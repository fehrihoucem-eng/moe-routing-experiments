"""Parquet store for routing records.

Layout under ``<store>/``::

    routing.parquet   prompt_id, phase, token_id, layer, expert_id, gate
    prompts.parquet   prompt_id, key, n_prompt_tokens, n_decode_tokens, source
    meta.json         n_layers, n_experts, top_k, source traces

``routing.parquet`` is the canonical X: one row per selected expert, so a
(token, layer) contributes exactly top_k rows and the dense tensor is only ever
materialised on demand. ``phase`` is dictionary-encoded, so keeping prefill
alongside decode costs almost nothing and the split stays a filter, not a
separate capture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .parse import Prompt, parse_trace

ROUTING_SCHEMA = pa.schema(
    [
        pa.field("prompt_id", pa.int32()),
        pa.field("phase", pa.dictionary(pa.int8(), pa.string())),
        pa.field("token_id", pa.int32()),
        pa.field("layer", pa.int16()),
        pa.field("expert_id", pa.int16()),
        pa.field("gate", pa.float32()),
    ]
)

PROMPTS_SCHEMA = pa.schema(
    [
        pa.field("prompt_id", pa.int32()),
        pa.field("key", pa.string()),
        pa.field("category", pa.string()),
        pa.field("n_prompt_tokens", pa.int32()),
        pa.field("n_decode_tokens", pa.int32()),
        pa.field("source", pa.string()),
        pa.field("text", pa.string()),
        pa.field("response", pa.string()),
    ]
)


def _routing_table(records: list[dict]) -> pa.Table:
    cols = {name: [] for name in ("prompt_id", "phase", "token_id", "layer", "expert_id", "gate")}
    for r in records:
        for name in cols:
            cols[name].append(r[name])
    arrays = [
        pa.array(cols["prompt_id"], pa.int32()),
        pa.array(cols["phase"], pa.string()).dictionary_encode(),
        pa.array(cols["token_id"], pa.int32()),
        pa.array(cols["layer"], pa.int16()),
        pa.array(cols["expert_id"], pa.int16()),
        pa.array(cols["gate"], pa.float32()),
    ]
    return pa.Table.from_arrays(arrays, schema=ROUTING_SCHEMA)


def _load_manifests(manifests: list[str | Path] | None) -> dict[str, dict]:
    """Map trace key -> manifest entry. The trace records only the key, so this
    is the only route by which a prompt's category and text reach the store."""
    by_key: dict[str, dict] = {}
    for path in manifests or []:
        blob = json.loads(Path(path).read_text())
        for entry in blob.get("prompts", []):
            by_key[entry["key"]] = entry
    return by_key


def _prompts_table(prompts: list[Prompt], by_key: dict[str, dict]) -> pa.Table:
    def field(p: Prompt, name: str) -> str:
        return str(by_key.get(p.key, {}).get(name, "") or "")

    return pa.Table.from_arrays(
        [
            pa.array([p.prompt_id for p in prompts], pa.int32()),
            pa.array([p.key for p in prompts], pa.string()),
            pa.array([field(p, "category") for p in prompts], pa.string()),
            pa.array([p.n_prompt_tokens for p in prompts], pa.int32()),
            pa.array([p.n_decode_tokens for p in prompts], pa.int32()),
            pa.array([p.source for p in prompts], pa.string()),
            pa.array([field(p, "text") for p in prompts], pa.string()),
            pa.array([field(p, "response") for p in prompts], pa.string()),
        ],
        schema=PROMPTS_SCHEMA,
    )


def build_store(
    traces: list[str | Path],
    out_dir: str | Path,
    n_experts: int = 256,
    n_layers: int | None = None,
    manifests: list[str | Path] | None = None,
) -> dict:
    """Parse ``traces`` into a parquet store at ``out_dir``. Returns the metadata.

    ``manifests`` are the JSON sidecars written by :func:`~routetrace.capture.capture`;
    they carry each prompt's category and text, which the trace itself does not.
    Defaults to ``<trace>.manifest.json`` beside each trace when present."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if manifests is None:
        manifests = [
            m for t in traces
            if (m := Path(str(t) + ".manifest.json")).exists()
        ]

    all_records: list[dict] = []
    all_prompts: list[Prompt] = []
    for trace in traces:
        records, prompts = parse_trace(trace, first_prompt_id=len(all_prompts))
        all_records.extend(records)
        all_prompts.extend(prompts)

    if not all_records:
        raise ValueError("no routing records parsed; are the traces empty?")

    observed_layers = max(r["layer"] for r in all_records) + 1
    if n_layers is None:
        n_layers = observed_layers
    max_expert = max(r["expert_id"] for r in all_records)
    if max_expert >= n_experts:
        raise ValueError(f"expert id {max_expert} exceeds n_experts={n_experts}")

    pq.write_table(_routing_table(all_records), out_dir / "routing.parquet", compression="zstd")
    pq.write_table(_prompts_table(all_prompts, _load_manifests(manifests)),
                   out_dir / "prompts.parquet", compression="zstd")

    # top_k is a property of the capture, not an assumption: read it back off the
    # data so a container with a different top-k does not silently mislabel X.
    # phase belongs in the key: token_id restarts at 0 per phase, so without it
    # a prompt's prefill token 0 and decode token 0 merge and top_k reads as 16.
    per_group: dict[tuple[int, str, int, int], int] = {}
    for r in all_records:
        key = (r["prompt_id"], r["phase"], r["token_id"], r["layer"])
        per_group[key] = per_group.get(key, 0) + 1
    top_k = max(per_group.values())

    meta = {
        "n_layers": int(n_layers),
        "n_experts": int(n_experts),
        "top_k": int(top_k),
        "top_k_min": int(min(per_group.values())),
        "n_rows": len(all_records),
        "n_prompts": len(all_prompts),
        "traces": [str(t) for t in traces],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def read_meta(store_dir: str | Path) -> dict:
    return json.loads((Path(store_dir) / "meta.json").read_text())


def read_routing(store_dir: str | Path) -> pa.Table:
    return pq.read_table(Path(store_dir) / "routing.parquet")


def read_prompts(store_dir: str | Path) -> pa.Table:
    return pq.read_table(Path(store_dir) / "prompts.parquet")
