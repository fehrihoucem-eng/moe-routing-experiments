#!/usr/bin/env python3
"""Aggregate a Store into the JSON the explorer page reads, then build the page.

    .venv/bin/python scripts/export_site_data.py

Writes ``site/data.json`` and inlines it into ``site/routing.html`` from
``site/routing.template.html``. The data is inlined rather than fetched because
the page is opened over ``file://``, where a cross-origin fetch of a sibling
JSON file is blocked.

Everything here reads the **train** Split only. The test Split stays unseen so
that nothing shown on the page can inform a decision it is meant to validate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routetrace import load_X, read_meta, read_prompts, read_split  # noqa: E402
from routetrace import transforms as T  # noqa: E402

STORE = ROOT / "data/stores/corpus_v1"
SITE = ROOT / "site"

LABELS = {
    "coding": "Coding",
    "math_reasoning": "Math / reasoning",
    "factual_expository": "Factual",
    "conversational_creative": "Conversational",
    "structured_extraction": "Structured extraction",
}
CATS = list(LABELS)

# The heatmap's colour cap, in multiples of an even split. Fixed rather than
# data-derived so the five category maps stay comparable to each other.
HEAT_CAP = 20.0
# Which decode token of the demo prompt to expand. Any token works; a mid-answer
# one avoids the first-token transient.
DEMO_TOKEN = 12


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def main() -> int:
    meta = read_meta(STORE)
    split = read_split(STORE)
    prompts = {r["prompt_id"]: r for r in read_prompts(STORE).to_pylist()}
    n_layers, n_slots = meta["n_layers"], meta["n_experts"]

    out: dict = {
        "meta": {
            "n_layers": n_layers,
            "n_slots": n_slots,
            "n_experts": n_layers * n_slots,
            "top_k": meta["top_k"],
            "n_rows": meta["n_rows"],
            "n_prompts": meta["n_prompts"],
            "n_train": split["n_train"],
            "n_test": split["n_test"],
        },
        "categories": [{"key": c, "label": LABELS[c]} for c in CATS],
    }

    # Per-Expert counts per Category. expert_histogram keeps the layer axis:
    # a slot number means nothing across layers.
    hist: dict[str, np.ndarray] = {}
    for cat in CATS:
        X, _ = load_X(STORE, phase="decode", split="train", categories=cat)
        hist[cat] = T.expert_histogram(X).astype(float)
        out.setdefault("tokens_train", {})[cat] = int(X.shape[0])
    out["tokens_test"] = {
        c: sum(prompts[i]["n_decode_tokens"] for i in split["by_category"][c]["test"])
        for c in CATS
    }

    # Heatmap: share within a layer, expressed as a multiple of an even split,
    # so a value of 1.0 means "exactly as often as chance".
    out["heat"] = {
        c: np.round(
            hist[c] / np.maximum(hist[c].sum(axis=1, keepdims=True), 1) * n_slots, 3
        ).tolist()
        for c in CATS
    }
    out["heat_cap"] = HEAT_CAP

    # The correction, both ways round: per-Expert profiles vs the same profiles
    # collapsed onto slot numbers.
    per_expert = {c: hist[c].ravel() / hist[c].sum() for c in CATS}
    per_slot = {c: hist[c].sum(axis=0) / hist[c].sum() for c in CATS}
    out["sim_expert"] = [[round(cosine(per_expert[a], per_expert[b]), 4) for b in CATS] for a in CATS]
    out["sim_slot"] = [[round(cosine(per_slot[a], per_slot[b]), 4) for b in CATS] for a in CATS]

    # Cross-category agreement per layer, over all ten pairs.
    per_layer = []
    for layer in range(n_layers):
        dist = {c: hist[c][layer] / max(hist[c][layer].sum(), 1) for c in CATS}
        pairs = [cosine(dist[a], dist[b]) for i, a in enumerate(CATS) for b in CATS[i + 1 :]]
        per_layer.append(
            {
                "layer": layer,
                "mean": round(float(np.mean(pairs)), 4),
                "min": round(float(np.min(pairs)), 4),
                "max": round(float(np.max(pairs)), 4),
            }
        )
    out["per_layer"] = per_layer

    # One slot index across all layers: 40 different Experts sharing a number.
    total = sum(hist[c] for c in CATS)
    slot = int(total.sum(axis=0).argmax())
    hot = np.unravel_index(total.argmax(), total.shape)
    out["collapse"] = {
        "slot": slot,
        "per_layer": [int(total[l, slot]) for l in range(n_layers)],
        "sum": int(total[:, slot].sum()),
        "hottest_expert": {"layer": int(hot[0]), "slot": int(hot[1]), "count": int(total[hot])},
    }

    # One token, fully expanded across the stack.
    X, index = load_X(STORE, phase="decode", categories="coding", prompts=[0])
    layers = []
    for layer in range(n_layers):
        nz = np.nonzero(X[DEMO_TOKEN, layer])[0]
        layers.append(
            [[int(s), round(float(X[DEMO_TOKEN, layer, s]), 4)]
             for s in sorted(nz, key=lambda s: -X[DEMO_TOKEN, layer, s])]
        )
    out["one_token"] = {
        "prompt": prompts[0]["text"][:120],
        "token_id": int(index["token_id"][DEMO_TOKEN]),
        "layers": layers,
    }

    SITE.mkdir(exist_ok=True)
    data_path = SITE / "data.json"
    data_path.write_text(json.dumps(out, separators=(",", ":")))

    template = (SITE / "routing.template.html").read_text()
    if "__DATA__" not in template:
        raise SystemExit("template lost its __DATA__ placeholder")
    page = SITE / "routing.html"
    page.write_text(template.replace("__DATA__", data_path.read_text()))

    print(f"{data_path}  {data_path.stat().st_size / 1024:.0f} KB")
    print(f"{page}  {page.stat().st_size / 1024:.0f} KB")
    print(f"hottest Expert: layer {hot[0]}, slot {hot[1]} ({int(total[hot])} selections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
