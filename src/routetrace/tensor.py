"""Materialise the routing store as X: [tokens, layers, n_experts].

X is the object every experiment starts from. Its token axis is the concatenation
of the selected prompts' tokens, in (prompt_id, token_id) order; ``index`` maps
each slice back to the prompt and position it came from, so a transform never has
to guess which rows belong together.

X is dense-by-default because at experiment scale it is small (a 1,280-token
decode split is 1,280 x 40 x 256 x 4 B = 52 MB) and every downstream transform is
a plain ndarray op. Past a few tens of thousands of tokens, ask for
``sparse=True`` and get the COO triplets instead -- the store itself is always
sparse, so nothing is lost either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .store import read_meta, read_prompts, read_routing

DECODE = "decode"
PREFILL = "prefill"


@dataclass
class COO:
    """Sparse X: ``coords`` is [3, nnz] of (token, layer, expert) row indices."""

    coords: np.ndarray
    values: np.ndarray
    shape: tuple[int, int, int]

    @property
    def nnz(self) -> int:
        return int(self.values.size)

    def to_dense(self) -> np.ndarray:
        X = np.zeros(self.shape, dtype=self.values.dtype)
        X[self.coords[0], self.coords[1], self.coords[2]] = self.values
        return X


def prompt_ids_for(store_dir: str | Path, categories: list[str] | str) -> list[int]:
    """prompt_ids belonging to the given category or categories."""
    if isinstance(categories, str):
        categories = [categories]
    rows = read_prompts(store_dir).to_pylist()
    known = {r["category"] for r in rows if r["category"]}
    unknown = set(categories) - known
    if unknown:
        raise ValueError(f"unknown categories {sorted(unknown)}; store has {sorted(known)}")
    return [r["prompt_id"] for r in rows if r["category"] in categories]


def load_X(
    store_dir: str | Path,
    phase: str | None = DECODE,
    split: str | None = None,
    prompts: list[int] | None = None,
    categories: list[str] | str | None = None,
    sparse: bool = False,
    dtype=np.float32,
):
    """Load X and its token index.

    ``phase`` selects prefill or decode and defaults to ``"decode"``; pass
    ``None`` to keep both. ``split`` selects ``"train"`` or ``"test"`` from the
    store's saved assignment. ``prompts`` restricts to explicit prompt ids, and
    ``categories`` to corpus categories. Every filter given is intersected.

    Returns ``(X, index)``. ``X`` is ``[tokens, layers, n_experts]`` dense, or a
    :class:`COO` when ``sparse=True``. ``index`` is a structured array with
    fields ``prompt_id`` and ``token_id``, one entry per row of X's token axis.
    """
    store_dir = Path(store_dir)
    meta = read_meta(store_dir)
    table = read_routing(store_dir)

    # `split` used to mean the phase. Catch the muscle-memory error loudly
    # rather than filter on a train/test value that matches no phase and raise
    # the confusing "no rows" further down.
    if split in (DECODE, PREFILL):
        raise ValueError(
            f"split={split!r} is a phase; pass phase={split!r} instead. "
            "split= selects 'train' or 'test'."
        )

    if categories is not None:
        by_cat = prompt_ids_for(store_dir, categories)
        prompts = by_cat if prompts is None else sorted(set(prompts) & set(by_cat))
    if split is not None:
        from .splits import prompt_ids_for_split

        by_split = prompt_ids_for_split(store_dir, split)
        prompts = by_split if prompts is None else sorted(set(prompts) & set(by_split))

    if phase is not None:
        # phase is dictionary-encoded; cast so the comparison is against strings.
        table = table.filter(pc.equal(table["phase"].cast("string"), phase))
    if prompts is not None:
        table = table.filter(pc.is_in(table["prompt_id"], value_set=pa.array(prompts, pa.int32())))
    if table.num_rows == 0:
        raise ValueError(
            f"no rows for phase={phase!r} split={split!r} "
            f"categories={categories!r} prompts={prompts!r}"
        )

    prompt_id = table["prompt_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    token_id = table["token_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    phase_col = np.asarray(table["phase"].cast("string").to_pylist())
    layer = table["layer"].to_numpy(zero_copy_only=False).astype(np.int64)
    expert = table["expert_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    gate = table["gate"].to_numpy(zero_copy_only=False).astype(dtype)

    # Token axis: unique (prompt_id, token_id), ordered. Packing the pair into one
    # integer key lets np.unique both order the axis and hand back the per-row
    # mapping into it; the pair is recovered by dividing the key back out, so the
    # index never depends on where a given row happened to sit in the file.
    # token_id restarts at 0 for each phase, so (prompt_id, token_id) alone is
    # NOT unique when both phases are loaded -- prefill token 0 and decode token 0
    # of the same prompt would collapse onto one slice of X. Phase joins the key.
    is_decode = (phase_col == DECODE).astype(np.int64)
    stride = int(token_id.max()) + 1
    key = (prompt_id * 2 + is_decode) * stride + token_id
    uniq, inverse = np.unique(key, return_inverse=True)
    inverse = inverse.reshape(-1)

    n_tokens = uniq.size
    n_layers = meta["n_layers"]
    n_experts = meta["n_experts"]

    index = np.empty(
        n_tokens,
        dtype=[("prompt_id", np.int32), ("phase", "U7"), ("token_id", np.int32)],
    )
    index["prompt_id"] = (uniq // stride // 2).astype(np.int32)
    index["phase"] = np.where((uniq // stride) % 2 == 1, DECODE, PREFILL)
    index["token_id"] = (uniq % stride).astype(np.int32)

    if sparse:
        coords = np.stack([inverse.astype(np.int32), layer.astype(np.int32), expert.astype(np.int32)])
        return COO(coords=coords, values=gate, shape=(n_tokens, n_layers, n_experts)), index

    X = np.zeros((n_tokens, n_layers, n_experts), dtype=dtype)
    X[inverse, layer, expert] = gate
    return X, index


def to_torch(X, device: str | None = None):
    """Convert a dense X or a :class:`COO` to torch, if torch is installed."""
    import torch  # imported lazily: torch is an optional extra

    if isinstance(X, COO):
        t = torch.sparse_coo_tensor(
            torch.from_numpy(X.coords.astype(np.int64)),
            torch.from_numpy(X.values),
            size=X.shape,
        )
    else:
        t = torch.from_numpy(np.ascontiguousarray(X))
    return t.to(device) if device else t


def prompt_slices(index: np.ndarray) -> dict[int, slice]:
    """Map each prompt_id to its contiguous slice of X's token axis."""
    out: dict[int, slice] = {}
    pid = index["prompt_id"]
    if pid.size == 0:
        return out
    starts = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    ends = np.r_[starts[1:], pid.size]
    for s, e in zip(starts, ends):
        out[int(pid[s])] = slice(int(s), int(e))
    return out


def describe(store_dir: str | Path) -> str:
    """One-screen summary of a store, for sanity-checking a capture."""
    meta = read_meta(store_dir)
    prompts = read_prompts(store_dir)
    lines = [
        f"store      {store_dir}",
        f"layers     {meta['n_layers']}",
        f"experts    {meta['n_experts']}",
        f"top_k      {meta['top_k']} (min observed {meta['top_k_min']})",
        f"rows       {meta['n_rows']:,}",
        f"prompts    {meta['n_prompts']}",
    ]
    for row in prompts.to_pylist():
        lines.append(
            f"  #{row['prompt_id']:<4} {row['key']:<24} "
            f"prefill={row['n_prompt_tokens']:<6} decode={row['n_decode_tokens']:<6} "
            f"({row['source']})"
        )
    return "\n".join(lines)
