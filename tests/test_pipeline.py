"""Pipeline tests. The synthetic traces pin the grammar and the phase rules;
the real smoke trace pins that we agree with an actual engine capture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from routetrace import (
    DECODE,
    PREFILL,
    build_store,
    load_X,
    parse_trace,
    make_split,
    prompt_ids_for,
    read_prompts,
    read_split,
)
from routetrace.parse import TraceFormatError
from routetrace.tensor import prompt_slices
from routetrace import transforms as T

REAL_TRACE = Path(__file__).resolve().parents[1] / "data/traces/smoke.trace"

# rt_trace prints gates as "%.4f": each carries up to 5e-5 of rounding, so a
# renormalised top-8 row sums to 1 only within 8 * 5e-5.
GATE_SUM_TOL = 8 * 5e-5


def _line(call, row, layer, pairs):
    return f"{call} {row} {layer} " + " ".join(f"{e}:{g:.4f}" for e, g in pairs)


def _write(tmp_path, lines, name="t.trace"):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def test_marker_splits_prefill_from_decode(tmp_path):
    """2-token prefill then 2 decode steps, across 2 layers."""
    lines = ["#prompt req-a 2"]
    call = 0
    for layer in (0, 1):  # prefill: one forward, layer-major, 2 rows
        for row in (0, 1):
            lines.append(_line(call, row, layer, [(10 + row, 0.6), (20 + row, 0.4)]))
        call += 1
    for step in range(2):  # decode: one row per forward
        for layer in (0, 1):
            lines.append(_line(call, 0, layer, [(30, 0.7), (40, 0.3)]))
            call += 1

    records, prompts = parse_trace(_write(tmp_path, lines))
    assert len(prompts) == 1
    assert prompts[0].key == "req-a"
    assert prompts[0].n_prompt_tokens == 2
    assert prompts[0].n_decode_tokens == 2

    prefill = [r for r in records if r["phase"] == PREFILL]
    decode = [r for r in records if r["phase"] == DECODE]
    # 2 tokens x 2 layers x 2 experts each
    assert len(prefill) == 8 and len(decode) == 8
    assert sorted({r["token_id"] for r in prefill}) == [0, 1]
    assert sorted({r["token_id"] for r in decode}) == [0, 1]


def test_multiple_prompts_get_distinct_ids(tmp_path):
    lines = []
    call = 0
    for key in ("req-a", "req-b"):
        lines.append(f"#prompt {key} 1")
        lines.append(_line(call, 0, 0, [(1, 1.0)]))  # prefill (1 token)
        call += 1
        lines.append(_line(call, 0, 0, [(2, 1.0)]))  # decode step 0
        call += 1

    records, prompts = parse_trace(_write(tmp_path, lines))
    assert [p.prompt_id for p in prompts] == [0, 1]
    assert [p.key for p in prompts] == ["req-a", "req-b"]
    # A single-token prompt is exactly the case row-counting would mislabel:
    # both its prefill and its decode forwards are one row wide.
    for pid in (0, 1):
        phases = {r["phase"] for r in records if r["prompt_id"] == pid}
        assert phases == {PREFILL, DECODE}


def test_first_prompt_id_offsets_across_files(tmp_path):
    lines = ["#prompt req-z 1", _line(0, 0, 0, [(1, 1.0)])]
    _, prompts = parse_trace(_write(tmp_path, lines), first_prompt_id=7)
    assert prompts[0].prompt_id == 7


def test_markerless_trace_treats_first_forward_as_prefill(tmp_path):
    """Traces captured before rt_prompt() existed must still parse."""
    lines = []
    for row in (0, 1, 2):
        lines.append(_line(0, row, 0, [(5, 1.0)]))
    lines.append(_line(1, 0, 0, [(6, 1.0)]))

    records, prompts = parse_trace(_write(tmp_path, lines))
    assert prompts[0].n_prompt_tokens == 3
    assert sum(r["phase"] == PREFILL for r in records) == 3
    assert sum(r["phase"] == DECODE for r in records) == 1


def test_wide_decode_forward_is_rejected(tmp_path):
    """A decode forward with >1 row means the prompt length was wrong."""
    lines = ["#prompt req-a 1", _line(0, 0, 0, [(1, 1.0)])]
    lines.append(_line(1, 0, 0, [(2, 1.0)]))
    lines.append(_line(1, 1, 0, [(3, 1.0)]))
    with pytest.raises(TraceFormatError, match="expected 1"):
        parse_trace(_write(tmp_path, lines))


def test_malformed_expert_field_is_rejected(tmp_path):
    with pytest.raises(TraceFormatError):
        parse_trace(_write(tmp_path, ["#prompt a 1", "0 0 0 12 13"]))


def test_marker_is_invisible_to_route_pairs_reader(tmp_path):
    """route_pairs.py skips lines with <4 fields; the marker must have 3."""
    p = _write(tmp_path, ["#prompt req-a 2"])
    fields = p.read_text().split()
    assert len(fields) == 3


# ---------------------------------------------------------------- real capture

requires_real = pytest.mark.skipif(
    not REAL_TRACE.exists(), reason="smoke.trace not present"
)


@requires_real
def test_real_trace_shapes_and_invariants(tmp_path):
    meta = build_store([REAL_TRACE], tmp_path / "store")
    assert meta["n_layers"] == 40
    assert meta["n_experts"] == 256
    assert meta["top_k"] == 8 and meta["top_k_min"] == 8

    X, index = load_X(tmp_path / "store", phase=DECODE)
    n_tokens = X.shape[0]
    assert X.shape == (n_tokens, 40, 256)
    assert X.dtype == np.float32

    # exactly 8 nonzero experts per (token, layer)
    nz = (X != 0).sum(axis=-1)
    assert nz.min() == 8 and nz.max() == 8

    # Gates are the renormalised ones, so each (token, layer) sums to 1 -- up to
    # the trace's own precision. rt_trace prints "%.4f", so each of the top_k
    # gates carries up to 5e-5 of rounding and their sum up to top_k * 5e-5.
    sums = X.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=GATE_SUM_TOL)

    # index is aligned and ordered
    assert index.shape == (n_tokens,)
    assert list(index["token_id"]) == sorted(index["token_id"])


@requires_real
def test_real_trace_prefill_and_decode_are_disjoint(tmp_path):
    build_store([REAL_TRACE], tmp_path / "store")
    Xd, _ = load_X(tmp_path / "store", phase=DECODE)
    Xp, _ = load_X(tmp_path / "store", phase=PREFILL)
    Xa, _ = load_X(tmp_path / "store", phase=None)
    # the smoke capture is 18 prefill rows and 31 decode steps
    assert Xp.shape[0] == 18
    assert Xd.shape[0] == 31
    # split=None shares a token axis, so the counts add only if ids do not collide
    assert Xa.shape[0] == 31 + 18


@requires_real
def test_sparse_matches_dense(tmp_path):
    build_store([REAL_TRACE], tmp_path / "store")
    X, _ = load_X(tmp_path / "store", phase=DECODE)
    coo, _ = load_X(tmp_path / "store", phase=DECODE, sparse=True)
    assert coo.nnz == 31 * 40 * 8
    assert np.array_equal(coo.to_dense(), X)


@requires_real
def test_prompt_slices_cover_the_token_axis(tmp_path):
    build_store([REAL_TRACE], tmp_path / "store")
    X, index = load_X(tmp_path / "store", phase=DECODE)
    slices = prompt_slices(index)
    assert sum(s.stop - s.start for s in slices.values()) == X.shape[0]


# -------------------------------------------------------------------- transforms


def _toy():
    X = np.zeros((2, 1, 8), dtype=np.float32)
    X[0, 0, [1, 3]] = [0.75, 0.25]
    X[1, 0, [3, 5]] = [0.50, 0.50]
    return X


def test_binarize_and_histograms():
    X = _toy()
    assert T.binarize(X).sum() == 4
    # expert_histogram keeps the layer axis: an Expert is a (layer, slot) pair
    assert T.expert_histogram(X).shape == (1, 8)
    assert T.expert_histogram(X)[0].tolist() == [0, 1, 0, 2, 0, 1, 0, 0]
    assert T.gate_mass(X)[0, 3] == pytest.approx(0.75)


def test_slot_histogram_collapses_layers_and_expert_histogram_does_not():
    """The distinction that made an earlier analysis wrong: slot 1 of layer 0
    and slot 1 of layer 1 are different Experts and must not be summed."""
    X = np.zeros((1, 2, 4), dtype=np.float32)
    X[0, 0, 1] = 1.0
    X[0, 1, 1] = 1.0
    assert T.expert_histogram(X).shape == (2, 4)
    assert T.expert_histogram(X)[0, 1] == 1 and T.expert_histogram(X)[1, 1] == 1
    assert T.slot_histogram(X).shape == (4,)
    assert T.slot_histogram(X)[1] == 2  # two different Experts, one index


def test_expert_profile_is_normalised_over_experts():
    X = _toy()
    p = T.expert_profile(X)
    assert p.shape == (1 * 8,)
    assert p.sum() == pytest.approx(1.0)


def test_topk_mask_then_renormalize():
    X = _toy()
    top1 = T.topk_mask(X, 1)
    assert (top1 != 0).sum(axis=-1).tolist() == [[1], [1]]
    assert top1[0, 0, 1] == pytest.approx(0.75)
    r = T.renormalize(top1)
    assert np.allclose(r.sum(axis=-1), 1.0)


def test_topk_mask_never_adds_experts():
    X = _toy()
    wide = T.topk_mask(X, 8)
    assert np.array_equal(wide, X)


def test_cooccurrence_and_transition():
    X = _toy()
    C = T.cooccurrence(X, 0)
    assert C[3, 3] == 2 and C[1, 3] == 1 and C[1, 5] == 0
    X2 = np.zeros((1, 2, 4), dtype=np.float32)
    X2[0, 0, 1] = 1.0
    X2[0, 1, 2] = 1.0
    assert T.transition(X2, 0)[1, 2] == 1


# --------------------------------------------------- real multi-prompt capture

SERVE_TRACE = Path(__file__).resolve().parent / "fixtures/serve3.trace"

requires_serve = pytest.mark.skipif(
    not SERVE_TRACE.exists(), reason="serve3.trace fixture not present"
)


@requires_serve
def test_serve_capture_splits_three_prompts(tmp_path):
    """A real SERVE run: 3 prompts of 7/5/3 prompt tokens, 15 decode steps each."""
    meta = build_store([SERVE_TRACE], tmp_path / "store")
    assert meta["n_prompts"] == 3
    assert meta["top_k"] == 8 and meta["top_k_min"] == 8

    X, index = load_X(tmp_path / "store", phase=DECODE)
    assert X.shape == (45, 40, 256)
    assert (X != 0).sum(axis=-1).min() == 8
    assert (X != 0).sum(axis=-1).max() == 8
    assert np.allclose(X.sum(axis=-1), 1.0, atol=GATE_SUM_TOL)

    # every prompt contributes a contiguous, equal-length block
    slices = prompt_slices(index)
    assert list(slices) == [0, 1, 2]
    assert all(s.stop - s.start == 15 for s in slices.values())
    # token_id restarts per prompt
    assert list(index["token_id"][slices[1]]) == list(range(15))

    Xp, _ = load_X(tmp_path / "store", phase=PREFILL)
    assert Xp.shape[0] == 7 + 5 + 3


@requires_serve
def test_both_phases_do_not_collide(tmp_path):
    """prefill token 0 and decode token 0 of one prompt are distinct slices."""
    build_store([SERVE_TRACE], tmp_path / "store")
    X, index = load_X(tmp_path / "store", phase=None)
    assert X.shape[0] == 45 + 15
    both = index[index["prompt_id"] == 0]
    assert set(both["phase"]) == {PREFILL, DECODE}
    assert (both["token_id"] == 0).sum() == 2  # one per phase, not merged


@requires_serve
def test_two_files_concatenate_without_id_collisions(tmp_path):
    meta = build_store([SERVE_TRACE, REAL_TRACE], tmp_path / "store")
    assert meta["n_prompts"] == 4  # 3 from serve + 1 markerless smoke
    X, index = load_X(tmp_path / "store", phase=DECODE)
    assert sorted(set(index["prompt_id"])) == [0, 1, 2, 3]
    assert X.shape[0] == 45 + 31


# ------------------------------------------------------------------ categories


def _mini_trace_and_manifest(tmp_path, cats):
    """Two-layer, 1-token-prefill + 2-decode-step trace for len(cats) prompts."""
    lines, call = [], 0
    for i in range(len(cats)):
        lines.append(f"#prompt p{i:05d} 1")
        for _ in range(3):  # 1 prefill forward + 2 decode forwards
            for layer in (0, 1):
                lines.append(_line(call, 0, layer, [(i, 0.6), (i + 1, 0.4)]))
                call += 1
    trace = _write(tmp_path, lines, "c.trace")
    manifest = tmp_path / "c.trace.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "prompts": [
                    {"key": f"p{i:05d}", "category": c, "text": f"t{i}", "response": f"r{i}"}
                    for i, c in enumerate(cats)
                ]
            }
        )
    )
    return trace


def test_manifest_attaches_category_and_text(tmp_path):
    trace = _mini_trace_and_manifest(tmp_path, ["coding", "math_reasoning"])
    build_store([trace], tmp_path / "store")
    rows = read_prompts(tmp_path / "store").to_pylist()
    assert [r["category"] for r in rows] == ["coding", "math_reasoning"]
    assert [r["text"] for r in rows] == ["t0", "t1"]
    assert [r["response"] for r in rows] == ["r0", "r1"]


def test_load_X_filters_by_category(tmp_path):
    trace = _mini_trace_and_manifest(tmp_path, ["coding", "math_reasoning", "coding"])
    build_store([trace], tmp_path / "store")

    assert prompt_ids_for(tmp_path / "store", "coding") == [0, 2]

    X, index = load_X(tmp_path / "store", phase=DECODE, categories="coding")
    assert sorted(set(index["prompt_id"])) == [0, 2]
    assert X.shape[0] == 4  # 2 prompts x 2 decode steps

    both, _ = load_X(
        tmp_path / "store", phase=DECODE, categories=["coding", "math_reasoning"]
    )
    assert both.shape[0] == 6


def test_categories_and_prompts_intersect(tmp_path):
    trace = _mini_trace_and_manifest(tmp_path, ["coding", "math_reasoning", "coding"])
    build_store([trace], tmp_path / "store")
    _, index = load_X(
        tmp_path / "store", phase=DECODE, categories="coding", prompts=[2, 1]
    )
    assert sorted(set(index["prompt_id"])) == [2]


def test_unknown_category_is_rejected(tmp_path):
    trace = _mini_trace_and_manifest(tmp_path, ["coding"])
    build_store([trace], tmp_path / "store")
    with pytest.raises(ValueError, match="unknown categories"):
        prompt_ids_for(tmp_path / "store", "nope")


def test_render_chat_matches_the_engine_template():
    from routetrace.capture import render_chat

    off = render_chat("hi")
    assert off == (
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    on = render_chat("hi", enable_thinking=True)
    assert on.endswith("<|im_start|>assistant\n<think>\n")
    # the assistant state is never left bare -- that is the untrained one
    assert not off.endswith("assistant\n") and not on.endswith("assistant\n")
    sys_ = render_chat("hi", system="be terse")
    assert sys_.startswith("<|im_start|>system\nbe terse<|im_end|>\n")


# ---------------------------------------------------------------- train / test


def _cat_store(tmp_path, cats):
    trace = _mini_trace_and_manifest(tmp_path, cats)
    build_store([trace], tmp_path / "store")
    return tmp_path / "store"


def test_split_is_stratified_and_exhaustive(tmp_path):
    cats = ["coding"] * 10 + ["math_reasoning"] * 10
    store = _cat_store(tmp_path, cats)
    sp = make_split(store, test_per_category=2, seed=0)

    assert sp["n_train"] == 16 and sp["n_test"] == 4
    for cat, d in sp["by_category"].items():
        assert len(d["test"]) == 2, cat
        assert len(d["train"]) == 8, cat
    # every prompt lands on exactly one side
    assert set(sp["train"]) | set(sp["test"]) == set(range(20))
    assert not set(sp["train"]) & set(sp["test"])


def test_split_is_deterministic_in_seed(tmp_path):
    cats = ["coding"] * 10 + ["math_reasoning"] * 10
    store = _cat_store(tmp_path, cats)
    a = make_split(store, test_per_category=2, seed=0, overwrite=True)
    b = make_split(store, test_per_category=2, seed=0, overwrite=True)
    c = make_split(store, test_per_category=2, seed=1, overwrite=True)
    assert a["test"] == b["test"]
    assert a["test"] != c["test"]


def test_split_will_not_silently_overwrite(tmp_path):
    store = _cat_store(tmp_path, ["coding"] * 4)
    make_split(store, test_per_category=1, seed=0)
    with pytest.raises(FileExistsError):
        make_split(store, test_per_category=1, seed=0)


def test_read_split_without_one_is_an_error(tmp_path):
    store = _cat_store(tmp_path, ["coding"] * 4)
    with pytest.raises(FileNotFoundError, match="make_split"):
        read_split(store)


def test_split_larger_than_category_is_rejected(tmp_path):
    store = _cat_store(tmp_path, ["coding"] * 3)
    with pytest.raises(ValueError, match="cannot hold out"):
        make_split(store, test_per_category=4, seed=0)


def test_load_X_by_split_does_not_leak(tmp_path):
    cats = ["coding"] * 10 + ["math_reasoning"] * 10
    store = _cat_store(tmp_path, cats)
    sp = make_split(store, test_per_category=2, seed=0)

    _, itr = load_X(store, phase=DECODE, split="train")
    _, ite = load_X(store, phase=DECODE, split="test")
    train_ids = set(itr["prompt_id"].tolist())
    test_ids = set(ite["prompt_id"].tolist())

    assert train_ids == set(sp["train"])
    assert test_ids == set(sp["test"])
    # the whole point: no prompt contributes tokens to both sides
    assert not train_ids & test_ids


def test_split_and_category_intersect(tmp_path):
    cats = ["coding"] * 10 + ["math_reasoning"] * 10
    store = _cat_store(tmp_path, cats)
    make_split(store, test_per_category=2, seed=0)
    _, idx = load_X(store, phase=DECODE, split="test", categories="coding")
    ids = set(idx["prompt_id"].tolist())
    assert len(ids) == 2 and all(i < 10 for i in ids)


def test_passing_a_phase_as_split_is_rejected(tmp_path):
    """split= used to mean the phase; that mistake must be loud, not silent."""
    store = _cat_store(tmp_path, ["coding"] * 4)
    with pytest.raises(ValueError, match="is a phase; pass phase="):
        load_X(store, split="decode")
