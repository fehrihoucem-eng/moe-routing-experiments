"""Predictor tests.

The synthetic Routing pins the estimator's arithmetic exactly -- counts, the
alpha=inf identity, tie-breaking, and the rule that a fitted table never sees a
scored row. The serve3 fixture pins that the compact loader agrees with a real
capture, including where Prompt boundaries fall.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from routetrace import build_store, make_split
from routetrace.predict import (
    Routing,
    coverage,
    fit_tables,
    folds,
    grid_rows,
    lag_rows,
    load_routing,
    score,
)
from routetrace.predict import _history_mass, _pair_counts

SERVE_TRACE = Path(__file__).resolve().parent / "fixtures/serve3.trace"
N_EXPERTS = 32
TOP_K = 4
BUDGETS = (1, 2, 4, 8)


def _routing(n_prompts=4, n_tokens=6, n_layers=3, seed=0) -> Routing:
    """Deterministic synthetic Routing with real Prompt boundaries."""
    rng = np.random.default_rng(seed)
    total = n_prompts * n_tokens
    slots = np.stack(
        [
            np.stack([rng.choice(N_EXPERTS, TOP_K, replace=False) for _ in range(n_layers)])
            for _ in range(total)
        ]
    ).astype(np.int16)
    gates = rng.random((total, n_layers, TOP_K)).astype(np.float32)
    gates /= gates.sum(axis=2, keepdims=True)

    prompt_id = np.repeat(np.arange(n_prompts, dtype=np.int32), n_tokens)
    token_id = np.tile(np.arange(n_tokens, dtype=np.int32), n_prompts)
    prev = np.arange(total, dtype=np.int64) - 1
    prev[token_id == 0] = -1

    return Routing(slots, gates, prompt_id, token_id, prev, N_EXPERTS)


def _store(tmp_path):
    build_store([SERVE_TRACE], tmp_path / "store")
    return tmp_path / "store"


# --- the compact form -------------------------------------------------------


@pytest.mark.skipif(not SERVE_TRACE.exists(), reason="serve3.trace fixture not present")
def test_load_routing_matches_a_real_capture(tmp_path):
    r = load_routing(_store(tmp_path), split=None)
    assert r.slots.shape == r.gates.shape
    assert r.n_layers == 40 and r.top_k == 8
    assert r.slots.shape[0] == r.n_tokens

    # Every (Token, Layer) selected top_k distinct Slots, and the gates it kept
    # renormalise to 1 within the trace's "%.4f" precision.
    for row in (0, r.n_tokens // 2, r.n_tokens - 1):
        for layer in (0, 20, 39):
            assert len(set(r.slots[row, layer].tolist())) == r.top_k
    assert np.allclose(r.gates.sum(axis=2), 1.0, atol=r.top_k * 5e-5)


@pytest.mark.skipif(not SERVE_TRACE.exists(), reason="serve3.trace fixture not present")
def test_prev_never_crosses_a_prompt_boundary(tmp_path):
    r = load_routing(_store(tmp_path), split=None)
    first = r.token_id == 0
    assert np.all(r.prev[first] == -1)
    assert np.all(r.prev[~first] >= 0)

    linked = np.flatnonzero(r.prev >= 0)
    assert np.all(r.prompt_id[linked] == r.prompt_id[r.prev[linked]])
    assert np.all(r.token_id[linked] == r.token_id[r.prev[linked]] + 1)


@pytest.mark.skipif(not SERVE_TRACE.exists(), reason="serve3.trace fixture not present")
def test_slots_are_ascending_within_a_cell(tmp_path):
    """The compact form orders by Slot, not by gate: the selection is the object.

    _persistence relies on the gate ordering being recovered explicitly rather
    than inherited from the array, so pin that it is not already gate-sorted.
    """
    r = load_routing(_store(tmp_path), split=None)
    assert np.all(np.diff(r.slots.astype(np.int64), axis=2) > 0)


# --- Coverage ---------------------------------------------------------------


def test_coverage_is_one_only_when_every_selected_expert_is_named():
    actual = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    scores = np.zeros((2, N_EXPERTS))
    np.put_along_axis(scores, actual, 1.0, axis=1)
    assert np.allclose(coverage(scores, actual, budgets=(4,)), 1.0)

    missed = np.zeros((2, N_EXPERTS))
    np.put_along_axis(missed, (actual + 16) % N_EXPERTS, 1.0, axis=1)
    assert np.allclose(coverage(missed, actual, budgets=(4,)), 0.0)


def test_budget_below_top_k_caps_coverage_by_arithmetic_alone():
    """A perfect Predictor still cannot exceed K/top_k. This is the reason the
    K-curve is reported instead of a single Top-5 number."""
    actual = np.array([[0, 1, 2, 3]])
    scores = np.zeros((1, N_EXPERTS))
    np.put_along_axis(scores, actual, 1.0, axis=1)
    got = coverage(scores, actual, budgets=(1, 2, 3, 4))
    assert np.allclose(got, [[1 / 4, 2 / 4, 3 / 4, 1.0]])


def test_coverage_is_monotone_in_the_budget():
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    rows = grid_rows(r, np.arange(r.n_tokens))
    s = score("cross_layer", r, tables, rows, layer=1)
    got = coverage(s, r.slots[rows, 1].astype(np.int64), budgets=BUDGETS)
    assert np.all(np.diff(got, axis=1) >= 0)


def test_ties_break_toward_the_lower_slot():
    actual = np.array([[5, 6, 7, 8]])
    scores = np.ones((1, N_EXPERTS))  # every Slot tied
    assert coverage(scores, actual, budgets=(1,))[0, 0] == 0.0  # picks Slot 0
    assert coverage(scores, actual, budgets=(9,))[0, 0] == 1.0  # Slots 0..8


# --- the estimator ----------------------------------------------------------


def test_pair_counts_matches_the_naive_loop():
    before = np.array([[0, 1], [1, 2]])
    after = np.array([[3, 4], [4, 5]])
    got = _pair_counts(before, after, N_EXPERTS)

    want = np.zeros((N_EXPERTS, N_EXPERTS))
    for b, a in zip(before, after):
        for i in b:
            for j in a:
                want[i, j] += 1
    assert np.array_equal(got, want)


def test_scores_are_a_distribution():
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    rows = grid_rows(r, np.arange(r.n_tokens))
    for name in ("popularity", "persistence", "prev_token", "cross_layer", "combined"):
        s = score(name, r, tables, rows, layer=2)
        assert s.shape == (rows.size, N_EXPERTS)
        assert np.allclose(s.sum(axis=1), 1.0)
        assert np.all(s >= 0)


def test_popularity_is_the_infinite_alpha_limit_of_the_transition():
    """A is not a separate Predictor, it is B with the conditioning shrunk away.

    This identity is what turns "does routing history help?" into "how far below
    infinity does alpha land?", so it is worth pinning rather than assuming.
    """
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    rows = grid_rows(r, np.arange(r.n_tokens))

    pop = score("popularity", r, tables, rows, layer=1)
    assert np.allclose(pop, tables.popularity[1][None, :])

    shrunk = score("cross_layer", r, tables, rows, layer=1, alpha=1e12)
    assert np.allclose(shrunk, pop, atol=1e-6)


def test_beta_reweights_conditioners_and_beta_zero_does_not():
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    rows = grid_rows(r, np.arange(r.n_tokens))
    unweighted = score("cross_layer", r, tables, rows, layer=1, beta=0.0)
    weighted = score("cross_layer", r, tables, rows, layer=1, beta=1.0)
    assert not np.allclose(unweighted, weighted)


def test_zero_alpha_ignores_popularity_where_counts_exist():
    r = _routing()
    rows = np.arange(r.n_tokens)
    tables = fit_tables(r, rows)
    s = score("cross_layer", r, tables, grid_rows(r, rows), layer=1, alpha=0.0)
    # With no smoothing, a Slot that never followed any conditioner scores zero.
    assert np.any(s == 0.0)


def test_an_unseen_conditioner_backs_off_to_popularity_even_at_zero_alpha():
    """c(i) = 0 makes the conditional 0/0. It backs off rather than producing NaN.

    Reachable in the real run: 5.7% of conditioning rows fire fewer than 30
    times on a fit set and some fire not at all, so this is the tail, not a
    contrived case.
    """
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    tables.cross[1][:] = 0.0  # no conditioner has ever been seen

    with np.errstate(invalid="raise", divide="raise"):
        s = score("cross_layer", r, tables, grid_rows(r, np.arange(r.n_tokens)), layer=1, alpha=0.0)
    assert np.all(np.isfinite(s))
    assert np.allclose(s, tables.popularity[1][None, :])


def test_a_fitted_table_never_sees_a_scored_row():
    """Fold hygiene: counts come only from fit rows, including the previous Token.

    A Prompt split across the fit/score boundary would otherwise leak a scored
    row's Slots into ``same`` through its predecessor.
    """
    r = _routing()
    fit = r.rows_for([0, 1])
    tables = fit_tables(r, fit)

    want = np.zeros((r.n_layers, N_EXPERTS, N_EXPERTS))
    for row in fit:
        p = r.prev[row]
        if p < 0 or p not in set(fit.tolist()):
            continue
        for layer in range(r.n_layers):
            for i in r.slots[p, layer]:
                for j in r.slots[row, layer]:
                    want[layer, i, j] += 1
    assert np.array_equal(tables.same, want.astype(np.float32))

    held = r.rows_for([2, 3])
    assert tables.cross.sum() == len(fit) * (r.n_layers - 1) * TOP_K * TOP_K
    assert not np.array_equal(fit_tables(r, held).cross, tables.cross)


# --- persistence ------------------------------------------------------------


def test_persistence_names_exactly_the_previous_tokens_slots():
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    rows = grid_rows(r, np.arange(r.n_tokens))
    got = coverage(
        score("persistence", r, tables, rows, layer=1),
        r.slots[rows, 1].astype(np.int64),
        budgets=(TOP_K,),
    )[:, 0]

    prev_slots = r.slots[r.prev[rows], 1]
    want = [
        len(set(p.tolist()) & set(a.tolist())) / TOP_K
        for p, a in zip(prev_slots, r.slots[rows, 1])
    ]
    assert np.allclose(got, want)


def test_persistence_ranks_carried_slots_by_the_previous_gate():
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    rows = grid_rows(r, np.arange(r.n_tokens))
    s = score("persistence", r, tables, rows, layer=2)

    prev = r.prev[rows]
    hottest = r.slots[prev, 2][np.arange(rows.size), r.gates[prev, 2].argmax(axis=1)]
    assert np.array_equal(s.argmax(axis=1), hottest)


def test_persistence_fills_the_budget_above_top_k_with_popularity():
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    rows = grid_rows(r, np.arange(r.n_tokens))
    s = score("persistence", r, tables, rows, layer=1)

    ranked = np.argsort(-s, axis=1, kind="stable")
    carried = set(r.slots[r.prev[rows[0]], 1].tolist())
    assert set(ranked[0, :TOP_K].tolist()) == carried
    # Beyond the carried Slots the ordering is the Layer's popularity.
    rest = [j for j in ranked[0, TOP_K:] if j not in carried]
    pop = tables.popularity[1][rest]
    assert np.all(np.diff(pop) <= 0)


# --- lags -------------------------------------------------------------------


def test_lag_rows_never_reaches_into_another_prompt():
    r = _routing(n_prompts=4, n_tokens=6)
    for lag in (1, 2, 4):
        src = lag_rows(r, lag)
        have = src >= 0
        assert np.array_equal(have, r.token_id >= lag)
        assert np.all(r.prompt_id[src[have]] == r.prompt_id[have])
        assert np.all(r.token_id[src[have]] == r.token_id[have] - lag)


def test_a_lag_deeper_than_the_prompt_is_available_nowhere():
    r = _routing(n_prompts=4, n_tokens=6)
    assert np.all(lag_rows(r, 6) < 0)


def test_history_at_lag_one_is_persistence_exactly():
    """The control the whole lag experiment rests on: if ``history`` at (1,) were
    merely close to persistence, a gap at (1, 2, 4) could be the encoding rather
    than the older Tokens."""
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    rows = grid_rows(r, np.arange(r.n_tokens))
    for layer in (1, 2):
        want = score("persistence", r, tables, rows, layer)
        for decay in (0.0, 0.5, 1.0):
            got = score("history", r, tables, rows, layer, lags=(1,), decay=decay)
            assert np.allclose(got, want)
        assert np.allclose(score("history_priority", r, tables, rows, layer, lags=(1,)), want)


def test_a_union_ranked_by_recency_is_persistence_at_the_routers_own_k():
    """Strict tiers mean the union cannot displace a Slot t-1 named, only fill
    the Budget above it -- so every priority row must tie persistence at K=top_k
    and can only differ above it. If one ever does not, the tiering is broken."""
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    rows = grid_rows(r, np.arange(r.n_tokens), max_lag=4)
    actual = r.slots[rows, 1].astype(np.int64)

    s = score("history_priority", r, tables, rows, 1, lags=(1, 2, 4))
    base = coverage(score("persistence", r, tables, rows, 1), actual, budgets=(TOP_K,))
    assert np.allclose(coverage(s, actual, budgets=(TOP_K,)), base)

    # Above K=top_k it names the union and nothing else, until the union runs
    # out. Whether those Slots are worth naming is the experiment's question;
    # that they are the ones named is the invariant.
    ranked = np.argsort(-s, axis=1, kind="stable")
    for i, row in enumerate(rows):
        union = set()
        for lag in (1, 2, 4):
            union |= set(r.slots[lag_rows(r, lag)[row], 1].tolist())
        assert set(ranked[i, : len(union)].tolist()) == union
        assert set(ranked[i, :TOP_K].tolist()) == set(r.slots[r.prev[row], 1].tolist())


def test_the_history_vote_accumulates_a_slot_named_at_several_lags():
    """Recurrence across lags is the only thing older history can say that t-1
    cannot, so it has to actually add rather than overwrite."""
    r = _routing(n_prompts=1, n_tokens=5)
    r.slots[1, 0] = r.slots[3, 0]  # t-2 repeats what t-4 selected, for row 3...
    rows = np.array([3])
    tables = fit_tables(r, np.arange(r.n_tokens))
    mass = _history_mass(r, rows, 0, lags=(1, 2), decay=1.0)
    assert np.isclose(mass.sum(), 1.0)

    shared = set(r.slots[2, 0].tolist()) & set(r.slots[1, 0].tolist())
    for j in shared:
        contrib = 0.5 * (r.gates[2, 0][list(r.slots[2, 0]).index(j)]
                         + r.gates[1, 0][list(r.slots[1, 0]).index(j)])
        assert mass[0, j] > 0 and np.isclose(mass[0, j], contrib / mass.sum(), rtol=1e-5)


def test_scoring_a_lag_the_grid_does_not_support_raises():
    """Silently scoring t-4 only where it exists, while t-1 scores everywhere,
    would decide the comparison on the grid. It has to be an error."""
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    rows = grid_rows(r, np.arange(r.n_tokens), max_lag=1)
    with pytest.raises(ValueError, match="t-4 is undefined"):
        score("history", r, tables, rows, 1, lags=(1, 4))
    with pytest.raises(ValueError, match="t-4 is undefined"):
        score("history_priority", r, tables, rows, 1, lags=(1, 4))


def test_decay_zero_collapses_a_union_onto_the_most_recent_token():
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    rows = grid_rows(r, np.arange(r.n_tokens), max_lag=4)
    collapsed = score("history", r, tables, rows, 1, lags=(1, 2, 4), decay=0.0)
    assert np.allclose(collapsed, score("persistence", r, tables, rows, 1))


# --- the grid and the folds -------------------------------------------------


def test_grid_rows_drops_every_prompts_first_token():
    r = _routing()
    rows = grid_rows(r, np.arange(r.n_tokens))
    assert np.all(r.token_id[rows] > 0)
    assert rows.size == r.n_tokens - len(np.unique(r.prompt_id))


def test_a_deeper_grid_drops_every_prompts_opening_tokens():
    r = _routing(n_prompts=4, n_tokens=6)
    for max_lag in (1, 2, 4):
        rows = grid_rows(r, np.arange(r.n_tokens), max_lag=max_lag)
        assert np.all(r.token_id[rows] >= max_lag)
        assert rows.size == r.n_tokens - max_lag * len(np.unique(r.prompt_id))


@pytest.mark.skipif(not SERVE_TRACE.exists(), reason="serve3.trace fixture not present")
def test_folds_partition_the_train_split(tmp_path):
    store = _store(tmp_path)
    make_split(store, test_per_category=1)
    f = folds(store, n_folds=2)

    from routetrace.splits import prompt_ids_for_split

    assert sorted(np.concatenate(f).tolist()) == sorted(prompt_ids_for_split(store, "train"))
    assert len(set.intersection(*[set(x.tolist()) for x in f])) == 0


def test_unknown_predictor_is_rejected():
    r = _routing()
    tables = fit_tables(r, np.arange(r.n_tokens))
    with pytest.raises(ValueError, match="unknown predictor"):
        score("markov", r, tables, grid_rows(r, np.arange(r.n_tokens)), layer=1)
