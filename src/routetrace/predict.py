"""Predictors of routing, and the Coverage they achieve.

A **Predictor** names ``K`` Slots for a target (Token, Layer) from routing that
is already known. The one number it is judged on is **Coverage**: of the 8
Experts the Router actually selected, the fraction the Predictor named. Coverage
is the whole story -- with the Router always selecting exactly 8, precision is
``Coverage * 8 / K`` and carries nothing extra, which is why this module exposes
no precision function for a caller to mistake for a second axis.

Predictors differ in their **Horizon**, and only Predictors sharing one are
comparable:

``token``
    Everything about the Tokens before t, known ~50 ms before Token t enters the
    model at all (40 Layers at ~20 tok/s) and longer still for the older ones.
    ``popularity``, ``persistence``, ``prev_token``, ``history``,
    ``history_priority``. A lag costs nothing extra in lead time -- t-4 was known
    200 ms ago -- so within this Horizon reaching further back is free, and the
    only question is whether it says anything t-1 has not already said.
``layer``
    That, plus the current Token's Slots one Layer down -- ~1.25 ms of warning.
    ``cross_layer`` and the combination.

The Horizon gap is the point of the exercise. One Expert is 1.6 MiB at int4
gs64, so a single miss costs ~0.33 ms to fetch at 5 GB/s and eight cost ~2.7 ms:
the ``layer`` Horizon cannot cover its own miss, while the ``token`` Horizon has
40x the room. A ``layer`` Predictor therefore has to win by a wide margin to be
worth anything, and this module exists to find out whether it does.

Every scoring Predictor here is one estimator with two knobs::

    P(j|i) = (c(i,j) + alpha * pop(j)) / (c(i) + alpha)
    score(j) = sum_i (g_i ** beta) * P(j|i)

``alpha`` shrinks a conditional row toward the target Layer's popularity, which
matters because the tail is thin: on a cross-validation fit set the median
(Layer, Slot) fires ~246 times but the 10th percentile fires 58, and 5.7% of
conditioning rows fire fewer than 30 times. The nesting is the useful part --
``alpha=inf`` *is* popularity and ``beta=0`` *is* the unweighted transition, so
"does routing history help?" becomes "how far below infinity does alpha land?"
rather than a comparison between separately built artefacts.

Fitting and scoring both happen inside the train Split; see ADR-0003.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .store import read_meta, read_prompts
from .tensor import COO, load_X

TOKEN_HORIZON = "token"
LAYER_HORIZON = "layer"

#: Budgets the K-curve is reported at. 8 is the Router's own top-k, so Coverage
#: below it is capped by arithmetic alone: at K=5 the ceiling is 5/8 = 62.5%.
BUDGETS = (1, 3, 5, 8, 12, 16)


@dataclass
class Routing:
    """Decode routing in its compact form: the Slots that fired, not a dense X.

    ``slots`` and ``gates`` are ``[tokens, layers, top_k]``, ordered by ascending
    Slot within each (Token, Layer) -- not by gate, since the selection is the
    object and the gate is an attribute of it. This is ~30 MB for the whole
    corpus where dense X is 638 MB, and every Predictor here wants the 8 Slots
    rather than 248 zeros.

    ``prev`` is the row index of the same Prompt's previous Token, or -1 at a
    Prompt's first Token. It is the only thing that makes the ``token`` Horizon
    expressible, and it never crosses a Prompt boundary: Prompts are independent
    by construction (ADR-0001), so Token 0 of one Prompt has no predecessor even
    though a row for it exists earlier in the array.
    """

    slots: np.ndarray
    gates: np.ndarray
    prompt_id: np.ndarray
    token_id: np.ndarray
    prev: np.ndarray
    n_experts: int

    @property
    def n_tokens(self) -> int:
        return int(self.slots.shape[0])

    @property
    def n_layers(self) -> int:
        return int(self.slots.shape[1])

    @property
    def top_k(self) -> int:
        return int(self.slots.shape[2])

    def rows_for(self, prompt_ids) -> np.ndarray:
        """Row indices belonging to the given Prompts."""
        return np.flatnonzero(np.isin(self.prompt_id, np.asarray(list(prompt_ids))))


def load_routing(store_dir: str | Path, split: str | None = "train", **kwargs) -> Routing:
    """Load decode routing compactly. ``split`` and ``categories`` pass through.

    Decode only, and not by preference: all Prefill Tokens of a Prompt are
    computed in one Forward, so Token t-1's routing is never known before Token
    t's. The ``token`` Horizon is not merely weaker in Prefill, it is undefined,
    and a table where half the Predictors have no column is not a comparison.
    """
    coo, index = load_X(store_dir, phase="decode", split=split, sparse=True, **kwargs)
    assert isinstance(coo, COO)
    meta = read_meta(store_dir)
    top_k = int(meta["top_k"])
    if int(meta["top_k_min"]) != top_k:
        raise ValueError(
            f"store has ragged top-k ({meta['top_k_min']}..{top_k}); the compact "
            "form assumes every (token, layer) selected exactly top_k experts"
        )

    token, layer, expert = coo.coords
    order = np.lexsort((expert, layer, token))
    n_tokens, n_layers, _ = coo.shape
    shape = (n_tokens, n_layers, top_k)
    if order.size != n_tokens * n_layers * top_k:
        raise ValueError(f"expected {n_tokens * n_layers * top_k} rows, got {order.size}")

    prompt_id = index["prompt_id"].astype(np.int32)
    token_id = index["token_id"].astype(np.int32)
    # Same Prompt and consecutive position: load_X orders the token axis by
    # (prompt_id, phase, token_id), so the predecessor is the row before -- but
    # only check, never assume, because a categories= filter can drop Prompts and
    # a future capture could interleave them.
    prev = np.full(n_tokens, -1, dtype=np.int64)
    contiguous = np.flatnonzero(
        (prompt_id[1:] == prompt_id[:-1]) & (token_id[1:] == token_id[:-1] + 1)
    )
    prev[contiguous + 1] = contiguous

    return Routing(
        slots=expert[order].reshape(shape).astype(np.int16),
        gates=coo.values[order].reshape(shape).astype(np.float32),
        prompt_id=prompt_id,
        token_id=token_id,
        prev=prev,
        n_experts=int(coo.shape[2]),
    )


@dataclass
class Tables:
    """Counts fitted on one fold's fit rows. Nothing here has seen a scored row.

    ``popularity`` is ``[layers, n_experts]`` normalised per Layer. ``cross`` and
    ``same`` are ``[layers, n_experts, n_experts]`` raw co-occurrence counts
    ``c(i, j)`` indexed by *target* Layer: ``cross[L]`` conditions on Layer L-1 of
    the same Token, ``same[L]`` on Layer L of the previous Token. Both are
    undefined at their first index (``cross[0]`` has no Layer below it) and left
    as zeros there rather than dropped, so the Layer axis keeps meaning L.

    Counts are unweighted even for the gate-weighted Predictor: ``beta`` weights
    the *conditioners* at scoring time, so one fitted table serves every beta.

    Both axes are Slot-indexed, and an entry denotes an Expert only because the
    Layer is pinned by the table it sits in. This is the one place the project's
    rule against naming an Expert by its Slot alone is bent, and it is safe only
    because a table is never summed across its first axis.
    """

    popularity: np.ndarray
    cross: np.ndarray
    same: np.ndarray


def fit_tables(r: Routing, rows: np.ndarray) -> Tables:
    """Count popularity and both transition families over ``rows``."""
    n_l, n_e = r.n_layers, r.n_experts
    slots = r.slots[rows].astype(np.int64)

    pop = np.zeros((n_l, n_e), dtype=np.float64)
    for layer in range(n_l):
        pop[layer] = np.bincount(slots[:, layer, :].ravel(), minlength=n_e)
    pop /= pop.sum(axis=1, keepdims=True)

    cross = np.zeros((n_l, n_e, n_e), dtype=np.float32)
    for layer in range(1, n_l):
        cross[layer] = _pair_counts(slots[:, layer - 1, :], slots[:, layer, :], n_e)

    # The previous Token must itself be in the fit set: a fold boundary that cut
    # mid-Prompt would otherwise let a scored row's Slots into the fitted counts.
    in_fit = np.zeros(r.n_tokens, dtype=bool)
    in_fit[rows] = True
    prev = r.prev[rows]
    usable = (prev >= 0) & in_fit[np.maximum(prev, 0)]
    same = np.zeros((n_l, n_e, n_e), dtype=np.float32)
    if usable.any():
        before = r.slots[prev[usable]].astype(np.int64)
        after = slots[usable]
        for layer in range(n_l):
            same[layer] = _pair_counts(before[:, layer, :], after[:, layer, :], n_e)

    return Tables(popularity=pop, cross=cross, same=same)


def _pair_counts(before: np.ndarray, after: np.ndarray, n_e: int) -> np.ndarray:
    """``c(i, j)`` over every (i in before, j in after) pair, per row.

    bincount over a flattened i*n_e+j key rather than ``np.add.at``, which is an
    unbuffered scatter and roughly two orders of magnitude slower at the ~600k
    pairs per Layer this sees.
    """
    keys = (before[:, :, None] * n_e + after[:, None, :]).ravel()
    return np.bincount(keys, minlength=n_e * n_e).reshape(n_e, n_e).astype(np.float32)


def _mixture(counts, pop, cond_slots, cond_gates, alpha, beta):
    """``score(j) = sum_i (g_i ** beta) * (c(i,j) + alpha*pop(j)) / (c(i) + alpha)``.

    Expanded so the smoothed 256x256 is never materialised: the count term is
    scaled per conditioning row once, and the backoff term collapses to a scalar
    per scored row times the popularity vector. ``alpha=inf`` short-circuits to
    popularity, which is the identity that makes A a special case of B rather
    than a separate Predictor.
    """
    weights = np.ones_like(cond_gates) if beta == 0 else cond_gates**beta
    if not np.isfinite(alpha):
        out = np.repeat(pop[None, :], cond_slots.shape[0], axis=0) * weights.sum(1, keepdims=True)
        return out / out.sum(1, keepdims=True)

    # A conditioning Slot that never fired in the fit set has c(i) = 0, so at
    # alpha=0 the conditional is 0/0. It is not undefined in any useful sense --
    # the row carries no information -- so it backs off entirely to popularity,
    # which is what every alpha > 0 already does in the limit.
    denom = counts.sum(axis=1) + alpha
    empty = denom == 0
    safe = np.where(empty, 1.0, denom)
    scaled = counts / safe[:, None]
    out = np.zeros((cond_slots.shape[0], pop.size), dtype=np.float64)
    for k in range(cond_slots.shape[1]):
        out += weights[:, k, None] * scaled[cond_slots[:, k]]
    backoff = (weights * np.where(empty, 1.0, alpha / safe)[cond_slots]).sum(axis=1)
    out += backoff[:, None] * pop[None, :]
    return out / out.sum(axis=1, keepdims=True)


def score(
    name: str,
    r: Routing,
    tables: Tables,
    rows: np.ndarray,
    layer: int,
    alpha: float = 10.0,
    beta: float = 0.0,
    w: float = 0.5,
    lags: tuple[int, ...] = (1,),
    decay: float = 1.0,
) -> np.ndarray:
    """``[len(rows), n_experts]`` scores for one Predictor at one target Layer.

    Higher is better; each row sums to 1 so that scores from different Predictors
    can be mixed. ``rows`` must already be restricted to the scoring grid.
    """
    # Negative indexing would make layer=0 quietly condition on Layer 39 and
    # return plausible numbers for a cell no cross-layer Predictor can enter.
    if layer < 1 and name != "popularity":
        raise ValueError(f"{name!r} has no Layer below {layer}; the grid starts at 1")

    pop = tables.popularity[layer]
    if name == "popularity":
        # The alpha -> inf limit of every other conditional Predictor here, and
        # computed directly rather than via _mixture so that it stays defined on
        # Layer 0, where it is the only Predictor that has anything to say. The
        # nesting is pinned by test_popularity_is_the_infinite_alpha_limit.
        return np.repeat(pop[None, :], rows.size, axis=0)
    if name == "persistence":
        return _persistence(r, rows, layer, pop)
    if name == "prev_token":
        prev = r.prev[rows]
        return _mixture(tables.same[layer], pop, r.slots[prev, layer].astype(np.int64),
                        r.gates[prev, layer], alpha, beta)
    if name == "cross_layer":
        return _mixture(tables.cross[layer], pop, _cond(r, rows, layer - 1),
                        _gates(r, rows, layer - 1), alpha, beta)
    if name == "cross_layer_raw":
        return _raw_counts(tables.cross[layer], pop, _cond(r, rows, layer - 1))
    if name == "combined":
        a = score("prev_token", r, tables, rows, layer, alpha, beta)
        b = score("cross_layer", r, tables, rows, layer, alpha, beta)
        return w * a + (1.0 - w) * b
    if name == "history":
        return _history(r, rows, layer, pop, lags, decay)
    if name == "history_priority":
        return _history_priority(r, rows, layer, pop, lags)
    if name == "cross_layer_history":
        # The history side enters as its own mass, not as the ranked form: a
        # mixture needs two distributions on one scale, and _history's (1, 2]
        # encoding is an ordering, not a scale. Popularity is left out of the
        # history term entirely here because the cross-layer term already
        # scores every Slot.
        h = _history_mass(r, rows, layer, lags, decay)
        b = _mixture(tables.cross[layer], pop, _cond(r, rows, layer - 1),
                     _gates(r, rows, layer - 1), alpha, beta)
        return w * h + (1.0 - w) * b
    raise ValueError(f"unknown predictor {name!r}")


def _raw_counts(counts, pop, cond_slots, top_m: int = 16):
    """colibri's shipped variant, for comparison: raw counts, truncated at top-M.

    ``c/tools/route_pairs.py`` writes only each conditioning event's top-16
    followers and stores raw co-occurrences with no per-conditioner
    normalisation, because "the consumer only needs their ranking". The engine
    then sums those counts over the observed Slot set (``COUPLE=``). That makes a
    busy conditioner dominate the sum in a way ``P(j|i)`` does not, which is the
    difference this Predictor exists to measure -- its docstring claims
    +3.6..+9.4pp over marginal heat, and reproducing that band is the check that
    our own harness is sound.

    Popularity is added as a tie-break only: counts are integer-valued so any two
    distinct scores differ by at least 1, and ``pop`` sums to 1, so it can only
    order Slots the truncated table scores equally -- most of them, since 240 of
    256 are zeroed per conditioner.
    """
    keep = np.zeros_like(counts)
    idx = np.argpartition(-counts, top_m, axis=1)[:, :top_m]
    np.put_along_axis(keep, idx, np.take_along_axis(counts, idx, axis=1), axis=1)

    out = np.zeros((cond_slots.shape[0], pop.size), dtype=np.float64)
    for k in range(cond_slots.shape[1]):
        out += keep[cond_slots[:, k]]
    out += pop[None, :]
    return out / out.sum(axis=1, keepdims=True)


def lag_rows(r: Routing, lag: int) -> np.ndarray:
    """``[n_tokens]`` row index of the Token ``lag`` positions back, or -1.

    Built by walking ``prev`` ``lag`` times rather than by subtracting from
    ``token_id``, so it inherits that field's Prompt-boundary guarantee instead
    of re-deriving it: a Token 3 positions into its Prompt has no t-4, even
    though a row 4 earlier in the array exists and belongs to another Prompt.
    Walking also stays correct if a ``categories=`` filter has removed Prompts
    from between.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    idx = np.arange(r.n_tokens, dtype=np.int64)
    for _ in range(lag):
        idx = np.where(idx >= 0, r.prev[np.maximum(idx, 0)], -1)
    return idx


def _lag_weights(lags, decay: float) -> tuple[tuple[int, ...], np.ndarray]:
    """``decay ** (lag - 1)``, normalised to sum to 1.

    Geometric in the Token distance, not in the position within ``lags``, so the
    weight a lag receives does not change when a different lag is added beside
    it -- ``t-4`` is weighted the same in ``(1, 4)`` as in ``(1, 2, 4)``, which
    is what makes those two rows of the result comparable. ``decay = 1`` is an
    equal vote; ``decay = 0`` collapses to ``t-1`` alone, which is the identity
    that makes every union here a superset of persistence rather than a rival
    to it.
    """
    lags = tuple(sorted({int(x) for x in lags}))
    if not lags:
        raise ValueError("lags must name at least one Token distance")
    w = np.array([float(decay) ** (x - 1) for x in lags], dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError(f"lags {lags} at decay={decay} carry no weight")
    return lags, w / w.sum()


def _history_mass(r: Routing, rows: np.ndarray, layer: int, lags, decay: float) -> np.ndarray:
    """``sum_lag w(lag) * gate_{t-lag}(j)``: a distribution over Slots, from history alone.

    Each lag contributes its own renormalised gate vector, so the result sums to
    1 and is directly mixable with a fitted Predictor's scores. A Slot selected
    at several lags accumulates -- that accumulation *is* the hypothesis under
    test, since a Slot the Router keeps returning to is the only thing older
    history can say that ``t-1`` cannot.

    Slots named at no lag score exactly 0. That is deliberate: this function is
    the evidence from history and nothing else, and a caller who wants the
    remaining Budget filled adds the filler it prefers (:func:`_history` uses
    popularity, ``cross_layer_history`` uses the Layer below).
    """
    lags, weights = _lag_weights(lags, decay)
    out = np.zeros((rows.size, r.n_experts), dtype=np.float64)
    buf = np.empty_like(out)
    for weight, lag in zip(weights, lags):
        src = lag_rows(r, lag)[rows]
        missing = int((src < 0).sum())
        if missing:
            raise ValueError(
                f"t-{lag} is undefined for {missing} of {rows.size} scored rows; "
                f"restrict the grid with grid_rows(..., max_lag={max(lags)}) so "
                "every lag scores the same cells"
            )
        buf.fill(0.0)
        np.put_along_axis(
            buf,
            r.slots[src, layer].astype(np.int64),
            weight * r.gates[src, layer].astype(np.float64),
            axis=1,
        )
        out += buf
    return out / out.sum(axis=1, keepdims=True)


def _history(r: Routing, rows: np.ndarray, layer: int, pop: np.ndarray, lags, decay: float):
    """The history vote, ranked, with popularity filling the Budget below it.

    Same encoding as :func:`_persistence` -- carried Slots in ``(1, 2]``, the
    rest in ``[0, 1)`` -- so that ``lags=(1,)`` reproduces persistence exactly
    at every Budget rather than approximately. That identity is the control for
    this whole experiment: any gap between ``history`` at ``(1,)`` and at
    ``(1, 2, 4)`` is older history and cannot be anything else.
    """
    mass = _history_mass(r, rows, layer, lags, decay)
    floor = np.repeat((pop / (pop.max() + 1.0))[None, :], rows.size, axis=0)
    out = np.where(mass > 0.0, 1.0 + mass, floor)
    return out / out.sum(axis=1, keepdims=True)


def _history_priority(r: Routing, rows: np.ndarray, layer: int, pop: np.ndarray, lags):
    """The literal set union, ranked newest-Token-first: t-1's 8, then t-2's, then t-4's.

    Recency is a strict tier, so at K=8 this *is* persistence -- the union
    cannot displace a Slot t-1 named, only fill the Budget above it. That makes
    the priority rows answer one question cleanly ("does older history fill a
    surplus Budget better than popularity does?") and say nothing at all about
    the other ("can older history improve the ranking?"), which is what the vote
    rows are for. Reporting both is the only way to tell those two apart.

    Within a tier the ordering is that Token's own gates, and a Slot named at
    several lags takes its most recent tier: lags are written oldest-first so
    the newest overwrites.
    """
    lags = tuple(sorted({int(x) for x in lags}))
    out = np.repeat((pop / (pop.max() + 1.0))[None, :], rows.size, axis=0)
    for i in range(len(lags) - 1, -1, -1):
        src = lag_rows(r, lags[i])[rows]
        missing = int((src < 0).sum())
        if missing:
            raise ValueError(
                f"t-{lags[i]} is undefined for {missing} of {rows.size} scored rows; "
                f"restrict the grid with grid_rows(..., max_lag={max(lags)})"
            )
        np.put_along_axis(
            out,
            r.slots[src, layer].astype(np.int64),
            (len(lags) - i) + r.gates[src, layer].astype(np.float64),
            axis=1,
        )
    return out / out.sum(axis=1, keepdims=True)


def _cond(r: Routing, rows: np.ndarray, layer: int) -> np.ndarray:
    return r.slots[rows, layer].astype(np.int64)


def _gates(r: Routing, rows: np.ndarray, layer: int) -> np.ndarray:
    return r.gates[rows, layer].astype(np.float64)


def _persistence(r: Routing, rows: np.ndarray, layer: int, pop: np.ndarray) -> np.ndarray:
    """Name the previous Token's own Slots, ranked by the gate they got there.

    The only Predictor that ships no table: free at runtime, with the full
    ``token`` Horizon. When K < 8 some of the 8 must be dropped, and the previous
    gate is the Router's own confidence ordering -- already captured, and the
    only ranking available that is not arbitrary. Below the 8, popularity fills
    the remaining Budget.

    Encoded as scores in ``(1, 2]`` for the carried Slots and ``[0, 1)`` for the
    rest, so that one top-K over the result reproduces exactly that ordering.
    """
    prev = r.prev[rows]
    out = pop[None, :] / (pop.max() + 1.0)
    out = np.repeat(out, rows.size, axis=0)
    carried = r.slots[prev, layer].astype(np.int64)
    np.put_along_axis(out, carried, 1.0 + r.gates[prev, layer].astype(np.float64), axis=1)
    return out / out.sum(axis=1, keepdims=True)


def coverage(scores: np.ndarray, actual: np.ndarray, budgets=BUDGETS) -> np.ndarray:
    """``[len(scores), len(budgets)]`` Coverage: fraction of ``actual`` named.

    Ties break toward the lower Slot index, via a stable sort on the negated
    scores. That only bites where scores are exactly equal -- popularity ties
    among never-fired Slots, mostly -- but it has to be pinned or two runs
    disagree at K=1.
    """
    ranked = np.argsort(-scores, axis=1, kind="stable")
    hit = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(hit, actual, True, axis=1)
    taken = np.take_along_axis(hit, ranked[:, : max(budgets)], axis=1)
    cum = np.cumsum(taken, axis=1)
    return cum[:, np.asarray(budgets) - 1] / actual.shape[1]


def folds(store_dir: str | Path, n_folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Stratified Prompt folds over the train Split, for cross-validation.

    Prompts, never Tokens, and stratified by Category for the same reason the
    train/test Split is (ADR-0001). Assignment is round-robin over each
    Category's shuffled Prompts, so the folds are as even as 16 Prompts per
    Category allow -- 4/3/3/3/3 -- rather than exactly equal.
    """
    from .splits import prompt_ids_for_split

    train = set(prompt_ids_for_split(store_dir, "train"))
    by_cat: dict[str, list[int]] = {}
    for row in read_prompts(store_dir).to_pylist():
        if row["prompt_id"] in train:
            by_cat.setdefault(row["category"] or "", []).append(int(row["prompt_id"]))

    rng = np.random.default_rng(seed)
    out: list[list[int]] = [[] for _ in range(n_folds)]
    for cat in sorted(by_cat):
        ids = np.array(sorted(by_cat[cat]))
        for i, pid in enumerate(rng.permutation(ids)):
            out[i % n_folds].append(int(pid))
    return [np.array(sorted(f)) for f in out]


def grid_rows(r: Routing, rows: np.ndarray, max_lag: int = 1) -> np.ndarray:
    """Restrict ``rows`` to the scoring grid: Tokens that have ``max_lag`` predecessors.

    The Layer half of the grid (1..39) is enforced by the caller's Layer loop.
    Together they are the common grid of ADR-0003: every Predictor scores the
    same cells, because a Predictor handed cells its rivals cannot enter wins on
    the arithmetic rather than on the signal.

    ``max_lag`` is the deepest history any Predictor in the comparison reaches
    back to, and it must be set from the *comparison*, not from the Predictor
    being scored. Letting t-8 score only where 8 predecessors exist while t-1
    scores everywhere would hand them different cells -- and not randomly
    different ones, since the dropped cells are every Prompt's opening Tokens,
    which is exactly where routing has the least history to be predicted from.
    """
    return rows[lag_rows(r, max_lag)[rows] >= 0]
