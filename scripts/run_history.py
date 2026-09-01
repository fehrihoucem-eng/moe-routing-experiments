#!/usr/bin/env python3
"""How much routing information survives past the previous Token, and where.

Follow-up 4 of ``docs/results/predictor-coverage.md``, widened: that result
measured only t-1, and a prefetcher would happily use t-2 or t-8 because they
cost nothing extra in lead time -- a Lag is not a Horizon, and t-4 was known 200
ms ago. So the question is not whether older Tokens are *available*, it is
whether they say anything t-1 has not already said.

Train only, 5-fold by Prompt, macro-averaged by Prompt, Layers 1-39 of Decode
(ADR-0003). The grid is the one change from the published run: every Predictor
here scores only Tokens with **8 predecessors**, because t-8 cannot enter a
Prompt's opening Tokens and handing t-1 cells its rivals cannot reach would
decide the comparison on arithmetic. That costs 5.3% of rows and no Prompts, and
it shifts every number slightly against the published table -- the t-1 row below
is this experiment's own baseline, not the 38.9% from that one.

Two ranking rules for a union, because "does older history help?" is two
questions that a single rule cannot separate:

``priority``
    The literal set union, newest Token first. At K=8 it *is* t-1, so it can
    only answer whether older history fills a surplus Budget better than
    popularity does.
``vote``
    Slots weighted by ``decay ** (lag - 1)`` times their gate, summed. Older
    Tokens can reorder t-1's own 8, so this is the rule that can answer whether
    recurrence across history improves the ranking at K=8 -- and the only one
    that can lose to t-1 as well as beat it.

Usage::

    .venv/bin/python scripts/run_history.py            # ~2 min
    .venv/bin/python scripts/run_history.py --quick    # coarse sweeps, ~40 s
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from routetrace.predict import (  # noqa: E402
    BUDGETS,
    coverage,
    fit_tables,
    folds,
    grid_rows,
    load_routing,
    score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_predictors import target_layers  # noqa: E402

STORE = Path("data/stores/corpus_v1")
CHOICE = Path("docs/results/predictor-coverage.json")
OUT_JSON = Path("docs/results/history-coverage.json")

#: The deepest Lag any Predictor in this comparison reaches. It sets the grid
#: for all of them, including t-1.
MAX_LAG = 8

#: Reported Budgets. K=8 is the Router's own top-k and the one Budget where a
#: single Token's 8 Slots can in principle be enough; 12 and 16 are where a
#: union has room to say something a copy cannot.
REPORT_AT = (8, 12, 16)
TUNE_AT = 8

DECAYS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
#: Dense below 0.3 and coarse above it. The history term is zero on 248 of 256
#: Slots, so it acts as a bonus on eight rather than as half of a blend, and the
#: optimum sits an order of magnitude lower than the 0.4 the published
#: table-vs-table mixture landed on. A uniform 0.1 grid straddles the peak.
WS = [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
QUICK_DECAYS = [0.0, 0.5, 1.0]
QUICK_WS = [0.0, 0.5, 1.0]

#: The union whose decay curve is swept and whose weight is tuned against the
#: cross-layer Predictor. Chosen before any number was looked at.
MAIN_LAGS = (1, 2, 4)


def rows_for_report(budgets) -> list[int]:
    return [list(BUDGETS).index(k) for k in budgets]


def fit_folds(r, fold_sets):
    """One :class:`Tables` per fold, fitted once and reused by every sweep.

    ~21 MB each, and the alternative is refitting the same counts inside every
    decay and every mixture weight. Nothing here depends on the Predictor, so
    there is nothing to leak by sharing them.
    """
    return [fit_tables(r, fit_rows) for fit_rows, _ in fold_sets]


def score_spec(r, fold_sets, tables_by_fold, layers, name, params):
    """One Predictor over every fold: ``(layer -> prompt -> [K], fold -> [K])``.

    The single entry point for scoring, used by the sweeps and by the result
    table alike. Splitting them was how the published study grew a sweep that
    weighted folds and a table that weighted Prompts -- a 0.02pp disagreement
    here, but the same shape of bug that cost two full re-runs there, and folds
    are 20/16/16/16/16 Prompts rather than equal, so the two are genuinely
    different statistics.
    """
    per_layer = {layer: {} for layer in layers}
    per_fold = {}
    for f, (_, score_rows) in enumerate(fold_sets):
        rows = grid_rows(r, score_rows, max_lag=MAX_LAG)
        pids = r.prompt_id[rows]
        uniq = np.unique(pids)
        acc = []
        for layer in layers:
            s = score(name, r, tables_by_fold[f], rows, layer, **params)
            cov = coverage(s, r.slots[rows, layer].astype(np.int64), budgets=BUDGETS)
            by_prompt = np.array([cov[pids == p].mean(axis=0) for p in uniq])
            for pid, row in zip(uniq, by_prompt):
                per_layer[layer][int(pid)] = row
            acc.append(by_prompt.mean(axis=0))
        per_fold[f] = np.mean(acc, axis=0)
    return per_layer, per_fold


def pooled_of(per_layer, layers) -> np.ndarray:
    """``[K]`` Coverage: macro by Prompt within a Layer, then flat over Layers."""
    return macro_of(per_layer, layers).mean(axis=0)


def macro_of(per_layer, layers) -> np.ndarray:
    """``[len(layers), K]``, averaged over Prompts within each Layer."""
    return np.array([
        np.mean([v for _, v in sorted(per_layer[layer].items())], axis=0)
        for layer in layers
    ])


def evaluate(r, fold_sets, tables_by_fold, layers, specs):
    """Every spec, on the same rows of the same folds."""
    per_prompt, per_fold = {}, {}
    for label, name, params in specs:
        per_prompt[label], per_fold[label] = score_spec(
            r, fold_sets, tables_by_fold, layers, name, params)
    return per_prompt, per_fold


def macro(per_prompt, label, layers) -> np.ndarray:
    return macro_of(per_prompt[label], layers)


def sweep_decay(r, fold_sets, tables_by_fold, layers, lags, decays):
    """Coverage at every reported Budget, for each decay. The curve *is* a result.

    Reported at all three Budgets rather than only the tuning one, because the
    published study already found the K=8 verdict inverting by K=16 once a
    Predictor runs out of Slots to name, and a decay tuned at 8 could easily be
    the wrong one at 16.
    """
    out = []
    for d in decays:
        pl, _ = score_spec(r, fold_sets, tables_by_fold, layers, "history",
                           {"lags": lags, "decay": d})
        m = pooled_of(pl, layers)
        out.append({"decay": d,
                    **{f"coverage_at_{k}": float(m[i])
                       for k, i in zip(REPORT_AT, rows_for_report(REPORT_AT))}})
    return out


def sweep_w(r, fold_sets, tables_by_fold, layers, lags, decay, ws, cross_params):
    """Mixture weight for cross-layer + temporal history, tuned at K=8 on train."""
    k = list(BUDGETS).index(TUNE_AT)
    out = []
    for w in ws:
        pl, _ = score_spec(r, fold_sets, tables_by_fold, layers, "cross_layer_history",
                           {**cross_params, "lags": lags, "decay": decay, "w": w})
        out.append({"w": w, "coverage_at_8": float(pooled_of(pl, layers)[k])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--choice", type=Path, default=CHOICE)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    r = load_routing(args.store, split="train")
    layers = list(target_layers(r.n_layers))

    fold_pids = folds(args.store, n_folds=args.folds, seed=args.seed)
    fold_sets = []
    for held in fold_pids:
        score_rows = r.rows_for(held)
        fit_rows = np.setdiff1d(np.arange(r.n_tokens), score_rows)
        fold_sets.append((fit_rows, score_rows))

    kept = grid_rows(r, np.arange(r.n_tokens), max_lag=MAX_LAG)
    print(f"train: {r.n_tokens} decode tokens, grid keeps {kept.size} "
          f"({100 * kept.size / r.n_tokens:.1f}%) with {MAX_LAG} predecessors, "
          f"{np.unique(r.prompt_id[kept]).size} prompts", file=sys.stderr)

    # The cross-layer Predictor enters at the hyperparameters the published study
    # chose for it, on train. Re-tuning it here would be tuning a rival on the
    # same folds it is about to be compared on.
    chosen = json.loads(args.choice.read_text())
    cross = dict(chosen["tuned"]["cross_layer_gated"])
    if cross.get("alpha") is None:
        cross["alpha"] = np.inf
    cross.pop("w", None)
    published_combined = dict(chosen["tuned"]["combined"])
    if published_combined.get("alpha") is None:
        published_combined["alpha"] = np.inf
    print(f"cross_layer (C) at published {cross}; "
          f"combined at published {chosen['tuned']['combined']}", file=sys.stderr)

    decays = QUICK_DECAYS if args.quick else DECAYS
    ws = QUICK_WS if args.quick else WS
    tables_by_fold = fit_folds(r, fold_sets)
    print(f"fitted {len(tables_by_fold)} folds ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)

    # --- how fast does history fade? ----------------------------------------
    decay_surface = sweep_decay(r, fold_sets, tables_by_fold, layers, MAIN_LAGS, decays)
    best_decay = max(decay_surface, key=lambda d: d[f"coverage_at_{TUNE_AT}"])["decay"]
    print(f"tuned decay for {MAIN_LAGS}: {best_decay} ({time.perf_counter() - t0:.0f}s)",
          file=sys.stderr)

    # One w per mixture, not one w shared. The question "does adding older Tokens
    # to the cross-layer Predictor help?" is only answerable if both sides are at
    # their own best weight; a w tuned for three lags and then handed to the
    # one-lag mixture would answer a different question badly.
    w_surface, best_w = {}, {}
    for key, lags in (("t-1", (1,)), ("history", MAIN_LAGS)):
        got = sweep_w(r, fold_sets, tables_by_fold, layers, lags, best_decay, ws, cross)
        w_surface[key] = got
        best_w[key] = max(got, key=lambda d: d["coverage_at_8"])["w"]
        print(f"tuned w for cross+{key}: {best_w[key]} ({time.perf_counter() - t0:.0f}s)",
              file=sys.stderr)

    # --- the table ----------------------------------------------------------
    specs = [
        ("popularity", "popularity", {}),
        ("t-1", "history", {"lags": (1,)}),
        ("t-2", "history", {"lags": (2,)}),
        ("t-4", "history", {"lags": (4,)}),
        ("t-8", "history", {"lags": (8,)}),
        ("t-1 u t-2 (priority)", "history_priority", {"lags": (1, 2)}),
        ("t-1 u t-4 (priority)", "history_priority", {"lags": (1, 4)}),
        ("t-1 u t-2 u t-4 (priority)", "history_priority", {"lags": (1, 2, 4)}),
        # Not asked for, but one line and it is the only row that says whether
        # the union saturates or just keeps buying Budget.
        ("t-1 u t-2 u t-4 u t-8 (priority)", "history_priority", {"lags": (1, 2, 4, 8)}),
        ("t-1 u t-2 (vote)", "history", {"lags": (1, 2), "decay": best_decay}),
        ("t-1 u t-4 (vote)", "history", {"lags": (1, 4), "decay": best_decay}),
        ("t-1 u t-2 u t-4 (vote)", "history", {"lags": (1, 2, 4), "decay": best_decay}),
        ("cross-layer (C)", "cross_layer", cross),
        # The published winner, re-scored on this grid so the new mixtures have
        # something to be compared against that was not fitted here.
        ("combined (B+D, published)", "combined", published_combined),
        ("cross-layer + t-1", "cross_layer_history",
         {**cross, "lags": (1,), "w": best_w["t-1"]}),
        ("cross-layer + history", "cross_layer_history",
         {**cross, "lags": MAIN_LAGS, "decay": best_decay, "w": best_w["history"]}),
    ]
    per_prompt, per_fold = evaluate(r, fold_sets, tables_by_fold, layers, specs)
    print(f"scored {len(specs)} predictors ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)

    labels = [s[0] for s in specs]
    result = {
        "store": str(args.store),
        "split": "train",
        "n_folds": args.folds,
        "seed": args.seed,
        "max_lag": MAX_LAG,
        "grid_rows": int(kept.size),
        "grid_rows_total": int(r.n_tokens),
        "n_prompts": int(np.unique(r.prompt_id[kept]).size),
        "budgets": list(BUDGETS),
        "report_at": list(REPORT_AT),
        "tune_at": TUNE_AT,
        "layers": layers,
        "chance": [k / r.n_experts for k in BUDGETS],
        "cross_params": {k: (None if k == "alpha" and not np.isfinite(v) else v)
                         for k, v in cross.items()},
        "main_lags": list(MAIN_LAGS),
        "tuned_decay": best_decay,
        "tuned_w": best_w,
        "decay_surface": decay_surface,
        "w_surface": w_surface,
        "specs": [{"label": lab, "predictor": n,
                   "params": {k: (list(v) if isinstance(v, tuple) else
                                  (None if k == "alpha" and not np.isfinite(v) else v))
                              for k, v in p.items()}}
                  for lab, n, p in specs],
        "pooled": {},
        "per_layer": {},
        "fold_spread": {},
        "thirds": {},
    }

    third = {"early": range(1, 14), "mid": range(14, 27), "late": range(27, 40)}
    for label in labels:
        curves = macro(per_prompt, label, layers)
        result["per_layer"][label] = curves.tolist()
        result["pooled"][label] = curves.mean(axis=0).tolist()
        result["fold_spread"][label] = [per_fold[label][f].tolist() for f in range(args.folds)]
        result["thirds"][label] = {
            name: curves[[layers.index(x) for x in rng]].mean(axis=0).tolist()
            for name, rng in third.items()
        }

    # Paired per-fold differences: every Predictor saw the same five folds, so
    # the comparison resolves far finer than the unpaired spread suggests.
    def paired(a, b):
        d = np.array([np.array(result["fold_spread"][a][f]) -
                      np.array(result["fold_spread"][b][f]) for f in range(args.folds)])
        return {f"K={k}": {"mean": float(d[:, i].mean() * 100),
                           "sd": float(d[:, i].std(ddof=1) * 100),
                           "min": float(d[:, i].min() * 100),
                           "max": float(d[:, i].max() * 100)}
                for k, i in zip(REPORT_AT, rows_for_report(REPORT_AT))}

    result["paired"] = {
        f"{a} over {b}": paired(a, b)
        for a, b in [
            ("t-1 u t-2 u t-4 (priority)", "t-1"),
            ("t-1 u t-2 u t-4 u t-8 (priority)", "t-1 u t-2 u t-4 (priority)"),
            ("t-1 u t-2 (priority)", "t-1"),
            ("t-1 u t-2 u t-4 (vote)", "t-1"),
            ("t-1 u t-2 (vote)", "t-1"),
            ("t-2", "t-8"),
            ("cross-layer + history", "cross-layer + t-1"),
            ("cross-layer + history", "cross-layer (C)"),
            ("cross-layer + t-1", "combined (B+D, published)"),
            ("cross-layer + history", "combined (B+D, published)"),
        ]
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")

    idx = rows_for_report(REPORT_AT)
    print(f"\n{'predictor':<28}" + "".join(f"  K={k:<5}" for k in REPORT_AT), file=sys.stderr)
    for label in labels:
        cells = "".join(f"  {result['pooled'][label][i] * 100:6.2f}" for i in idx)
        print(f"{label:<28}{cells}", file=sys.stderr)
    print(f"\nwrote {args.out} in {time.perf_counter() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
