#!/usr/bin/env python3
"""Fit and score every Predictor, and write the Coverage result.

Everything here happens inside the train Split, by 5-fold cross-validation over
Prompts (ADR-0003). The test Split is not read, not even to report it: this
script chooses a Predictor, and choosing is exactly what the holdout must not
inform. Confirming the winner on test is a separate, once-only run --
``confirm_on_test.py``.

Usage::

    .venv/bin/python scripts/run_predictors.py            # ~8 min
    .venv/bin/python scripts/run_predictors.py --quick    # coarse grid, ~1 min
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
    LAYER_HORIZON,
    TOKEN_HORIZON,
    coverage,
    fit_tables,
    folds,
    grid_rows,
    load_routing,
    score,
)
from routetrace.store import read_prompts  # noqa: E402

STORE = Path("data/stores/corpus_v1")
OUT_JSON = Path("docs/results/predictor-coverage.json")

ALPHAS = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, np.inf]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
WS = [round(0.1 * i, 1) for i in range(11)]
QUICK_ALPHAS = [0.0, 10.0, 1000.0, np.inf]
QUICK_BETAS = [0.0, 1.0]
QUICK_WS = [0.0, 0.5, 1.0]

#: Selection criterion for the hyperparameters. K=8 is the Router's own top-k --
#: the one Budget where full Coverage is arithmetically reachable, so it is the
#: least distorted point on the curve to tune at.
TUNE_AT = 8

# The four Predictors as specified, plus persistence (the free bar the token
# Horizon has to clear), colibri's shipped variant, and the combination that is
# the honest ceiling of the layer Horizon.
PREDICTORS = [
    ("popularity", TOKEN_HORIZON, "A: the Layer's hottest Slots, no history"),
    ("persistence", TOKEN_HORIZON, "D0: Token t-1's own Slots, ranked by its gates"),
    ("prev_token", TOKEN_HORIZON, "D: P(E_t | E_t-1), same Layer"),
    ("cross_layer", LAYER_HORIZON, "B: P(E_l+1 | E_l), unweighted (beta=0)"),
    ("cross_layer_gated", LAYER_HORIZON, "C: the same, gate-weighted (beta=1)"),
    ("cross_layer_raw", LAYER_HORIZON, "B-raw: colibri's raw counts, top-16"),
    ("combined", LAYER_HORIZON, "B+D: the layer Horizon's ceiling"),
]

# beta is pinned for the canonical B and C rows; the tuned surface is reported
# separately so the table stays the one that was asked for.
CANONICAL = {"cross_layer": {"beta": 0.0}, "cross_layer_gated": {"beta": 1.0}}
ALIAS = {"cross_layer_gated": "cross_layer"}


def target_layers(n_layers: int) -> range:
    """Layers 1..n-1: the common grid. Layer 0 has nothing below it, so no
    cross-layer Predictor can enter it, and it is also the Layer the literature
    singles out as anomalous (Mixtral measures consecutive-Token repetition
    there at chance). Granting popularity a Layer its rivals cannot reach would
    decide the comparison on the grid rather than on the signal."""
    return range(1, n_layers)


def _fold_scores(r, tables, rows, name, layer, params):
    return score(ALIAS.get(name, name), r, tables, rows, layer, **params)


def sweep(r, fold_sets, name, params_list, layers) -> list[tuple[dict, float]]:
    """Coverage@TUNE_AT for each parameter set, on the same metric as the result.

    Macro-averaged by Prompt, exactly as :func:`macro` does, so a sweep number is
    directly comparable to a table number. Tuning on a cell-weighted mean and
    reporting a Prompt-weighted one would put two different measurements under
    one name -- a small discrepancy here, given how flat the surface is, but the
    kind that costs an hour to explain later.
    """
    out = []
    for params in params_list:
        per_layer = []
        for fit_rows, score_rows in fold_sets:
            tables = fit_tables(r, fit_rows)
            rows = grid_rows(r, score_rows)
            pids = r.prompt_id[rows]
            uniq = np.unique(pids)
            for layer in layers:
                s = _fold_scores(r, tables, rows, name, layer, params)
                cov = coverage(s, r.slots[rows, layer].astype(np.int64), budgets=(TUNE_AT,))
                per_layer.append(np.mean([cov[pids == p].mean() for p in uniq]))
        out.append((params, float(np.mean(per_layer))))
    return out


def evaluate(r, fold_sets, layers, tuned, budgets, names=None):
    """Per-(Predictor, Layer, Prompt) Coverage. Macro-averaging happens after."""
    names = names or [n for n, _, _ in PREDICTORS]
    per_prompt = defaultdict(lambda: defaultdict(dict))  # name -> layer -> pid -> [K]
    per_fold = defaultdict(lambda: defaultdict(list))  # name -> fold -> [K]

    for f, (fit_rows, score_rows) in enumerate(fold_sets):
        tables = fit_tables(r, fit_rows)
        rows = grid_rows(r, score_rows)
        pids = r.prompt_id[rows]
        uniq = np.unique(pids)
        for name in names:
            params = tuned.get(name, {})
            fold_acc = []
            for layer in layers:
                s = _fold_scores(r, tables, rows, name, layer, params)
                cov = coverage(s, r.slots[rows, layer].astype(np.int64), budgets=budgets)
                by_prompt = np.array([cov[pids == p].mean(axis=0) for p in uniq])
                for pid, row in zip(uniq, by_prompt):
                    per_prompt[name][layer][int(pid)] = row
                # Macro by Prompt here too, so the fold spread is the spread of
                # the statistic actually reported and not of a cell-weighted
                # cousin of it.
                fold_acc.append(by_prompt.mean(axis=0))
            per_fold[name][f] = np.mean(fold_acc, axis=0)
    return per_prompt, per_fold


def macro(per_prompt, name, layers, pids=None) -> np.ndarray:
    """[len(layers), K] Coverage, averaged over Prompts within each Layer.

    Prompts, not cells: the Prompt is this project's unit of independence, and
    Decode length varies ~3x across Categories, so a cell-weighted mean would
    quietly weight the long Prompts and let Category imbalance into a number
    that is not about Category.
    """
    rows = []
    for layer in layers:
        vals = per_prompt[name][layer]
        keep = [v for p, v in sorted(vals.items()) if pids is None or p in pids]
        rows.append(np.mean(keep, axis=0))
    return np.array(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="coarse grid, for a smoke run")
    args = ap.parse_args()

    t0 = time.perf_counter()
    r = load_routing(args.store, split="train")
    layers = list(target_layers(r.n_layers))
    print(f"train: {r.n_tokens} decode tokens, {len(np.unique(r.prompt_id))} prompts, "
          f"layers {layers[0]}..{layers[-1]}", file=sys.stderr)

    fold_pids = folds(args.store, n_folds=args.folds, seed=args.seed)
    fold_sets = []
    for held in fold_pids:
        score_rows = r.rows_for(held)
        fit_rows = np.setdiff1d(np.arange(r.n_tokens), score_rows)
        fold_sets.append((fit_rows, score_rows))

    alphas = QUICK_ALPHAS if args.quick else ALPHAS
    betas = QUICK_BETAS if args.quick else BETAS
    ws = QUICK_WS if args.quick else WS

    # --- tune, inside train only -------------------------------------------
    grid = [{"alpha": a, "beta": b} for a in alphas for b in betas]
    surface, tuned = {}, {}
    for name in ("prev_token", "cross_layer"):
        got = sweep(r, fold_sets, name, grid, layers)
        surface[name] = [
            {"alpha": None if not np.isfinite(p["alpha"]) else p["alpha"],
             "beta": p["beta"], "coverage_at_8": v}
            for p, v in got
        ]
        best = max(got, key=lambda kv: kv[1])[0]
        tuned[name] = dict(best)
        print(f"tuned {name}: {best} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)

    wgrid = [{**tuned["cross_layer"], "w": w} for w in ws]
    got = sweep(r, fold_sets, "combined", wgrid, layers)
    surface["combined"] = [{"w": p["w"], "coverage_at_8": v} for p, v in got]
    tuned["combined"] = dict(max(got, key=lambda kv: kv[1])[0])
    print(f"tuned combined: {tuned['combined']} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)

    # The table reports B and C at their canonical betas (0 and 1); the surface's
    # own argmax is kept separately, because "where did beta actually land" is
    # the question the nesting was built to answer and the override would hide it.
    tuned_free = {k: dict(v) for k, v in tuned.items()}
    tuned["cross_layer"] = {**tuned["cross_layer"], **CANONICAL["cross_layer"]}
    tuned["cross_layer_gated"] = {**tuned_free["cross_layer"], **CANONICAL["cross_layer_gated"]}

    # --- the result ---------------------------------------------------------
    per_prompt, per_fold = evaluate(r, fold_sets, layers, tuned, BUDGETS)

    cats = {row["prompt_id"]: row["category"] for row in read_prompts(args.store).to_pylist()}
    by_cat = defaultdict(set)
    for pid, cat in cats.items():
        by_cat[cat].add(pid)

    result = {
        "store": str(args.store),
        "n_folds": args.folds,
        "seed": args.seed,
        "n_train_prompts": int(len(np.unique(r.prompt_id))),
        "n_train_tokens": int(r.n_tokens),
        "top_k": r.top_k,
        "budgets": list(BUDGETS),
        "layers": layers,
        "tune_at": TUNE_AT,
        "chance": [k / r.n_experts for k in BUDGETS],
        "predictors": [{"name": n, "horizon": h, "note": d} for n, h, d in PREDICTORS],
        "tuned": {k: {kk: (None if kk == "alpha" and not np.isfinite(vv) else vv)
                      for kk, vv in v.items()} for k, v in tuned.items()},
        "tuned_free": tuned_free,
        "surface": surface,
        "per_layer": {},
        "pooled": {},
        "fold_spread": {},
        "by_category": {},
    }
    for name, _, _ in PREDICTORS:
        curves = macro(per_prompt, name, layers)
        result["per_layer"][name] = curves.tolist()
        result["pooled"][name] = curves.mean(axis=0).tolist()
        result["fold_spread"][name] = [per_fold[name][f].tolist() for f in range(args.folds)]
        result["by_category"][name] = {
            cat: macro(per_prompt, name, layers, pids).mean(axis=0).tolist()
            for cat, pids in sorted(by_cat.items())
        }

    # Popularity on the full grid, including Layer 0, as the footnote the common
    # grid owes the reader.
    full = list(range(r.n_layers))
    pp_full, _ = evaluate(r, fold_sets, full, {}, BUDGETS, names=["popularity"])
    result["popularity_full_grid"] = macro(pp_full, "popularity", full).tolist()
    result["popularity_layer_0"] = macro(pp_full, "popularity", [0]).tolist()[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.out} in {time.perf_counter() - t0:.0f}s", file=sys.stderr)

    for name, horizon, _ in PREDICTORS:
        cov = result["pooled"][name]
        cells = "  ".join(f"{c * 100:5.1f}" for c in cov)
        print(f"{name:<18} {horizon:<6} {cells}", file=sys.stderr)


if __name__ == "__main__":
    main()
