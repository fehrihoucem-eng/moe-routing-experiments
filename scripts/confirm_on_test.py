#!/usr/bin/env python3
"""Score one already-chosen Predictor on the held-out test Split. Once.

This is a separate script from ``run_predictors.py`` on purpose. That one
chooses; this one confirms, and the two must never be the same run, or the
holdout has been spent on the choice it was meant to check (ADR-0003).

It refuses to run without ``--confirm``, and it takes exactly one Predictor:
sweeping several here, or re-running after seeing the answer, is the same
mistake as scoring the comparison on test in the first place.

Tables are fitted on the **whole** train Split, not on a cross-validation fold,
because the choice has already been made and there is nothing left to hold out
from within train.

Usage::

    .venv/bin/python scripts/confirm_on_test.py --predictor combined --confirm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from routetrace.predict import (  # noqa: E402
    BUDGETS,
    coverage,
    fit_tables,
    grid_rows,
    load_routing,
    score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_predictors import ALIAS, target_layers  # noqa: E402

STORE = Path("data/stores/corpus_v1")
CHOICE = Path("docs/results/predictor-coverage.json")
OUT = Path("docs/results/test-confirmation.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--choice", type=Path, default=CHOICE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--confirm", action="store_true",
                    help="required: this spends the held-out Split")
    args = ap.parse_args()

    if not args.confirm:
        raise SystemExit(
            "refusing to read the test Split without --confirm. "
            "Choose the Predictor with run_predictors.py first; this run is the "
            "one-off confirmation and re-running it after seeing the answer "
            "makes test a validation set."
        )
    if args.out.exists():
        raise SystemExit(
            f"{args.out} exists: test has already been spent for this result. "
            "Delete it deliberately if the Corpus or the Predictor has changed."
        )

    chosen = json.loads(args.choice.read_text())
    params = chosen["tuned"].get(args.predictor, {})
    if params.get("alpha", 0) is None:
        params = {**params, "alpha": np.inf}

    train = load_routing(args.store, split="train")
    test = load_routing(args.store, split="test")
    tables = fit_tables(train, np.arange(train.n_tokens))

    rows = grid_rows(test, np.arange(test.n_tokens))
    pids = test.prompt_id[rows]
    layers = list(target_layers(test.n_layers))

    per_layer = []
    for layer in layers:
        s = score(ALIAS.get(args.predictor, args.predictor), test, tables, rows, layer, **params)
        cov = coverage(s, test.slots[rows, layer].astype(np.int64), budgets=BUDGETS)
        per_layer.append([cov[pids == p].mean(axis=0) for p in np.unique(pids)])

    per_layer = np.array(per_layer)          # [layers, prompts, budgets]
    by_prompt = per_layer.mean(axis=0)       # macro over Layers, per Prompt
    pooled = by_prompt.mean(axis=0)

    result = {
        "predictor": args.predictor,
        "params": chosen["tuned"].get(args.predictor, {}),
        "fitted_on": "train (all 80 prompts)",
        "n_test_prompts": int(np.unique(pids).size),
        "n_test_tokens": int(test.n_tokens),
        "budgets": list(BUDGETS),
        "layers": layers,
        "pooled": pooled.tolist(),
        "per_prompt_std": by_prompt.std(axis=0, ddof=1).tolist(),
        "per_layer": per_layer.mean(axis=1).tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")

    train_pooled = chosen["pooled"][args.predictor]
    print(f"{args.predictor}  budgets {list(BUDGETS)}", file=sys.stderr)
    print("  train CV  " + "  ".join(f"{c * 100:5.1f}" for c in train_pooled), file=sys.stderr)
    print("  test      " + "  ".join(f"{c * 100:5.1f}" for c in pooled), file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
