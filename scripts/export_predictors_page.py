#!/usr/bin/env python3
"""Build ``site/predictors.html`` from the template and the Coverage result.

The page inlines its data so it works over ``file://`` with no server, the same
arrangement as ``site/routing.html``. Its source is
``docs/results/predictor-coverage.json``, which is committed — so this script
does not need the GPU, the engine or ``data/``.

**Train only.** The payload is built from the cross-validated train result. If a
test number is ever produced by ``confirm_on_test.py``, it belongs in the
results document and must not be inlined here: the page is meant to stay safe to
look at while a modelling decision is still open (ADR-0003).

Usage::

    .venv/bin/python scripts/export_predictors_page.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULT = Path("docs/results/predictor-coverage.json")
TEMPLATE = Path("site/predictors.template.html")
OUT = Path("site/predictors.html")

# Display names, in the order the page's SERIES list expects. Keys are the
# Predictor names run_predictors.py writes; the page never sees the internal ones.
SERIES = {
    "popularity": "Layer popularity",
    "cross_layer": "Cross-layer Markov",
    "cross_layer_gated": "Gate-weighted Markov",
    "prev_token": "Previous-token Markov",
    "persistence": "Persistence (copy t-1)",
}
PLACEHOLDER = "__DATA__"


def payload(result: dict) -> dict:
    """The subset the page needs, rounded to 3dp to keep the file small."""
    import statistics

    out = {
        "budgets": result["budgets"],
        "layers": result["layers"],
        "n_experts": 256,
        "top_k": result["top_k"],
        "popularity_layer0": result["popularity_layer_0"],
        "cats": sorted(result["by_category"]["persistence"]),
        "tuned": result.get("tuned_free", result["tuned"]),
        "series": {},
        "pooled": {},
        "fold_sd": {},
        "by_category": {},
    }
    for key, label in SERIES.items():
        folds = result["fold_spread"][key]
        out["series"][label] = [[round(v * 100, 3) for v in row] for row in result["per_layer"][key]]
        out["pooled"][label] = [round(v * 100, 3) for v in result["pooled"][key]]
        out["fold_sd"][label] = [
            round(statistics.stdev([f[i] for f in folds]) * 100, 3)
            for i in range(len(result["budgets"]))
        ]
        out["by_category"][label] = {
            cat: [round(v * 100, 3) for v in vals]
            for cat, vals in sorted(result["by_category"][key].items())
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, default=RESULT)
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    result = json.loads(args.result.read_text())
    html = args.template.read_text()
    if PLACEHOLDER not in html:
        raise SystemExit(f"{args.template} has no {PLACEHOLDER} to substitute")

    html = html.replace(PLACEHOLDER, json.dumps(payload(result), separators=(",", ":")))
    args.out.write_text(html)
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes, data inlined)")


if __name__ == "__main__":
    main()
