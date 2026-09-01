#!/usr/bin/env python3
"""Build ``site/history.html`` from the template and the Lag result.

Same arrangement as ``export_predictors_page.py``: the page inlines its data so
it works over ``file://`` with no server, and its source is the committed
``docs/results/history-coverage.json``, so this script needs neither the GPU,
the engine, nor ``data/``.

**Train only**, and for the same reason as the other page -- it is meant to stay
safe to look at while a modelling decision is open (ADR-0003). The one test
number this project has spent lives in the results document, not here.

Usage::

    .venv/bin/python scripts/export_history_page.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RESULT = Path("docs/results/history-coverage.json")
TEMPLATE = Path("site/history.template.html")
OUT = Path("site/history.html")
PLACEHOLDER = "__DATA__"

#: Everything the page draws. Anything not listed is dropped, so the inlined
#: payload stays small and the page cannot quietly start depending on a field
#: the script does not promise.
KEEP = [
    "popularity",
    "t-1", "t-2", "t-4", "t-8",
    "t-1 u t-2 (priority)",
    "t-1 u t-4 (priority)",
    "t-1 u t-2 u t-4 (priority)",
    "t-1 u t-2 u t-4 u t-8 (priority)",
    "t-1 u t-2 (vote)",
    "t-1 u t-4 (vote)",
    "t-1 u t-2 u t-4 (vote)",
    "cross-layer (C)",
    "combined (B+D, published)",
    "cross-layer + t-1",
    "cross-layer + history",
]


def pct(x) -> float:
    return round(x * 100, 3)


def payload(r: dict) -> dict:
    missing = [k for k in KEEP if k not in r["pooled"]]
    if missing:
        raise SystemExit(f"{RESULT} has no rows for {missing}; re-run scripts/run_history.py")

    n_k = len(r["budgets"])
    out = {
        "budgets": r["budgets"],
        "report_at": r["report_at"],
        "layers": r["layers"],
        "n_experts": 256,
        "top_k": 8,
        "grid": {
            "rows": r["grid_rows"],
            "total": r["grid_rows_total"],
            "prompts": r["n_prompts"],
            "max_lag": r["max_lag"],
        },
        "tuned": {"decay": r["tuned_decay"], "w": r["tuned_w"]},
        "cross_params": r["cross_params"],
        "decay_surface": r["decay_surface"],
        "w_surface": r["w_surface"],
        "paired": r["paired"],
        "pooled": {},
        "per_layer": {},
        "fold_sd": {},
        "thirds": {},
    }
    for k in KEEP:
        folds = r["fold_spread"][k]
        out["pooled"][k] = [pct(v) for v in r["pooled"][k]]
        out["per_layer"][k] = [[pct(v) for v in row] for row in r["per_layer"][k]]
        out["fold_sd"][k] = [
            round(statistics.stdev([f[i] for f in folds]) * 100, 3) for i in range(n_k)
        ]
        out["thirds"][k] = {t: [pct(v) for v in vals] for t, vals in r["thirds"][k].items()}
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
