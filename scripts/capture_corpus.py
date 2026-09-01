#!/usr/bin/env python3
"""Capture the corpus_v1 routing traces and build the store.

    .venv/bin/python scripts/capture_corpus.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from routetrace import build_store, describe  # noqa: E402
from routetrace.capture import PromptSpec, capture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/home/houcem-fehri/Models/qwen36_i4_gs64")
ENGINE = Path("/home/houcem-fehri/colibri/c/qwen36")
TRACE = ROOT / "data/traces/corpus_v1.trace"
STORE = ROOT / "data/stores/corpus_v1"

# Greedy: serve_sample() takes temp<=0 as exact argmax, so the capture is
# reproducible. Thinking is off, so the template pre-closes the <think> block
# and the model answers directly.
MAX_TOK = 200
TEMP = 0.0

ENV = {
    "COLI_CUDA": "1",
    "COLI_GPUS": "0",
    "CUDA_EXPERT_GB": "auto",
    "HEAT_FILE": "/home/houcem-fehri/colibri-run/heat.bin",
    "OMP_NUM_THREADS": "16",
    "OMP_WAIT_POLICY": "ACTIVE",
    "OMP_PROC_BIND": "close",
}


def main() -> int:
    corpus = json.loads((ROOT / "prompts/corpus_v1.json").read_text())
    specs = [PromptSpec(text=p["text"], category=p["category"]) for p in corpus["prompts"]]
    print(f"corpus {corpus['name']}: {len(specs)} prompts, max_tok={MAX_TOK}, temp={TEMP}")

    t0 = time.time()
    replies = capture(
        prompts=specs,
        trace_path=TRACE,
        model_dir=MODEL,
        engine=ENGINE,
        max_tok=MAX_TOK,
        temp=TEMP,
        enable_thinking=False,
        env_extra=ENV,
        progress=True,
    )
    dt = time.time() - t0

    failed = [r for r in replies if r.error]
    empty = [r for r in replies if not r.error and not r.response.strip()]
    print(f"\ncaptured {len(replies)} prompts in {dt / 60:.1f} min")
    if failed:
        print(f"  {len(failed)} FAILED:")
        for r in failed[:10]:
            print(f"    {r.key} {r.category}: {r.error}")
    if empty:
        print(f"  {len(empty)} produced no text: {[r.key for r in empty]}")

    build_store([TRACE], STORE)
    print()
    print("\n".join(describe(STORE).splitlines()[:6]))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
