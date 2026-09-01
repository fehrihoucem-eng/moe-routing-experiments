"""Turn colibri ROUTE_TRACE dumps into routing tensors.

    from routetrace import build_store, load_X

    build_store(["data/traces/smoke.trace"], "data/stores/smoke")
    X, index = load_X("data/stores/smoke")     # decode only, [tokens, 40, 256]
"""

from .parse import DECODE, PREFILL, Prompt, TraceFormatError, parse_trace
from .predict import (
    BUDGETS,
    LAYER_HORIZON,
    TOKEN_HORIZON,
    Routing,
    Tables,
    coverage,
    fit_tables,
    folds,
    grid_rows,
    load_routing,
    score,
)
from .splits import TEST, TRAIN, make_split, prompt_ids_for_split, read_split
from .store import build_store, read_meta, read_prompts, read_routing
from .tensor import COO, describe, load_X, prompt_ids_for, prompt_slices, to_torch

__all__ = [
    "BUDGETS",
    "COO",
    "DECODE",
    "LAYER_HORIZON",
    "PREFILL",
    "TEST",
    "TOKEN_HORIZON",
    "TRAIN",
    "Prompt",
    "Routing",
    "Tables",
    "TraceFormatError",
    "build_store",
    "coverage",
    "describe",
    "fit_tables",
    "folds",
    "grid_rows",
    "load_X",
    "load_routing",
    "make_split",
    "parse_trace",
    "prompt_ids_for",
    "prompt_ids_for_split",
    "prompt_slices",
    "read_meta",
    "read_prompts",
    "read_routing",
    "read_split",
    "score",
    "to_torch",
]
