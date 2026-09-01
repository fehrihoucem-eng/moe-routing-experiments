"""Turn colibri ROUTE_TRACE dumps into routing tensors.

    from routetrace import build_store, load_X

    build_store(["data/traces/smoke.trace"], "data/stores/smoke")
    X, index = load_X("data/stores/smoke")     # decode only, [tokens, 40, 256]
"""

from .parse import DECODE, PREFILL, Prompt, TraceFormatError, parse_trace
from .splits import TEST, TRAIN, make_split, prompt_ids_for_split, read_split
from .store import build_store, read_meta, read_prompts, read_routing
from .tensor import COO, describe, load_X, prompt_ids_for, prompt_slices, to_torch

__all__ = [
    "COO",
    "DECODE",
    "PREFILL",
    "TEST",
    "TRAIN",
    "Prompt",
    "TraceFormatError",
    "build_store",
    "describe",
    "load_X",
    "make_split",
    "parse_trace",
    "prompt_ids_for",
    "prompt_ids_for_split",
    "prompt_slices",
    "read_meta",
    "read_prompts",
    "read_routing",
    "read_split",
    "to_torch",
]
