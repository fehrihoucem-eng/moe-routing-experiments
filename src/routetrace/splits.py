"""Train/test assignment, stratified by category.

The split is **by prompt, never by token**. Tokens inside one prompt share a
prefix, a topic and a decode trajectory, so a token-level split would put
near-duplicate rows on both sides and quietly inflate every held-out score. A
prompt is the smallest independent unit the capture produces -- ``serve_one()``
resets KV per request, and a prompt was measured to route identically whether it
ran alone or as #85 of 100.

The assignment is written to ``<store>/split.json`` rather than recomputed on
demand. Two experiments that "both used seed 0" must be comparable even if one
of them re-ran ``build_store`` in between, and an assignment held only in code is
one refactor away from silently changing.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .store import read_prompts

TRAIN = "train"
TEST = "test"
SPLIT_FILE = "split.json"


def make_split(
    store_dir: str | Path,
    test_per_category: int = 4,
    seed: int = 0,
    overwrite: bool = False,
) -> dict:
    """Assign every prompt to train or test, ``test_per_category`` per category.

    Deterministic in ``seed``: categories are visited in sorted order and each
    draws from its own sorted prompt_id list, so the result does not depend on
    row order in the store. Writes ``<store>/split.json`` and returns it.
    """
    store_dir = Path(store_dir)
    out = store_dir / SPLIT_FILE
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True to replace it")

    rows = read_prompts(store_dir).to_pylist()
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"] or ""].append(int(r["prompt_id"]))

    rng = random.Random(seed)
    by_category: dict[str, dict[str, list[int]]] = {}
    train: list[int] = []
    test: list[int] = []
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        if test_per_category > len(ids):
            raise ValueError(
                f"category {cat!r} has {len(ids)} prompts, "
                f"cannot hold out {test_per_category}"
            )
        held = sorted(rng.sample(ids, test_per_category))
        kept = [i for i in ids if i not in set(held)]
        by_category[cat] = {TRAIN: kept, TEST: held}
        train.extend(kept)
        test.extend(held)

    split = {
        "seed": seed,
        "test_per_category": test_per_category,
        "n_train": len(train),
        "n_test": len(test),
        "by_category": by_category,
        TRAIN: sorted(train),
        TEST: sorted(test),
    }
    out.write_text(json.dumps(split, indent=2) + "\n")
    return split


def read_split(store_dir: str | Path) -> dict:
    path = Path(store_dir) / SPLIT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; create it with make_split(store_dir)"
        )
    return json.loads(path.read_text())


def prompt_ids_for_split(store_dir: str | Path, split: str) -> list[int]:
    """prompt_ids on one side of the train/test assignment."""
    if split not in (TRAIN, TEST):
        raise ValueError(f"split must be {TRAIN!r} or {TEST!r}, got {split!r}")
    return list(read_split(store_dir)[split])
