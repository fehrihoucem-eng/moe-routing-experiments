# What to do next

The single ordered queue. Each item is a pointer: the reasoning lives in the
results document that produced it, and is not repeated here, so there is one
place to change when a number changes.

**Two constraints apply to everything below.**

`corpus_v1`'s **test Split is spent** — it went on the combined Predictor
(41.6% at K=8, `docs/results/test-confirmation.json`). Any new Predictor needs a
`corpus_v2` capture for a clean holdout, ~13 min on the GPU via
`scripts/capture_corpus.py`. Deleting that file to re-run makes test a
validation set.

**The order of 1 and 2 matters.** Item 1 settles *what the best Predictor is*;
item 2 is a policy layered on top of whatever that turns out to be. Fitting the
per-Layer choice against a baseline that item 1 then replaces means running it
twice.

## Next

**1. Re-examine `cross-layer + t-1` deliberately.**
It beat the published winner by **+6.11pp** at K=8 (46.35 vs 40.26 on the same
grid) with a *simpler* Predictor carrying no second table, and it got one tuned
weight in an experiment that was not about it. Give it its own sweep. If it
holds, it — not B+D — is what a `corpus_v2` should confirm.
→ [temporal-history.md](./results/temporal-history.md#an-unplanned-result-the-published-combination-was-mixing-the-wrong-thing)

**2. Fit the per-Layer Horizon choice properly.**
The 42.7%-vs-39.1% oracle is an upper bound taken with the answer in hand;
choosing per Layer on the fit folds and scoring on the held-out one says what
the +3.6pp is really worth. Still the largest untaken gain. The Lag work adds a
second thing to choose per Layer: the union is worth +11.5pp at Layer 20 and
+0.8pp at Layer 1.
→ [predictor-coverage.md](./results/predictor-coverage.md#a-global-prefetch-policy-is-the-wrong-design),
[temporal-history.md](./results/temporal-history.md#depth)

## After those

**3. Union over a residency Budget, not per-cell Coverage.**
For a real cache the question is which Experts to keep given a fixed VRAM budget
across all 40 Layers at once — a different objective. The finding that older
Tokens are good at filling *surplus* Budget is exactly this shape.
→ [predictor-coverage.md](./results/predictor-coverage.md#if-this-is-continued) item 5

**4. Position-controlled rerun.**
Drop the first n Decode Tokens (n = 1, 5, 20) and recompute. If
`structured_extraction` catches up, the Category spread was about Prompt length.
The Lag experiment's `max_lag=8` grid is already a mild version of this and
moved every number ~0.3pp.
→ [predictor-coverage.md](./results/predictor-coverage.md#if-this-is-continued) item 3

**5. Prefill.**
Untouched. The `token` Horizon is undefined there — all Prefill Tokens of a
Prompt share one Forward — so it needs its own Predictor set, not a rerun.

## Outside this repo

**Tell colibri about normalisation.** +4.2pp on their own table for a division,
and a two-line change to `route_pairs.py`'s consumer. The most consistent gap in
the whole study (sd 0.09) and the one directly actionable result for the engine.
→ [predictor-coverage.md](./results/predictor-coverage.md#six-things-the-numbers-say) point 5

## Done

- ~~Confirm on test, once.~~ 41.6% at K=8, above the 40.6% train CV for a
  structural reason. → [predictor-coverage.md](./results/predictor-coverage.md#confirmed-on-test)
- ~~Longer Horizons.~~ Widened to t-2/t-4/t-8 and their unions: nothing at K=8,
  **+8.1pp at K=16**, saturating at t-4, memory longest mid-stack.
  → [temporal-history.md](./results/temporal-history.md)
