# How predictable is expert routing?

Coverage of seven Predictors on `corpus_v1`, by 5-fold cross-validation inside
the train Split. Numbers are `%` of the Router's 8 selected Experts that the
Predictor named, macro-averaged by Prompt, on Layers 1-39 of Decode. Protocol
and its reasons: [ADR-0003](../adr/0003-evaluate-predictors-on-a-common-grid.md).
Raw numbers: [`predictor-coverage.json`](./predictor-coverage.json), rebuilt by
`scripts/run_predictors.py` (~6 min, CPU only). Interactive version, with the
Coverage-by-Layer curve at every Budget: `site/predictors.html` (self-contained,
opens over `file://`; rebuilt by `scripts/export_predictors_page.py`).

**The test Split has been spent, once, on the combined Predictor** — see
[Confirmed on test](#confirmed-on-test). Every other number below is train, and
every *choice* below was made on train alone.

## The headline

Knowing the previous Token is worth almost as much as knowing the Layer below,
and it is known 40x earlier.

The cheapest Predictor that works — copy the previous Token's own 8 Slots, no
table, no training, no lookup — reaches **38.9%** Coverage at K=8. The best
Predictor of any kind reaches **40.6%**. That **+1.60pp** is what an entire
extra Layer of hindsight buys, and it has to be spent inside 1.25 ms instead of
50 ms.

The caveat that matters: this holds *at K=8*. Give a prefetcher a Budget of 16
and the gap widens to +13.02pp, because copying can only ever name 8 Slots.

## Confirmed on test

The combination — the Predictor `run_predictors.py` chose, at its tuned
`alpha = 0, beta = 0.75, w = 0.4` — scored once on the 20 held-out Prompts
(3,569 Decode Tokens), with tables refitted on all 80 train Prompts. Raw
numbers: [`test-confirmation.json`](./test-confirmation.json).

| | K=1 | K=3 | K=5 | K=8 | K=12 | K=16 |
|---|---|---|---|---|---|---|
| train, 5-fold CV | 8.12 | 20.82 | 30.30 | **40.57** | 49.97 | 56.70 |
| test, held out | 8.13 | 21.26 | 31.07 | **41.63** | 51.36 | 58.20 |
| difference | +0.01 | +0.45 | +0.77 | +1.06 | +1.39 | +1.50 |

**It did not fall, and the direction it moved is the expected one.** The
confirmation fits on all 80 train Prompts where each CV fold fits on 64, so its
tables carry ~25% more counts; the gap widening monotonically with Budget
(+0.01pp at K=1 to +1.50pp at K=16) is the signature of a denser table, which
can only help where the fold-fitted one had thin conditioning rows — out in the
tail, past the Slots that were already obvious. Read the train CV number as the
honest estimate and this as the same estimate with more fitting data, not as a
1pp improvement.

It is also well inside the noise either way. Coverage@8 has a standard deviation
of 5.90pp *across the 20 test Prompts*, so the standard error on their mean is
1.32pp and the +1.06pp difference is 0.8 of one. The train folds themselves
spanned 39.83–41.22.

The depth profile reproduces on held-out Prompts, which is the part worth more
than the pooled number. Coverage@8 by third: early (1–13) **40.4** against
train's 39.2, mid (14–26) **38.6** against 38.0, late (27–39) **45.8** against
44.5 — the same dip in the middle of the stack and the same late-stack peak,
with Layer 39 at 47.2%. The structure in [Depth](#depth) is not an artefact of
the Prompts it was found on.

**What this does not confirm.** One Predictor was scored, so the comparisons
are all still train-only — including the headline one. That free persistence is
within 1.60pp of this number at K=8 remains a cross-validated claim, not a
held-out one. Test is now spent for `corpus_v1`; anything further, an HMM
included, needs a second Corpus rather than a second run of this script.

## The K-curve

`±` is the standard deviation across the 5 folds.

| Predictor | Horizon | K=1 | K=3 | K=5 | K=8 | K=12 | K=16 |
|---|---|---|---|---|---|---|---|
| chance | — | 0.4 | 1.2 | 2.0 | 3.1 | 4.7 | 6.3 |
| **A** popularity | token | 1.6 ±0.2 | 4.0 ±0.4 | 6.2 ±0.6 | 9.2 ±0.7 | 12.8 ±0.8 | 16.1 ±1.0 |
| **D0** persistence | token | 7.1 ±0.1 | 19.0 ±0.3 | 28.6 ±0.3 | **38.9** ±0.4 | 41.5 ±0.4 | 43.7 ±0.4 |
| **D** prev-token | token | 6.8 ±0.1 | 17.0 ±0.2 | 24.6 ±0.3 | 33.0 ±0.4 | 41.3 ±0.5 | 47.5 ±0.5 |
| **B** cross-layer | layer | 7.8 ±0.2 | 19.7 ±0.3 | 28.7 ±0.4 | 38.5 ±0.5 | 47.7 ±0.6 | 54.3 ±0.6 |
| **C** cross-layer, gated | layer | 8.0 ±0.1 | 20.3 ±0.3 | 29.3 ±0.4 | 39.1 ±0.5 | 48.3 ±0.6 | 55.0 ±0.6 |
| B-raw (colibri's) | layer | 7.1 ±0.2 | 17.8 ±0.4 | 25.7 ±0.5 | 34.3 ±0.6 | 42.5 ±0.6 | 48.6 ±0.6 |
| **B+D** combined | layer | 8.1 ±0.1 | 20.8 ±0.3 | 30.3 ±0.4 | **40.6** ±0.5 | 50.0 ±0.6 | 56.7 ±0.6 |

Coverage below K=8 is capped at `K/8` by arithmetic alone — 62.5% at K=5 — so
only the K>=8 columns are comparable against 100%. Misses, the count you would
stall on, is `8 * (1 - Coverage)`: 4.7 for the combined Predictor at K=8, 7.3 for
popularity, 7.8 for chance.

### Which gaps are real

The `±` above is the unpaired spread across folds, and it overstates the
uncertainty on a *comparison*: every Predictor was scored on the same five
folds, so the differences are paired. Per-fold differences at K=8, with the
range across the five folds:

| Comparison | mean | sd | folds |
|---|---|---|---|
| B+D over D0 persistence | +1.60 | 0.61 | +0.83 .. +2.33 |
| B+D over D0, **at K=16** | +13.02 | 0.34 | +12.63 .. +13.37 |
| B normalised over B-raw | +4.18 | 0.09 | +4.11 .. +4.32 |
| D0 persistence over D prev-token | +5.92 | 0.42 | +5.54 .. +6.50 |
| C gated over B unweighted | +0.63 | 0.10 | +0.53 .. +0.78 |
| B cross-layer over D0 persistence | −0.45 | 0.65 | −1.25 .. **+0.38** |

Every one of these is consistent in sign across all five folds except the last,
which straddles zero: **B and D0 are tied at K=8**, and the entire advantage of
the layer Horizon at that Budget comes from combining it with the token Horizon,
not from being a Layer closer.

## Six things the numbers say

**1. Popularity is nearly useless, and that is itself the finding.** 9.2% at
K=8 against a 3.1% chance line: knowing nothing but which Experts are hot buys a
factor of 2.9. Routing in this model is close to balanced, which is what an
auxiliary load-balancing loss is for (`router_aux_loss_coef = 0.001` in the
container's config). Any prefetch scheme resting on a static hot set is starting
from almost nothing — and colibri's `HEAT_FILE` placement table is exactly such
a set.

**2. Copying the previous Token beats learning how it transitions**, at every
Budget up to 8. D0 reaches 38.9% where the fitted `P(E_t | E_{t-1})` reaches
33.0% — **+5.92pp paired**, consistent across all five folds. The learned table
spreads mass over Slots that *often* follow, and in doing so loses the identity
structure that dominates at small K.

The two cross over between K=8 and K=12, and the reason is structural: D0 has
only 8 Slots to name. Past K=8 it is padding with popularity, which is worth 9%,
so its curve flattens — 38.9 -> 41.5 -> 43.7 — while D keeps climbing to 47.5.
**Below K=8 copy; above it, learn.**

**3. Gate weighting is real, consistent, and tiny.** C beats B by 0.63pp at
K=8, positive in all five folds with sd 0.10 — so it is not noise, though the
unpaired ±0.5 columns make it look like it might be. The free exponent lands at
**beta = 0.75**, between the two canonical settings, and moving beta across its
whole range moves Coverage by 0.66pp. The top gate carries only 3.2x the bottom
one's weight, so there was never much to reweight. Worth taking, since it costs
one exponentiation; not worth arguing about.

**4. Shrinkage buys nothing; the counts are already dense enough.** Coverage@8
is flat to two decimals from alpha = 0 to alpha = 30, decays past 100, and
collapses to 9.22% at alpha = inf — which is popularity's 9.2%, confirming
empirically that A is B's infinite-shrinkage limit rather than a separate
Predictor. The thin tail we designed the backoff for (5.7% of conditioning rows
firing under 30 times) simply does not carry enough scored cells to matter.

**5. Normalisation is worth more than the entire gate question.** colibri's
shipped variant — raw co-occurrence counts, truncated at top-16, no
per-conditioner normalisation — scores 34.3% where the same table normalised to
`P(j|i)` scores 38.5%. **+4.18pp for a division** (sd 0.09, the most consistent
gap in the whole study), against +0.63pp for gate weighting. Without
normalisation a busy conditioning Expert dominates the sum regardless of how
informative it is. This is the one directly actionable result here for
`c/tools/route_pairs.py` and `COUPLE=`.

**6. The combination is the best Predictor, and at K=8 it is barely worth
having.** B+D reaches 40.6% against D0's free 38.9% — **+1.60pp paired**, in a
1.25 ms deadline instead of 50 ms, for a 2.6M-cell table. The mixture weight
lands at **w = 0.4**, leaning toward the cross-layer term, and the `w` sweep
spans 33.0% (pure D) to 40.6%.

At K=16 the verdict inverts: **+13.02pp**, because D0 has run out of Slots to
name and the layer Horizon has not. If a prefetcher can afford a Budget of 16 —
25.6 MiB per Layer at 1.6 MiB an Expert — the cross-layer table earns its keep.
If it can only afford 8, copying the previous Token is within 1.6pp of the best
anything here can do.

## Depth

Coverage@8 by Layer, sampled:

| Layer | A | D0 | D | C | B+D |
|---|---|---|---|---|---|
| 1 | 7.2 | 16.2 | 26.4 | 36.7 | 40.3 |
| 5 | 7.6 | 33.4 | 30.6 | 37.4 | 40.3 |
| 10 | 8.7 | 34.0 | 31.8 | 37.0 | 39.5 |
| 15 | 11.4 | 42.6 | 32.3 | 38.4 | 38.3 |
| 20 | 10.2 | 48.8 | 34.2 | 35.9 | 37.1 |
| 25 | 8.5 | 42.5 | 33.4 | 35.7 | 37.9 |
| 30 | 7.5 | 47.6 | 39.7 | 42.5 | 44.6 |
| 35 | 8.6 | 45.9 | 37.1 | 39.5 | 43.1 |
| 39 | 10.1 | 26.1 | 31.2 | 47.4 | 46.4 |

Averaged in thirds — early (1-13), mid (14-26), late (27-39):

| Predictor | early | mid | late |
|---|---|---|---|
| A popularity | 8.6 | 10.0 | 9.1 |
| D0 persistence | 32.7 | **42.9** | 41.3 |
| C cross-layer | 37.1 | 37.1 | **43.1** |
| B+D combined | 39.2 | 38.0 | **44.5** |

**Persistence is a mid-stack phenomenon**, and this reproduces a published shape
independently. Mixtral reports consecutive-Token repetition at chance in Layer 0
(~14% against a 12.5% floor) and markedly higher at Layers 15 and 31
[Jiang et al. 2024, Table 5]. Our D0 runs 16.2% at Layer 1, rises to 48.8% by
Layer 20, and falls off a cliff at Layer 39 (26.1%). Different metric, same
curve — which is worth something, because it is the one place our result can be
checked against somebody else's model.

The final Layer inverts the whole ranking: D0 collapses to 26.1% while
cross-layer still holds 47.4% there (its own peak is 49.7% at Layer 34). The last
Layer's routing appears strongly determined by Layer 38 and weakly by what the
previous Token did — consistent with the literature's observation that
final-Layer hidden states correlate with the output embedding rather than with
local context.

### A global prefetch policy is the wrong design

D0 beats the cross-layer Predictor outright in **24 of 39 Layers**, but not in a
contiguous block: the two Horizons trade places with depth. Coverage@8, C vs D0:

| Layers | Winner | D0 persistence | C cross-layer |
|---|---|---|---|
| 1–5 | C | 24.5 | **36.6** |
| 6–31 | **D0** | **42.1** | 37.9 |
| 32–39 | C | 37.7 | **44.9** |

An oracle that picks the better Horizon at each Layer scores **42.7%**, against
**39.1%** for the best single global Predictor. **That +3.6pp is larger than the
+1.60pp the combined Predictor buys over free persistence** — so specialising the
policy by depth is worth more than a better Predictor, and worth more than the
whole `layer`-Horizon apparatus.

The oracle is an upper bound, not a policy: it is a per-Layer choice between two
fixed Predictors, made with the answer in hand. But the choice is between only
two options over 39 Layers, so it is a far smaller thing to fit than either
transition table, and it is the obvious next thing to measure honestly.

This also inverts the shape one might expect. Early Layers are where routing
persists *least* between Tokens — matching Mixtral's Layer-0-at-chance — and the
late stack is where the previous Token stops mattering and the Layer below takes
over. Neither end is "history-dependent" in the same sense.

## Category

Coverage@8, the shared Predictors evaluated per Category (no Predictor was
fitted per Category — see below):

| Predictor | coding | conversational | factual | math | structured |
|---|---|---|---|---|---|
| A popularity | 6.3 | 11.9 | 11.5 | 11.8 | 4.7 |
| D0 persistence | 39.4 | 46.4 | 43.5 | 34.6 | 30.8 |
| C cross-layer | 36.5 | 47.3 | 39.5 | 39.3 | 33.1 |
| B+D combined | 38.6 | 48.6 | 41.8 | 40.1 | 33.7 |

The spread is large — 15pp between `conversational_creative` and
`structured_extraction` — and it moves the two Horizons differently:
persistence leads on `coding` and `factual_expository`, the cross-layer
Predictor leads on `math_reasoning` and `structured_extraction`. A prefetcher
tuned on one Category would be mistuned on another.

`structured_extraction` is hardest on every Predictor *and* has the most
balanced routing (popularity 4.7%). Two candidate explanations we cannot
separate here: its answers are genuinely short (69 Decode Tokens per Prompt on
average, against a 200 cap that 67 of 100 responses hit), so its Prompts
contribute fewer, more heterogeneous Tokens; or extraction output really does
move through Expert space faster. Distinguishing them needs a
position-controlled rerun.

## What this does not establish

**We did not reproduce colibri's +3.6..+9.4pp.** `route_pairs.py` claims that
band for coupling over "marginal heat"; our B-raw beats popularity by **25pp**
at K=8. Almost certainly not the same measurement — their baseline is a
*placement* heat table sized by `CUDA_EXPERT_GB`, i.e. a resident set of some
size, not a top-8 prediction, and the figure is in-sample on a different model.
B-raw was built to reproduce that band as a check on this harness, and it did
not: nothing here validates our numbers against an external one. The
normalisation finding (point 5) stands on its own, since it is an internal
comparison between two scoring rules on identical counts.

**Category-conditioned tables were not fitted.** A fit set splits five ways to
~1,920 Tokens per Category, which drops the median (Layer, Slot) firing count
from 246 to ~49 — the shrinkage would do most of the work and the tables would
largely reproduce the pooled one. The stratified evaluation above is the
answerable version. Fitting per Category needs a Corpus roughly 5x larger.

**Prefill is untouched.** All Prefill Tokens of a Prompt share one Forward, so
Token t-1's routing is never available before Token t's and the `token` Horizon
is undefined there. For context, a whole Prefill Forward touches a mean of 85
distinct Experts per (Prompt, Layer) — median 81, range 47-220 — from ~36.6
Tokens, so it is not the degenerate all-256 case one might assume.

**The shared Expert is absent.** Qwen3.6 routes 8 of 256 alongside 1 shared
Expert that `rt_route()` never sees, so every number here describes the routed
residual.

**Prefetch is simulated, not observed.** All 10,240 Experts fit in this box's
VRAM, so no Coverage number here has been converted into a latency. The lead-time
arithmetic (1.6 MiB per Expert, 1.25 ms per Layer, 50 ms per Token at 20 tok/s)
is arithmetic, not measurement.

**80 Prompts.** Unpaired fold spreads run ±0.1 to ±1.0pp. Paired comparisons
resolve much finer — the +0.63pp C-over-B gap is consistent across all five
folds — but a paired test on five folds over 80 Prompts is not a substitute for
a second Corpus, and none of these Predictors has been seen on text this model
was not asked to generate greedily at `temp=0`.

## If this is continued

The ordered queue is [`docs/queue.md`](../queue.md), which interleaves these
with what the Lag experiment added. This list is kept as the record of what
*this* study left open, and its numbering is what the queue's pointers cite.

1. ~~**Confirm on test, once.**~~ Done: 41.6% at K=8, above the 40.6% train CV
   for the structural reason given in [Confirmed on test](#confirmed-on-test).
   The holdout for `corpus_v1` is now spent.
2. **Tell colibri about normalisation.** +4.2pp on their own table for a
   division, and it is a two-line change to `route_pairs.py`'s consumer.
3. **Position-controlled rerun.** Drop the first n Decode Tokens (n = 1, 5, 20)
   and recompute. If `structured_extraction` catches up, the Category spread was
   about Prompt length.
4. ~~**Longer horizons.**~~ Done, and widened, in
   [temporal-history.md](./temporal-history.md): t-2/t-4/t-8 and their unions.
   Older Tokens add nothing at K=8 but **+8.1pp at K=16**, saturating at t-4,
   and routing memory is longest mid-stack. That work also found `cross-layer +
   t-1` beating the combination below by **+6.11pp** — see its point 2.
5. **Union over Budgets, not Coverage.** For a real cache the question is the
   *residency* one: which Experts to keep, given a fixed VRAM budget across all
   40 Layers at once — a different objective from per-cell top-K.
6. **Fit the per-Layer Horizon choice properly.** The 42.7% oracle above is an
   upper bound taken with the answer in hand. Choosing per Layer on the fit
   folds and scoring on the held-out one would say what the +3.6pp is really
   worth, and it is the cheapest remaining gain on the table.
