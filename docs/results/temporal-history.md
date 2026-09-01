# Does routing history reach past the previous Token?

Coverage of 16 Predictors on `corpus_v1`, by 5-fold cross-validation inside the
train Split. Numbers are `%` of the Router's 8 selected Experts that the
Predictor named, macro-averaged by Prompt, on Layers 1-39 of Decode. Protocol:
[ADR-0003](../adr/0003-evaluate-predictors-on-a-common-grid.md). This is
follow-up 4 of [predictor-coverage.md](./predictor-coverage.md), which measured
only t-1. Raw numbers: [`history-coverage.json`](./history-coverage.json),
rebuilt by `scripts/run_history.py` (~2 min, CPU only). Interactive version, with
the Coverage-by-Layer curve at every Budget and the per-Layer union gain:
`site/history.html` (self-contained, opens over `file://`; rebuilt by
`scripts/export_history_page.py`).

**Train only.** The test Split was spent on the combined Predictor and is not
read here.

**The grid moved, so the baselines moved with it.** Every Predictor below scores
only Decode Tokens that have **8 predecessors**, because t-8 cannot enter a
Prompt's opening Tokens and letting t-1 score cells its rivals cannot reach
would decide the comparison on arithmetic rather than on signal. That keeps
11,360 of 12,000 train rows (94.7%) and all 80 Prompts. It costs about 0.3pp
uniformly: the published combined Predictor scores **40.26** here against
**40.57** on the shallower grid, and t-1 scores **38.55** against **38.9**. Do
not read a row of this document against a row of that one.

## The answer

**No at K=8, emphatically yes at K=12 and K=16.** Older Tokens cannot improve
the ranking of the top 8 — nothing here moves that by more than a tenth of a
point in either direction — but they are much better than popularity at filling
a Budget that t-1 has run out of Slots to fill.

The published study noted that persistence flattens above K=8 (38.9 -> 41.5 ->
43.7) because it has only 8 Slots to name and pads the rest with popularity.
Older Tokens are the fix: **+5.36pp at K=12 and +8.08pp at K=16**, paired,
consistent in sign across all five folds.

## Single Lags

Each row copies the Slots of exactly one earlier Token, ranked by the gates it
gave them, with popularity filling the Budget below.

| Predictor | K=8 | K=12 | K=16 |
|---|---|---|---|
| chance | 3.13 | 4.69 | 6.25 |
| popularity | 9.06 | 12.58 | 15.86 |
| **t-1** | **38.55** | 41.05 | 43.19 |
| t-2 | 31.11 | 34.00 | 36.49 |
| t-4 | 26.51 | 29.65 | 32.31 |
| t-8 | 24.65 | 27.94 | 30.72 |

**A Token eight back is still worth 2.7x popularity.** t-8 alone reaches 24.65%
against popularity's 9.06% and chance's 3.13%. Routing does not decorrelate
within the window a prefetcher cares about; it decays to a high floor. The fall
from t-1 to t-2 (-7.44pp) is steeper than the whole of t-2 to t-8 (-6.54pp,
paired, sd 0.87), so most of what recency is worth is spent in a single step and
the rest of the curve is nearly flat.

That flat tail is the trap in this experiment. A Predictor can look strong on
its own and still be redundant, and t-4 and t-8 are almost entirely redundant.

## Unions

Two ranking rules, because "does older history help?" is two questions and one
rule cannot separate them.

**priority** is the literal set union, newest Token first: t-1's 8 Slots, then
whatever t-2 adds, then t-4. Recency is a strict tier, so at K=8 it *is* t-1 by
construction and can only answer whether older history fills a surplus Budget
better than popularity does.

**vote** scores a Slot by `decay ** (lag - 1)` times the gate it got, summed
over Lags. A Slot the Router keeps returning to accumulates, so older Tokens can
reorder t-1's own 8 — this is the only rule that can answer whether recurrence
across history improves the ranking, and the only one that can lose.

| Predictor | K=8 | K=12 | K=16 |
|---|---|---|---|
| t-1 | 38.55 | 41.05 | 43.19 |
| t-1 ∪ t-2 (priority) | 38.55 | 46.02 | 49.15 |
| t-1 ∪ t-4 (priority) | 38.55 | 45.38 | 48.59 |
| t-1 ∪ t-2 ∪ t-4 (priority) | 38.55 | **46.42** | 51.28 |
| t-1 ∪ t-2 ∪ t-4 ∪ t-8 (priority) | 38.55 | 46.43 | **51.67** |
| t-1 ∪ t-2 (vote) | 38.37 | 46.01 | 49.15 |
| t-1 ∪ t-4 (vote) | 38.55 | 45.38 | 48.59 |
| t-1 ∪ t-2 ∪ t-4 (vote) | **38.60** | 46.56 | 51.28 |

### Which gaps are real

Paired per-fold differences — every Predictor saw the same five folds, so the
comparison resolves far finer than any unpaired spread would suggest.

| Comparison | K=8 | K=12 | K=16 |
|---|---|---|---|
| ∪(1,2,4) priority over t-1 | +0.00 | **+5.36** ±0.19 | **+8.08** ±0.20 |
| ∪(1,2) priority over t-1 | +0.00 | +4.96 ±0.17 | +5.95 ±0.16 |
| ∪(1,2,4) vote over t-1 | +0.05 ±0.07 | +5.50 ±0.20 | +8.08 ±0.20 |
| ∪(1,2) vote over t-1 | **−0.18** ±0.05 | +4.96 ±0.17 | +5.95 ±0.16 |
| adding t-8 to ∪(1,2,4) | +0.00 | +0.02 ±0.00 | +0.39 ±0.05 |

**Voting cannot beat copying at K=8.** The three-Lag vote gains +0.05pp and its
fold range straddles zero (−0.04 .. +0.13); the two-Lag vote *loses* 0.18pp, in
all five folds. Whatever a Slot's recurrence across t-1, t-2 and t-4 says about
it, the Router's own gate at t-1 already said it better. This is the same result
the published study found comparing D0 to D — copying beats learning how routing
transitions — reappearing in a form that has no fitted table in it at all, which
makes it a fact about the routing rather than about an estimator.

**The union saturates at t-4.** Adding t-8 to the union buys +0.02pp at K=12 and
+0.39pp at K=16, against the +8.08pp the first two additions bought. By t-8 the
Slots on offer have almost all been named already: the standalone curve is flat
out there because it is measuring the *same* Experts, not different ones.

## Depth

The question was whether this depends on Layer depth. It does, and in one
direction throughout.

Coverage@8 by Layer, sampled, with the ratio t-8/t-1 as a crude memory length:

| Layer | t-1 | t-2 | t-4 | t-8 | t-8/t-1 | ∪ gain @16 |
|---|---|---|---|---|---|---|
| 1 | 16.1 | 9.8 | 8.4 | 8.9 | 0.55 | +0.8 |
| 5 | 32.8 | 24.2 | 19.1 | 17.5 | 0.53 | +6.9 |
| 10 | 33.4 | 25.4 | 20.1 | 17.9 | 0.54 | +7.7 |
| 15 | 42.4 | 38.0 | 33.7 | 32.0 | 0.76 | +11.2 |
| 20 | 48.4 | 43.9 | 39.7 | 37.8 | **0.78** | +11.5 |
| 25 | 42.2 | 36.2 | 31.7 | 29.1 | 0.69 | +10.1 |
| 30 | 46.8 | 38.4 | 32.1 | 28.9 | 0.62 | +9.7 |
| 35 | 45.4 | 34.9 | 28.9 | 27.2 | 0.60 | +7.2 |
| 39 | 26.4 | 19.1 | 16.5 | 16.0 | 0.61 | +3.1 |

Averaged in thirds:

| | t-1 | t-2 | t-4 | t-8 | t-8/t-1 | ∪ gain @16 |
|---|---|---|---|---|---|---|
| early (1-13) | 32.2 | 24.4 | 20.3 | 19.0 | 0.59 | +6.7 |
| mid (14-26) | **42.5** | **37.3** | **32.7** | **30.4** | **0.72** | **+10.4** |
| late (27-39) | 40.9 | 31.7 | 26.6 | 24.5 | 0.60 | +7.1 |

**Routing memory is longest in the middle of the stack, and it is one phenomenon
rather than two.** The mid-stack does not merely persist more strongly from t-1
(42.5 against 32.2 early) — it also *decays more slowly*, holding 72% of its t-1
Coverage at t-8 where both ends hold ~60%. Across the 39 Layers the correlation
between a Layer's t-1 Coverage and its t-8/t-1 ratio is +0.65: Layers that
remember the previous Token also remember the one before that.

This extends rather than merely reproduces the published depth result. That one
established *where* persistence lives (mid-stack, matching Mixtral's Layer-0-at-
chance shape); this one says the mid-stack window is also where routing has the
longest memory, and that the two ends of the stack are short-memoried for
different reasons — Layer 1 because routing there barely persists at all (t-1
only 16.1%), Layer 39 because it has handed over to the Layer below.

The extremes are where the union earns least: +0.8pp at Layer 1 and +3.1pp at
Layer 39 against +11.5pp at Layer 20. A prefetcher filling a Budget of 16 from
older Tokens would be doing useful work in the middle 26 Layers and close to
nothing at either end.

## How fast history should be discounted

The vote's `decay`, swept on train and tuned at K=8:

| decay | K=8 | K=12 | K=16 |
|---|---|---|---|
| 0.000 (t-1 only) | 38.55 | 41.05 | 43.19 |
| 0.125 | 38.55 | 46.44 | 51.28 |
| 0.375 | 38.60 | 46.53 | 51.28 |
| **0.500** | **38.60** | **46.56** | 51.28 |
| 0.625 | 38.54 | 46.52 | **51.30** |
| 0.750 | 38.11 | 46.28 | 51.29 |
| 1.000 (equal vote) | 36.07 | 44.27 | 50.43 |

The surface is flat from 0.125 to 0.625 and falls off a cliff above it: an equal
vote costs **2.54pp** at K=8 against the tuned 0.5. So the ordering by recency
has to be respected, but *how* it is respected barely matters — anything that
puts t-1 clearly first and does not throw the older Tokens away lands within
0.05pp of the optimum. The optimum at 0.5 means t-4 enters at an eighth of t-1's
weight, which is roughly the point at which an older Token can no longer displace
a Slot t-1 named. That is consistent with everything above: the useful role of
history is to extend the list, not to re-rank it.

## With the cross-layer Predictor

Mixing the history mass with the gated cross-layer Predictor C at its published
`alpha = 0, beta = 1`, at a weight tuned per mixture on train.

| Predictor | Horizon | K=8 | K=12 | K=16 |
|---|---|---|---|---|
| cross-layer (C) | layer | 38.88 | 47.98 | 54.59 |
| combined (B+D, published) | layer | 40.26 | 49.59 | 56.30 |
| cross-layer + t-1 (`w=0.12`) | layer | 46.35 | 56.40 | 63.09 |
| **cross-layer + history** (`w=0.15`) | layer | **47.63** | **57.63** | **64.33** |

| Comparison | K=8 | K=12 | K=16 |
|---|---|---|---|
| cross+history over cross+t-1 | +1.28 ±0.08 | +1.23 ±0.06 | +1.24 ±0.10 |
| cross+history over C alone | +8.76 ±0.32 | +9.68 ±0.30 | +9.76 ±0.26 |
| cross+t-1 over published B+D | +6.11 ±0.36 | +6.83 ±0.28 | +6.81 ±0.27 |

**Older history is worth about +1.24pp inside the best Predictor, flat across
Budgets** — positive in all five folds and at 38 of 39 Layers (Layer 2 is the
exception, at −0.37pp). It concentrates where everything else in this document
does: +1.67pp mid-stack against +1.04 early and +1.14 late, peaking at +2.00pp
at Layer 20.

### An unplanned result: the published combination was mixing the wrong thing

`cross-layer + t-1` reaches **46.35%** at K=8 where the published winner reaches
**40.26%** on this same grid — **+6.11pp paired**, consistent across all five
folds, and larger than any gap in the original study. The two Predictors have
the same Horizon and the same cross-layer term. The only difference is what the
`token` side contributes: the published `combined` mixes in **D**, the fitted
transition table `P(E_t | E_t-1)`, and this mixes in **D0**, the raw copy of the
previous Token's 8 Slots.

That study's own finding 2 said copying beats learning the transition — 38.9%
against 33.0% standalone — and then finding 6 built the combination out of the
learned table anyway, because `combined` was defined as a mixture of two fitted
distributions and D0 ships no table to mix. The copy is not a distribution in
the same sense, but its gate vector is, and mixing *that* is both legal and much
better. The mixture weight tells the same story: it lands at `w = 0.12` here
against `w = 0.4` there, because the copy puts all its mass on 8 Slots and needs
only a small nudge to lift them above the cross-layer ranking.

This was not what the experiment set out to measure and it should be treated as
a lead rather than a result: it has been scored on train only, at one tuned
weight, and the test Split for `corpus_v1` is already spent.

## What this does not establish

**The gains above K=8 are Budget, not accuracy.** Every union result says the
same thing in different words: give a Predictor 16 Slots and older Tokens are
good candidates for the extra 8. Nothing here says routing is more *predictable*
than the published study found — at the Router's own top-k, it is not.

**Prefetch is still simulated.** No number here is a latency. A Budget of 16 is
25.6 MiB per Layer at 1.6 MiB an Expert, and whether that is affordable is a
question for colibri, not for this repo. What has changed is that filling those
16 from t-1 ∪ t-2 ∪ t-4 costs no table and no lead time — t-4 was known 200 ms
ago at 20 tok/s — where the published K=16 result needed a 2.6M-cell
cross-layer table inside 1.25 ms.

**Only one Lag set was swept for decay.** `(1, 2, 4)` was fixed before any
number was looked at, and the decay surface is reported for it alone. The
saturation result makes a wider sweep unpromising, but it was not run.

**Position and Lag are confounded at the grid edge.** Restricting to Tokens with
8 predecessors is itself a mild version of follow-up 3's position control, and
it moved every number by ~0.3pp. A Prompt's opening Tokens are both the ones
where deep Lags are unavailable and the ones where routing is least predictable;
this experiment removes them rather than explaining them.

**80 Prompts, one Corpus, greedy at temp=0.** Unchanged from the published
study, and the paired fold statistics above are not a substitute for a second
Corpus.

## If this is continued

The ordered queue is [`docs/queue.md`](../queue.md) — go there rather than
picking from this list, because these items interleave with the ones the
published study left. What this experiment added to it:

- **Re-examine `cross-layer + t-1` deliberately** — queue item 1, and the reason
  it is first: it settles what the best Predictor is, which item 2 then builds a
  policy on top of.
- **A second thing to choose per Layer** — the union is worth +11.5pp at
  Layer 20 and +0.8pp at Layer 1, so the per-Layer Horizon choice (queue item 2)
  now has two knobs rather than one.
- **Prefill is still untouched**, and the `token` Horizon remains undefined
  there — queue item 5.
