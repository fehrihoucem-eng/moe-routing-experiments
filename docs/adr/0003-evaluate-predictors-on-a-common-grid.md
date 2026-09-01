# Predictors are scored on a common grid, inside train, on one metric

Comparing Predictors is itself a modelling decision, so it happens entirely
inside the train Split — 5-fold cross-validation over the 80 train Prompts,
stratified by Category, folds of 16 — and the test Split stays unspent for a
single confirmation run of whichever Predictor wins. Every Predictor is scored
on the same cells, **Layers 1-39 and Decode Tokens 1..n-1**, because the
Predictors are not all defined everywhere and an uneven grid would decide the
comparison before any counting happened. Results are reported as **Coverage@K,
macro-averaged by Prompt**, per Layer.

## Considered options

Scoring on test directly would have spent the holdout on a four-way comparison
and left nothing to validate the winner. A fixed validation carve (12 fit / 4
validation per Category) was simpler but gives one draw and no variance
estimate, and it thins the fit set exactly where the transition tables are
already thin: on a cross-validation fit set the median (Layer, Slot) fires 246
times, but the 10th percentile fires 58 and 5.7% of conditioning rows fire fewer
than 30 times.

Letting each Predictor use every cell it is defined on was rejected because the
cells are not interchangeable. Cross-layer Predictors cannot enter Layer 0 at
all, and Layer 0 is the one Layer the literature agrees is anomalous — Mixtral
measures consecutive-Token repetition there at chance (~14% against a 12.5%
floor) while Layers 15 and 31 sit well above it. Granting a popularity Predictor
a Layer its rivals cannot enter, which is also the Layer where the effect its
rivals depend on is known to vanish, biases the comparison in both directions at
once.

Reporting Precision@K and Recall@K alongside Coverage was rejected because the
Router always selects exactly 8, so `|Predicted n Actual| / 8` **is** Recall@K
identically, and Precision@K is `Coverage * 8 / K`. Three columns would have
looked like triangulation while being one measurement printed three ways.

## Consequences

Per-Layer curves are the result and the pooled number is only a summary:
averaging 40 Layers folds a Layer-0 null into whatever structure sits mid-stack.
Popularity's full-grid numbers are reported separately as a footnote so nothing
is hidden by the common grid.

Macro-averaging by Prompt is what makes the fold spread meaningful, since folds
are Prompt-sets and the Prompt is this project's unit of independence
(ADR-0001). It also keeps Category imbalance out of a number that is not about
Category: Decode length varies roughly threefold across Categories, so
micro-averaging over cells would quietly weight the long Prompts.

Any later change to the grid, the fit discipline or the metric invalidates every
Coverage figure the project has published, in the same way a change to ADR-0001
would. Record such a change as its own decision rather than re-running with new
settings and comparing across them.
