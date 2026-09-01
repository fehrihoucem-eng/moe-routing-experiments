# Routing Experiments

Studies which experts a Mixture-of-Experts model selects while it answers, and
how that selection varies with the kind of question. The model is
Qwen3.6-35B-A3B run on colibri; this repo owns the capture, the dataset and the
analysis, never the engine.

## Language

### What the model does

**Layer**:
One of the model's 40 stages. Every layer has its own router and its own set of
experts.
_Avoid_: block, level

**Slot**:
A position in a layer's expert table, numbered 0 to 255. A slot number is
meaningless on its own, because each layer numbers its own experts from zero.
_Avoid_: expert id, expert number, expert index

**Expert**:
One small feed-forward network, identified by the pair (Layer, Slot). There are
10,240 of them — 40 layers times 256 slots — and colibri counts them exactly
that way. Slot 151 of layer 0 and slot 151 of layer 20 are different Experts and
must never be summed together.
_Avoid_: naming an Expert by its Slot alone

**Router**:
The part of a layer that scores all 256 of its Experts for one Token and selects
the best 8.
_Avoid_: gate network, gating

**Gate**:
The weight the Router gives one selected Expert, after renormalisation. The 8
gates of one (Token, Layer) sum to 1.
_Avoid_: score, logit, weight, probability

### Producing the data

**Prompt**:
One question submitted to the engine, and the unit of independence in this
project: the engine resets its state between Prompts, so no Prompt influences
another. Everything that must not leak across train and test is decided per
Prompt.
_Avoid_: request, query, sample, example, item

**Category**:
The kind of question a Prompt is, one of five: `coding`, `math_reasoning`,
`factual_expository`, `conversational_creative`, `structured_extraction`.
_Avoid_: domain, topic, task type, class, label

**Corpus**:
A fixed, versioned set of Prompts with their Categories. `corpus_v1` is 100
Prompts, 20 per Category.
_Avoid_: dataset, prompt set

**Capture**:
One run of the engine over a Corpus, producing a Trace. A Capture is
reproducible: greedy decoding makes two Captures of the same Prompts byte-equal.
_Avoid_: run, generation, inference pass

**Trace**:
The engine's raw output file, one line per (Forward, Layer), listing the 8
selected Slots and their Gates. Written by colibri, never by this repo.
_Avoid_: log, dump, routing file

**Marker**:
The `#prompt` line a Trace carries at the start of each Prompt. It states where
one Prompt's rows begin and how long its Prefill is, so neither has to be
inferred.
_Avoid_: header, delimiter, separator

**Forward**:
One pass of the model over one or more Tokens. A Prefill Forward covers the
whole question at once; a Decode Forward covers exactly one Token.
_Avoid_: step, iteration, call

**Phase**:
Whether Tokens were read or written — `prefill` or `decode`. Always the word for
this distinction; never "split".
_Avoid_: split, stage, mode

**Prefill**:
The Phase that reads the Prompt.

**Decode**:
The Phase that writes the answer. The only Phase the experiments currently use.
_Avoid_: generation, sampling

**Token**:
One position the model processed. Tokens are numbered from zero within a Prompt
*and within a Phase*, so a Token is only identified by (Prompt, Phase, position).
_Avoid_: word, step, position alone

### Working with the data

**Store**:
The parsed, canonical form of one or more Traces: a directory of Parquet files
holding one row per selected Expert. The Store is the source of truth; anything
else is derived from it.
_Avoid_: database, cache, dataset

**X**:
The tensor every experiment starts from, shaped [Tokens, Layers, 256], holding
Gates and zeros elsewhere. Exactly 8 entries along the last axis are non-zero.
_Avoid_: matrix, data, features, tensor (unqualified)

**Index**:
The table that names each row of X's Token axis, as (Prompt, Phase, Token). X
without its Index cannot be traced back to a Prompt.
_Avoid_: labels, metadata, ids

**Split**:
Which side of the train/test division a Prompt is on — `train` or `test`.
Assigned per Prompt and stratified by Category. Never means Prefill vs Decode.
_Avoid_: phase, fold, partition, subset

**Transform**:
A function of X that produces the input to one experiment. Experiments are
defined as Transforms so that they share one capture.
_Avoid_: feature, preprocessing, pipeline

### Predicting the routing

**Predictor**:
A function from routing that is already known to a ranked list of Slots for a
target (Token, Layer). Every Predictor names the same kind of thing, whether it
is the crudest or the best, so none of them is privileged by its name.
_Avoid_: model, estimator, baseline

**Horizon**:
How much a Predictor knows when it fires. `token` means everything about the
Tokens before this one, known before the current Token enters the model at all;
`layer` means that plus the current Token's Experts at the Layer below. Two
Predictors are only comparable within one Horizon, because the wider Horizon is
strictly more information.
_Avoid_: lookahead, context, window

**Lag**:
How many Tokens back a `token`-Horizon Predictor reaches, written t-1, t-2, t-4.
A Lag is not a Horizon: every Lag is available at the same moment, so reaching
further back costs no lead time and buys no wider Horizon.
_Avoid_: window, order, history length

**Budget**:
How many Slots a Predictor names, written K. Distinct from the Router's own 8,
which is fixed by the model — a Budget below 8 cannot reach full Coverage no
matter how good the Predictor is.
_Avoid_: top-k, k

**Coverage**:
Of the 8 Experts the Router actually selected, the fraction a Predictor named.
The one number a Predictor is judged on. Precision is `Coverage * 8 / K` and
carries no separate information.
_Avoid_: recall, hit rate, accuracy

**Miss**:
A selected Expert the Predictor did not name. The same measurement as Coverage,
counted rather than expressed as a fraction: `Misses = 8 * (1 - Coverage)`.
_Avoid_: error, failure, fault
