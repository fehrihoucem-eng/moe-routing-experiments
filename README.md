# moe-routing-experiments

> Vocabulary lives in [CONTEXT.md](./CONTEXT.md); decisions in [docs/adr/](./docs/adr/).
> Note **Expert** means a (layer, slot) pair — a slot number alone names 40 different Experts.

Routing-tensor experiments on Qwen3.6-35B-A3B, captured from
[colibri](https://github.com/JustVugg/colibri)'s `ROUTE_TRACE` stream.

Deliberately **outside** the colibri checkout: nothing here touches that repo's
branches. The one change colibri *does* need is vendored as a patch, see
[Engine patch](#engine-patch).

## The object: X

Everything is a transform of `X`, shaped `[tokens, layers, n_experts]`:

- `n_experts = 256`, `n_layers = 40`
- exactly **8 nonzero entries** along the expert axis per `(token, layer)` — the
  router's top-8
- the value is the **gate**, post-renormalisation, so each `(token, layer)` row
  sums to 1

```python
from routetrace import build_store, load_X, describe
from routetrace import transforms as T

build_store(["data/traces/serve3.trace"], "data/stores/serve3")
X, index = load_X("data/stores/serve3")          # decode only, [45, 40, 256]

T.expert_histogram(X)                            # [40, 256] per Expert (layer, slot)
T.cooccurrence(X, layer=12)                      # [256, 256] within-layer
T.transition(X, layer=12, delta=1)               # [256, 256] cross-layer
```

`index` is a structured array of `(prompt_id, phase, token_id)`, one entry per
row of X's token axis, so a slice of X always maps back to where it came from.
`prompt_slices(index)` gives `{prompt_id: slice}`.

`load_X(..., sparse=True)` returns a `COO` (`coords [3, nnz]`, `values [nnz]`)
instead — worth it past a few tens of thousands of tokens, where dense stops
being free (a 20k-token decode split is ~820 MB dense, ~6.4M nonzeros sparse).
`to_torch(X)` converts either form, and is the only thing that imports torch.

### Prefill vs decode

Both phases are captured; **every loader defaults to `phase="decode"`**. Pass
`phase="prefill"` or `phase=None` for the rest.

`token_id` restarts at 0 in each phase, so `(prompt_id, token_id)` is *not*
unique when both phases are loaded — the token axis keys on
`(prompt_id, phase, token_id)`. That is why `index` carries `phase`.

## Corpus

`prompts/corpus_v1.json` — 100 English prompts, 20 per category: `coding`,
`math_reasoning`, `factual_expository`, `conversational_creative`,
`structured_extraction`.

Captured with `scripts/capture_corpus.py` (greedy `temp=0`, thinking off,
`max_tok=200`) into `data/stores/corpus_v1`: 100 prompts, **15,569 decode
tokens**, 6.15M routing rows, 13 min. 67 of 100 responses hit the 200-token cap,
so decode length is censored at 200 — `structured_extraction` is by far the
shortest (69 tokens/prompt on average; extraction answers are simply short).

Slice by category:

```python
X, index = load_X("data/stores/corpus_v1", categories="coding")
X, index = load_X("data/stores/corpus_v1", categories=["coding", "math_reasoning"])
```

The full decode split is `[15569, 40, 256]` = **638 MB dense**; use
`sparse=True` when you want the whole corpus at once.

### Train / test

The split is **by prompt, never by token**: tokens inside one prompt share a
prefix, a topic and a decode trajectory, so a token-level split would put
near-duplicate rows on both sides and inflate every held-out score.

```python
make_split("data/stores/corpus_v1", test_per_category=4, seed=0)  # writes split.json
X, index = load_X("data/stores/corpus_v1", split="train")   # [12000, 40, 256]
X, index = load_X("data/stores/corpus_v1", split="test")    # [3569, 40, 256]
```

`corpus_v1` at `seed=0` is 16 train / 4 test in each of the five categories.
Note the token shares are *not* 80/20 — decode length varies per prompt, so the
test side carries 22.9% of decode tokens, ranging 20.0% (`factual_expository`,
`math_reasoning`) to 28.0% (`structured_extraction`). Stratification is on prompt
count, which is what makes the units independent; token balance is a consequence.

The assignment is written to `<store>/split.json` rather than recomputed, so two
experiments that "both used seed 0" stay comparable even if one rebuilt the store
in between. `make_split` refuses to overwrite without `overwrite=True`.

> `phase=` selects prefill/decode. `split=` selects train/test. Passing
> `split="decode"` raises rather than silently filtering to nothing.

## Predicting the routing

`docs/results/predictor-coverage.md` is the first result: how much of the
Router's top-8 you can name in advance, and from what.

A **Predictor** names K Slots for a target (Token, Layer); **Coverage** is the
fraction of the 8 it got. Predictors are grouped by **Horizon** — how much is
known when they fire — because that, not accuracy, is what decides whether a
prediction is usable:

- `token`: everything about Token t−1, known ~50 ms ahead (40 Layers at 20 tok/s)
- `layer`: that, plus the current Token one Layer down — ~1.25 ms

One Expert is 1.6 MiB at int4 gs64, so a single miss costs ~0.33 ms to fetch at
5 GB/s. The `layer` Horizon cannot cover its own miss; the `token` Horizon has
40x the room.

```python
from routetrace import load_routing, fit_tables, score, coverage, grid_rows
import numpy as np

r = load_routing("data/stores/corpus_v1", split="train")   # compact: slots + gates
tables = fit_tables(r, fit_rows)                           # counts, fit rows only
rows = grid_rows(r, score_rows)                            # tokens with a predecessor
s = score("combined", r, tables, rows, layer=20, alpha=0.0, beta=0.75, w=0.4)
coverage(s, r.slots[rows, 20].astype(np.int64))            # [rows, len(BUDGETS)]
```

Every conditional Predictor is one estimator with two knobs, so `popularity` is
not a separate thing — it is the `alpha → ∞` limit, and `cross_layer` at
`beta=0` and `beta=1` are the unweighted and gate-weighted variants:

```
P(j|i)   = (c(i,j) + alpha * pop(j)) / (c(i) + alpha)
score(j) = sum_i (g_i ** beta) * P(j|i)
```

The headline, at K=8: copying the previous Token's Slots — free, no table —
gets **38.9%**; the best Predictor of any kind gets **40.6%**. See the results
document for the K-curve, the depth profile and the caveats.

```sh
.venv/bin/python scripts/run_predictors.py                    # ~6 min, CPU only
.venv/bin/python scripts/export_predictors_page.py            # site/predictors.html
.venv/bin/python scripts/confirm_on_test.py --predictor combined --confirm
```

`site/predictors.html` is the interactive version — the K-curve table, and
Coverage against Layer for every Budget. Self-contained, so it opens over
`file://`. Like `routing.html` it shows **train only**.

`run_predictors.py` reads **train only** and chooses; `confirm_on_test.py` spends
the holdout, refuses to run without `--confirm`, and refuses to run twice.

### Determinism

`serve_sample()` treats `temp <= 0` as exact argmax. Verified, not assumed:
two separate engine processes over the same prompts produce **byte-identical**
traces, and a prompt routes identically whether it runs alone or as #85 of 100 —
`serve_one()` resets KV per request, so requests do not leak into each other.

## The store

`build_store()` writes a parquet store, which is the canonical form:

```
routing.parquet   prompt_id, phase, token_id, layer, expert_id, gate
prompts.parquet   prompt_id, key, category, n_prompt_tokens, n_decode_tokens,
                  source, text, response
meta.json         n_layers, n_experts, top_k, n_rows, n_prompts, traces
split.json        seed, test_per_category, train/test prompt ids (make_split)
```

One row per selected expert, so a `(token, layer)` contributes exactly `top_k`
rows and X is only ever materialised on demand. `top_k` is measured off the data
rather than assumed, so a container with a different top-k cannot be mislabelled
silently. Several traces concatenate into one store with `prompt_id` offset per
file.

## Capturing

`SERVE=1` runs many prompts through one engine process, so the ~12 s of weight
load and VRAM warmstart is paid once rather than per prompt:

```python
from routetrace.capture import capture

capture(
    prompts=["What is 2+2?", "Name one primary colour."],
    trace_path="data/traces/run.trace",
    model_dir="/home/houcem-fehri/Models/qwen36_i4_gs64",
    engine="/home/houcem-fehri/colibri/c/qwen36",
    max_tok=128,
    env_extra={"COLI_CUDA": "1", "COLI_GPUS": "0", "CUDA_EXPERT_GB": "auto",
               "OMP_NUM_THREADS": "16"},
)
```

`np + max_tok` must stay under the engine's context (8192 by default) or
`serve_one()` rejects the request — the same trap that makes a bare `coli chat`
answer every message with HTTP 400.

## Engine patch

Upstream qwen36 emits no routing telemetry: it never included `route_trace.h`,
so `ROUTE_TRACE=` was silently ignored. `patches/colibri-qwen36-route-trace.patch`
adds it, against colibri `184e052`:

- `rt_route()` after the top-k renormalisation, so traced gates are the applied ones
- `rt_prompt()` at each SERVE request and once on the argv path, emitting a
  `#prompt <key> <n_prompt_tokens>` marker
- `"qwen36"` added to `rt_engine_names[]`; `route_trace.h` added to the Makefile deps

The marker is deliberately **three** whitespace-separated fields, because
`c/tools/route_pairs.py` skips lines with fewer than four — a four-field marker
would make every existing reader die parsing `"prompt"` as an int.

```sh
cd /path/to/colibri && git apply /path/to/patches/colibri-qwen36-route-trace.patch
make -C c qwen36 CUDA=1 CUDA_HOME=/usr CUDA_ARCH=compute_90   # this box: nvcc 12.4 vs sm_120
```

Tracing does not perturb the model: routing is decided on the CPU even when the
CUDA tier runs the experts, and measured throughput was unchanged (20.04 tok/s
traced vs 20.68 baseline).

## Explorer

`site/routing.html` is a self-contained page (data inlined, so it works over
`file://`) covering the shape of X, the Expert-vs-Slot distinction, where
categories separate by depth, and the corpus split. Rebuild it with:

```sh
.venv/bin/python scripts/export_site_data.py
```

It reads the **train** split only, so nothing on the page can inform a decision
the held-out split is meant to validate.

## Layout

```
src/routetrace/
  parse.py       trace -> long-form records; phase and token_id assignment
  store.py       parquet store
  tensor.py      load_X, COO, to_torch, prompt_slices, describe
  splits.py      make_split / read_split: stratified train-test by prompt
  transforms.py  transforms of X (expert_histogram keeps the layer axis)
  predict.py     Predictors, the shrinkage estimator, Coverage, CV folds
  capture.py     SERVE-mode driver + chat template
tests/           64 tests; fixtures/serve3.trace is a real 3-prompt capture
patches/         the colibri engine change
prompts/         corpus_v1.json
scripts/         capture_corpus.py, export_site_data.py, run_predictors.py
site/            routing.html (built from routing.template.html)
docs/results/    predictor-coverage.md and its JSON (committed: not regenerable
                 without the GPU capture)
data/            traces and stores (gitignored; regenerate with capture_corpus.py)
```

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```
