# moe-routing-experiments

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

T.expert_histogram(X)                            # [256]  selections per expert
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

Both phases are captured; **every loader defaults to `split="decode"`**. Pass
`split="prefill"` or `split=None` for the rest.

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

## Layout

```
src/routetrace/
  parse.py       trace -> long-form records; phase and token_id assignment
  store.py       parquet store
  tensor.py      load_X, COO, to_torch, prompt_slices, describe
  transforms.py  transforms of X
  capture.py     SERVE-mode driver + chat template
tests/           23 tests; fixtures/serve3.trace is a real 3-prompt capture
patches/         the colibri engine change
prompts/         corpus_v1.json
scripts/         capture_corpus.py
data/            traces and stores (gitignored; regenerate with capture_corpus.py)
```

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```
