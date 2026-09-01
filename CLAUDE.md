# moe-routing-experiments

Analysis of expert routing in Qwen3.6-35B-A3B, captured from colibri. This repo
owns the capture, the dataset and the analysis — never the engine.

## Read these before starting work

| File | What it holds | When you need it |
|---|---|---|
| `CONTEXT.md` | The glossary. Opinionated, with `_Avoid_` lists. | Before using any project term. It is a glossary only — put nothing else in it. |
| `docs/adr/` | Decisions that are hard to reverse, and why. | Before changing how capture or the split works. |
| `docs/research/moe-expert-specialisation.md` | The MoE routing literature, cited to primary sources, with unverifiable claims flagged. | **Before researching this topic again — it has already been done.** |
| `README.md` | How to run things. | — |

## Traps that have already cost time

**An Expert is a `(layer, slot)` pair.** Each of the 40 layers numbers its own 256
experts from zero, so slot 151 of layer 0 and slot 151 of layer 20 are different
weight matrices. `expert_histogram()` keeps the layer axis; `slot_histogram()`
collapses it and is almost never what you want. Conflating these once produced a
published-to-the-user result that was wrong by a wide margin (category cosine
read 0.88–0.94 collapsed, 0.20–0.54 correct).

**`phase=` and `split=` are different arguments.** `phase` is `prefill`/`decode`;
`split` is `train`/`test`. `load_X(split="decode")` raises on purpose.

**The test split is held out.** Analysis, plots and the explorer page read
`split="train"` only. Do not look at test to make a modelling decision.

**The shared expert is not in the data.** Qwen3.6 has 1 shared expert alongside
the 256 routed ones, but `rt_route()` fires only inside the routed top-k loop, so
it never reaches the trace. Routed profiles are therefore a *residual* and will
look more differentiated than published numbers for architectural reasons alone.

**Full support over 256 slots is not a finding.** With 8 selections per token,
P(a slot is never picked) is ~1e-14 even for the smallest category. Every slot
being used was forced by sample size. Use MaxVio or counts either side of
`k/N = 3.125%` instead.

**Gates are only accurate to `top_k * 5e-5`.** The trace prints them as `%.4f`,
so a renormalised row sums to 1 within 4e-4, not to machine precision.

## The engine is patched, not stock

Upstream qwen36 emits no routing telemetry. `patches/colibri-qwen36-route-trace.patch`
adds it, against colibri `184e052`. On this machine the build needs both overrides
(distro nvcc at `/usr/bin`, and CUDA 12.4 cannot target the RTX 5090's sm_120):

```sh
make -C c qwen36 CUDA=1 CUDA_HOME=/usr CUDA_ARCH=compute_90
```

## Commands

```sh
.venv/bin/python -m pytest -q                      # 33 tests
.venv/bin/python scripts/capture_corpus.py         # re-capture corpus_v1 (~13 min, needs the GPU)
.venv/bin/python scripts/export_site_data.py       # rebuild site/routing.html
```

`data/` is gitignored and regenerable; everything in it comes from those scripts.
