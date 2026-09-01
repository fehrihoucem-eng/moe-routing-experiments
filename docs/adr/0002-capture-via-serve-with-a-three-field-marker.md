# Capture through SERVE mode, delimited by a three-field Marker

A Capture runs every Prompt through one engine process (`SERVE=1`) rather than
one process per Prompt, because the ~12 s of weight load and VRAM warmstart would
otherwise be paid 100 times. That leaves every Prompt's rows concatenated into
one Trace, so the engine emits a `#prompt <key> <n_prefill_tokens>` Marker at each
request; Prompt boundaries and the Prefill/Decode split are then stated by the
producer instead of inferred by the reader.

The Marker has exactly **three** whitespace-separated fields, and that is not
cosmetic: colibri's own `tools/route_pairs.py` skips any line with fewer than
four fields, so a three-field Marker is invisible to it, while a four-field one
would be parsed as data and crash on `int("prompt")`. Any future Marker must
keep the field count under four.

## Considered options

Inferring boundaries from row counts instead of a Marker was rejected: a
single-Token Prompt has a one-row Prefill Forward and one-row Decode Forwards,
which are indistinguishable by width. Inferring from the Layer counter alone was
also rejected — it cannot see a boundary in a single-Layer model at all.
