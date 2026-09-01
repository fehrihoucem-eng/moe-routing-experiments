# Train/test splits are assigned per Prompt, not per Token

A Token-level split looks attractive because it gives far more units — 15,569
Decode Tokens against 100 Prompts — but Tokens within one Prompt share a prefix,
a topic and a single Decode trajectory, so the two sides would hold
near-duplicates of each other and any held-out score would be inflated. We
therefore assign whole Prompts, stratified by Category (16 train / 4 test each),
and accept that the test side is small and that its share of Tokens is 22.9%
rather than 20%, because Decode length varies per Prompt.

The Prompt is the right unit specifically because it is the independent one: the
engine resets its KV state per request, and a Prompt was measured to route
identically whether it ran alone or as #85 of 100.

## Consequences

Token-share imbalance across Categories (20.0% to 28.0%) is a consequence of
stratifying on Prompt count. If an experiment needs balanced Token mass instead,
that is a different split and should be recorded as its own decision, not by
silently reweighting this one.
