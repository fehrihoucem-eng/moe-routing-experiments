# Expert specialisation and routing behaviour in sparse MoE language models

A literature review, written to inform analysis of `corpus_v1` Captures.
Every claim below is attributed inline to a paper the author retrieved and read.
Claims that could not be traced to a primary source are marked **[UNVERIFIED]**.

Terminology follows `CONTEXT.md` (Layer, Slot, Expert, Router, Gate, Category,
Decode). Where a paper's own vocabulary differs, the paper's word is quoted.

---

## 1. Summary: established, contested, not addressed

**Established** (multiple independent primary sources agree):

- Routing correlates strongly with **token identity and surface form**, not with
  topic, in coarse-grained MoEs. Mixtral [Jiang et al. 2024, §5], OpenMoE
  [Xue et al. 2024, §4.1–4.2] and ST-MoE's decoder [Zoph et al. 2022, §7.2] all
  report this independently.
- Routing decisions **stabilise very early in pre-training** and barely move
  afterwards, including through fine-tuning. OLMoE measures ~60% of the top-8
  set already fixed after 1% of pre-training [Muennighoff et al. 2024, §5.1];
  OpenMoE finds checkpoint-to-checkpoint expert preferences "almost totally
  overlapped" [Xue et al. 2024, §4.2]; ExFlow finds inter-layer affinity is
  acquired "at the very early training stage" [Yao et al. 2024].
- **Consecutive Tokens share Experts far above chance**, and much more so in
  middle and late Layers than in Layer 0 [Jiang et al. 2024, §5, Table 5].
- The **auxiliary load-balancing loss actively suppresses measurable domain
  specialisation**. This is now shown causally three ways: by scope
  [Qiu et al. 2025, §4.2], by removing the loss entirely
  [DeepSeek-AI 2024, §4.5.3 and Fig. 9], and theoretically
  [Wang, Hayou & Nalisnick 2026, §4.1, Prop. 2].
- **Architecture changes the answer.** Fine-grained (many small) Experts plus a
  shared Expert measurably reduce redundancy between routed Experts
  [Dai et al. 2024, §4.5]; from-scratch training rather than upcycling appears
  necessary for domain specialisation to appear at all
  [Muennighoff et al. 2024, §5.3].

**Contested:**

- **Whether domain/topic specialisation exists at all.** Mixtral says no
  [Jiang et al. 2024, §5]; OLMoE says yes and blames Mixtral's upcycling
  [Muennighoff et al. 2024, §5.3]; DeepSeek-V3 says yes but only once the
  auxiliary loss is removed [DeepSeek-AI 2024, §4.5.3]; a 2026 preprint argues
  the whole framing is confounded because routing similarity is fully explained
  by hidden-state similarity [Wang, Hayou & Nalisnick 2026, §4.1].
- **Whether routing is context-independent.** OpenMoE concludes routing follows
  "the Token ID instead of high-level semantics" [Xue et al. 2024, §4.1];
  Wang, Hayou & Nalisnick directly contradict this on newer models, reporting
  high same-Token cross-context overlap only in the first Layer and shrinking
  overlap after, and attribute OpenMoE's result to "insufficient training"
  [2026, §4.2, Fig. 4].
- **The layer-wise shape of specialisation** — see §3 below. Language-conditioned
  and domain-conditioned routing show *opposite* depth profiles in the one paper
  that measures both [Bandarkar et al. 2026, §4.3].

**Not addressed by the literature** (as far as this review found):

- Routing over **Decode Tokens the model itself produced**, as a population.
  Essentially all published routing analyses are forward passes over a fixed
  corpus. The single exception found is [Wang, Hayou & Nalisnick 2026, §5.2],
  and its result is that the two are *not* interchangeable. See §5.
- Any agreed **null model** for routing-distribution distances (permutation
  baselines, bootstrap CIs, sample-size correction).
- Any standard for **Category-conditioned analysis of task types** as opposed to
  data domains, languages or programming languages.
- **Snowflake Arctic** has no primary technical paper — architecture is described
  only in Snowflake's blog and "cookbook" posts. Any Arctic routing claim is
  **[UNVERIFIED]** here. **JetMoE** [Shen et al. 2024] was checked directly and
  contains *no* routing or specialisation analysis; it discusses load balancing
  only as a training constraint (§2.4) and explicitly says it could not afford
  architecture ablations (§7).

---

## 2. Question 1 — Topic vs token-level specialisation

### 2.1 What Mixtral actually claims

The relevant section is **§5, "Routing analysis"** of *Mixtral of Experts*
[Jiang et al. 2024]. It analyses Layers 0, 15 and 31 (of 32) on **The Pile
validation set**, over these subsets (Table 5): ArXiv, DM Mathematics, Github,
Gutenberg, PhilPapers, PubMed Abstracts, StackExchange, Wikipedia (en).

Findings, precisely:

1. **No domain specialisation.** "Surprisingly, we do not observe obvious
   patterns in the assignment of experts based on the topic." Expert-selection
   distributions are near-identical across ArXiv, PubMed Abstracts and
   PhilPapers at every Layer measured. The sole exception is DM Mathematics,
   which shows "a marginally different distribution of experts", attributed to
   its synthetic nature and narrow vocabulary, and visible mostly at the first
   and last Layers "where the hidden states are very correlated to the input and
   output embeddings respectively".
2. **Syntactic structure instead.** "This suggests that the router does exhibit
   some structured syntactic behavior." Concretely: multi-token words such as
   `self` in Python or `Question` in English tend to route together, and in code
   "the indentation tokens are always assigned to the same experts". Figure 8
   shows selection "more aligned with the syntax rather than the domain".
3. **Positional locality.** Consecutive Tokens frequently share Experts. Table 5
   gives the proportion where Token *i* and *i+1* share the first-choice Expert:
   ~13.6–14.9% at Layer 0, ~23.6–28.4% at Layer 15, ~19.7–26.3% at Layer 31,
   against a uniform baseline of 1/8 = 12.5%. For "first or second choice" the
   numbers are ~45–50% (L0), ~62–67% (L15), ~44–54% (L31) against a ~46%
   baseline. The caption's own summary: "Repetitions at the first layer are
   close to random, but are significantly higher at layers 15 and 31."

So the widely repeated summary of Mixtral is accurate, with two important
qualifications usually dropped: the analysis covers **only 3 of 32 Layers**, and
the Layer-0 repetition rate is **at chance**, i.e. the "high consecutive-token
repetition" finding is a mid/late-Layer phenomenon.

### 2.2 What later work says

**Confirming the token-level view.**
*OpenMoE* [Xue et al. 2024, §4] is the strongest confirmation and goes further.
On RedPajama subsets, TheStack code languages, TED parallel text and MT-Bench,
domain-level routing is near-uniform, while Token-ID plots show identical Token
IDs going to the same Experts regardless of context — their conclusion is that
"MoE simply routes based on the Token ID instead of high-level semantics" (§4.1).
Critically for anyone measuring domain effects, they show the confound
explicitly: Expert 21 appears code/math-specialised only because it prefers
`=`, `and` and `\n` (§4.2, Table 9). They also replicate Token-ID specialisation
on DeepSeek-MoE-16B but *not* on Mixtral, which they attribute to Mixtral's
upcycling from dense Mistral-7B (§4.4).

*ST-MoE* [Zoph et al. 2022] predates Mixtral and splits the difference by
component. **§7.1 "Encoder Experts Exhibit Specialization"** finds experts for
"punctuation, conjunctions & articles, verbs, visual descriptions, proper names,
counting & numbers" (Table 13) — i.e. *syntactic/lexical* categories, not
topics. **§7.2 "Decoder Experts Lack Specialization"**: "we also do not observe
meaningful specialization (semantics or syntax) in decoder experts", supported
by Table 14 showing decoder routing entropy near the uniform value of 3.5 for
32 experts. §7.3 adds "we find no evidence of language specialization".
This matters here: **a decoder-only model's routing is the case ST-MoE found
*least* specialised.**

**Qualifying or contradicting it.**
*OLMoE* [Muennighoff et al. 2024] is the most detailed routing analysis
available and reaches the opposite conclusion for its own model (16 Layers,
64 Experts, top-8, trained from scratch). §5 is organised as router saturation
(§5.1), expert co-activation (§5.2), domain specialisation (§5.3) and vocabulary
specialisation (§5.4).

- **§5.3 domain specialisation.** Metric (Eq. 7): the proportion of a domain's
  Tokens for which Expert *i* is in the top-k, against a uniform baseline of
  8/64 = 12.5%. On arXiv, GitHub and C4 they report "many examples of experts
  that are activated significantly above or below random chance", with a Layer-0
  Expert "nearly 100% specialized" for arXiv, and GitHub/arXiv frequently
  co-activating at Layer 7. Generic C4 activations are "much more balanced".
- **The direct Mixtral comparison.** In the same figure (Fig. 22, bottom) they
  measure Mixtral-8x7B and find it "exhibits little domain specialization across
  both unique and generic domains", near the 2/8 = 25% uniform baseline. Their
  explanation: "The initialization from a dense model may limit the amount of
  possible specialization in the experts as they all start from the same local
  optimum." This is a **direct, same-methodology replication of Mixtral §5 that
  agrees with it** while arguing the result does not generalise.
- **§5.4 vocabulary specialisation** is where OLMoE's specialisation is
  clearest, and it is *lexical*, not topical: Expert 27 takes non-Latin scripts,
  Expert 58 punctuation and brackets, Expert 7 religious vocabulary
  (`Jesus` 98%, `God` 90%), Expert 37 days/events (`Sunday` 100%), Expert 3
  kinship terms. Note this is compatible with Mixtral's and OpenMoE's picture:
  strong specialisation over *token identity*, which then produces apparent
  domain effects.
- **§5.2 co-activation** (Eq. 6, N(Ei,Ej)/N(Ei)): "there is no strong
  co-activation among experts in one layer, with only few exceptions", read as
  "little redundancy across different experts".

*DeepSeekMoE* [Dai et al. 2024] argues architecturally rather than by
measurement. §3.1 motivates fine-grained segmentation by combinatorics: N=16
top-2 gives 120 combinations; splitting each into 4 (64 Experts, top-8) gives
4,426,165,368. §4.5 gives the only quantitative specialisation evidence, all
indirect via Pile loss: masking the highest-probability Experts degrades
DeepSeekMoE much faster than GShard×1.5 ("This sensitivity suggests a lower
level of parameter redundancy in DeepSeekMoE", Fig. 4); disabling the shared
Expert and activating one more routed Expert raises Pile loss "from 1.808 to
2.414"; and even 4 activated routed Experts match GShard (Fig. 5). **There is no
domain-conditioned routing analysis in DeepSeekMoE** — the "ultimate expert
specialization" of the title is argued, not measured.

*DeepSeek-V3* [DeepSeek-AI 2024] supplies the causal result. 61 Layers, 256
routed + 1 shared Expert, top-8, first 3 Layers dense (§4.2). Balancing is
auxiliary-loss-free: a per-Expert bias is added to affinity scores for
*selection only* — "the bias term is only used for routing" (§2.1.2) — with a
sequence-wise balance loss at α = 0.0001 kept purely "to avoid extreme imbalance
within any single sequence". §4.5.3 compares two 16B models trained with and
without the auxiliary loss on Pile-test domains, measuring normalised load ("the
ratio between the actual expert load and the theoretically balanced expert
load"). Fig. 9's caption: "The auxiliary-loss-free model shows greater expert
specialization patterns than the auxiliary-loss-based one." The mechanism given
is scope — batch-wise balancing "does not enforce in-domain balance on each
sequence", and this "flexibility allows experts to better specialize in
different domains".
*(Appendix C carries the per-Layer version of this figure; its body could not be
retrieved — the arXiv HTML truncated before it. The §4.5.3 text and Fig. 9
caption above were read directly.)*

*Demons in the Detail* [Qiu et al. 2025] is the cleanest controlled evidence
that the *measurement* is an artefact of the training loss. They define
"Balance BSZ" as the number of Tokens over which Expert selection frequency is
computed for the loss (§4.1). Under standard micro-batch balancing "the
micro-batch LBL is almost calculated at the sequence level", which they call
"an overly strict constraint". Measuring per-domain top-k selection frequency on
held-out SFT-Code, SFT-Math and EN-Literature (§4.2): under micro-batch
balancing frequencies are near-uniform and "none exceed 0.15"; under global-batch
balancing "many experts in SFT-Math having frequencies exceeding 0.2". A shuffle
control (§5) confirms token *diversity*, not token *count*, drives it.

*Qwen3* [Qwen Team 2025, §2] adopts exactly this: 128 Experts, 8 activated,
fine-grained segmentation after DeepSeekMoE, no shared Experts, and "we adopt
the global-batch load balancing loss ... to encourage expert specialization".
So Qwen-family MoEs are trained under the regime that Qiu et al. show *permits*
domain specialisation to appear.

*The Myth of Expert Specialization in MoEs* [Wang, Hayou & Nalisnick 2026] is
the strongest contrary voice and is directly relevant to your method. Because
the Router is a linear map, they argue "hidden state similarity is both
necessary and sufficient to explain expert usage similarity" (§1, Prop. 1), so
observed specialisation is "an emergent property of the representation space,
not of the routing architecture itself". Empirically (§5.1, Fig. 6, Jaccard
overlap at top-p = 0.8 over rollouts of HMMT problems on three MoEs) two
different models answering the *same* question share only "∼60% expert overlap
on average", which is comparable to the same model on *different* questions and
far below the same-model same-question condition — "solving the same problem
with a different model is as 'foreign' as solving an entirely different
problem". **This is a 2026 preprint, listed as v1 with no venue; treat as
unreplicated.**

### 2.3 Verdict on Question 1

The honest reading: **specialisation over token identity and surface form is
established; specialisation over topic is real but small, architecture-dependent
and heavily modulated by the load-balancing regime.** Nobody has shown large
topic specialisation in a model trained with per-micro-batch balancing. The two
papers reporting clear domain effects (OLMoE, DeepSeek-V3) both trained from
scratch, and DeepSeek-V3's effect only appears once the auxiliary loss is gone.

---

## 3. Question 2 — Depth

There is **no single established pattern**, but there are three recurring and
partly reconcilable observations.

**(a) Layer 0 is an outlier, in every paper that looks.**
Mixtral: consecutive-Token repetition at Layer 0 is at chance (~14% vs 12.5%),
unlike Layers 15 and 31 [Jiang et al. 2024, Table 5]. OLMoE: "Layer 0 is an
outlier saturating significantly more slowly than other layers", linked to
DeepSeekMoE's choice to omit an MoE in the first Layer because load balancing
converges slowly there [Muennighoff et al. 2024, §5.1]. DeepSeek-V3 makes the
first three Layers dense outright [DeepSeek-AI 2024, §4.2]. Bandarkar et al.
find Phi-3.5-MoE "activates the same few experts in the first layer for all
languages" [2026, §4.3]. Wang et al. find same-Token cross-context overlap is
highest in the first Layer [2026, §4.2, Fig. 4].

**(b) First and last Layers are where input/output embedding structure shows.**
Mixtral attributes both its DM Mathematics exception and its strongest syntactic
effects to the first and last Layers, "where the hidden states are very
correlated to the input and output embeddings respectively" [Jiang et al. 2024,
§5]. Lo et al. find Mixtral Expert-weight similarity drops with depth but "many
values significantly grow to greater than 0.8 in the last two layers", and for
DeepSeek "the values in the last layer are significantly larger"
[Lo et al. 2024, §5.1]. OLMoE finds "routing in later layers saturates earlier
during pretraining" [Muennighoff et al. 2024, §5.1].

**(c) The depth profile of conditioned routing depends on what you condition
on — and this is the key result for your dataset.**
[Bandarkar et al. 2026] is the only work found that measures both. §4.3,
Finding 1, on four MoEs (OLMoE, Qwen3-30B-A3B, Phi-3.5-MoE, GPT-OSS-20B), for
**languages**: "For all languages, there is much higher routing divergence from
English in the first and last layers than in the intermediate" — "The overall
trend is this U-shape for all languages". For **domains** (AlpaCare-MedInstruct
medical, GSM8K-Instruct math, against English FLoRes), the same section reports
the reverse: "For these domains, routing divergence from the generic domain
exhibits the opposite pattern: higher divergence exists in the middle layers
(more of a ∩-shape). However, these patterns are weaker, as domains are less
different than languages." They read this as "separation of parameterization
between multilinguality and task-specific capabilities", supported in §5.2 by
zero overlap between multilingual and task Experts at threshold τ ≥ 0.3.

**(d) A contrary depth claim.** Wang, Hayou & Nalisnick report the opposite tail
behaviour: at prefill, unrelated sequences "activate nearly identical experts in
deeper layers", which they tie to representation rank collapse — Tokens "become
more correlated with depth", which "decreases the diversity of expert usage"
[2026, §4.1–4.2, Fig. 2]. Their synthesis is that early-Layer states reflect
input semantics while "later-layer representations may instead encode the
model's anticipated output".

**(e) Cross-layer structure is not independent.** ExFlow measures "the
conditional probability in tokens' routing across multiple layers" and finds
pre-trained GPT MoE models "implicitly exhibit a strong inter-layer expert
affinity", exploited for placement to cut cross-GPU routing latency "up to 67%"
across configurations "from 8 to 64" experts [Yao et al. 2024]. Path-Constrained
MoE frames the same object as "expert paths" — the sequence of Expert choices
across all Layers — and reports that tokens "cluster into a small fraction of
paths that align with linguistic function" [Gu et al. 2026]. *(Abstract page only
was read for Gu et al.; its quantitative cross-layer correlation figures were
not verified.)*

**Verdict on Question 2:** established that Layer 0 behaves differently and that
depth matters; established that later Layers saturate earlier during training;
**contested** whether late Layers converge (Wang et al.) or diverge (Bandarkar
et al. for languages) across conditions. For *task/domain* conditioning
specifically, the only direct measurement reports **maximum divergence in the
middle Layers**.

---

## 4. Question 3 — Standard metrics and analysis methods

| Metric | Definition as used | Primary source |
|---|---|---|
| **Expert utilisation / load** | Fraction of Tokens dispatched to Expert *i*, `f_i`, and fraction of Router probability mass `P_i` | Switch Transformer [Fedus et al. 2021, Eq. 5–6] |
| **Auxiliary load-balancing loss** | `α · N · Σ f_i P_i`, α = 1e-2, "encourages uniform routing since it is minimized under a uniform distribution" | [Fedus et al. 2021, Eq. 4] |
| **MaxVio** | `(max_i Load_i − mean Load) / mean Load`, in global and per-batch variants | [Wang et al. 2024, §4.1, Eq. 4] |
| **Routing entropy** | Entropy of the Router distribution, compared against `log(n_experts)` | ST-MoE [Zoph et al. 2022, §7.2, Table 14] |
| **Entropy of top-k mass** | Sum of top-k Gate values per Layer, used *instead of* entropy | [Qiu et al. 2025, §5, Fig. 4] |
| **Domain selection frequency** | Proportion of a domain's Tokens for which Expert *i* is in the top-k, vs. uniform k/N | OLMoE [Muennighoff et al. 2024, §5.3, Eq. 7] |
| **Normalised expert load** | Actual load / theoretically balanced load, per domain per Layer | DeepSeek-V3 [DeepSeek-AI 2024, §4.5.3, Fig. 9] |
| **Expert specialisation Δ** | Difference in relative activation frequency between a task corpus and a generic baseline, range [−1,1] | [Bandarkar et al. 2026, §5.2, Eq. 3] |
| **Entropy-normalised JSD** | Jensen-Shannon divergence between mean-pooled Gate vectors, divided by `log E − ½(H(q₁)+H(q₂))` | [Bandarkar et al. 2026, §4.3, Eq. 2; App. A.4, Eq. 6–8] |
| **Expert co-activation** | `N(E_i, E_j) / N(E_i)` within a Layer | OLMoE [Muennighoff et al. 2024, §5.2, Eq. 6] |
| **Cross-layer routing transition** | Conditional probability of Layer *n+1* Expert given Layer *n* Expert | ExFlow [Yao et al. 2024] |
| **Expert paths** | Full per-Token sequence of Expert choices across all Layers, clustered | [Gu et al. 2026] |
| **Router saturation** | Mean over Tokens of `card(E_i^(t) ∩ E_i^(T)) / k` — overlap between an intermediate checkpoint's top-k set and the final checkpoint's | OLMoE [Muennighoff et al. 2024, §5.1, Eq. 5] |
| **Consecutive-token repetition** | P(Token *i* and *i+1* share first / first-or-second Expert), vs. 1/k and the 2-of-k collision baseline | Mixtral [Jiang et al. 2024, §5, Table 5] |
| **Token-to-token Jaccard consistency** | Jaccard similarity of activated Expert sets over sampled Token pairs within a sequence | [Bandarkar et al. 2026, §4.4] |
| **Router Hamming similarity** | Averaged pairwise overlap of binary top-k masks; the authors say it "measures expert usage similarity more directly" than cosine | [Wang, Hayou & Nalisnick 2026, App. A, Eq. 11] |
| **Sequence-level frequency cosine** | Expert visit counts normalised to a probability vector, compared by cosine | [Wang, Hayou & Nalisnick 2026, App. A, Eq. 13–14] |
| **Top-p Jaccard overlap** | Smallest Expert set covering probability mass *p*, compared by Jaccard | [Wang, Hayou & Nalisnick 2026, App. A, Eq. 18–19] |

Three notes on the metric landscape:

- **There is no consensus set.** [Falke et al. 2026] state the field suffers from
  a "lack of established metrics" and propose a small-scale testbed with a
  domain-separable data mixture and a reference Router giving an upper bound;
  they report that "balancing scope is the crucial factor". *(Abstract page only
  was read; the metric definitions in that paper were not verified.)*
- **The cosine-of-frequency-vectors metric you are already using is a published
  one** [Wang, Hayou & Nalisnick 2026, Eq. 13–14], but the same paper prefers
  Hamming/Jaccard set overlap on the grounds that it is more direct.
- **Nobody publishes a null baseline** for these distances beyond the uniform
  line. That gap is yours to fill (§6).

---

## 5. Question 4 — Pitfalls

**P1. The load-balancing loss suppresses exactly what you are measuring.**
Best-evidenced pitfall in the field. Switch's loss is "minimized under a uniform
distribution" [Fedus et al. 2021]. OLMoE names the cost: removing it is "an
important direction for future research as it constrains the flexibility of the
model by forcing it to use all experts approximately equally", and offers this as
the reason prior work "failed to find strong evidence of expert specialization"
[Muennighoff et al. 2024, §4.1.6]. Qiu et al. show scope alone moves per-domain
frequencies from "none exceed 0.15" to "exceeding 0.2" [2025, §4.2]. DeepSeek-V3
shows removing the loss yields "greater expert specialization patterns"
[DeepSeek-AI 2024, Fig. 9]. Wang et al. give the mechanism: near a minimiser the
Router satisfies `Π_e P μ ≈ 0`, so "the router effectively ignores the shared
direction and routes using only the residual variations" [2026, §4.1, Prop. 2].
**Consequence for you:** a small measured specialisation is a *lower bound* on
what the weights encode, and its magnitude is a property of the training recipe
as much as of the model's knowledge.

**P2. Slot index has no meaning across Layers.**
This is architectural, not a literature finding: each Layer has its own Router,
independently initialised, so Slot 151 of Layer 0 and Slot 151 of Layer 20 are
unrelated. Your `CONTEXT.md` already encodes this correctly. What the literature
adds is that the *relation* between Layers is nonetheless real and measurable —
but as a conditional distribution, not as index identity [Yao et al. 2024;
Gu et al. 2026]. Never sum, average or correlate across the Slot axis of
different Layers.

**P3. Top-k renormalisation destroys comparability of Gate values.**
Models differ in whether softmax is applied before or after top-k selection:
Lo et al. note explicitly that "DeepSeek/Grok apply softmax before top-k, Mixtral
after", so absolute Gate scores are not comparable across models [2024, App. E].
DeepSeek-V3 uses a sigmoid affinity, normalises across only the selected scores,
and adds its balancing bias to the *selection* score but not the Gate — "the bias
term is only used for routing" [DeepSeek-AI 2024, §2.1.2].
**Consequence for you:** your Gates sum to 1 per (Token, Layer) by construction,
so a Gate of 0.30 means "30% of this Token's Layer budget", not "the Router was
confident". Confidence, if you want it, must come from the pre-top-k logits,
which your Trace does not carry. Also note that a Gate-weighted Category profile
and a count-based Category profile are different objects and will not agree.

**P4. Token frequency confounds with topic.**
The canonical demonstration is OpenMoE: Expert 21 looks code/math-specialised
purely because it prefers `=`, `and` and `\n` [Xue et al. 2024, §4.2, Table 9].
OLMoE's vocabulary analysis is the same phenomenon read positively — punctuation,
scripts and function words dominate expert-level associations [2024, §5.4].
Any corpus-level Category profile is dominated by high-frequency tokens, so a
"coding routes differently" result may be entirely a statement about whitespace,
brackets and identifiers. **Mitigation used in the literature:** condition on
Token ID (OpenMoE), or compare *paired/parallel* text so the token distribution
is matched (Bandarkar et al. use FLoRes parallel sentences precisely for this).

**P5. Position, BOS and template effects.**
Mixtral reports "some degree of positional locality" [2024, §5]. OpenMoE finds
neighbouring positions favour the same Experts and documents
"Drop-towards-the-End", where later positions are dropped once Expert capacity
fills — with fine-tuning failing to fix it [Xue et al. 2024, §4.3]. Most
directly relevant to Decode Captures: Wang et al. attribute gpt-oss's prefill
routing collapse to "input-invariant initial decoding tokens" produced by its
reasoning chat template, and show that removing the template reduces the collapse
[2026, §5.3]. Since every Prompt in a Category shares a template and often an
opening phrase, the first few Decode Tokens may carry Category signal that is
purely formatting.

**P6. Small-sample effects on count-based profiles.**
Bandarkar et al. chose mean-pooled routing *weights* over discrete counts because
counts "yield sharper, higher-variance distributions that are more difficult to
mean-pool" [2026, §4.3]. With 8 non-zero entries out of 256 per (Token, Layer),
a Category with few Tokens produces a profile whose sampling noise is large
relative to its signal, and that noise *reduces* cosine similarity — i.e. it
biases toward a "specialisation" conclusion.

**P7. Comparing distributions of different entropy or support size.**
Bandarkar et al. normalise JSD by `log E − ½(H(q₁)+H(q₂))` because KL/JS is
"highly sensitive on entropy", noting raw plots "were overshadowed by the trends
in entropy" [2026, App. A.4]. If your Categories differ in routing entropy — and
they almost certainly do, since Bandarkar et al. Finding 3 reports entropy
falling across Layers at different rates for different conditions — then an
unnormalised distance between Category profiles partly measures entropy
difference, not profile difference.

**P8. Upcycling and architecture confound cross-model comparison.**
OLMoE [§5.3] and OpenMoE [§4.4] independently attribute Mixtral's flat domain
profile to its dense initialisation. Do not treat Mixtral's null result as a
statement about MoEs in general.

**P9. Decode Tokens are not corpus Tokens.**
See §5.1 below — this is a pitfall specific to your setup and the literature
gives it exactly one data point.

---

## 6. How this relates to the captured dataset

### 6.0 Setup notes worth stating up front

Per the official model card, Qwen3.6-35B-A3B has "Number of Layers: 40",
"Number of Experts: 256", "Number of Activated Experts: 8 Routed + 1 Shared"
[Qwen 2026, model card]. Two consequences:

- **A shared Expert exists and is not in your Trace.** DeepSeekMoE's rationale is
  that shared Experts capture "common knowledge", and disabling theirs raised
  Pile loss "from 1.808 to 2.414" [Dai et al. 2024, §4.5]. So the 8 routed Slots
  you observe are already the *residual* after common computation is handled
  elsewhere. All else equal, this should make routed-Expert profiles look **more**
  differentiated than in a model without a shared Expert (Mixtral, OLMoE, Qwen3).
  Any comparison of your effect sizes against those papers is biased upward by
  this alone.
- **The balancing regime probably favours specialisation.** Qwen3 states it
  adopts "the global-batch load balancing loss ... to encourage expert
  specialization" [Qwen Team 2025, §2], the exact intervention Qiu et al. show
  makes domain frequencies rise above 0.2. Whether Qwen3.6 inherits this is
  **[UNVERIFIED]** — the model card does not state the training loss. If it does,
  you are measuring a model *selected* for the property you are testing, and
  Mixtral is the wrong baseline for effect size.

### 6.1 Observation 1 — Category routing profiles have pairwise cosine 0.20–0.54

**Known, in kind. Novel, in degree — but not yet interpretable.**

- The *metric* is published: sequence-level Expert-frequency vectors compared by
  cosine is [Wang, Hayou & Nalisnick 2026, Eq. 13–14].
- The *finding* that task/domain conditions route measurably differently is
  established for models trained with global-batch balancing
  [Qiu et al. 2025, §4.2] and for from-scratch MoEs
  [Muennighoff et al. 2024, §5.3; DeepSeek-AI 2024, §4.5.3].
- What is **not** established is that 0.20–0.54 is *low*. Nobody publishes a null
  distribution for this statistic. Two concerns:
  1. For non-negative frequency vectors, cosine is bounded in [0,1] and two
     near-uniform 256-dimensional profiles would sit close to 1.0. A range of
     0.20–0.54 is therefore either a very large effect or an artefact of how the
     vectors are built (raw counts vs. rates, per-Layer vs. concatenated over all
     10,240 Experts, centred vs. uncentred). **State which, in writing, before
     interpreting.** Cosine on mean-centred profiles is a correlation and has an
     entirely different null.
  2. P4 and P6 both push this statistic in the direction you observed. Coding
     Prompts generate different *tokens*, and different tokens route differently
     for reasons OpenMoE showed are lexical, not topical.
- **The strongest contrary reading:** Wang, Hayou & Nalisnick argue routing
  similarity is fully explained by hidden-state similarity [2026, Prop. 1], so
  a Category-level cosine gap may be restating "coding text and prose have
  different hidden states" — true, but not evidence of expertise.

### 6.2 Observation 2 — U-shaped cross-Category agreement (0.76 at L0, 0.28 at L14, 0.58 at L39)

**Known — and this is the observation with the strongest literature support.**

[Bandarkar et al. 2026, §4.3] measure exactly this shape on four MoEs including
Qwen3-30B-A3B: for **domains** (medical, math, vs. generic), "routing divergence
from the generic domain exhibits the opposite pattern: higher divergence exists
in the middle layers (more of a ∩-shape)". Maximum divergence mid-stack is
maximum *dis*agreement mid-stack, i.e. exactly your minimum at Layer 14. Their
selected middle intervention window for Qwen3-30B-A3B is Layers 8–35 of 48
(~17–73% depth); your minimum at Layer 14 of 40 sits at 35% depth, inside it.

Three independent supports for the two ends:

- **Layer 0 high agreement:** Mixtral finds Layer 0 repetition at chance
  [2024, Table 5]; OLMoE calls Layer 0 a saturation outlier [2024, §5.1];
  Bandarkar et al. find Phi-3.5-MoE routes all languages to the same few
  first-Layer Experts [2026, §4.3]; DeepSeek-V3 makes early Layers dense
  [2024, §4.2]. Early routing is dominated by embedding-space structure, which
  is Category-agnostic.
- **Late-Layer recovery:** consistent with Wang et al.'s report that unrelated
  inputs "activate nearly identical experts in deeper layers" [2026, §4.2] and
  with Lo et al.'s finding that Mixtral Expert similarities "significantly grow
  to greater than 0.8 in the last two layers" [2024, §5.1].
- **Caution:** Bandarkar et al. also note domain patterns "are weaker, as domains
  are less different than languages", and that for *languages* the shape is
  inverted (U-shaped divergence, i.e. high disagreement at both ends). If any of
  your Prompts differ in language or script, you are mixing the two effects.

**Novel part:** doing this over *task Categories* of a single language, on 40
Layers with 256 Slots, at Decode time. Bandarkar et al.'s domain result is
stated in prose only, without a figure, on two domains. Your version is denser
and better sampled. That is a publishable increment on an existing finding, not
a new phenomenon.

### 6.3 Observation 3 — All 256 Slots used by every Category

**Known, and — importantly — statistically uninformative at your sample size.**

This is what the load-balancing loss is designed to produce
[Fedus et al. 2021: "minimized under a uniform distribution"], and OLMoE's whole
point in §5.3 is that specialisation shows up as activation *above or below*
chance, not as zero support (they still note "0%" Experts would be removable,
implying they expect near-zero cases to be rare).

More pointedly: with ~3,100 Decode Tokens per Category × 8 selections, each Layer
sees ~24,900 selections over 256 Slots — about 97 expected hits per Slot. Under
uniform random routing, P(a given Slot is never selected) ≈ (1 − 8/256)^3100
= 1.8 × 10⁻⁴³. Across all 40 Layers × 256 Slots × 5 Categories the expected
number of unused Slots is 9 × 10⁻³⁹. **Full support was guaranteed before you
looked**;
observing it rules out nothing except a truly dead Expert. Report it as a data
sanity check, not as a finding. The informative version is the *shape* of the
usage distribution — Gini, entropy, or MaxVio [Wang et al. 2024, Eq. 4] — and how
far the tails depart from k/N = 3.125%.

### 6.4 The Decode-Token caveat you asked about

**Yes, this is a real and under-studied gap, and you should flag it prominently.**

Every routing analysis reviewed here except one is a forward pass over fixed
corpus text: Mixtral on The Pile validation set [2024, §5]; OLMoE on "a random
0.5% of the C4 validation data" [2024, §5.1–5.2]; OpenMoE on RedPajama/TheStack/
TED [2024, §4]; Qiu et al. on held-out SFT sets [2025, §4.2]; DeepSeek-V3 on
Pile test [2024, §4.5.3]; Bandarkar et al. on FLoRes sequences, explicitly
"averaging routing weights over all tokens of each sequence" and not separating
prompt from generated Tokens [2026, §4.3].

The exception is [Wang, Hayou & Nalisnick 2026, §5.2–5.3], and its result is a
warning: across matched query pairs, "during the prefill phase, all three query
pairs show nearly identical expert usage" while during generation similarity
either persists or "drops sharply as the rollout lengthens", and "which regime
applies is not predictable from the prompt alone". Their conclusion —
"prefill-phase expert usage is not a reliable proxy for full generation usage
pattern" — cuts both ways: **published corpus-based results are not a reliable
proxy for your Decode-phase results either.** Your effect sizes are not directly
comparable to Mixtral's or OLMoE's numbers, and you should say so rather than
claim to have replicated or refuted them.

There is a second, subtler issue. Your Decode Tokens are the model's own output
under greedy decoding, so Category is confounded with *output* token
distribution, not input topic. A Category effect could be entirely mediated by
"coding Prompts make the model emit code tokens, and code tokens route to code
Experts" — the P4 confound, amplified. Your Capture already holds the Prefill
rows; the Prefill/Decode comparison is the cleanest available test of this.

---

## 7. What the literature suggests measuring next

Ordered by how much they change the interpretation of what you already have.

1. **A permutation null for the Category cosine.** Shuffle Category labels across
   Prompts (not Tokens — respect ADR-0001) 1,000 times and recompute the pairwise
   cosine matrix. Report the observed 0.20–0.54 against that null. Without it,
   the number is uninterpretable, because no paper publishes a baseline for it
   (§4, closing note).
2. **A per-Layer bootstrap CI on the agreement curve.** Resample Prompts within
   Category with replacement; plot the U-shape with a band. The literature's
   depth results are all point estimates, so a curve with CIs is an immediate
   contribution — and it will tell you whether 0.58 at Layer 39 is really a
   recovery or noise.
3. **Replicate Mixtral's consecutive-Token repetition, per Layer, all 40.**
   [Jiang et al. 2024, Table 5] gives you a directly comparable statistic, but
   **recompute the baselines for your geometry** — Mixtral's 12.5% and ~46.4%
   are 1/8 and 1 − C(6,2)/C(8,2) for top-2 of 8. For top-8 of 256 the
   first-choice baseline is 1/256 = 0.39% and the "share at least one Expert"
   baseline is 1 − C(248,8)/C(256,8) = 22.7%. Mixtral's
   result was on 3 Layers of a 32-Layer model over corpus text; yours would be
   40 Layers over Decode. Any deviation from the Mixtral shape (chance at L0,
   peak mid-stack) is a genuine finding.
4. **Condition on Token ID before concluding anything topical.** OpenMoE's
   Expert-21 example [2024, §4.2] is the exact failure mode. Recompute
   Category profiles (a) restricted to Tokens that occur in all five Categories,
   and (b) with each Token's contribution reweighted to a common marginal Token
   distribution. If the U-shape survives both, it is not a frequency artefact.
   This is the single highest-value robustness check available.
5. **Prefill vs Decode on the same Prompts.** Directly tests §6.4. You already
   have the rows and the Marker states the boundary (ADR-0002). If Category
   separation is much weaker in Prefill, your effect is about generated text,
   not about question type — which is still interesting but is a different
   claim.
6. **Position-controlled profiles.** Drop the first *n* Decode Tokens (try
   n = 1, 5, 20) and recompute. Wang et al.'s "input-invariant initial decoding
   tokens" collapse [2026, §5.3] and Mixtral's positional locality [2024, §5]
   both predict the first Tokens are unrepresentative. Also plot agreement as a
   function of Decode position.
7. **Report expert-usage skew, not support.** Replace observation 3 with MaxVio
   per Layer per Category [Wang et al. 2024, Eq. 4], plus the count of Slots
   above/below the uniform k/N = 3.125% line, which is OLMoE's presentation
   [2024, §5.3]. This is the informative version of "all Slots are used".
8. **Entropy-normalised JSD alongside cosine.** [Bandarkar et al. 2026, Eq. 2 and
   App. A.4]. Report routing entropy per (Layer, Category) first; if entropies
   differ, the unnormalised distances are partly measuring that (P7).
9. **Set-overlap metrics as a cross-check.** Router Hamming similarity on binary
   top-8 masks and Jaccard at top-p [Wang et al. 2026, Eq. 11, 18–19]. The
   authors argue these measure Expert usage similarity more directly than cosine;
   agreement between the two families would strengthen the result considerably.
10. **Cross-Layer transition matrices.** For adjacent Layers, estimate
    P(Slot at Layer L+1 | Slot at Layer L), per Category. ExFlow shows this is
    strongly non-uniform in trained models [Yao et al. 2024]; Gu et al. frame the
    per-Token full-depth version as an "expert path" [2026]. Nobody has published
    this *conditioned on task Category*, and your Store already contains it. This
    is the most likely source of a genuinely novel result — but beware P2 and the
    sample size: a 256×256 transition matrix from ~3,100 Tokens is 65,536 cells
    from ~25,000 observations, so aggregate (e.g. cluster Slots first, or measure
    only mutual information between adjacent Layers) rather than plotting raw.
11. **Within-Layer co-activation per Category.** OLMoE's Eq. 6 restricted to a
    Category, compared against the all-Category baseline. OLMoE found "no strong
    co-activation among experts in one layer" globally [2024, §5.2]; whether that
    holds within a Category is open.
12. **Anchor the effect size honestly.** Recompute OLMoE's §5.3 statistic —
    per-Category top-k selection frequency against the k/N uniform line — because
    it is the one number in this literature that is directly comparable across
    models. Then state the two biases from §6.0 (shared Expert absent from the
    Trace; probable global-batch balancing) when comparing to their figures.

Not worth doing yet, and why: **router saturation** [Muennighoff et al. 2024,
§5.1] needs intermediate checkpoints, which a served model does not give you;
**expert pruning/ablation** needs engine changes this repo explicitly does not
own; **cross-model comparison** needs a second Capture on a model with a
different balancing regime, which is the natural follow-up once (1)–(6) are done.

---

## 8. References

All entries below were retrieved and read by the author of this document. Where
only the abstract page was read, that is stated.

1. Jiang, Albert Q.; Sablayrolles, Alexandre; Roux, Antoine; Mensch, Arthur;
   Savary, Blanche; Bamford, Chris; Chaplot, Devendra Singh; de las Casas,
   Diego; Bou Hanna, Emma; Bressand, Florian; Lengyel, Gianna; Bour, Guillaume;
   Lample, Guillaume; Renard Lavaud, Lélio; Saulnier, Lucile; Lachaux,
   Marie-Anne; Stock, Pierre; Subramanian, Sandeep; Yang, Sophia; Antoniak,
   Szymon; Le Scao, Teven; Gervet, Théophile; Lavril, Thibaut; Wang, Thomas;
   Lacroix, Timothée; El Sayed, William. **"Mixtral of Experts."** 2024.
   arXiv:2401.04088. https://arxiv.org/abs/2401.04088
   *(Full text §5 and Table 5 read.)*

2. Muennighoff, Niklas; Soldaini, Luca; Groeneveld, Dirk; Lo, Kyle; Morrison,
   Jacob; Min, Sewon; Shi, Weijia; Walsh, Pete; Tafjord, Oyvind; Lambert,
   Nathan; Gu, Yuling; Arora, Shane; Bhagia, Akshita; Schwenk, Dustin; Wadden,
   David; Wettig, Alexander; Hui, Binyuan; Dettmers, Tim; Kiela, Douwe;
   Farhadi, Ali; Smith, Noah A.; Koh, Pang Wei; Singh, Amanpreet; Hajishirzi,
   Hannaneh. **"OLMoE: Open Mixture-of-Experts Language Models."** 2024.
   arXiv:2409.02060. https://arxiv.org/abs/2409.02060
   *(Full text §4.1, §4.3, §5.1–5.4 read.)*

3. Dai, Damai; Deng, Chengqi; Zhao, Chenggang; Xu, R.X.; Gao, Huazuo; Chen,
   Deli; Li, Jiashi; Zeng, Wangding; Yu, Xingkai; Wu, Y.; Xie, Zhenda; Li,
   Y.K.; Huang, Panpan; Luo, Fuli; Ruan, Chong; Sui, Zhifang; Liang, Wenfeng.
   **"DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts
   Language Models."** 2024. arXiv:2401.06066. https://arxiv.org/abs/2401.06066
   *(Full text §3.1, §4.4, §4.5 read.)*

4. DeepSeek-AI. **"DeepSeek-V3 Technical Report."** 2024 (v2, 2025).
   arXiv:2412.19437. https://arxiv.org/abs/2412.19437
   *(Full text §2.1.2, §4.2, §4.5.3 and Fig. 9 caption read; Appendix C body
   could not be retrieved — the HTML render truncated before it.)*

5. Xue, Fuzhao; Zheng, Zian; Fu, Yao; Ni, Jinjie; Zheng, Zangwei; Zhou,
   Wangchunshu; You, Yang. **"OpenMoE: An Early Effort on Open Mixture-of-Experts
   Language Models."** 2024. arXiv:2402.01739. https://arxiv.org/abs/2402.01739
   *(Full text §4.1–4.4 read.)*

6. Zoph, Barret; Bello, Irwan; Kumar, Sameer; Du, Nan; Huang, Yanping; Dean,
   Jeff; Shazeer, Noam; Fedus, William. **"ST-MoE: Designing Stable and
   Transferable Sparse Expert Models."** 2022. arXiv:2202.08906.
   https://arxiv.org/abs/2202.08906
   *(Full text §3.3–3.4, §7.1–7.3, Tables 13–14 read.)*

7. Fedus, William; Zoph, Barret; Shazeer, Noam. **"Switch Transformers: Scaling
   to Trillion Parameter Models with Simple and Efficient Sparsity."** 2021
   (JMLR). arXiv:2101.03961. https://arxiv.org/abs/2101.03961
   *(Full text, load-balancing loss section, Eq. 4–6 read.)*

8. Qiu, Zihan; Huang, Zeyu; Zheng, Bo; Wen, Kaiyue; Wang, Zekun; Men, Rui;
   Titov, Ivan; Liu, Dayiheng; Zhou, Jingren; Lin, Junyang. **"Demons in the
   Detail: On Implementing Load Balancing Loss for Training Specialized
   Mixture-of-Expert Models."** 2025. arXiv:2501.11873.
   https://arxiv.org/abs/2501.11873
   *(Full text §1, §2.2, §3, §4.1–4.2, §5 read.)*

9. Wang, Lean; Gao, Huazuo; Zhao, Chenggang; Sun, Xu; Dai, Damai.
   **"Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts."**
   2024. arXiv:2408.15664. https://arxiv.org/abs/2408.15664
   *(Full text §2.2, §3, §4.1 read.)*

10. Bandarkar, Lucas; Yang, Chenyuan; Fayyaz, Mohsen; Hu, Junlin; Peng, Nanyun.
    **"Multilingual Routing in Mixture-of-Experts."** 2025 (v2 2026; ICLR 2026).
    arXiv:2510.04694. https://arxiv.org/abs/2510.04694
    *(Full text §3, §4.2–4.4, §5.2, §5.4, App. A.1, A.4 read.)*

11. Wang, Xi; Hayou, Soufiane; Nalisnick, Eric. **"The Myth of Expert
    Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily
    Domain Expertise."** 2026. arXiv:2604.09780.
    https://arxiv.org/abs/2604.09780
    *(Full text §1, §4.1–4.2, §5.1–5.3, §6, App. A–B read. Preprint, v1, no
    stated venue — treat as unreplicated.)*

12. Lo, Ka Man; Huang, Zeyu; Qiu, Zihan; Wang, Zili; Fu, Jie. **"A Closer Look
    into Mixture-of-Experts in Large Language Models."** 2024.
    arXiv:2406.18219. https://arxiv.org/abs/2406.18219
    *(Full text §4.1–4.3, §5.1–5.4, §6–9, App. C–E read.)*

13. Yao, Jinghan; Anthony, Quentin; Shafi, Aamir; Subramoni, Hari; Panda,
    Dhabaleswar K. **"Exploiting Inter-Layer Expert Affinity for Accelerating
    Mixture-of-Experts Model Inference."** 2024. arXiv:2401.08383.
    https://arxiv.org/abs/2401.08383
    *(Abstract page read; full text not read. Cited only for the existence and
    direction of the inter-layer conditional-probability finding.)*

14. Gu, Zijin; Likhomanenko, Tatiana; Thilak, Vimal; Ramapuram, Jason; Jaitly,
    Navdeep. **"Path-Constrained Mixture-of-Experts."** 2026. arXiv:2603.18297.
    https://arxiv.org/abs/2603.18297
    *(Abstract page read; full text not read. Preprint, under review. Cited only
    for the "expert paths" framing and the claim that tokens cluster into a
    small fraction of paths.)*

15. Falke, Tobias; Anastassacos, Nicolas; Tan, Samson; Meas, Chankrisna Richy;
    Prakash, Chandana Satya; Sekhar, Nitesh; Bari, M Saiful; Kompella, Krishna;
    Elsayed, Gamaleldin F. **"MoE Routing Testbed: Studying Expert
    Specialization and Routing Behavior at Small Scale."** 2026.
    arXiv:2604.07030. https://arxiv.org/abs/2604.07030
    *(Abstract page read; full text not read. Preprint. Cited only for the
    "lack of established metrics" observation and the "balancing scope is the
    crucial factor" conclusion.)*

16. Chi, Zewen; Dong, Li; Huang, Shaohan; Dai, Damai; Ma, Shuming; Patra,
    Barun; Singhal, Saksham; Bajaj, Payal; Song, Xia; Mao, Xian-Ling; Huang,
    Heyan; Wei, Furu. **"On the Representation Collapse of Sparse Mixture of
    Experts."** 2022 (NeurIPS 2022). arXiv:2204.09179.
    https://arxiv.org/abs/2204.09179
    *(Abstract page read; full text not read. Background only — routing training
    "encourages token clustering around expert centroids".)*

17. Shen, Yikang; Guo, Zhen; Cai, Tianle; Qin, Zengyi. **"JetMoE: Reaching
    Llama2 Performance with 0.1M Dollars."** 2024. arXiv:2404.07413.
    https://arxiv.org/abs/2404.07413
    *(Full text checked specifically for routing analysis: none present.)*

18. Qwen Team. **"Qwen3 Technical Report."** 2025. arXiv:2505.09388.
    https://arxiv.org/abs/2505.09388
    *(Full text §2 Architecture read.)*

19. Qwen. **Qwen3.6-35B-A3B model card.** Hugging Face, 2026.
    https://huggingface.co/Qwen/Qwen3.6-35B-A3B
    *(First-party model card; Model Overview section read. Note the card's
    sidebar reports "36B params" against the overview's 35B.)*

### Sources deliberately not cited as evidence

- **Snowflake Arctic.** No arXiv technical report exists; the architecture is
  described only in Snowflake's own blog and "cookbook" posts. Because the task
  brief asked about Arctic, this is recorded as **[UNVERIFIED]**: the widely
  repeated "128 experts, top-2, dense-MoE hybrid" description traces to a
  first-party blog post, not a paper, and no first-party routing analysis of
  Arctic was found.
- **Qwen1.5-MoE / Qwen2 MoE routing analyses.** None found; Qwen3's §2 is the
  first Qwen-family statement on specialisation and it is a design choice, not a
  measurement.
- Any Medium, blog or survey summary of the above. Several were surfaced during
  search and none are cited.
