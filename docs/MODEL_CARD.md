# Model card: EditGuard auxiliary models

Both auxiliary models completed user-executed GPU training and run in the learned
application and comparison pipeline. The selected exports are
`artifacts/scope/pilot-v3-neutral2` and `artifacts/answer/pilot-v2-polarity1`.
Model binaries, tokenizers and full-state recovery checkpoints remain local,
ignored artifacts; publishing their metadata does not distribute deployable
weights. The frozen five-configuration evaluation is complete and sealed. The
original oracle export contains 319 CUDA runtime errors and is an incomplete
generation measurement. A separately named fixed-configuration supplement now
recovers all 319 selected errors after GPU availability was restored, preserving
the original results. The development and final results below remain distinct.

Training uses user-authorized `automated_development` review. The user also
authorized `automated_final` label audit for the final packet. All records declare
`human_reviewed_data=false`. No independent human semantic review is claimed.
The application's structured answers are separately labeled and never counted
as trained generator successes.

## Roles and pinned pretrained provenance

The scope model classifies question–correction applicability. A frozen embedding
model retrieves candidates, deterministic compatibility controls eligibility, and
the trained cross-encoder scores eligible pairs. The trained answer model generates
short text from an applicable correction, question and unchanged supporting
context. The distinct frozen base retrieves the initial fictional handbook.
New approved corrections update memory without per-edit retraining.

| Role | Backbone and exact pretrained revision | Official source |
|---|---|---|
| Frozen embeddings | `sentence-transformers/all-MiniLM-L6-v2`, `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | [Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2); masked mean pooling, L2 normalization |
| Trained pair classifier | `distilbert/distilbert-base-uncased`, `12040accade4e8a0f71eabdb258fecc2e7e948be` | [Model card](https://huggingface.co/distilbert/distilbert-base-uncased); binary classification head |
| Trained answer and distinct frozen base | `google/flan-t5-small`, `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` | [Model card](https://huggingface.co/google/flan-t5-small); encoder-decoder generation |

Loading DistilBERT pretraining weights into a classifier initially reports unused
language-model-head keys and newly initialized classifier keys. That expected
task-head change does not imply that the trained export lacks a classifier. The
selected trained export's tensors and checkpoints pass integrity checks. See
[the official architecture documentation](https://huggingface.co/docs/transformers/en/model_doc/distilbert).

## Selected exports

| Evidence | Scope | Answer |
|---|---|---|
| Local artifact | `artifacts/scope/pilot-v3-neutral2` | `artifacts/answer/pilot-v2-polarity1` |
| Updates completed / selected | 360 / 360 | 240 / 240 |
| Selected validation objective | Pair cross-entropy: 0.043264720 | Weighted token NLL: 0.168240236 |
| Manifest SHA-256 | `c85e77193ff3cf5137baf0a569d142585cfac3a1734393fadf49ce95f9b88459` | `8e435ac9bcd5486bb967c7ca09c38bf1786b810c97a7c95a365c6250ceeb0a72` |
| Recorded training loop | 112.546 s | 264.537 s |
| Peak allocated / reserved GPU | 1.318 / 1.410 GiB | 1.504 / 1.656 GiB |

Measurements are from the inspected RTX 4050 laptop, PyTorch 2.9.0+cu130 and
Transformers 5.16.1 environment. Loop time excludes complete startup and final
export wall time; no full-command duration was recorded. Validation loss is not
an error percentage, and weighted answer loss is not numerically comparable to
the original uniform-loss selection result.

The independent [targeted training audit](../reports/targeted-training-audit.json)
passes 986 checks, including selected export/checkpoint tensor equality,
optimizer updates, retained state, source/data/review identities and finite caps.
[Scope](../reports/targeted-training/scope/manifest.json) and
[answer](../reports/targeted-training/answer/manifest.json) manifests preserve exact
resources, environments and hashes. Historical initial runs and scope revision 1
remain documented in [PILOT_TRAINING_RESULTS.md](PILOT_TRAINING_RESULTS.md) and
[PILOT_V3_RESULTS.md](PILOT_V3_RESULTS.md).

## Data and optimization

Four entity families supplied 240 scope pairs and 96 answer examples for training;
two disjoint families supplied 120 pairs and 48 answers for validation. The
504-row pilot has typed, exact-row-hash-bound automated review. Counterfactual
values and paraphrases stay with their family. The 80-family catalogue is the
source draft; its entire training partition was not used for these models.
No final questions entered optimization, checkpoint selection or calibration.

Both runs use seed 42, float32, batch size 2, AdamW learning rate 0.00002,
weight decay 0.01, gradient clipping 1.0 and a fixed linear schedule. Training
inputs have a 192-token budget and answer targets 48 tokens. Accepted pilot rows
fit those budgets. The comparison pipeline uses compatible 384-token context
budgets and 48 new answer tokens, greedy deterministic generation and one beam.

Scope revision 2 uses `entity_neutral_v1`: matching entity/alias, plan and version
literals are normalized in the classifier input. Wrong selectors, relation words
and quantities remain distinguishable. Opaque source/date strings are excluded
from that classifier descriptor; metadata remains available to deterministic
eligibility rules. This is a disclosed adaptation, not learned entity resolution.
Eight same-label abstracted pairs overlap training and validation; no conflicting
labels overlap. Results for those eight and the other 112 are reported separately.

Answer revision 1 uses `comparison_first_token_8x`: both Yes and No first tokens
in accepted comparison targets receive weight 8; other valid tokens receive 1,
padding 0. The loss divides summed weighted NLL by summed weights. This changes
optimization, without changing catalogue facts, target answers or labels.

Full-state checkpoints preserve model, optimizer, scheduler, RNG, sampler, global
step and selection state. Exact-state resume rejects changed source, configuration,
data and review identities. Completed/capped runs refuse further training before
ML initialization. Source snapshots are preserved in
[`reports/training-source-v2`](../reports/training-source-v2/manifest.json);
earlier sources remain in `reports/training-source-v1`.

## Development evidence used for selection

| Component | Same-population development result | Meaning and limit |
|---|---|---|
| Raw scope at 0.5 | 119/120; TP 48, FP 1, TN 71, FN 0 | Previous revision: 98/120 on these same raw pairs. This is validation, not final test accuracy. |
| Raw scope without eight shared abstracted inputs | 111/112 | Shared eight: 8/8. Removing them does not remove template familiarity. |
| Selected scope operating point | Temperature 1.0, threshold 0.97, margin 0.025; 48 TP, 0 FP, 71 TN, one negative clarification | Coverage 119/120. Selection on this population makes this an optimistic development estimate. |
| Oracle free-running answer generation | 34/48 complete assertions; 9 incorrect and 5 uncredited pending cases | 12/12 direct, 6/12 paraphrases, 16/24 comparisons. Routing is correct by construction. |
| Negative comparison generation | 5/12 correct No endorsements | Seven still fail; quantity copying alone is insufficient. |
| Matched trained editor | 34/120 complete assertions, 76 pending, 9 incorrect, one clarification | Primary scorer requires complete subject/relation/scope endorsement; pending remains in the denominator. |
| Separate request-bound QA diagnostic | Updated RAG 91/120; trained editor 86/120 | Exploratory development metric; updated RAG leads. It credits supported short responses without proving a complete scoped assertion. |

Primary development results use `editguard-short-facts-v3`. Before final freeze,
review added the whole-assertion storage alias “supports N GB of storage.” The
rule still requires correct identity, relation, unit, value and comparison
polarity; negated/examples/rejected possibilities do not qualify. All compared
outputs were rescored under the same version in fresh derived records; original
outputs and earlier scores remain archived. This scoring repair was made from
development evidence, before final question access.

See [selected development summary](../reports/targeted-selected-development-evaluation.json),
[complete-assertion metrics](../reports/targeted-candidate-v3-scored/metrics.json),
[request-bound diagnostic](../reports/targeted-candidate-v3-short-qa/summary.json),
and [component figures](../reports/targeted-figures/README.md).

## Frozen final measurements

The five matched configurations complete 4,080 observations on 720 standard
questions and 96 challenges, with 4,009 answers, 71 clarifications and zero
runtime errors. On the 720 standard questions, the frozen request-bound metric
credits updated RAG on 649/720 (90.14%), similarity on 632/720 (87.78%), the
trained scope editor on 613/720 (85.14%), prompt-only on 533/720 and the unchanged
base on 500/720. The editor has 23 clarifications and 33 additional pending
responses, all retained in its denominator. Updated RAG leads this controlled
short-answer comparison; no overall learned-superiority claim is supported.

The editor's applicable direct, paraphrase and comparison results are 72/72,
39/72 and 97/144. Its out-of-scope correctness is 405/432, with retention 405/428
among baseline-QA-correct cases. Zero observed false applications out of 432
out-of-scope cases coexists with 23 clarifications, reducing coverage to 697/720.
The paired standard-family interval for editor minus updated RAG is -5.00
percentage points, 95% interval [-6.67, -3.06], using the frozen 2,000-resample
family bootstrap. Twelve shared-template families limit external validity.

Raw final scope at threshold 0.97 has TP 288, FP 19, TN 413 and FN 0: 701/720
accuracy, precision 288/307, recall 288/288, Brier 0.03195 and 10-bin ECE 0.03218.
The nine abstracted training-overlap inputs are 9/9 correct; the remaining 711
are 692/711. These raw metrics differ from guarded runtime application and
generated-answer quality. Pre-eligibility retrieval reaches 283/288 at rank 1
and 288/288 at rank 3 in the bounded alternate-world memories.

Primary full-assertion scoring gives the editor 207/720 credits, 443 pending,
47 wrong generated responses and 23 clarifications. All 720 base/RAG/prompt
responses remain pending under that more restrictive grammar. Their zero primary
credit is not zero factual accuracy; the separate request-bound measure avoids
presenting this output-format mismatch as evidence of learned superiority.

On 96 challenges, the editor answers 48 and clarifies 48, with zero primary
credits and 48 pending outputs. All challenge wording is unsupported by the
frozen request-bound parser, so it cannot establish semantic accuracy there.
Raw scope recall falls to 72/96, and Brier/ECE worsen to 0.24834/0.25476.
This exposes genuine scope/coverage generalization limits even though unsupported
generated phrases cannot all be classified as factually wrong.

The original oracle measurement is 45/384 primary credits, with 65 answered,
319 CUDA runtime errors, eight pending and 12 incorrect generated responses.
Errors remain uncredited in the full denominator, not relabeled as model
wrongness or excluded to report 45/65 capability. A separate fixed-configuration
recovery first stopped during model loading with no CUDA GPU available and zero
new generations. After GPU availability was restored, its healthy resumed
session answered all 319 selected cases with no new runtime errors. The combined
384-case oracle view retains 65 original answers: 247 primary credits, 90 pending
and 47 incorrect generated answers. Standard primary credit is 207/288 and
request-bound credit is 208/288; challenge primary credit is 40/96 with 56 pending.
All 96 challenge forms remain unsupported by the secondary parser, so secondary
zero credit does not establish zero factual accuracy. The same frozen weights,
inputs and settings are used; this is operational recovery, not an ML improvement.
See the [separate completed manifest](../reports/final/oracle-runtime-recovery/manifest.json).
The healthy recovery's before/after base fingerprints match. The
original failed GPU instance's final fingerprint is unavailable;
fresh closure verifies unchanged files and fresh-process weights only. It does
not prove that lost original in-memory state. Read the
[runtime failure disclosure](../reports/final/runtime-failure-disclosure.json).

Standard trained-editor serial request latency is p50 241.78 ms and p95 906.22 ms
including routing/generation and clarifications, excluding model loading,
scoring, IO and HTTP/browser overhead. Fixed configuration order, shared caches
and resumed sessions limit timing comparisons. Complete tables, primary and
secondary denominators, paired uncertainty, challenge failures and execution
incidents are in [FINAL_EVALUATION.md](FINAL_EVALUATION.md).

The separate post-restoration HTTP engineering workload answers 30/30 measured
requests with HTTP 200 and zero errors/clarifications, at warm client p50
230.42 ms and p95 372.04 ms. The cold lazy-loading request is 68,781.35 ms, separate
from three excluded warmups and the 30 warm samples. It repeats three development
questions and includes API before-answer generation, transport and JSON decoding.
The four-device comparison still produces an incorrect Yes; `ml_accuracy=null`.
This proves finite workload availability/timing, not 30 correct answers or a
held-out accuracy result. Fixture state and model/configuration bindings remain
unchanged; the live database is untouched. See [HTTP evidence](../reports/final/neural-http.json).

## Intended use and limitations

Use for English short factual support questions in the controlled fictional
domain. These models do not establish arbitrary dialogue, novel relation types,
multi-hop reasoning, multilingual support or production readiness. Training and
validation are small and template-correlated; automated consistency review can
miss natural-language ambiguity. Pretraining may introduce unsupported outputs.

Nine of 720 final scope inputs abstract to same-label training pairs; 711 do not,
and none matches validation. Raw entities and lineages remain disjoint. Final
challenge questions are agent-authored, machine-audited drafts, with no human
authorship or human-review claim. Final results and uncertainty must preserve
these distinctions; see [FINAL_EVALUATION.md](FINAL_EVALUATION.md).

The system is **SERAC-inspired**. Deterministic governance and source provenance
do not certify generated prose. A well-routed correction can still yield a wrong
answer, and updated RAG can outperform the editor. Numerical resume claims must
identify the actual population, metric and review policy. Reporting distinguishes
the complete matched comparison from the original incomplete oracle measurement
and any recovery supplement. No final result was used to revise models,
thresholds or scorers. Release completion depends on reproducible evidence and
documentation, rather than a promised accuracy or superiority number.
