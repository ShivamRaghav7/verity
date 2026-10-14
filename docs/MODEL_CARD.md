# Model card: EditGuard auxiliary models

**Status:** Both auxiliary models completed user-executed GPU training and run
in the learned comparison pipeline. The current scope artifact is the completed
pilot-v3 revision: 360 updates, selected step 360. The answer artifact remains the
initial pilot-v2 model: completed and selected step 240. One bounded answer
polarity revision and the second, final scope revision are prepared with
source-bound profiles and preflight-valid commands;
neither has trained. Exported
models, tokenizers and full-state recovery checkpoints are present locally and
pass recorded integrity checks. Model binaries are ignored local artifacts;
publishing metadata and hashes does not distribute deployable weights.

Training and validation use explicitly automated development review,
`human_reviewed_data=false`. They are not independent human validation or final
test results. Developer smoke/profile checkpoints remain ineligible. Structured
answers are the application's default, separately labeled from generation.
[Initial training records](PILOT_TRAINING_RESULTS.md) and
[current pilot-v3 results](PILOT_V3_RESULTS.md) provide full evidence.

## Model roles and pinned provenance

The scope model classifies whether a correction applies to a short product-support
question. A fixed compact embedding model retrieves candidates; a trained compact
cross-encoder scores question–correction pairs. The answer model generates a short
response from an applicable correction, question and necessary unchanged context.
Both auxiliaries train across edits; new approved corrections later update memory
without per-edit retraining. The separately frozen base retrieves the initial
fictional handbook. Validation recorded an unchanged base tensor fingerprint.

| Role | Backbone | Primary source and recorded role |
|---|---|---|
| Embedding baseline | `sentence-transformers/all-MiniLM-L6-v2` | [Official model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; masked mean pooling and L2 normalization. |
| Pair classifier | `distilbert/distilbert-base-uncased` | [Official model card](https://huggingface.co/distilbert/distilbert-base-uncased). Revision `12040accade4e8a0f71eabdb258fecc2e7e948be`; trained binary classifier. |
| Conditional answer and separate frozen base | `google/flan-t5-small` | [Official model card](https://huggingface.co/google/flan-t5-small). Revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`; distinct frozen and fine-tuned states. |

Initial profiles and separate synthetic
[checkpoint/resume verification](TRAINER_SMOKE.md) preceded the real runs.
The current trained artifacts are `artifacts/scope/pilot-v3-revision1` and
`artifacts/answer/pilot-v2-initial`. The archived scope-v2 export remains available
for common-input comparisons. Exact revisions, tensor/file hashes, environments
and selected states are in the [initial artifact audit](../reports/pilot-v2-training-audit.json)
and [529-check scope revision audit](../reports/pilot-v3-training-audit.json).
The exact historical trainer sources are preserved from Git `350db38` in
[`reports/training-source-v1`](../reports/training-source-v1/manifest.json).
Existing pretraining can introduce bias or unsupported generations; the bounded
task and evaluation do not remove that risk.

## Training data and optimization

The controlled fictional pilot contains 240 scope/96 answer training examples
and 120 scope/48 answer validation examples across four and two disjoint entity
families. All 504 labels used a separate user-authorized automated development
registry; immutable source review flags remain pending independent human review.
Scope examples include positives, ordinary negatives and five difficult-negative
categories. Answer training uses applicable corrections and short factual targets.
Counterfactual values remain in the same family partition. Pending or stale review
records cannot authorize training. No test/challenge rows enter fitting, early
stopping, calibration or model selection. See [the dataset card](DATASET_CARD.md).

Both runs use seed 42, float32, batch size 2, AdamW at initial learning rate
0.00002, weight decay 0.01, clipping at 1.0 and a fixed linear schedule. Inputs
have a 192-token budget; answer targets have 48 tokens. Accepted rows fit those
budgets without truncation. Scope uses two-class cross-entropy; the answer model
uses conditional token negative log-likelihood. Teacher-forced answer loss is
not factual accuracy; autoregressive outputs can still endorse incorrect answers.
Pilot-v3 balances the same negative scope questions across the four forms without
changing catalogue facts, labels, partitions or example count. It has fresh
typed automated review. Positive questions and all answer inputs/targets remain
unchanged; 54 of 72 validation negative questions change wording.

The artifacts record source/config/data/review/environment identities. Full-state
checkpoints retain model, optimizer, scheduler, RNG, deterministic sampler, global
step and validation selection. Resume rejects changed identities. Neither initial
run records a resume event; the separate recovery regressions provide that evidence.

## Archived initial pilot-v2 evaluation

| Record | Measured evidence |
|---|---|
| Scope training | Completed 360 updates, selected step 120; loop 89.448 s, peak allocated CUDA 1.318464 GiB |
| Answer training | Completed/selected 240 updates; loop 245.856 s, peak allocated CUDA 1.508413 GiB |
| Scope operating point | Validation-selected temperature 1.5, threshold 0.11 and ambiguity margin 0 |
| Matched comparisons | 120 questions per configuration, 600 model-generated records; separate 48-question oracle diagnostic |
| Raw calibrated scope pairs | Precision 36/36, recall 36/48, false positives 0/72; all 12 misses use the direct question form |
| Oracle answer credit | 31/48, with 12 wrong comparison endorsements and five pending semantic reviews |
| Inference resources | Recorded serial CLI p50/p95 and sampled memory; no full HTTP/UI latency or randomized-order comparison claim |
| Frozen final test | Not evaluated; independent final review and configuration/scorer/test freezing remain pending |

Training-loop durations exclude model loading, tokenization and final export;
complete command wall time is unavailable. Scope calibration and threshold
selection use the reported validation population and are descriptive rather
than independent calibration measurements. A zero selected margin does not
establish safe ambiguity handling. The detailed result documents disclose full
denominators, latency limits, calibration curves and raw failure examples.

The primary full-assertion scorer credits the trained scope editor on 19/120
questions and leaves incomplete baseline fragments pending. This lower bound
does not establish baseline factual failure or learned superiority. A separate
[post-hoc request-bound QA diagnostic](SHORT_QA.md) credits updated RAG on
110/120 questions and the trained editor on 91/120; updated RAG leads that
exploratory comparison. The five configurations share approved memory and
compatible answer budgets. Structured outputs receive no generation credit.
There is no frozen-test or statistical superiority claim.

## Current pilot-v3 scope and answer evaluation

The first [bounded scope revision](SCOPE_DATA_REVISION.md) completed all 360
user-executed updates and selected the final checkpoint. Its training loop took
43.402 seconds, with peak allocated/reserved GPU memory of 1.318464/1.410156 GiB
and maximum sampled process RSS of 2.028103 GiB. Full process wall time is
unavailable. An independent frozen-model check reproduces validation NLL
0.485679 within approximately `9.2e-9` of the training record. Loss is not an
error percentage.

On the **same 120 pilot-v3 validation pairs**, the original scope model obtains
NLL 1.477381 and raw accuracy 56/120 at threshold 0.5. The revision obtains NLL
0.485679, AUROC 0.863715 and raw accuracy 98/120, with recall 46/48, precision
46/66 and 20/72 false positives. All 12 direct positives are detected. The
remaining raw false positives are ten wrong-entity and ten wrong-version pairs;
raw entity/version discrimination remains weak. The original v2 selected NLL
0.390796 used different negative wording and is not a common-input comparison.

Runtime compatibility guards reject incompatible edits before classifier routing.
The new calibration selector therefore includes those established guards in its
operating-point objective; the earlier raw-pair selector optimized a different
decision from the runtime. It uses the same finite grid, costs and frozen
validation scores without changing weights. The current policy has temperature
0.5, threshold 0.47 and ambiguity margin 0.025. It applies 46/48 positive edits,
falsely applies 0/72 negatives and clarifies five of 120 pairs, giving 115/120
coverage. These are **validation-selected classifier-plus-guard results**.
They do not supersede the raw metrics or establish generated-answer accuracy.

The five-way run records 600 predictions, including five deterministic
clarifications, and 48 separate oracle answers. Strict complete-assertion credit
is 30/120 for the trained editor and 31/120 for similarity; the three baseline
generators have 120 pending/uncredited fragments each. Zero strict baseline
credit is not zero factual correctness. The unchanged exploratory request-bound
short-QA rules credit updated RAG on 91/120 and the trained editor on 80/120.
Updated RAG leads that diagnostic. All abstentions remain uncredited in both
denominators, and structured facts never count as learned generation.

Oracle answer credit remains 31/48, with five pending paraphrases and all 12
expected-`No` comparisons incorrectly answered `Yes`. A teacher-forced diagnostic
reproduces the original mean token loss near 0.044412, but finds the first `No`
token correct on only 2/24 training and 0/12 validation questions. Low average
token loss can obscure a wrong initial decision followed by easily copied facts.
[Answer-objective evidence](ANSWER_OBJECTIVE.md) explains this hypothesis and its
limits; it does not prove a single cause.

The subsequent isolated workflow shows that successful two-family validation
routing does not establish reliable transfer to Nimbus. Stage E has prepared one
opt-in `comparison_first_token_8x`
experiment, weighting the first token equally for reviewed `Yes` and `No`
comparisons while retaining the same full targets, facts and partitions. The
user-run cap is 240 updates, five passes through the 96-row answer training set.
It has not trained, and improvement is not promised. The
[released run guide](ANSWER_REVISION_TRAINING.md) supplies the guarded profile,
exact inputs, artifacts, stop condition and exact-state recovery. The answer
question tests generated comparison polarity while preserving factual answers;
its conditional planning budgets are 532 seconds, 2.25 GiB GPU and 6 GiB storage.
Weighted selection loss must not be compared directly with the original uniform
loss or presented as accuracy.

The archived [initial Nimbus learned-workflow check](LEARNED_WORKFLOW.md) also found an
out-of-scope false application outside the two-family validation slice: the
device correction was selected for a price question, generating
`Nimbus Basic version v2 costs 3 devices.` A four-device comparison also endorsed
yes where no was required. Passing governance/integration checks does not count
as answer accuracy. The pilot's 0/72 out-of-scope edit selections therefore
cannot establish reliable locality on other products or wording. These examples
use the initial scope artifact; they are not a newly measured v3 workflow result.

## Subsequent unfamiliar-product workflow and final scope experiment

The [actual scope-v3 Nimbus workflow](../reports/pilot-v3-learned-workflow.json)
passes **31/31 governance/integration checks**, including approval, conflicts and
rollback, but fails to transfer the approved device correction from `5` to `3`.
The corrected direct question is routed to the frozen base and returns stale
`5`; a device paraphrase returns `9 USD per month`; both capacity comparisons
clarify. In separate oracle probes, the unchanged answer checkpoint produces
the correct direct `3` but still incorrectly starts the negative comparison
with `Yes`. Routing and generation failures are distinct. The handwritten
fixture includes repeated questions and is not a representative accuracy study.

This measured failure motivates **scope revision 2, the final scope experiment**.
The one training adaptation is the versioned `entity_neutral_v1` preprocessing
policy. It canonicalizes correction-referenced product names/aliases, plan and
version selectors to consistent phrases and normalizes the correction descriptor.
Relation, quantities and units remain inputs; unmatched selector text remains
visible. Original correction metadata continues to enforce compatibility,
approval, effective dates and provenance. This is deterministic preprocessing,
not evidence that the classifier learned identity/date verification.
The policy applies across edits using their supplied correction descriptors;
it consumes neither gold labels nor expected facts and adds no Nimbus training
examples.

The same pilot-v3 training/validation rows and typed review registry, DistilBERT
backbone, seed 42, learning rate and 360-step cap are retained. The separately
prepared answer revision uses the same original v2 inputs and 240-step cap.
Neither new model has trained. Their guarded profiles and released cards passed
preflight. Scope's conditional budgets are 1,331 seconds, 2.0 GiB GPU and 5 GiB
storage; answer's are given above. These four-update planning estimates are not
guarantees or full-run measurements. The [released guide](ANSWER_REVISION_TRAINING.md)
requires one run at a time and the unchanged card for exact-state recovery from
the latest completed checkpoint. Completed exports and capped checkpoints are
refused before ML initialization. No measured improvement is claimed.
After these finite runs, evaluate component quality and the same
workflow, select or reject artifacts from development evidence and freeze the
final protocol. Final frozen evaluation and release polish remain pending.
No third scope experiment or added catalogue scale is planned.

## Intended use and limitations

Use within the documented fictional English support domain and selected scopes.
The small two-family validation population shares templates with training and
does not establish unfamiliar-wording generalization. Family-cluster uncertainty
from only two clusters is weak evidence for population comparisons. No guarantee
extends to arbitrary questions, new relation types, unknown products, adversarial
phrasing, multi-hop reasoning or production support dialogue.

Deterministic compatibility and conflict rules remain separate safeguards.
Generated prose can endorse an incorrect value or subject despite a correct
retrieved source. The UI distinguishes `model_generated`, structured facts and
clarification, displays provenance and training review policy, and exposes a
read-only validation dashboard. Sources and training provenance do not verify
generated prose. Numerical resume claims must follow the corresponding reviewed,
frozen measurement; this pilot establishes training and diagnosis rather than
final-project completion.
