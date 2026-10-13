# Model card: EditGuard auxiliary models

**Status:** Both pilot-v2 auxiliary models completed user-executed GPU training
and run in the learned comparison pipeline. The scope run completed 360 updates
and selected step 120; the answer run completed and selected step 240. Exported
models, tokenizers and full-state recovery checkpoints are present locally and
pass recorded integrity checks. Model binaries are ignored local artifacts;
publishing metadata and hashes does not distribute deployable weights.

Training and validation use explicitly automated development review,
`human_reviewed_data=false`. They are not independent human validation or final
test results. Developer smoke/profile checkpoints remain ineligible. Structured
answers are the application's default, separately labeled from generation.
[Training records](PILOT_TRAINING_RESULTS.md) and
[validation diagnostics](PILOT_VALIDATION_RESULTS.md) provide full evidence.

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
The trained artifacts are `artifacts/scope/pilot-v2-initial` and
`artifacts/answer/pilot-v2-initial`. Exact revisions, tensor/file hashes,
training environments and selected states are in the
[87-check artifact audit](../reports/pilot-v2-training-audit.json).
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

The artifacts record source/config/data/review/environment identities. Full-state
checkpoints retain model, optimizer, scheduler, RNG, deterministic sampler, global
step and validation selection. Resume rejects changed identities. Neither initial
run records a resume event; the separate recovery regressions provide that evidence.

## Initial development evaluation

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

The first [bounded scope-data revision](SCOPE_DATA_REVISION.md) balances existing
negative question forms in pilot-v3 without changing facts, example counts,
families or splits. It has fresh review identities and establishes no new model
result. Further training requires independently reviewed inputs and its own
finite, profiled run card. The answer model's 12 `No` → `Yes` errors remain a
separate generation problem even with oracle-correct routing.

The [isolated Nimbus learned-workflow check](LEARNED_WORKFLOW.md) also found an
out-of-scope false application outside the two-family validation slice: the
device correction was selected for a price question, generating
`Nimbus Basic version v2 costs 3 devices.` A four-device comparison also endorsed
yes where no was required. Passing governance/integration checks does not count
as answer accuracy. The pilot's 0/72 out-of-scope edit selections therefore
cannot establish reliable locality on other products or wording.

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
