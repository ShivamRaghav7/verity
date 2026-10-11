# Model card: EditGuard auxiliary models

**Status:** Design and pipeline scaffolding. No EditGuard-trained scope or answer
checkpoint is present at initial delivery. No accuracy, training time or memory
result is claimed here. The current application answers deterministically.

## Intended model roles

The scope model classifies whether a correction applies to a short product-support
question. Candidate retrieval uses a fixed compact embedding baseline; a compact
cross-encoder is trained on question–correction pairs if real-checkpoint profiling
shows feasibility. The answer model generates a short response from the question,
an applicable correction and necessary unchanged context. Both auxiliaries train
across edits; adding a new approved edit later updates memory only.

The frozen base assistant retrieves the initial fictional support handbook. Its
weights and initial fact snapshot do not change during editing. The final model
registry must identify this base separately from the fine-tuned answer model.

## Candidate provenance

| Role | Candidate | Primary source and selection status |
|---|---|---|
| Embedding baseline | `sentence-transformers/all-MiniLM-L6-v2` | [Official model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Profile inference and truncation; revision not selected. |
| Pair classifier encoder | `distilbert/distilbert-base-uncased` | [Official model card](https://huggingface.co/distilbert/distilbert-base-uncased). Binary head trained locally; revision not selected. |
| Conditional answer and frozen generator candidate | `google/flan-t5-small` | [Official model card](https://huggingface.co/google/flan-t5-small). Separate frozen and fine-tuned artifacts; revision not selected. |
| Conditional generation alternative | `google-t5/t5-small` | [Official T5 documentation](https://huggingface.co/docs/transformers/model_doc/t5). Evaluate only if the initial candidate has a diagnosed feasibility issue. |

These model IDs are candidates, not installed or trained model claims. Capture
resolved revision, download checksum where available, license and tokenizer
version at first real-checkpoint use. Existing pretraining may introduce bias or
unsupported generations; the bounded task and evaluation do not remove that risk.

## Training data and supervision

Use accepted reviewed rows from [DATASET_CARD.md](DATASET_CARD.md). Scope training
uses positive, ordinary negative and five difficult-negative categories. Answer
training uses only applicable corrections and target short answers. Alternate
counterfactual values remain in the same family partition. Exclude pending labels.
No test examples or challenge wording enter fitting, calibration, early stopping
or threshold/model selection.

The initial scope objective is binary cross-entropy. The initial answer objective
is conditional token negative log-likelihood. Calibrate probabilities and select
scope threshold and ambiguity margin on validation only. Validate unseen edits and
entities; evaluate raw classifier quality as well as deployed compatibility gates.

## Required training record

Before a substantial user-executed run, provide a finite run card with measured
profile and estimated bounds. Store resolved model revision; tokenizer/max lengths;
seed; batch and effective batch; precision; optimizer/scheduler; learning rate;
step/epoch cap; checkpoint and early stopping rules; data/review/config hashes;
device/runtime versions; and actual wall time, peak CUDA allocation and host memory.

Checkpoint weights, optimizer/scheduler/scaler, RNG state and global step. Resume
must validate data/config/model identities. Select by validation results and save
both the selected checkpoint and selection rule. File artifacts, not a notebook
screenshot, are the reproducible deliverable.

## Evaluation record: currently unmeasured

| Record | Status |
|---|---|
| Trained scope checkpoint and training resource measurements | Not available |
| Trained answer checkpoint and training resource measurements | Not available |
| Calibration parameters, threshold and margin | Not selected |
| Frozen test and scorer/config/model hashes | Not frozen |
| Five-configuration component and end-to-end comparison | Not performed |
| Scope precision/recall/FAR, coverage/calibration, answer accuracy | Not measured |
| Cold/warm latency p50/p95 and inference memory | Not measured |

Populate these fields from exported experiment records after actual runs. Include
sample counts, abstention handling, uncertainty and failure examples. Synthetic
random-weight smoke profiling only tests execution/memory on that shape; it cannot
support performance claims or definitive real-checkpoint throughput estimates.

## Use limits and disclosure

Use within the documented fictional English support domain and selected scopes.
Ambiguous context, unresolved conflicts and uncertain routing request clarification.
Generated text may endorse a wrong value or wrong subject despite a correct source.
Show provenance as evidence and label `model_generated`; do not label prose
"verified" merely because retrieval succeeded. Report structured outputs separately.

Measured validation/test behavior constrains any resume or panel claim. There is no
guarantee for arbitrary questions, unseen relation types, adversarial phrasing,
multi-hop reasoning, unknown products or real-world deployment.
