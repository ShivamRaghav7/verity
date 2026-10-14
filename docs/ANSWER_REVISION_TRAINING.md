# Two bounded development revisions

The remaining scope revision is **revision 2 of 2**. The answer objective is
**revision 1 of at most 2**. Neither has completed training or demonstrated an
improvement. Do not compare a weighted token loss with the original uniform
loss as if either were answer accuracy.

## Questions and inputs

| Run | Concrete question | Inputs retained | Sole intended training change |
| --- | --- | --- | --- |
| Scope | Does removing catalogue name and provenance cues improve applicability on unchanged held-out edits and the disclosed Nimbus workflow, without increasing false applications? | Pilot-v3: 240 train / 120 validation pairs, exact manifest and automated review registry; pinned DistilBERT, seed 42, batch 2, float32, learning rate 2e-5 | `entity_neutral_v1` classifier preprocessing; relation, values, quantities and unmatched selectors remain visible |
| Answer | Does additional equal weight on both opening Yes/No tokens improve generated comparison polarity while preserving factual short answers? | Pilot-v2: 96 train / 48 validation answers, exact manifest, automated reviews, inputs and targets; pinned FLAN-T5-small, seed 42, batch 2, float32, learning rate 2e-5 | `comparison_first_token_8x`, with weighted-loss checkpoint selection |

All 504 labels use disclosed automated development review. No human approval,
frozen-test accuracy or unrestricted-language reliability is claimed. Original
entity and lineage partitions remain fixed. Scope normalization creates eight
identical same-label pairs across those partitions. Report all 120 validation
rows, the eight overlapping rows and the 112 nonoverlapping rows separately
when assessing generalization. No contradictory transformed labels were found.

The hypothesis and limitations are in [scope preprocessing](SCOPE_INPUT_POLICY.md)
and [answer objective](ANSWER_OBJECTIVE.md). The recorded Nimbus failure remains
in [the original workflow evidence](../reports/pilot-v3-learned-workflow.json).

## Release measurements

Both released cards under `configs/run_cards/targeted-revisions/` passed
preflight. The RTX 4050 Laptop GPU profiles use the actual guarded trainer,
four optimizer updates, two full validations and two checkpoint writes.

| Measurement / planning budget | Scope | Answer |
| --- | ---: | ---: |
| Median of three steady optimizer samples | 0.106 s | 0.385 s |
| Full validation, observed range | 1.283–1.391 s / 120 rows | 2.544–2.624 s / 48 rows |
| Checkpoint write, observed range | 79.925–79.999 s | 11.229–11.834 s |
| Measured peak reserved GPU memory | 1.410 GiB | 1.656 GiB |
| Buffered GPU memory budget | 2.0 GiB | 2.25 GiB |
| Buffered runtime estimate | 1,331 s, about 22 min | 532 s, about 9 min |
| Checkpoint/export storage budget | 5 GiB | 6 GiB |

These are conditional planning estimates, **not guarantees**. The card uses the
largest observed steady-step, validation and checkpoint costs, observed loading/
tokenization/export overhead, and a 2× time buffer. Memory has a 25% reserved-memory
margin rounded upward. Scope profiling overlapped the app regression suite;
loading and storage costs varied widely across the preserved profiles. Storage,
OneDrive activity, CPU contention, thermal state and other applications can
change the actual duration. No complete run duration or accuracy is inferred
from four updates. The planned storage budgets cover retained and in-progress
checkpoints plus exports; existing artifacts also consume disk space.

Final exact profiles: [scope](../reports/profiles/guarded-scope-neutral.json)
and [answer](../reports/profiles/guarded-answer-polarity.json).
Earlier profiles/cards remain immutable [historical evidence](PROFILE_EVIDENCE.md).

Run **one command at a time**, after the preceding run reports completion:

```powershell
cd "C:\Users\shiva\OneDrive\Desktop\projekt-KE"
.\scripts\run-scope-final-revision.ps1
.\scripts\run-answer-revision.ps1
```

To inspect without training, add `-PreflightOnly` to either wrapper. Full runs
remain for the user to execute; only four-step profiles have run for these new
policies. The wrappers use the existing isolated `.venv-ml` and cached pinned
weights in offline mode. Do not reuse the earlier preliminary cards.

## Finite stopping and artifacts

Scope stops at exactly **360 optimizer updates**, three passes. It validates and
saves a checkpoint every 120 updates. Answer stops at exactly **240 updates**,
five passes, validating and checkpointing every 48 updates. There is no automatic
retry, budget extension or accuracy-based continuation. Both start afresh from
their pinned pretrained checkpoints.

Scope writes `artifacts/scope/pilot-v3-neutral2`; answer writes
`artifacts/answer/pilot-v2-polarity1`. Each directory contains `model/`,
`tokenizer/`, `manifest.json`, `training_log.jsonl`, `run_card.json`, and retained
`checkpoints/`. Earlier trained artifacts are preserved.

## Recovery and verification

Keep the exact card, environment and source files unchanged during a run or its
recovery. A completed checkpoint stores model, optimizer, scheduler, RNG, sampler,
global step, dataset, objective, preprocessing and configuration identities.
Interrupted uncommitted work is discarded when resuming the latest completed
checkpoint. The fixed stopping cap and scheduler do not restart.

For example, after an interrupted scope run has a completed checkpoint:

```powershell
$scopeCheckpoint = (Get-Content -Raw artifacts/scope/pilot-v3-neutral2/latest_checkpoint.json | ConvertFrom-Json).path
.\scripts\run-scope-final-revision.ps1 -ResumeCheckpoint $scopeCheckpoint
```

Answer recovery uses its own `artifacts/answer/pilot-v2-polarity1/latest_checkpoint.json`
and `run-answer-revision.ps1 -ResumeCheckpoint`.

Only the latest completed checkpoint from that run may be resumed. A completed
export or checkpoint at the stopping cap is rejected **before ML initialization
or manifest modification**. Corrupt state or changed identities are rejected.
Tiny CPU tests verify exact recovery and refused-resume artifact preservation;
these implementation checks are not trained-model performance measurements.

An interruption before the first scheduled checkpoint has no completed state
to resume. Preserve the partial evidence and prepare a fresh output/card rather
than deleting it or pretending exact recovery is available.

## After the user executes training

Audit the selected exports and retained checkpoints. For scope, report raw
classification, guard-aware calibration, coverage/abstentions and the Nimbus
workflow separately. For answer, report unweighted loss, opening-token polarity,
oracle-generated factual answers and all five matched end-to-end configurations.
Keep the old and new outputs and select or reject each artifact from development
evidence. No numerical improvement is promised by these objectives.

Then fix the final configurations and scoring protocol, independently review
and freeze the held-out/challenge labels, and run the final comparison once.
Test findings enter the failure analysis rather than triggering release retraining.
Finish the report, scientific plots, panel demonstration and deployment checks.
