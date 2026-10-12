# Finite auxiliary training and exact resume

`scripts/train_auxiliary.py` trains the scope classifier or the edit-conditioned
answer model. The structured app imports no ML package through this module.
Production training is deliberately gated on reviewed train/validation data and
a preflight-valid run card. No frozen test file is read by the trainer. No
substantial command is released without reviewed inputs and actual trainer
measurements. The [pilot run instructions](PILOT_TRAINING.md) contain the initial
finite commands and their conditional estimates.

## Implemented optimizer path

- Scope: pinned cached `AutoModelForSequenceClassification`, two labels, plain
  question as tokenizer input 1 and correction text as input 2. Class 1 means the
  edit applies. Entity/plan/version/date filtering remains in deterministic
  routing; request selectors are not silently added to classifier text.
- Answer: pinned cached `AutoModelForSeq2SeqLM`, exact dataset `input_text` to
  `target_text`; padding labels are `-100` and excluded from token loss.
- Both: batch 2, float32, AdamW (betas 0.9/0.999, epsilon 1e-8), default weight
  decay 0.01, global gradient clipping at norm 1, linear learning-rate decay to
  zero at the unchanged `max_steps` cap. Eager attention and deterministic Torch
  algorithms are requested; unsupported nondeterministic operations stop the run.
- A separate Python shuffle RNG records full order, epoch and cursor. A final
  smaller batch is retained. Validation is unshuffled and uses loss weighted by
  rows for scope, by non-padding target tokens for answer.
- `validation_every_steps` defaults to `checkpoint_every_steps`. The final cap
  always validates. The lowest validation loss selects the exported model; no
  test results influence selection. Ties retain the earlier checkpoint.
- Input or target truncation is rejected with measured token counts. Revise the
  declared lengths and reprofile rather than discard the correction or target.

The source code requires cached pretrained weights and a resolved 40-character
model revision. It does not download a different model during a run. The frozen
base generator is a separate artifact and is never trained by this script.

Fresh-machine preparation must cache all starting backbones in the same directory
the trainer resolves. Follow [ML_SETUP.md](ML_SETUP.md), then use the absolute
project cache location in each training shell before importing Transformers:

```powershell
New-Item -ItemType Directory -Force runtime/hf-cache | Out-Null
$env:HF_HOME = (Resolve-Path runtime/hf-cache).Path
.\.venv-ml\Scripts\python.exe scripts/cache_models.py --model all --aux-config configs/models.json
```

This explicit download command validates pinned revisions and caches FLAN-T5,
MiniLM and DistilBERT. Duplicate model/revision pairs are loaded once. The cache
tool's default `both` selection covers only frozen baselines, so it does not
prepare DistilBERT for scope training. Subsequent trainer commands remain
cached-only and never turn a missing checkpoint into an implicit download.

## Reviewed-run contract

The existing `scripts/preflight_training.py` checks exact manifest, row-file,
review-registry and profile hashes; train/validation family, entity and lineage
separation; model revision; profiled batch/lengths/precision; finite `max_steps`;
and the checkpoint plan. `train_auxiliary.py` calls this before importing Torch.
Accepted rows must have current hash-bound decisions under the declared review
policy. Human review remains the default. The user-authorized
`automated_development` policy requires typed machine decisions and an independent
audit hash, and writes `human_reviewed_data=false` to model artifacts. Passing preflight is
not a claim that training, calibration or final evaluation succeeded.

The actual device must match the completed profile. New optional run-card
fields are `validation_every_steps`, `weight_decay`, `max_grad_norm`,
`cpu_threads` (default 2) and `scheduler` (only `linear` is implemented).

The release checklist for a substantial command remains:

1. A concrete experiment question and fixed reviewed inputs.
2. Actual trainer measurements for realistic pilot lengths, validation and
   checkpoint I/O, with hardware and software records.
3. Expected artifacts and a measured estimate with exclusions and assumptions.
4. Exact-state resume instructions and the fixed stopping cap.
5. User execution of the long GPU command. Developer smoke is separate.

## Checkpoints and resume

Each `checkpoints/step-000000NN/` is written to a temporary sibling directory,
flushed, and atomically renamed only after state and metadata are complete.
`state.pt` stores model, AdamW, scheduler, Python/NumPy/Torch CPU and initialized
CUDA RNG states, sampler order/cursor/RNG, global step, identity and current
validation selection. Torch loads tensor/primitive state with `weights_only=True`.
The metadata checksum detects accidental corruption before state restoration.

Identity includes exact run-card bytes, configuration, reviewed row content,
review and dataset-manifest hashes, initial pretrained tensor values, tokenizer
vocabulary/special tokens, trainer/data/preflight source hashes and the software,
device (including NVIDIA driver and GPU UUID for CUDA) and deterministic-settings
fingerprint. A difference fails resume. Do
not change `max_steps`, learning rate, dataset, review registry, code, hardware
or package versions while calling the operation an exact resume.

Use the same original run-card file with CLI `--resume <checkpoint-directory>`;
the operational resume argument does not modify the saved card. The directory
must be inside that run's output. `--stop-after-steps N` is an absolute early
checkpoint boundary; it retains the original cap and schedule. It does not add
an unscheduled validation decision. An interrupted manifest is unavailable to
the application.

Existing output is never silently overwritten by a fresh run. On resume,
uncommitted log suffixes after the checkpoint step are backed up before trimming.
Retention preserves the last configured checkpoints plus the selected best
checkpoint if it is older. Full AdamW state consumes substantially more disk
than a weights-only export; inspect actual checkpoint sizes in the pilot.

Atomic rename prevents consumers from seeing partial completed checkpoint
directories. It is not a claim of immunity to storage failure or power loss.

## Artifact manifest consumed by inference

`manifest.json` declares:

```json
{
  "schema_version": 1,
  "artifact_type": "editguard_auxiliary",
  "purpose": "trained",
  "status": "completed",
  "task": "scope",
  "model_path": "model",
  "tokenizer_path": "tokenizer",
  "input_format_version": "editguard-input-v1",
  "input_format": "question_then_correction_pair",
  "positive_class_index": 1,
  "num_labels": 2,
  "selection": {
    "metric": "validation_loss",
    "value": "actual measured loss is a number",
    "global_step": "actual selected step is an integer",
    "checkpoint": "step-000000NN"
  },
  "files_sha256": {"model/model.safetensors": "actual SHA-256"}
}
```

The illustrative placeholders above are not results. Actual manifests also
contain pinned source model identity, dataset identity, all resume hashes,
environment details, tokenization statistics, final and selected tensor-state
hashes, completed/max steps and resource observations. Every file under `model/`
and `tokenizer/` is hashed. The answer manifest uses task `answer`, input format
`dataset_input_text_v1`, and null classification fields.

The final exported model is the selected checkpoint; `training.final_state_sha256`
describes the last optimizer step, which can differ from the selected model.
The application must reject smoke, running, interrupted, incomplete or
integrity-mismatched artifacts. Scope calibration remains a separate
validation-only artifact; training completion alone does not establish a usable
decision threshold or reliability claim.

Outputs are the exact `run_card.json`, atomic checkpoints and metadata,
`latest_checkpoint.json`, per-step/resource `training_log.jsonl`, the selected
Hugging Face model and tokenizer, and the final manifest. Recorded elapsed loop
time is per invocation segment on resume and includes validation/checkpoint I/O;
optimizer timings exclude validation, checkpoint I/O, loading and tokenization.
RSS is recorded when psutil is installed; GPU allocated/reserved peaks are real
Torch observations, not estimates of total system memory.

## Developer smoke and implemented verification

Explicit `--smoke` mode uses literal synthetic Nimbus fixtures, cached real
pretrained weights, batch 2 and a cap of four optimizer steps. No human review
is invented. Artifacts declare `purpose=smoke_test`, even when a real neural
optimizer completed; the runtime must refuse them as production trained models.
Smoke supports both tasks, CPU or CUDA, early checkpoints and exact resume.
It validates mechanics and feasibility, not corrected-answer accuracy.

Normal app CI runs the sampler, hash mismatch, atomic-write and lazy-import
checks without Torch. The optional real CPU regression uses a tiny classifier
with dropout, AdamW and the same checkpoint loop. With
`EDITGUARD_RUN_ML_TESTS=1`, `python -m unittest discover -s tests -p test_trainer.py -v`
also checks uninterrupted four steps against two steps plus exact resume:
parameter tensors, losses, batch orders, validation selection and subsequent
Python/NumPy/Torch draws must match exactly. It also corrupts a checkpoint and
checks refusal before restore.

On the inspected global Python/Torch environment these seven tests passed in
23.740 seconds, including ML import time. Five pure checks pass in the isolated
application environment and the two optional optimizer tests are explicitly
skipped there. This CPU fixture proves the resume implementation under its tested
conditions; it is not a pretrained-model training measurement or an accuracy
number. Actual pretrained smoke/trainer profiles must be recorded separately.
