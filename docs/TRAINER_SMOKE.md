# Actual trainer smoke verification

Recorded on 2026-10-11 in the isolated `.venv-ml` environment. These are
developer feasibility measurements on literal synthetic fixtures. They are
neither human-reviewed training results nor accuracy measurements. The source
record is [`reports/trainer-smoke.json`](../reports/trainer-smoke.json).

## Question and finite execution

Can the cached pinned DistilBERT scope backbone and FLAN-T5-small answer backbone
perform real CUDA updates, save a complete checkpoint at step 2, and resume to
the original four-step stopping cap? Both tasks completed this sequence. The
runtime rejected both exported artifacts because `purpose=smoke_test`.

No catalogue training data, reviewed pilot, validation development dataset or
frozen test dataset was read. Scope used four synthetic training pairs and two
validation pairs. Answer used two synthetic training rows and one validation
row. Actual maximum input lengths were 53 scope tokens and 82 answer tokens;
the models were padded to the declared 192 tokens. Answer targets were padded
to 48 tokens. No row was truncated.

## Measured resources and timing

Hardware: NVIDIA GeForce RTX 4050 Laptop GPU, driver 581.86. Software:
Python 3.14.3, Torch 2.9.0+cu130, Transformers 5.16.1. Both paths used batch 2,
float32, eager attention, deterministic algorithms and AdamW at an initial
learning rate of 2e-5. Validation and checkpoints ran after every smoke step.

| Measurement | Scope | Answer |
| --- | ---: | ---: |
| Optimizer steps, including resumed segment | 4 | 4 |
| Median optimizer time over all four steps | 0.368 s | 0.546 s |
| Peak observed Torch allocated GPU memory | 1.318 GiB | 1.508 GiB |
| Peak observed Torch reserved GPU memory | 1.410 GiB | 1.664 GiB |
| Maximum observed process RSS at logged steps | 1.712 GiB | 2.283 GiB |
| First segment loop time, including validation and checkpoint writes | 13.217 s | 18.922 s |
| Resumed segment loop time, including validation and checkpoint writes | 13.328 s | 32.718 s |
| First segment process time | 88.608 s | 34.042 s |
| Resumed segment process time | 36.992 s | 63.364 s |
| Full optimizer checkpoint directory, each retained checkpoint | 803,602,299 bytes | 1,055,374,138 bytes |
| Selected model and tokenizer export | 268,545,255 bytes | 310,295,648 bytes |

These four-step medians include startup effects; there is no separate warmup
sample in this trainer verification. Process time includes imports, pinned
model loading, hashing, tokenization, training, checkpoint I/O and final export
where applicable. Loop time excludes loading, tokenization and final export.
RSS is sampled at logged steps rather than continuously. Torch memory counters
do not describe every allocation by the driver or other applications.

Checkpoint I/O materially exceeded optimizer compute in this workspace, and
varied across segments. A substantial run estimate must measure the actual
reviewed pilot, chosen validation schedule and checkpoint frequency. These
small fixtures do not justify promising a full-training completion time.

The FLAN loader again reported that checkpoint shared and output weights differ
despite the saved tying configuration. The trainer preserved the pinned
loader's behavior instead of silently tying the values. The DistilBERT binary
classification head was newly initialized, as expected for this adaptation.

## Resume and rejection evidence

Each pretrained run retained the original four-step cap and learning-rate
schedule while resuming from `step-00000002`. Logs contain steps 1–4 exactly
once, with full optimizer/scheduler/RNG/sampler state restored. Saved artifacts
record model revisions, environment and code/data identity hashes. This verifies
actual pretrained checkpoint recovery; this smoke did not additionally compare
a separate uninterrupted CUDA run tensor by tensor.

The separate tiny CPU optimizer regression passed all seven trainer tests in
7.439 seconds. It compared uninterrupted four steps with two steps followed by
resume and required exact equality of parameters, losses, batch orders,
validation selection and subsequent Python/NumPy/Torch RNG draws. Corrupt
checkpoint state was refused before restoration. That controlled CPU result
does not guarantee identical behavior under a changed device, package stack or
training configuration.

Both selected exports were also reloaded on CPU using the official
`AutoModelForSequenceClassification` and `AutoModelForSeq2SeqLM` classes, with
their sibling `tokenizer/` directories and `local_files_only=True`. Every
selected tensor-state digest matched the original manifest exactly; tokenizer
vocabulary/special-token/padding identities also matched. One inference forward
pass on each task's synthetic training fixture produced finite logits and loss.
Scope logits had shape `[4, 2]`; answer logits had shape `[2, 48, 32128]`.
The smoke manifests stayed byte-for-byte unchanged. This checks export/load
compatibility only: the artifacts remain ineligible for trained-model routing,
and no generated-answer quality was measured. The answer reload preserved the
same untied-weight behavior and emitted the documented loader warning.

## Reproduction

Use the same pinned cached model revisions from `configs/models.json`, install
the recorded isolated ML dependencies, and set `HF_HOME` to the absolute
`runtime/hf-cache` directory. `HF_HUB_OFFLINE=1` keeps this verification cached.
For each task, use a fresh output directory and invoke:

```powershell
.\.venv-ml\Scripts\python.exe scripts/train_auxiliary.py --smoke --task scope --device cuda --output runtime/smoke-scope --max-steps 4 --stop-after-steps 2
.\.venv-ml\Scripts\python.exe scripts/train_auxiliary.py --smoke --task scope --device cuda --output runtime/smoke-scope --max-steps 4 --resume runtime/smoke-scope/checkpoints/step-00000002
```

Replace `scope` with `answer` and use a distinct `runtime/smoke-answer` directory
for the answer task. Existing run directories cannot be silently overwritten.
Do not change trainer, preflight or data preparation source between segments;
resume validates their hashes. The report records the exact executed commands.

For the independent CPU regression:

```powershell
$env:EDITGUARD_RUN_ML_TESTS = "1"
.\.venv-ml\Scripts\python.exe -m unittest discover -s tests -p test_trainer.py -v
```

Full reviewed-data training, scope threshold calibration and final test
evaluation remain separate milestones.
