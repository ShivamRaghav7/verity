# Reproduce one VLKEB/MEND edit

Use [the Colab notebook](../notebooks/02_vlkeb_mend_smoke.ipynb) to test the
published editor before building the new application around it. It executes one
fixed development edit, generated answers, patch persistence, and exact rollback.
It does not run meta-training or final evaluation.

## What this run can establish

The concrete question is whether the released BLIP-2/MEND editor can load,
produce an actual parameter edit, generate short answers, save/reload that edit,
and restore its six editable matrices on the allocated runtime. The technical
checks and answer quality are separate outcomes. A successful save/reload does
not mean that the target answer was learned successfully.

We reuse [VLKEB](https://github.com/VLKEB/VLKEB) at commit
`10951b7b3788928f578b73f07eac9e1eaa0316f3` and its released MEND checkpoint.
[MEND](https://arxiv.org/abs/2110.11309) supplies the learned editing method; the
checkpoint was trained by its upstream authors, not by this project. The wrapper
adds a bounded reproduction, integrity checks, generated-answer diagnostics, and
a persistent patch lifecycle. All compatibility adaptations are recorded by the
backend in the run report.

Published paper/repository results may be quoted as **upstream results**, with
their source, model, dataset and scoring protocol. They must not be presented as
measurements from our wrapper or copied into our experiment output. Our claims
come from saved local run evidence, including unsuccessful attempts. Reusing a
pretrained editor is disclosed reuse; this project has not meta-trained MEND.

The notebook pins our wrapper to
`a25a37034e6ff7ca1b73648405958e26b0e93959`. A later deliberate pin change should
identify the new commit and preserve the original run evidence.

The single fixture comes from upstream **TRAIN** rows 0 and 1. It may have been
seen during the editor's original training. Its corrected answer is a
counterfactual benchmark label, not an approved real-world correction. Neither a
successful edit nor a failure on this example estimates held-out accuracy.

## Run the notebook

1. Open `notebooks/02_vlkeb_mend_smoke.ipynb` in Colab and choose **Runtime →
   Change runtime type → GPU**.
2. Run the configuration and checkout cells. They fetch the pinned wrapper into
   `/content/editguard-vlkeb`. An existing checkout must be clean and match the
   pin; no existing work is reset or deleted.
3. Run the hardware inventory and environment cells. The isolated Python is
   `/content/editguard-vlkeb/artifacts/vlkeb/environment/bin/python`. The
   notebook kernel's Python remains unchanged.
4. Run the selective upstream import check before downloading models. If it
   fails, stop and preserve the error; later cells enforce stage guards.
5. Inspect the asset plan, then run the separate explicit download cell. It
   fetches the pinned files, checks their declared integrity values, prepares the
   training-only fixture, and verifies it again.
6. Run the planning cell to create a fresh output-directory name, then the smoke
   cell. It runs in the foreground. A nonzero exit, out-of-memory failure, or
   timeout is reported as a failure and retained in the logs.
7. Read the generated answers and technical checks. The optional final cell
   exports evidence, including failures. Set `DOWNLOAD_RESULTS = True` to start
   the browser download; no Google Drive authorization is required.

The notebook has ordinary Python code cells, without shell magics or background
workers managed by the notebook. The CLI itself deliberately executes editing
and reload in separate model processes. Detailed process output is written to
`edit.log` and `reload.log`; use Colab's Files panel to inspect them while the
foreground cell is running.

## Inputs and download size

`configs/vlkeb-assets.json` records 25 inputs totaling **13,218,483,144 bytes**
(13.218 GB / 12.311 GiB). They include OPT-2.7B, BERT, EVA, BLIP-2 Q-Former,
the released MEND editor, attribution/license notices, training annotations,
and the source image archive. The complete archive is required to extract the
fixed fixture's four images. Test annotation rows are not downloaded or evaluated.

The layout under `artifacts/vlkeb/assets/` is:

```text
opt-2.7b/
bert-base-uncased/
eva_vit_g.pth
blip2_pretrained_opt2.7b.pth
mend.pt
upstream-train.json
upstream-images.tar.gz
train-fixture.json
images/...
notices/...
asset-records.json
```

Downloads use pinned source revisions, exact declared byte counts and published
SHA-256 or MD5 values. The two GCS components publish MD5; local SHA-256 values
are recorded as well. These checks establish which upstream files were used;
they do not make arbitrary checkpoint files trustworthy. Load only the declared
research assets. OPT's model license permits noncommercial research; the code,
editor, annotations and source images have separate terms in the manifest.

The 13.218 GB figure excludes dependency packages, installation caches and saved
patches. Hardware inventory does not establish model VRAM or host-memory fit.
The first T4 session installed the initial dependency environment in 57.568
seconds. After adding the processor's missing OpenCV dependency, its selective
upstream import succeeded in 8.381 seconds. Those are setup observations, not
model-run time or memory estimates. Full-model loading and editing require their
own actual measurements.

## First measured T4 attempt

The preserved [first-run summary](../reports/vlkeb/t4-smoke-1/smoke-1/smoke.json)
uses wrapper commit `c945561659d31a6766fa53f200d4ce35dff795f1`, raw questions
and the release target-token convention. It **failed the complete smoke gate**:

| Observation | Measured result |
| --- | --- |
| Edit-phase duration | 128.027 seconds, including model loading and diagnostics |
| Same-process checks | 3 passed: weights changed, all six matrices restored exactly, baseline generated answers restored |
| Direct generated target | Did not match `Albany, New York`; the edited answer repeated `any` |
| Baseline answers | Empty for all four cases |
| Neighbor preservation | False; preserving a wrong answer would not establish correctness anyway |
| Fresh-process reload | Killed with return code `-9`; no phase report was written |
| Sampled peak process RSS | 11,912,224,768 bytes |
| CUDA peak allocated / reserved | 10,946,286,080 / 11,616,124,928 bytes |

Return code `-9` means the process was killed; high RAM use makes memory pressure
a plausible cause, but the return code alone does not prove an out-of-memory
kill. The resource figures describe this attempt, not expected training or
serving requirements. The saved report records the patch hash and weight changes;
that establishes a parameter update, not a correct generated answer. The first
Colab runtime's temporary disk was subsequently lost. Its JSON/log evidence was
exported locally, but the roughly 629 MB patch was not retained. It cannot now
be reloaded from that report alone.

The next adapter moves the loaded EVA encoder to CUDA earlier to reduce host
memory overlap without changing its weights or dtypes. That loading-order change
requires a fresh profile.

## Explicit diagnostic variants

The CLI and notebook default to **`--question-format raw --target-alignment
release`**. Keep those defaults when reproducing the first configuration.

- `raw` preserves the fixture question. `vqa` uses exactly
  `Question: {question} Answer:` for every edit and diagnostic query. This
  scaffold follows the [official Salesforce generation example](https://github.com/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb).
- `release` uses the released collator's standalone target-token labels.
  `context` takes labels from the target suffix of the jointly tokenized
  question, space and target. It requires the question tokens to remain an exact
  prefix and refuses an ambiguous boundary. This is a documented adaptation,
  rather than exact reproduction of release collation.

`Albany, New York` has six standalone tokens versus four after a separating
space with the selected OPT tokenizer. Selecting a loss suffix using the former
count can supervise question positions. The context variant addresses that
alignment defect, but the released editor was itself trained using release
collation, so improved generation remains an experimental question. Each
variant uses a new output directory and records its options. Do not sweep
examples until one looks successful.

The next fixed development attempt uses **`vqa` + `context`** on the same fixture
and executes the complete edit, save/reload and rollback workflow. This is a
planned diagnostic configuration, not a successful result. To run it in the
notebook, explicitly set `QUESTION_FORMAT = "vqa"` and
`TARGET_ALIGNMENT = "context"` before planning. Historical defaults remain
`raw` / `release`.

## Finite execution and recovery

Each worker is capped at **1,800 seconds**, and each answer at **24 new tokens**.
The first process generates four baseline answers, performs one edit, generates
the same four questions, rolls back, and generates them again. The second
process constructs the base model afresh, reloads the saved patch, generates the
four questions, rolls back, and generates them again. This is **20 generation
calls across five answer states**, for one edit.

The four cases are the direct edit question, a paraphrase, another photograph,
and one neighboring question. No target answer is supplied to the generated
query call. No further examples are selected according to observed success.

The timeout is a stopping cap, not a prediction of runtime. The command records
worker duration, sampled process RSS, and CUDA allocator peaks. It does not
measure serving p50/p95 latency. Process-RSS sampling can miss short spikes, and
CUDA allocator figures omit some driver allocations; those limits are included
in the report.

Interrupted `.part` downloads can resume if the runtime's disk survives; already
verified files are reused. A lost Colab session loses its temporary disk unless
the user separately saves it. A checksum mismatch stops the download rather than
silently selecting a newer upstream file.

A model worker does not resume partial state. Choose a new output directory for
a retry and retain the earlier failure. In the notebook, rerun the planning cell
to obtain a fresh timestamped directory. The model configuration and fixture
remain fixed. A failure does not justify switching examples until a favorable
answer appears.

## Equivalent commands

From the repository root on Linux, the setup steps are:

```bash
python scripts/bootstrap_vlkeb_colab.py --fetch-upstream --install

artifacts/vlkeb/environment/bin/python -c 'from pathlib import Path; from ke_repro.bridge import _upstream_modules; print([m.__name__ for m in _upstream_modules(Path("tmp/vlkeb-upstream").resolve())])'

artifacts/vlkeb/environment/bin/python scripts/prepare_vlkeb_assets.py --root artifacts/vlkeb/assets
artifacts/vlkeb/environment/bin/python scripts/prepare_vlkeb_assets.py --download --root artifacts/vlkeb/assets
artifacts/vlkeb/environment/bin/python scripts/prepare_vlkeb_assets.py --verify --root artifacts/vlkeb/assets
```

The first asset command only prints a plan. The `--download` command explicitly
fetches the large files. After it succeeds, inspect the finite run plan:

```bash
artifacts/vlkeb/environment/bin/python -m ke_repro plan \
  --upstream tmp/vlkeb-upstream \
  --assets artifacts/vlkeb/assets \
  --manifest configs/vlkeb-assets.json \
  --output artifacts/vlkeb/runs/smoke-001 \
  --device cuda --max-new-tokens 24 --timeout-seconds 1800 \
  --question-format raw --target-alignment release
```

Then execute that same configuration with `smoke` instead of `plan`. The selected
output directory must not already exist:

```bash
artifacts/vlkeb/environment/bin/python -m ke_repro smoke \
  --upstream tmp/vlkeb-upstream \
  --assets artifacts/vlkeb/assets \
  --manifest configs/vlkeb-assets.json \
  --output artifacts/vlkeb/runs/smoke-001 \
  --device cuda --max-new-tokens 24 --timeout-seconds 1800 \
  --question-format raw --target-alignment release
```

The smoke command verifies the authenticated fixture and model assets again
before checkpoint deserialization. It uses local assets in offline mode during
model execution and does not silently switch to CPU if CUDA is unavailable.

## Optional diagnosis of a retained patch

`ke_repro.diagnose` is useful only when the actual `patch` file from an earlier
run still exists. A JSON report or patch hash cannot reconstruct it. In
particular, it cannot recover the lost first-run patch. Use the matching pinned
base assets and a fresh output directory:

```bash
artifacts/vlkeb/environment/bin/python -m ke_repro.diagnose \
  --upstream tmp/vlkeb-upstream \
  --assets artifacts/vlkeb/assets \
  --manifest configs/vlkeb-assets.json \
  --patch artifacts/vlkeb/runs/smoke-001/patch \
  --output artifacts/vlkeb/runs/diagnose-001 \
  --device cuda --timeout-seconds 1800
```

This performs **no new edit or training**. One fresh model answers the same four
fixed questions using raw and VQA formats before and after loading that patch
(16 generation calls). It also records eight teacher-forced scores under
release/context alignment, two tokenization reports, and exact-weight rollback.
It verifies the existing patch's hash before loading, writes `run_plan.json`,
`preflight.json`, `diagnostic.json`, and `diagnostic.log`, and stops after at most
1,800 seconds. That cap is not a runtime estimate. Preserve these files as a
separate diagnostic configuration; do not count teacher-forced scores as
generated-answer accuracy.

## What the results mean

| Field or file | Interpretation |
| --- | --- |
| `technical_gate_passed` | Eight edit, identity, persistence and exact-restoration checks passed across the two processes. It is not an answer-accuracy score. |
| `direct_edit_target_matched` | The whole generated short answer conservatively matched the fixed target after case/punctuation normalization. |
| `generated_answer_diagnostics` | The four edited-state answers and their individual target matches. Read paraphrase and alternate-image outcomes separately. |
| `neighbor_answer_preserved` | The neighbor's generated answer stayed the same. Its factual target match is a separate check; preserving a wrong answer is not factual correctness. |
| `accuracy_estimate: null` | Intentional: one training-seen edit cannot support a benchmark accuracy estimate. |
| `phase_execution` | Each process's actual completion/failure and resource observations. |
| `run_plan.json` and `preflight.json` | The declared finite question/configuration and verified input identities. |
| `edit.json` / `reload.json` | Generated answers, matrix hashes, technical checks, compatibility adaptations and resources. |
| `edit.log` / `reload.log` | Detailed model/worker output, including failures that may occur before a JSON report can be written. |
| `patch` | One file containing original and edited tensors for the six editable matrices, with identity metadata. It requires matching base assets and is not a standalone model. |

Conservative whole-answer matching may reject a semantically correct longer
answer. Such cases should be inspected and disclosed rather than counted as
automatic successes. Conversely, simply mentioning a target within a negation
or example is insufficient. This first smoke is a feasibility profile; a later
fixed, reviewed evaluation protocol is required for broader quality claims.

Teacher-forced target-token diagnostics condition on the full target answer.
They are useful for locating alignment or decoding errors, but must never be
counted as independently generated-answer accuracy. Record the question format
and target alignment alongside all comparisons.

The export includes run JSON/logs/patch files, the hardware inventory, dependency
receipt and resolved requirements, import record, asset manifest, integrity
records, and an export manifest with file hashes. It excludes full model weights,
the image archive and environment packages. Save that archive before releasing a
temporary Colab runtime. The existing local website is not replaced by this run.
