# EditGuard: Visual Knowledge Editing Workbench

EditGuard lets a researcher propose a correction to a vision–language model,
inspect its generated answers and side effects, approve a saved parameter patch,
activate it, and restore the original weights exactly. It integrates the released
**VLKEB/MEND editor with BLIP-2 OPT-2.7B**, alongside a bounded last-layer
fine-tuning baseline and an updated visual retrieval comparison.

The saved GPU development runs establish actual parameter changes, patch
persistence, fresh-process reload and exact rollback. On the fixed development
example, fine-tuning generates the requested full answer on the direct question
and another photograph, but misses the full paraphrase target and changes an
unrelated answer to repetitive output. This demonstrates both editing and its
risks. It is one training-seen example, not a benchmark success rate or an
arbitrary-edit guarantee.

Start with the [project explanation](docs/VISUAL_PROJECT_EXPLAINED.md),
[measured results and limitations](docs/VLKEB_RESULTS.md), and
[workbench guide](docs/VLKEB_WORKBENCH.md). Published paper numbers remain in a
separate [attributed reference](docs/VLKEB_UPSTREAM_RESULTS.md).

## What this project contributes

| Component | Responsibility |
| --- | --- |
| Pretrained visual model and MEND network | Reused upstream research assets; we did not train the foundation model or meta-train the released editor. |
| Reproduction bridge | Pinned assets, integrity checks, bounded generated answers, explicit token alignment, saved tensors and exact restoration. |
| Last-layer FT | Real per-edit optimization: 15 fixed AdamW steps on two OPT matrices, with disclosed precision and answer-termination adaptations. |
| Updated RAG comparison | The current correction in image/question memory, pretrained text and visual features, and a fixed applicability gate. |
| Research workbench | Inactive candidate preview, explicit approval and activation, provenance, history, evidence export and rollback. |
| Experiment controls | A 20-edit, five-method protocol with sealing, strict whole-answer scoring, linked cases, paired uncertainty and separate latency measurements. |

The current measured evidence is described in [the results report](docs/VLKEB_RESULTS.md).
The final five-method evaluation has not yet produced a completed report. An
implemented method or passing application test is not a measured model result.

## Run the workbench

Use the [standalone Colab workbench notebook](notebooks/03_visual_editing_workbench.ipynb)
for the pinned environment, verified asset download, CUDA hardware checks and
private notebook viewer. The [one-edit reproduction notebook](notebooks/02_vlkeb_mend_smoke.ipynb)
and [reproduction guide](docs/VLKEB_SMOKE.md) preserve the earlier development
configurations and failures.

After preparing the Linux/Colab environment and assets, run from the repository root:

```bash
artifacts/vlkeb/environment/bin/python -m ke_repro.workbench --port 8003
```

Open `http://127.0.0.1:8003/` on that host, or use the notebook's signed-in port
viewer. Wait for **Model ready**, select a development fixture, generate its
original answer, choose MEND or FT, then click **Preview correction**. Review the
candidate's target answer and unrelated probe before **Approve and activate**.
Ask a paraphrase, export the evidence, then roll back and regenerate.

Preview restores the base before returning; the patch remains inactive until
explicit approval. Only one patch can be active. The session token and loopback
binding support a local research demonstration; they are not a production
multi-user administration system.

The measured full-model path uses a Tesla T4. The laptop's 6 GB GPU is not the
demonstrated target. Model/data assets require a separate download of about
13.2 GB before dependencies and patches; each saved candidate patch is about
600 MiB. Preserve exported evidence and patches before releasing a temporary
Colab runtime. Never run simultaneous full-model jobs on the same GPU.

## Evidence and reproduction

The [results report](docs/VLKEB_RESULTS.md) links the original machine-readable
receipts, including the unsuccessful first run. The
[fixed evaluation protocol](docs/VLKEB_EVALUATION_PROTOCOL.md) defines 20 source
entities, five cases per edit and five configurations: base, prompt-only, updated
RAG, context-aligned MEND and adapted last-layer FT. Missing and failed answers
remain in the assigned denominators. The selection is a small nonrandom slice;
source-directory disjointness does not prove absence from pretraining.

`scripts/render_vlkeb_results.py` renders saved evidence into PNG/PDF figures and
CSV/JSON data with source hashes. It loads no model and estimates no missing
metrics. Renderer verification uses visibly marked synthetic fixtures; those
figures are never project results.

Focused application checks run without the large model:

```bash
python -m unittest tests.test_ke_repro_workbench
```

The 11 workbench contract tests use a tiny fake backend to verify candidate
review, activation, failure handling, exports and restoration behavior. Actual
model evidence comes from GPU reports. [Container and HTTP details](docs/VLKEB_WORKBENCH.md)
document the packaged Linux path and its validation limits.

## Limits to the claim

The project does not guarantee that an arbitrary correction works or stays
local. Teacher-forced token scores can be high while generated answers repeat
or fail. The development fixture comes from upstream training annotations, and
its labels are controlled counterfactual associations. We have not independently
human-reviewed those labels or established real-world factual truth from them.
Our loading, contextual alignment and FT objective adaptations also prevent
claiming reproduction of the paper's full benchmark numbers. Advantage over
retrieval, prompt correction or other models must follow measured comparisons.

## Preserved earlier projects

The earlier **VisionPatch** CLIP projection-editing studio remains separately
runnable at port 8002 and preserved at Git tag
`visionpatch-preserved-2026-10-22`. Its completed study belongs to that project:
read the [studio guide](docs/visionpatch/WORKBENCH_GUIDE.md),
[V2 results](docs/visionpatch/RESULTS_V2.md), and
[reproduction instructions](docs/visionpatch/REPRODUCING_V2.md). Start its existing
environment with `scripts/run-visionpatch.ps1`.

The earlier customer-support **EditGuard V1** is preserved at `editguard-v1.0.0`
with its [original README](docs/EDITGUARD_V1_README.md). Its text-task results and
the VisionPatch results are historical evidence for those systems, separate
from this VLKEB visual question-answering experiment.
