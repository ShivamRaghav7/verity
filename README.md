# VisionPatch

The new research backend is **EditGuard: Visual Model Editing Workbench**.
It reuses the published VLKEB/MEND editor, with documented compatibility and
token-alignment adaptations. Start with the [Colab reproduction guide](docs/VLKEB_SMOKE.md)
and [new workbench instructions](docs/VLKEB_WORKBENCH.md).
The first GPU attempt and its failures are preserved in `reports/vlkeb/t4-smoke-1`;
published paper results are not our measured results. The application below is
the preserved earlier project, and remains separately runnable.

The next-project [upstream research decision](docs/UPSTREAM_PROJECT_SCOUT.md)
recommends reproducing a published multimodal editor before replacing this app.
The current studio and results are preserved at `visionpatch-preserved-2026-10-22`.

**Targeted visual model editing with measured generalization and exact rollback.**

**Current claim:** a controlled model-repair study and working editor. Practical
advantage over stronger models and simpler alternatives remains unproven.
[Usefulness assessment and the remaining evidence gate](docs/visionpatch/USEFULNESS.md).

VisionPatch changes a learned visual association inside CLIP: a misleading printed
word can make the model misidentify an object. A small learned projection patch
repairs that behavior on new photographs without changing the frozen encoders.
Registered area: Knowledge Editing; specifically **visual model / concept editing**.

This branch contains a correction-authoring application and a completed,
fixed-configuration selectivity study. It does not generate captions or unrestricted visual answers.
The preserved customer-support EditGuard V1 remains at the `editguard-v1.0.0` tag;
its original instructions are saved in [the V1 README](docs/EDITGUARD_V1_README.md).

## V2: selective repair and interacting corrections

Open **http://127.0.0.1:8002/** after starting the server. Supply a failing
image and matched reference, fit a new parameter update, inspect collateral
changes, approve it, activate it, compare actual inference and roll back. SQLite
persists the version chain across restarts. Preview and approval do not activate
parameters; stale parents and revisions are rejected.

The V2 study covers three misleading words, held-out fonts and placements,
legitimate text recognition, five comparison methods, and all six orders of
three sequential edits. It uses 80 fitting/reference photos, 32 fresh validation
photos and 48 fresh final photos. All 224 photographs observed in V1 are excluded
from V2 validation/final selection.

| Misleading word | Original CLIP | Object-only edit | Protected edit | Matched fine-tuning | Oracle mask |
| --- | --- | --- | --- | --- | --- |
| camera | 55/96 | 92/96 | 91/96 | 92/96 | 96/96 |
| laptop | 31/96 | 88/96 | 77/96 | 86/96 | 96/96 |
| watch | 68/96 | 92/96 | 92/96 | 92/96 | 96/96 |

These are **96 questions from 48 photos at two positions**, not 96 independent
photos. Every method retained 48/48 clean-object answers. The object-only watch
edit damaged two text-card answers (22/24); its protected counterpart retained
24/24, while laptop repair was weaker with protection. Sequential edits interfered:
final laptop accuracy ranged from 51/96 to 74/96 across orders. All six orders
restored the exact original projection on rollback. Oracle masking knows both the
synthetic region and task; it is a privileged control, not a deployed detector.

The study demonstrates repair, preservation tradeoffs and interference. It does
not establish superiority to fine-tuning, arbitrary editing, or production
reliability. Labels received automated auditing, without independent human review.

![Measured V2 selectivity](output/visionpatch/study-v2/selectivity.png)

Start with [the ground-up studio guide](docs/visionpatch/WORKBENCH_GUIDE.md), then
read [the V2 results and failure analysis](docs/visionpatch/RESULTS_V2.md),
[architecture](docs/visionpatch/ARCHITECTURE_V2.md) and
[predeclared study protocol](docs/visionpatch/STUDY_V2.md). Complete observations,
validation selection, resources and the final seal are in `reports/visionpatch/study-v2`.

The first visual pilot is archived in Git and its
[results](docs/visionpatch/RESULTS.md) and
[protocol](docs/visionpatch/PROTOCOL.md). Its measured evidence and preparation
artifacts are preserved, but its earlier interface is no longer part of the website.

## Run the prepared demonstration

```powershell
.\scripts\run-visionpatch.ps1
```

Open http://127.0.0.1:8002/ for the persistent correction studio;
http://127.0.0.1:8002/workbench opens the same application. Create a candidate,
inspect its evidence, approve it, activate it and compare live recognition.
Use **Rollback** to restore a saved projection. The study section displays the
separate completed experiment. API documentation:
http://127.0.0.1:8002/docs. Stop the server with Ctrl+C.

Uploaded images run the actual pinned encoder. The eight candidate similarities
are uncalibrated scores; an unknown object still receives one of those candidates.
Candidates, approvals, activation history and rollback states persist in the
studio's registry. This loopback demo has no production authentication.

## Prepare a fresh checkout

The image archive and model weights are downloaded separately, not stored in Git.
CPU is sufficient. The original measured environment used Python 3.14.3,
PyTorch 2.9.0, Transformers 5.16.1, NumPy 2.4.2 and Pillow 12.3.0.

```powershell
py -3.14 -m venv .venv-vision
.\.venv-vision\Scripts\python.exe -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-vision\Scripts\python.exe -m pip install -r requirements-vision.txt
.\.venv-vision\Scripts\python.exe -m visionpatch.prepare --allow-download
.\.venv-vision\Scripts\python.exe -m visionpatch.profile --device cpu --allow-download --output reports/visionpatch/profile-local.json
.\.venv-vision\Scripts\python.exe -m visionpatch.pilot --device cpu --report reports/visionpatch/pilot-local.json
.\scripts\run-visionpatch.ps1
```

The `pilot` command prepares the frozen-feature and projection artifacts consumed
by the studio; it does not launch an older website. Use unused report/artifact
paths for repeat experiments. This preparation stops after
one closed-form solve and 25 small projection fine-tuning steps. Completed
artifacts are never overwritten. The application reads the matching report saved
beside its generated patch. Fresh-machine measurements may differ.

The final evaluation has two phases: `python -m visionpatch.finalize --prepare-only`
seals the data, code and trained-artifact hashes before inference; running without
that flag consumes the seal. Recorded final outputs are immutable. Reproduction
on a different checkout needs distinct `--dataset`, `--artifacts`, `--seal` and
`--report` paths; do not overwrite the published evidence.

For V2, [the reproduction guide](docs/visionpatch/REPRODUCING_V2.md) explains exact
manifest-based data recovery and finite `profile → develop → seal → finalize`
commands. `scripts/run-visionpatch-study.ps1` writes new run directories, preserving
published results. Saved feature/model artifacts stay outside Git; input
manifests, code, configurations, selections, observations and figures are
versioned. The application uses SQLite and ordinary files; no external database
or experiment server is required.

## Architecture

```mermaid
flowchart LR
    A[Photograph] --> B[Frozen CLIP image encoder]
    B --> C[Visual projection + learned low-rank patch]
    C --> E[Normalized class similarities]
    D[Fixed labels] --> T[Frozen CLIP text encoder]
    T --> E
    S[Four paired correction images] --> F[Regularized parameter fit]
    R[Separate retention images] --> F
    F --> P[Saved patch + parent hash]
    P --> C
```

This is an adaptation of associative-memory editing, not a novel editing algorithm
or a faithful reproduction of the original spatial classifier editor. Only the
final visual projection changes. [Research foundation](https://arxiv.org/abs/2112.01008).

## Validation and deployment

Run `python -m unittest discover -s tests -v` in the ML environment. The original
45 visual tests and new registry, study, workbench and recovery modules cover
fitting, split separation, denominators, approval, API behavior, artifact
integrity, sealing, conflicts, persistence and exact rollback. Optional checks
and environment-specific skips are reported explicitly.
The workbench API regression tests exercise candidate creation, live inference,
approval, activation and restoration. The archived `visionpatch.verify_http`
module concerns the earlier pilot API, which is disabled in normal serving.
`python -m visionpatch.plot_results` regenerates the archived pilot's scientific
PNG/SVG figures; the V2 reproduction guide describes current study outputs.

The latest recorded studio CPU HTTP sample has p50 664 ms and p95 896 ms, from
12 serial requests for one support image. Earlier samples were faster; their
cause was not isolated. These are small local measurements, not a concurrency
benchmark or a runtime promise. All samples remain in the results report.

A CPU container recipe is provided in `Dockerfile.visionpatch`; see
[container instructions](docs/visionpatch/DEPLOYMENT.md). GitHub Actions checks the
original application and the new visual components. Large data/model files stay
outside the image and are mounted for inference.

Caltech101 images: Li, Andreeto, Ranzato and Perona,
[DOI 10.22002/D1.20086](https://doi.org/10.22002/D1.20086), CC BY 4.0.
Source-label and split audits are automated; no independent human or
pretraining-contamination review is claimed.
