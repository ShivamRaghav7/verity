# VisionPatch

**Targeted visual model editing with measured generalization and exact rollback.**

VisionPatch changes a learned visual association inside CLIP: a misleading printed
word can make the model misidentify an object. A small learned projection patch
repairs that behavior on new photographs without changing the frozen encoders.
Registered area: Knowledge Editing; specifically **visual model / concept editing**.

This branch contains a correction-authoring application and a completed,
fixed-configuration selectivity study. It does not generate captions or unrestricted visual answers.
The preserved customer-support EditGuard V1 remains at the `editguard-v1.0.0` tag;
its original instructions are saved in [the V1 README](docs/EDITGUARD_V1_README.md).

## V2: selective repair and interacting corrections

Open **http://127.0.0.1:8002/workbench** after starting the server. Supply a failing
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

## Preserved V1 results

Fixed vocabulary: eight object categories. Printed distractor: `camera`.
Four cup/laptop support pairs fit the patch; 64 separate clean reference images
provide retention statistics. The final set contains 80 additional original
photographs, excluded by content hash from all 144 development images.

| Configuration | Final printed-label accuracy | Final clean accuracy |
| --- | --- | --- |
| Original CLIP | 49/80 | 80/80 |
| Learned projection edit | 80/80 | 80/80 |
| Projection fine-tuning | 80/80 | 80/80 |
| Oracle label masking | 80/80 | 80/80 |

The editor repaired 31/31 induced errors in this particular holdout and preserved
10/10 genuine camera predictions. **Fine-tuning and privileged masking tied it.**
These small, controlled results are not a guarantee for other words, fonts,
objects, candidate lists or deployment conditions. A development failure remains
in the report; test results were not used to modify the configuration.

![Frozen holdout comparison](output/visionpatch/figures/accuracy.png)

Read [the beginner guide and website instructions](docs/visionpatch/README.md),
[measured results and limitations](docs/visionpatch/RESULTS.md), and
[the evaluation protocol](docs/visionpatch/PROTOCOL.md).

## Run the prepared demonstration

```powershell
.\scripts\run-visionpatch.ps1
```

Open http://127.0.0.1:8002/workbench for the persistent V2 studio.
The original gallery remains at http://127.0.0.1:8002. Approve **Apply learned patch**, inspect the gallery,
compare all baselines in the frozen-holdout dashboard, try an uploaded photograph,
and use **Rollback** to restore the original projection. API documentation:
http://127.0.0.1:8002/docs. Stop the server with Ctrl+C.

The gallery uses cached frozen-encoder features. Uploaded images run the actual
pinned encoder. The eight candidate similarities are uncalibrated scores; an
unknown object still receives one of those candidates. Activation history is
session-local in the original gallery; the V2 studio uses a separate persistent
registry. This loopback demo has no production authentication.

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

Use unused report/artifact paths for repeat experiments. The pilot stops after
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
`python -m visionpatch.verify_http` exercises real image inference and restoration;
`python -m visionpatch.plot_results` regenerates scientific PNG/SVG figures.

The measured CPU HTTP sample has p50 243 ms and p95 327 ms, from 12 serial requests
for one development image. First model loading took 11.74 seconds in that check.
This is a small latency sample, not a concurrency benchmark.

A CPU container recipe is provided in `Dockerfile.visionpatch`; see
[container instructions](docs/visionpatch/DEPLOYMENT.md). GitHub Actions checks the
original application and the new visual components. Large data/model files stay
outside the image and are mounted for inference.

Caltech101 images: Li, Andreeto, Ranzato and Perona,
[DOI 10.22002/D1.20086](https://doi.org/10.22002/D1.20086), CC BY 4.0.
Source-label and split audits are automated; no independent human or
pretraining-contamination review is claimed.
