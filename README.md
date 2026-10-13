# EditGuard

**A SERAC-inspired knowledge editing and verification platform for bounded AI
customer support.** Registered research topic: Knowledge Editing.

EditGuard targets support answers that become outdated when product limits or
policies change. An administrator publishes a precise correction; the system
must apply it within the right product, property, plan, version and date scope,
preserve unrelated facts, expose provenance, resolve conflicts and support rollback.

## Current implementation

The first complete **engineering vertical slice** is runnable: customer questions,
before/after comparison, authenticated draft creation and approval, scoped fact
resolution, history, conflict resolution, rollback and JSON export. The application
uses a curated fictional Nimbus handbook. The dataset and products are controlled
demonstrations, not claims about real companies.

**Both auxiliary models completed the user-executed pilot-v2 training runs.**
The compact scope classifier and correction-conditioned answer model have
validation-selected local exports, calibration and integrated inference.
Structured answers remain the default. Optional neural modes implement the
unchanged base, properly updated RAG, prompt-only correction, similarity editor
and trained scope editor. Missing or invalid required artifacts return HTTP 503.
Model binaries are local ignored artifacts; Git contains their configuration,
recorded identities and reproduction tooling. A clean checkout needs verified
exports or reproduced training to enable learned inference.

The initial measured validation uses two held-out entity families and shared
question forms. Scope precision is 36/36 and recall 36/48; all 12 missed
applicable questions are direct requests. The answer model has 31/48 credited
answers with oracle routing, including 12 incorrect `No` → `Yes` endorsements.
Strict complete-assertion scoring leaves short fragments pending and uncredited;
zero credited baseline assertions does not mean zero factual correctness.
An explicitly exploratory request-bound QA analysis credits updated RAG on
110/120 questions and the trained scope editor on 91/120. The learned editor
does not outperform RAG in that diagnostic. These are automated development
measurements, not final-test results or arbitrary-question guarantees.

The editor wraps a frozen retrieval-based assistant with approved memory,
a trained pair scope classifier and a trained edit-conditioned seq2seq generator.
Adding an approved correction will update memory without per-edit retraining.
[Architecture and adaptations](docs/ARCHITECTURE.md) explain why this is
SERAC-inspired rather than a faithful reproduction.

[ML setup](docs/ML_SETUP.md), [matched comparisons](docs/COMPARISONS.md),
[trainer and resume](docs/TRAINER.md), [scoring rules](docs/SCORER.md) and
[data review](docs/DATA_REVIEW.md) document the reproducible pipeline.
The finite pilot contains 360 scope and 144 answer examples across disjoint
training/validation families and no test partition. All 504 labels received an
independent **automated development review**, followed by integrity and target
checks. The separate registry records 504 accepted and zero flagged; no human
approval is claimed. Immutable source rows retain their pending
human-review flags. [Per-row evidence](reports/pilot-v2-independent-review.json)
and four portable human-review pages are linked from the data-review guide.

[Training results](docs/PILOT_TRAINING_RESULTS.md),
[validation and failure analysis](docs/PILOT_VALIDATION_RESULTS.md) and
[exploratory short QA](docs/SHORT_QA.md) disclose denominators, limitations and
all five comparisons. [Pilot-v3](docs/SCOPE_DATA_REVISION.md) repairs the observed
scope question-form imbalance without adding rows or changing facts. It is the
first bounded development revision; no v3 trained-model result exists yet.
Its fresh machine review and four-step resource profile are complete;
[scope revision training](docs/SCOPE_REVISION_TRAINING.md) gives the verified
360-step command, measured planning budgets and exact-state recovery instructions.
An [isolated learned-workflow check](docs/LEARNED_WORKFLOW.md) also exposes a
same-product/different-property false application: a device correction was used
for Nimbus's price. Workflow consistency checks do not imply answer correctness.

![Measured pilot-v2 training loss and resource use](reports/pilot-v2-figures/training-resources.png)

## Run locally

Python 3.12+; the application was tested here with Python 3.14.3 on Windows.
No Node installation, GPU, model download, API key or frontend build is needed
for the first workflow. Production ML dependencies are kept separate.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
$env:EDITGUARD_ADMIN_TOKEN = 'replace-with-your-local-admin-token'
.\scripts\run.ps1
```

Linux/macOS:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
export EDITGUARD_ADMIN_TOKEN='replace-with-your-local-admin-token'
.venv/bin/python -m editguard
```

Open [the application](http://127.0.0.1:8000) or
[interactive API documentation](http://127.0.0.1:8000/docs).
Enter the administrator token in the Correction editor to enable mutation requests.
If no token is configured, the **localhost-only demo default** is `local-demo-only`.
The shared token is a demonstration boundary, not production identity management.

SQLite data persists at `runtime/editguard.sqlite3`. Restarting is idempotent;
seeding never reactivates withdrawn corrections. No answer cache is used, and
revision-consistent snapshots keep responses and exports coherent with history.
Server binds to localhost; public deployment requires proper authentication.

With the isolated [ML environment](docs/ML_SETUP.md), cached pinned backbones and
verified local pilot-v2 exports present, start the optional learned application:

```powershell
$env:EDITGUARD_INFERENCE_CONFIG = 'configs/inference-pilot-v2.json'
$env:EDITGUARD_EVALUATION_REPORT = 'reports/pilot-v2-evaluation.json'
.\.venv-ml\Scripts\python.exe -m editguard
```

The Evaluation lab loads recorded validation diagnostics alongside a separate
deterministic regression. It performs no training or model evaluation on refresh.
It shows checkpoint selection, automated-review provenance, full denominators,
pending fragments, calibration, oracle answers, matched configurations and
exportable evidence. Retrieved sources do not verify generated prose.

## Panel demonstration

1. Select Nimbus / Basic / v2 and date 2026-10-11. Ask “How many devices does Basic
   support?” The original value is five.
2. In Corrections select the device fact, original value `5`, corrected value `3`,
   effective date `2026-10-01`, and source “Fictional release notice EG-001”. Save
   a draft, inspect it, then explicitly approve it from History.
3. Ask the direct question and “What is Basic's device limit?”. Ask “Can I connect
   four devices on Basic?”. Structured answers show three and “No”, with provenance.
4. Ask Basic's monthly price, switch to Pro's devices, then Basic v1. They retain
   nine USD/month, twelve devices and five devices respectively. Set date before
   the correction to observe the initial value in v2.
5. Save an overlapping competing draft using original `3`, corrected `4`, same
   scope/date, without supersession. Approval is rejected; the affected question
   exposes the conflict. Withdraw the competing draft in Conflicts.
6. Withdraw the approved edit in History. Basic v2 returns to five devices; audit
   events remain. A withdrawn edit is immutable; a new reviewed draft can restore it.

Supported structured grammar intentionally covers short factual questions and
“Can I connect four devices on Basic?”. Unsupported wording, unknown tiers,
comparative capacity phrases and ambiguous scopes clarify. General natural-language
capability is limited by the measured pilot results; it is not inferred from this
structured demonstration. Switch answer modes to inspect real generated outputs
and their routing, including documented scope and comparison-polarity failures.

## Tests and reproducibility

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check editguard/static/app.js
.\.venv\Scripts\python.exe scripts/prepare_data.py --output data/generated/controlled-v2
```

GitHub Actions defines the Python regression suite, JS syntax check, a separate
CPU optimizer/resume regression job and a Docker workflow/restart verification job.
The optimizer job uses tiny synthetic models without pretrained downloads.
GPU backbone smoke remains a separate local check. Tests cover
authority, exact scope, historical dates, half-open intervals, conflicting drafts,
supersession, rollback, persistence, concurrent revisions, API token checks,
leakage constraints and hash-bound review/freeze gates. The dashboard runs an
isolated engineering fixture and never mutates live memory. Its optional ML
summary is read-only recorded validation evidence, separate from those checks.

The independent data pipeline starts with 80 fictional entity families, 4,800 scope
pairs and 1,920 short answer examples. It splits
families before augmentation, keeps all variants together, and generates a human
review queue. Training rejects unresolved or stale review records and defaults to
human review. The explicitly selected `automated_development` policy preserves
machine-review provenance in development runs; final-test locking still requires
human review. Generated labels are not automatically accepted as truth.
[Dataset card](docs/DATASET_CARD.md) describes counts, provenance and limitations.

## Training and evaluation stages

The observed host has an RTX 4050 Laptop GPU with 6 GB VRAM, 23.71 GiB RAM and
12 logical CPU threads. [Hardware inventory](reports/hardware.json) records the
actual inspected environment. The initial DistilBERT and FLAN-T5-small profiles
passed at batch size 2, with peak GPU allocation of 1.27 and 1.46 GiB respectively.
Profiling does not create a trained checkpoint. See the exact shapes, revisions and
limitations in [profiling evidence](docs/PROFILING.md).

`scripts/profile_models.py` profiles at most ten optimizer steps with bounded
batch/length settings; defaults require cached weights. It writes timing, peak
CUDA memory, package versions and model revision. It never reads the test set.
The first measured profile uses the existing host ML installation, separately
from the isolated application environment; full training must use its own pinned
environment and data/config hashes.

[Training preflight](docs/TRAINING_PREFLIGHT.md) validates reviewed inputs and
finite run cards. The separate [trainer](docs/TRAINER.md) implements training and
checkpoint restoration; preflight alone does not verify a completed training run.

[Initial pilot training instructions](docs/PILOT_TRAINING.md) provide the concrete
questions, reviewed inputs, measurements, conditional runtime/memory estimates,
artifacts and exact-resume rules. Those initial user-executed runs completed at
360 scope updates and 240 answer updates; selected checkpoints are steps 120 and
240 respectively. [Actual training measurements](docs/PILOT_TRAINING_RESULTS.md)
replace the planning estimates for these runs. Four-step profiles remain
ineligible as trained deployment artifacts. The v3 revision requires its own
reviewed inputs, bounded run card and profile-based estimates before execution.

[Measured embedding-baseline curves](docs/SIMILARITY_RESULTS.md) show raw and
compatibility-gated behavior on 120 development validation pairs, with explicit
denominators and the missed question. They are threshold-selection evidence,
not final-test or generated-answer performance.

The five comparison configurations are unchanged base, properly updated RAG,
prompt-only corrections, embedding retrieval corrections and trained scope-aware
editing. All updated baselines receive the same approved memory and context.
[Evaluation protocol](docs/EVALUATION_PROTOCOL.md) separates retrieval, raw scope,
oracle-routed generation and end-to-end quality, keeps abstentions in denominators,
and scores subject, relation, endorsed value and polarity. No LLM-only judge is used.

## Container

```powershell
$env:EDITGUARD_ADMIN_TOKEN = 'replace-with-your-admin-token'
docker compose up --build
```

The container runs the same structured application as a nonroot user, with a
persistent SQLite volume and localhost port mapping. Container definitions are
provided; a Docker build is not claimed unless recorded in verification evidence.
GPU training stays outside this lightweight application image.

The structured image was built and verified locally with Docker 29.2.1,
Python 3.12.15 and UID 1000. All 23 isolated HTTP lifecycle checks passed, as did
three restart/persistence checks. The verification container is stopped; its
dedicated volume and image remain available. [Deployment evidence and reproduction](docs/DEPLOYMENT.md)
distinguish this structured image from optional local neural inference.

## Project records

- [Five-person roles and finite A–H milestones](docs/PROJECT_PLAN.md)
- [Architecture](docs/ARCHITECTURE.md) and [interface contract](docs/IMPLEMENTATION_CONTRACT.md)
- [API behavior](docs/API.md)
- [Dataset card](docs/DATASET_CARD.md) and [model card](docs/MODEL_CARD.md)
- [Evaluation protocol](docs/EVALUATION_PROTOCOL.md)
- [Failure analysis](docs/FAILURE_ANALYSIS.md) and [interim report](docs/REPORT.md)
- [Measured profiling](docs/PROFILING.md) and [structured HTTP samples](reports/structured-http.json)

Stages A/B and the baseline harness are implemented; D/E completed their initial
real training and validation runs, and learned inference is integrated. Scientific
graphs and an honest development report are available. Final independently
reviewed challenges, frozen-test evaluation, broader uncertainty evidence and
the polished final report remain pending. Resume claims can describe the models
trained and five configurations evaluated; no superiority or final numerical
improvement claim follows from this pilot.
