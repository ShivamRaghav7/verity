# Project scope and finite execution plan

Build a local, container-deployable support editing application backed by a frozen
assistant, approved correction memory, a trained applicability classifier and a
trained correction-conditioned generator. Finish with reproducible comparisons,
scientific figures, failure analysis, a report and a panel demonstration. The first
domain covers numeric plan/product facts with four relations. Neither guaranteed
accuracy nor winning every comparison is a completion criterion.

The software slice first demonstrates original answer → administrator correction
→ direct/paraphrased answer → untouched neighboring facts → explicit conflict →
withdrawal and restoration. It runs in structured mode while models are absent.
This makes the governance workflow reviewable without fabricating ML evidence.

## Five-person work breakdown

| Person | Primary responsibility | Concrete outputs and review handoff |
|---|---|---|
| 1: Data and scientific protocol | Independent handbook, entity families, split assignments, label review, truth schema | Dataset card, reviewed pilot data, locked test manifest, scoring specification; person 5 independently reviews challenge wording. |
| 2: Backend and governance | Fact store, correction validation, transactional approval, conflicts, rollback, API | Tested state machine, provenance exports and API documentation; person 1 checks scope semantics. |
| 3: Retrieval and scope ML | Embedding baseline, profiling, pair classifier, calibration, threshold selection | Saved scope weights, retrieval/precision/recall/calibration artifacts, reproducible training config; person 4 checks leakage. |
| 4: Answer ML and base assistant | Frozen handbook assistant, conditional seq2seq training, oracle evaluation, prompt/RAG comparisons | Saved answer weights, generator/oracle metrics, baseline configuration records; person 3 checks conditioning and matched settings. |
| 5: Integration, interface and evidence | Customer/admin UI, comparison/history/conflicts, end-to-end harness, CI/container, report/demo | Working app, exports, scientific plots, latency/resource measurements, failure analysis and report; persons 1–4 review their evidence. |

Every person owns tests and documentation for their work. Share one versioned
schema and avoid separate incompatible notebooks as the primary pipeline.

## A–H milestones and gates

| Stage | Finite work | Acceptance evidence | What is not a release claim yet |
|---|---|---|---|
| A: Contract and data | Finalize schemas/interfaces, initial 80-family split, review protocol and fixed scoring rules | Explicit truth fields, zero cross-split families/lineages, generated manifests and regression tests | Generated labels are not human-reviewed. |
| B: First workflow | One catalogue, administrator correction, historical scope, conflict resolution and withdrawal | Repeatable six-part panel sequence; integration tests and local runnable UI | Structured answers are not learned-model accuracy. |
| C: Base and harness | Load a frozen generator with initial handbook retrieval; implement all five configurations and component logs | Validation pilot runs under matched question/answer budgets; no unsupported fictional prior-knowledge assumption | No final test result or superiority claim. |
| D: Scope model | Profile real checkpoint, train one compact cross-encoder, calibrate on validation, compare embedding routing | Saved checkpoint/config/reviewed dataset hash, finite training log, precision/recall/FAR and coverage curves | New-edit reliability is conditional on domain and measured generalization. |
| E: Answer model | Profile seq2seq checkpoint, train on correction-conditioned answers, evaluate with oracle routing | Saved checkpoint, unseen-edit/entity validation, subject/relation/polarity scoring and failure examples | Correct structured fallback is not counted as generation evidence. |
| F: Integration | Enable learned mode only with checkpoint/calibration registry, connect memory routing, run regression suite | Direct/paraphrase/unaffected/conflict/rollback cases with distinct answer kinds; repeatable version transitions | Arbitrary natural-language scope is not guaranteed. |
| G: Freeze and final evaluation | Lock model/config/scorer/test hashes and thresholds; run once across five configurations | Raw per-question outputs, denominators, paired uncertainty, resource/latency records and disclosed exclusions | Test failures cannot trigger test-guided tuning of this release. |
| H: Presentation | Polish UI, render scientific graphs from results, complete cards/report/demo/container/docs | Clean checkout reproduction, locally deployable app, real saved weights, report and tested demonstration | Resume numbers exist only when backed by recorded results. |

Current delivery implements A's contracts and B's complete structured workflow.
C's five-configuration harness records 600 prediction records per development
run and a separate 48-question oracle diagnostic. D/E completed the initial
user-executed pilot-v2 runs: 360 scope updates and 240 answer updates, with selected
checkpoints at steps 120 and 240. Both models are eligible local trained artifacts;
synthetic optimizer smoke remains ineligible. Training used separately recorded,
user-authorized automated development review of 504 labels, with no independent
human-review claim. [Training](PILOT_TRAINING_RESULTS.md) and
[validation evidence](PILOT_VALIDATION_RESULTS.md) document the actual results.

F integrates the learned editor, trained-artifact provenance and a read-only
validation dashboard while retaining separate deterministic answer labels.
The initial scope recall was 36/48, with all 12 direct-form positives missed.
The user then completed the first bounded scope revision at 360 updates,
selecting step 360. On the exact same pilot-v3 validation pairs, raw accuracy
improves from 56/120 for the original model to 98/120 for the revision and NLL
from 1.477 to 0.486. Raw entity/version matching remains a limitation. Comparing
the original v2 selected loss of 0.391 directly with v3's 0.486 would compare
different negative-question distributions, not identical tasks.

The first [bounded scope-data revision](SCOPE_DATA_REVISION.md) derives pilot-v3
by balancing existing negative wording, preserving the same 504 rows, families,
splits and facts. Fresh automated review, profiling and user-executed training
are complete. Calibration selection previously optimized a raw-pair decision
although runtime rejects incompatible edits first. A separate guard-aware
selector fixes that policy mismatch using the same finite grids, costs and frozen
probabilities. It changes the operating point, not the trained weights. It routes
46/48 applicable pairs, falsely applies 0/72 negatives and clarifies 5/120 pairs,
for 115/120 coverage. These are selection-data routing metrics with deterministic
guards; they are not raw model performance or answer accuracy.

These measurements establish a pilot component with those documented guards.
E's original answer model still receives oracle
credit on 31/48 answers and produces `Yes` on all 12 comparisons requiring `No`.
The current editor's strict automatic answer credit is 30/120. Short baseline
fragments remain pending; zero primary credit does not prove factual failure.
The separate, pre-existing [exploratory short-QA diagnostic](SHORT_QA.md) credits
updated RAG on 91/120 and the learned editor on 80/120; no superiority is claimed.
[Pilot-v3 results](PILOT_V3_RESULTS.md) preserve all denominators and component
attribution. Initial records and the exact historical trainer source from Git
commit `350db38` remain archived.

A subsequent [isolated Nimbus workflow](../reports/pilot-v3-learned-workflow.json)
passes all 31 governance/integration checks while failing to apply an approved
device correction to an unfamiliar product name. The direct question returns
stale `5` instead of `3`; its device paraphrase returns `9 USD per month`; capacity
questions clarify. An oracle-routed direct answer gives the corrected `3`, while
the oracle negative comparison still says `Yes`. These handpicked and repeated
cases localize remaining failures but do not estimate general accuracy.

Stage D has prepared **scope revision 2, the final scope experiment**.
Its one change is the versioned `entity_neutral_v1` input adaptation: selectors
referenced by the correction are canonicalized and the scope descriptor is
normalized. Original metadata remains authoritative for compatibility, dates,
approval and provenance. The same pilot-v3 rows, reviews, backbone, seed,
learning rate and 360-step cap are retained. No labels, facts or splits are
rewritten, no new catalogue is added, and no scope revision 3 is planned.
This new model has not trained and has no measured improvement.

One bounded answer experiment is prepared. The
[comparison-first-token objective](ANSWER_OBJECTIVE.md) emphasizes the initial
`Yes`/`No` decision equally for both polarities without changing facts, labels,
targets or partitions. The released user-run cap is 240 updates, five passes
through 96 training examples. It has not trained and has no measured improvement.
The [released run guide](ANSWER_REVISION_TRAINING.md) binds both commands to
completed guarded profiles and preflight-valid cards. Scope tests transfer to
unfamiliar product names without increasing false applications; answer tests
generated comparison polarity while retaining complete factual answers. The
conditional budgets are 1,331 seconds, 2.0 GiB GPU and 5 GiB storage for scope;
532 seconds, 2.25 GiB GPU and 6 GiB storage for answer. These are profiled planning
estimates, not guarantees. Both support exact-state recovery from the latest
completed checkpoint using the unchanged card and reject completed-run resumes
before ML initialization. This separate answer experiment retains the v2 inputs.

G's independent human-reviewed challenges, final configuration/scorer freeze and
single held-out test run remain pending. H has measured development graphs and
documentation, but the final report and release reproduction are incomplete.

## Remaining finite milestones

1. The user executes the [released 360-step scope and 240-step answer commands](ANSWER_REVISION_TRAINING.md)
   one at a time. Their exact inputs, source-bound resource profiles, preflight
   and recovery checks are prepared. Both full runs are currently untrained;
   no accuracy gain is assumed.
2. Evaluate unweighted loss, first-token polarity, oracle-generated factual
   answers and the same five end-to-end configurations on development questions.
   Also repeat the isolated Nimbus workflow and report its actual generated
   failures separately from governance checks. Select or reject each revised
   artifact from development evidence, calibrate the selected scope policy using
   the existing protocol, then fix both weights. Do not add a third scope run.
3. Fix the final model/configuration, baselines, thresholds, answer budgets and
   scoring rules. Independently review the final held-out labels and manually
   authored unfamiliar-wording challenges, then freeze their hashes before final
   inference. Do not silently extend the development-only automated review claim
   into human-reviewed final evidence.
4. Run one matched final evaluation and report component errors, coverage,
   abstentions, out-of-scope preservation and factual correctness, paired
   uncertainty, latency and resources. Test findings become failure analysis;
   they do not authorize test-guided retraining for this release.
5. Finish the scientific graphs, model/dataset cards, final report and six-part
   demonstration. Verify the locally deployable app, container, regression/CI
   and clean-checkout reproduction, then push the completed release evidence.

## Bound the experiments

Start with a small **reviewed** pilot drawn only from training/validation families.
Pending generated examples may test tensor shapes in a clearly labeled smoke
profile, but they cannot support trained-model or evaluation claims. After a real
checkpoint profile, propose one finite scope run and one finite answer run.

Each substantial run has a written run card containing the question resolved,
input manifest and review hashes, model revision, expected artifacts, measured
steps/second and peak memory, extrapolated runtime with assumptions, checkpoint
interval and resume behavior, maximum steps/epochs, early-stopping rule and disk
budget. The user executes long GPU commands after preparation and verification.

Limit development to an initial run plus at most two targeted validation-driven
revisions per auxiliary model. Predeclare their questions and bounds. If results
remain disappointing, document errors and ship the measured bounded system; do
not keep adding backbones, datasets or infrastructure. A new dataset scale is
justified only by a diagnosed issue after the first reproducible pipeline works.

Checkpoint model, optimizer, scheduler, scaler (if used), RNG state, global step,
configuration and data hashes. Resuming against changed data/config must fail.
No training-time or memory promise is made before profiling.

## Completion checklist and deliverable boundaries

- Tested edit, overlap/conflict, historical scope, revision and rollback behavior.
- Actual trained scope and answer checkpoint artifacts with model/data versions.
- Independent reviewed data with family-first splits and locked final test set.
- Five fair configurations, component diagnostics, abstention-aware denominators,
  paired uncertainty, latency/resource logs and failure analysis.
- Runnable application, isolated environment, CI, container, API docs and exports.
- Scientific plots generated from recorded results, complete report, architecture,
  dataset/model cards, README and six-part panel demonstration instructions.
- Substantial tested improvements committed and pushed to the configured remote.
  If no remote or credentials exist, record the exact missing condition and keep
  the local change reviewable; do not pretend a push succeeded.

Truthful current resume wording: "Built a SERAC-inspired support knowledge editor
with transactional correction memory and rollback; trained compact applicability
and correction-conditioned answer models and evaluated five configurations on a
controlled held-out development population." Describe the deployable structured
container and optional local neural application separately. Numerical improvement
and final-test claims require their corresponding reviewed, frozen evidence.

## Report structure

The final report contains problem and bounded requirements; related primary work;
architectural adaptations; data provenance/review/splits; training and profiling;
fair comparison protocol; measured results with denominators and uncertainty;
component error attribution; limitations and failed cases; engineering tests;
deployment and reproduction; and panel demonstration. Blank result fields remain
explicitly "not measured" until artifacts exist.
