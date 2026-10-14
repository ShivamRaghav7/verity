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

Current delivery implements A/B's contracts, complete structured workflow and
governance tests. C's five-configuration harness records 600 predictions for
each 120-question development run, plus 48 oracle-routed answers. D/E completed
the initial pilot runs, two finite scope revisions and one finite answer
revision. F integrates eligible trained exports, calibrated routing, provenance
and distinct answer kinds with the customer/admin application. All pilot
reviews are independently recorded automated development decisions; source
human-review flags remain pending and no human approval is claimed.

The selected scope model is `pilot-v3-neutral2`, selected at update 360; the
selected answer model is `pilot-v2-polarity1`, selected at update 240. Their
independent artifact audit passes 986 checks. On the same 120 pilot-v3 validation
pairs, raw scope accuracy improves from revision 1's 98/120 to 119/120. The
calibrated application policy with compatibility guards applies 48/48 applicable
corrections, falsely applies 0/72 negatives and clarifies one question. These are
validation-selection routing measurements, not guaranteed new-question accuracy.
Entity-neutral preprocessing produces eight identical same-label pairs across
raw disjoint training/validation families; shared templates and two validation
families limit generalization claims.

Holding the selected scope fixed, complete oracle answer credit improves from
31/48 to 34/48, and complete end-to-end fact credit from 31/120 to 34/120.
The existing separate short-QA diagnostic improves from 83/120 to 86/120; the
updated RAG baseline remains stronger at 91/120. The answer model still fails
seven of twelve negative comparisons, and its comparable unweighted and weighted
token losses became worse. All controls were rescored under the same versioned
scorer, preserving original outputs and historical scores. No superiority or
arbitrary natural-language reliability is claimed. The
[selected revision record](TARGETED_REVISION_RESULTS.md) contains exact hashes,
counts, measured resources, controls and limitations. Earlier
[pilot-v3 failures](PILOT_V3_RESULTS.md) and the isolated Nimbus workflow that
motivated revision 2 remain archived; passing governance checks never establish
generated-answer correctness.

Training and development model selection are closed for this release. No third
scope run, second answer experiment, larger training catalogue or new backbone
is planned. The final configuration selects these two trained artifacts and the
existing calibrated policies. Stage G now locks model/configuration/scorer hashes,
completes a separately recorded independent **automated** final-label audit
authorized by the user, freezes the evaluation packet before inference and runs
one matched final evaluation. This is an explicit change from the earlier human
review gate; it must be disclosed as machine review in every final artifact.
Human semantic review is absent. H's final report, figures, release reproduction
and panel demonstration remain incomplete until that evidence exists.

## Remaining finite milestones

1. Fix the selected model/configuration, baselines, thresholds, answer budgets and
   scorer. Independently audit final held-out labels and authored unfamiliar-wording
   challenges under the explicit user-authorized automated final policy, then
   freeze their hashes before final inference. Record machine review separately
   from development review, with no human-reviewed evidence claim.
2. Run one matched final evaluation and report component errors, coverage,
   abstentions, out-of-scope preservation and factual correctness, paired
   uncertainty, latency and resources. Test findings become failure analysis;
   they do not authorize test-guided retraining for this release.
3. Finish the scientific graphs, model/dataset cards, final report and six-part
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
