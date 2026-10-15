# EditGuard: interim project report

**Status: both auxiliary models are trained and integrated; the completed scope
revision and guard-aware calibration are evaluated on the controlled pilot.
One bounded answer-polarity experiment and the second, final scope experiment
are prepared with profiled, preflight-valid commands, not trained.
Final independently reviewed evaluation remains pending.**

## Problem and registered topic

Knowledge Editing is the registered topic. Product support systems become stale
when limits or policies change. EditGuard aims to publish corrections without
per-correction retraining while maintaining precise scope, provenance and rollback.
The initial task is short factual English questions over a controlled fictional
product catalogue; unrestricted dialogue and dependency reasoning are deferred.

## Technical approach

The architecture is SERAC-inspired: a frozen base assistant is wrapped by approved
external memory, a learned applicability classifier and a correction-conditioned
generator. Our governance and temporal compatibility rules are adaptations. See
[architecture](ARCHITECTURE.md) for the primary-paper reading and full diagram.
The prototype exercises those governance rules with separately labeled structured
answers and real trained neural editing. Updated RAG applies approved facts before
retrieval; prompt-only correction receives the same approved descriptors.
[Comparison contracts](COMPARISONS.md) disclose routing and benchmark boundaries.

## Data and protocol

The independent fictional training catalogue is separate from the ten-fact Nimbus
UI fixture. Entity families are partitioned before questions and counterfactual
values are augmented. Typed, hash-bound review controls eligibility; human review
is the default. The user authorized an independent automated development review
of the corrected 504-row pilot: 504 accepted, zero flagged, with an additional
144-target assertion check. The immutable source flags remain pending human
review; the separate registry explicitly discloses machine review. Pilot-v3
balances the existing negative question forms, retains all 504 rows, facts and
partitions, and has fresh automated decisions bound to its new row identities.
All development records disclose `human_reviewed_data=false`. The test partition
is not frozen. See [dataset card](DATASET_CARD.md),
[current per-row review](PILOT_V3_REVIEW.md) and [evaluation protocol](EVALUATION_PROTOCOL.md).

## Engineering evidence

Regression covers approval, conflicts, half-open dates, explicit supersession,
rollback, neighboring scope retention, persistence, authority checks and coherent
concurrent reads. The live dashboard exposes validated trained-artifact metadata,
automated-review provenance and read-only recorded validation results. It retains
separate isolated workflow checks, denominators and fixture service latency.
Engineering checks demonstrate database/API behavior; they are not learned-model
accuracy. Changing loaded model or calibration dependencies requires restart
rather than silently associating stale inference with new artifact metadata.

The new trainer implements both pretrained auxiliary tasks, fixed stopping caps,
validation-loss selection and atomic full-state resume with data, configuration,
environment, source-code and pretrained-state identities. The scorer rejects
wrong subjects/relations/polarity, contradictory assertions and unendorsed target
mentions. Unsupported fragments enter a blinded hash-bound review queue and remain
uncredited until adjudicated. [Scorer rules](SCORER.md) retain abstentions and
missing predictions in denominators and separate structured from generated evidence.

The UI provides customer questions, scoped before/after comparisons, an approval
workflow, history, conflict resolution and exports. Minimal infrastructure consists
of FastAPI, Pydantic, SQLite and build-free static assets. The core regression and
CPU optimizer/resume CI jobs passed for commit `c89e528`; the source record is
`reports/ci-c89e528.json`. Deployment CI execution is recorded separately.
All three jobs also passed for `6f32ee4`, including container lifecycle and restart
verification; [that CI run](https://github.com/ShivamRaghav7/verity/actions/runs/38149158719)
is distinct from the current local changes.

The complete workflow was exercised in a separate browser fixture, including
conflict rejection and restoration. Mobile layout had no horizontal overflow and
the browser reported no console errors. Docker Desktop's Linux engine later started
successfully. The structured image passed 23 isolated HTTP workflow checks and
three checks after restart, preserving revision 7 and both withdrawn corrections.
It ran as UID 1000 on Python 3.12.15 with a dedicated volume. See
[deployment records](DEPLOYMENT.md); these checks make no neural-deployment claim.

## Initial profiling observations

Real DistilBERT and FLAN-T5-small checkpoints completed one warmup and three
measured float32 optimizer steps at batch 2/input length 192 (answer length 48).
Peak allocated GPU memory was 1.274 and 1.462 GiB respectively. Synthetic shapes
and short runs establish feasibility only; no checkpoint or accuracy result was
produced. [Raw profiling and limitations](PROFILING.md) include resolved revisions.

A separate 30-request engineering benchmark of the structured app observed
HTTP p50 15.592 ms and p95 19.616 ms, with zero errors/abstentions in the selected
supported forms. This result includes HTTP/JSON overhead and excludes ML inference.

Both actual auxiliary trainers subsequently completed four real CUDA steps through
an interrupted two-step segment and resumed two-step segment. The separate CPU
regression matched uninterrupted/resumed parameters, losses, selection and RNG.
These synthetic checkpoints are explicitly rejected by the runtime. Actual
timings, memory and checkpoint I/O are in [TRAINER_SMOKE.md](TRAINER_SMOKE.md).
They do not establish a full-training runtime estimate.

The frozen neural smoke completed six questions across the three available
baselines, preserving explicit unavailability of the two editors. Base parameter
fingerprints remained identical. [NEURAL_SMOKE.md](NEURAL_SMOKE.md) records wrong
answers and uncredited fragments, with all raw inputs and outputs retained.
No result in that engineering fixture is a final accuracy claim.

Actual four-step pilot profiles now include the full reviewed validation partitions
and checkpoint writes. Peak allocations were 1.318 GiB (scope) and 1.508 GiB
(answer); checkpoint operations took 8.22–9.62 and 10.27–17.07 seconds.
[Finite run cards](PILOT_TRAINING.md) stop at 360/240 updates, with conditional
planning estimates of 441/432 seconds. These are estimates, not measured full
training times. Profile artifacts remain ineligible for deployment.

The frozen MiniLM baseline selected cosine threshold 0.46 using all 120 reviewed
validation pairs: 48 positive and 72 negative. Without compatibility gates it
applied 47 correct and 48 incorrect corrections; with gates it applied 47 correct
and zero incorrect corrections, missing one positive. Sixty incompatible pairs
were rejected by metadata. This is validation selection on single focal pairs,
not test performance, full-memory retrieval or generated-answer accuracy. Raw
score/grid artifacts are in `reports/similarity-pilot-v2`; no trained-auxiliary
superiority claim follows from this result.

![Structured application HTTP latency](../reports/structured-http-latency.png)

![Measured four-step synthetic trainer resources](../reports/trainer-smoke-resources.png)

![Similarity routing on 120 development validation pairs, with explicit denominators and scope gates](../reports/similarity-pilot-v2/operating-points.png)

[The baseline result record](SIMILARITY_RESULTS.md) documents the missed
paraphrase, raw/gated behavior and selection limitations. It makes no generated
answer or final-test claim.

## Experiments and results

| Required result | Current evidence |
|---|---|
| Trained auxiliary weights | Initial scope/answer runs completed at 360/240 updates; current scope-v3 revision also completed 360, selecting step 360; original answer selected step 240 |
| Training time and memory | Scope-v3 loop 43.402 s, allocated/reserved CUDA peaks 1.318464/1.410156 GiB; initial answer loop 245.856 s, allocated peak 1.508413 GiB; complete command wall time unavailable |
| Applicable-edit retrieval recall | Raw unfiltered retrieval 47/48 at rank 1 and 48/48 at rank 3; eligible routing is a separate measure |
| Raw scope model on common v3 pairs | At 0.5, original accuracy 56/120 and NLL 1.477; revised accuracy 98/120 and NLL 0.486, AUROC 0.864; recall 46/48, precision 46/66, false positives 20/72 |
| Validation-selected scope with guards | Temperature 0.5, threshold 0.47, margin 0.025; 46/48 positive corrections, 0/72 false applications, five clarifications, 115/120 coverage; selected on the reported validation pairs |
| Oracle generation | Strict and exploratory QA credit 31/48; 12 incorrect comparison endorsements and five pending answers |
| Five matched configurations | 120 questions each, 600 predictions including five clarifications; strict learned-editor credit 30/120; separate exploratory QA learned 80/120, updated RAG 91/120 |
| Out-of-scope retention/correctness | Primary strict baseline-correct retention undefined at 0/0; exploratory QA credit 50/72 learned and 53/72 base/RAG/similarity; learned retains 50/53 baseline-QA-correct answers, with three clarifications |
| Latency and uncertainty | Serial CLI p50/p95 recorded; two-family cluster bootstrap is insufficient for precise population claims |
| Conflict/rollback consistency | Separate deterministic/deployment checks and isolated learned-workflow evidence |
| Final frozen-test results | Not available; final review and configuration/scorer/test freeze remain pending |

These measurements establish real training and a reproducible development
comparison. They do not establish superiority or final-project completion.
[Training results](PILOT_TRAINING_RESULTS.md) disclose run identities, resources,
selection and the 87-check artifact audit.
[Archived initial validation](PILOT_VALIDATION_RESULTS.md) retains the first
measurements. [Current pilot-v3 results](PILOT_V3_RESULTS.md) disclose completed
scope training, identical-input model comparison, calibration selection,
all raw records, denominators and failure attribution. The historical trainer
sources from Git `350db38` are preserved in
[`reports/training-source-v1`](../reports/training-source-v1/manifest.json).

The primary scorer requires a complete factual assertion. On pilot-v3 it credits
30/120 learned-editor and 31/120 similarity-editor outputs; incomplete baseline
fragments remain pending and uncredited. Zero baseline credit is not evidence
of zero baseline factual correctness. An explicitly
[post-hoc short-QA diagnostic](SHORT_QA.md) binds supported fragments to the
requested subject/relation and credits properly updated RAG on 91/120 questions
and the trained scope editor on 80/120. Updated RAG leads that exploratory
comparison; learned superiority is not claimed. Structured/database answers
are never credited as learned generation.

The answer model makes 12 `No` → `Yes` mistakes even with oracle-correct routing,
despite including the correct limit. Routing cannot explain these polarity
failures. The initial scope model missed all 12 direct positives. The original
pilot gave negative labels only to direct questions, creating a wording shortcut. The first
[bounded revision](SCOPE_DATA_REVISION.md) rotates those existing negatives
across direct, paraphrase and yes/no comparison forms without adding rows or
changing facts, splits or lineages. The user completed its 360-step run. When both
exports are evaluated on the same v3 questions, NLL decreases from 1.477 to 0.486,
raw accuracy increases from 56/120 to 98/120 and all 12 direct positives are
detected at 0.5. Wrong-entity and wrong-version pairs account for all 20 raw false
positives; metadata guards reject those pairs. The original v2 selected loss of
0.391 used different negatives, so comparing it directly with 0.486 would be
misleading. These measured development changes do not rewrite the archived v2
results or access the test/challenge corpus.

The original calibration selector minimized operating cost across raw pairs,
although runtime rejects incompatible edits first. Including the existing
compatibility gate in a separate selector corrects that mismatch without fitting
new weights or expanding the finite search grid. Its selected policy routes
46/48 applicable edits, falsely applies 0/72 negatives and clarifies 5/120 pairs.
All five clarifications remain uncredited in answer accuracy. Scope selection
and its reported validation results share data, so these are optimistic
development measures, not independent final-test estimates.

The answer diagnostic reproduces mean validation token NLL near 0.044412 while
finding first-token `No` correctness of only 2/24 on training and 0/12 on
validation. For validation questions requiring `No`, first-token NLL is 1.0646
versus 0.02154 for remaining tokens. A low teacher-forced mean can obscure the
incorrect initial decision because the following fact tokens are easy to copy.
This supports one bounded objective experiment without proving the sole cause
of the error. [The objective specification](ANSWER_OBJECTIVE.md) gives equal
8x emphasis to the first token of reviewed `Yes` and `No` comparison targets,
leaving all other valid tokens at weight one and padding at zero.

That answer revision is prepared and **has not trained**. Its finite cap is
240 updates, five passes through the same 96 training rows. The
[released run guide](ANSWER_REVISION_TRAINING.md) provides the measured guarded
profile, exact inputs and artifacts, and checkpoint/resume behavior. Its question
is whether equal extra emphasis on the opening Yes/No token improves generated
comparison polarity while preserving factual answers. The conditional planning
budget is 532 seconds, 2.25 GiB GPU and 6 GiB checkpoint/export storage. This is
an estimate from four updates, not a full-run measurement or guarantee.
Weighted selection loss is not comparable to the initial
uniform loss and is not accuracy. Unweighted loss, first-token polarity, oracle
generation and the full matched comparison must determine whether the result
is retained; no improvement is promised.

An archived [initial learned-workflow check](LEARNED_WORKFLOW.md) exposed another locality
failure: Nimbus's price question selected a device correction and generated
`Nimbus Basic version v2 costs 3 devices.` A four-device comparison also produced
yes when no was expected. These actual generated failures remain separate from
passing governance/integration checks. The pilot's 0/72 out-of-scope edit selections
cannot be generalized to unseen products, properties or questions. This fixture
used the initial scope artifact and is not a new v3 workflow measurement.

A subsequent [scope-v3 Nimbus workflow](../reports/pilot-v3-learned-workflow.json)
uses the actual completed revision and passes **31/31 governance/integration
checks**. It nevertheless misses the approved device correction on the direct
question, returning stale `5` instead of `3`; a device paraphrase returns
`9 USD per month`, and capacity questions clarify. With oracle routing the answer
model produces the correct direct `3`, but its negative comparison still says
`Yes`. These are handpicked and partly repeated engineering cases, not a general
accuracy estimate. Passing transactional approval and rollback checks does not
make those generated answers correct.

This evidence motivates **scope revision 2, the final scope experiment**. Its
single adaptation is the versioned `entity_neutral_v1` classifier input policy:
correction-referenced product/alias, plan and version selectors become consistent
phrases, and the descriptor is normalized while relation, values and units remain
visible. Original metadata still governs compatibility, approval, dates and
provenance. It keeps the same pilot-v3 rows and reviews, backbone, seed, learning
rate and 360-step cap. Facts, labels and partitions are not changed. The model
has not trained. Its source-bound guarded profile and released card passed
preflight; the conditional planning budget is 1,331 seconds, 2.0 GiB GPU and
5 GiB checkpoint/export storage. The [released guide](ANSWER_REVISION_TRAINING.md)
states the question, artifacts and recovery contract: use the unchanged card
and latest completed checkpoint; completed exports or checkpoints at the cap
are refused before ML initialization. It is separate from the first weighted
answer revision, which keeps the v2 inputs and 240-step cap. Neither has measured
improvement. Final evaluation and the polished release report remain pending.

![Scope models compared on the same pilot-v3 pairs before deterministic guards](../reports/pilot-v3-figures/scope-diagnosis.png)

The following plots remain archived initial pilot-v2 evidence; they are not the
current scope-v3 metrics.

![Archived pilot-v2 strict outcomes, with pending semantic review disclosed](../reports/pilot-v2-figures/validation-outcomes.png)

![Validation-selected scope calibration and operating points](../reports/pilot-v2-figures/scope-calibration.png)

![Actual pilot-v2 training loss and resource use](../reports/pilot-v2-figures/training-resources.png)

The figure manifest binds all plotted inputs and output hashes. The dashboard
renders a hash-bound summary against the configured model identities and permits
JSON export; refresh loads recorded evidence rather than rerunning inference.
The final report still requires independent final review, fixed configuration
and scoring rules, manually authored unfamiliar-wording challenges and a single
matched held-out test evaluation with honest uncertainty and failure disclosure.

## Limitations and next acceptance gates

Current language coverage is deliberately narrow. A correct source does not
guarantee a generated answer endorses it. Scope ambiguity, unfamiliar wording,
token shortcuts and temporal errors need component diagnostics. See
[failure analysis](FAILURE_ANALYSIS.md) for concrete findings.

The finite [A–H plan](PROJECT_PLAN.md) lists ownership, run-card requirements,
maximum development revisions and full completion criteria. Complete the second,
final scope experiment and the first prepared answer experiment, evaluate
component quality and repeat the isolated workflow, then select or reject each
from development evidence. Fix the model/baseline/scorer protocol, independently
review and freeze
the final held-out questions/challenges, then run the five-way final test once.
Retain failures rather than tune against that test. Finish the report, graphs,
demo, locally deployable application and CI reproduction. No third scope run or
open-ended model search is planned. This interim artifact
does not mark the ML-focused project complete.
