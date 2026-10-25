# Project scope and finite execution plan

This document preserves the earlier support-editor release and its evidence.
For the current ReasonKE conversational product, use the
[project journey and finish plan](PROJECT_JOURNEY_AND_FINISH_PLAN.md).

Build a locally deployable support editing application with a frozen assistant,
approved correction memory, a trained applicability classifier and a trained
edit-conditioned generator. Deliver reproducible comparisons, scientific figures,
failure analysis, a report and a panel demonstration. The bounded domain has four
numeric product/plan relations. Guaranteed accuracy and winning every baseline
are not completion criteria.

Both auxiliary models are trained and selected. Model search is closed. The
matched frozen final comparison is sealed; original oracle generation remains
incomplete due to disclosed runtime errors. Its separately declared recovery is
complete after GPU availability was restored, preserving the original record.
The finite A–H implementation, model training, evaluation and presentation work
are complete, including the fresh learned-HTTP measurement. All three release CI
jobs passed on source commit `2c7938c`, and the release is signed off. The structured app,
recorded dashboard, figures, report and CPU regression/CI evidence remain separate
from the operational supplement.

## Five-person work breakdown

| Person | Primary responsibility | Concrete output and handoff |
|---|---|---|
| 1: Data and scientific protocol | Fictional handbook, family-first splits, truth schema, label audit and fixed evaluation protocol | Dataset card, versioned pilot/final packets, review provenance and test lock; person 5 checks challenge authorship and semantic-review limits. |
| 2: Backend and governance | Fact store, correction validation, transactional approval, conflicts, rollback and API | Tested state machine, history/provenance exports and API documentation; person 1 checks scope semantics. |
| 3: Retrieval and scope ML | Embedding baseline, profiling, pair classifier, calibration and threshold selection | Saved scope export, retrieval/precision/recall/calibration/coverage evidence; person 4 checks leakage and deterministic adaptations. |
| 4: Answer ML and frozen assistant | Initial-handbook base, conditional training, oracle diagnostics and prompt/RAG comparisons | Saved answer export, generator failures and matched baseline records; person 3 checks conditioning and shared budgets. |
| 5: Integration, interface and evidence | Customer/admin UI, history/conflicts, harness, CI/container, report and demo | Runnable app, exports, plots, latency/resources, failure analysis and report; persons 1–4 review the interpretation of their evidence. |

These are team responsibilities, not claims that five named people independently
reviewed every row. Actual pilot and final review are typed automated decisions,
explicitly authorized by the user. There is no independent human-review or
human-authored-challenge claim. Every contributor owns meaningful tests and
documentation; shared schemas and runnable scripts are the primary pipeline.

## A–H milestone status

| Stage | Finite scope | Current evidence and status |
|---|---|---|
| A: Contract and data | Explicit schemas, family splits, review and acceptance rules | Implemented. Facts/edits encode identity, relation, values, plan/version/dates, provenance, approval and lineage. Raw families/entities/lineages are disjoint; transformed-input overlap is separately disclosed. |
| B: First workflow | One correction, appropriate questions, neighboring facts, conflict and rollback | Implemented and tested. Structured workflow and historical state are distinct from generated-answer quality. Local container evidence records 23/23 lifecycle and 3/3 restart checks. |
| C: Base and harness | Frozen initial-handbook generator, all five fair configurations and component logs | Implemented. Updated RAG materializes approved changes; updated methods receive the same corrections/context and compatible generation budgets. |
| D: Scope ML | Profile, train compact cross-encoder, embedding baseline and validation calibration | Complete for the selected release. Final scope export `pilot-v3-neutral2`, 360 updates, selected step 360; preprocessing/guards are declared adaptations. |
| E: Answer ML | Profile, train correction-conditioned seq2seq, free-generation validation and failure diagnosis | Training/development complete. Selected `pilot-v2-polarity1`, 240 updates, selected step 240. Original final oracle runtime errors remain disclosed; separately named fixed-configuration recovery is complete. |
| F: Integration | Artifact/calibration gates, learned routing, memory transitions, provenance and answer-kind separation | Implemented. Actual selected-model isolated workflow passes 31/31 state/routing checks; its factual diagnostics are not 31 correct neural answers. |
| G: Frozen evaluation | Lock models/config/scorers/questions, audit final labels, one matched evaluation | Matched comparison sealed: 4,080 records, 4,009 answers, 71 clarifications, zero runtime errors. Original oracle includes 319 CUDA errors. Its separate recovery completes all 319 after GPU restoration; the combined 384-case view retains 65 original answers. Original scores remain unchanged. |
| H: Release presentation | Final report/PDF, scientific figures, browser/demo verification, clean-checkout/CI evidence | Complete. The 13-page report and six landscape scientific figures are rendered and reviewed; dashboard/downloads verified. Local release checks pass 516 tests with 18 optional skips and 34 separate CPU numeric checks. Oracle recovery and learned-HTTP evidence are incorporated. All three release CI jobs passed on source commit `2c7938c`. |

## Recorded scientific outcome

The selected models train on four entity families and validate on two others:
scope 240/120 rows, answer 96/48 rows. The original 80-family draft was not all
used for training. All 504 pilot labels have exact-row-hash-bound automated
development review. The selected export audit passes 986 checks. Actual training
loops took 112.55 seconds for scope and 264.54 seconds for answers on the inspected
RTX 4050 laptop; full-command startup/export time is not recorded.

The final 12-family packet contains 720 standard questions, 288 oracle answers
and 96 agent-authored challenges, with 1,104 automated-final audit decisions.
It is frozen and training-ineligible. Nine scope inputs overlap training after
literal abstraction; 711 do not, and none overlaps validation. Shared templates
still limit novel-language claims.

On the frozen standard request-bound metric, the five configurations obtain
500/720 base, 649/720 updated RAG, 533/720 prompt-only, 632/720 similarity and
613/720 trained editor. Updated RAG leads. The paired trained-minus-RAG difference
is -5.00 percentage points, 95% family-bootstrap interval [-6.67, -3.06]. All
23 learned standard clarifications and 33 additional pending outputs remain in
the denominator. The editor's raw scope classification is 701/720, while runtime
observes zero false edit applications out of 432 out-of-scope cases after guards.
Its out-of-scope correctness is 405/432 and baseline-correct retention 405/428;
these are different measures.

The editor answers only 48/96 challenge questions. Their wording is unsupported
by the frozen request-bound parser, so secondary zero credit does not mean all
generated text is wrong. Raw challenge scope recall is 72/96 and calibration
degrades. Primary complete-assertion baseline zeros also reflect pending short
fragments, not zero factual accuracy. Both metrics and raw outputs remain visible.

The original oracle run has 65/384 answered, 319 runtime errors and 45/384 primary
credits. Those errors are not generated wrong answers or a license to remove
cases. A declared 319-case operational supplement stopped during preload with
`No CUDA GPUs are available`, making zero new generations/reservations in that
session. After GPU availability was restored, the same declared supplement
answered all 319 selected cases with zero new runtime errors. Its combined view
retains the 65 original answers: primary 247/384 (90 pending, 47 incorrect),
standard request-bound 208/288. The original files and interrupted-session
history remain preserved. The healthy session has matching base fingerprints;
it cannot reconstruct the unavailable final weight fingerprint of the original
failed GPU instance. No model or scoring setting changed.

See [FINAL_EVALUATION.md](FINAL_EVALUATION.md), [MODEL_CARD.md](MODEL_CARD.md),
[DATASET_CARD.md](DATASET_CARD.md) and [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)
for exact identities, denominators, uncertainty, incidents and limitations.

## Release checklist

- Complete: the [13-page report PDF](../output/pdf/EditGuard-Report.pdf), six
  scientific figures, cards, API/deployment documentation and six-part
  [panel demonstration](DEMO.md). Every rendered PDF page, current dashboard and
  JSON/CSV download was checked. Governance instructions distinguish structured
  answers from separately measured trained generation.
- Complete: local application/JS regressions (516 passed, 18 explicit optional
  skips), a separate 34-test CPU numeric/checkpoint group, recorded structured
  container lifecycle/restart checks and clean-source figure reproduction.
- Complete: the declared 319-case oracle recovery and finite isolated learned
  HTTP workload, reported separately in the final report and dashboard while
  retaining the original errors and timing boundaries.
- Complete: improvements pushed and all three CI jobs verified on exact source
  commit `2c7938c`. The [CI receipt](../reports/final/engineering/github-ci-recovered.json)
  records regression, CPU optimizer recovery and structured container/restart
  success. This sign-off documentation does not change measured source or ML claims.

The learned HTTP workload has 30/30 measured HTTP 200 answers, zero errors or
clarifications, warm p50/p95 230.42/372.04 ms and cold lazy loading 68,781.35 ms.
Three warmups are separate. It repeats three development questions, includes API
before-answer generation and reports `ml_accuracy=null`. A repeated comparison
answer remains wrong; these are availability/timing measurements, not 30 correct
answers. The isolated fixture state and model/configuration identities are stable;
the live database is unchanged. See [HTTP evidence](../reports/final/neural-http.json).

The device interruption does not authorize retraining, new backbones, a larger
dataset, test-guided scoring changes or another model search. Fixed-configuration
operational recovery and reporting are complete. The pushed release source passed
all three CI jobs, as recorded in the release checklist above.

## Experiment and recovery bounds

Development used an initial run plus two scope revisions and one answer revision.
Each substantial command had a run question, exact input/review/configuration
hashes, real-profile runtime/memory estimates, expected artifacts, checkpoints,
resume rules and finite step cap. The user executed the substantial GPU runs.
No further training is planned for this release.

Full-state checkpoints preserve model, optimizer, scheduler, RNG, sampler,
global step and validation selection. Changed inputs/source/configuration/reviews
refuse exact-state resume. Completed/capped runs refuse further training before
ML initialization. Final evaluation is separately single-use; interruption
recovery preserves committed observations. Its operational supplements are
separately declared and cannot replace the original run or tune against it.

## Completion boundaries

Completion requires tested edits/conflicts/rollback; both trained auxiliaries;
versioned, leakage-audited evaluation and fair baselines; a locally deployable
application; saved models/configuration; reproducible scripts; regression/CI;
container/API documentation; scientific figures; honest failure analysis; a
polished report and a tested demo. The lightweight structured image and optional
local neural environment are separate deployment paths. Local ignored model
binaries must be retained or reproduced; metadata in Git does not ship weights.

Current truthful resume wording: “Built a SERAC-inspired support knowledge editor
with transactional correction memory and rollback; trained compact applicability
and edit-conditioned answer models, evaluated five matched configurations on a
controlled held-out dataset, and containerized the structured application.” Any
number must name its population/metric and link to evidence. Keep the original
oracle incident distinct from its completed operational recovery. Do not claim
human review, production readiness, universal accuracy or learned superiority
over updated RAG.
