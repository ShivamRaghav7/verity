# EditGuard: interim project report

**Status: engineering foundation and first workflow. Final ML experiments pending.**

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
The prototype currently exercises those governance rules with structured answers.

## Data and protocol

The independent fictional training catalogue is separate from the ten-fact Nimbus
UI fixture. Entity families are partitioned before questions and counterfactual
values are augmented. Hash-bound human review controls eligibility; the test
partition is not frozen yet. See [dataset card](DATASET_CARD.md) and
[evaluation protocol](EVALUATION_PROTOCOL.md). All generated labels remain pending.

## Engineering evidence

Regression covers approval, conflicts, half-open dates, explicit supersession,
rollback, neighboring scope retention, persistence, authority checks and coherent
concurrent reads. The live dashboard exposes actual isolated workflow checks,
their denominators and fixture service latency. These checks demonstrate database
and API behavior, not learned editing performance.

The UI provides customer questions, scoped before/after comparisons, an approval
workflow, history, conflict resolution and exports. Minimal infrastructure consists
of FastAPI, Pydantic, SQLite and build-free static assets. Docker and CI definitions
are included; their execution must be recorded independently.

## Experiments and results

| Required result | Current evidence |
|---|---|
| Trained auxiliary weights | Pending; absent |
| Real training time and memory | Pending full runs; finite profiling recorded separately |
| Applicable-edit retrieval recall | Not measured |
| Raw scope precision/recall, false application and calibration | Not measured |
| Oracle answer and end-to-end factual accuracy | Not measured |
| Heldout paraphrases, entity generalization and baseline retention | Not measured |
| Five fair configurations with paired uncertainty | Not run |
| Conflict/rollback consistency | Engineering regression and local demonstration |
| Structured service/HTTP latency | Engineering-only benchmark artifacts when run |

No ML performance numbers or superiority claims may be inferred from empty fields.
The final report will replace this interim status only after user-executed training,
reviewed data, validation-driven selection, final configuration/scorer/test freeze,
and the matched final comparison. Scientific figures must be generated from those
recorded per-question results, with sample sizes and uncertainty.

## Limitations and next acceptance gates

Current language coverage is deliberately narrow. A correct source does not
guarantee a generated answer endorses it. Scope ambiguity, unfamiliar wording,
token shortcuts and temporal errors need component diagnostics. See
[failure analysis](FAILURE_ANALYSIS.md) for concrete findings.

The finite [A–H plan](PROJECT_PLAN.md) lists ownership, run-card requirements,
maximum development revisions and full completion criteria. This interim artifact
does not mark the ML-focused project complete.
