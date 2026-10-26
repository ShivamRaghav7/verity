# Implemented contributions and the evidence behind them

This is a map of substantial work already present in the repository. It helps a
team allocate ownership, review the implementation and explain the project after
finishing it. It does **not** assign invented historical authorship or imply that
one person independently developed an upstream algorithm. Each claimed
contribution should be demonstrated, understood and linked to its evidence.

The useful product goal is to maintain a conversational assistant's facts when a
change also affects a connected answer: service ownership changes, so its on-call
contact changes, while old releases and other properties remain valid. The
demonstration domain and all service/team/person assignments are fictional.

## Work packages that can be owned and demonstrated

| Work package | Concrete implemented work | Reviewable artifacts | How to demonstrate understanding |
| --- | --- | --- | --- |
| Governed knowledge lifecycle | Canonical entity/property/scope keys, explicit approval, version lineages, competing-version rejection, withdrawal/restoration and atomic rollback | [store](../editguard_chat/store.py), [store tests](../tests/test_chat_store.py), [API rules](CHAT_API.md) | Publish a paraphrased fact with the same canonical IDs, resolve the conflict explicitly, then roll back. Explain the unique active-key invariant and why both rollback events share one revision. |
| Durable recovery and historical states | Text-workspace export, strict digest/schema/reference checks, event-state replay, transactional empty-target restore and approved memory at an earlier revision | [snapshot validation](../editguard_chat/workspace.py), [workspace tests](../tests/test_chat_workspace.py), [historical API](../editguard_chat/server.py) | Restore an exported text workspace into a fresh directory, compare old/current fact views, and show that corrupt input or a nonempty destination cannot partially overwrite work. |
| Learned model deployment | Integration of the attributed trained ReasonKE checkpoint, pinned model revision, 4-bit GPU loading, bounded input/output, separate authenticated worker, raw generation/resource metadata | [runtime](../editguard_reasonke/runtime.py), [client](../editguard_chat/reasonke_backend.py), [model card](REASONKE_MODEL_CARD.md), [real T4 profiles](../reports/reasonke/operations-profiles-v1/README.md) | Trace a question through selected facts into a genuine model generation. Explain upstream training, per-edit frozen weights, NF4 versus FP16 computation, token caps and explicit error handling. |
| Context selection and conversation consistency | Shared two-stage question-only lexical retrieval, matched retrieved fact IDs across models, active-memory bounds, exclusion of stale generated history and symmetric comparison contexts | [retrieval](../editguard_benchmark/retrieval.py), [controller](../editguard_chat/server.py), [ReasonKE contract tests](../tests/test_chat_reasonke.py), [deployment selection](../configs/deployment-policy.json) | Follow a two-hop question through first-hop selection and expansion, then show why prior-revision answers must not contaminate rollback or the without-context comparator. |
| Independent controlled dataset | Forty-eight isolated fictional fact graphs, explicit original/updated ownership, supporting-fact IDs, eight question categories, lineage-first partition and deterministic graph audit | [data authoring](../editguard_benchmark/data.py), [dataset packet](../data/reasonke/operations-v1), [dataset card](OPERATIONS_DATASET_CARD.md), [data tests](../tests/test_operations_benchmark.py) | Reconstruct the owner-to-contact join; distinguish the old team's valid contact from the newly selected contact. Show how paraphrases stay in the same split and why similar-name/old-release controls matter. |
| Fair and finite experiment execution | Five matched configurations, two sequential pinned checkpoints, profiled budgets, frozen source/config/data protocol, hard watchdogs, atomic request ledgers and exact-plan resume without wrong-answer retries | [runner](../editguard_benchmark/run.py), [sequential launcher](../scripts/run_operations_pipeline.py), [configuration](../configs/operations-evaluation.json), [frozen protocol](../reports/reasonke/operations-final/protocol.json), [benchmark contracts](../tests/test_operations_benchmark.py) | Explain each condition and why updated RAG receives current truth. Interrupt a software fixture and show that only unexecuted tasks can resume while completed failures remain recorded. |
| Scientific scoring and evidence integrity | Whole-endorsed-answer scoring, explicit outcome denominators, retrieval/oracle separation, locality and preservation, paired lineage bootstrap, latency/resource reporting, independent host replay and tamper rejection | [scorer](../editguard_benchmark/scoring.py), [metrics](../editguard_benchmark/metrics.py), [host audit](../scripts/audit_operations_evidence.py), [tamper tests](../tests/test_operations_evidence.py), [real development audit](../reports/reasonke/operations-development-v1/host-audit.json) | Recompute a saved score and explain why a target merely mentioned in an explanation is not credited. Distinguish accuracy, coverage, preservation and correctness; explain independent lineages and uncertainty limits. |
| Usable application and visual communication | Persistent normal chat, sourced correction editor, search, conflict choices, audit timeline, historical comparison dialog, fictional example approval, current results dashboard, PNG/SVG/CSV/JSON/PDF exports | [browser application](../editguard_chat/static), [results selection](../editguard_chat/results.py), [figure generator](../scripts/build_operations_report.py), [PDF generator](../scripts/build_operations_pdf.py), [browser evidence](../output/operations-2026-10-26/ui/README.md), [demo walkthrough](OPERATIONS_DEMO.md) | Perform the complete ownership-change demo and show current evidence separately from the older visual study. Explain which UI elements show model output and which are deterministic controls. |
| Reproducible delivery and regression engineering | Small isolated dependency sets, pinned Colab launchers, lightweight website container, safe evidence packaging, CI and real HTTP/SQLite checks that deliberately load no model | [chat requirements](../requirements-chat.txt), [GPU requirements](../requirements-reasonke.txt), [container](../Dockerfile.chat), [CI](../.github/workflows/ci.yml), [deployment checker](../editguard_chat/check_deployment.py), [24-check receipt](../reports/reasonke/engineering/workspace-deployment-3.json) | Build/run the website, inspect an honest unavailable-model state, reproduce a software check, then identify which additional GPU evidence is required for neural quality. |

These packages overlap at defined interfaces rather than requiring an arbitrary
number of presentation sections. A team can group or share them according to
actual implementation/review responsibility. Meaningful ownership includes
being able to explain the relevant source, reproduce its checks and interpret
its failure modes; it is not established by counting screens or lines of code.

## What has actually been measured

The [matched development study](../reports/reasonke/operations-development-v1/README.md)
contains **120 actual model requests**, the same 24 questions per condition over
three predetermined development lineages. ReasonKE updated retrieval scored
24/24, ReasonKE full memory 21/24, Qwen updated retrieval 24/24 and Qwen full memory
22/24. R0 had 12/24 against updated truth; its changed-fact answers were stale while
its 12 locality answers were correct. No execution errors or unexecuted records
were removed from that population. These are development results, not a held-out
final accuracy claim, and do not establish a ReasonKE advantage.

The retrieval policy was selected from that development evidence before final
inference. The [final protocol](../reports/reasonke/operations-final/protocol.json)
assigns 1,312 requests with source/data/configuration identities. A protocol and
launched workload are not completed results: final numerical claims require
the completed raw receipts, audit and aggregate. The current dashboard selects
saved evidence with explicit phase and execution-state labels rather than
inventing missing scores.

The [software deployment receipt](../reports/reasonke/engineering/workspace-deployment-3.json)
records 24 real HTTP/SQLite contract checks with **zero model generations** and
`model_accuracy: null`. Its passing status supports transaction, identity,
rollback and recovery behavior, not answer-model accuracy. Unit fixtures and
browser checks have the same distinction. The separate
[twelve-request HTTP workload](../scripts/verify_operations_http.py) prepares
actual website latency and post-restoration answer measurements; its fixtures
are not substituted for an executed GPU receipt.

The [typeset development report](../output/pdf/editguard-development-study.pdf)
and [derived figures](../reports/reasonke/operations-development-v1/derived)
come from those recorded outputs. No model generation occurs during plotting,
typesetting or host integrity replay.

## How earlier training contributes without being misrepresented

The repository also contains project-trained auxiliary models from the earlier
SERAC-inspired support editor: a DistilBERT pair classifier and a FLAN-T5 short
answer generator. Their training objectives, selected checkpoints, exact
pretrained provenance, measured resources and independent training audit are
documented in [MODEL_CARD.md](MODEL_CARD.md),
[training manifests](../reports/targeted-training), and
[the training audit](../reports/targeted-training-audit.json).

That earlier work provides a separately reproducible ML study. It does not mean
we trained ReasonKE, that its classifier routes current ReasonKE requests, or
that the earlier support/visual scores estimate current conversational accuracy.
The older methods' limitations and outcomes remain evidence, not a reason to
invent a history of planned success.

## Defensible claims for a report or resume

Claims supported by implementation include:

- Built a reversible knowledge-maintenance application with explicit correction
  approval, canonical scoped identities, historical comparison and transactional
  recovery.
- Integrated a pinned upstream-trained edit-conditioned reasoner on a measured
  T4 configuration and preserved raw answer/provenance/resource receipts.
- Authored a controlled fictional connected-fact dataset with lineage-controlled
  splits and evaluated matched retrieved/full-memory conditions against an
  official same-size instruction model.
- Implemented fixed-denominator scoring, retrieval/generation diagnostics,
  paired uncertainty, failure analysis and reproducible scientific exports.
- Trained and audited the earlier support editor's auxiliary models, with their
  distinct data, checkpoints and reported results.

Numerical claims must name their dataset, phase, denominator and comparison.
Do not claim a new foundation-model algorithm, training of the released ReasonKE
weights, reproduction of an undocumented paper score, human-reviewed labels,
arbitrary semantic conflict detection or guaranteed multihop/paraphrase accuracy.
The correct conclusion is the actual measured behavior and a working auditable
product, whether or not a specialized checkpoint beats a strong updated-RAG
baseline.
