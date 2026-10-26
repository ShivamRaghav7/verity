# Current implementation: governed memory and operations benchmark

The current branch now implements canonical fact identity, portable **text**
workspace snapshots, a current-model evidence dashboard, a controlled operations
dataset, and bounded two-model evaluation. This document describes runnable
implementation. The matched development run is complete; the larger frozen
final inference was launched on 26 October 2026 and has its own receipt ledger.
Only a completed saved final report is presented as final accuracy.

## Application

In Knowledge, optionally expand **Canonical fact identity** and supply all three
IDs: entity, property, and version/plan scope. Reusing these IDs shares the same
lineage despite different question wording. A competing active version is rejected
until an operator explicitly replaces it. Other properties/scopes remain separate.
Aliases are recorded metadata; they are not a trained entity resolver.

The server also implements `--context-policy retrieved`: the same declared
two-stage lexical retriever supplies at most four facts from up to 200 active
records. Full-memory mode retains its 20-record limit. The release's default
policy is retrieved context, selected before final inference from the matched
development comparison (24/24 versus 21/24 for ReasonKE); a storage/retrieval bound is
not a measured neural accuracy guarantee at 200 records. The frozen benchmark
compares both policies on identical nine-fact worlds.

**Export workspace JSON** saves text conversations, approved corrections, all
versions and audit events. Restore validates the checksum, field schema, references,
lineages, active-key uniqueness and complete revision history before inserting in
one SQLite transaction. It refuses a nonempty destination; launch the server with
a new `--storage` directory before restoring. A checksum detects corruption, not
the truth of the imported facts. The operator explicitly approves restoration.

Portable JSON snapshots intentionally refuse conversations with local images;
back up their SQLite database plus images directory privately. Current ReasonKE
chat is text-only. Export before a temporary Colab runtime terminates.

Research results opens **current text-model evidence**, including the five-condition
matched development scorecard, category outcomes, preservation, retrieval and
latency. The five exportable PNG/SVG figures and CSV/report are derived from the
actual 120 requests. Development results are explicitly labelled as such. The
older visual study is behind an archive button. A completed final result is read from
`reports/reasonke/operations-final/results.json`; missing results are never invented.

The Knowledge tab can explicitly approve a nine-fact fictional example handbook,
then draft an ownership change for a second explicit approval. Search the library
by question, value, source or canonical identity. Approval/rollback history shows
every event and revision. Chat's comparison dialog can generate against a previous
approved-memory revision, including the original nine-fact handbook, without
changing the live state. These are implemented product operations, not scores
credited to the model.

## Dataset actually authored

`data/reasonke/operations-v1` contains 48 isolated fictional service/team graphs,
partitioned into 16 development and 32 final lineages before wording augmentation.
Each has eight questions: direct, direct paraphrase, two-hop, two-hop paraphrase,
another property, similarly named service, another release, and unrelated fact.
The 384 question paths pass the deterministic graph audit. Training did not run.
Review is automated, not independent human semantic review. Shared templates and
repeated nonentity values limit broad language-generalization claims.

Every world supplies an initial fact view and a resolved updated view. Supporting
facts are available to all updated configurations. Nine facts fit within the
bounded memory; unrelated facts act as distractors. The key ownership edit is
applied only to release 2; release 1 and other service properties remain explicit.

The general instruction baseline is the official Qwen2.5-7B-Instruct checkpoint
at `a09a35458c702b33eeacc393d103063234e8bc28`, verified from
[the official metadata](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct).
The [model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) identifies its
instruction use. It is a same-size comparison; exact ancestry of the released
ReasonKE checkpoint remains unverified.

## Run stages

Use the isolated CUDA environment from the ReasonKE launcher. Both models are
loaded sequentially, never together. `configs/operations-evaluation.json` fixes
the five conditions, budgets and watchdogs. The retrieval baseline is two-stage
BM25-style lexical retrieval with question-only first pass and answer-text query
expansion; both generators receive exactly the same selected fact IDs.

```text
python -m editguard_benchmark.run --model reasonke --phase profile --output artifacts/operations/profile-reasonke.json
python -m editguard_benchmark.run --model instruction --phase profile --output artifacts/operations/profile-instruction.json
```

Each profile has exactly 12 declared requests, a hard 1,800-second loading
watchdog and a hard 240-second request watchdog. The parent process terminates
an over-budget worker and preserves error/unexecuted records. This is separate
from Transformers' soft 120-second generation setting. No training occurs.
Profile outputs include actual model/software revisions, loading and request
time, token counts, CUDA allocation/reservation and raw generations.

Run development only after reviewing profiles, with at most 72 ReasonKE and
48 instruction requests across their conditions. Settings and scoring must be
selected there. Do not search new models or tune against final answers.

After both profiles match the final source/data/settings, create a frozen protocol:

```text
python -m editguard_benchmark.run --freeze-profiles artifacts/operations/profile-reasonke.json artifacts/operations/profile-instruction.json --output artifacts/operations/protocol.json
```

The resulting protocol prints mean/max-profile extrapolations, with their
limitations. Review the profile-grounded runtime/memory budget before launching
the substantial final workload. Final inference refuses mismatched source, data,
configuration or protocol identities. It is assigned 784 ReasonKE and 528 Qwen
requests: five 256-question main conditions plus two 16-question oracle-support
diagnostics. Results are judged against current truth; R0 is the stale reference,
and its old-truth matches are separately recorded.

```text
python -m editguard_benchmark.run --model reasonke --phase final --protocol artifacts/operations/protocol.json --output artifacts/operations/final-reasonke.json
python -m editguard_benchmark.run --model instruction --phase final --protocol artifacts/operations/protocol.json --output artifacts/operations/final-instruction.json
python -m editguard_benchmark.metrics artifacts/operations/final-reasonke.json artifacts/operations/final-instruction.json --output reports/reasonke/operations-final/results.json
```

Use `--resume` only with identical identities. Completed wrong/error outcomes are
not retried; only unexecuted records run. Raw outputs and verdicts are saved
atomically after every request. A final run cannot be started without a frozen
profile-bound protocol. Report interruption honestly and preserve its ledger.

## Scoring and controls

The primary score matches the whole endorsed final span to predeclared aliases
for the requested entity, relation and scope. Mentioning a target in an example,
negation or rejected alternative is not success. Correct, wrong, abstain,
execution error and unexecuted outcomes remain distinct; all assigned questions
stay in the primary denominator. Bare short answers are valid for this task.

Metrics include each category, applicable/locality correctness, coverage,
answered-only precision, false edited-value endorsements, R0-correct retention,
support-fact recall, complete two-hop support recall, oracle-support performance,
pipeline p50/p95 and resources. Paired uncertainty resamples entire lineages.
Pipeline latency includes context selection, local worker communication and
inference, not the browser network; website latency requires its own workload.

The regression tests use explicit software fixtures and no GPU inference. They
are evidence for schema, routing, scoring, transaction and finite-plan behavior,
not neural factual accuracy. Neural results must come from the saved CUDA runs.

After final inference, generate the report, CSV scorecard and five scientific
figures from actual receipts using an analysis environment with Matplotlib:

```text
python -m scripts.build_operations_report artifacts/operations/final-reasonke.json artifacts/operations/final-instruction.json --output reports/reasonke/operations-final/derived
```

This postprocessing does not load either model. Figure manifests bind raw input
hashes and plotting-source identity. The report retains failed outcomes and
documents latency scope, strict answer matching and the small controlled sample.

See [the completed development evidence](../reports/reasonke/operations-development-v1/README.md)
for actual scores, model revisions, failures and measured T4 resource use. A
separate twelve-request website workload is available as
`python -m scripts.verify_operations_http --base-url http://127.0.0.1:8005 --output <new-file.json>`.
It requires a ready model and an isolated empty active memory; it publishes the
three development worlds sequentially, measures direct/paraphrase/two-hop answers,
restores each original owner, and measures the restored answer. Its HTTP latency
includes SQLite and website/worker communication. It is not a new independent
held-out sample, and its software fixtures are not neural results.

A further **four-generation** operational check runs in a different empty
workspace using `python -m scripts.verify_operations_history --output <new-file.json>`.
It asks the initial two-hop contact question, compares approved memory at revision
9 against the ownership update at revision 10 (two fresh generations), atomically
rolls back to revision 11, and asks for the restored contact. It checks model and
memory provenance, rollback events and every assigned answer, without retries.
Failures, abstentions and unavailable comparison results retain their denominator.

After both final model jobs finish, `scripts.finalize_operations_release` performs
integrity and claim-support audits, generates measured charts, runs the twelve-
request and four-generation website checks sequentially against the same loaded
checkpoint, then serves a fresh demo. It refuses incomplete final ledgers,
existing output directories and occupied ports. Its evidence bundle includes no
user chat database, model weights or session credentials. Final study accuracy
and the sixteen operational generations remain separate populations.
