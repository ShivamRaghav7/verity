# Human review and finite development pilots

The review tool helps people inspect and label data. Automated integrity checks
never accept a row or impersonate a reviewer. Current generated data and pilot
rows remain pending until actual human decisions are imported. Independent
engineering and explicitly synthetic smoke tests can proceed while review is pending.

## Create the small pilot

```powershell
.\.venv\Scripts\python.exe scripts/review_dataset.py pilot --data-dir data/generated/controlled-v1 --output data/generated/pilot-v1
```

The default selects four training and two validation families, deterministically
with seed 42. Both tasks share these family selections: 240 training and 120
validation scope pairs; 96 training and 48 validation answer rows. Each family
retains all correction lineages, alternate worlds, four positives and six negative
types. No test question file or challenge file is read or copied, and the pilot
catalogue contains only selected development families. `--task scope` or
`--task answer` produces only that task's development files.

Source v1 cross-family negatives form a cyclic ring. Taking their transitive
whole-family closure would expand a one-family selection to the entire split.
The useful pilot therefore derives only foreign negative questions: their product
name, explicit request context and expected fact are rerouted to another selected
family in the same split, preserving entity role/property/tier/version and label.
All original/corrected values, lineages, variants and positive questions remain
unchanged. These are new derived examples requiring fresh human review, not an
unmodified subset represented as original data.

Every pilot row has a fresh ID and pending state. `derivation.json` binds its source
row ID/content hash to the new content hash and records whether rerouting occurred.
The manifest records source dataset identity, source manifest/development-file/
catalogue hashes, counts, selection and derivation hashes. The pilot has its own
dataset/version identity. Source data and candidate test artifacts are unchanged.
An output must be empty/new, and a frozen output cannot be reused.

`--policy closure` is an optional exact-source dependency closure. Its default
16-family-per-split budget rejects large cyclic expansion with an explicit reason;
it does not silently turn the small pilot into full-scale review.

## Inspect integrity without creating decisions

```powershell
.\.venv\Scripts\python.exe scripts/review_dataset.py inspect --data-dir data/generated/pilot-v1 --task scope --split train
```

Use `--family family-012` to narrow a view. Checks compare structured scope and
effective interval to labels; correction/original/expected values to catalogue;
comparison amounts to yes/no polarity; target subject/relation/value forms; and
support facts to the unchanged catalogue. Flags are prompts for investigation.
No flag-free result proves that an unfamiliar question has the intended meaning.

## Review through portable HTML or a terminal

The prepared pilot is `editguard-controlled-v1-pilot-v1-42-7b7a9b1b`. A copy of
its actual manifest is tracked at `reports/pilot-manifest.json`. Its 504 rows are
pending: 360 scope pairs and 144 answer examples. The derivation records 60
foreign-negative reroutes. Four standalone review pages are ready:

- `reports/review-scope-train.html`: 240 rows.
- `reports/review-scope-validation.html`: 120 rows.
- `reports/review-answer-train.html`: 96 rows.
- `reports/review-answer-validation.html`: 48 rows.

Each page contains its dataset manifest hash and exact row hashes. The embedded
JavaScript passed `node --check`; semantic inspection and explicit decisions
remain the human reviewer's responsibility. Review all retained training and
validation rows before creating a substantial run card.

```powershell
.\.venv\Scripts\python.exe scripts/review_dataset.py export-html --data-dir data/generated/pilot-v1 --task scope --split train --output reports/review-scope-train.html
.\.venv\Scripts\python.exe scripts/review_dataset.py export-html --data-dir data/generated/pilot-v1 --task answer --split validation --output reports/review-answer-validation.html
```

Open each standalone HTML file in a browser. It requires no server, account or
network request. Every row starts pending. Inspect question, correction, explicit
scope, ground truth, comparison polarity, answer target and supporting context.
Supply your actual name and explicitly accept or reject each row; rejection
requires a reason. Export the resulting JSON before closing the page. Unsaved
decisions are kept in the page session only. Rows and embedded JSON are escaped,
and dynamic text uses `textContent`; untrusted question text cannot become markup.

Alternatively, use `interactive` with `--registry path/to/reviews.json`. It prints
one row at a time and offers accept, reject, skip or quit. It does not batch-accept.
Import conflicts leave the `.incoming` file available so human work is not lost.

## Import and preserve the authoritative registry

```powershell
.\.venv\Scripts\python.exe scripts/review_dataset.py import-decisions --data-dir data/generated/pilot-v1 --incoming path/to/review-decisions.json --registry data/generated/pilot-v1/reviews.json
```

Decisions must reference the exact row SHA-256, a named reviewer, accept/reject,
an ISO timestamp with timezone (`Z` or an explicit offset), and a rejection reason.
The browser writes a UTC timestamp only when a person records a decision.
Manifest/row hashes are checked; pending or stale decisions fail training preflight.

Imports reject duplicate JSON keys, unknown/test row IDs and conflicting existing
decisions. Identical reimports are idempotent. Existing unknown fields and decisions
are preserved. No silent replacement changes earlier decisions; use a newly
declared corrected dataset/review record if adjudication is necessary. A review
registry snapshot referenced by a frozen test lock is immutable; use a separate
development registry for subsequent development decisions. Test review/editing is
deliberately outside this development tool.

Accepted and rejected decisions are procedural human records, not a digital
identity proof. Do not script accepted decisions into a production registry. Tests
use visibly synthetic reviewer fixtures solely to verify software behavior.

## Smoke fixtures and next training stage

For bounded programmer tests, use explicit toy question–correction rows or the
curated Nimbus handbook with purpose `smoke_test`. Such fixtures can validate
tokenization, backward passes and checkpoint restoration without being declared
human-reviewed research data. Runtime must refuse smoke-purpose checkpoints as
production learned models. A reviewed pilot run still requires the real trainer,
measured profile, pinned model revision, finite run card and exact-state resume
verification before a substantial GPU command is released.
