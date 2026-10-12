# Development review and finite pilots

The active 504-row pilot v2 has separate automated development review:
**504/504 inspected, 504 accepted, 0 flagged**. The user authorized this review;
no human inspection or trained-model accuracy is claimed. Source rows and their
manifest remain immutable, with zero human acceptances/rejections and 504 pending
human decisions. The [independent v2 audit](../reports/pilot-v2-independent-review.json)
and [typed automated registry](../reports/pilot-v2-reviews.automated.json) preserve
per-row hashes, provenance and `human_reviewed_data=false`.

## Restore the exact active pilot

```powershell
.\.venv\Scripts\python.exe scripts/pilot_bundle.py unpack --source data/pilots/pilot-v2-source.zip --manifest reports/pilot-v2-manifest.json --output data/generated/pilot-v2
```

On a clean checkout this restores the exact measured bytes, verifies the recorded
manifest and all allowed file hashes, and refuses an existing output directory.
The finite bundle contains development rows/catalogue/derivation metadata, with
no models, review registries, test questions or challenges. The archival v1 packet
is retained at `data/pilots/pilot-v1-source.zip`, bound to `reports/pilot-manifest.json`.
Original v1 reports and review pages remain unchanged.

The [v2 repair](../reports/pilot-repair.json) changed grammar in 28 correction
descriptors, 4 questions, 8 answer targets and 8 answer inputs. Structured values,
labels and fact/entity/edit/lineage identities were retained; all 504 review rows
received fresh IDs/hashes. `scripts/repair_pilot.py` records current source-code
hashes in any new build, so a later rebuild can have a different manifest identity.
Use the recorded bundle to reproduce the measured packet. Repair never copies
reviews or reads/replaces historical test/challenge files.

The original bounded pilot selected four training and two validation families
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

New full-data generation defaults to `controlled-v2`, generator 3 and input format
`editguard-input-v2`; the active pilot is a recorded derivation of the archived v1
pilot. Original v1 data and candidate test files remain unchanged.

## Inspect integrity without creating decisions

```powershell
.\.venv\Scripts\python.exe scripts/review_dataset.py inspect --data-dir data/generated/pilot-v2 --task scope --split train
```

Use `--family family-012` to narrow a view. Checks compare structured scope and
effective interval to labels; correction/original/expected values to catalogue;
comparison amounts to yes/no polarity; target subject/relation/value forms; and
support facts to the unchanged catalogue. Flags are prompts for investigation.
No flag-free result proves that an unfamiliar question has the intended meaning.

## Review through portable HTML or a terminal

The active pilot is `editguard-controlled-v1-pilot-v2-42-7b7a9b1b-grammar-80ce36e5`,
with input format `editguard-input-v2`. Its recorded manifest is
`reports/pilot-v2-manifest.json`; 360 scope pairs and 144 answer examples remain
pending human review in source. The derivation preserves 60 foreign-negative
reroutes. Four current standalone human review pages use the v2 hashes:

- `reports/review-v2-scope-train.html`: 240 rows.
- `reports/review-v2-scope-validation.html`: 120 rows.
- `reports/review-v2-answer-train.html`: 96 rows.
- `reports/review-v2-answer-validation.html`: 48 rows.

Each page contains its dataset manifest hash and exact row hashes. The embedded
JavaScript has bounded synthetic regression and syntax checks; semantic inspection
and explicit decisions remain the human reviewer's responsibility. An explicitly
typed automated registry is a separate development path, described below.

```powershell
.\.venv\Scripts\python.exe scripts/review_dataset.py export-html --data-dir data/generated/pilot-v2 --task scope --split train --output reports/review-v2-scope-train.html
.\.venv\Scripts\python.exe scripts/review_dataset.py export-html --data-dir data/generated/pilot-v2 --task answer --split validation --output reports/review-v2-answer-validation.html
```

Open each standalone HTML file in a browser. It requires no server, account or
network request. Every row starts pending. Inspect question, correction, explicit
scope, ground truth, comparison polarity, answer target and supporting context.
Supply your actual name and explicitly accept or reject each row; rejection
requires a reason. Export the resulting JSON before closing the page. Unsaved
decisions are kept in the page session only. Progress shows accepted/rejected/pending
counts, with navigation to every row or the next pending row. Filenames include
dataset/task/split, and new exports declare human review policy and reviewer type.
Explicitly load that page's prior exported JSON to resume. The page validates
dataset/manifest/row hashes, decision, name, zoned timestamp and rejection reason
before merging; duplicate keys and conflicts reject the whole load. Loaded controls
reflect preserved reviewer/timestamp/decision records and cannot replace them.
Legacy human exports remain readable; automated registries are explicitly refused.
There is no automatic save or registry/API write. Rows and embedded JSON are escaped,
and dynamic text uses `textContent`; untrusted question text cannot become markup.

Alternatively, use `interactive` with `--registry path/to/reviews.json`. It prints
one row at a time and offers accept, reject, skip or quit. It does not batch-accept.
Import conflicts leave the `.incoming` file available so human work is not lost.

## Import and preserve the authoritative registry

```powershell
.\.venv\Scripts\python.exe scripts/review_dataset.py import-decisions --data-dir data/generated/pilot-v2 --incoming path/to/exported-human-decisions.json --registry path/to/human-reviews.json
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

Accepted and rejected human decisions are procedural records, not digital identity
proof. Never script decisions as though a person inspected them. Tests use visibly
synthetic reviewer fixtures solely to verify software behavior.

The separate `reports/pilot-v2-reviews.automated.json` registry records authorized
automated development review and the independent audit hash. The loader and
development CLIs default to `human` and reject its machine-typed rows. Only explicit
`--review-policy automated_development` selects that registry; training run cards
likewise declare the policy. Calibration/comparison records, grids and manifests
bind review hashes/policy and `human_reviewed_data=false`. Scope calibration records
training and validation review provenance separately. This does not approve final
testing or satisfy independent human assessment. Pending/stale/unresolved records
still fail the selected gate; source pending flags are not mutated.

## Smoke fixtures and next training stage

For bounded programmer tests, use explicit toy question–correction rows or the
curated Nimbus handbook with purpose `smoke_test`. Such fixtures can validate
tokenization, backward passes and checkpoint restoration without being declared
human-reviewed research data. Runtime must refuse smoke-purpose checkpoints as
production learned models. A reviewed pilot run still requires the real trainer,
measured profile, pinned model revision, finite run card and exact-state resume
verification before a substantial GPU command is released. The v2 tokenizer audit
counted all 504 development inputs with no truncation at 192/256 tokens and all
144 answer targets with no truncation at 48 tokens; lengths are not model accuracy
or throughput. Pilot repair, automated review and tokenization did not read or
change archived v1 test/challenge question files. Final frozen evaluation and
independent human challenge questions remain separate milestones.
