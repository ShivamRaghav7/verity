# Independent automated review of pilot-v3

All **504 revised development rows were audited afresh: 504 accepted, zero
flagged**. This is explicitly machine-only consistency review under the
user-authorized `automated_development` policy. It is not independent human
semantic review. No previous v2 review decisions were copied or migrated.
No model, GPU, test or challenge corpus was used, and both source datasets
remain byte-identical to their pre-review states.

| Partition | Audited and accepted | Flagged |
|---|---:|---:|
| Scope training | 240/240 | 0 |
| Scope validation | 120/120 | 0 |
| Answer training | 96/96 | 0 |
| Answer validation | 48/48 | 0 |
| Total | 504/504 | 0 |

The new dataset is
`editguard-controlled-v1-pilot-v3-42-7b7a9b1b-grammar-80ce36e5-scope-balanced-8ab5b924`.
It retains four training and two validation entity families, 36 separate edit
worlds, 360 scope pairs and 144 answer examples. Its source manifest still
records pending rows and `eligible_for_training: false`; acceptance lives in a
separate, typed and hash-bound development registry. The human-review default
continues to reject that registry. Training requires explicit selection of the
automated development review policy.

## Independent judgments

The executable evaluator is `scripts/audit_pilot_v3.py`. It imports only Python's
standard library. It does not import the generator, formatter, balancing script,
integration integrity checks or scoring functions. Its full executable source
and both source-text/file SHA-256 hashes are embedded in
`reports/pilot-v3-independent-review.json`, alongside a decision for every row.

Anchored parsers independently recognize the twelve known question forms,
resolve the exact entity name or alias, plan, version and requested relation,
and require one compatible catalogue fact. The evaluator compares the parsed
request against the structured scope and checks half-open effective dates.
An edit applies only when entity, relation, plan, version and effective date
match. Applicable rows use the correction's new value; inapplicable rows use
the queried catalogue fact's original value. Neither a numeric overlap nor a
supplied binary label determines applicability.

The numeric comparisons are computed independently: requested devices, storage
or support quantity must be at most the allowance, while monthly price must be
at most the question's budget. **Comparison answer polarity and correction
applicability are different labels**. An out-of-scope comparison can correctly
have a `Yes` answer while its focal-edit scope label remains zero. All 252
comparison rows have independently recomputed polarity and noun agreement;
null polarity is checked on the other 252 rows.

A separate assertion parser validates all 144 answer targets for their endorsed
subject, relation, plan, version, numeric value, unit and required yes/no prefix.
It checks two unchanged other-property supporting facts per answer, their scope
and effective dates, and the exact correction/question/date/support components
in the sequence-to-sequence input. These checks concern dataset targets; they
do not count as evidence that a trained model generated a correct answer.

All 216 scope negatives are checked against their named difficulty category:
same entity/different property, different entity/same property, similar name,
wrong tier, wrong version or ordinary negative. All rows are checked for explicit
pending source metadata and simulated approval status. Simulated catalogue edits
are not production administrator approvals.

## Preservation and removed form shortcut

Each v3 row is bound to its new content hash and corresponding v2 source hash.
The auditor verifies unchanged factual truth, original correction metadata,
scope label, family split and edit lineage, and confirms each derivation's
reported changed fields. Applicable question fields and all answer-model input
and target fields remain unchanged. Catalogue, selection and split files are
byte-identical to v2. Changed row identities require fresh decisions even when
the answer's field values were preserved.

The four actual wording styles now have the same 40% applicable prior in both
partitions, as computed from parsed questions and independently computed labels:

| Actual style | Training applicable / inapplicable | Validation applicable / inapplicable |
|---|---:|---:|
| Direct | 24 / 36 | 12 / 18 |
| Alias paraphrase | 24 / 36 | 12 / 18 |
| Comparison with yes answer | 24 / 36 | 12 / 18 |
| Comparison with no answer | 24 / 36 | 12 / 18 |

This confirms removal of the observed marginal question-form shortcut in the
data. It does not establish that the revised classifier improves. The old
classifier's direct-question misses and answer-model polarity failures remain
documented in [the v2 validation results](PILOT_VALIDATION_RESULTS.md). No model
was retrained or evaluated as part of this audit.

The audit contains 23 row-check categories with explicit denominators and 46
passing global checks. Global checks cover exactly 504 unique rows, partition
counts, file hashes, four/two family assignments, disjoint family/entity/lineage
references, complete edit worlds, matched style priors and distinct
counterfactual values/scenarios within their original splits. Counterfactual
edit worlds remain separate; their contradictory values must not be activated
together in one memory snapshot.

## Registry cross-check and immutable evidence

`scripts/register_automated_review.py` now supports only the complete unfrozen
v2 development pilot or this supported v3 revision. For v3 it additionally
requires origin preservation, balanced-style checks, v3 derivation checks and
matching independently computed style/polarity. It cross-checks all row hashes,
scopes, facts and supplied independent observations, then rescored all 144
dataset targets with the application's complete-assertion parser. All 504
rows passed that integration cross-check. Accepted rows with unresolved issues,
stale hashes or disagreeing observations are refused; flagged independent
decisions receive explicit rejection and retain their reason.

The resulting `reports/pilot-v3-reviews.automated.json` records
`review_policy: automated_development`, `human_reviewed_data: false`, exact
audit/manifest hashes and per-row reviewer type, timestamp, reason and content
hash. It does not modify the dataset, elevate customer messages, or fabricate
human approvals. Existing output filenames and unsupported/test artifacts are
refused. The archived v2 registry and audit are unchanged.

| Evidence | SHA-256 |
|---|---|
| V3 manifest | `5e4790e2edf9c15e98f37ce455ab275a387fa1ae866c35cde40a0322de62dc2e` |
| Scope training rows | `a47f9ed7b8849c7c8071419f4a13649d802a219ee8ac13a4bd2a47c1e8abaf04` |
| Scope validation rows | `683f840682c2b62f12fbc36d9fc208a405c5cfd2d0d1b2dbff4d8b0f07630a39` |
| Independent audit | `82fbdf73e8a11fabb37bbeb49cb2056974b1b6e16299f6648d156bd409eb4da3` |
| Machine review registry | `3bcff20d4e82cfae18d04445fa51123ee43aa2592a8dacf93aeaf536d6f84546` |
| Independent evaluator file | `9bafc7c3e621d59684aea64c7af1fa24b45231d3a7d5b69ef17df9975cf2e725` |
| Registration cross-check file | `1d32639dccd2f5a230baf64e8200eb4b43d6874eb5196a5ce8b327a150f3e9dd` |

## Reproduction and verification

The exact timestamp makes all decisions and the full report reproducible.
Choose fresh output filenames; the example below reproduces the recorded audit
and registry without replacing them:

```powershell
.venv\Scripts\python.exe scripts/audit_pilot_v3.py --output runtime/pilot-v3-audit-copy.json --reviewed-at "2026-10-13T06:22:09.440596+00:00"
.venv\Scripts\python.exe scripts/register_automated_review.py --dataset data/generated/pilot-v3 --audit reports/pilot-v3-independent-review.json --output runtime/pilot-v3-registry-copy.json
.venv\Scripts\python.exe -m unittest tests.test_automated_registration tests.test_pilot_v3_review -v
```

Actual reproduction produced byte-identical copies of both the full audit and
registry in ignored runtime paths. The focused review suite passed 19 tests in
19.837 seconds. Seven new v3 tests verify complete machine provenance and source
preservation, deterministic decisions, refusal of stale origin/missing balance/
false style/human claims, recomputation of tampered scope labels and comparison
polarity, refusal of a synthetic test-named marker before reading its content,
and rejection of v2 decisions presented for v3. Twelve existing v2 registration
tests still pass, including stale hashes and observation disagreement checks.

The auditor handles only known controlled question forms; unsupported prose or
ambiguous subject resolution is flagged rather than guessed. This finite review
cannot establish natural-language robustness, production policy interpretation,
or independent human annotation quality. The fictional daily support quantity is
an allowance/availability fact, not a time-of-day operating schedule. Profiling,
training, calibration and validation after the revision are separate work with
their own finite run cards and honest measurements.
