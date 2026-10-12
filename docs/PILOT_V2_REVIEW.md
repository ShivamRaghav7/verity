# Independent automated development review of pilot v2

This review read every one of the 504 scope/answer training and validation rows
in editguard-controlled-v1-pilot-v2-42-7b7a9b1b-grammar-80ce36e5. The user explicitly authorized the implementation
agents to review these development labels. Reviewer type: automated_development.
**No human review or human approval is claimed.**

Decisions: **504 accept / 0 flagged / 504 reviewed**.
Acceptance means consistency under the controlled demonstration semantics; it is
not ML accuracy or an arbitrary-language reliability claim. No registry was written.
Dataset rows and their pending source flags remain unchanged.

The integration subsequently cross-checked all 504 decisions and all 144 targets
and registered them separately at `reports/pilot-v2-reviews.automated.json`.
The default human-review gate refuses that registry; development runs explicitly
select `automated_development`. Human review and final-test review remain separate.

The complete evidence is reports/pilot-v2-independent-review.json. Each row has
an exact hash, supplied/computed applicability and polarity, independently computed
expected fact, parsed question fact,
parsed target where applicable, checks, issues and decision. The report preserves
the independent executable review source and SHA-256, all input hashes, complete
template inspection, counterfactual cases and numeric-overlap negatives.

## Reviewed inputs

| File | Rows |
|---|---:|
| scope_train.jsonl | 240 |
| scope_validation.jsonl | 120 |
| answer_train.jsonl | 96 |
| answer_validation.jsonl | 48 |
| Total | 504 |

The 96 fictional pilot catalogue facts and derivation/split metadata supplied the
truth and provenance. No test/challenge rows, review registry, model or GPU was used.
The source manifest SHA-256 is 8ab5b924549b2ff4039384c476e09075a74f58d471195cb7f9e6ed0c62eb7559.

## Independent method

Every v2 row is parsed afresh by complete-question forms and exact catalogue
subject resolution. Applicability requires product, property, plan and version
equality plus an effective interval containing the question date. Expected numeric
truth is selected from the original/corrected fact independently of target text.
Inclusive comparisons use price <= requested budget and requested device/storage/
support quantity <= the corresponding allowance.

A separate complete-assertion parser checks the endorsed subject, property, plan,
version, unit, numeric value and yes/no prefix. Supporting facts must be two
unchanged other properties in the same scope. Correction and answer-input text
must preserve the structured metadata, source, dates, question and support facts.
Numeric-one device/hour forms require correct singular/plural agreement.

Every row is bound to its v2 derivation hash and input version. Global checks
verify all partition/file hashes, unique IDs, family/entity/lineage isolation and
distinct counterfactual values/scenarios. No prepare/review/repair/formatter/scorer
function is imported or used for judgments. The v1 audit's results or decisions
were not copied.

## Template, counterfactual and overlap cases

All twelve generated question forms were inspected with train/validation answer
representatives: direct, alias paraphrase and comparison wording for device limit,
monthly price, storage and daily support hours. All forms are shared across the
two splits, so this review does not establish unfamiliar-wording generalization.

All twelve two-variant lineages and their direct/paraphrase/yes/no targets were
reviewed. An identical six-device request is no in a five-device world and yes in
a six-device world. Each world must have its own focal correction memory; the
variants must never be coactivated as simultaneous approved facts.

The 6 / 216 negative rows whose answers numerically equal the
correction's new value are explicitly recorded. They remain negative because
product/property/tier/version differs. Numeric target presence does not establish
applicability. The v1 singular-device errors were rechecked in the actual v2
questions, descriptors and targets, with new row hashes and fresh decisions.

## Decision use and limits

Root must cross-check these decisions before constructing an authorized
development registry. Any resulting registry must state automated development
review provenance and must not present these labels as human-reviewed. This
report itself changes no training eligibility flags or approval history.

The pilot covers only core Basic v2 edits and constant temporal metadata.
Pro/v1/Plus negatives test rejection scope, but do not show editing of those
scopes. Device/storage comparisons assume total allowance without usage/hardware
conditions. Support hours mean fictional daily allowance/availability rather
than a time-of-day operating schedule.

Training results, calibration, independent human assessment, unfamiliar wording
challenges, frozen final-test evaluation and deployed reliability remain separate
requirements. The full per-row decisions and source/method hashes make this finite
automated review inspectable.
