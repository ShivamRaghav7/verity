# Dataset card: EditGuard Controlled Support v1

**Status:** Independent generated draft for a controlled fictional demonstration.
No generated label has been human-reviewed by creating these files. This is not
customer data, an external benchmark, or evidence of real-world support accuracy.

Version v1 / generator 2 replaces the unreviewed v0 draft before any test freeze:
v0 had disjoint old/new numeric supports and unbalanced comparison polarity.
Neither draft supports performance claims. The version change records the label
shortcut correction rather than silently presenting it as the same dataset.

## Purpose and independence

The task is applicability and short corrected answers for product/plan facts. The
catalogue and numeric corrections are authored deterministically for this project,
without importing SERAC, zsRE, Wikipedia or another editing dataset. Facts and
questions describe fictional products; the controlled handbook supplies truth.
Pretrained model corpora are outside our control and their provenance belongs in
model documentation. Dataset independence does not establish generalization to
real companies or arbitrary language.

## Initial scale

`scripts/prepare_data.py` defaults to 80 entity families, split 56 training,
12 validation and 12 test families. Each family contains one product, a nearby
product name, Basic/Pro contexts and v1/v2 contexts. The four relations are device
limit, monthly USD price, storage GB and support hours per day.

| Artifact | Total | Train / validation / test |
|---|---:|---|
| Initial structured catalogue facts | 1,280 | 896 / 192 / 192 |
| Independent correction scenarios | 480 | 336 / 72 / 72 |
| Scope question–correction pairs | 4,800 | 3,360 / 720 / 720 |
| Applicable correction answer examples | 1,920 | 1,344 / 288 / 288 |
| Challenge drafts: wording / unseen values | 96 | 0 / 0 / 96 |

The 480 scenarios arise from 320 editable core Basic-v2 facts plus alternate
counterfactual values for two relations. Alternate worlds are mutually exclusive
scenarios, not simultaneously approved edits. Each scenario has four positives
(direct, paraphrase, yes comparison, no comparison) and six negatives. This scope is small
enough to inspect and reproduce while testing critical boundaries. Correlated
rows are not 4,800 independent observations; uncertainty is grouped by family.
Start training with a reviewed pilot and use the full draft only after review and
a working pipeline. Do not increase scale to disguise model errors.

## Explicit ground truth and row schema

Facts include `fact_id`, `entity_id`, display name, aliases, relation, numeric
string value, unit, plan, version, effective dates, source, lineage, family and
split. Corrections include original/corrected values, entity aliases, exact scope,
effective dates, source, edit/scenario/lineage IDs, predecessor and scenario
approval status. Every record has an explicit pending review state.

The approval string `approved_in_simulated_scenario` defines the counterfactual
scenario only. It is not an administrator approval and does not authorize import
into application memory. Operational corrections pass the administrator workflow.

Scope JSONL rows contain:

```text
id, split, family_id, edit_id, lineage_id,
question, request_context={entity_id,plan,version,as_of},
correction={fact_id,entity_id,entity_name,entity_aliases,relation,
  original_value,corrected_value,unit,plan,version,effective_from,
  effective_to,source,approval_status,edit_id,lineage_id,supersedes,scenario_id},
correction_text, label (0/1), negative_type, question_type,
expected_fact={fact_id,entity_id,relation,value,unit,plan,version,as_of},
reviewed=false, review_status=pending
```

Answer rows retain the identifiers/question/correction/expected fact and add
`supporting_context`, `input_text`, `target_text`, `expected_polarity` and question
type. Supporting context contains other unchanged relations; it excludes the
edited relation. The corrected value is supplied through the correction input.
Positive comparison questions require correct yes/no polarity and the endorsed
fact. Merely mentioning the correct numeral is insufficient.

## Negative taxonomy

Every edit scenario includes one negative for each of:

- Same entity but a different property.
- Another entity but the same property.
- A near-name sibling product with the same property.
- The wrong subscription tier.
- An incompatible version.
- An ordinary unrelated product/property combination.

Negatives from another family are chosen inside the same split, so their question
entities cannot leak into another partition. All near-name products, aliases,
plans, versions, counterfactual worlds and fact lineages remain with their family.
Additional temporal, ambiguity, conflict and multi-edit challenges are reviewed
separately; the initial generated pairs do not cover all these phenomena.

## Leakage controls and remaining limitations

Assign families **before** facts, questions and paraphrases are augmented. Fixed
seed assignment and hashes make partitions reproducible. Tests verify disjoint
families, entity IDs and lineages across partitions and validate each negative
against its structured truth. New test entities and edits are never training rows.

Current paraphrase templates are shared across splits. This controls entity and
edit leakage but does not remove template familiarity. Numeric names and a small
relation vocabulary can make the problem easier than natural support language.
Held-out entities therefore do not imply novel wording. Effective dates in the
generated pairs are constant; historical behavior is tested through deterministic
governance and application cases. No temporal learning claim follows from these rows.
Original and corrected values share the same per-relation numeric pools with
equality excluded. Every edit scenario has both comparison polarities, so relation
and alternate-world index do not determine yes/no labels. Counterfactual variants
help diagnose value memorization; they do not prove it is
absent. Report results for direct questions, template paraphrases and independent
human challenge wording separately.

`challenge_drafts.jsonl` contains 48 agent-authored unfamiliar wording drafts and
48 challenge-only correction scenarios with values absent from the standard
numeric pools (17 devices, 53 USD/month, 512 GB or 24 support hours/day).
The latter embed their own correction and must not be merged with primary
scenarios. These are planned unseen-value diagnostics, not measured generalization.
They are explicitly **not** claimed to be manually authored human challenge
questions. Before final evaluation, a team member independently revises or writes
challenges, confirms intended subject/relation/scope/polarity, and a second member
reviews labels. Keep these questions hidden from model/threshold selection.

## Review protocol

Two team members first review catalogue semantics, units, alias uniqueness and
scenario meaning. Review generated labels rather than trusting templates or an
LLM. Check product/property/version/tier boundaries and whether each question has
one supported interpretation. Remove unclear rows with documented reasons or
rewrite them in a new declared dataset version.

Human decisions live in a registry bound to exact canonical row content:

```json
{
  "rows": {
    "scope:example-id": {
      "decision": "accept",
      "reviewer": "actual reviewer name",
      "reviewed_at": "2026-10-11T12:00:00+05:30",
      "row_sha256": "sha256 of canonical JSON for this exact row"
    }
  }
}
```

Use `decision=reject` with reason for exclusions. The script does not infer review
from a generated `reviewed` flag. `load_reviewed_rows()` rejects pending,
malformed or stale review decisions and returns accepted rows; unresolved rows
are fatal by default. This is a procedural gate, not a cryptographic identity
system. Never populate reviewer entries automatically as if a human inspected them.

Training inputs may use accepted training rows; development/calibration uses
accepted validation rows. Tensor-shape smoke profiling on unreviewed drafts must
be labeled as such, with no quality metric or learned result reported.

## Reproduction and test freeze

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe -m unittest tests.test_data -v
```

The manifest records SHA-256 hashes for catalogue, edits, assignment and all
JSONL files. No timestamp affects generated content. Existing differing outputs
require explicit overwrite before freezing or a new output directory. The test
set is currently a candidate test partition, **not frozen or evaluated**.

After independent review and final configuration/scoring decisions, set the
evaluation configuration status to `frozen`, record concrete model/checkpoint and
scorer identities, then run the bounded freeze operation:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py --freeze-test --reviews path/to/reviews.json --evaluation-config configs/evaluation.json
```

It requires resolved reviews on all test pairs/answers/challenges and stores test
file, review registry and evaluation-config hashes in `test_lock.json`. Changed
test hashes cannot be regenerated over that lock. This file guard supplements
team discipline: it cannot prevent copying examples elsewhere or repeated
unlogged test runs. Preserve raw outputs and declare any changed benchmark.

## Intended use and exclusions

Use for bounded support editing research and a controlled panel demonstration.
Do not infer production readiness, open-domain safety, multilingual performance,
or reliability on real customer cases. There is no personal customer information.
The team must set a license and document any later external contributions before
publishing dataset artifacts.
