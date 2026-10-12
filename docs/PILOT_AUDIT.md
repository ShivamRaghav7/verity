# Independent machine audit of pilot v1

This is an advisory machine audit of the controlled development pilot
`editguard-controlled-v1-pilot-v1-42-7b7a9b1b`. Human review is still pending.
No acceptance decisions were recorded, no registry was read or written, and no
dataset file was changed. No model, GPU, test partition or challenge questions
were used. The audit does not establish training eligibility or model accuracy.

The complete evidence is in `reports/pilot-independent-audit.json`. It records
exact input/code hashes, per-check denominators, representative questions for
every template, all affected defect IDs, numeric collision cases and all twelve
two-variant lineages. Its machine-rule implementation identifier is
`pilot-independent-structured-v1`.

## Scope and method

| Input | Rows |
|---|---:|
| `scope_train.jsonl` | 240 |
| `scope_validation.jsonl` | 120 |
| `answer_train.jsonl` | 96 |
| `answer_validation.jsonl` | 48 |
| Total task rows | 504 |

The audit also read the pilot's 96 fictional catalogue facts and its manifest,
selection, derivation and split metadata. Four training families and two
validation families contain 36 edit scenarios across 24 lineages. Scope rows
contain 144 positives and 216 negatives, with 36 examples of each declared
negative type. Answer rows repeat the 144 applicable scope questions with targets
and support context; they are not 144 additional independent question samples.

The generator and pilot derivation code were inspected to understand their
construction. The audit did not call their integrity checker, question/answer
formatters, data builder or the project's scoring functions. A separate standard
library procedure parsed the complete question using twelve explicit forms,
resolved its full subject against the pilot catalogue, and derived applicability
from entity, relation, plan, version and effective interval. Numeric comparisons
used `Decimal`, with price <= requested budget and requested device/storage/support
quantity <= the relevant allowance. This independently checks the comparison
direction rather than trusting supplied labels or target strings.

A separate four-form assertion parser checked the target's subject, relation,
numeric value, unit and yes/no prefix. Support facts were compared with the
unchanged catalogue; input text was decomposed into correction, question, date
and supporting assertions. The audit also checked pair counterparts, row hashes,
split isolation and counterfactual lineage membership. The JSON report contains
the explicit regexes, scope/inequality rules and a hash of the method description.

## Structured consistency findings

No structured inconsistency was found under the controlled semantics and these
specified parsing rules. This is machine consistency evidence, not human approval
or evidence that arbitrary natural-language questions are safe.

| Independent check | Checked / eligible |
|---|---:|
| Question has one recognized form and uniquely resolved scoped subject/property | 504 / 504 |
| Scope label or answer applicability agrees with independently derived scope | 504 / 504 |
| Expected subject/property/unit and numeric value agree with the scenario | 504 / 504 |
| Correction original value and scoped metadata agree with the base fact | 504 / 504 |
| Correction descriptor agrees with its structured fields | 504 / 504 |
| Row/fact references remain in their declared split | 504 / 504 |
| Rows and correction review flags remain pending | 504 / 504 |
| Derived row hashes match pilot derivation records | 504 / 504 |
| Yes/no label agrees with the independent numeric inequality | 144 / 144 |
| Negative type has its declared product/property/tier/version difference | 216 / 216 |
| Answer target endorses the requested fact and comparison polarity | 144 / 144 |
| Answer context preserves two other unmodified scoped facts | 144 / 144 |
| Answer has an identical positive scope counterpart | 144 / 144 |
| Two-variant lineages have distinct values/scenarios within one split | 12 / 12 |
| Declared pilot file hashes match audited content | 8 / 8 |

Family, entity and edit-lineage intersections between training and validation
were empty. The pilot redirects 60 foreign negative rows among selected families;
their resulting scopes, labels and row hashes passed the checks above. The audit
did not reopen original source rows to validate their historical hashes; those
hashes remain derivation provenance, rather than independently reverified source
content in this audit.

All four relations include both positive and negative comparisons. Across task
rows there are 72 yes and 72 no comparison labels; each answer comparison repeats
a corresponding scope comparison. For example, an edited 19 USD/month price
answers yes to a 19 USD budget and no to an 18 USD budget. A five-device limit
answers yes for five devices and no for six. These are the intended inclusive
boundary semantics.

Six of the 216 negative scope rows happen to have a numeric answer equal to the
correction's new value. Their labels correctly remain negative because product,
property, tier or version differs. The report lists these IDs so reviewers can
confirm that numeric overlap is not mistaken for applicability.

Counterfactual worlds require care. For Aster 015 Basic v2, a five-device edit
and a six-device edit share one lineage. The identical question about connecting
six devices is no in the first world and yes in the second. Both targets are
internally consistent; the edits must never be simultaneously activated as two
approved facts. Each world needs its own focal correction memory.

## Concrete surface defect

The singular quantity one incorrectly retains the plural noun `devices`.
The factual values and polarity labels remain consistent, but the text is
grammatically awkward and should be flagged during human review.

| Field | Affected rows / rows containing the field |
|---|---:|
| Question (`connect 1 devices`) | 4 / 504 |
| Correction descriptor (`to 1 devices`) | 28 / 504 |
| Answer target (`supports 1 devices`) | 8 / 144 |

The affected edits are:

- `edit:aster-055-core:Basic:v2:device_limit:variant-0` (training).
- `edit:aster-044-core:Basic:v2:device_limit:variant-0` (validation).

The eight answer IDs are `pilot-v1:42:answer:` followed by either edit ID and
`:p0`, `:p1`, `:p2` or `:p3`. The report spells out every affected ID and scope.
Descriptor counts include each edit's ten scope rows and four answer rows;
the question defect appears in the matching `p2` scope and answer rows.

Pilot v1 is preserved as the audited snapshot. If a separate corrected pilot is
derived, its hashes and review exports must be refreshed before decisions are
recorded. These findings do not authorize carrying approvals between versions.

## Coverage limits and human review priorities

Every one of the twelve question forms appears in both training and validation.
They comprise direct, alias-based paraphrase and quantity-comparison forms for
four relations. This pilot separates entities and edit lineages; it does not
provide unfamiliar wording. Independent manually authored wording remains a
later review/evaluation requirement and was not examined here.

Every correction targets the core product's Basic v2 scope. Pro, v1 and Plus
facts occur as scope negatives, which is useful for rejecting inappropriate
edits. They do not demonstrate that an answer model can edit those scopes.
All question dates are 2026-10-11 and all edit effective dates are 2026-09-01.
There are no before-effective or expired-edit examples. Date handling claims
must follow the deterministic gate/workflow tests, rather than imply learned
temporal generalization from this pilot.

Human reviewers should confirm the following assumptions and inspect the exact
rows before acceptance:

- Device and storage comparisons refer to total plan allowance, without existing
  usage, hardware restrictions or another unstated condition.
- Daily support-hours wording denotes the fictional catalogue's daily support
  allowance/availability; it does not specify an operating-hours schedule.
- Product names and aliases identify the intended entity, including near-name
  Plus products and `+` aliases.
- Corrections are fictional single-edit scenarios. The simulated approval field
  is neither a real administrator approval nor a human dataset review decision.
- Counterfactual variants stay in their assigned split and remain mutually
  exclusive scenario states.

The evidence covers a finite development slice and assists review. It cannot
replace independent human review, final test freezing, trained-model evaluation,
calibration, or deployed reliability measurements.
