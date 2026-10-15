# Dataset card: EditGuard Controlled Support

This is an independent, controlled fictional demonstration dataset for short
product-support knowledge editing. It contains no real customer records and is
not an external benchmark. The selected models use a 504-row development pilot.
A separate final packet has 1,104 machine-audited scope/answer/challenge labels,
is frozen, and is ineligible for training. The matched final comparison is sealed;
the original oracle measurement contains separately disclosed runtime errors.
The user authorized both the automated
development review and a fully disclosed automated final audit. Neither claims
independent human semantic review: `human_reviewed_data=false` throughout.

## Purpose, origin and scale

The project authors its catalogue, numeric corrections and question templates
deterministically. It does not import SERAC, zsRE, Wikipedia or another editing
dataset. The handbook defines ground truth for fictional products. Pretrained
model corpora remain outside our control; dataset independence does not imply
unrestricted-language or real-company generalization.

The original `controlled-v1` draft uses 80 entity families, assigned to 56 train,
12 validation and 12 test families before question augmentation. Each family
contains a core product, a near-name sibling, Basic/Pro plans and v1/v2 versions.
The four numeric relations are device limit, monthly price in USD, storage in GB
and support hours per day.

| Original source artifact | Total | Train / validation / test |
|---|---:|---|
| Structured catalogue facts | 1,280 | 896 / 192 / 192 |
| Independent correction scenarios | 480 | 336 / 72 / 72 |
| Scope question–correction pairs | 4,800 | 3,360 / 720 / 720 |
| Applicable-edit answer examples | 1,920 | 1,344 / 288 / 288 |
| Challenge drafts | 96 | 0 / 0 / 96 |

The selected models did **not** train on the entire original training partition.
The modest pilot uses four training and two validation families. This finite
scale made the review, training, failure diagnosis and complete workflow feasible
before any increase in dataset size. No scale increase was used to conceal errors.

| Actually used population | Families | Scope pairs | Answer rows | Review policy |
|---|---:|---:|---:|---|
| Pilot training | 4 | 240 | 96 | `automated_development` |
| Pilot validation | 2 | 120 | 48 | `automated_development` |
| Frozen final test | 12 | 720 | 288 | `automated_final` |
| Final challenges, within test families | Same 12 | 96 matched questions | Same 96 oracle questions | `automated_final` |

The 96 challenge records are one population evaluated in two settings, not 192
independent labels. The final audit therefore totals 720 + 288 + 96 = 1,104 rows.
Rows sharing a fact, family, template or counterfactual scenario are correlated;
reported uncertainty clusters by entity family.

## Explicit facts, edits and ground truth

Facts contain `fact_id`, `entity_id`, display name, aliases, relation, numeric
string value, unit, plan, version, effective interval, source, lineage, family and
split. Corrections retain original/corrected values, exact scope, source,
edit/scenario/lineage identifiers, predecessor and simulated approval status.

`approved_in_simulated_scenario` defines an evaluation's alternate world. It is
not an application administrator approval and never authorizes a customer
message or dataset import to change operational memory. Application edits use the
administrator workflow and transactional governance.

Scope rows contain the question, request entity/plan/version/date context, full
correction descriptor, explicit binary applicability label, negative category,
question form and expected scoped fact. Answer rows also contain unchanged
supporting context, conditional input, target response and comparison polarity.
Supporting context excludes the edited relation. Corrected values come through
the correction input; targets are not hidden inference inputs.

Every correction scenario has direct and paraphrase questions and both comparison
polarities. Each has six negatives: same entity/different property, different
entity/same property, similar product name, wrong plan, wrong version and an
ordinary unrelated case. All counterfactual values and paraphrases of a fact
remain in its family partition. Alternate worlds are mutually exclusive scenarios,
not contradictory edits activated together.

Original and corrected values use overlapping relation-specific numeric pools,
excluding equality. Both Yes and No comparison cases occur for each scenario.
These controls remove obvious value/polarity shortcuts; they do not establish
that a model has learned general reasoning.

## Versioned pilot derivations

Pilot v2 repairs singular device/hour display grammar while retaining facts,
labels and partitions. Its exact 504-row source bundle and separate automated
review registry are preserved. Pilot v3 balances negative wording across direct,
paraphrase, Yes-comparison and No-comparison forms. It changes 162 negative
question strings across train/validation, preserving row count, truth, labels,
entity assignments, positives and answer examples. The scope model trains on v3;
the answer model trains on v2. Each uses its own exact manifest/review identities.

The current input format is `editguard-input-v2`. Source pending-human flags are
not overwritten to invent human decisions. Typed automated registries provide
separate exact-row-hash-bound acceptance records. Pending, stale or mismatched
registries cannot authorize training. The default human-review tool does not
silently accept an automated registry; development commands explicitly declare
their review policy.

Restore exact development bytes on a clean checkout:

```powershell
.\.venv\Scripts\python.exe scripts/pilot_bundle.py unpack --source data/pilots/pilot-v2-source.zip --manifest reports/pilot-v2-manifest.json --output data/generated/pilot-v2
.\.venv\Scripts\python.exe scripts/pilot_bundle.py unpack --source data/pilots/pilot-v3-source.zip --manifest reports/pilot-v3-manifest.json --output data/generated/pilot-v3
```

The bounded bundles verify hashes, refuse differing existing outputs, and exclude
model weights and test/challenge questions. Re-generating with current code is a
new derivation; unpacking the archived bundle reproduces the measured bytes.

## Final packet and review policy

The original final question files were not read during development training,
calibration, model selection or scoring repair. The selected model exports,
calibration, inference settings, scoring rules and preparation/audit algorithms
were frozen before final question access. The new packet is
`data/final/final-v2`, bound by
[`reports/final/protocol-v2.json`](../reports/final/protocol-v2.json) and
[`reports/final/test_lock-v2.json`](../reports/final/test_lock-v2.json).

Preparation preserves original questions, facts, labels, entity/plan/version/date
scope, partitions and lineages. It adapts correction and answer display strings
to the already declared input format and adds explicit oracle/challenge routing
metadata. It does not overwrite the original source files. An initial startup
dependency-guard rejection occurred before any model inference; its source
amendment and logs are retained, and the active packet's question/truth bytes
equal those of the rejected packet. See [FINAL_EVALUATION.md](FINAL_EVALUATION.md).

The final audit accepts 1,104 and flags zero rows. It checks controlled-format
question meaning, arithmetic/polarity, expected fact/value/unit, request scope,
correction compatibility, unchanged context, aliases, counterfactual separation
and raw partition disjointness without loading models or using GPU inference.
Its per-row decisions, hashes and checks are in the packet's `audit.json` and
`reviews.json`. This establishes machine consistency under a finite grammar,
not independent human natural-language truth assessment. Zero flagged labels
does not imply perfect labels or perfect model accuracy.

The active packet is `eligible_for_final_evaluation=true`,
`eligible_for_training=false`, `test_frozen=true`. The original draft remains
archived with its original review status. Exact hash checks reject mutation;
the final runner refuses a second fresh evaluation and permits only continuation
of the same unfinished execution. These file controls supplement team discipline;
they cannot prevent unlogged copies or use elsewhere.

## Leakage controls and remaining limitations

Raw final entities, families and edit lineages are disjoint from both pilot
training/validation and the original declared training/validation partitions.
Foreign negative references remain inside their split. Similar-name siblings,
plans, versions and counterfactual variants remain together.

Scope preprocessing normalizes matching product/plan/version literals. This
creates eight same-label abstracted train/validation pair overlaps and nine
same-label final/train overlaps despite raw identity separation. No contradictory
abstracted labels overlap. Of 720 final scope pairs, nine share a training input,
zero share a validation input, and 711 share neither. Report performance on the
shared and unshared groups separately. Shared logical templates remain even in
the unshared group; entity-heldout performance is not novel-language proof.

All positive edits target core Basic-v2 facts. Pro, v1 and sibling products are
negative cases; results do not demonstrate learned positive editing of those
contexts. Generated pairs use constant effective dates. Historical correctness,
conflict handling and rollback are engineering tests, not temporal learning
results.

The challenge population contains 48 agent-authored unfamiliar-wording drafts
and 48 separate correction scenarios with values outside the standard pools: 17
devices, 53 USD/month, 512 GB or 24 support hours/day. The latter carry their own
corrections and must not be merged with standard scenarios. The questions are
explicitly not claimed to be human-authored. The user's automated-final-audit
choice supersedes the earlier planned human review; the lack of independent
human wording/review remains a reported limitation.

## Measured coverage and interpretation

All five configurations complete the same 720 standard and 96 challenge questions
with zero runtime errors; the trained editor clarifies 23 standard and 48 challenge
questions. No abstention is removed from the corresponding accuracy denominator.
On standard short QA, updated RAG gets 649/720 and the trained editor 613/720.
The editor's raw scope classification is 701/720, including 9/9 abstracted inputs
shared with training and 692/711 unshared inputs. These are measurements within
the declared controlled dataset, not open-language accuracy.

All 96 challenge question forms are unsupported by the frozen request-bound
scorer and remain without secondary credit; that does not establish that every
generated answer is wrong. The editor's raw challenge scope recall is 72/96,
and runtime answers only 48/96. These observed losses in scope/coverage limit
claims about unfamiliar wording or values. No scorer was extended after final
question access. Primary complete-assertion results, pending outputs and raw
responses remain available for inspection.

The original 384-case oracle export has 65 answers and 319 CUDA runtime errors;
all 384 remain in its fixed denominator. This is a runtime measurement failure,
not a reason to change labels, remove rows or claim model wrongness. Any fixed
configuration recovery is separately named and preserves the original export.
Its first session stopped before any generation because CUDA was unavailable.
After GPU availability was restored, the same declared supplement answered all
319 selected cases with zero new runtime errors and retained the 65 original
answers in a separate combined view. It does not change the packet, labels,
original frozen scores, denominator, models or scoring rules. The combined view
has 247/384 primary credits with 90 pending; standard request-bound credit is
208/288. Challenge request-bound scoring still cannot measure unsupported wording.
The separate 30-request learned HTTP workload repeats three development
demonstration questions, reads no held-out questions and reports `ml_accuracy=null`.
Its availability/timing counts are not additional dataset accuracy evidence.
See [FINAL_EVALUATION.md](FINAL_EVALUATION.md) and the
[runtime disclosure](../reports/final/runtime-failure-disclosure.json).

## Intended use and exclusions

Use for bounded support editing research and a controlled panel demonstration.
Do not infer production readiness, open-domain reliability, multilingual support
or real-customer accuracy. No personal customer information is present. A license
for dataset distribution and any later external contributions must be explicit.
See [the model card](MODEL_CARD.md) for trained-data scale and
[the final evaluation protocol](FINAL_EVALUATION.md) for fixed denominators,
abstentions and scoring limits.
