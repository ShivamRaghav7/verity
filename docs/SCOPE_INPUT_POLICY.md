# Explicit scope-input policies

The second and final permitted scope revision tests a single representation
change: remove the controlled catalogue's literal product names from matching
scope inputs. The 240 training rows and 120 validation rows, with their disclosed
automated development review,
backbone, seed, objective and 360-step cap stay fixed. This change is prepared;
it is **not evidence of a trained model improvement**.

The first two raw-input pilots learned narrow cues, and the revision-1 model
missed applicable `Nimbus` questions in the isolated real application workflow.
Good guarded validation-pair results did not establish generalization to the
application's unseen product name. That failure remains in the archived reports.

## Policies and boundaries

`editguard.scope_inputs.scope_pair(question, correction_descriptor, policy)`
returns the two classifier input strings. It uses only the question and the
standard correction descriptor. No label, expected fact, answer target or request
truth enters this function.

| Policy | Behavior |
| --- | --- |
| `raw_v1` | Default for legacy artifacts. Returns both strings exactly, including their whitespace and descriptor shape. |
| `entity_neutral_v1` | Parses the project's standard structured numeric correction descriptor, replaces matching literal product names/aliases, plan and version selectors in the question with shared phrases, and constructs a neutral correction descriptor. |

For a correction changing Nimbus Basic v2's device limit from five to three:

```text
Question:
How many devices does Nimbus Basic version v2 support?

Normalized question:
How many devices does this product this plan this version support?

Normalized correction:
Correction for this product, this plan, this version: device limit changed from 5 to 3 devices.
```

The product aliases are escaped as literals and matched longest first. Identifier
continuations are protected: `NimbusMini`, `Nimbus-Plus`, `A015+`, `v20` and
`v2.1` cannot silently become matches for `Nimbus`, `A015` or `v2`. A separately
spelled suffix, such as `Nimbus Mini`, remains visible after the matching `Nimbus`
literal is replaced; exact product compatibility is still checked from original
metadata before routing. Unmatched `Pro`, `v1` and unfamiliar product text remain
in the question. Property words and comparison quantities also remain: a price
question is still a price question, and “connect 4 devices” still contains four.

The neutral correction retains the edited relation, original value, corrected
value and unit. Relation underscores become spaces. It removes the descriptor's
literal subject/aliases, effective date and opaque source URI so that family and
lineage identifiers in those strings do not act as catalogue cues. All original
metadata remains unchanged in the approved memory, API, source/history display,
deterministic compatibility gates and evaluation truth. Dates in the customer's
question remain as written.

This version explicitly supports the four declared numeric relations:
`device_limit`, `price_monthly`, `storage_gb`, and `support_hours`, with their
existing units. Malformed descriptors, unsupported policies, incompatible units,
invalid dates and selector aliases that collide with a plan/version fail
explicitly. Broadening this descriptor contract requires a deliberate new policy
version; the parser does not guess an unfamiliar schema.

## What is deterministic and what is learned

Literal selector normalization is **deterministic preprocessing**, and approval,
product/plan/version/date compatibility remains a **deterministic guard**. Their
successful boundaries must not be presented as learned entity identification or
learned temporal reasoning. The trained cross-encoder still learns whether the
question's requested property and meaning fall within the conditioned correction.
Its probabilities, uncertainty, raw failures and guarded outcomes are evaluated
separately. Normalization does not authorize a correction, decide a label, choose
an answer, or verify arbitrary prose.

Unrestricted language, ambiguous aliases, and unseen wording remain limitations.
A product alias that is an ordinary word can be ambiguous in natural language;
the function performs literal replacement, while the bounded application's
context resolver remains responsible for ambiguity and selector conflicts. This
pilot does not measure general temporal reasoning or unrestricted dialogue.

## Reproducibility and integration

The scope policy has its own version, `editguard-scope-input-v1`, separate from
the global `editguard-input-v2` formatter used by the existing answer model and
baselines. `scope_policy_identity(policy)` supplies:

```text
scope_input_policy
scope_input_policy_version
scope_input_source_sha256
```

The last field hashes exact `scope_inputs.py` bytes. New scope training artifacts
bind this identity into their run/checkpoint metadata and manifest. If any scope
policy fields are present, runtime requires the complete supported identity;
different code or versions are rejected. Legacy scope artifacts without these
fields retain `raw_v1`. Existing answer artifacts need no new policy metadata.

Training applies the shared function before tokenization and length auditing.
`ScopeCheckpoint.score` applies it once before runtime tokenization. No special
tokens or vocabulary rows are added, and no base, embedding or answer-model
input is normalized by this scope policy. Changes to the neural runtime source
require fresh baseline calibration evidence under the existing source-hash
contract; archived evidence remains immutable.

The representation, checkpoint contract and source-bound guarded profile have
been checked; its released run card passed preflight. The user executes the
finite 360-step run using the [released guide](ANSWER_REVISION_TRAINING.md).
The concrete question is whether this normalization improves routing on the
unchanged held-out edits and Nimbus workflow without increasing false
applications. Its conditional budgets are 1,331 seconds, 2.0 GiB GPU and 5 GiB
checkpoint/export storage. These are four-update planning estimates, not
guarantees or trained-model measurements. Recovery uses the unchanged card and
latest completed checkpoint; completed exports and checkpoints at the cap are
refused before ML initialization. Validation selection and the same unseen-name workflow must then
test whether it helped. Document failure if it did not; do not extend the cap,
add a hyperparameter sweep, or tune against the frozen test set. Final evaluation
and release polish remain pending. Generated `No`
errors belong to the separate answer model and are not solved by this policy.

The current reviewed pilot's 360 scope rows all parse under this policy. Each
partition retains unique pairs (240 train, 120 validation), with no contradictory
labels after transformation. **Eight identical neutral pairs occur in both
partitions**, with the same label. Original entities and lineages remain split,
but this abstraction overlap must remain disclosed: those eight validation rows
cannot demonstrate an unseen logical form. A separately authored and reviewed
wording challenge and the separate unseen-name application workflow remain
necessary to assess practical generalization after training. No human-review
completion or frozen challenge-test result is claimed here.
