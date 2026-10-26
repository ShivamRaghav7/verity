# Dataset card: EditGuard controlled operations handbook

This is a new, independently authored **fictional demonstration dataset**, not a
collection of real company policies or a reuse of MQuAKE labels. Its purpose is
to measure how approved service-ownership changes affect direct and connected
questions while leaving other facts alone.

## Contents and ground truth

The versioned packet is `data/reasonke/operations-v1`. It contains 48 isolated
worlds, each with nine facts, an explicit ownership edit, and eight questions.
There are 384 question records. Each world describes a named service, a similarly
named Pro service, old/new owning teams and their on-call contacts, a hosting
region, a historical release owner and an unrelated records office.

An example relationship is:

```text
Service in release 2 → owning team → on-call person
```

Changing the first relationship changes both the direct owner answer and the
derived on-call answer. The old team's contact remains a valid fact about that
other team, making it a useful distractor. Release 1 is explicitly unaffected.

Each fact has an ID, entity ID, relation ID, scope ID, human-readable question,
answer, source and scope. Each edit records its original/corrected value,
approval state and lineage. Each question records its requested entity,
relation/scope, required supporting-fact IDs, current answer, initial answer,
fixed answer aliases and inapplicable edited-value aliases.

## Split and review

Lineage IDs were partitioned with seed 42 **before** constructing wording
variants: 16 development worlds and 32 final worlds. All questions, counterfactual
values and fact variants for a world stay in its partition. Service, team and
person names are disjoint across the partitions. Simple region/city labels may
repeat as values; shared templates are deliberately disclosed.

The packet's deterministic audit checks all 384 question-to-fact paths, the one
changed fact per world, supporting joins, budgets, category counts, and split
disjointness. Automated audit is not independent human review. No customer data,
private contacts, real service names or actual on-call assignments are used.

The 24-request profile covers selected development examples. The capped main
development comparison uses the same 24 questions per configuration: eight
categories over three predetermined development worlds. The full 128-row
development partition is not all executed in that comparison. No final answers
are used for model/prompt/retrieval/scoring selection.

## Final-study population

Each of the 32 final worlds contributes one question in each category:

| Category | Assigned questions per configuration |
| --- | ---: |
| Direct edit | 32 |
| Direct paraphrase | 32 |
| Two-hop edit | 32 |
| Two-hop paraphrase | 32 |
| Same entity, another property | 32 |
| Similar entity, same property | 32 |
| Another version/tier scope | 32 |
| Unrelated fact | 32 |
| Total | 256 |

The first four categories are applicable-edit questions; the others are locality
controls. Thus each main configuration has 128 applicable and 128 locality
questions. There are 64 paraphrase and 64 two-hop questions, overlapping on the
two-hop paraphrase category. These totals are assigned denominators, not scores.

The independent evaluation unit is the edit lineage, not each wording variant.
Paired uncertainty resamples the 32 lineages. Oracle diagnostics use a fixed
pre-inference selection of 16 two-hop questions per model and do not add new
independent question families.

## Fair use and limitations

All configurations have access to the same initial/supporting information.
Updated configurations use a resolved current fact view, with superseded values
replaced. Retrieval is question-only and cannot use hidden answers or gold
support IDs. Gold support is used only in the labelled oracle diagnostic.

Every assigned answer remains in the denominator, including abstentions,
execution errors and unexecuted cases. Primary scoring matches the complete
endorsed final span to fixed aliases for its requested entity/relation/scope.
This is a short-answer task with controlled language, not unrestricted dialogue.
Whole-answer matching can reject a semantically valid longer form; raw outputs
are retained for separate review, without test-guided alias changes.

The label audit is automated, the names and relations are synthetic, worlds are
small, and question templates recur. This supports a controlled experiment, not
claims about arbitrary companies, all paraphrases, large repositories, or lack
of overlap with unknown upstream model training. Shared component words are not
proof of a semantic entity-resolution test. No new model training uses this packet.

The manifest binds each input file and the authoring script with SHA-256 hashes.
The evaluation protocol separately freezes inputs, models, prompts, scoring and
inference sources before final inference. See [the implementation guide](OPERATIONS_BENCHMARK.md).
