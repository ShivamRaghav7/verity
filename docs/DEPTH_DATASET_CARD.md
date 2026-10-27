# Separate depth challenge: controlled Indian-context data

This is a new dataset, **editguard-depth-v1**, rather than a change to the
completed operations-v1 study. Its purpose is to examine a meaningful boundary:
does an ownership correction propagate through one, two and three connected
relations, and does failure arise before or after the necessary facts reach the
model? A score below 100% is not a design objective. No model outcome was used to
construct this packet, and no performance number is asserted by its existence.

## Content and ground truth

The dataset has **48 independent fictional edit lineages**, twelve facts per
lineage and eight questions per lineage: 384 question records in total. The
service, team, person and office assignments are invented. Indian river/range
names, plausible person names and city labels make the examples familiar to an
Indian audience without claiming actual company or government arrangements.
The fact source explicitly identifies every assignment as controlled fiction.

Each graph contains four distinct chains: old ownership, updated ownership,
similarly named Pro-service ownership and historical release-1 ownership. Only
the release-2 ownership fact changes. The old and new teams' contacts and those
people's office cities remain independently valid throughout the edit.

```text
service / release 2 -> owner team -> on-call person -> office city
                    one hop       two hops          three hops
```

| Question family | Questions per world | Required current facts |
| --- | ---: | --- |
| Direct owner and paraphrase | 2 | ownership |
| On-call contact and paraphrase | 2 | ownership + team contact |
| Contact's office city and paraphrase | 2 | ownership + team contact + person office |
| Historical release-1 owner | 1 | separate historical ownership |
| Same service's hosting zone | 1 | separate property |

Every row includes requested subject/relation/scope, expected whole-answer
aliases, initial answer, edit applicability, category, depth and ordered gold
support IDs. The two locality categories have `depth: 0` to distinguish controls
from the edit-affected chain; each still needs its one explicit fact. Fact rows
also carry subject and object entity IDs, allowing the audit to check actual
joins rather than merely counting three retrieved strings.

## Splits and wording

Seed **43** assigns sixteen development and thirty-two final lineages **before**
constructing wording variants. Development contains 128 questions; final
contains 256. Each final category has 32 questions. Questions from one edit
lineage cannot appear in both splits. Service, team and person subject aliases
are disjoint between the partitions and are also distinct from the website's
India demonstration entities and the original operations-v1 service names.

Development and final use disjoint authored question-template banks, with three
constructions per category per split. This is a stronger wording separation
than entity substitution alone, but templates still repeat *within* a split.
Canonical fact syntax, relation vocabulary, generic release labels and simple
city values can repeat across splits. The disjoint scope assertion concerns
the composite **entity + relation + scope identity**, not the generic strings
`release-1`, `release-2` or `current`. No claim of foundation-pretraining or
upstream fine-tuning disjointness is made.

The packet has a new final population. It must receive its own frozen protocol
before final inference. The completed original scorecard remains unchanged;
its questions and this challenge must not be pooled into a single improved
headline accuracy. Increasing hop depth does not create independent samples:
the primary independent unit remains the whole lineage, not its eight related
questions or repeated model configurations.

## Authoring and automated review

The authoring module is [editguard_depth/data.py](../editguard_depth/data.py).
It uses explicitly authored templates and deterministic graph construction,
not an LLM that invents labels without checks. The audit validates both current
and initial paths for all 384 questions (768 path checks), exactly one changed
fact, join identities and display labels, requested relation/scope, fixed
current/initial support paths, unambiguous within-world aliases, category counts,
split membership and cross-split alias/composite-key separation.

The manifest records separate factual-input, label and wording-bank SHA-256
identities, file checksums and generator-source identity. Loading verifies those
identities and reconstructs the path audit. Writing refuses an existing output
directory. Tests deliberately corrupt labels, scope, joins, support branches and
file bytes; these are software fixtures, not model performance measurements.

**Review is automated. There is no independent human semantic review claim.**
The city labels are familiar names, but their fictional office assignments are
ground truth only inside these controlled graphs. The packet defines no
current real-world policy, historical fact update or actual person's workplace.

## Use and limitations

```text
python -m editguard_depth.data --output <new-packet-directory>
python -m unittest tests.test_depth_data -q
```

The expected shared-retrieval and oracle-support experiments are documented in
[the finite run plan](DEPTH_RUN_PLAN.md). The retriever must not receive gold support IDs except
in an explicitly named oracle condition. Report three-hop retrieval completeness
separately from answer accuracy; otherwise a missing third fact can be mistaken
for failed reasoning. Models should receive matched information and budgets.

This is a bounded synthetic study with 32 final lineages, repeated templates,
short fixed answer aliases and twelve-fact worlds. It does not establish
arbitrary paraphrase robustness, unrestricted conversation, longer graph-depth
reliability or behavior at a production-scale memory size. All assigned
abstentions, wrong answers, errors and unexecuted cases must remain visible.
The original 256-question perfect retrieval result is neither erased nor used
as a target that this separate challenge must artificially lower.
