# Storage capacity alias in scorer v3

`editguard-short-facts-v3` recognizes one additional complete assertion form:
`<entity> <plan> <version> supports <value> <unit> of storage`.
The rule requires the entire assertion and the existing reviewed entity aliases,
plan, version, numeric value, unit and question polarity checks. The phrase
`of storage` is mandatory. It does not infer correctness from a retrieved source,
a target-number substring or an arbitrary generated sentence.

This is a development-discovered scoring correction, before final configuration
and test freeze. Four saved candidate oracle answers used this form with a
correct `No` and fully scoped factual continuation. Scorer v2 left those answers
pending review. They were not classified as incorrect. Their requests were
21/26 GB against the Aster 044 quotas of 20/25 GB, and 26/41 GB against the
Aster 045 quotas of 25/40 GB. Changing scoring does not change the generated
answers, targets, training data or model weights.

The controlled handbook represents `storage_gb` as a plan storage quota, measured
in GB. The existing questions ask the storage quota or whether a requested amount
can be stored; `supports N GB of storage` supplies that same capacity relation.
This interpretation is specific to the controlled short factual support task.
It does not establish that unspecified hardware capabilities, RAM amounts,
per-device allowances or arbitrary uses of the verb “supports” mean a plan quota.
See [the dataset card](DATASET_CARD.md), `scripts/prepare_data.py`'s storage
question forms, and `editguard/short_qa.py`'s bounded capacity-question mapping.

The exact previous scorer bytes are preserved in
`reports/scoring-source-v2/scoring.py`, with SHA-256
`b2c595a7d5ae9efc463a86c4ad6498553a7efa536a48152cef2afd2ac31b0ab1`
and a separate archive manifest. Historical evaluations retain their original
scores and source hashes. Fresh rescoring must apply v3 to all compared
configurations under the same rules and save new artifacts. Attribute any
result difference caused solely by this rule to scoring coverage, rather than
to a model improvement. Remaining unrecognized answers retain pending review
and receive no automatic accuracy credit.

`tests/test_storage_endorsement_alias.py` checks complete direct and yes/no
answers, reviewed aliases, wrong scope/value/unit/property, negation, examples,
conditionals, rejected possibilities, missing fields and contradictory answers.
It also imports the archived v2 scorer to verify the explicit v2-pending /
v3-credited difference and unchanged existing assertion verdicts. No temporary
grammar patch or alternate target-matching path is used by these tests.
