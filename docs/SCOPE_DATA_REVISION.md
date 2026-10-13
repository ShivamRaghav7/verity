# Bounded scope-data revision: pilot-v3

The initial pilot's validation analysis found 12 missed applicable direct
questions. It also exposed a data confound: all 144 training negatives and all
72 validation negatives used direct questions, while applicable edits appeared
in direct, paraphrased and comparison forms. Direct questions consequently had
a 14.3% applicable prior; the other forms had a 100% applicable prior. This
supports testing a specific data repair. It does not prove that question form
caused every model error.

Pilot-v3 revises these existing negatives without expanding the catalogue or
adding examples. It retains the same six entity families, 36 edit worlds,
360 scope rows, 144 answer rows, split assignments and counterfactual lineages.
The runtime input format remains `editguard-input-v2`. The v2 data, reviews,
training artifacts and measured results remain intact.

Each split sorts its edit identifiers and rotates the six existing negative
types through four existing question forms:

```text
style = (sorted_edit_index_within_split + negative_type_index) % 4
```

The negative-type order remains same entity/different property, different
entity/same property, similar name, wrong tier, wrong version and ordinary.
The four forms are direct, alias paraphrase, comparison with a yes answer and
comparison with a no answer. Applicability is still zero for every negative;
the comparison's yes/no answer refers to the queried fact, not applicability.

| Split | Form | Applicable | Inapplicable | Applicable prior |
|---|---|---:|---:|---:|
| Train | Direct | 24 | 36 | 40% |
| Train | Alias paraphrase | 24 | 36 | 40% |
| Train | Comparison: yes | 24 | 36 | 40% |
| Train | Comparison: no | 24 | 36 | 40% |
| Validation | Direct | 12 | 18 | 40% |
| Validation | Alias paraphrase | 12 | 18 | 40% |
| Validation | Comparison: yes | 12 | 18 | 40% |
| Validation | Comparison: no | 12 | 18 | 40% |

The queried fact's original catalogue value determines each negative comparison.
For device, storage and support limits, comparison amounts equal the value
for yes and exceed it by one for no. For monthly price, the question asks
whether the price is *at most* an amount: yes uses the value, and no uses one
less. These directions are checked independently against structured ground truth.
Aliases come from that queried entity's reviewed-data candidate catalogue record.

Actual derivation changed 162 question strings. All 216 negatives changed their
question-type metadata from `out_of_scope` to the actual form and gained
`expected_polarity` metadata: 108 null entries and 108 yes/no entries. Every
scope label, requested subject/relation, fact value, correction, scope and
lineage stays fixed. All applicable questions and all answer-model inputs and
targets remain byte-identical as field values. Catalogue and split files are
copied byte-for-byte. Every one of the 504 rows receives a new `pilot-v3` ID,
content hash and pending decision; previous reviews cannot silently authorize
the revised dataset.

Run this finite, CPU-only derivation in the app environment:

```powershell
.\.venv\Scripts\python.exe scripts/balance_scope_questions.py
```

Defaults read immutable `data/generated/pilot-v2`, publish a fresh
`data/generated/pilot-v3`, and write `reports/pilot-v3-manifest.json` and
`reports/scope-balancing.json`. Existing or partial destinations are refused.
To reproduce separately, supply fresh `--output`, `--manifest-report` and
`--report` paths. Sources with frozen-test markers, test/challenge files,
unknown files, incomplete worlds or incompatible schemas are refused before
unsupported files are read. Publication stages complete bytes and renames the
directory; a failed publication cleans only the checked staging directory.
Source hashes are rechecked before publication.

The manifest binds the source dataset, every source-file hash, derivation code,
new files and per-row lineage. The actual v3 manifest SHA-256 is
`5e4790e2edf9c15e98f37ce455ab275a387fa1ae866c35cde40a0322de62dc2e`.
Six automated regression tests verify matched priors, original-value comparison
semantics, source/answer/fact immutability, reproducibility, invalid-source refusal
and atomic failure cleanup using separate synthetic fixtures.

This derivation approves no reviews, loads no models and uses no GPU. The new
504 decisions must be independently reviewed before training eligibility.
That subsequent [fresh machine audit](PILOT_V3_REVIEW.md) is now complete:
504 accepted, zero flagged. A separate four-step resource profile and run-card
preflight prepared [scope revision 1](SCOPE_REVISION_TRAINING.md) for user execution.
The derivation itself still approves no reviews and produces no trained model.
Automated development review must retain its explicit provenance and cannot
be reported as human review. No test or challenge corpus was read or evaluated.
Question forms and catalogue size remain limited; removing this observed
shortcut does not establish broad natural-language reliability. A subsequent
finite validation experiment should resolve whether direct-edit detection
improves while false applications and other question forms remain acceptable.
