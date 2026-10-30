# Supplementary failure diagnosis for the technical presentation

This is a post-hoc descriptive analysis of saved final outputs, dated 30 October 2026. It makes **zero model calls**, preserves every frozen score and excludes R0 from the updated-method failure cohort because R0 is a staleness diagnostic against updated truth.

All **63 updated-method nonsuccesses** have their required support supplied and no token/time-budget-reached flag. This locates the observed failures after support selection under the tested configuration. It does not identify a causal model-internal mechanism.

| Cohort | Assigned main questions | Nonsuccesses | Observed description |
| --- | ---: | ---: | --- |
| Operations Reason-KE++ full memory | 256 | 25 | Every wrong final exactly matches another supplied fact value. |
| Operations Qwen retrieved | 256 | 15 | Literal UNKNOWN on two-hop paraphrases. |
| Operations Qwen full memory | 256 | 15 | 10 off-target fact values, 3 concatenations, 2 UNKNOWN. |
| Depth Qwen iterative | 256 | 8 | 7 direct two-hop UNKNOWN and 1 three-hop paraphrase UNKNOWN. |

## Full-memory Reason-KE++ associations

The exact matched fact roles are 15 Pro-service owners, 3 release-2 owners, 3 unrelated office cities, 2 previous-owner contacts, 1 hosting region and 1 current-owner contact. These roles follow the declared operations-world schema, not guessed semantic categories. They sum to 25. Sixteen failures are edit-applicable and nine are locality questions.

In `R-ALL:ops-006-direct`, the requested Garnet Relay release-2 owner is Silver unit 77, but the answer is Saffron unit 137 from Pro-service fact f5. The generated explanation explicitly cites f5 while treating it as the requested service. In `R-ALL:ops-045-direct`, the correct owner Cedar unit 116 appears as a boxed subanswer, followed by the contact Nira Wren as the final answer to a team question. Two wrong-final records contain the expected value in an action-stage boxed answer under the conservative parser. This is a text diagnostic, not certification of all intermediate reasoning.

## Qwen: abstention versus strict mismatch

The 15 original retrieved abstentions all occur in the 32 two-hop paraphrases. In the depth set, template `final:two_hop:1` has 7 abstentions in 9 direct questions. Other direct two-hop templates are 23/23 correct, and all 32 two-hop paraphrases are correct. The three-hop paraphrases are 31/32. These post-test concentrations do not prove a causal wording effect and do not authorize prompt retuning on the final set.

The 13 strict G-ALL mismatches include ten exact off-target fact values. Two other outputs contain a compatible correct owner/contact chain but add owner text despite a person-only request (`ops-045` and `ops-047` two-hop paraphrases). A third joins a Pro owner with the correct requested contact (`ops-005`). The fixed primary scorer still labels all thirteen wrong. We do not add aliases or turn those cases into successes afterward.

The original oracle set includes only one original G-RAG failure question ID. The depth oracle set contains none of its eight main failure IDs. Their 16/16 outcomes therefore cannot be described as recovery of the complete failure cohorts.

## Reproduce and inspect

- `failure-audit.json` contains all 63 records, raw output, supplied facts, unchanged outcomes and input hashes.
- `failure-ledger.csv` is the compact per-record index.
- Run `python -m scripts.analyze_presentation_failures --output NEW_DIRECTORY` from the repository root. Existing output directories are refused.
- The generator imports the existing string normalizer for descriptive value association only. It does not call or change the frozen scorer.
- The three concatenation diagnoses use the specified inspected IDs and must not be presented as a general semantic judge.

These analyses support descriptive failure findings in slides 20–21 and backup slide 31. They do not establish arbitrary-language accuracy, independently reviewed labels, faithful explanations, causal effects of the training algorithm, or large-memory reliability.
