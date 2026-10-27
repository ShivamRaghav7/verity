# Final comparison: failure analysis and practical trade-offs

The frozen primary study completed all **1,312 assigned generations** over **32 held-out fictional edit lineages**: five matched conditions of 256 questions plus two preselected oracle diagnostics of 16 questions each. Every assigned output remains in the denominator. Execution errors and unexecuted records were zero across the primary study. Completed wrong or abstained answers were not retried. This analysis uses saved outputs and the fixed scorer; it makes no model calls or score changes.

## What the outcomes mean

| Configuration | Correct / assigned | Scored wrong | Abstained | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| R0 | 136/256 | 120 | 0 | 53.12% |
| R-RAG | 256/256 | 0 | 0 | 100.00% |
| R-ALL | 231/256 | 25 | 0 | 90.23% |
| G-RAG | 241/256 | 0 | 15 | 94.14% |
| G-ALL | 241/256 | 13 | 2 | 94.14% |

R-RAG and G-RAG received the same updated retrieved fact IDs; R-ALL and G-ALL received the full current nine-fact memory. R0 received initial facts and was scored against updated targets. Its 136/256 is a staleness diagnostic rather than separately measured accuracy against its original handbook: 12/128 applicable and 124/128 locality answers were correct.

**The retrieval-condition difference is entirely an abstention difference.** G-RAG produced 241 correct answers and 15 literal `UNKNOWN` responses, with zero incorrect endorsed answers. All 15 abstentions occurred among the 32 two-hop paraphrases: 17/32 correct, 15/32 abstained. The other seven categories were 32/32 correct each. R-RAG answered all 256 correctly. G-RAG therefore had 94.14% coverage and 241/241 answered-only precision; primary accuracy remains 241/256 because abstentions receive zero credit.

G-ALL also scored 241/256, but with a different reliability profile: 13 scored-wrong answers and 2 abstentions. Its coverage was 254/256 (99.22%) and its answered-only precision 241/254 (94.88%). Equal aggregate accuracy does not mean equal wrong-answer or abstention behavior.

## Qwen failures by category

Each category contains 32 assigned questions per condition. Cells show **scored wrong / abstained**, including zeros.

| Category | G-RAG | G-ALL |
| --- | ---: | ---: |
| Direct ownership | 0 / 0 | 2 / 0 |
| Ownership paraphrase | 0 / 0 | 3 / 0 |
| Two-hop contact | 0 / 0 | 2 / 1 |
| Two-hop paraphrase | 0 / 15 | 5 / 0 |
| Other property | 0 / 0 | 0 / 0 |
| Similar service name | 0 / 0 | 1 / 0 |
| Older release | 0 / 0 | 0 / 1 |
| Unrelated office city | 0 / 0 | 0 / 0 |

All 15 G-RAG abstentions and all 15 G-ALL nonsuccesses had every required supporting fact supplied. Across the complete retrieval condition, the shared retriever found 320/320 required facts and complete support for 64/64 two-hop questions. These outcomes occur after necessary context was supplied; they are not missing-required-support cases. This observation does not identify a model-internal causal mechanism.

Both oracle conditions scored 16/16 on the same preselected gold-support questions. Those 16 are a diagnostic subset, not targeted recovery attempts for the 15 G-RAG abstentions. Only one of those abstained question IDs belongs to the oracle subset, so the oracle result cannot be reported as a 15-case recovery result.

## Inspectable examples

The following are actual endorsed outputs; for these Qwen examples the saved raw output is the same short string. Full questions, supplied facts and generation metadata remain in the [Qwen receipt](final-instruction.json).

| Record ID | Question | Expected | Actual raw/endorsed output | Outcome |
| --- | --- | --- | --- | --- |
| G-RAG:ops-008-two_hop_paraphrase | In release 2, identify Ibis Relay's owning team and give only that team's on-call person's name. | Mara Reeve | `UNKNOWN` | abstain |
| G-ALL:ops-006-direct_paraphrase | For release 2 of Garnet Relay, name the team responsible for ownership. | Silver unit 77 | `Saffron unit 137` | wrong |
| G-ALL:ops-027-two_hop | Who is the on-call contact for the team that owns Dahlia Registry in release 2? | Orin Lorne | `Mara Arden` | wrong |
| G-ALL:ops-044-similar_entity_same_relation | Which team owns Ibis Scheduler Pro in release 2? | Silver unit 175 | `Willow unit 115` | wrong |
| G-ALL:ops-046-wrong_version_or_tier | Which team owns Kestrel Scheduler in release 1? | Amber unit 237 | `UNKNOWN` | abstain |
| G-ALL:ops-045-two_hop_paraphrase | In release 2, identify Juniper Scheduler's owning team and give only that team's on-call person's name. | Nira Wren | `Cedar unit 116 Nira Wren` | wrong |

- In `G-RAG:ops-008-two_hop_paraphrase`, supplied facts `ops-008-f1` and `ops-008-f3` explicitly connect Ibis Relay to Saffron unit 79 and that team to Mara Reeve. The output was `UNKNOWN` despite that support.
- In `G-ALL:ops-006-direct_paraphrase`, the output Saffron unit 137 matches the owner of Garnet Relay **Pro** (`ops-006-f5`), while the requested Garnet Relay owner is Silver unit 77 (`ops-006-f1`).
- In `G-ALL:ops-027-two_hop`, Mara Arden is the previous owner team's contact (`ops-027-f2`). The current ownership and contact facts (`ops-027-f1`, `ops-027-f3`) endorse Orin Lorne.

**Scoring limitation to disclose:** three G-ALL scored-wrong outputs concatenate a team and a person. Two (`ops-045-two_hop_paraphrase`, `ops-047-two_hop_paraphrase`) contain a compatible correct owner/contact chain, but violate the request to give only the person's name and fail the frozen whole-answer matcher. The third (`ops-005-two_hop_paraphrase`) combines the Pro-service owner with the correct requested contact. Consequently, all 13 scored-wrong cases must not be described as 13 false factual endorsements. These raw outputs remain visible; no aliases or scores were changed after the test.

The separate [ReasonKE analysis](REASONKE_FAILURE_ANALYSIS.md) catalogues all 25 R-ALL wrong answers and their supplied-fact associations. R-ALL had 16 applicable and 9 locality errors. R-RAG retained 124/124 R0-correct locality answers and achieved 128/128 overall locality accuracy; R-ALL retained 117/124 and achieved 119/128 locality accuracy. Preservation and factual correctness have different denominators.

## Accuracy, uncertainty and latency together

R-RAG exceeded G-RAG by **5.859375 percentage points** (15/256 additional correct answers). The paired 95% lineage-bootstrap interval is **[3.906250, 8.203125] points**, from 10,000 resamples of 32 lineages (seed 42). Wording variants share a lineage and are not independent samples.

| Configuration | Generated responses timed | p50 seconds | p95 seconds |
| --- | ---: | ---: | ---: |
| R-RAG | 256 | 15.0568 | 16.7135 |
| G-RAG | 256 | 0.9162 | 1.1981 |
| R-ALL | 256 | 15.3450 | 18.7162 |
| G-ALL | 256 | 1.4398 | 1.6594 |

The R-RAG p50 was **16.43 times** the G-RAG p50; its p95 was **13.95 times** the G-RAG p95. These are ratios of the reported latency quantiles as configured, not paired per-question speed ratios or normalized compute comparisons. Timings include context selection, local IPC and generation, including generated abstentions; they exclude loading and the website/browser round trip. The separate website workload measures operational HTTP behavior.

Full-memory Qwen exceeded full-memory ReasonKE by 3.90625 points on the fixed scorecard; the reported R-ALL minus G-ALL interval is [-7.8125, -0.390625] points. ReasonKE therefore did not win every configuration. The observed deployment choice combines high corrected-answer coverage with a substantial latency cost.

## Defensible conclusion

On this frozen controlled population, ReasonKE with updated retrieval answered all 256 questions correctly, including both edit and locality questions, while Qwen with the same retrieval abstained on 15 two-hop paraphrases. Correct supporting facts were available in those cases. Full memory introduced wrong associations for both checkpoints, and the strict scoring contract also rejected some overlong Qwen answers. The platform makes these outcomes inspectable alongside approved fact versions and reversible edits.

The evidence is bounded by 32 fictional lineages, nine-fact worlds, repeated wording templates/simple values, automated labels without independent human semantic review, fixed short-answer matching and single-operator deployment. Both checkpoints were trained upstream; comparing them does not isolate a particular training step. The 256/256 result supports this measured configuration and dataset, not guaranteed accuracy for arbitrary edits, paraphrases, longer reasoning chains or unrestricted conversations.

Sources: [aggregate metrics](results.json), [mechanical audit](host-audit.json), [ReasonKE raw outputs](final-reasonke.json), [Qwen raw outputs](final-instruction.json), and [fixed protocol](protocol.json).
