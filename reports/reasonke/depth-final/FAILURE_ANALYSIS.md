# Failure analysis: separate frozen depth study

The completed run recorded **544/544 assigned outcomes**: 256 main questions and 16 preselected three-hop oracle questions for each checkpoint. CPU replay reproduced the frozen task, request, routing, model provenance, extraction and scoring checks. This analysis makes no new model calls or scoring changes. The original operations-v1 study stays separate.

## Every outcome stays in its denominator

| Condition | Correct / assigned | Wrong | Abstain | Execution errors | Unexecuted |
| --- | ---: | ---: | ---: | ---: | ---: |
| ReasonKE + iterative retrieval | 256/256 | 0 | 0 | 0 | 0 |
| Qwen + iterative retrieval | 248/256 | 0 | 8 | 0 | 0 |
| ReasonKE + gold support | 16/16 | 0 | 0 | 0 | 0 |
| Qwen + gold support | 16/16 | 0 | 0 | 0 | 0 |

Qwen's main accuracy and answer coverage are both **96.875% (248/256)**. Its answered-only precision is 248/248, but that filtered figure does not replace the assigned denominator. An `UNKNOWN` response receives zero primary credit. It is a generated abstention, not a crash, a completed false factual answer or an omitted request. The eight abstentions appear in the raw ledgers unchanged.

## All eight non-correct records

Every row below has literal `UNKNOWN` as both the raw model output and extracted final response. None reached its token or generation-time budget. Seven are two-hop questions; one is a three-hop paraphrase. All are from the main retrieval condition; neither oracle condition contains a failure.

| Record ID | Category | Expected endorsed answer | Actual | Required support present |
| --- | --- | --- | --- | ---: |
| `G-RAG:depth-001-two_hop` | two_hop | Diya Desai | `UNKNOWN` | 2/2 |
| `G-RAG:depth-004-two_hop` | two_hop | Arjun Desai | `UNKNOWN` | 2/2 |
| `G-RAG:depth-016-two_hop` | two_hop | Arjun Nair | `UNKNOWN` | 2/2 |
| `G-RAG:depth-024-three_hop_paraphrase` | three_hop_paraphrase | Indore | `UNKNOWN` | 3/3 |
| `G-RAG:depth-025-two_hop` | two_hop | Diya Kapoor | `UNKNOWN` | 2/2 |
| `G-RAG:depth-028-two_hop` | two_hop | Arjun Kapoor | `UNKNOWN` | 2/2 |
| `G-RAG:depth-031-two_hop` | two_hop | Ishita Kapoor | `UNKNOWN` | 2/2 |
| `G-RAG:depth-040-two_hop` | two_hop | Arjun Menon | `UNKNOWN` | 2/2 |

The two-hop failures all use wording template `final:two_hop:1`: “Identify [service]'s release-2 owning team and give only its on-call contact's name.” Qwen abstained on **7 of 9** questions instantiated from this template; it answered the other two correctly. The other two direct two-hop templates produced **23/23 correct** results. This is a descriptive wording concentration observed after evaluation, not a proven causal diagnosis or permission to revise the frozen prompts. No alternate prompts or retries were run for these failures.

| Depth group, per main condition | ReasonKE correct / assigned | Qwen correct / assigned | Qwen abstentions |
| --- | ---: | ---: | ---: |
| Locality controls (depth 0) | 64/64 | 64/64 | 0 |
| One hop | 64/64 | 64/64 | 0 |
| Two hops | 64/64 | 57/64 | 7 |
| Three hops | 64/64 | 63/64 | 1 |

The 32 direct and 32 paraphrased two-hop questions combine to the 64-question two-hop group. Qwen answered all 32 two-hop paraphrases correctly. Its three-hop direct questions scored 32/32 and three-hop paraphrases scored 31/32. The deeper group therefore had a higher observed score than the two-hop group; these data do not support a claim that error must rise monotonically with hop count. Wording, entity assignments and distractor interactions vary as well.

## Retrieval succeeded; answer production still abstained

The iterative selector supplied **448/448 required fact occurrences** and complete support for **256/256 main questions**, identically for both checkpoints. Within three-hop questions it supplied **192/192 required occurrences** over **64/64 complete paths**. Duplicating the same selector with two generators is not two independent retrieval studies.

Every abstention had all its required support in the actual supplied context. The observed failures therefore occur after successful support selection, at answer production under the fixed prompt/context/decoding configuration. This separates them from missing-support retrieval failures. The receipts do not reveal a causal internal mechanism; they cannot distinguish a learned abstention preference from sensitivity to wording or distractors without a separate planned experiment.

For `G-RAG:depth-001-two_hop`, the supplied facts explicitly say Narmada Gateway's release-2 owner is Vindhya unit 111 and that unit's on-call contact is Diya Desai. The expected short answer is therefore Diya Desai, but Qwen generated `UNKNOWN`. Its four-fact context also contains the Pro service owner and the selected contact's office city. These were the frozen selector's output, not facts injected after observing failure.

For `G-RAG:depth-024-three_hop_paraphrase`, the supplied chain is Kaveri Scheduler -> Malwa unit 134 -> Aarav Kapoor -> Indore. Qwen again generated `UNKNOWN`; ReasonKE returned `Indore` on the same main question. ReasonKE's saved explanation includes the explicit subanswers `Malwa unit 134`, `Aarav Kapoor` and `Indore`. This is one illustrative inspected trace, not a semantic audit of all explanations or a claim that generated prose is fully verified.

Both models answered the 16 preselected gold-support three-hop requests correctly. **None of the eight main failure question IDs is in that oracle subset.** Consequently the oracle result cannot be described as repairing or explaining those eight failures. It measures a separate fixed subset. Gold support also changes the supplied context, so its 16/16 cannot replace main retrieval accuracy.

## Accuracy has a measured cost

ReasonKE's main warm pipeline p50/p95 was **15.94/19.28 seconds (n=256)**; Qwen's was **1.12/1.38 seconds (n=256)**. The Qwen latency population includes all eight generated abstentions. Times include support selection, local IPC and generation; model loading, downloads and browser network are excluded. On this controlled population, ReasonKE produced eight more correct main answers while taking longer. This is a measured trade-off rather than a general claim of model superiority.

The paired main accuracy difference was **+3.125 percentage points**, with a **95% lineage-bootstrap interval [1.5625, 5.078125]** over 32 lineages. The three-hop difference was **+1.5625 points**, interval **[0, 4.6875]**. Lineages, not wording variants, are the resampled units (10,000 draws, seed 42). Intervals summarize this sampled population; they do not prove universal reliability or causally isolate upstream training.

## Why a perfect ReasonKE score remains limited

No non-correct ReasonKE final answers were observed, and no failures are invented to make the study look harder. Its 256/256 is the measured whole-final-answer alias result for this frozen packet. The packet has only 32 final graph lineages, twelve facts per world and repeated templates within each split. City labels and relation vocabulary repeat even though service/team/person identities and wording banks are separated by development/final split. All labels and ground-truth paths were audited automatically, without independent human semantic review.

The scoring checks the extracted endorsed final span against frozen controlled aliases. It does not certify all explanation sentences, arbitrary paraphrases, ordinary conversation or inference over a large production corpus. The recorded 64/64 locality-control answers per model establish factual correctness on those controlled questions; this new study has no unchanged-base condition, so it does not independently estimate baseline-correct preservation. The original operations study's preservation results remain separately reported.

The models were trained upstream. This project did not fine-tune these checkpoints or change their weights for new corrections. The comparison uses fixed pinned checkpoints, matched facts and inference budgets, but the Qwen checkpoint is not a verified causal ablation of ReasonKE's training. Published upstream metrics are not substituted for these results. Runtime and database/rollback checks are operational evidence, separate from neural answer accuracy.

## Preservation and recovery

`resume-audit.json` binds the seven raw/protocol file hashes and compares saved checkpoints 8, 9, 10 and 11 with the completed ReasonKE ledger. All previously recorded rows are unchanged (174, 197, 235 and 247 rows respectively). The observed browser disconnection did not require an evaluator restart. An earlier temporary development runtime expired before export; its unavailable outcomes are disclosed and are not reconstructed or counted in the successful replacement development denominator.

Reproduce the mechanical final audit with:

```text
python -m editguard_depth.metrics reports/reasonke/depth-final/final-reasonke.json reports/reasonke/depth-final/final-instruction.json --protocol reports/reasonke/depth-final/protocol.json --output NEW_AUDIT_RESULTS.json
```

Use a new output path; preserve the released files. The raw final receipts contain every generated answer, support fact, latency, token count and resource record. `derived/REPORT.md`, CSVs and figures summarize those measurements without new inference.
