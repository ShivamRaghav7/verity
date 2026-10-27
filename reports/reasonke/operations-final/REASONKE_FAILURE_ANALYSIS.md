# ReasonKE final-stage failure analysis

This analysis uses the completed frozen ReasonKE conditions. The matched cross-model comparison is reported separately in the full scorecard. No model calls, score changes, alias changes or answer retries were performed for this analysis.

## Evidence and population

- Raw ledger: `reports/reasonke/operations-final/final-reasonke.json`; SHA-256 `d6d8cad4a87e6036c2bd3ff0bb9cfd21fa15c2a37b7a801eb7b03ed15ca0af4b`.
- 32 held-out fictional lineages; 256 assigned questions per main condition, with 32 questions per category. The separate oracle diagnostic has 16 preselected two-hop questions.
- All 784 ReasonKE requests produced outputs. Execution-error, abstention and unexecuted counts are zero in this stage; wrong answers retain zero primary credit.
- R-RAG: 256/256 correct (128/128 applicable; 128/128 locality). R-ALL: 231/256 correct (112/128 applicable; 119/128 locality). R-ORACLE: 16/16 correct. These are distinct populations, not extra independent samples.

## Where full-memory answers failed

| Question category | Wrong / assigned | Applicable to the edit? |
| --- | ---: | --- |
| Direct ownership | 7/32 | Yes |
| Ownership paraphrase | 7/32 | Yes |
| Two-hop contact | 1/32 | Yes |
| Two-hop paraphrase | 1/32 | Yes |
| Same service, other property | 3/32 | No |
| Similarly named service | 2/32 | No |
| Older release | 3/32 | No |
| Unrelated office city | 1/32 | No |

R-ALL failed 16 applicable questions and 9 locality questions. Each failed question had all required supporting fact IDs among its nine supplied facts (25/25). Every wrong endorsed value exactly matched the answer value of another supplied fact (25/25). This is observable evidence of incorrect fact association at the answer level; it does not establish an internal attention mechanism or a causal training defect.

The wrong answer matched the Pro-service owner in 15 cases, the release-2 owner in 3 cases, an unrelated office city in 3 cases, an unrelated hosting region in 1 case, the previous owner team's contact in 2 cases, and the current owner team's contact when the question requested the team itself in 1 case. These groups follow exact supplied-fact value matches in this dataset; they are a post-hoc descriptive analysis and do not modify the scorer.

## Representative cases

### R-ALL:ops-006-direct

Question: Which team owns Garnet Relay in release 2?

Expected: **Silver unit 77**. Endorsed answer: **Saffron unit 137**.

The supplied context included:

- `ops-006-f1`: Which team owns Garnet Relay in release 2? -> Silver unit 77
- `ops-006-f5`: Which team owns Garnet Relay Pro in release 2? -> Saffron unit 137

### R-ALL:ops-045-direct

Question: Which team owns Juniper Scheduler in release 2?

Expected: **Cedar unit 116**. Endorsed answer: **Nira Wren**.

The supplied context included:

- `ops-045-f1`: Which team owns Juniper Scheduler in release 2? -> Cedar unit 116
- `ops-045-f3`: Who is the on-call contact for Cedar unit 116? -> Nira Wren

### R-ALL:ops-001-wrong_version_or_tier

Question: Which team owns Beryl Relay in release 1?

Expected: **Willow unit 192**. Endorsed answer: **Amber unit 72**.

The supplied context included:

- `ops-001-f6`: Which team owns Beryl Relay in release 1? -> Willow unit 192
- `ops-001-f1`: Which team owns Beryl Relay in release 2? -> Amber unit 72

### R-ALL:ops-005-same_entity_other_relation

Question: Which region hosts Fable Relay in release 2?

Expected: **South sector**. Endorsed answer: **Olven**.

The supplied context included:

- `ops-005-f7`: In which city is Kestrel records office 6? -> Olven
- `ops-005-f4`: Which region hosts Fable Relay in release 2? -> South sector

### R-ALL:ops-025-unrelated

Question: In which city is Garnet records office 26?

Expected: **Velora**. Endorsed answer: **South sector**.

The supplied context included:

- `ops-025-f4`: Which region hosts Beryl Registry in release 2? -> South sector
- `ops-025-f7`: In which city is Garnet records office 26? -> Velora

### R-ALL:ops-041-two_hop

Question: Who is the on-call contact for the team that owns Fable Scheduler in release 2?

Expected: **Ilan Wren**. Endorsed answer: **Tovan Solen**.

The supplied context included:

- `ops-041-f3`: Who is the on-call contact for Amber unit 112? -> Ilan Wren
- `ops-041-f2`: Who is the on-call contact for Silver unit 52? -> Tovan Solen
- `ops-041-f1`: Which team owns Fable Scheduler in release 2? -> Amber unit 112

## Preservation and correctness are separate

R0 supplied the initial handbook but was scored against post-edit targets: 136/256 correct overall, comprising 12/128 applicable answers and 124/128 locality answers. Its 120 wrong outcomes must not all be described as stale changed-fact answers: 116 were applicable and four were locality errors. R0 accuracy against its original-world targets was not separately evaluated by this scorecard, so 136/256 is not its original-handbook accuracy.

Among the 124 locality answers that R0 answered correctly, R-RAG retained 124/124 and R-ALL retained 117/124. Their overall locality accuracy is respectively 128/128 and 119/128. Therefore R-RAG also answered all four R0-incorrect locality cases correctly; R-ALL answered two of those four correctly while losing seven R0-correct locality cases. Preserving baseline errors would not be factual correctness.

The narrow false-correction metric counted 3/128 R-ALL locality questions endorsing one of the edited values, versus 0/128 for R-RAG. R-ALL had nine locality errors in total, so this narrow metric does not capture every incorrect fact association.

## Complete R-ALL wrong-answer catalogue

| Record ID | Expected | Endorsed actual | Exact matching supplied fact |
| --- | --- | --- | --- |
| R-ALL:ops-001-wrong_version_or_tier | Willow unit 192 | Amber unit 72 | ops-001-f1 |
| R-ALL:ops-005-same_entity_other_relation | South sector | Olven | ops-005-f7 |
| R-ALL:ops-006-direct | Silver unit 77 | Saffron unit 137 | ops-006-f5 |
| R-ALL:ops-007-similar_entity_same_relation | Amber unit 138 | Indigo unit 78 | ops-007-f1 |
| R-ALL:ops-014-direct | Silver unit 85 | Saffron unit 145 | ops-014-f5 |
| R-ALL:ops-014-wrong_version_or_tier | Amber unit 205 | Saffron unit 145 | ops-014-f5 |
| R-ALL:ops-025-unrelated | Velora | South sector | ops-025-f4 |
| R-ALL:ops-027-direct | Copper unit 98 | Cedar unit 158 | ops-027-f5 |
| R-ALL:ops-027-direct_paraphrase | Copper unit 98 | Cedar unit 158 | ops-027-f5 |
| R-ALL:ops-027-wrong_version_or_tier | Silver unit 218 | Cedar unit 158 | ops-027-f5 |
| R-ALL:ops-032-direct | Saffron unit 103 | Violet unit 163 | ops-032-f5 |
| R-ALL:ops-032-direct_paraphrase | Saffron unit 103 | Violet unit 163 | ops-032-f5 |
| R-ALL:ops-032-similar_entity_same_relation | Violet unit 163 | Saffron unit 103 | ops-032-f1 |
| R-ALL:ops-034-direct_paraphrase | Violet unit 105 | Willow unit 165 | ops-034-f5 |
| R-ALL:ops-036-direct_paraphrase | Willow unit 107 | Silver unit 167 | ops-036-f5 |
| R-ALL:ops-036-same_entity_other_relation | North sector | Lyris | ops-036-f7 |
| R-ALL:ops-037-direct | Cedar unit 108 | Indigo unit 168 | ops-037-f5 |
| R-ALL:ops-041-two_hop | Ilan Wren | Tovan Solen | ops-041-f2 |
| R-ALL:ops-041-two_hop_paraphrase | Ilan Wren | Tovan Solen | ops-041-f2 |
| R-ALL:ops-042-direct_paraphrase | Violet unit 113 | Willow unit 173 | ops-042-f5 |
| R-ALL:ops-044-direct_paraphrase | Willow unit 115 | Silver unit 175 | ops-044-f5 |
| R-ALL:ops-045-direct | Cedar unit 116 | Nira Wren | ops-045-f3 |
| R-ALL:ops-046-same_entity_other_relation | East sector | Tirren | ops-046-f7 |
| R-ALL:ops-047-direct | Indigo unit 118 | Amber unit 178 | ops-047-f5 |
| R-ALL:ops-047-direct_paraphrase | Indigo unit 118 | Amber unit 178 | ops-047-f5 |

## Supported interpretation and limits

- Within this controlled experiment, retrieved context improved ReasonKE final-answer accuracy over full memory by 25/256 = 9.765625 percentage points. The frozen lineage-bootstrap comparison reports R-ALL minus R-RAG = -9.765625 points, 95% interval [-13.671875, -6.25], using 32 lineages and 10,000 resamples (seed 42). This estimates this matched configuration difference on the controlled population, not arbitrary conversational reliability.
- Shared retrieval found 320/320 required supporting facts and complete support for all 64 two-hop questions in R-RAG. Retrieval accuracy and answer accuracy are separate measures. The nine-fact full-memory condition already contained the necessary facts, yet still produced wrong associations.
- R-RAG's 256/256 includes held-out entities and edit lineages, but wording templates and simple values repeat across splits. Labels were machine-audited without independent human semantic review. Short endorsed-answer aliases were fixed before final inference. This result does not demonstrate 100% performance on arbitrary paraphrases, longer chains, larger fact stores or unrestricted dialogue.
- The 16/16 oracle result is a small preselected generation diagnostic, not an independent larger accuracy claim. Correct final answers do not certify all generated explanation prose or faithful internal reasoning.
- Neither checkpoint was trained by this current study. No Qwen superiority/inferiority claim follows from this ReasonKE-only analysis.
