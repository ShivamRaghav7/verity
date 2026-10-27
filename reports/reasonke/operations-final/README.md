# Frozen final operations study

The final protocol was sealed before held-out inference on 26 October 2026.
Its SHA-256 is
`23914664df3378be46be463ea60cd8b5b258eb13b6b55c68b38f76d3b15c94c5`.
`protocol.json` is the actual exported Colab input, including source/config/data
hashes, the two real profiles, development policy selection and the preselected
explanation-audit population. Its source hashes were checked against the host.

The finite T4 workload was launched: 784 ReasonKE requests, then 528 official
Qwen instruction requests. The five main conditions each have 256 questions
from 32 held-out fictional lineages; the two oracle diagnostics have 16 questions
per model. No wrong-answer retries, parameter search or test-guided scoring
changes are permitted. Each assigned outcome remains in its receipt ledger.

## Completed ReasonKE stage: 27 October 2026

All 784 ReasonKE outputs were generated and independently replay-audited on the
laptop. No execution errors or unexecuted records occurred in this stage.

| Condition | Correct / assigned | Wrong | Abstained |
| --- | ---: | ---: | ---: |
| R0, initial facts scored against updated targets | 136/256 | 120 | 0 |
| R-RAG, updated retrieved facts | 256/256 | 0 | 0 |
| R-ALL, updated full memory | 231/256 | 25 | 0 |
| R-ORACLE, preselected gold supporting facts | 16/16 | 0 | 0 |

`final-reasonke.json` preserves every output. `reasonke-stage-audit.json` checks
source/data/configuration identities, request context, provenance, extraction
and frozen scoring without a model call. `resume-audit.json` verifies exact
equality of all 483 answers saved before the user-requested pause; the remaining
301 requests completed after resumption. Completed wrong answers were retained.

The 528-request Qwen stage is running separately. The full matched comparison,
aggregate and scientific exports follow both completed ledgers and their audit.
The website continues to label the development study until that full result is
available. The bounded 256-question retrieval result is not a guarantee for
arbitrary language, edits or dialogue, and does not establish superiority over
Qwen before its matched results are available.
