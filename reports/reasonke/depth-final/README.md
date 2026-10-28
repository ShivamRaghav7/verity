# Completed separate depth challenge

All **544 assigned final generations** completed on a Tesla T4. The main
comparison contains the same 256 questions over 32 held-out fictional graph
lineages per model. Each model also receives 16 preselected three-hop questions
with gold support. These diagnostics are separate from the primary denominator.

| Configuration | Correct / assigned | Wrong | Abstain | Execution errors | Unexecuted |
| --- | ---: | ---: | ---: | ---: | ---: |
| ReasonKE + iterative retrieval | 256/256 | 0 | 0 | 0 | 0 |
| Qwen + same iterative retrieval | 248/256 | 0 | 8 | 0 | 0 |
| ReasonKE + gold support | 16/16 | 0 | 0 | 0 | 0 |
| Qwen + gold support | 16/16 | 0 | 0 | 0 | 0 |

ReasonKE scored 64/64 in each main depth group. Qwen scored 64/64 on locality
controls, 64/64 on one-hop, 57/64 on two-hop and 63/64 on three-hop questions.
The selector supplied all 448 required fact occurrences and all 256 complete
paths for each main model condition. It is the same selector, not two independent
retrieval experiments. Abstentions receive zero primary accuracy credit.

The paired main difference is **+3.125 percentage points**, with a 95% bootstrap
interval of **+1.5625 to +5.078125 points**, resampling whole lineages rather
than individual wording variants. The three-hop difference interval includes
zero. Warm main-request p50/p95 latency is **15.94/19.28 seconds** for ReasonKE
and **1.12/1.38 seconds** for Qwen, including selection, local IPC and generation,
excluding loading, downloads and browser network. Both loading peaks allocated
6.59 GiB. Full raw resource and timing samples remain available.

The [frozen protocol](protocol.json) binds the source, dataset, configurations,
scoring and four profile/development receipts. It was published in commit
`08d274e204e0dc8c7fcff94cc3aabcefda0debef` before final inference. The laptop
[host audit](host-audit.json) mechanically replayed all extraction, identities,
retrieval and scores with zero model calls. Each response was saved atomically;
completed failures were never retried for a better answer. An earlier expired
development session lost its unsaved outputs; its unknown outcomes are disclosed
and excluded, with preserved profiles restored unchanged for replacement development.

Read the [five-page report](../../../output/pdf/editguard-depth-study.pdf),
[measured Markdown report](derived/REPORT.md), [scorecard CSV](derived/scorecard.csv),
[retrieval CSV](derived/retrieval.csv), [latency and resource CSV](derived/latency.csv)
and [figure provenance](derived/figure-manifest.json). The plain-language
[extension guide](../../../docs/UNDERSTAND_DEPTH_EXTENSION.md) explains the
timeline, metrics and India website workflow.

The [portable evidence ZIP](../../../output/depth-2026-10-28/editguard-depth-evidence.zip)
and [file-hash manifest](../../../output/depth-2026-10-28/manifest.json) contain
the released evidence, report and reproduction helpers. Model weights, access
tokens and the user chat database are excluded.

This study does not replace or pool the completed operations-v1 results. These
short fictional assignments, repeated within-split templates, automated labels
and twelve-fact worlds limit generalization. There was no independent human
semantic review. A correct final answer does not certify its explanation.
Both checkpoints were trained upstream; this project integrates and evaluates
them with versioned external memory and does not claim new checkpoint training,
per-correction weight updates, universal accuracy or faithful reproduction of a
published paper score. The India website acceptance workload is a separate
development demonstration, not extra held-out benchmark evidence.

## Completed India website acceptance

The separate fictional display workload completed **14/14 neural answers** and
all **12 transaction/provenance consistency checks**. It covers initial direct,
two-hop and three-hop answers; an approved ownership replacement; paraphrase,
locality and historical-release questions; historical/current office comparison;
and atomic rollback followed by restored answers. An unresolved competing
approval was rejected without changing memory. The initial nine facts were
restored at revision 11, with versions and conversations retained.

The [raw website receipt](india-acceptance.json) was mechanically replayed on the
laptop by the [India host audit](india-host-audit.json), checking pinned model
identity, raw-output extraction, iterative memory provenance and all scores
without model calls. Fourteen generations used thirteen HTTP generation requests
because one comparison generated two answers. Loopback request p50/p95 was
17.11/33.39 seconds (n=13), including the two-generation comparison. This is not
per-generation latency or fourteen independent factual lineages. It adds no
held-out benchmark observations and does not certify arbitrary generated prose.
