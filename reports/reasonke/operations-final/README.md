# Completed frozen operations evaluation

Execution: completed; 1312 assigned generations across 32 held-out fictional edit lineages.

Settings, source/data identities and scoring were frozen before final inference. All wrong answers, abstentions, execution errors and unexecuted records retain denominator credit. No benchmark question was retried for a better answer.

| Condition | Correct / assigned | Wrong | Abstained | Errors | Unexecuted |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0 | 136/256 | 120 | 0 | 0 | 0 |
| R-RAG | 256/256 | 0 | 0 | 0 | 0 |
| R-ALL | 231/256 | 25 | 0 | 0 | 0 |
| R-ORACLE | 16/16 | 0 | 0 | 0 | 0 |
| G-RAG | 241/256 | 0 | 15 | 0 | 0 |
| G-ALL | 241/256 | 13 | 2 | 0 | 0 |
| G-ORACLE | 16/16 | 0 | 0 | 0 | 0 |

The separate explanation audit retains 32 preselected traces. Only bounded explicit indexed subanswers are assessed; surrounding prose and internal reasoning are not certified.

`final-reasonke.json` and `final-instruction.json` preserve all outputs; `protocol.json` identifies the actual frozen inputs. `host-audit.json` checks identities, extraction and scoring without a model call. `derived/` contains the measured report, CSV and scientific PNG/SVG figures.

`http-acceptance.json` contains twelve development-world website requests. `history-acceptance.json` contains four further generations for historical comparison and atomic rollback. These operational workloads are not extra independent held-out data and do not alter the primary scorecard. Dataset review and explanation review are automated, without independent human semantic review.

## Completed website and rollback workload

The separate operational workload recorded **12/12 correct workflow answers and
4/4 correct historical-comparison/atomic-rollback answers**. All eleven recorded
history/rollback consistency checks passed; cleanup errors, execution errors and
unexecuted operational cases were zero. The actual twelve-request website
p50/p95 was **15.42/18.78 seconds**, including storage and worker HTTP. These
development-world generations are separate from the held-out scorecard.
`operational-host-audit.json` independently replays all sixteen raw-output
extractions, model identities and fixed scores without a model call.

## Read and reproduce the evidence

- [Ten-page final report](derived/study.pdf), [report manifest](derived/study-manifest.json)
  and [measured Markdown report](derived/REPORT.md).
- [Full failure analysis](FAILURE_ANALYSIS.md) and [all 25 full-memory ReasonKE errors](REASONKE_FAILURE_ANALYSIS.md).
- [Independent laptop replay](independent-host-audit.json), in addition to the
  original finalizer's `host-audit.json`.
- [Exact pause/resume preservation](resume-audit.json): all 483 previously
  completed records stayed unchanged; only 301 unfinished ReasonKE cases ran.

Figures were redrawn on the laptop to remove overlapping scatter labels. Their
manifest records the actual plotting source, library version and original raw
receipt hashes. No model output, score, population or frozen inference setting
changed during graph/PDF publication.

Both checkpoints were trained upstream. Context policy was selected on
development before final inference. The measured ReasonKE retrieval advantage
over Qwen retrieval was +5.86 percentage points (95% lineage-bootstrap interval
+3.91 to +8.20); it came from fifteen Qwen abstentions. ReasonKE was slower and
scored below Qwen with full memory. Shared templates, automated labels and strict
short-answer matching limit the conclusion; there is no universal accuracy,
arbitrary-edit safety, faithful-explanation or causal-training claim.
