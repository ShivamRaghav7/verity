# Separate depth challenge: development preparation

The replacement development evaluation completed on 28 October 2026. Both
checkpoints scored **24/24**, with zero wrong answers, abstentions, execution
errors or unexecuted rows. The same 24 questions cover three development
lineages, including six questions at each depth/control group. These are
development measurements, not held-out accuracy. Raw ledgers and the replayed
`development-results.json` retain every answer and its provenance.

ReasonKE's warm p50/p95 was 15.25/17.99 seconds; Qwen's was 1.10/1.37 seconds
(24 generated requests each; includes retrieval/IPC, excludes model loading).
The [sixteen-request profiles](../depth-profiles-v1/README.md) and measured
budgets remain separate. No final results guided configuration or scoring.
The completed operations-v1 study and its results are preserved unchanged.

An earlier development session expired before results were downloaded. Its
actual outcomes are unavailable and are not claimed as evidence. The saved
profiles were restored unchanged, and the declared development population was
run again as an explicitly disclosed replacement rather than an outcome-based
retry. The final protocol binds only the complete replacement ledgers.

`retrieval-selection.json` records a reproducible development-only comparison
of six bounded lexical policies. The selected policy supplies at most four
facts, starts from two lexical seeds and expands through answer text for at
most three rounds. Its development support coverage is 128/128 packet questions
plus both declared India display states (230/230 required fact occurrences).
These are retrieval diagnostics, not generated-answer accuracy or held-out
results. The original two-stage retriever supplied all support on 96/128 packet
questions, including 0/32 three-hop questions.

`retrieval-selection-packet-only.json` preserves the earlier selection before
the display query exposed a distractor. Both selection stages preceded any
neural depth run or final freeze. No final model output guided the choice.

`local-feasibility.json` records the read-only laptop check. The current fixed
loader's saved T4 loading allocation exceeds laptop VRAM. Both checkpoints are
uncached, and no measured local estimate supports a two-hour run. This does not
establish impossibility for every alternative loader or model configuration.

The [dataset card](../../../docs/DEPTH_DATASET_CARD.md) documents the new split,
graph joins, automated label audit and limitations. The
[finite run plan](../../../docs/DEPTH_RUN_PLAN.md) specifies profile inputs,
artifacts, exact-plan resume and the separate 544-request final population.
Run the 16-request profile first when an idle T4 is available. Report errors,
abstentions and wrong answers with their assigned denominators; do not retry
completed outcomes for a better score.
