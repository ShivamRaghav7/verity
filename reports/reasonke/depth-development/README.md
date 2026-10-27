# Separate depth challenge: development preparation

No neural depth profile, development or final inference is reported here.
The completed operations-v1 study and its results are preserved unchanged.

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
