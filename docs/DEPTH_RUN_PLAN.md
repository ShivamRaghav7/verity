# Separate depth challenge: preparation and finite run plan

The completed operations-v1 study remains unchanged. Its 256/256 ReasonKE
retrieval result is an actual result on controlled, correlated questions, not
universal language accuracy. We add depth and new wording to measure a different
generalization question, not to manufacture a lower score.

**Current state, 28 October 2026: all 544 frozen final assignments are complete,
backed up and replay-audited.** ReasonKE scored 256/256 main answers; Qwen scored
248/256 with eight abstentions. Both separate gold-support diagnostics scored
16/16. There are no execution errors or unexecuted final assignments. See the
[complete evidence](../reports/reasonke/depth-final/README.md). Both checkpoints
also scored 8/8 on profiles and 24/24 on development; these earlier populations
are separate from final accuracy.
The protocol was published in commit `08d274e` before final inference, with
SHA-256 `df120622131d617b37b7bfdfdfd141cbe23fe07fa5bf9ff914732d99dcfe1941`.
An earlier development session expired before its outcomes were downloaded;
those unavailable outcomes are not claimed as evidence. The preserved profiles
were restored unchanged for the disclosed replacement development run.

Colab had initially refused GPU allocation because of its usage limit. A
read-only laptop check also found that the fixed loader's previously
measured 6.59 GiB peak allocation exceeds the RTX 4050's approximately 6 GiB
capacity. Both checkpoints are uncached and their combined source tensors exceed
currently free disk space. No measured local runtime supports the user's
two-hour limit, so no model downloads or local inference were started. See the
[local feasibility receipt](../reports/reasonke/depth-development/local-feasibility.json).
This is a configuration-specific assessment, not a claim that every quantized
7B implementation is impossible on this laptop. No GPU training is required or
scheduled automatically.

## Question and populations

Can an approved ownership update propagate through service -> team -> on-call
person -> office city, with new entity assignments and wording families, while
version and property controls stay correct?

The new packet has 48 fictional Indian-context graph lineages and 384 questions.
Partitioning happens first: 16 development and 32 final lineages. Each world has
twelve facts and eight questions: two each at depths one, two and three, plus
two locality controls. The final population contains 256 unique questions,
including 64 three-hop questions, over 32 independent graph lineages. Wording
patterns are reserved by split; entity and person names are disjoint. City
values and relation vocabulary intentionally repeat. See the
[dataset card](DEPTH_DATASET_CARD.md).

Together the two studies cover 64 held-out graph lineages and 512 question
records. They have different populations and context policies, so their
accuracies must be reported separately rather than pooled into a headline.

## What changed in retrieval, and why

The original two-stage four-fact retriever completed all one/two-hop development
paths but none of the 32 three-hop paths. That is a retrieval limitation, not a
measurement of the neural model's reasoning.

We declared six bounded candidates: one/two lexical seeds, three expansion
rounds, and a four/five/six-fact cap. Selection used only the 128 development
questions and the explicitly declared India display's initial/current three-hop
states. It selected **four facts, two seeds, up to three rounds**: 130/130 paths
and 230/230 required fact occurrences were present. This is support recall,
not answer accuracy. The earlier packet-only diagnostic is preserved alongside
the [complete selection record](../reports/reasonke/depth-development/retrieval-selection.json).

Retrieval starts from the question, then expands with selected answer text.
Expected answers, gold supporting IDs and structured question labels never
enter ordinary retrieval. More or different context can still harm generation.

Both pinned checkpoints receive the same selected fact IDs and generation
budgets. Each has one main iterative-retrieval condition and a sixteen-question
hash-preselected three-hop oracle diagnostic. Oracle support is labelled and
cannot be presented as normal retrieval performance or targeted recovery.

## Stages and stopping conditions

Use `notebooks/06_depth_challenge.ipynb`, pinned to its stated source commit.
Use an idle T4 runtime. The hardware cell refuses CPU-only execution before
model download. Do not run a demo worker alongside the experiment.

For the India website alone, use `notebooks/07_india_chat.ipynb`; its isolated
workspace uses iterative retrieval. The preserved `04_reasonke_chat.ipynb`
launcher continues to reproduce the original two-stage serving policy. Starting
either website does not run the new evaluation.

| Stage | Concrete question | Fixed requests | Expected artifacts / stop |
| --- | --- | ---: | --- |
| Profile | How much time, output and CUDA memory do actual depth questions use? | 8 per checkpoint, sequentially; 16 total | Two raw profile ledgers, logs, model/resource metadata; stop after all 16 outcomes |
| Development | Does the fixed method handle the declared development wording and controls? | 24 per checkpoint; 48 total | Two development ledgers and separate audited summary; stop after 48 outcomes |
| Freeze | Are source, data, scoring, measured budgets and populations fixed? | Zero generations | Protocol identifying all source/data/configuration hashes and four complete input receipts |
| Final | What is held-out performance by depth and wording, with routing isolated? | 272 per checkpoint; 544 total | Two complete final ledgers, protocol-bound audit and depth scorecard; stop after 544 outcomes |
| Score | Do the saved outputs reproduce the declared scoring and denominators? | Zero generations | Audited `results.json`; refuses running/failed jobs and preserves existing results |

Only profile is enabled initially. Later phases, including scoring, have explicit switches. Review
the completed profile before enabling development, and finalise settings before
freezing. No final answers may guide settings or aliases. A disappointing result
is retained; it does not trigger an unbounded search for a better score.

### Runtime and memory

The new T4 profiles supply a measured, uncertain depth-run estimate:

| Checkpoint | Requests profiled | Mean / maximum seconds | 272-request mean / maximum extrapolation |
| --- | ---: | ---: | ---: |
| ReasonKE | 8 | 15.97 / 17.91 | 72.42 / 81.18 minutes |
| Qwen | 8 | 1.09 / 2.49 | 4.94 / 11.27 minutes |

These request-only extrapolations exclude loading and downloads. With cached
reloading, approximately 80-95 minutes was budgeted for the two-model final run;
later variation can exceed this estimate. Both loading peaks allocated 6.59 GiB.
See the [raw measured profiles](../reports/reasonke/depth-profiles-v1/README.md).
Loading pinned NF4 checkpoints
uses FP16 computation and transfers approximately 15.23 GB of source tensors per
checkpoint; fresh downloads vary. Both checkpoints remain cached for exact-plan
resume. The notebook requires at least 36 GiB free disk for those tensors and
the isolated environment, and inventories the actual GPU.

The profile records actual load time, per-request pipeline time, token counts
and CUDA allocation/reservation. Freeze computes 272-request extrapolations
from each eight-request profile mean and maximum, explicitly excluding fresh
downloads/loading. These are uncertain estimates, not guarantees. Before a
substantial final run, show those measured values, required inputs, output
directory and remaining count.

Hard watchdogs are 1,800 seconds for loading, 240 seconds per request and eight
hours per model. These are safety caps, not forecasts. The same 512-new-token
budget applies to both checkpoints. Report truncation/format failures instead
of silently increasing the cap after seeing test answers.

### Resume and safe storage

Every outcome is written atomically after its request. `--resume` verifies the
identical source/task/context/scoring plan and skips every completed correct,
wrong, abstained or error result. Only `unexecuted` rows can run. Preserve all
profile/development receipts beside `protocol.json`: final verification checks
their hashes and measured timings as well.

Interrupted ledgers are audited individually during resume. Publish the combined
paired scorecard after both final ledgers finish; partial combined paired
statistics are not a supported report in this frozen runner.

Download the notebook's whitelisted evidence ZIP after each phase, before
closing a temporary runtime. It contains JSON receipts/protocol, not model
weights, tokens or customer chats. Resumption requires the same pinned source
and restored files; never repeat completed failures for a better outcome.

```text
python -m scripts.run_depth_pipeline --phase profile --directory artifacts/depth-run
python -m scripts.run_depth_pipeline --phase development --directory artifacts/depth-run
python -m editguard_depth.run --freeze-directory artifacts/depth-run --output artifacts/depth-run/protocol.json
python -m scripts.run_depth_pipeline --phase final --protocol artifacts/depth-run/protocol.json --directory artifacts/depth-run
python -m editguard_depth.metrics artifacts/depth-run/final-reasonke.json artifacts/depth-run/final-instruction.json --protocol artifacts/depth-run/protocol.json --output artifacts/depth-run/results.json
```

Run one stage at a time in the pinned CUDA environment. The later commands
require their earlier completed evidence; copying these lines is not permission
to skip measured-budget review.

## Reporting rules

Report all assigned outcomes, depth/category denominators, coverage, required
fact recall, complete-path retrieval and oracle generation separately. The
complete-support subgroup is descriptive; it cannot replace the primary
denominator. Paired uncertainty resamples whole lineages. Distinguish model
answers from administrator/database state checks. Automated labels and shared
templates remain limitations even if the score is again perfect.
