# Selected auxiliary revisions: development results

**Historical development record.** Future-tense statements below describe the
status when these revisions were selected. Stage G and the separately declared
oracle runtime recovery have since completed. See [final evaluation](FINAL_EVALUATION.md)
for their results and [the beginner guide](BEGINNER_GUIDE.md) for metric definitions.
The original development measurements and limitations remain unchanged.

The user completed both released runs on 2026-10-14. The selected scope model is
`artifacts/scope/pilot-v3-neutral2`; the selected answer model is
`artifacts/answer/pilot-v2-polarity1`. Both are completed local trained exports,
not profiling artifacts. No additional training experiment is planned for this
release. Stage G will lock the configuration and scoring rules, independently
audit the held-out evaluation labels under the user-authorized automated policy,
freeze their hashes and run one final matched evaluation. That final audit is
machine review, not human approval. These results are development evidence only.

## Training identity and actual measurements

| Measurement | Scope revision 2 | Answer revision 1 |
|---|---:|---:|
| Dataset / reviewed train rows / validation rows | pilot-v3 / 240 / 120 | pilot-v2 / 96 / 48 |
| Updates completed / selected checkpoint | 360 / 360 | 240 / 240 |
| Input / target token caps | 192 / not applicable | 192 / 48 |
| Selected validation objective | Binary classification NLL | Comparison-first-token 8x weighted token NLL |
| Recorded selected objective | 0.043264720 | 0.168240236 |
| Recorded training-loop time | 112.55 seconds | 264.54 seconds |
| Peak allocated / reserved GPU memory | 1.318 / 1.410 GiB | 1.504 / 1.656 GiB |

Recorded loop time includes that loop's validation and checkpoint writes. Full
process startup, final export and total user-command wall time are unavailable;
the table is not a complete command runtime. Inputs and targets did not truncate.
The [independent training audit](../reports/targeted-training-audit.json) passed
986 checks covering recorded identities, weights and available checkpoint
state. Audit checks are artifact integrity evidence, not answer-accuracy samples.

The scope manifest SHA-256 is
`c85e77193ff3cf5137baf0a569d142585cfac3a1734393fadf49ce95f9b88459`;
the answer manifest SHA-256 is
`8e435ac9bcd5486bb967c7ca09c38bf1786b810c97a7c95a365c6250ceeb0a72`.
Copied metadata and logs are in [the training evidence](../reports/targeted-training).
Exact training inputs, backbones, source versions, finite caps and recovery rules
remain in [the released run guide](ANSWER_REVISION_TRAINING.md).

## Scope results on the same validation population

Both scope models were evaluated on the same 120 pilot-v3 pairs: 48 applicable
and 72 out of scope, from two held-out entity families. Revision 2 applies the
documented [entity-neutral input policy](SCOPE_INPUT_POLICY.md); raw entity,
lineage, facts, labels and split assignments remain unchanged. Its abstracted
inputs have **eight identical same-label pairs across train and validation**,
affecting 8/120 validation rows, and zero contradictory-label input pairs.
Shared question templates and that overlap limit claims about unseen logical
forms or wording, even though raw entities are held out.

| Raw classifier metric, threshold 0.5 | Revision 1 | Selected revision 2 |
|---|---:|---:|
| Correct pairs / all pairs | 98/120 | 119/120 |
| True applications / applicable pairs | 46/48 | 48/48 |
| False applications / out-of-scope pairs | 20/72 | 1/72 |
| Precision | 46/66 | 48/49 |
| Validation NLL reproduced from exported weights | 0.485679001 | 0.043264630 |

These raw metrics exclude deterministic compatibility guards and uncertainty
handling. The selected validation policy uses temperature 1.0, threshold 0.97
and ambiguity margin 0.025. After the same product, plan, version and date
compatibility rules used in the application, it applies 48/48 applicable pairs,
falsely applies 0/72 negatives, rejects 71 negatives and clarifies one negative.
Coverage is 119/120. The clarification remains in the denominator. Zero observed
false applications on 72 selection-data negatives is not a guarantee for new
questions or a model-only result. See [raw scope diagnosis](../reports/pilot-v3-neutral-scope-diagnosis/summary.json)
and [selected guarded policy](../reports/scope-neutral2-runtime/calibration.json).

## Answer results and tradeoffs

The unchanged 144-row pilot-v2 answer inputs and targets were used for both
models. The fixed revision weights the first target token of each reviewed
comparison row by eight, equally for `Yes` and `No`. It changes no target value
and trains no deterministic answer adapter. [Frozen teacher-forced diagnosis](../reports/pilot-v2-answer-revision-diagnosis/summary.json)
reports both objectives for both models:

| Validation metric on the same 48 dataset-context rows | Original answer | Selected revision |
|---|---:|---:|
| Unweighted token NLL | 0.044412011 | 0.061731466 |
| Fixed 8x weighted token NLL | 0.162544489 | 0.168240240 |
| First-token top-1 correct, required `No` | 0/12 | 4/12 |
| First-token top-1 correct, required `Yes` | 12/12 | 11/12 |
| First-token top-1 correct, all rows | 36/48 | 39/48 |

Both comparable aggregate losses became worse. The recorded weighted selection
loss of 0.16824 must not be compared directly with the original uniform selection
loss of 0.04441 as the same objective. Teacher forcing supplies earlier gold
tokens and does not measure complete generated-answer accuracy.

Free generation with correct routing and application-selected supporting context
credits 34/48 complete facts for the revision, versus 31/48 for the original
answer model under the same selected scope configuration. The revised model
gets 5/12 negative comparisons correct and still fails seven; it gets 11/12
positive comparisons, 12/12 direct answers and 6/12 paraphrases correct. Five
paraphrases remain uncredited pending semantic review and one is incorrect.
The first-token diagnostic's 4/12 and free generation's 5/12 negative decisions
use different supporting contexts; they are distinct measurements.

## Fair comparisons and scoring disclosure

The five configurations receive the same 120 questions and approved information,
with matched frozen base, retrieval and output budgets. Three development
configurations isolate the scope and answer changes. Saved outputs from **all
three** are rescored under `editguard-short-facts-v3`; the previous scored
artifacts remain intact. The [storage alias change](SCORER_STORAGE_ALIAS.md)
accepts complete scoped `supports N GB of storage` assertions. Any difference
caused only by this change is scoring coverage, not a model improvement.

| Selected configuration | Primary complete fact credit / 120 | Primary pending / 120 | Exploratory requested short-QA credit / 120 |
|---|---:|---:|---:|
| Unchanged base | 0 | 120 | 66 |
| Properly updated RAG | 0 | 120 | 91 |
| Prompt-only corrections | 0 | 120 | 69 |
| Similarity correction editor | 34 | 77 | 87 |
| Trained scope-aware editor | 34 | 76 | 86 |

Baseline zero primary credit means their short fragments do not establish a
complete scoped assertion; it does **not** mean zero factual correctness.
Requested short-QA is a separate existing diagnostic accepting bounded whole
quantities or yes/no decisions. It does not establish an exact scoped fact for
bare comparison answers. The updated RAG assistant remains strongest on that
diagnostic. No superiority claim follows from this table.

Holding the selected scope and its policy fixed, the original answer gives
31/120 primary fact credit and 83/120 short-QA credit; the revision gives
34/120 and 86/120. The prior scope plus original answer gives 30/120 and 80/120.
All denominators retain pending answers, failures and clarifications. The selected
trained editor has one clarification, 76 pending assertions and nine incorrect
answered assertions. Its out-of-scope short-QA credit is 52/72 versus the base's
53/72, reflecting one clarification. Primary baseline-correct retention is
undefined (0/0), because no baseline out-of-scope fragment establishes the full
fact; that undefined value must not be reported as perfect retention.

Per-question outputs and metrics are in
[selected comparisons](../reports/targeted-candidate-v3-scored/metrics.json),
[selected short QA](../reports/targeted-candidate-v3-short-qa/summary.json),
[scope-only control](../reports/scope2-original-answer-v3-scored/metrics.json),
and [the previous scope control](../reports/pilot-v3-current-v3-scored/metrics.json).
The selected learned configuration's observed end-to-end p50 is 201.58 ms and
p95 is 956.75 ms over 120 requests. Different development runs had different
system load; their timings do not establish a causal speed improvement.

The isolated Nimbus workflow passes 31/31 governance/integration checks across
33 API requests. Actual generated outputs and separate oracle probes remain in
[the workflow evidence](../reports/targeted-candidate-learned-workflow.json).
Repeated handpicked cases are not representative accuracy samples. Structured
database answers, provenance displays and workflow checks never enter the
learned model's accuracy numerator.

## Release decision and remaining work

Select scope revision 2 for its improved raw routing and guarded behavior, and
answer revision 1 for its measured improvement in complete oracle/end-to-end
answers despite worse token losses and remaining polarity errors. The finite
development question has been answered; no third scope revision or second
answer experiment is planned. `configs/inference-final.json` fixes these models
and the existing calibrated policies for stage G.

All development labels have separately recorded automated review, with no
human-review claim. Final evaluation will use a separately recorded independent
machine audit authorized by the user, followed by a hash freeze before inference.
That policy change must be explicit in the protocol and final report. Stage G's
final outputs and stage H's final report, graphs, release reproduction and panel
demo remain pending. Test findings will be documented rather than used to tune
this release.
