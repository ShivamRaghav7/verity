# Scope revision 1: interpreting the 0.486 loss

The new scope model learned substantially more of the intended task than the initial model on the **same balanced validation questions**. Its raw classification is still imperfect. The recorded `0.4856789919237296` is binary negative log-likelihood, not a 48.6% error rate or a factual-answer score.

This is a controlled development pilot with four training and two validation product families. All 504 data labels use explicitly disclosed automated development review. Neither this experiment nor its graphs establish final-test performance, human-reviewed data quality or accuracy on arbitrary natural-language questions.

## Recorded training and independent reproduction

Revision 1 completed its fixed 360 optimizer steps, equivalent to three passes through 240 training pairs at batch size two. Validation NLL decreased across the three scheduled checks; the final step was selected by the declared minimum-validation-loss rule. The exported model independently reproduces the selected validation NLL as `0.4856790011362158`, a difference of about `9.2e-9` from the trainer's value. This agreement supports the consistency of export, input order, labels and loss evaluation.

The finite diagnostic evaluated 240 training and 120 validation pairs for each of two exported models: **720 pair forward passes**, no gradient updates and no new parameter search. The diagnostic validates artifact hashes, source snapshots, typed review records, structured scope labels, question–correction input order, positive class index and input formatting. Train/validation family, queried entity, fact, edit, lineage and exact question–correction pair overlaps are all empty. Every input fits within 192 tokens; no input is truncated.

These checks found no evidence of a reversed-label, input-order, truncation, partition-leakage or loss/export implementation fault. They do not prove that every component is bug-free. The generation model's previously documented wrong-`Yes` comparison answers are a separate failure and are not repaired by a better scope classifier.

## Compare both models on identical inputs

The original initial model selected a validation loss near `0.391` on **pilot-v2**, whose negatives used only direct question forms. Pilot-v3 changes 54 of the 72 validation negative questions and keeps all 48 positive questions and correction texts fixed. Directly comparing `0.391` and `0.486` across these different input distributions would be misleading.

The table below instead evaluates **both exports on the exact same 120 pilot-v3 validation pairs**, using a fixed raw probability threshold of 0.5. No compatibility guards contribute to these numbers.

| Raw scope measure | Initial model on v3 | Balanced revision 1 on v3 |
| --- | ---: | ---: |
| Validation NLL, lower is better | 1.477381 | 0.485679 |
| Brier score, lower is better | 0.459192 | 0.160941 |
| AUROC, higher is better | 0.542824 | 0.863715 |
| Correctly classified pairs | 56/120 (46.7%) | 98/120 (81.7%) |
| Positive recall | 34/48 (70.8%) | 46/48 (95.8%) |
| Applicability precision | 34/84 (40.5%) | 46/66 (69.7%) |
| False positive applications | 50/72 (69.4%) | 20/72 (27.8%) |
| Direct positive questions detected | 0/12 | 12/12 |

A constant predictor assigning 0.4 positive probability to every row has NLL about `0.673`; a constant 0.5 predictor has NLL about `0.693`. Revision 1's lower loss is useful evidence of learning. Loss also penalizes uncertainty: its positive probabilities remain close to 0.5, so high recall can coexist with a nonzero average loss.

Revision 1's exported-model training NLL is `0.483570` and validation NLL is `0.485679`. Their small difference does not show the classic pattern of excellent training fit and poor validation fit. It supports an incomplete-fit explanation within this tiny pilot; it does not establish a unique cause or prove that simply training longer would help.

## The remaining raw classifier failures

The first dataset allowed a wording shortcut: all paraphrase/comparison forms were positive while negatives were direct. Revision 1 balances the positive prior at 40% in all four forms without changing facts, lineages, partitions, model architecture or total row count. The initial model's behavior on the new negatives is consistent with exploiting that shortcut: all 36 comparison negatives are classified positive at raw 0.5, while all 12 direct positives are missed. The balanced revision removes this particular observed failure.

Revision 1 separates relation mismatches reasonably well in this validation partition. Its remaining raw false positives are concentrated in entity and version matching:

| Validation negative type | Raw false positives at 0.5 | Mean predicted applicability |
| --- | ---: | ---: |
| Same entity, different property | 0/12 | 0.364 |
| Different entity, same property | 10/12 | 0.532 |
| Similar product name | 0/12 | 0.127 |
| Wrong subscription tier | 0/12 | 0.116 |
| Wrong product version | 10/12 | 0.536 |
| Ordinary negative | 0/12 | 0.122 |

The 48 positive probabilities range from `0.4939` to `0.5673`, with mean `0.5395`. Wrong-entity and wrong-version probabilities overlap that range. The model has not learned sufficiently strong raw entity/version discrimination. Two false negatives are support-hours paraphrases; all direct and both comparison-positive groups are detected at 0.5.

All 20 raw false positives at this fixed threshold violate the application's deterministic entity/version compatibility checks. Applying those guards to the supplied pairs gives TP=46, FP=0, TN=72 and FN=2: **118/120 correctly handled pairs, 46/48 positive recall, 0/72 false applications**. These are explicitly **classifier-plus-guard pair diagnostics**, not raw model performance or end-to-end factual-answer accuracy. The guard checks entity, plan, version and effective date; relation applicability remains a learned decision.

## Calibration is a separate policy decision

The first revision-1 calibration used the existing raw-pair objective and selected temperature 0.5, threshold 0.58 and ambiguity margin 0.1. It clarified **all 48 applicable pairs**, plus 24 of 72 negatives, leaving 48/120 answer coverage and zero positive correction recall. Zero false applications under that policy is not a useful correction-success claim. It reflects conservative abstention in the presence of overlapping raw probabilities.

Runtime rejects incompatible pairs before the scope decision. A separate guard-aware calibration uses the **same** finite temperatures, thresholds, margins and declared costs, with the existing compatibility gate included in the operating-point objective. Temperature selection still minimizes raw-pair NLL. It loads frozen probabilities and performs no model fitting or new forward passes. The original raw-pair calibration stays archived unchanged.

The guard-aware artifact selects temperature **0.5**, threshold **0.47**, and ambiguity margin **0.025**. Among the same 120 validation pairs, it applies 46/48 positive corrections, clarifies 2/48 positives and 3/72 negatives, rejects the other 69/72 negatives and falsely applies 0/72 negatives. Answer coverage is **115/120 (95.8%)**. Scope precision is 46/46 and positive recall is 46/48. These are **policy-plus-guard validation metrics selected on the same data**, not independent test estimates or generated-answer accuracy. The 60 incompatible pairs rejected by deterministic rules remain counted, and the five clarifications remain unsuccessful in overall accuracy denominators.

This fixes an operating-policy mismatch without changing trained weights. No additional training is justified merely by the printed loss value. End-to-end factual answers still require a separate comparison with the unchanged answer model; improving routing cannot repair a wrong answer generated after correct routing. Abstentions must remain in every end-to-end denominator.

## End-to-end validation after the policy correction

The completed five-configuration run contains **600 predictions for 120 pilot-v3 questions**, plus **48 oracle-routed answer predictions**. Every configuration receives the same approved correction scenarios and supporting catalogue facts. The scope revision is used with the original answer checkpoint; no extra training or test-set access occurs. The frozen base parameter fingerprint is identical before and after the run.

The original primary scorer requires a complete endorsement of subject, relation and answer. It does not automatically certify short fragments such as `Yes` or `75 GB`; those outputs remain uncredited and pending semantic review. Its credited fractions below are consequently **strict automatic lower bounds**, not measured zero factual correctness for the three base-generator configurations.

| Configuration | Strict complete-fact credit | Pending/uncredited score | Explicit abstentions | Generated-answer coverage |
| --- | ---: | ---: | ---: | ---: |
| Unchanged base | 0/120 | 120/120 | 0/120 | 120/120 |
| Properly updated RAG | 0/120 | 120/120 | 0/120 | 120/120 |
| Prompt-only correction | 0/120 | 120/120 | 0/120 | 120/120 |
| Similarity editor | 31/120 | 77/120 | 0/120 | 120/120 |
| Trained scope editor | 30/120 | 78/120 | 5/120 | 115/120 |

Pending scores include the explicit clarifications, so those columns are not disjoint. Both editors also have 12/120 strictly incorrect comparison answers. Every question remains in the 120-question primary denominator; clarifications neither count as generated-answer successes nor disappear from the reported fraction. No deterministic structured fact answer contributes to any generated-answer numerator.

The trained editor selects an approved correction on 46 questions, uses the frozen base on 69 and clarifies five. On the 48 applicable questions, its strict credit is **12/12 direct**, **6/12 paraphrase**, and **12/24 comparative** answers. Two applicable paraphrases are clarified; four other applicable paraphrase answers remain pending. The 12 direct questions are unchanged from pilot-v2, where the initial scope policy missed all 12, so the direct-question routing improvement has a valid common-question interpretation. The full 120-question v2 and v3 populations differ, so their aggregate end-to-end fractions must not be presented as a fully paired improvement.

The oracle answer model still obtains **31/48** strict credit: 12/12 direct, 7/12 paraphrase and 12/24 comparative. Five paraphrases remain pending; all 12 expected-`No` comparisons generate `Yes` while stating the correct edited limit. This localizes the unresolved comparison failure to answer generation rather than retrieval or scope routing. The learned editor does not beat the similarity editor on this primary validation score; no superiority claim is warranted.

Both explicit editors apply a correction falsely on **0/72 out-of-scope questions**. Under the strict complete-fact scorer, all 72 out-of-scope outputs remain pending/uncredited; for the trained editor these include three clarifications. Baseline-correct retention is **0/0, undefined**, because this strict automatic grader establishes no baseline-correct out-of-scope pairs. Literal output preservation and factual accuracy must remain separate; neither the raw lower bound nor unchanged prose establishes semantic correctness.

The archived raw, unfiltered applicable-edit retrieval diagnostic is **47/48 at top 1** and **48/48 at top 3** for correction-retrieval configurations. It measures retrieval before eligibility filtering, independently of the successful scope decisions. It is not a direct measurement of the eligible candidate list used for routing.

Observed per-question CLI p50/p95 latency is 165.075/288.726 ms for the base, 144.777/272.923 ms for updated RAG, 148.636/285.188 ms for prompt-only, 200.688/905.931 ms for similarity and 198.386/942.516 ms for the trained editor. These values exclude model initialization, scoring and the extra raw-pair diagnostic forward pass. Fixed configuration order and shared embedding-cache warming prevent a controlled latency-superiority claim; they are not HTTP end-to-end measurements.

Paired family-cluster bootstrap intervals are archived in the metrics, but only two validation families are available. Identical or narrow intervals can reflect this tiny clustered sample; they should not be presented as strong statistical evidence. The final frozen test remains unevaluated.

## Separate exploratory short-answer diagnostic

The previously declared request-bound short-QA analyzer is also applied to these archived predictions. It binds an explicit question subject, relation and scope before accepting a whole-output quantity or a whole-output `Yes`/`No` predicate. It does not extract a desired value from arbitrary prose, examples or rejected alternatives, and a strict incorrect primary score cannot be overridden. Bare comparison polarity can establish the requested predicate without proving the exact underlying limit.

This is the existing **post-hoc development diagnostic**, not a replacement for the primary scorer or a new final-test scoring rule. Primary model outputs and metrics remain unchanged. All abstentions and pending outputs stay uncredited in the requested-QA denominator.

| Configuration | Requested QA credit | Requested numeric value | Requested comparison predicate | Out-of-scope requested QA |
| --- | ---: | ---: | ---: | ---: |
| Unchanged base | 66/120 | 26/60 | 40/60 | 53/72 |
| Properly updated RAG | 91/120 | 42/60 | 49/60 | 53/72 |
| Prompt-only correction | 69/120 | 26/60 | 43/60 | 52/72 |
| Similarity editor | 84/120 | 45/60 | 39/60 | 53/72 |
| Trained scope editor | 80/120 | 41/60 | 39/60 | 50/72 |

Properly updated RAG has the strongest requested-QA result on this development partition. Its primary `0/120` complete-assertion lower bound must not be used to make it appear factually incapable. The trained editor's narrower coverage explains three of its out-of-scope requested-QA losses: it retains 50/53 baseline-QA-correct out-of-scope answers, while three otherwise correct answers are clarified. This QA retention is separate from the primary strict retention, which remains undefined at 0/0. The trained model's 12 wrong comparison answers persist even under oracle routing and this supplementary scoring.

## Subsequent unfamiliar-product workflow

The [completed scope-v3 Nimbus workflow](../reports/pilot-v3-learned-workflow.json)
uses a separate SQLite fixture and actual trained model exports. All **31/31
governance/integration checks** pass, including administrator approval, conflict
resolution and withdrawal/rollback. Those checks deliberately do not require
generated-answer factual correctness.

After approval of a device-limit correction from `5` to `3`, the learned direct
question returns stale `5` through the frozen-base route. Its calibrated edit
probability is approximately 0.3885, below the selected application boundary.
A device paraphrase returns `9 USD per month` through that route, with edit
probability approximately 0.4110. Capacity questions for three and four devices
clarify, with probabilities approximately 0.4647 and 0.4565. The favorable routing
metrics on the two held-out numbered Aster families therefore do not establish
reliable transfer to this unfamiliar product name and alias.

Two separate oracle probes localize the failures: the answer checkpoint produces
`Nimbus Basic version v2 supports 3 devices.` when supplied the correct direct
edit, but incorrectly starts the four-device comparison with `Yes` while stating
the limit of three. The former is a routing failure; the latter remains an
answer-model polarity failure. The fixture's handpicked and repeated questions
are engineering demonstrations, not a representative accuracy population. Its
13-question answer diagnostic must not be extrapolated to general accuracy or
used to replace the 120-question recorded validation.

## Finite development decision

Stage D has a measured first scope revision with documented deterministic guards,
but the subsequent Nimbus workflow exposes failed transfer that the favorable
pilot routing score does not resolve. It motivates **scope revision 2, the final
scope experiment**. The single adaptation is the explicit `entity_neutral_v1`
classifier input policy: canonicalize selectors referenced by the correction and
normalize the descriptor while retaining relation, quantities and units. Original
metadata remains authoritative for compatibility, approval, dates and provenance.
The same pilot-v3 rows, reviews, backbone, seed, learning rate and 360-step cap
are retained. This is preprocessing, not evidence of learned entity verification.
No facts, labels or partitions are rewritten, and no third scope revision is
planned. The new model has not trained or established an improvement. Its
source-bound guarded profile and released card passed preflight. Conditional
planning budgets are 1,331 seconds, 2.0 GiB GPU and 5 GiB checkpoint/export
storage; four updates cannot guarantee full-run time or memory.
The preprocessing applies across edits using the question and supplied correction
descriptor, without gold labels or expected facts; it adds no Nimbus training
examples.

Stage E has separately prepared its first bounded experiment addressing the answer
model's comparison-polarity failure. Its question is whether emphasizing the
first `Yes`/`No` token improves the 12 failing negative comparisons while retaining
short factual answers. It keeps the same v2 answer inputs and 240-step cap and
**has not trained or established an improvement**. Its source-bound guarded
profile and released card also passed preflight, with conditional budgets of
532 seconds, 2.25 GiB GPU and 6 GiB checkpoint/export storage.
The [released guide](ANSWER_REVISION_TRAINING.md) contains both concrete questions,
commands and expected artifacts. Execute one at a time. Exact-state recovery
uses the unchanged card and latest completed checkpoint; completed exports or
capped checkpoints are refused before ML initialization. These readiness checks
do not establish new model accuracy. Scope quality, guarded routing, oracle generation and end-to-end answers
remain separate. After these experiments, select or reject artifacts using
development evidence, fix the final protocol, independently review/freeze the
test and challenges, and evaluate once without test-guided tuning.

The historical training source used by both initial models and scope revision 1 is archived under [`reports/training-source-v1`](../reports/training-source-v1/manifest.json), with exact hashes from Git commit `350db38`. Subsequent answer-training changes do not rewrite those historical source identities or the completed scope diagnosis. The diagnosis's training-source hash describes the implementation present when these scope weights were trained and evaluated, rather than silently claiming the newer trainer produced them.

## Reproduce and inspect

The script only reads pilot-v2/pilot-v3 train and validation files. It never reads frozen test/challenge files or the full original catalogue corpus. The output contains 720 row-bound probabilities, input hashes, grouped metrics, recorded validation checks, token audits and explicit limitations.

```powershell
.\.venv-ml\Scripts\python.exe scripts/diagnose_scope_revision.py `
  --revised-calibration reports/scope-pilot-v3/calibration.json `
  --output runtime/reproduced-scope-diagnosis

.\.venv-ml\Scripts\python.exe scripts/plot_scope_diagnosis.py `
  --source reports/pilot-v3-scope-diagnosis `
  --output runtime/reproduced-scope-figures
```

Existing output directories are refused so previous evidence is not overwritten. No training, model changes, new thresholds or final-test evaluation occur in these commands. The archived diagnostic binds the initial raw-pair revision-1 calibration; later guard-aware policies must use their own explicit artifact rather than rewriting it.

Artifacts: [`summary.json`](../reports/pilot-v3-scope-diagnosis/summary.json), [`probabilities.jsonl`](../reports/pilot-v3-scope-diagnosis/probabilities.jsonl), [`scope-diagnosis.png`](../reports/pilot-v3-figures/scope-diagnosis.png), [`guard-aware calibration`](../reports/scope-pilot-v3-runtime/calibration.json), [`five-way metrics`](../reports/pilot-v3-validation-routed/metrics.json), [`exploratory short-QA metrics`](../reports/pilot-v3-short-qa/summary.json), [`subsequent Nimbus workflow`](../reports/pilot-v3-learned-workflow.json).
