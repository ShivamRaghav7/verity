# Scope revision 1: implementation audit

The completed scope model is weak on important scope boundaries. The evidence does
not identify a broken training loop, an inverted label, or an inference/training
format mismatch. This is an independent, read-only implementation and development
data audit conducted after the 360-step pilot-v3 run. No optimizer updates, GPU
execution, or test/challenge data access were performed for this audit.

## What the loss means

The reported `0.4856789919237296` is mean binary cross-entropy on 120 validation
question–correction pairs. It is **not an accuracy percentage**. For this dataset's
40% positive prior, always predicting probability 0.4 gives cross-entropy
`0.6730116670`; predicting 0.5 gives `0.6931471806`. The selected model improves
this probability loss, but that does not establish useful precision, recall, or
safe routing at the selected operating threshold.

The initial model's `0.3907963567` validation loss used different negative
questions. Comparing those two selected losses is not a matched before/after
experiment. The 48 positive validation questions are unchanged and can support a
separate paired comparison.

## Implementation checks

| Area | Inspected behavior | Finding |
| --- | --- | --- |
| Labels and output indices | Training creates `torch.long` labels 0/1, loads `num_labels=2`, and records positive class index 1. Runtime reads softmax index 1. | Consistent; no inversion found. |
| Loss | Installed Transformers 5.16.1 `DistilBertForSequenceClassification.forward` chooses single-label cross-entropy for these long labels. | Correct classification objective. |
| Input order | Training and runtime both tokenize `(question, correction_text)` as a pair. | Consistent. |
| Lengths | Manifest records maximum train/validation pair lengths 85/83 tokens, both below the declared 192-token cap. | No truncated training or validation rows. |
| Train/evaluation modes | Optimizer loop calls `.train()`, zeroes gradients, computes loss, calls `.backward()`, clips gradients, and steps optimizer/scheduler. Validation calls `.eval()` inside `torch.inference_mode()` and restores `.train()`. | No accidental training under inference mode or validation dropout found. |
| Auxiliary encoder | Pretrained sequence classifier is loaded normally; training does not freeze its parameters. Runtime freezes the completed classifier for inference. | Correct separation of training and inference. |
| Schedule | AdamW starts at `2e-5`; linear decay reaches zero after the declared 360 steps. | Matches run card; no extra updates or implicit early stop. |
| Selection/export | Lowest validation loss among steps 120/240/360 wins; selected state is loaded before model export. | Step 360 is correctly selected. |
| Resume | Checkpoint includes model, optimizer, scheduler, RNG, sampler order/cursor, global step, identity, and best validation. Hash/configuration checks reject a changed stopping cap. | No material resume defect found by source inspection; this audit does not claim a new execution of resume tests. |

The loader's newly initialized `classifier` and `pre_classifier` weights are
expected when using the pretrained masked-language-model backbone for a new
classification task. Unexpected vocabulary-head keys describe that task change;
they are not evidence that completed fine-tuning failed.

The PyTorch TF32 deprecation warning also does not explain the validation loss.
The declared run uses deterministic float32 and disables TF32. Updating the API
is routine maintenance requiring a new profile/source identity, not a proposed
accuracy fix.

## Data and trajectory checks

Independent standard-library checks inspected **all 360 scope rows**. For every
row, the binary label agrees with explicit entity/relation/plan/version and date
compatibility; the original value matches the catalogue; the expected value
matches the applicable correction or unchanged queried fact; and the question's
exact product/alias, plan and version identify the expected subject. All
referenced facts stay within the row's split. There are no duplicate token-input
text pairs with contradictory labels and no shared entity families, entities or
edit lineages between train and validation.

| Population | Rows | Edit worlds | Edit lineages | Entity families |
| --- | ---: | ---: | ---: | ---: |
| Train | 240 | 24 | 16 | 4 |
| Validation | 120 | 12 | 8 | 2 |

These rows share twelve question templates and repeated correction worlds.
Neither 240 training rows nor 120 validation rows represent that many independent
products or independent language patterns. Labels have automated development
review, not independent human review. Effective dates do not vary in this pilot;
the learned model's temporal generalization remains unmeasured. Deterministic
compatibility checks are separate application behavior.

| Pass end | Mean online training loss during pass | Validation loss |
| --- | ---: | ---: |
| 120 | 0.681737 | 0.678070 |
| 240 | 0.678347 | 0.640191 |
| 360 | 0.567464 | 0.485679 |

Online training loss includes training dropout and parameters changing across the
pass; it is not an evaluation-mode training loss. The continuing decline and
best checkpoint at the stopping cap suggest incomplete optimization, without
proving that a longer run will improve held-out performance. All 360 updates have
finite gradient norms above the clipping threshold of 1.0. This standard clipping
setting limits updates; that observation alone does not demonstrate a clipping
bug or justify changing several optimization settings together.

## The remaining failure is specific

The saved validation probabilities expose a scope shortcut. Easy negative cues
are separated while numbered product identities and product versions are not:

| Pair group | Denominator | Mean probability correction applies | Range |
| --- | ---: | ---: | --- |
| Applicable | 48 | 0.5395 | 0.4939–0.5673 |
| Different product, same property | 12 | 0.5321 | 0.4890–0.5588 |
| Wrong version | 12 | 0.5363 | 0.4914–0.5663 |
| Same product, different property | 12 | 0.3639 | 0.2367–0.4947 |
| Similar product name | 12 | 0.1269 | 0.1247–0.1305 |
| Wrong tier | 12 | 0.1157 | 0.1119–0.1183 |
| Ordinary negative | 12 | 0.1221 | 0.1190–0.1239 |

This is consistent with learning `Pro`/`Plus` cues before learning exact numbered
entity and version equality. It is a diagnosis from observed predictions, not a
causal attribution to a single token or layer. The classifier's raw probabilities
cannot reliably distinguish the important difficult negatives from positives.
Calibration can select a cautious operating point, but cannot create separation
that the model has not learned. Global loss alone concealed that failure.

## Bounded recommendation

Do not release this artifact solely because its loss decreased, and do not lower
the routing threshold simply to improve headline recall. First report calibrated
precision/recall, clarification coverage, each difficult-negative group, and the
unchanged positive questions alongside the initial model. Keep the initial model
and revision artifacts immutable so that the outcome can be inspected.

### Material selection-policy mismatch discovered after this training audit

The archived scope calibrator selected threshold/margin cost on **all raw pairs**.
The existing application first rejects corrections outside the approved product,
plan, version and date scope, then applies the classifier's uncertainty rule to
the remaining candidates. The raw selector charged false applications or
clarification costs for incompatible pairs that cannot reach that runtime rule.
The similarity baseline already selected its operating point after the same
compatibility gate. This mismatch is a material calibration-policy bug; it is
separate from the correct training loop and the weak raw boundary discrimination.

The archived selection is preserved: temperature 0.5, threshold 0.58, margin 0.1
clarified all 48 positives and 24 negatives, and rejected 48 negatives. It remains
an explicit failed development result; neither its artifact nor its probabilities
were overwritten.

`scripts/calibrate_scope_runtime.py` now performs an offline selection on those
**same frozen 120 raw probabilities**, with the original temperature grid,
threshold grid, margin grid, declared costs, and tie breaks. Temperature still
minimizes raw-pair negative log-likelihood. The operating point now minimizes cost
after the existing approved product/plan/version/date gate; no expected fact or
label determines eligibility. Raw-pair outcomes remain reported separately.
No model executes or trains during this correction. Fourteen regression checks
cover denominators, gate precedence over uncertainty, raw failures, source/model/
review integrity, no ML imports, and compatibility with the unchanged loader.

The root-run selector chose temperature 0.5, threshold 0.47 and margin 0.025:

| Diagnostic at the corrected operating point | True applications | False applications | True rejections | Positive clarifications | Negative clarifications |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw classifier pairs | 46 | 21 | 45 | 2 | 6 |
| Existing compatibility gate plus classifier | 46 | 0 | 69 | 2 | 3 |

Both rows retain all 120 pairs: 48 positives and 72 negatives. Guarded recall is
46/48; decision coverage is 115/120. Guarded precision is 46/46 in this small
validation population. The zero guarded false applications are evidence of the
**combined deterministic and learned routing policy**; they do not establish
that the classifier learned entity/version boundaries. Its 21 raw false
applications stay visible. Same-validation selection and reporting are
optimistic development evidence, not a frozen-test result.

These are single focal question–correction pairs. They do not establish retrieval,
multi-correction competition, conflict handling or generated-answer correctness.
The next action is to evaluate the complete existing memory-routing workflow and
matched baselines using this corrected selector. **No additional training is
recommended now.** If that evaluation still justifies the remaining permitted
scope revision, incomplete optimization may motivate one predeclared finite run
on the same reviewed data, with a fresh identity and profiling. Do not change the
completed run's stopping cap and call that an exact resume. No hyperparameter
sweep, final-test tuning, or accuracy guarantee is justified by this audit.

## Audited source identities

- Training: `editguard/training.py`, SHA-256 `52952f6ec9e4c383c71b39e77e097f2e7c8bd98d2a6bfb60d742026f0e658218`.
- Runtime: `editguard/neural.py`, SHA-256 `e5f780dd0d258ccc442187477992d1c4b3119a8d101473adb87f737e8ebb3c51`.
- Formatting: `editguard/formatting.py`, SHA-256 `d5fc6f268509d44fab951d8c5c9c3e71a4202421438e8a8346a9e896550f7d6b`.
- Revision artifact: `artifacts/scope/pilot-v3-revision1/manifest.json`, SHA-256 `2374ff41593ff6fd7a9890ddfc05cef0845a667c5651bc38664ca968c1e6cd55`.
- Saved probabilities: `reports/scope-pilot-v3/validation_probabilities.jsonl`, SHA-256 `b4ff6bbb205990d93fc8febf005a0e71a19cd51a65c96adf659fa4d34083a091`.

The exact installed implementation inspected was
`.venv-ml/Lib/site-packages/transformers/models/distilbert/modeling_distilbert.py`,
class `DistilBertForSequenceClassification`. This audit uses source inspection and
saved development artifacts; it is not a new training experiment or a final
evaluation result.
