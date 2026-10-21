# VisionPatch Development and Evaluation Protocol

## Freeze and provenance

Before variants, group exact duplicate source images by SHA-256. The first feasibility run has support, clean covariance/retention, and probe partitions; all are development data. Keep every source and all its variants in one partition. Inspect suspected near duplicates and record decisions. Content hashes prevent exact leakage, not all visual similarity or possible overlap with CLIP's unknown pretraining images.

Save source paths, labels, hashes, split seed, class prompts, transformation parameters, model revision, and preprocessing. Audit source labels without upgrading an automated check to human review. Initially use one default font and different support/probe label positions. Class-disjoint and unfamiliar-font tests require later explicit partitions. No frozen final set exists yet; create and seal it before final evaluation, after the development feasibility gate.

The eight-class development subset uses 18 originals per class: two support, eight retention, and eight probe. This yields 144 original photographs and 64 probe originals. Only cup and laptop support images fit the patch, giving four paired examples; support slots from other classes are unused. The distractor word is `camera`; genuine camera photographs remain in evaluation. Report probe performance separately on support classes and the other six classes. Verify class folder names, available images, and exact group counts during preparation; the generated manifest is authoritative.

## The edit objective

Let `W` be CLIP's original visual projection with shape `[out, in]`, `K` the rows of attacked support features with shape `[N, in]`, and `V` the corresponding projected features with a matched blank label, shape `[N, out]`. Let `R` contain rows of clean retention features and `E = V - K W^T`. The implementation defines:

`C = R^T R / len(R) + covariance_ridge * scale * I`

Here `scale` is the mean diagonal of the unregularized second-moment matrix, or one when it is zero. Fit an additive projection update `D` by minimizing:

`|| K D^T - E ||²_F + fit_ridge * tr(D C D^T)`

For positive regularization values, the implemented solution is:

`D = E^T (K C^-1 K^T + fit_ridge * I)^-1 K C^-1`

The two regularizers have different roles: `covariance_ridge` stabilizes and fills unobserved covariance directions; `fit_ridge` controls the tradeoff between fitting the paired targets and changing retained representations. Compute `C` only from clean retention features. Use CPU float64 linear solves without constructing explicit inverses, record the numerical settings, and verify finite outputs. The update has rank no greater than the number of support pairs. Normalize final image embeddings and score against the same frozen text embeddings in all learned-model arms. This objective encourages retention; it does not guarantee unchanged recognition.

Support size, regularization, and fine-tuning steps may be selected on development results within a recorded finite search. Save every attempted configuration. Publish measured edit time and memory, not estimates presented as measurements.

## Comparisons and controls

- Original CLIP: unchanged checkpoint and prompts.
- Projection patch: the stated closed-form objective.
- Projection fine-tuning: same editable matrix, support pairs, targets, and retention examples; finite development-selected step budget.
- Oracle masking: blank the known generated label region before running original CLIP. This is a privileged preprocessing control, not a learned editor.

Run clean, misleading-label, and identically positioned blank-label versions. The blank version helps separate text influence from object occlusion. Maintain a fixed candidate set containing every evaluated object class and the distractor. Never remove the distractor from inference or use ground truth to decide when the patch applies.

## Metrics and denominators

Report counts with rates for every arm and partition:

- Clean accuracy: correct clean predictions / all source images.
- Attacked accuracy: correct misleading-label predictions / all source images.
- Attack-induced error: baseline clean-correct and attacked-wrong sources from non-distractor classes / baseline clean-correct sources from non-distractor classes. A camera label on a genuine camera is not a misleading-class attack.
- Repair rate: baseline clean-correct, attacked-wrong non-distractor sources corrected by the patch / baseline attack-induced errors under that eligibility rule.
- New-error rate: baseline-correct predictions made wrong / baseline-correct predictions, separately for clean and attacked inputs.
- Genuine distractor accuracy and text-bearing challenge accuracy, with their own sample counts.
- Prediction changes, macro class accuracy, patch size, runtime, peak memory, and warm inference p50/p95.

Also publish repair count divided by all source images; a high conditional repair rate on a tiny eligible subset can mislead. Zero denominators are undefined and shown explicitly. If abstention is added later, it counts as incorrect in raw accuracy and is reported separately.

Use paired comparisons and bootstrap intervals grouped by source image, keeping its variants together. Report the number of independent sources and classes; extra transformations do not create independent photographs. Show representative successes and failures under a documented selection rule.

## Regression and release checks

Verify disjoint hashes; deterministic transformations; unchanged frozen encoders, prompts, and preprocessing; identical patch application across classes; repeatable patch reload; and exact parent-weight restoration. Compare rollback predictions with the original predictions using recorded numerical tolerance. Check both accidental overwrites and applying a patch to the wrong parent checkpoint.

Development measurements guide implementation. After configurations and scoring rules are fixed, evaluate the sealed final split once. Any subsequent correction is a versioned follow-up evaluation with the original results retained. No performance numbers are claimed by this protocol.
