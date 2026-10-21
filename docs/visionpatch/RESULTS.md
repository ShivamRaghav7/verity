# VisionPatch measured development results

Evidence: `reports/visionpatch/pilot-v1.json`, completed 21 October 2026, and the separate `reports/visionpatch/encoder-profile.json`. This page reports the first fixed development configuration. It contains no final-test result.

## Experiment in plain language

We used 144 Caltech101 photographs from eight classes. Four paired cup/laptop support photographs fitted the edit; 64 different clean images supplied the retention reference. The 64 probe photographs, eight per class, were not used to fit either learned approach. Each probe has clean, blank-label, and printed-`camera` versions. These 192 rendered inputs come from **64 independent source photographs**, not 192 independent originals.

Both encoders and eight candidate prompts were frozen. The edited visual projection is 512 by 768. The patch has rank at most four; this is an algebraic upper bound, not a claim that the entire model has four parameters. Original weights are retained for exact rollback.

## Accuracy and preservation

| Configuration | Clean correct | Blank-label correct | Printed-label correct | Label-induced errors repaired | New errors on baseline-correct printed inputs |
|---|---:|---:|---:|---:|---:|
| Original CLIP | 64/64 | 62/64 | 42/64 | 0/22 | 0/42 |
| Projection edit | 64/64 | 62/64 | 63/64 | 21/22 | 0/42 |
| Projection fine-tuning | 64/64 | 62/64 | 63/64 | 21/22 | 0/42 |
| Oracle blank-region masking | 64/64 | 62/64 | 62/64 | 21/22 | 1/42 |

**The two learned approaches tie on these aggregate measurements.** This experiment does not show that closed-form editing outperforms ordinary projection fine-tuning. The mask is a privileged control because it knows the generated label location; its one new error explains why it can repair 21 mistakes yet finish with one fewer correct prediction.

Accuracy is correct predictions divided by every probe source for that input variant. All methods produce a candidate label; there are no abstentions omitted from the denominator. The edit's printed-label accuracy is 98.44% in this small controlled development set, not a general model reliability estimate.

The original model was correct on all clean probes. Among the 56 non-camera sources, adding `camera` induced 22 mistakes: 22/56, or 39.29%. Genuine cameras are excluded from this attack denominator because printing their correct class is not a misleading-class attack. The edit repaired 21 of those 22 errors, or 95.45%. Expressed against **all** probe sources, the repairs are 21/64, or 32.81%.

All 64 original clean predictions remained correct after editing, and all eight genuine camera photographs remained correct. Those finite observations support preservation on this set; they do not guarantee preservation elsewhere.

## Generalization and the remaining failure

On the two support classes, unseen cup/laptop probes improved from 6/16 to 16/16 printed-label predictions. On the other six classes, the edit and fine-tuning improved from 36/48 to 47/48, repairing 11/12 eligible errors. This demonstrates development transfer beyond the support classes. It does not establish unfamiliar-word or unfamiliar-font transfer: the same camera word and default font were used, with a different label position.

The remaining edited failure is `cellphone/image_0037.jpg`. The original clean prediction is correct. The printed-label base prediction is camera; the edited prediction is watch, which is still incorrect. The matching blank-label prediction is also watch. This is consistent with difficulty under the covering label, but the experiment does not establish a causal explanation. An improvement away from camera must not be counted as a repair unless it reaches the correct source class.

The source-grouped paired bootstrap estimates a printed-label accuracy gain over original CLIP of **32.81 percentage points**, with a 95% interval of **21.88 to 43.75 points**, using 2,000 resamples and seed 42. Projection fine-tuning has the same reported interval. These intervals describe resampling this development subset, not population-wide deployment confidence.

## Measured resources and restoration

| Measurement | Recorded value | Interpretation |
|---|---:|---|
| CPU feature preparation | 64.116 s | Encode the manifest's 432 rendered inputs; separate from fitting |
| CPU model load | 2.062 s | Pilot's recorded load, not complete application startup |
| CPU closed-form fit | 0.189 s | Fit the projection patch from cached features |
| CPU fine-tuning | 0.133 s | Exactly 25 projection-only optimizer updates from cached features |
| Saved patch size | 1,617,586 bytes | Includes original projection snapshot and low-rank factors |
| Earlier GPU profile, batch of eight | 0.02198 s median | Random-input forward-pass resource profile only |
| Earlier GPU peak allocated memory | 642,368,512 bytes, about 613 MiB | Profile allocation, not total GPU/process memory |

The GPU profile used an RTX 4050 Laptop GPU, two warmups and eight measured batches. CUDA was unavailable later, so the actual development pilot explicitly ran on CPU. The profile cannot be substituted for measured end-to-end upload latency or gradient-training GPU memory. The two subsecond fitting times are single small runs and do not establish a robust speed advantage for either method.

The pilot verified exact restoration of projection weights and all cached predictions after rollback. The API additionally tests explicit approval, artifact-parent mismatch rejection, image rendering, upload handling, and rollback using small synthetic fixtures. Those software tests are not recognition-accuracy evidence.

## Limits and next evidence

This is a small controlled development experiment with one distractor, one font, a fixed eight-class vocabulary, and source dataset labels. Exact-byte deduplication was checked; independent human label review, near-duplicate/physical-object auditing, natural text-bearing challenge evaluation, and broad pretrained-data contamination auditing were not performed. Caltech101 may have appeared in CLIP's pretraining material.

A separate fixed-configuration held-out evaluation is the next milestone. Development successes and failures remain part of the record. No final results, broad safety claims, or commercial performance claims are inferred from these measurements.
