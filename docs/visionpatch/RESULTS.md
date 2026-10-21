# VisionPatch measured results

Evidence: `reports/visionpatch/pilot-v1.json`, `reports/visionpatch/final-v1.json`, its preceding `final-seal-v1.json`, the separate `encoder-profile.json`, and `http-smoke-v2.json`, all under `reports/visionpatch/`. The development pilot, frozen final evaluation, and live HTTP workflow check were completed on 21 October 2026. These finish the bounded controlled study.

## Frozen held-out result

Before final inference, a seal fixed the original model revision, fitted patch and fine-tuning weights, candidate vocabulary, transform, scoring rule, bootstrap settings, and runtime bindings. The final manifest contains 80 additional Caltech101 originals, ten per class. Exact hashes overlap with none of the 144 development originals. There was no new training or configuration selection during final evaluation. These are new source images from the same dataset, not a new external deployment domain.

| Frozen configuration | Clean correct | Blank-label correct | Printed-label correct | Induced errors repaired | New errors on baseline-correct printed inputs |
|---|---:|---:|---:|---:|---:|
| Original CLIP | 80/80 | 80/80 | 49/80 | 0/31 | 0/49 |
| Projection edit | 80/80 | 80/80 | 80/80 | 31/31 | 0/49 |
| Projection fine-tuning | 80/80 | 80/80 | 80/80 | 31/31 | 0/49 |
| Oracle blank-region masking | 80/80 | 80/80 | 80/80 | 31/31 | 0/49 |

All three interventions tie on the final aggregate outcomes. Perfect counts on these 80 controlled examples do not imply perfect accuracy on arbitrary images, different text, or other candidates. They do not demonstrate superiority of editing over the simpler controls.

The attack denominator is 70 baseline-clean-correct non-camera sources; 31 acquire errors when the camera word is added, or 44.29%. All 31 are repaired by each intervention. Against every final source, the repairs are 31/80, or 38.75%. All ten genuine cameras remain correctly recognized. There are no abstentions; every prediction is scored.

Unseen images from the two support classes improve from 7/20 to 20/20 printed-label predictions, repairing 13/13 eligible errors. The other six classes improve from 42/60 to 60/60, repairing 18/18 eligible errors; that attack denominator excludes ten genuine cameras and is therefore 50, not 60. Exact rollback restores original weights and all final predictions.

The paired source-bootstrap accuracy gain is **38.75 percentage points**, with a 95% interval of **28.75 to 50.00 points**, for each intervention versus original CLIP, using 2,000 resamples and seed 42. This summarizes uncertainty from resampling the small same-source final set, not a production guarantee. The shared bootstrap helper's output includes the inherited phrase “Small development subset”; here its 80-source inputs are the sealed final observations, as the final report status and manifest show.

CPU final feature preparation took 31.746 s for 240 variants in 30 batches; model loading took 1.274 s. CPU peak RAM was not instrumented. These batch measurements are separate from single-request HTTP latency.

## Development experiment in plain language

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

## Live HTTP workflow and latency

`http-smoke-v2.json` explicitly records CPU inference. It uploads one development airplane image with the printed camera label: original prediction camera, patched prediction airplane, then rollback prediction camera with the original projection hash restored.

The first HTTP inference took **11.737 s**, starting before the encoder was loaded. Twelve subsequent serial requests on the same image measured **p50 243.249 ms** and **p95 326.934 ms**; p95 uses nearest rank `ceil(0.95 × 12)`. This verifies the upload path and gives a small warm-latency sample. It is not a recognition benchmark, throughput measurement, or concurrency test. The earlier `http-smoke-v1.json` did not record its device, so its timings are not used as authoritative CPU measurements.

## Scope and limitations

This is a small controlled development and held-out study with one distractor, one font, a fixed eight-class vocabulary, and source dataset labels. Exact-byte deduplication was checked; independent human label review, near-duplicate/physical-object auditing, natural text-bearing challenge evaluation, and broad pretrained-data contamination auditing were not performed. Caltech101 may have appeared in CLIP's pretraining material.

The small fixed-configuration held-out evaluation and workflow checks are complete. Development successes and failures remain part of the record, including the cell-phone failure above. New words, fonts, text-bearing tasks, external-domain images, and production hardening are optional extensions requiring their own data and evaluation; they are not established by the finished study. No broad safety or commercial reliability claim follows from these measurements.
