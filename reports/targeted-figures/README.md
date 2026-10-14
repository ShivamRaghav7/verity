# Targeted revision figures

These figures show completed **development** measurements from the controlled fictional dataset. Labels received automated development review; they are not claimed as human-reviewed data. Frozen final evaluation remains pending.

| Figure | What it measures | What it does not establish |
|---|---|---|
| [Scope diagnosis](scope-neutral-diagnosis.png) · [SVG](scope-neutral-diagnosis.svg) | Raw classifier decisions and NLL on the same 120 pilot-v3 validation pairs, with shared and unshared abstract inputs separated | Calibrated retrieved routing, generated-answer accuracy or arbitrary-language identity verification |
| [Answer polarity diagnosis](answer-polarity-diagnosis.png) · [SVG](answer-polarity-diagnosis.svg) | Teacher-forced first-token polarity and both common loss objectives on identical 48 pilot-v2 validation targets | Free-running complete-answer accuracy or reliable unfamiliar arithmetic |
| [Training resources](targeted-training-resources.png) · [SVG](targeted-training-resources.svg) | Recorded capped training-loop durations, validation objectives, GPU peaks and sampled process RSS | Full command duration, startup/export cost, lifetime RSS peak or inference latency |

Scope accuracy at the uncalibrated threshold 0.5 increased from **98/120 to 119/120**, while mean pair NLL decreased from **0.485679 to 0.043265**. After excluding the eight identical abstracted inputs shared with training, accuracy increased from **90/112 to 111/112**. All eight shared inputs are positive. Entity-neutral preprocessing is a deterministic literal-normalization adaptation; it does not train the model to verify product identity.

For the answer model, expected-No first-token decisions improved from **0/12 to 4/12**, and expected-Yes decisions decreased from **12/12 to 11/12**. Both common full-target validation objectives became worse: unweighted token NLL increased from **0.044412 to 0.061731**, and the fixed comparison-first-token 8× objective increased from **0.162544 to 0.168240**. These are teacher-forced measurements; model selection must also consider complete generated answers.

The recorded training loops took **112.55 seconds** for 360 scope updates and **264.54 seconds** for 240 answer updates. Loop timing includes validation and checkpoint writes but excludes startup, tokenization and final export. Complete command wall times were not recorded.

Reproduce all figures into a fresh directory:

```powershell
.\.venv-ml\Scripts\python.exe scripts/plot_targeted_results.py --output runtime/targeted-figures-reproduced
```

The script reads only completed diagnostic records and archived training metadata. It recomputes counts and losses, checks recorded SHA-256 hashes, confirms original inputs remain unchanged, and exports PNG and SVG with input provenance. It loads no neural models and reads no dataset, test or challenge files. A second run reproduced all six exports and the manifest byte for byte in the measured environment (Matplotlib 3.10.7). Input and output hashes are recorded in [manifest.json](manifest.json).
