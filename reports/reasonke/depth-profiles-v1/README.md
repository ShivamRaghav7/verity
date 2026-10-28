# Measured depth profiles: development only

These sixteen requests were completed on 28 October 2026 using source
`7da6d1df5a790c51c7cc95d4378117f5eae052c3` and pinned NF4/FP16 checkpoints.
Each checkpoint received the same eight questions from one development lineage.
The raw ledgers were exported and replay-audited on the laptop. They are timing
and memory profiles, not held-out accuracy or eight independent graph samples.

| Checkpoint | Correct / assigned | Mean request seconds | Maximum request seconds | Observed cold load wall seconds |
| --- | ---: | ---: | ---: | ---: |
| ReasonKE | 8/8 | 15.9742 | 17.9072 | 285.55 |
| Qwen instruction | 8/8 | 1.0889 | 2.4864 | 215.07 |

Both used a Tesla T4, Torch 2.11.0/CUDA 13.0, Transformers 4.56.1,
Accelerate 1.10.1 and bitsandbytes 0.48.1. Peak loading allocation was
7,071,596,544 bytes (6.59 GiB). Peak generation allocations were 5,779,961,856
bytes for ReasonKE and 5,788,171,264 bytes for Qwen. The raw files retain token
counts, actual answers, formats, supplied facts and resource measurements.

Multiplying each measured mean/maximum by 272 gives request-only final forecasts
of 4,344.99/4,870.76 seconds for ReasonKE and 296.19/676.30 seconds for Qwen.
These eight-sample extrapolations exclude loading, downloads and later runtime
variation. They are estimates, not guarantees.

A development phase was subsequently dispatched, but its Colab session expired
before its results were backed up. Its actual outcomes are unavailable; they
are not reported as successes or used for configuration selection. These
preserved profiles are restored unchanged before a disclosed replacement
development run. No final evaluation had started at that interruption.
