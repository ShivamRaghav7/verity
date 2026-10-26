# Measured two-checkpoint development profiles

Both pinned checkpoints loaded on a fresh Tesla T4 and each completed 12 assigned
requests. These are **profiling and development observations**, not the final
held-out comparison. The profile conditions deliberately use different balanced
question subsets, so their total correct counts must not be compared as model
accuracy. The subsequent development comparison uses matched questions.

| Model | Download/loading seconds | Mean request seconds | Slowest sampled request | Load peak reserved CUDA |
| --- | ---: | ---: | ---: | ---: |
| Author ReasonKE_plus | 359.76 | 14.21 | 16.46 | 7,140,802,560 bytes |
| Official Qwen2.5-7B-Instruct | 259.57 | 1.12 | 2.94 | 7,140,802,560 bytes |

Both used NF4 double quantization with FP16 computation, greedy decoding,
4,096 input tokens and at most 512 output tokens. Maximum reserved CUDA during
these profile generations was 7,205,814,272 bytes. These are PyTorch measurements,
not total process/system VRAM. Request timing covers context selection, local
worker communication and inference; it excludes model loading and the browser
network. Model identity and resource fields are preserved per generation.

The source checkout was `e4f82a3e6d25b99464327dded2052719446db028`. Each receipt's
source/data/model/scoring hashes were checked against the local files. The
development dataset and question-only retriever are fixed in the same identity.
No training ran. The generic checkpoint's exact ancestry relationship with the
released ReasonKE weights is not established.

## What actually happened

- The stale R0 examples returned values from their supplied initial world.
  They are marked wrong against the post-edit target, as intended for the stale
  diagnostic. This is not evidence that the model failed to read the initial facts.
- ReasonKE matched the four sampled retrieval/locality controls and both sampled
  direct full-context targets. Its two sampled full-context two-hop questions
  returned the old team's contact despite identifying the new owner correctly.
  The raw outputs preserve this join failure; no response was retried.
- The instruction model matched eleven of its twelve sampled questions. The
  remaining full-context two-hop paraphrase returned `UNKNOWN` and is scored as
  abstention, with zero correct-answer credit. The strict raw outputs are retained.
- There were no execution errors or unexecuted requests in these two completed
  profiles. This does not mean there were no incorrect neural answers.

An earlier loading attempt was stopped after additional pre-run ledger checks
were added. It produced zero model generations. Its original receipt remains
`interrupted-loading-zero-generations.json`, with its original running status;
that status does not indicate an active process. The owned processes were stopped,
and the revised source used a new output directory. No measured answer was removed
or replaced by that restart.

## Forecasts, not promised runtimes

For the 1,312 planned final-stage requests (1,280 main plus 32 oracle), mean-profile
extrapolation is about 11,138 seconds for 784 ReasonKE requests and 591 seconds for
528 instruction-model requests: roughly 3.26 hours of request work combined.
Using each profile's slowest request gives about 4.02 hours. Cold loading and
download time are additional; these small-sample extrapolations are uncertain.
Hard load, request and per-model wall watchdogs remain separate safeguards.

For the 120-request matched development stage, the same samples imply about
18–22 minutes of request work plus loading. No final model or scoring choice
will be made using held-out outputs. The platform's context-policy setting will
be selected from development observations before the final freeze.
