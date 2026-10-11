# Initial measured feasibility and engineering benchmark

Date: 11 October 2026. Raw records are stored under `reports/`. These measurements
do not score knowledge editing accuracy and do not constitute full model training.

## Observed environment

Windows 11, Python 3.14.3, RTX 4050 Laptop GPU, 6.0 GiB VRAM, 23.71 GiB host RAM,
12 logical CPU threads. The existing host ML installation was PyTorch
`2.9.0+cu130` and Transformers `5.16.1`; CUDA was available. GPU driver was
581.86. The application separately uses its isolated `.venv` and pinned lockfile.
The full ML run must reproduce/reprofile its own isolated environment.

## Real-checkpoint synthetic optimizer profiles

| Candidate | Resolved revision | Median optimizer step | Peak allocated / reserved CUDA memory |
|---|---|---:|---:|
| DistilBERT binary pair classifier | `12040accade4e8a0f71eabdb258fecc2e7e948be` | 0.0816 s | 1.274 / 1.363 GiB |
| FLAN-T5-small conditional generator | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` | 0.1311 s | 1.462 / 1.607 GiB |

Each profile used batch size 2, padded input length 192, float32, AdamW at 2e-5,
one warmup optimizer step and three measured steps on a repeated synthetic
question/correction. The generator target length was 48. Both forward/backward
and optimizer update were timed after CUDA synchronization. Parameters were
updated only in transient profiling memory; no model checkpoint was written.

DistilBERT's binary head was newly initialized, as expected for this encoder
checkpoint. The FLAN-T5 loader reported different saved shared/output weights and
kept them separate. Record this loader behavior when reproducing with the pinned
Transformers version; do not silently change weight tying in future runs.

These measurements demonstrate feasibility for this particular shape. They exclude
model download/load, dataset tokenization, varied examples and lengths, validation
generation, checkpoint I/O, host data-loading cost and other workloads. Three
steady samples are insufficient for a stable full-run runtime estimate. No
accuracy, generalization or full-training duration is inferred from declining
synthetic-batch loss. Profile a reviewed pilot using the eventual trainer before
releasing a substantial user-executed command.

The initial cached-only profile failed because the checkpoint was absent. It is
preserved as `scope-profile-pending.json`, with `status=incomplete`. Subsequent
public checkpoint downloads into the ignored project cache completed successfully.
`scope-profile.json` and `answer-profile.json` are the measured successful runs.

## Structured application's HTTP roundtrip

One sequential localhost run measured 30 requests after 3 excluded warmups,
using a new HTTP connection per request and three known supported questions.
All 30 returned `answered` and `deterministic_structured`; errors and abstentions
were zero. Memory revision remained 1. This is coverage on those three supported
forms, not accuracy on arbitrary questions.

Observed HTTP roundtrip p50 was **15.592 ms** and p95 **19.616 ms**. Timing includes
HTTP transport and JSON decoding. It differs from the dashboard's in-process
fixture service timing. Raw timings and denominator metadata are in
`reports/structured-http.json` and `.csv`. A single small local run does not
characterize concurrency, production service load or neural inference latency.

![Measured structured HTTP timings](../reports/structured-http-latency.png)

## Reproduction

Run the app, then use the bounded `scripts/benchmark_http.py` command described in
the README. `scripts/plot_measurements.py` reads saved JSON and writes PNG/SVG;
install `requirements-plots.txt` in a separate plotting environment if needed.
Its output checks require a completed stable structured benchmark.

The GPU profiler defaults to cached files and has explicit limits of 1–10 measured
steps, batch 1–8 and input lengths 128/192/256. Cache defaults follow Hugging Face;
for project-local reproduction set `HF_HOME` to `runtime/hf-cache`. Checkpoint
downloads require explicit `--allow-download`. No profiler reads the test split.
