# Validation scope calibration

`scripts/calibrate_scope.py` prepares a validation-only operating point for a
completed trained scope classifier. It requires an integrity-checked auxiliary
artifact, the exact declared dataset manifest and the complete human decisions
for `scope_validation.jsonl`. It refuses smoke-purpose models, test-tagged rows,
stale/pending reviews, duplicate IDs, changed files and different datasets.
The input formatter must match training and runtime. Only questions and correction
descriptors enter the classifier; labels enter the subsequent calibration fit.

No calibration model run has been executed yet. Pure numerical fitting and input
gate tests use explicitly synthetic fixtures and do not establish model quality.

## Finite selection procedure

The procedure first selects the temperature with the smallest binary negative
log-likelihood from `0.5, 0.75, 1, 1.5, 2, 3`. An exact NLL tie prefers
the temperature nearest to 1, then the smaller value.
It then evaluates 404 operating points: thresholds `0..1` in increments of `0.01`
and ambiguity margins `0, 0.025, 0.05, 0.1`. A calibrated probability whose distance
from the threshold is at most the margin requests clarification. Other pairs
apply the edit when the probability is above the threshold and reject it below.
This matches the runtime's inclusive uncertainty-band rule.

Operating-point selection minimizes these explicit development costs per pair:

| Outcome | Cost |
| --- | ---: |
| Applying an irrelevant correction | 4 |
| Rejecting an applicable correction | 1 |
| Requesting clarification | 0.25 |
| Correct application or rejection | 0 |

Equal cost first favors fewer false applications and then greater coverage.
Remaining ties prefer a smaller margin and a lower threshold. These costs encode
the project's preference for avoiding incorrect correction application; they are
an implementation choice rather than an accuracy guarantee. Keep costs and grids
fixed before the final test evaluation. Change them through a declared development
configuration/version, never by tuning against test results.

The artifact reports counts, total and mean cost, coverage, clarification rate,
precision, recall, false application rate and selective accuracy. All accepted
validation pairs remain in the accuracy denominator; positive clarifications
remain in the recall denominator. Selective accuracy separately uses covered
pairs. Reviewed rejected dataset rows are disclosed as exclusions. A zero-sized
selective population has an unavailable rate rather than perfect accuracy.

## Inspect inputs before execution

This template checks future reviewed inputs without loading model packages or
writing results. Replace placeholders with the actual completed artifact and
human-review registry when available:

```powershell
.\.venv\Scripts\python.exe scripts/calibrate_scope.py --scope-artifact path/to/completed-scope-run --dataset data/generated/pilot-v1 --reviews path/to/human-reviews.json --output reports/scope-calibration-run-1 --max-rows 120 --preflight-only
```

The row cap applies to the whole accepted validation partition. A larger
partition causes an error; the tool never silently calibrates a convenient subset.
Both positive and negative pairs must be present. The output directory must be
fresh and separate from the model and dataset source directories. The default
device for future inference is CPU; choosing CUDA requires the isolated ML
environment and an explicit device argument. An execution command will be
prepared after trained scope artifacts and genuine reviewed inputs exist.

## Recorded artifacts and limits

Execution produces `validation_probabilities.jsonl`, `selection_grid.json` and
`calibration.json`. The frozen calibration records the scope manifest SHA-256,
dataset manifest and validation-file hashes, review registry hash, original row
hashes, grid, costs, script hash, selected operating metrics and measured elapsed
time. The final calibration file is published atomically after the detailed
artifacts. The runtime accepts it only for the corresponding scope artifact. Set
`calibration_artifact` in an inference configuration to the eventual
`calibration.json` path. Inputs are checked again before publication if their
files changed during inference.

These metrics describe selection on validation data and are optimistic estimates
of behavior on unseen data. Final test evaluation remains a separate stage with
frozen configurations and scoring rules. Pair classification calibration also
does not measure retrieval failure, competition among multiple edits or generator
correctness; report these components and end-to-end results separately.
