# Training preflight contract

`scripts/preflight_training.py` checks a concrete finite run card and performs no
training, model import or download. It is preparatory validation. A successful
preflight does **not** mean a training loop, checkpoint restoration or model
quality has been verified. The next ML stage must implement and test those before
providing substantial user-executed commands.

The card requires explicit absolute paths for data directory, task-specific
training/validation JSONL, review registry, profile and output directory. Include
exact manifest/review/profile SHA-256 hashes, resolved 40-character model revision,
model ID, batch size, input/target lengths and precision matching the measured
profile. Only accepted human-reviewed training/validation rows are eligible;
pending decisions or stale content hashes fail. Test inputs are rejected, and
reviewed families/entities/lineages must be disjoint between train and validation.

Declare a concrete experimental question, seed, learning rate, positive finite
step cap (initial guard: at most 100,000), checkpoint interval within the cap,
checkpoint retention, expected artifact names, exact-state resume plan and runtime
and GPU-memory estimates with their assumptions. Estimates cannot be below the
measured optimizer-step extrapolation or peak allocation; they must also account
for validation/generation/I/O limitations before an actual full run is released.

The current profiler measures a repeated synthetic batch from a real checkpoint.
Its latency/memory establish bounded shape feasibility, with explicit limitations.
They do not establish reviewed-data length distribution, final accuracy or complete
training wall time. Preflight preserves those limitations in its output.

Checkpoint plans include model, optimizer, scheduler, RNG, global step, data hashes
and configuration hash; add scaler state if mixed precision is later introduced.
An intended resume path must be under the declared output directory and contain
matching task/model/data/review metadata with a step below the stopping cap.
Preflight checks declared metadata; restoration itself remains next-stage work.

Expected artifacts include `model`, `training_log.jsonl`, `run_card.json` and
`checkpoints`. The output directory must not overwrite the input dataset. No
example containing invented profiling measurements is provided as a runnable
production card. Unit-test measurements are explicitly synthetic fixtures.

The bounded preflight command, after a real card is prepared, is:

```powershell
.\.venv\Scripts\python.exe scripts/preflight_training.py path/to/real_run_card.json
```

Human label review implements the user's dataset-quality protocol. It is not an
extra permission request or a training approval workflow.
