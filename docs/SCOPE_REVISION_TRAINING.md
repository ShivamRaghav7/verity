# Scope revision 1: prepared finite run

This run asks whether balancing the existing negative question forms improves
direct-edit detection on held-out pilot entities without increasing false
correction applications or reducing paraphrase/comparison recall. The original
classifier missed all 12 applicable direct validation questions. The isolated
Nimbus workflow also falsely applied a device correction to a price question.
Those are observed failures; the wording imbalance is a hypothesis about a
contributing cause, rather than an established explanation of every error.

This is the first of at most two targeted scope revisions. It does not expand
the catalogue or row count. The answer checkpoint stays fixed at its completed
initial pilot export; its comparison-polarity failure remains documented.
No substantial revised training run has been executed by Codex.

## Exact inputs and outputs

- Dataset: `data/generated/pilot-v3`, with 240 scope training pairs and 120
  validation pairs. Entity families, facts, lineages, scope labels and split
  assignments are unchanged. Each of the four question forms now has a 40%
  applicable prior. No test or challenge partition is read.
- Review: `reports/pilot-v3-reviews.automated.json`; all 504 rows were independently
  machine-audited afresh, with 504 accepted and zero flagged. This remains
  `automated_development`, with `human_reviewed_data: false`.
- Model: pinned DistilBERT revision
  `12040accade4e8a0f71eabdb258fecc2e7e948be`, initialized from pretrained weights.
  The same seed 42, float32, batch size 2, 192-token budget, optimizer and learning
  rate are retained. This is not continuation of the initial trained checkpoint.
- Run card: `configs/run_cards/pilot-v3/scope.json`, bound to exact data, review,
  profile, trainer source and environment hashes, plus the initial scope manifest.
- Output: fresh `artifacts/scope/pilot-v3-revision1`, containing model, tokenizer,
  completed manifest, run card, contiguous training log and exact-state checkpoints.
  Initial model exports, v2 data and v2 evaluation records are preserved.

All 360 scope inputs fit 192 tokens without truncation. The
[tokenization audit](../reports/pilot-v3-scope-tokenization.json),
[independent review](PILOT_V3_REVIEW.md) and
[data derivation](SCOPE_DATA_REVISION.md) record these separate checks.

## Profile-grounded planning estimate

The actual trainer performed exactly four profiling updates on this reviewed
packet, with full 120-row validation and checkpoint writes at steps 2 and 4.
The GPU was the inspected RTX 4050 Laptop with 6,141 MiB capacity and no other
allocated GPU memory immediately before profiling. Profiling artifacts are
`smoke_test` and are refused as trained models by the application.

| Measured profiling component | Observation |
|---|---:|
| Three steady optimizer samples | 0.094430, 0.083044, 0.091242 seconds |
| Full validation samples | 1.140310, 1.151962 seconds |
| Checkpoint write/pruning samples | 6.156470, 13.228943 seconds |
| Measured load/tokenize/export overhead | 22.745251 seconds |
| Peak allocated / reserved CUDA memory in recorded window | 1.318464 / 1.410156 GiB |
| Largest sampled process RSS | 1.667858 GiB |

The prepared card budgets **200 seconds, 2.0 GiB GPU memory and 5 GiB disk**.
The runtime estimate uses the slowest observed steady optimizer sample × 360,
the slower full validation and checkpoint samples × 3, measured overhead and
a 2× time buffer. The GPU budget uses a 25% reserved-memory margin, rounded up
to a quarter GiB. These are conditional planning estimates, not guarantees;
disk/OneDrive activity, thermal throttling and competing processes can change
them. Sampled RSS is not a lifetime host-memory peak or a guaranteed RAM bound.

Raw observations are in [the profile](../reports/pilot-v3-scope-profile.json).
Exact profile manifest and log bytes are archived under
`reports/pilot-v3-profile/scope`; they contain no deployable trained weights.

## User-executed command and finite recovery

Both the run-card preflight and the PowerShell wrapper's preflight have passed.
From this repository root, execute:

```powershell
.\scripts\run-scope-revision.ps1
```

The stopping condition is **global step 360**, or three passes over accepted
training pairs. Validation and checkpointing occur every 120 steps. The exported
checkpoint has the lowest validation loss, with earlier step winning a tie.
No accuracy target extends the run, and the script launches no answer training,
automatic revision, calibration or final-test evaluation.

After an interruption, preserve the run directory and use the same unchanged
card with the **latest completed checkpoint** directory. Inspect
`artifacts/scope/pilot-v3-revision1/latest_checkpoint.json` for its path:

```powershell
.\scripts\run-scope-revision.ps1 -ResumeCheckpoint artifacts/scope/pilot-v3-revision1/checkpoints/step-00000120
```

Use the example above only if step 120 actually is the latest completed checkpoint
and precedes the cap; it does not claim that this directory already exists. The
wrapper refuses an older checkpoint or one from another run before training.
Resuming an older retained state can collide with later checkpoint directories,
which the trainer refuses to overwrite. Exact resume restores model, optimizer,
linear scheduler, RNG, sampler cursor and step, and refuses changed data,
configuration, source or environment identity. The two latest checkpoints plus
any distinct earlier best checkpoint are retained. To validate without training:

```powershell
.\scripts\run-scope-revision.ps1 -PreflightOnly
```

## Assessment after completion

Audit the exported checkpoint, select its operating threshold using only the
declared validation rule, and run the same finite comparison harness. Reassess
the embedding baseline under the corresponding validation protocol. Report raw
scope precision/recall, false applications, per-form performance and coverage,
then separate actual routing from generated-answer quality.

The 48 applicable validation questions are unchanged and support a paired
before/after diagnostic. The 72 revised negative questions differ in wording;
the full 120-question comparison must not be presented as the identical original
paired population. Any improvement remains development evidence. Final scoring
rules/configurations and test freezing stay pending; test data must not be used
to choose another revision.

For exact packet restoration on a fresh checkout, use the manifest-bound
`data/pilots/pilot-v3-source.zip` via `scripts/pilot_bundle.py unpack`. The packet
contains nine allowlisted development files, no review registry or test files;
the separate tracked machine registry supplies its explicitly typed decisions.
