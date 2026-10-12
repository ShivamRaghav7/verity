# Initial development pilot: ready for user execution

All 504 labels have an independent automated development review and a separate
integration check. The registry is `reports/pilot-v2-reviews.automated.json`;
**no human approval is claimed**. Neither auxiliary has an eligible trained
checkpoint yet. Four-step profiles are intentionally rejected by inference.

## Questions, inputs and fixed stopping conditions

| Task | Experimental question | Accepted train / validation | Stop |
|---|---|---:|---|
| Scope | Can DistilBERT learn applicability on held-out pilot entities and reduce false applications relative to the embedding baseline? | 240 / 120 | 360 optimizer steps: 3 passes |
| Answer | Can FLAN-T5-small use unseen-entity corrections while preserving subject, relation, value and yes/no polarity under oracle routing? | 96 / 48 | 240 optimizer steps: 5 passes |

Both cards use batch 2, float32, input 192, seed 42, AdamW at learning rate
`2e-5` and linear decay; answer targets have length 48. Inputs are bound to the
same immutable pilot manifest, typed review registry, pinned backbone revisions,
measured profiles and current source/environment hashes. No test input is read.
Lowest validation loss chooses the exported checkpoint; equal loss retains the
earlier step. No accuracy target extends the stopping cap.

The concrete cards are [scope](../configs/run_cards/pilot-v2/scope.json) and
[answer](../configs/run_cards/pilot-v2/answer.json). These local cards contain this
workspace's absolute paths. On another machine restore the exact pilot bundle,
prepare the pinned ML environment, reprofile there and regenerate cards with
`scripts/prepare_pilot_runs.py`; changing paths/hardware is not an exact resume.

## Measured basis and conditional estimates

The actual trainer ran four optimizer updates per task, with two full validation
passes and two checkpoint writes. Reports are
[scope profile](../reports/pilot-v2-scope-profile.json) and
[answer profile](../reports/pilot-v2-answer-profile.json).

| Measurement or plan | Scope | Answer |
|---|---:|---:|
| Observed steady optimizer median | 0.143 s | 0.329 s |
| Observed full validation | 1.22–1.67 s | 1.98–2.42 s |
| Observed checkpoint write/pruning | 8.22–9.62 s | 10.27–17.07 s |
| Observed peak allocated GPU memory | 1.318 GiB | 1.508 GiB |
| Observed peak reserved GPU memory | 1.410 GiB | 1.664 GiB |
| Conditional total-time planning estimate | 441 s (7.35 min) | 432 s (7.20 min) |
| Conditional GPU-memory planning estimate | 2.00 GiB | 2.25 GiB |
| Declared disk budget | 5 GiB | 6 GiB |

Time estimates use the largest of three measured optimizer samples, measured full
validation/checkpoint costs and loading/tokenization/export overhead, then a 2×
planning buffer. Memory uses 25% above observed reserved memory, rounded up to
0.25 GiB. These are conditional estimates, **not guarantees or full-run
measurements**. Thermal behavior, disk/OneDrive variation and competing processes
can change them. The inspected RTX 4050 Laptop GPU has 6 GiB; profile records
contain the actual package, driver, process and tensor-resource information.

## Execute

The isolated `.venv-ml`, pinned cached backbones and reviewed packet are already
prepared in this workspace. Both cards and the wrapper's preflight-only path
passed without starting a full run.

```powershell
.\scripts\run-pilot-training.ps1 -PreflightOnly
.\scripts\run-pilot-training.ps1
```

The second command trains scope and answer sequentially. To run one task use
`-Task scope` or `-Task answer`. The wrapper sets the absolute project `HF_HOME`
and cached-only loading. Missing inputs or invalid preflight stop before a new
optimizer run. It does not install packages, modify labels or read test data.

Expected outputs are `artifacts/scope/pilot-v2-initial` and
`artifacts/answer/pilot-v2-initial`, each containing selected model/tokenizer,
`manifest.json`, `run_card.json`, `training_log.jsonl`, checkpoints and the latest
checkpoint pointer. The artifacts record `review_policy=automated_development`
and `human_reviewed_data=false`. They are model training results, not final
evaluation results or independent human validation.

## Checkpoint and resume

Checkpoint and validation occur once per pass: every 120 scope updates or 48
answer updates, plus the final cap. Checkpoints contain model, optimizer,
scheduler, RNG, sampler/order/cursor, global step, selected validation state and
data/configuration/environment/source identities. Retention keeps the last two
and a distinct earlier best checkpoint. Writes publish atomically; incomplete
temporary state is not resumed.

For an interruption, use the last **completed** directory listed in
`latest_checkpoint.json`, with the same card and source/environment:

```powershell
.\scripts\run-pilot-training.ps1 -Task scope -ResumeCheckpoint 'C:\full\path\to\completed\step-00000120'
```

That path is an example, not a claim that a checkpoint currently exists. Do not
change steps, learning rate, data, decisions, source, hardware or packages and
call it an exact resume. CPU regression already verifies uninterrupted/resumed
parameters, losses, selection, sampler and RNG equality; earlier CUDA smoke
exercised actual checkpoint restoration separately.

## After these runs

Check artifact integrity, calibrate the scope model on the 120 accepted validation
pairs, evaluate the answer model separately under oracle routing, then run all
five matched configurations with recorded failures and fixed denominators. Keep
threshold selection and validation reporting separate from final-test performance.
The pilot has six entity families and twelve shared templates, with positive
edits only in Basic-v2. It cannot establish unrestricted wording, dense memory,
other-plan editing or temporal generalization. At most two targeted development
revisions per model are allowed by the project plan; disappointing results do not
expand this experiment indefinitely.
