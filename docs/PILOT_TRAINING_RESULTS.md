# Initial pilot-v2 training results

Both auxiliary models completed the user-executed, finite GPU runs. The scope
classifier completed 360 optimizer steps; the edit-conditioned answer model
completed 240. Their validation-selected models, tokenizers and exact-state
checkpoints are present locally. This section reports training and artifact
measurements. **Validation loss is not factual accuracy, scope precision/recall,
calibration, or end-to-end answer quality.** Those evaluations remain separate.

The dataset is the fictional controlled development pilot
`editguard-controlled-v1-pilot-v2-42-7b7a9b1b-grammar-80ce36e5`, with input format
`editguard-input-v2`. Its labels used the user-authorized
`automated_development` review policy. Both model manifests explicitly record
`human_reviewed_data: false`; independent human review has not been performed.
No test or challenge partition was read for these training results or this audit.

## Runs and checkpoint selection

| Measurement | Scope classifier | Edit-conditioned answer model |
|---|---|---|
| Backbone | DistilBERT base uncased | FLAN-T5-small |
| Pinned pretrained revision | `12040accade4e8a0f71eabdb258fecc2e7e948be` | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` |
| Accepted training / validation rows | 240 / 120 | 96 / 48 |
| Completed / maximum optimizer steps | 360 / 360 | 240 / 240 |
| Training passes at batch size 2 | 3 | 5 |
| Validation and checkpoint interval | 120 steps | 48 steps |
| Selected checkpoint step | 120 | 240 |
| Selected validation loss | 0.3907963567 | 0.0444124042 |
| Exported local artifact | `artifacts/scope/pilot-v2-initial` | `artifacts/answer/pilot-v2-initial` |

Both runs used seed 42, float32, AdamW with initial learning rate 0.00002,
weight decay 0.01, gradient clipping at 1.0, and a fixed linear schedule reaching
zero at the cap. The input budget was 192 tokens; the answer target budget was
48 tokens. All accepted input/target rows fit those budgets without truncation.
The saved environment identifies Windows 11, Python 3.14.3, Torch 2.9.0+cu130,
Transformers 5.16.1, tokenizers 0.23.3, NumPy 2.4.2 and safetensors 0.8.0.
Both runs used the inspected NVIDIA GeForce RTX 4050 Laptop GPU and driver 581.86,
two CPU threads, deterministic algorithms, eager attention and disabled TF32.

Scope validation loss is averaged over examples. Answer validation loss is
teacher-forced and weighted by non-padding target tokens. It can be low while an
autoregressive answer endorses a wrong fact or subject. No loss value is converted
to an answer accuracy number.

| Scope validation step | Loss |
|---:|---:|
| 120 | 0.3907963567 |
| 240 | 0.4536316596 |
| 360 | 0.4449242949 |

The scope run's later checkpoints had higher validation loss, so the exported
classifier uses step 120, after one training pass, despite completing the declared
three-pass run. This preserves the validation-based selection rule and avoids
claiming that more optimizer steps improved held-out behavior.

| Answer validation step | Loss |
|---:|---:|
| 48 | 0.3720425941 |
| 96 | 0.1462760383 |
| 144 | 0.0724859082 |
| 192 | 0.0493472222 |
| 240 | 0.0444124042 |

The selected answer checkpoint is the final capped step. These are single-seed
observations on a small development dataset with shared wording templates; they
do not establish robustness across seeds, unfamiliar wording, arbitrary support
dialogue, or edits to other plan/version scopes.

## Measured time and resources

| Recorded measurement | Scope | Answer |
|---|---:|---:|
| Training-loop duration | 89.448 s | 245.856 s |
| Sum of optimizer-phase durations | 36.348 s | 76.747 s |
| Median optimizer-phase duration | 0.096467 s (n=360) | 0.304167 s (n=240) |
| Sum of full validation durations | 4.119 s (3 passes) | 9.804 s (5 passes) |
| Sum of checkpoint writes and pruning | 47.333 s (3 writes) | 158.239 s (5 writes) |
| Other recorded loop overhead | 1.648 s | 1.066 s |
| Peak allocated CUDA memory in the recorded window | 1.318464 GiB | 1.508413 GiB |
| Peak reserved CUDA memory in the recorded window | 1.410156 GiB | 1.664062 GiB |
| Maximum sampled process RSS | 2.014477 GiB | 3.578484 GiB |

One GiB equals 2^30 bytes. CUDA peaks were collected after the trainer's loop
reset. Process RSS is the largest recorded sample, not an operating-system
lifetime high-water mark. Model loading/tokenization precede the loop timer;
selected-checkpoint loading/export and final manifest writing follow it.
Their wall time and memory are not fully measured by the loop records.
**Complete process wall time is unavailable.** Do not compare these loop totals
directly with the run cards' estimated full-run durations or report them as the
entire training command's duration.

The final log rows precede the final checkpoint writes: they show 78.384 s and
189.237 s, respectively. The manifest's 89.448 s and 245.856 s loop totals include
those final writes. Reporting only the last log timestamp would understate the
measured loop cost. Checkpoint I/O accounted for substantial time in these runs;
the result does not imply a stable disk speed or a future runtime guarantee.

## Independent artifact and data audit

`reports/pilot-v2-training-audit.json` records 87 passing identity, checksum,
selection and tensor checks, with raw measurements and exact hashes. The audit
used the core environment and standard library; it loaded no Torch objects,
models or GPU tensors and executed no training. It verified:

- Runtime artifact eligibility and every exported model/tokenizer file hash.
- Dataset manifest, four source data files, review registry, audit binding,
  accepted-row hashes and tokenization counts for each task.
- Exact saved run-card bytes, configuration/environment hashes, pinned model
  revisions, trainer/profile source hashes and profile-file hashes.
- Contiguous logs covering precisely the 360/240 step caps, finite losses and the
  lowest logged validation loss with the declared earlier-step tie rule.
- Byte checksums and metadata/payload identities for all retained checkpoints.
- Exact selected-checkpoint/export tensor bytes and independently reproduced
  selected/final tensor-state SHA-256 values.

Checkpoint inspection used an explicit allowlist to reconstruct opaque tensor
and storage references from the PyTorch archive. It did not import Torch or
instantiate models, optimizers or tensor objects. Safetensors headers and raw
bytes were compared with the selected checkpoint, then hashed using the trainer's
name/dtype/shape/value algorithm. For FLAN-T5, the serialized export has 190
tensor keys while the checkpoint model state has 192: the encoder and decoder
embedding names share the exported `shared.weight` values. Both omitted aliases
were checked by shape, dtype and exact bytes; the separate `lm_head.weight`
remains present. No head weights were silently retied by the audit.

| Identity | SHA-256 |
|---|---|
| Data manifest | `8ab5b924549b2ff4039384c476e09075a74f58d471195cb7f9e6ed0c62eb7559` |
| Automated development registry used by both runs | `a43554b6b39268472bb7c467e6bc502f0a0301efd6cdd880deb1958a7c824688` |
| Scope run card | `4ed5f458d11f8337d1efb2f4de9da26a1b09399ab4f6f20d6afafb5a209ec65c` |
| Answer run card | `8c0c6fae4c91b5ea59c8380c82b0b1ff58e26fca381c6091ea56f478c20f09a2` |
| Scope selected model tensor state | `64e979e679e583444a080c5a3b61df659f850c4342c1f575eedeba954e3d9bd5` |
| Answer selected model tensor state | `ec06257db751aa90739592ceb4486550e0f2dbb8122ca334a446cb19255520b3` |
| Scope exported safetensors file | `77877e99b49ed22c385f17288f6ac3a9baf08feaec58b1e6a02742d6e5c22cd0` |
| Answer exported safetensors file | `ddcbe06010ea5a627551cdc5852baf828e3681ca57e280cedff44ec6e3396d0f` |

The exact trainer versions and every retained checkpoint/export hash are stored
in the machine-readable audit. Input and artifact metadata hashes stayed unchanged
through the audit. Recorded selected states differ from the trainer's recorded
pretrained states; pretrained-state fingerprints are trainer provenance, rather
than newly reconstructed pretrained-model states in this audit.

## Recovery and remaining evidence

Scope checkpoints retained locally are steps 120, 240 and 360: the two most recent
plus the selected best checkpoint. Answer checkpoints retained are steps 192 and
240. Earlier answer checkpoints were pruned according to the declared policy,
while their validation losses and write measurements remain in the log/manifest.
Each checkpoint contains model, optimizer, scheduler, RNG, deterministic sampler,
global step, run/data/environment identity and best-validation state, all bound to
its recorded file checksum. Latest pointers identify scope 360 and answer 240.

Both saved run cards have `resume_checkpoint: null`. Their contiguous logs and
increasing elapsed timers are consistent with uninterrupted capped runs. This
audit claims no resume event for these two runs. Resume capability and the prior
separate recovery regressions remain distinct evidence; recovery must retain the
original caps, schedule, data/review identity and environment.

Model/checkpoint binaries are local ignored artifacts. Publishing their metadata
and hashes does not distribute deployable weights; deployment requires these
actual verified exports or reproduction of the pinned runs. These two trained
auxiliaries satisfy the initial training milestone. Scope calibration, oracle
generation, five matched baselines and learned integration now have separate
[development validation evidence](PILOT_VALIDATION_RESULTS.md), including the
reported scope and comparison-polarity failures. The
[exploratory short-QA analysis](SHORT_QA.md) preserves the original strict scores.
The bounded scope-data revision, independent final review, configuration/test
freeze, final held-out evaluation and polished final report remain incomplete.
