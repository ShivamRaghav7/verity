# Bounded generative fine-tuning baseline

This baseline asks whether a short, fixed per-edit optimization can teach the
requested answer **and its termination**, then survive patch export, fresh-model
reload and exact rollback. It is a comparison with released MEND, not MEND
meta-training and not a claim that we invented a new editing method.

The [VLKEB FT configuration](https://github.com/VLKEB/VLKEB/blob/10951b7b3788928f578b73f07eac9e1eaa0316f3/hparams/FT/blip2.yaml)
supplies the fixed 15 steps, AdamW learning rate `2e-4`, and the two layer-31
weights `fc1.weight` and `fc2.weight`. We use the implementation's default
weight decay `0.01` (its YAML's separate zero value is not used by the actual
AdamW call), betas `(0.9, 0.999)` and epsilon `1e-8`. All other base weights are
frozen. No hyperparameter search or adaptive step count is exposed.

Our implementation deliberately differs from upstream full-model FP32 training:

- The base retains its released dtypes and evaluation mode. Only the two
  editable weights have FP32 optimizer master copies, with updates copied back
  to their original model dtype. This reduces additional memory; actual memory
  and runtime still require the GPU profile.
- Labels come from the actual contextual suffix of the joined question and
  target. Prefix retokenization is rejected.
- The training target is the requested short answer followed by a newline.
  The newline is included in the supervised suffix, teaching the same stopping
  token used by the generator. The desired answer is recorded separately from
  this training target. No target is supplied during answer generation.

The termination choice is a declared generative adaptation: the earlier MEND
diagnostic could predict target tokens with teacher forcing while continuing to
repeat text during free generation. Teacher-forced token accuracy therefore
does not establish a correct, completed generated answer. This baseline is not
a faithful full-FP32 reproduction of the upstream FT implementation, and it has
no KL regularization.

Inputs are the same verified base assets and fixed development fixture used by
the smoke workflow. Before fitting, the transaction snapshots the patch weights.
It runs exactly 15 updates, checks scalar losses, gradients, master weights and
model weights for finiteness, then records loss at every step, final
teacher-forced NLL, selected parameter names, dtypes and optimizer recipe.
Exceptions restore original weight bytes; all paths clear gradients and editing
hooks and restore the original gradient flags. A successful transaction uses the
existing save/reload/rollback format.

The first GPU profile will use one development edit and the same four diagnostic
queries, followed by fresh-process patch reload and rollback. Its finite update
count is 15; the model worker timeout remains a cap rather than an estimate.
No full-model FT timing, resource requirement or generated-answer result has yet
been measured. Synthetic regression tests establish optimizer and transaction
behavior, not BLIP2 accuracy. Preserve every profile outcome and keep any final
held-out evaluation separate from this development check.
