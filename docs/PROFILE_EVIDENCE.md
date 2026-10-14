# Training profile evidence

EditGuard releases a substantial training command only after a small, actual
trainer profile binds the chosen inputs, objective, source files and environment.
A profile is a four-update development artifact, not an evaluated trained model.
The application rejects its `smoke_test` model export.

## Historical and preliminary measurements

Existing profile records and their copied run metadata remain immutable. They
document the code and measurements used at that point in development. A later
change to training, preprocessing or preflight source does not erase those
measurements, but it makes their source-bound run cards unsuitable for releasing
a new run with the changed implementation.

This applies to the preliminary answer polarity and entity-neutral scope
profiles, including the records under `reports/answer-polarity-profile`,
`reports/answer-polarity-final-profile` and `reports/scope-neutral-profile`.
Earlier run cards under `configs/run_cards/answer-polarity1` and
`configs/run_cards/scope-neutral2` are retained as planning history. Their names
do not establish eligibility under the current source. Current preflight must
check the exact profile and source hashes; an old card must not be repaired by
rewriting the old measurement or its hashes.

## Final guarded release

The release profiles use fresh paths:

| Component | Profile record | Released run card | Fixed stopping cap |
| --- | --- | --- | --- |
| Scope revision 2 | `reports/profiles/guarded-scope-neutral.json` | `configs/run_cards/targeted-revisions/scope.json` | 360 optimizer steps |
| Answer revision 1 | `reports/profiles/guarded-answer-polarity.json` | `configs/run_cards/targeted-revisions/answer.json` | 240 optimizer steps |

These paths identify the final release contract. A file's absence means its
profile or card has not been completed; this document does not claim a
measurement before that artifact exists. The card records the concrete question,
exact train/validation inputs and review registry, pinned pretrained model,
expected exports, checkpoint plan and a finite stopping condition.

Time planning uses three steady optimizer samples, two complete validation
passes, two checkpoint writes and observed loading/tokenization/export overhead.
The estimate adds a time buffer and a reserved GPU memory margin. It remains
conditional on the measured hardware, software, data shape and storage behavior;
four updates cannot establish full-run runtime or peak memory with certainty.
Read the released card and its bound profile for actual numbers.

The wrappers `scripts/run-scope-final-revision.ps1` and
`scripts/run-answer-revision.ps1` first run preflight and support
`-PreflightOnly`. The user executes the full runs. Resume uses the unchanged
card and the latest complete checkpoint from that run's output. A completed
model or checkpoint at the stopping cap is refused before ML initialization,
and its saved files remain unchanged. Exact-state restoration additionally
checks the model, optimizer, scheduler, RNG, sampler and identity records.

Passing preflight is evidence of a consistent plan. Passing loss/gradient and
checkpoint tests is evidence of the implementation. Neither is evidence of
generated-answer accuracy, generalization or a frozen-test result. Those claims
require the separate evaluation stage after training.
