# Model and editing-system card

EditGuard is a research workbench for maintaining a visual language model. It
uses the published VLKEB BLIP-2/OPT-2.7B architecture and MEND editing checkpoint.
The upstream authors trained the foundation components and learned MEND
transforms. We do not claim that training as our work.

## Components and our changes

| Component | Parameters and behavior | Training attribution |
|---|---|---|
| Base | EVA visual encoder, Q-Former and OPT-2.7B; fixed initial checkpoint | Published pretrained weights |
| MEND | Learned gradient transforms update six OPT MLP matrices in layers 29–31 | Published pretrained editor; our explicit factor/lifecycle adapter |
| Last-layer FT | Fifteen AdamW updates at 2e-4 on layer 31's fc1/fc2 weights | Actual per-correction optimization in our runs; not editor meta-training |
| Updated visual RAG | Current correction memory, MiniLM question vectors and BLIP visual-prefix vectors | Published frozen encoders; our threshold selected on separate development rows |
| Prompt-only | Current correction supplied as a QA example, with unchanged weights | No training |

The complete model is not converted to FP32. FT uses FP32 AdamW master copies
for the two selected matrices, retaining base precision and evaluation mode.
Its contextual labels include an answer-terminating newline. MEND's contextual
label variant fixes the standalone/joined tokenizer mismatch; its historical
release-label mode remains available. These are documented adaptations, not a
claim of faithful reproduction of every upstream benchmark setting.

The generated-answer path receives the question and image, without the target.
Teacher-forced diagnostics deliberately include the target and remain separate.
Our patch format stores original and edited tensors. Rollback copies originals;
subtracting a rounded FP16 update would not guarantee byte-identical restoration.

## Intended use and practical limits

The intended operator is a model maintainer investigating a specific visual
factual correction. Preview, inspect target matching and one unrelated probe,
then explicitly approve activation or discard the candidate. An approval is a
recorded operator decision, not a proof of universal correctness. Only one
parameter patch is active at a time; composition and lifelong editing are not
implemented. A conflicting activation is rejected until the active edit is
rolled back.

Development runs demonstrate both successful changes and collateral damage.
Exact restoration establishes reversibility of the selected tensors, not answer
accuracy. Sources and hashes establish provenance, not verification of arbitrary
generated prose. The small benchmark does not establish private-item recognition,
general dialogue reliability, production readiness, or arbitrary-edit success.

The completed [20-edit evaluation](VLKEB_RESULTS.md) retains all 500 assigned
answer records. Its source, raw outcomes, uncertainty and arithmetic verification
are preserved separately from development examples. FT matched 60/60 correction
queries but 0/40 unrelated targets; MEND retained the five initially correct
unrelated answers with lower correction success. The five-answer retention
denominator is too small to establish broad locality reliability.

Measured development runs use a Tesla T4 with 15,360 MiB VRAM. The FT worker's
recorded CUDA peak is 11,971,964,416 allocated bytes and 12,465,471,488 reserved
bytes. These figures concern that fixed run, not every query or environment.
The laptop's 6 GB GPU is not the demonstrated full-model deployment target.

## Reproduction and rights

Exact source revisions, checksums, dependency pins and required notices are in
`configs/vlkeb-assets.json`, `requirements-vlkeb.txt` and the run receipts.
The additional MiniLM files have their own pinned manifest and integrity checks.
Large weights and saved patches stay outside Git; preparation scripts fetch the
declared published assets, and the application exports its own patches.

VLKEB code/editor, OPT, LAVIS components, MiniLM and source images have separate
terms. In particular, an MIT editor does not remove OPT's noncommercial research
restrictions or individual image rights. See the asset notices and
[upstream audit](UPSTREAM_PROJECT_SCOUT.md). Final claims must cite our saved
evaluation, with author-reported numbers identified separately.
