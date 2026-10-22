# Model and dataset record

## V2 extension

The backbone and eight object candidates below are unchanged. V2 adds three
independent edit intents (`camera`, `laptop`, `watch`), each fitted from four
training-support pairs. Text-protected editing uses second-moment retention from
64 clean photos plus 32 controlled word cards. A fixed validation grid selects
one protection weight and one fine-tuning rate globally. The object-only ablation
and task-aware privileged masking control are also reported. All six sequential
orders use parent-bound fits before final scoring.

V2 has 80 training originals reused from V1 support/retention, 32 fresh validation
and 48 fresh final originals. Validation/final exclude all 224 V1 originals by
exact hash and identifier. Training/validation/final fonts and text-card template
groups differ. Final photo overlays use two positions; 96 target questions share
48 underlying photos. The 24 text cards cover eight words across three templates,
not 24 independent semantic concepts. Labels have automated audits and no
independent human review. Near-duplicate/physical-object overlap and pretraining
contamination remain unknown.

The frozen V2 study reports target repair, genuine-object accuracy, text
preservation, other-word collateral errors, grouped uncertainty and sequential
interference. Protection preserved the watch edit's text answers but reduced
laptop repair relative to object-only editing. It does not establish general
locality or superiority to fine-tuning. See [RESULTS_V2.md](RESULTS_V2.md),
[STUDY_V2.md](STUDY_V2.md) and the exact committed manifests in
`data/manifests/visionpatch`.

The interactive studio fits **one** supplied pair and maintains its own persistent
registry. Its previews evaluate development references and do not inherit the
formal four-pair study's final accuracy. Approved patches alter actual projection
parameters; user-supplied expected labels do not enter recognition requests.

The remainder records the preserved V1 experiment.

## Model

- Backbone: `openai/clip-vit-base-patch32`.
- Pinned revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- Task: choose the main object from eight fixed labels, including the printed
  distractor's genuine class.
- Trainable/editable component: the bias-free 512-by-768 visual projection.
  Both pretrained encoders, class prompts and text embeddings remain fixed.
- Editor: covariance-regularized low-rank projection update, learned from four
  attacked/blank paired photographs and 64 distinct clean retention references.
  Rank is at most four. It minimizes a representation-matching objective; it
  does not implement a handwritten prediction override.
- Comparison: 25 gradient updates to the same projection, with the same example
  pairs and reference photographs. The finite pilot configurations were not an
  exhaustive hyperparameter comparison. Both learned methods tied on the final
  study.

All images use the same explicit bicubic square crop and CLIP normalization. The
crop uses Pillow `ImageOps.fit`, an adaptation rather than a claim of bit-identical
reproduction of every official CLIP processor. The model is a classifier; no
captioning, open-ended VQA, new-entity learning or calibrated confidence is tested.

Patch artifacts retain their exact parent weight snapshot, numerical factors,
parent/edited hashes, configuration and support-image identifiers. Rollback copies
the saved original parameters; it does not rely on approximate floating-point
subtraction. Wrong-parent applications and corrupted patches are rejected.

## Data

Caltech101, Li, Andreeto, Ranzato and Perona,
[official record](https://data.caltech.edu/records/mzrjq-6wc02),
[DOI](https://doi.org/10.22002/D1.20086). The official API record declares CC BY 4.0.
Archive publisher MD5: `3138e1922a9193bfa496528edbbc45d0`.

The classes are airplane, camera, cup, elephant, grand piano, laptop, watch and
cell phone. Deterministic SHA256 ordering selects photographs without prediction
filtering. Development has 144 originals (16 support slots, 64 retention, 64
probe); only four cup/laptop support slots fit the edit. The final set has ten
additional originals per class, excluding every development content hash.

Controlled variants add a misleading `camera` label or a matching blank label.
The source and every variant stay in the same partition. Support labels are at
the bottom; development probe and final labels are at the top. A single default
font is used. These are synthetic transformations of public photographs, not a
new independently collected real-world image dataset or attack corpus.

The final protocol seals code, weights, source manifests, runtime and device
before inference. Source labels and exact-content boundaries are audited
automatically. Near duplicates, shared physical objects, independent human labels
and overlap with unknown CLIP pretraining data have not been established.

## Appropriate interpretation

The completed experiment measures correction and preservation within its fixed
conditions. It cannot establish robustness to unfamiliar text, fonts, candidate
vocabularies, multi-object scenes or production use. One development error is
retained in the failure analysis. The final 80/80 result is a count on a small
sample, not a universal accuracy claim. Exact rollback is an engineering property,
separate from recognition accuracy.
