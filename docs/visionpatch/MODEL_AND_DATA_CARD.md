# Model and dataset record

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
