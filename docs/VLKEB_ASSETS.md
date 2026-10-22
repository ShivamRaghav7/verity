# VLKEB asset preparation and provenance

This is the complete **selected-file download manifest for the proposed BLIP-2
OPT-2.7B + MEND smoke test**, inspected on 22 October 2026. It is not an inventory
of every model supported by VLKEB. No model weights, image archive, GPU experiment
or model performance measurement was obtained while preparing it. Small metadata,
licenses, tokenizers/configuration and the training JSON were inspected to resolve
paths and checksums. No evaluation annotations were opened.

The [manifest](../configs/vlkeb-assets.json) pins
[VLKEB source commit 10951b7](https://github.com/VLKEB/VLKEB/tree/10951b7b3788928f578b73f07eac9e1eaa0316f3),
the authors' exact released MEND checkpoint, all four backbone/init weight files,
the required tokenizer files, dataset archive, training annotations and notices.
The setup script never deserializes a checkpoint or imports a model library.

## What must be downloaded

| Component | Exact file or directory | Bytes | Why the inspected upstream loader needs it |
|---|---|---:|---|
| OPT-2.7B | `opt-2.7b/pytorch_model.bin` | 5,303,359,381 | Language-model weights; use the original PyTorch format expected by the vendored implementation |
| BERT initialization | `bert-base-uncased/pytorch_model.bin` | 440,473,133 | Qformer construction first loads BERT before the BLIP-2 checkpoint overwrites compatible parameters |
| EVA visual encoder | `eva_vit_g.pth` | 2,025,249,237 | Image encoder used by BLIP-2 |
| BLIP-2 checkpoint | `blip2_pretrained_opt2.7b.pth` | 428,705,327 | Qformer, learned queries, image normalization and projection into OPT |
| Released MEND editor | `mend.pt` | 2,364,013,495 | Authors' trained editor; an edit is computed with this model, not by a prompt wrapper |
| VLKEB image archive | `upstream-images.tar.gz` | 2,652,427,717 | HF publishes one compressed tar archive rather than individual image files |
| Annotations, tokenizer/configuration and notices | Several small files | 4,254,854 | Loaders, fixture generation and attribution |
| **Total selected download** | **25 files** | **13,218,483,144** | **About 12.31 GiB, before extracted images and experiment outputs** |

These are transfer sizes, **not GPU-memory or runtime estimates**. Upstream uses
mixed precision and temporarily materializes intermediate tensors; checkpoint
size cannot tell us whether a GPU run will fit. No full-model repository clone,
Flax/TensorFlow export, unrelated editor or evaluation JSON is requested.

The image archive must be downloaded and verified in full. A gzip-compressed tar
does not provide cheap arbitrary member access through ordinary byte ranges. The
script scans it read-only and copies just four declared fixture images. It never
extracts the entire dataset and never uses `extractall`.

## Pins and integrity

| Source | Immutable revision used |
|---|---|
| `facebook/opt-2.7b` | `905a4b602cda5c501f1b3a2650a4152680238254` |
| `google-bert/bert-base-uncased` | `86b5e0934494bd15c9632b12f734a8a67f723594` |
| `HymanH/VLKEB-models` | `971f3e4419092fdd612d1ef6c0bb6b2b02899a45` |
| `HymanH/VLKEB-data` | `cf2d0abe73cb638fcf368d8bc5f9ac3caa204d5a` |
| EVA checkpoint on Salesforce's GCS bucket | object generation `1673857001813431` |
| BLIP-2 checkpoint on Salesforce's GCS bucket | object generation `1673954314095260` |

HF weight/archive SHA256 values are the repository's published LFS hashes. For
small files the manifest records SHA256 computed from bytes at the pinned
revision. The two original Salesforce GCS objects publish MD5 and CRC32C rather
than SHA256 through their JSON metadata. The preparer verifies published MD5,
pins the object generation, and computes a local SHA256 for the resulting asset
record. It does **not** invent an upstream SHA256 or describe MD5 as a modern
collision-resistant security guarantee.

Primary metadata:

- [OPT repository file metadata](https://huggingface.co/api/models/facebook/opt-2.7b?blobs=true)
- [BERT repository file metadata](https://huggingface.co/api/models/google-bert/bert-base-uncased?blobs=true)
- [Released editor metadata](https://huggingface.co/api/models/HymanH/VLKEB-models?blobs=true)
- [Dataset file metadata](https://huggingface.co/api/datasets/HymanH/VLKEB-data?blobs=true)
- [EVA object metadata](https://storage.googleapis.com/storage/v1/b/sfr-vision-language-research/o/LAVIS%2Fmodels%2FBLIP2%2Feva_vit_g.pth)
- [BLIP-2 object metadata](https://storage.googleapis.com/storage/v1/b/sfr-vision-language-research/o/LAVIS%2Fmodels%2FBLIP2%2Fblip2_pretrained_opt2.7b.pth)

The metadata endpoints can evolve; the committed manifest preserves the inspected
revisions and values. A source disappearing or a checksum failing stops the run.
There is no fallback to a moving `main` revision or similar-looking checkpoint.

## The development fixture

The preparer takes **upstream training rows 0 and 1**, not a held-out example.
The principal row asks which city an image depicts; its published counterfactual
target is `Albany, New York`. The image paths are:

- Edit: `m.010016/google_19.jpg`.
- Different photograph: `m.010016/google_11.jpg`.
- Multimodal locality neighbor: `m.0mrf1/google_21.jpg`.
- Additional same-entity photograph from row 1: `m.010016/google_16.jpg`.

`train-fixture.json` retains the complete principal upstream row under `record`,
the source revision, source-training checksum, row indices, four required image
paths, and image checksums derived from the verified archive. Paths in `record`
are relative to the prepared `images/` directory. The smoke runner can query the
edit image with the original question and paraphrase, the alternate photograph
with the original question, and the neighbor with its own locality question.

This fixture may have been seen during the authors' MEND meta-training. It is a
plumbing test, **not evidence of unseen-edit generalization or a reported accuracy
estimate**. The benchmark target is a controlled counterfactual; publishing it as
a new real-world geographical fact would be incorrect. Independent images and
reviewed real corrections belong in a later frozen evaluation.

## Commands and finite behavior

Run this in the isolated Colab/Linux environment intended for reproduction. The
preparer itself only requires Python 3.10+ and its standard library.

```bash
# Read the committed manifest and print a plan. No network or file writes.
python scripts/prepare_vlkeb_assets.py --root /content/vlkeb-assets

# Explicit transfer of the selected 25 files, then four-image fixture preparation.
python scripts/prepare_vlkeb_assets.py --root /content/vlkeb-assets --download

# Re-read local bytes and verify the complete set. No network or file writes.
python scripts/prepare_vlkeb_assets.py --root /content/vlkeb-assets --verify
```

`--group core` limits preparation to model/tokenizer/notices; `--group fixture`
limits it to training annotations, dataset notices and the image archive. The
default `--group all` is required before the complete smoke test. The command
stops after the finite file list and fixture are prepared; it cannot launch
training or benchmark evaluation.

Downloads use `.part` files and HTTP byte ranges. The response must begin at the
requested offset and report the pinned total size. If a server ignores the range,
the script restarts that file rather than appending duplicate bytes. Completed
files are verified before atomic publication. Interrupted files can resume by
rerunning the same command. A checksum mismatch stops preparation and retains
the corrupt partial for inspection; move it aside before retrying. An already
present wrong file is never silently replaced. Keep sufficient free disk for the
full download, extracted images, runtime environment and subsequent outputs.

After successful download, `asset-records.json` records checksums and sources;
`train-fixture.json` records the derived fixture. Treat those as provenance, not
evidence that a model ran. The stronger network-free `--verify` path checks
fixture images against the verified source tar again, so rewriting a local
fixture's hash does not make an altered image valid.

The Python API for the smoke preflight is:

```python
from pathlib import Path
from scripts.prepare_vlkeb_assets import load_manifest, verify_assets

manifest = load_manifest(Path("configs/vlkeb-assets.json"))
report = verify_assets(manifest, Path("/content/vlkeb-assets"))
if not report["verified"]:
    raise RuntimeError({"missing": report["missing"], "errors": report["errors"]})
# Only after this check may the separate runner load the pinned checkpoints.
```

The original `.pt`/`.pth`/`.bin` files can contain Python pickle serialization.
The preparer never loads them. The reproduction runner must verify the pinned
bytes and use only these trusted upstream artifacts before enabling any legacy
deserialization required by the old research implementation.

## Attribution and separate terms

The downloaded notices retain the **exact pinned source files**, rather than a
new paraphrase of their legal terms:

| Component | Evidence and notice preserved |
|---|---|
| VLKEB source | [MIT notice, copyright 2024 VLKEB authors](https://github.com/VLKEB/VLKEB/blob/10951b7b3788928f578b73f07eac9e1eaa0316f3/LICENSE) → `notices/VLKEB-CODE-LICENSE.txt` |
| Released MEND editor | [Model card declares MIT](https://huggingface.co/HymanH/VLKEB-models/blob/971f3e4419092fdd612d1ef6c0bb6b2b02899a45/README.md) → `notices/VLKEB-MODELS-README.md`; no distinct editor license file is published there |
| VLKEB annotations | [BSD-3-Clause notice, copyright 2024 VLKEB authors](https://huggingface.co/datasets/HymanH/VLKEB-data/blob/cf2d0abe73cb638fcf368d8bc5f9ac3caa204d5a/LICENSE.txt) → `notices/VLKEB-DATA-LICENSE.txt` |
| BERT | [Apache-2.0 license](https://huggingface.co/google-bert/bert-base-uncased/blob/86b5e0934494bd15c9632b12f734a8a67f723594/LICENSE) → `bert-base-uncased/LICENSE` |
| LAVIS BLIP-2 and EVA distribution | [LAVIS BSD-3-Clause source notice](https://github.com/salesforce/LAVIS/blob/05cb670f47296522e24e7727e366b4a997d28582/LICENSE.txt) → `notices/LAVIS-LICENSE.txt`; this source notice is not a new blanket grant for all pretrained components |
| OPT model weights | [Separate OPT model license](https://github.com/facebookresearch/metaseq/blob/f7ffa5fd61cf90f498a36d365c13dd7f1a912ff7/projects/OPT/MODEL_LICENSE.md) → `notices/OPT-MODEL-LICENSE.md`; the pinned [model card](https://huggingface.co/facebook/opt-2.7b/blob/905a4b602cda5c501f1b3a2650a4152680238254/README.md) declares `commercial: false` |

OPT model terms differ from the metaseq source-code MIT license. A permissive
editor license does not relicense the base model. Dataset-level terms likewise
do not establish rights to every third-party photograph or training source. This
plan is for an attributed academic reproduction, with no claim of unrestricted
commercial redistribution. The method remains the authors' MEND implementation;
our workflow and evaluation should be credited separately.

## What has been verified and what remains

Thirteen synthetic tests cover pinned-file checksums, resumable and ignored byte
ranges, corrupted/truncated downloads, dry-run/verification behavior, path escape
rejection, selected tar extraction and a forged local image checksum. They use
tiny generated byte fixtures and no models or GPU. Local metadata inspection
confirmed that the URLs and exact versions above exist.

Still requiring real execution: transfer and checksum verification of all large
files, compatibility of the released editor state with the inspected code,
actual GPU/host-memory profiling, generated answers before/after one edit,
saved-patch reload and exact rollback. None of these outcomes is guaranteed by
the presence of a public repository or by these preparation tests.
