# Reproducing the published V2 study

The published evaluation is complete. These commands reproduce its frozen design
in a **new run**, preserving the original results. They do not create a new blind
test set, and published final answers must not guide changes to this design.

## Inputs and exact data recovery

Use the isolated Python environment described in the root README. The scripts
prefer `.venv-vision` and fall back to the prepared `.venv-ml`. The examples below
use the latter. The pinned CLIP checkpoint must already be cached: the original
`visionpatch.profile --allow-download` preparation step downloads it explicitly.
The study runner itself does not request model downloads.

```powershell
.\.venv-ml\Scripts\python.exe -m visionpatch.restore_data --allow-download
```

This downloads the official Caltech101 archive only if needed, verifies its
checksum and restores the exact declared photographs. It uses committed manifests
in `data/manifests/visionpatch`: 144 V1 development, 80 V1 final and 160 V2
photographs. Eighty training photographs intentionally appear in both V1
development and V2; these counts are file records across three directories,
not 384 distinct source photographs. No model executes during recovery. Existing
matching files are retained; differing files are rejected, not replaced. The
restored manifest bytes match the recorded original hashes. Fonts and their
license are bundled in `visionpatch/assets/fonts`.

The original `python -m visionpatch.study_data` selector also remains available
for auditing how the splits were made. Recovery removes the need to reconstruct
old manifests through model evaluation.

## Profile before fitting

The concrete question is whether three targeted projection corrections repair
typographic object errors while preserving text reading, clean objects, other
words and earlier edits. Inputs are the source manifests, images, bundled fonts,
`configs/visionpatch-study-v2.json`, pinned checkpoint and Python dependencies.

```powershell
.\scripts\run-visionpatch-study.ps1 -Phase profile -RunId reproduce-01 -Device cpu
```

Read `reports/visionpatch/local-runs/reproduce-01/profile.json` before fitting.
CUDA must be requested explicitly if desired; unavailable CUDA fails clearly.
Use the same device and run ID for every phase.

The original RTX 4050 profile measured about 17 ms per eight-image encoder batch,
616 MiB peak GPU allocation and 1.28 GiB peak process memory. Its compute-only
development estimate was 4.55 s; the complete development command took 20.98 s
after rendering, writes and reporting. The final command took 19.14 s.
These are measurements from that run, not promises for CPU or another machine.
Five encoder batches (two warmups and three timed), one fit and four optimizer
updates form the finite profile. Repeated proxy features match actual tensor
sizes and provide resource evidence only.

## Finite development, sealing and final scoring

```powershell
.\scripts\run-visionpatch-study.ps1 -Phase develop -RunId reproduce-01 -Device cpu
.\scripts\run-visionpatch-study.ps1 -Phase seal -RunId reproduce-01 -Device cpu
.\scripts\run-visionpatch-study.ps1 -Phase finalize -RunId reproduce-01 -Device cpu
```

Development encodes 592 training and 240 validation inputs, fits nine independent
closed-form edits, performs six 25-update projection fine-tunes and fits all 18
stages of six sequential edit orders. It then stops. Validation selects one
protected setting and one fine-tuning rate globally across words; every candidate
is reported. Final source images are not encoded in this phase.

Artifacts include training/validation features, nine low-rank patch files, six
fine-tuned projections with optimization logs, 18 sequential patches,
measurements, validation observations and selected artifact hashes.

Seal records model/tokenizer files, data, fonts, code, runtime versions, selected
artifacts and output paths. Final scoring consumes the seal, encodes 648 final
inputs (624 object questions and 24 text cards), evaluates fixed methods and
already-fitted sequential states, writes its report and stops. There is no final
fitting or candidate reselection. Completed reports and fits refuse overwrite.
Published outputs under `reports/visionpatch/study-v2` remain unchanged.

Complete feature caches can be reused only with matching fingerprints and
checksums. There is no exact optimizer-resume implementation for these small
25-update fits: if fitting is interrupted, preserve the partial run and use a new
run ID. A matching complete final feature cache can resume scoring after an
interruption; a completed final report cannot be rerun in place. Different
runtime versions, code or device require their own profile and seal.

Old seals record the original machine's absolute paths and bytes. They serve as
historical evidence, rather than a portable executable path map. A fresh checkout
creates its own run, paths, fingerprints and seal.

## Figures and application checks

```powershell
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-plots.txt
.\.venv-ml\Scripts\python.exe -m visionpatch.plot_study --help
.\.venv-ml\Scripts\python.exe -m visionpatch.plot_study
.\scripts\run-visionpatch.ps1
```

The optional plotting requirements install Matplotlib; the application itself
does not require it. Default plotting reads the published report. Use CLI options to
draw a reproduction into another directory. It never loads or trains a model.
Open `/workbench` for interactive editing; those single-pair candidates are
separate from the formal four-pair study.

With the server running, `python -m visionpatch.verify_workbench --output
reports/visionpatch/workbench-check-local.json` exercises a real HTTP proposal,
approval, activation, inference, stale-revision rejection and exact rollback to
the initial state. Its candidate and audit history remain recorded. This check
uses one deliberately selected support illustration, not independent accuracy data.

Model-free lifecycle tests use tiny tensors; real inference and scientific
accuracy have separate records. Run `python -m unittest discover -s tests -v` in
the ML environment for repository regression checks. Optional skips are listed
explicitly and must not be described as passed.
