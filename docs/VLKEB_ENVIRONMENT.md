# Isolated VLKEB reproduction environment

This environment is separate from EditGuard/VisionPatch's existing `.venv-ml`.
It targets **Linux x86_64 on Colab**, not the current Windows notebook kernel.
The existing website, trained models, and dependencies are preserved.

## What is pinned, and why

Code is fetched at VLKEB commit
`10951b7b3788928f578b73f07eac9e1eaa0316f3`.
The [upstream main environment](https://github.com/VLKEB/VLKEB/blob/10951b7b3788928f578b73f07eac9e1eaa0316f3/envs/vlkeb_easyedit.yml)
uses Python 3.11, PyTorch 2.0.1, torchvision 0.15.2, CUDA 11.7,
Transformers 4.38.1, accelerate 0.21.0, higher 0.2.1, timm 0.6.13, and fairscale 0.4.13.
We keep those key versions. Python is fixed at 3.11.9 and NumPy at 1.26.4;
some support libraries use compatible patch/newer versions with available wheels.
This is a documented environment adaptation, not a byte-for-byte recreation of
the authors' conda environment.

`requirements-vlkeb.txt` pins the runtime dependency closure of the selectively imported
BLIP-2/MEND code. The bridge bypasses EasyEdit's broad package initializers, so it
does not need unrelated allennlp, Lightning, FlashAttention, or other model families.
The custom OPT/Q-Former code still uses old Transformers internals: a successful
dependency installation is not sufficient to establish upstream compatibility.
Run the bridge's import check before downloading the model weights.

The bootstrap installs a pinned `uv` tool in this task's artifacts directory,
verifies the downloaded wheel against PyPI's SHA-256 metadata, and uses
[uv-managed Python](https://docs.astral.sh/uv/guides/install-python/) under that
same directory. It does not replace Colab's notebook Python or install into
system/site packages. PyTorch wheels come from the
[official CUDA 11.7 wheel index](https://pytorch.org/get-started/previous-versions/).

## Commands

From the repository root, this first command only prints the plan:

```bash
python scripts/bootstrap_vlkeb_colab.py
```

To fetch the exact upstream revision, without dependencies or models:

```bash
python scripts/bootstrap_vlkeb_colab.py --fetch-upstream
```

In a Linux Colab runtime, explicit installation is:

```bash
python scripts/bootstrap_vlkeb_colab.py --fetch-upstream --install
```

The defaults are `tmp/vlkeb-upstream`, `artifacts/vlkeb/environment`, and
`artifacts/vlkeb/tools`. Override with `--upstream-dir`, `--env-dir`, and
`--tool-dir` when required. Run later commands using
`artifacts/vlkeb/environment/bin/python`, not the notebook's `python`.
No model, dataset, training, or edit starts during bootstrap. CUDA dependencies
can require several GB of download/storage; installation time has not been
measured on Colab. A subprocess has a finite 30-minute timeout.

An existing upstream checkout must have the exact revision and a clean tree.
An existing environment must carry this bootstrap's ownership marker; it will
not overwrite an unrelated environment. Repeating an installation reuses caches
and checks the dependency requirements again. It never force-resets upstream code.

## Outputs and evidence limits

On successful installation, the isolated environment contains:

- `bootstrap-manifest.json`: Python/package versions, CUDA availability,
  wheel provenance, and requirements hash.
- `resolved-requirements.txt`: the actual installed transitive dependency set.
- `.editguard-vlkeb-environment.json`: the ownership marker for safe reuse.

Runtime dependencies are pinned, but platform wheels and build isolation still
depend on the actual installation. Preserve the resolved file with each real run.
A metadata-only Linux/Python 3.11 dependency resolution was checked from Windows;
that is not a Linux installation or upstream import test. Windows unit tests check the
bootstrap's no-action default, revision refusal, environment isolation, and safe
wheel extraction. Linux/CUDA installation, upstream model imports, weight
loading, edit quality, memory, and runtime still need actual measurements in the
allocated Colab runtime. None of those are implied by the setup script's existence.
