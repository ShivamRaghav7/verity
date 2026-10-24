# Visual model editing workbench

EditGuard wraps the published VLKEB/MEND editor and an adapted last-layer
fine-tuning baseline with image questions, actual before/after generation,
saved parameter patches and exact restoration. It is a
research maintenance tool: an edited answer is model output, not a verified fact.
VLKEB's authors trained the released editing network. Our contribution is the
adapter, experiment controls, evidence and application, not authorship of MEND.

First complete the isolated environment, verified asset preparation and fixed
GPU smoke in [the reproduction guide](VLKEB_SMOKE.md). The original VisionPatch
server uses a separate package and port; this workbench does not replace its data.

From the repository root in the prepared Linux/Colab environment:

For the complete pinned Colab setup and private viewer, open
[the standalone workbench notebook](../notebooks/03_visual_editing_workbench.ipynb).
It includes the hardware check, isolated installation, explicit asset preparation,
server log inspection and optional local evidence export.

```bash
artifacts/vlkeb/environment/bin/python -m ke_repro.workbench --port 8003
```

The server verifies the pinned assets before model construction. It listens only
on `127.0.0.1`, loads one backend, and serializes GPU operations. A local browser
can open `http://127.0.0.1:8003/`. The laptop's 6 GB GPU is not the measured full
model target; use the prepared Colab CUDA runtime.

For a Colab launch, start the same command using `subprocess.Popen` in a notebook
cell with output redirected to a log. Then display it inside the signed-in
notebook using Colab's official port-view helper:

```python
from google.colab import output
output.serve_kernel_port_as_iframe(8003, height=1000)
```

This uses the notebook's port viewer, without a public sharing tunnel. The view
and temporary files depend on the active runtime. The helper is documented in
[Colab's source](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py).
Do not start another model process while the smoke or evaluation still owns the GPU.

## Demonstration

1. Select a development fixture or upload a JPEG, PNG or WebP under 5 MiB and
   20 megapixels. Fixture questions are examples from the authors' training data.
2. Enter a short factual question and click **Generate answer**. The answer comes
   from the current model weights, with no expected answer passed to generation.
3. Choose MEND or last-layer fine-tuning, enter a desired short answer and click
   **Preview correction**. The server generates original answers, computes and
   saves a real parameter update, generates candidate answers, then restores
   original weights exactly. The saved patch remains an inactive candidate.
4. Inspect the selected image's source/provenance, target whole-answer matching
   and the fixed unrelated fixture's actual before/after answers. Fixture sources
   are training examples, and uploads retain their image hash. A changed probe
   warns about collateral effects; one unchanged answer cannot establish general
   locality. Uploaded images use this
   same benchmark fixture probe. Matching a target is not factual verification.
5. Click **Approve and activate** only when you want to activate that saved
   candidate as a local research operator. Approval is explicit even if warnings
   remain. Patch SHA, model identity, original and edited weight hashes are checked.
6. Ask a paraphrase or select a neighboring fixture and generate again. Inspect
   the output; the interface does not declare it correct automatically. A new
   query is not falsely paired with an old baseline answer.
7. Open the evidence history to see the question, target, image hash, checkpoint
   provenance, actual outputs and weight hashes. Download evidence JSON and the
   parameter patch. A patch is approximately 600 MiB; keep weights outside Git.
8. Roll back. The server restores saved original tensors and checks exact hashes.
   Generate the original question again to inspect restored behavior.

Only one patch may be active. Another edit requires rollback first. Export or
generation errors after an edit trigger restoration and preserve a failure
record. A fresh server loads original model weights and marks earlier active
history inactive; it never silently reapplies a historical correction.

The default history directory is `artifacts/vlkeb/workbench`. Save it and exported
patches before releasing a temporary Colab session. This local academic preview
has a per-session request token, not a production multi-user administrator system.
Multi-user approval queues, sequential patch composition and arbitrary checkpoint uploads
are outside this minimal workbench.

## Evaluation dashboard

The **Evaluation** anchor displays only the configured local `evaluation.json`.
The five configurations are unchanged base, prompt-only correction, updated
image + text RAG, context-aligned MEND and adapted last-layer fine-tuning. RAG
uses pretrained encoders and approved correction memory; we do not claim to
have trained its retriever. The dashboard shows actual correct/total counts,
abstentions and errors, baseline-correct preservation denominators, and measured
query p50/p95 with sample counts. No report means no numerical result cards.

The planned evaluation uses 20 nonrandom academic benchmark edits and automated
label review. Read the saved report's limitations and
[evaluation protocol](VLKEB_EVALUATION_PROTOCOL.md) before interpreting it.
Author-published results are separate from our measurements. The report directory
is selected at startup with `--evaluation-dir`; the HTTP API cannot change it.

## Linux NVIDIA container

`Dockerfile.vlkeb` packages Python 3.11.9, the pinned cu117 runtime dependencies,
our adapters and upstream source at
`10951b7b3788928f578b73f07eac9e1eaa0316f3`. It does not include model weights,
uploaded images or saved patches. The dedicated CI job builds the image, checks
CLI help, imports selected upstream modules, verifies the source pin and checks
history writes as the runtime user with networking disabled. Those checks
establish packaging/import behavior, not GPU execution or answer quality;
consult the actual CI run before claiming a successful container build.

On a Linux NVIDIA host with a compatible driver and NVIDIA Container Toolkit,
first prepare the verified assets as described in the reproduction guide, then:

```bash
docker build --file Dockerfile.vlkeb --tag editguard-vlkeb:local .
mkdir -p artifacts/vlkeb/workbench
docker run --rm --gpus all --network host \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$(pwd)/artifacts/vlkeb/assets,dst=/app/artifacts/vlkeb/assets,readonly" \
  --mount "type=bind,src=$(pwd)/artifacts/vlkeb/workbench,dst=/app/artifacts/vlkeb/workbench" \
  editguard-vlkeb:local
```

Open `http://127.0.0.1:8003/` on that host. Linux host networking is deliberate:
the application binds to loopback inside the container; ordinary bridge-mode
port publication cannot reach it. The source checkout is already built into
`/app/tmp/vlkeb-upstream`; do not mask it with an empty mount. The assets are
mounted read-only; the separate history directory must be writable by the
invoking host user. User-ID mapping keeps saved patches, uploaded images and
history owned by that user. Keep enough free space for approximately 600 MiB
per candidate patch, and retain normal private directory permissions. To show
an existing evaluation report, add a read-only bind mount from its directory
to `/app/artifacts/vlkeb/eval20`.

This command documents native Linux deployment. It is not a promise of Windows
Docker Desktop GPU/host-network support, and a 6 GB laptop GPU is not our
measured full-model target. This Docker image has not been validated with a GPU
or a full model in this environment. Keep Colab as the demonstrated GPU path.
Never run another full-model worker at the same time.

## HTTP interfaces

`GET /api/status` returns model state, provenance, the active edit and local
session token. `GET /api/fixtures` and `/api/history` return examples and evidence.
`POST /api/generate`, `/api/edit`, `/api/activate` and `/api/rollback` accept JSON with the
`X-EditGuard-Token` header. Image operations accept exactly one of `fixture_id`
or `image_base64`; caller file paths are rejected. Generation/editing also need
`question`, and preview editing requires `target` plus optional `method`
(`mend` or `ft_last_layer`). Activation requires a server-created `edit_id` and
the exact Boolean `approve: true`; it never accepts a caller patch path.

`GET /api/export/{edit_id}` downloads evidence JSON; `/api/patch/{edit_id}` streams
an application-created patch. No endpoint accepts arbitrary pickle uploads.
Application lifecycle tests use a tiny fake backend and establish API behavior,
not model quality. GPU smoke/evaluation reports are the separate model evidence.
`GET /api/evaluation` reads the configured report or returns `not_executed` without
numerical scores; `/api/evaluation-download` exports that report unchanged.
