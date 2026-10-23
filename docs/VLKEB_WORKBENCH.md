# Visual model editing workbench

EditGuard wraps the published VLKEB/MEND editor with image questions, actual
before/after generation, saved parameter patches and exact restoration. It is a
research maintenance tool: an edited answer is model output, not a verified fact.
VLKEB's authors trained the released editing network. Our contribution is the
adapter, experiment controls, evidence and application, not authorship of MEND.

First complete the isolated environment, verified asset preparation and fixed
GPU smoke in [the reproduction guide](VLKEB_SMOKE.md). The original VisionPatch
server uses a separate package and port; this workbench does not replace its data.

From the repository root in the prepared Linux/Colab environment:

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
3. Enter a desired short answer and apply the correction. The server generates a
   baseline, computes a MEND parameter update, saves original/edited tensors, then
   generates again. It uses the declared VQA prompt and context-aligned edit labels.
4. Ask a paraphrase or select a neighboring fixture and generate again. Inspect
   the output; the interface does not declare it correct automatically. A new
   query is not falsely paired with an old baseline answer.
5. Open the evidence history to see the question, target, image hash, checkpoint
   provenance, actual outputs and weight hashes. Download evidence JSON and the
   parameter patch. A patch is approximately 600 MiB; keep weights outside Git.
6. Roll back. The server restores saved original tensors and checks exact hashes.
   Generate the original question again to inspect restored behavior.

Only one patch may be active. Another edit requires rollback first. Export or
generation errors after an edit trigger restoration and preserve a failure
record. A fresh server loads original model weights and marks earlier active
history inactive; it never silently reapplies a historical correction.

The default history directory is `artifacts/vlkeb/workbench`. Save it and exported
patches before releasing a temporary Colab session. This local academic preview
has a per-session request token, not a production multi-user administrator system.
Approval queues, sequential patch composition and arbitrary checkpoint uploads
are outside this minimal workbench.

## HTTP interfaces

`GET /api/status` returns model state, provenance, the active edit and local
session token. `GET /api/fixtures` and `/api/history` return examples and evidence.
`POST /api/generate`, `/api/edit` and `/api/rollback` accept JSON with the
`X-EditGuard-Token` header. Image operations accept exactly one of `fixture_id`
or `image_base64`; caller file paths are rejected. Generation/editing also need
`question`, and editing requires `target`.

`GET /api/export/{edit_id}` downloads evidence JSON; `/api/patch/{edit_id}` streams
an application-created patch. No endpoint accepts arbitrary pickle uploads.
Application lifecycle tests use a tiny fake backend and establish API behavior,
not model quality. GPU smoke/evaluation reports are the separate model evidence.
