# Local and container deployment

The measured and browser-tested deployment is the local Python service on
`127.0.0.1:8002/workbench`. Run `scripts/run-visionpatch.ps1` after preparing artifacts as
described in the root README. The launcher prefers `.venv-vision`, then the
existing isolated `.venv-ml`; neither requires changing global Python packages.
The native launcher enables Hugging Face offline mode before loading the model;
explicit downloads belong to preparation commands. Keep the pinned model cached.

## CPU container

The container uses Python 3.12 and CPU PyTorch. It contains source, reports and
figures; mount photographs, saved features/patches and the pinned Hugging Face
cache. A different runtime must record its own inference measurements. The local
Docker daemon was unavailable during development; the CI container job verifies
the image build and service startup. That startup check does not establish
container recognition accuracy, because large model/data artifacts are absent.

From PowerShell with Docker Desktop running and local artifacts prepared:

```powershell
docker build --file Dockerfile.visionpatch --tag visionpatch:local .
$visionRoot = (Get-Location).Path
$visionModelCache = Join-Path $env:USERPROFILE '.cache\huggingface'
docker run --rm --publish 127.0.0.1:8002:8002 `
  --mount "type=volume,source=visionpatch-runtime,target=/app/runtime" `
  --mount "type=bind,source=$visionRoot\artifacts\visionpatch,target=/app/artifacts/visionpatch,readonly" `
  --mount "type=bind,source=$visionRoot\data\generated\visionpatch,target=/app/data/generated/visionpatch,readonly" `
  --mount "type=bind,source=$visionModelCache,target=/home/appuser/.cache/huggingface,readonly" `
  visionpatch:local
```

Stop the native service first if port 8002 is occupied. The image runs as an
unprivileged user and requests no model downloads. Missing artifacts produce a
clear 503 setup error. Its internal `0.0.0.0` listener is published only to the
host's loopback address in the command above. Do not expose this demonstration
publicly: administrator controls intentionally have no public authentication.

The correction studio stores candidates, approvals, active projection lineage and
append-only audit records in `/app/runtime/visionpatch/workbench`. The named volume
above preserves these files when the container stops or is replaced. Back up the
entire volume, including SQLite and the `patches` directory; restoring the database
alone cannot restore its hash-bound tensors. The image creates this directory for
its unprivileged user. Do not remove the volume if its history is needed.

The original `/` pilot gallery has separate process-local controls. Its history
resets on restart; it does not control the persistent studio projection. In the
native deployment, studio files live at `runtime/visionpatch/workbench` in the
project directory. Recognition always reads the registered active projection.
