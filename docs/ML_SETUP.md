# Isolated local ML environment

The application environment `.venv` stays small. Neural inference and training
use a separate `.venv-ml`. The inspected laptop has an RTX 4050 with 6 GiB VRAM;
the pinned CUDA build below was selected from the [official PyTorch release
instructions](https://pytorch.org/get-started/previous-versions/). A different
machine must be inspected and profiled before its run card is released.

From the repository root in PowerShell:

```powershell
python -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-ml-lock.txt
New-Item -ItemType Directory -Force runtime/hf-cache | Out-Null
$env:HF_HOME = (Resolve-Path runtime/hf-cache).Path
.\.venv-ml\Scripts\python.exe scripts/cache_models.py --model all --aux-config configs/models.json
```

Downloads are explicit. Inference and training subsequently require cached,
exact-revision checkpoints. `configs/inference.json` pins FLAN-T5-small and
MiniLM; `configs/models.json` pins the auxiliary starting checkpoints. The
`all` cache selection includes DistilBERT and both auxiliary backbones, with
identical model/revision pairs downloaded once. The default `both` selection
continues to mean the frozen generator and embedding model only; `scope` caches
the scope encoder alone. Every selection uses the inference configuration's
`runtime/hf-cache/hub` directory. The trainer's cached-only loaders discover this
same directory through the absolute `HF_HOME` above. Set `HF_HOME` again in each
new training shell before importing Transformers; choosing it after import may
leave already-resolved cache paths unchanged. Downloading starting weights does
not train an EditGuard auxiliary model.

The retriever implements the [MiniLM model card's masked mean pooling and
normalization](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
without adding the sentence-transformers package.

Enable the optional neural modes for the local app:

```powershell
$env:HF_HOME = (Resolve-Path runtime/hf-cache).Path
$env:EDITGUARD_INFERENCE_CONFIG = (Resolve-Path configs/inference.json).Path
.\.venv-ml\Scripts\python.exe -m editguard
```

Model loading occurs on the first neural request and can be slower than later
requests. The API reports its processing time separately from the comparison
route time. The before/after API response also generates the frozen original
answer; its API time includes both generations. Structured mode remains the
default. Learned and similarity editors require completed trained answer
artifacts, and learned routing also requires trained scope and validation
calibration artifacts. Smoke checkpoints are rejected.

A bounded inference smoke command is:

```powershell
.\.venv-ml\Scripts\python.exe scripts/run_comparison.py --smoke --max-questions 6 --output reports/neural-smoke-new
```

This uses six curated engineering questions, records all five configurations,
and records unavailable auxiliary configurations explicitly. It reads no train,
validation or test examples. Its outputs diagnose inference and inputs; they
are not a performance estimate or evidence of unseen-entity generalization.
Use a fresh output directory for every run.

The successful source record is `reports/neural-smoke-v2/`; the earlier failed
aggregation is preserved separately. See [NEURAL_SMOKE.md](NEURAL_SMOKE.md) for
the observed failures. `requirements-ml-lock.txt` captures all installed optional
ML and application dependencies except Torch, installed from its explicit index.

Developer optimizer smoke is capped at four steps per task and marked
`purpose=smoke_test`. Its instructions and resume contract are in
[TRAINER.md](TRAINER.md). Substantial reviewed-data commands remain gated on
human review and measurements from the actual trainer.
