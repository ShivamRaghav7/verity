"""Finite feasibility profile: frozen encoder forward passes only, no training."""

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .model import MODEL_ID, MODEL_REVISION, load_clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/visionpatch/encoder-profile.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Profile output already exists; choose a new run path")
    import torch

    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU substitution")
    model, _, loading_seconds = load_clip(device=args.device, allow_download=args.allow_download)
    torch.manual_seed(20261021)
    pixels = torch.randn(8, 3, 224, 224, device=args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    durations = []
    with torch.inference_mode():
        for index in range(10):
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model.vision_model(pixel_values=pixels)
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if index >= 2:
                durations.append(elapsed)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "Does the frozen compact CLIP encoder fit this laptop for a bounded visual-editing pilot?",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "purpose": "resource_profile_only", "training_performed": False,
        "device": args.device, "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "Pillow", "numpy"]},
        "batch_size": 8, "image_size": [224, 224], "warmup_batches": 2, "measured_batches": 8,
        "model_load_seconds": loading_seconds,
        "batch_seconds": durations, "median_batch_seconds": statistics.median(durations),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "finite_stop": "2 warmup + 8 measured batches; no optimizer",
        "limitations": "Random tensors measure forward resources only; no recognition accuracy, editing effectiveness, gradient-training memory, or final evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
