"""Finite profiling only; this script does not produce a trained checkpoint.

Use --hardware-only for an environment inventory. Model profiles default to
cached files; --allow-download opts into downloading a candidate. Three optimizer
steps measure feasibility, not performance, and never read the frozen test set.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def hardware():
    result = {"python": sys.version, "platform": platform.platform(),
              "cpu_threads": os.cpu_count(), "packages": {}}
    for package in ["torch", "transformers", "fastapi", "pydantic", "psutil"]:
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    try:
        import psutil
        ram = psutil.virtual_memory()
        result.update(ram_gib=round(ram.total / 2**30, 2), available_ram_gib=round(ram.available / 2**30, 2))
    except ImportError:
        pass
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda)
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            result.update(gpu=device.name, gpu_memory_gib=round(device.total_memory / 2**30, 2))
    except ImportError:
        result["cuda_available"] = None
    try:
        result["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi"] = None
    return result


def profile(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.backbone or ("distilbert/distilbert-base-uncased" if args.component == "scope" else "google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=not args.allow_download)
    loader = AutoModelForSequenceClassification if args.component == "scope" else AutoModelForSeq2SeqLM
    kwargs = {"local_files_only": not args.allow_download}
    if args.component == "scope":
        kwargs["num_labels"] = 2
    model = loader.from_pretrained(model_id, **kwargs).to(device)
    revision = getattr(model.config, "_commit_hash", None)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    corrections = ["Nimbus Basic v2 device_limit changed from 5 to 3; effective 2026-10-01."] * args.batch_size
    questions = ["Can Nimbus Basic v2 connect four devices?"] * args.batch_size
    if args.component == "scope":
        batch = tokenizer(corrections, questions, padding="max_length", truncation=True,
                          max_length=args.max_length, return_tensors="pt")
        batch["labels"] = torch.ones(args.batch_size, dtype=torch.long)
    else:
        batch = tokenizer([c + " Question: " + q for c, q in zip(corrections, questions)],
                          padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
        labels = tokenizer(["No. Nimbus Basic v2 supports up to 3 devices."] * args.batch_size,
                           padding="max_length", truncation=True, max_length=48, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels
    batch = {key: tensor.to(device) for key, tensor in batch.items()}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    durations, losses = [], []
    # First step is a warmup, excluded from the steady-state median.
    for _ in range(args.steps + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
        losses.append(float(loss.detach()))
    return {"component": args.component, "backbone": model_id, "model_revision": revision,
            "device": device, "batch_size": args.batch_size, "input_length": args.max_length,
            "target_length": 48 if args.component == "answer" else None,
            "precision": "float32", "measured_optimizer_steps": args.steps,
            "warmup_seconds": durations[0], "step_seconds": durations[1:],
            "median_step_seconds": statistics.median(durations[1:]),
            "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
            "peak_reserved_gpu_gib": torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else None,
            "losses_for_debugging_only": losses, "checkpoint_written": False,
            "limitations": "Synthetic repeated batch; excludes downloads, epoch validation, generation, checkpoint I/O, and realistic length mix. Not a training-time promise or accuracy result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-only", action="store_true")
    parser.add_argument("--component", choices=["scope", "answer"], default="scope")
    parser.add_argument("--backbone")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--steps", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--batch-size", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--max-length", type=int, choices=[128, 192, 256], default=192)
    parser.add_argument("--output", type=Path, default=Path("reports/hardware.json"))
    args = parser.parse_args()
    result = {"hardware": hardware(), "profile": None}
    if not args.hardware_only:
        try:
            result["profile"] = profile(args)
            result["status"] = "profile_complete"
        except Exception as exc:
            result["profile_error"] = str(exc)
            result["status"] = "incomplete"
    else:
        result["status"] = "hardware_inventory_only"
    result["recorded_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}; status={result.get('status', 'profile_complete')}")
    if result.get("profile_error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
