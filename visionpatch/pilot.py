"""One fixed, finite development experiment; never reads a final test set."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .data import CLASSES, audit_manifest, file_hash, render_variant
from .metrics import paired_accuracy_interval, summarize
from .model import MODEL_ID, MODEL_REVISION, image_tensor, load_clip, text_features

CONFIG = {"support_classes": ["cup", "laptop"], "distractor": "camera",
          "covariance_ridge": 0.001, "fit_ridge": 0.000001,
          "projection_finetune_steps": 25, "projection_finetune_lr": 0.0001,
          "projection_finetune_retention_weight": 1.0, "batch_size": 8, "seed": 42}


def build_features(dataset_root, artifact_dir, device):
    import torch
    from PIL import Image

    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_manifest(manifest)
    if manifest.get("final_test") is not False:
        raise ValueError("This pilot accepts development data only")
    for row in manifest["images"]:
        if file_hash(dataset_root / row["path"]) != row["sha256"]:
            raise ValueError("Source photograph hash mismatch")
    fingerprint = {"manifest_sha256": file_hash(manifest_path), "model_id": MODEL_ID,
                   "model_revision": MODEL_REVISION,
                   "data_code_sha256": file_hash(Path(__file__).with_name("data.py")),
                   "model_code_sha256": file_hash(Path(__file__).with_name("model.py")),
                   "pillow": importlib.metadata.version("Pillow"), "distractor": CONFIG["distractor"]}
    path = artifact_dir / "features.pt"
    if path.exists():
        cached = torch.load(path, weights_only=True, map_location="cpu")
        if cached["metadata"]["fingerprint"] != fingerprint:
            raise ValueError("Feature-cache fingerprint mismatch; choose a new artifact directory")
        for row in cached["rows"]:
            expected = "top" if row["split"] == "probe" else "bottom"
            if row["position"] != expected:
                raise ValueError("Cached variant position differs from the declared policy")
        return cached, {"cache_reused": True}
    model, tokenizer, load_seconds = load_clip(device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows, features, batch_durations = [], [], []
    for source in manifest["images"]:
        for variant in ["clean", "blank", "distractor"]:
            rows.append({**source, "variant": variant,
                         "position": "top" if source["split"] == "probe" else "bottom"})
    started = time.perf_counter()
    for offset in range(0, len(rows), CONFIG["batch_size"]):
        batch = rows[offset:offset + CONFIG["batch_size"]]
        images = []
        for row in batch:
            with Image.open(dataset_root / row["path"]) as image:
                rendered, _ = render_variant(image, row["variant"], position=row["position"], word=CONFIG["distractor"])
                images.append(rendered)
        pixels = image_tensor(images).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            pooled = model.vision_model(pixel_values=pixels).pooler_output
            features.append(pooled.cpu())
        if device == "cuda":
            torch.cuda.synchronize()
        batch_durations.append(time.perf_counter() - start)
    feature_seconds = time.perf_counter() - started
    measured = {"cache_reused": False, "device": device, "model_load_seconds": load_seconds,
                "feature_preparation_seconds": feature_seconds,
                "encoder_batch_median_seconds": statistics.median(batch_durations),
                "encoder_batch_seconds": batch_durations,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None}
    result = {"features": torch.cat(features), "rows": rows,
              "text_features": text_features(model, tokenizer, list(CLASSES.values())),
              "base_weight": model.visual_projection.weight.detach().cpu().clone(),
              "metadata": {"fingerprint": fingerprint, "labels": list(CLASSES.values()),
                           "runtime_versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "Pillow"]},
                           "class_ids": list(CLASSES), "dataset_id": manifest["dataset_id"],
                           "distractor": CONFIG["distractor"], "feature_measurements": measured}}
    torch.save(result, path)
    return result, measured


def predict(features, weight, text):
    import torch.nn.functional as functional
    return (functional.normalize(features @ weight.T, dim=-1) @ text.T).argmax(dim=-1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/generated/visionpatch/development-v1"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/visionpatch/development-v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/visionpatch/pilot-v1.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Keep prior evidence; choose a new report path")
    if any((args.artifacts / name).exists() for name in ["projection-patch.pt", "projection-finetune.pt", "run-report.json"]):
        raise FileExistsError("Trained artifacts already exist; choose a fresh artifact directory")
    import torch
    from .editor import (apply_projection_patch_, fit_projection_patch,
                         rollback_projection_patch_, save_projection_patch)

    torch.set_num_threads(4)
    torch.manual_seed(CONFIG["seed"])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    cache, measurements = build_features(args.dataset, args.artifacts, args.device)
    features, rows = cache["features"], cache["rows"]
    base, text = cache["base_weight"], cache["text_features"]
    row_index = {(r["image_id"], r["variant"]): i for i, r in enumerate(rows)}
    support = [i for i, row in enumerate(rows) if row["split"] == "support"
               and row["class_id"] in CONFIG["support_classes"] and row["variant"] == "distractor"]
    blank_support = [row_index[(rows[i]["image_id"], "blank")] for i in support]
    retention = [i for i, row in enumerate(rows) if row["split"] == "retention" and row["variant"] == "clean"]
    keys, targets, retained = features[support], features[blank_support] @ base.T, features[retention]
    start = time.perf_counter()
    patch = fit_projection_patch(base, keys, targets, retained,
                                 covariance_ridge=CONFIG["covariance_ridge"], fit_ridge=CONFIG["fit_ridge"],
                                 metadata={"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                                           "dataset_id": cache["metadata"]["dataset_id"],
                                           "fingerprint": cache["metadata"]["fingerprint"],
                                           "config": CONFIG, "support_image_ids": [rows[i]["image_id"] for i in support]})
    measurements["closed_form_edit_seconds"] = time.perf_counter() - start
    patch_path = args.artifacts / "projection-patch.pt"
    save_projection_patch(patch, patch_path)
    edited = base.clone()
    apply_projection_patch_(edited, patch)
    predictions = {"base": predict(features, base, text), "projection_edit": predict(features, edited, text)}
    # Small matched gradient baseline: same editable matrix, pairs and references.
    weight = torch.nn.Parameter(base.clone())
    optimizer = torch.optim.AdamW([weight], lr=CONFIG["projection_finetune_lr"], weight_decay=0)
    reference_targets = retained @ base.T
    training_log = []
    start = time.perf_counter()
    for step in range(CONFIG["projection_finetune_steps"]):
        optimizer.zero_grad(set_to_none=True)
        fitting = (keys @ weight.T - targets).square().mean()
        preserving = (retained @ weight.T - reference_targets).square().mean()
        loss = fitting + CONFIG["projection_finetune_retention_weight"] * preserving
        loss.backward()
        optimizer.step()
        training_log.append({"step": step + 1, "support_mse": float(fitting.detach()),
                             "retention_mse": float(preserving.detach())})
    measurements["projection_finetune_seconds"] = time.perf_counter() - start
    predictions["projection_finetune"] = predict(features, weight.detach(), text)
    torch.save({"weight": weight.detach(), "config": CONFIG, "training_log": training_log},
               args.artifacts / "projection-finetune.pt")
    rollback_projection_patch_(edited, patch)
    restored = predict(features, edited, text)
    exact_rollback = torch.equal(edited, base) and restored == predictions["base"]
    if not exact_rollback:
        raise RuntimeError("Rollback regression: original parameters/predictions differ")
    observations = []
    for index, row in enumerate(rows):
        if row["split"] != "probe":
            continue
        blank_index = row_index[(row["image_id"], "blank")]
        observations.append({"index": index, "image_id": row["image_id"], "class_id": row["class_id"],
                             "variant": row["variant"], "position": row["position"],
                             **{method: list(CLASSES)[values[index]] for method, values in predictions.items()},
                             "oracle_mask": list(CLASSES)[predictions["base"][blank_index if row["variant"] == "distractor" else index]]})
    methods = ["base", "projection_edit", "projection_finetune", "oracle_mask"]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "completed_development_pilot",
        "question": "Can four paired examples repair printed-word misidentification on unseen development photographs and a held-out patch location, while preserving clean recognition?",
        "config": CONFIG, "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "dataset": {"id": cache["metadata"]["dataset_id"], "original_images": len(rows) // 3,
                    "support_pairs_used": len(support), "retention_images": len(retention), "probe_images": len(observations) // 3,
                    "source_record": "https://data.caltech.edu/records/mzrjq-6wc02",
                    "manifest_sha256": cache["metadata"]["fingerprint"]["manifest_sha256"],
                    "final_test": False, "human_reviewed": False},
        "measurements": measurements, "patch_rank_bound": patch.rank_bound,
        "patch_bytes": patch_path.stat().st_size, "patch_sha256": file_hash(patch_path),
        "exact_projection_and_prediction_rollback": exact_rollback,
        "results": {method: summarize(observations, method, distractor=CONFIG["distractor"]) for method in methods},
        "generalization": {group: {method: summarize([r for r in observations if (r["class_id"] in CONFIG["support_classes"]) == seen], method)
                                   for method in methods} for group, seen in [("support_classes", True), ("other_classes", False)]},
        "paired_attacked_accuracy_vs_base": {method: paired_accuracy_interval(observations, method) for method in methods[1:]},
        "observations": observations,
        "limitations": ["Development pilot only; no frozen final test or production reliability claim",
                        "Controlled printed-word overlays; same font, different position; not a real-world attack benchmark",
                        "Only the visual projection is edited; frozen CLIP encoders; no captioning or generative VQA",
                        "Source dataset labels; independent human and near-duplicate audits not performed",
                        "Current baseline settings are fixed pilot configurations, not an exhaustive baseline tuning study",
                        "Oracle mask knows the generated label region and is privileged preprocessing, not model editing",
                        "Candidate-vocabulary softmax values are relative scores, not calibrated confidence"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.artifacts / "run-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "results": report["results"],
                      "measurements": measurements, "exact_rollback": exact_rollback}, indent=2))


if __name__ == "__main__":
    main()
