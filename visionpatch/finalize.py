"""Seal, then evaluate a fixed visual editor on disjoint held-out photographs.

Run --prepare-only first. Evaluation consumes that immutable seal; it contains
no training, hyperparameter search, test-based selection, or silent overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import tarfile
import time
from datetime import datetime, timezone
import zipfile

from .data import CLASSES, SOURCE_MD5, SOURCE_RECORD, audit_manifest, file_hash, render_variant
from .metrics import paired_accuracy_interval, summarize
from .model import MODEL_ID, MODEL_REVISION, image_tensor, load_clip


FINAL_PER_CLASS = 10
EVALUATION = {
    "variants": ["clean", "blank", "distractor"], "position": "top", "distractor": "camera",
    "font": "Pillow ImageFont.load_default(size=34)", "batch_size": 8,
    "bootstrap_samples": 2000, "bootstrap_seed": 42,
    "methods": ["base", "projection_edit", "projection_finetune", "oracle_mask"],
    "scoring": "fixed-candidate normalized CLIP similarity argmax; no abstentions",
}


def _write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _final_audit(manifest: dict, development: dict, output: Path, *, per_class: int) -> None:
    if manifest.get("final_test") is not True or manifest.get("classes") != CLASSES:
        raise ValueError("Expected the fixed held-out final dataset and vocabulary")
    if manifest.get("per_class_count") != per_class:
        raise ValueError("Final image counts differ from the declared protocol")
    development_hashes = {row["sha256"] for row in development["images"]}
    hashes, identifiers = set(), set()
    counts = dict.fromkeys(CLASSES, 0)
    for row in manifest["images"]:
        category = row["class_id"]
        if category not in CLASSES or row["split"] != "final" or row["label"] != CLASSES[category]:
            raise ValueError("Final source category, label, or split mismatch")
        if row["sha256"] in development_hashes:
            raise ValueError("Final photograph overlaps development source content")
        if row["sha256"] in hashes or row["image_id"] in identifiers:
            raise ValueError("Final source content or identifier repeats")
        hashes.add(row["sha256"])
        identifiers.add(row["image_id"])
        counts[category] += 1
        relative = Path(row["path"])
        target = (output / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(output.resolve()):
            raise ValueError("Final source path escapes its dataset directory")
        if file_hash(target) != row["sha256"]:
            raise ValueError("Final photograph differs from its sealed content hash")
    if any(count != per_class for count in counts.values()):
        raise ValueError("Final dataset must contain exactly the fixed count in every class")


def prepare_final_dataset(archive: Path, development_manifest: Path, output: Path,
                          *, per_class: int = FINAL_PER_CLASS) -> dict:
    """Select by content hash only, excluding every development original.

    This function never loads a model or ranks images by predictions. Source
    archive members are read through a whitelist, not extracted to arbitrary
    paths. Prepared final manifests cannot be overwritten.
    """
    if type(per_class) is not int or per_class <= 0:
        raise ValueError("per_class must be a positive integer")
    if (output / "manifest.json").exists():
        raise FileExistsError("Final manifest exists; preserve its sealed split")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Final dataset directory is nonempty; refusing partial-state overwrite")
    if file_hash(archive, "md5") != SOURCE_MD5:
        raise ValueError("Official source archive checksum mismatch")
    development = json.loads(development_manifest.read_text(encoding="utf-8"))
    audit_manifest(development)
    if development.get("final_test") is not False or development.get("classes") != CLASSES:
        raise ValueError("Development manifest and fixed vocabulary disagree")
    if development["source"]["archive_sha256"] != file_hash(archive):
        raise ValueError("Final and development data must use the same approved source archive")
    excluded = {row["sha256"] for row in development["images"]}
    candidates = {category: {} for category in CLASSES}
    with zipfile.ZipFile(archive) as outer:
        names = [name for name in outer.namelist()
                 if name.split("/")[-1] == "101_ObjectCategories.tar.gz"]
        if len(names) != 1:
            raise ValueError("Expected one official image tarball")
        with outer.open(names[0]) as raw, tarfile.open(fileobj=raw, mode="r|gz") as inner:
            for member in inner:
                parts = member.name.replace("\\", "/").split("/")
                if not member.isfile() or len(parts) < 2 or not parts[-1].lower().endswith(".jpg"):
                    continue
                category = parts[-2].lower()
                if category not in candidates:
                    continue
                source = inner.extractfile(member)
                if source is None:
                    continue
                payload = source.read()
                digest = hashlib.sha256(payload).hexdigest()
                if digest not in excluded:
                    candidates[category].setdefault(digest, (parts[-1], payload))
    # Plan the entire selection before writing any photograph.
    selection, seen = [], set()
    for category in CLASSES:
        selected = 0
        for digest in sorted(candidates[category]):
            if digest in seen:
                continue
            filename, payload = candidates[category][digest]
            selection.append((category, digest, filename, payload))
            seen.add(digest)
            selected += 1
            if selected == per_class:
                break
        if selected != per_class:
            raise ValueError(f"Insufficient unseen distinct photographs for {category}")
    rows = []
    for category, digest, filename, payload in selection:
        relative = Path("images") / category / filename
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
        rows.append({"image_id": f"{category}/{filename}", "class_id": category,
                     "label": CLASSES[category], "split": "final", "sha256": digest,
                     "path": relative.as_posix()})
    manifest = {
        "dataset_id": "visionpatch-caltech101-final-v1", "final_test": True,
        "purpose": "sealed_held_out_evaluation", "classes": CLASSES,
        "per_class_count": per_class, "source": development["source"],
        "development_manifest_sha256": file_hash(development_manifest),
        "excluded_development_originals": len(excluded),
        "sampling": "First distinct SHA256-sorted originals per fixed class after excluding every development hash; no model filtering",
        "label_review": "Automated source-label and content-hash audit; no independent human review",
        "limitations": ["Held out from development content, not from unknown model pretraining",
                        "Exact-byte duplicates excluded; near-duplicate and physical-object grouping not audited"],
        "images": rows,
    }
    _final_audit(manifest, development, output, per_class=per_class)
    _write_new_json(output / "manifest.json", manifest)
    return manifest


def _runtime_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "Pillow")}


def _load_trained(artifacts: Path, development_manifest: Path):
    """Inspect existing artifacts; no fit function or optimizer is imported."""
    import torch
    from .editor import load_projection_patch, projection_sha256

    cache = torch.load(artifacts / "features.pt", weights_only=True, map_location="cpu")
    finetuned = torch.load(artifacts / "projection-finetune.pt", weights_only=True, map_location="cpu")
    patch = load_projection_patch(artifacts / "projection-patch.pt")
    fingerprint = cache["metadata"]["fingerprint"]
    if (fingerprint["model_id"] != MODEL_ID or fingerprint["model_revision"] != MODEL_REVISION
            or fingerprint["manifest_sha256"] != file_hash(development_manifest)):
        raise ValueError("Training artifacts are not bound to the approved model/development manifest")
    if cache["metadata"]["class_ids"] != list(CLASSES) or cache["metadata"]["labels"] != list(CLASSES.values()):
        raise ValueError("Trained candidate vocabulary differs from the fixed final vocabulary")
    if patch.metadata.get("fingerprint") != fingerprint:
        raise ValueError("Projection patch and cached base refer to different training inputs")
    if patch.metadata.get("config") != finetuned.get("config"):
        raise ValueError("Frozen editing and fine-tuning configurations disagree")
    if patch.metadata["config"].get("distractor") != EVALUATION["distractor"]:
        raise ValueError("Final distractor differs from the frozen edit configuration")
    if projection_sha256(cache["base_weight"]) != patch.before_sha256:
        raise ValueError("Cached base projection does not match the saved patch parent")
    weight = finetuned["weight"]
    if weight.shape != cache["base_weight"].shape or not bool(torch.isfinite(weight).all()):
        raise ValueError("Invalid frozen fine-tuning projection")
    text = cache["text_features"]
    if text.shape != (len(CLASSES), weight.shape[0]) or not bool(torch.isfinite(text).all()):
        raise ValueError("Invalid frozen candidate text embeddings")
    return cache, patch, finetuned


def _bindings(args) -> dict:
    paths = {
        "archive": args.archive, "development_manifest": args.development_manifest,
        "final_manifest": args.dataset / "manifest.json",
        "development_features": args.trained_artifacts / "features.pt",
        "projection_patch": args.trained_artifacts / "projection-patch.pt",
        "projection_finetune": args.trained_artifacts / "projection-finetune.pt",
    }
    for name in ("data", "model", "editor", "metrics", "finalize"):
        paths[f"code_{name}"] = Path(__file__).with_name(f"{name}.py")
    return {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in paths.items()}


def load_and_validate_seal(path: Path) -> dict:
    """Refuse changed inputs/configuration before reading any final prediction."""
    seal = json.loads(path.read_text(encoding="utf-8"))
    if (type(seal.get("format_version")) is not int or seal.get("format_version") != 1
            or seal.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}):
        raise ValueError("Unknown final seal format or model")
    if seal.get("evaluation") != EVALUATION or seal.get("classes") != CLASSES:
        raise ValueError("Final evaluation configuration changed after sealing")
    if seal.get("final_per_class") != FINAL_PER_CLASS:
        raise ValueError("Final split size changed after sealing")
    required = {"archive", "development_manifest", "final_manifest", "development_features",
                "projection_patch", "projection_finetune", "code_data", "code_model",
                "code_editor", "code_metrics", "code_finalize"}
    if set(seal.get("bindings", {})) != required:
        raise ValueError("Final seal is missing required input bindings")
    for name, binding in seal["bindings"].items():
        if file_hash(Path(binding["path"])) != binding["sha256"]:
            raise ValueError(f"Sealed input changed: {name}")
    return seal


def prepare(args) -> dict:
    if args.seal.exists() or args.report.exists() or (args.artifacts / "features.pt").exists():
        raise FileExistsError("Final evidence already exists; refusing to reseal or overwrite it")
    cache, patch, finetuned = _load_trained(args.trained_artifacts, args.development_manifest)
    manifest = prepare_final_dataset(args.archive, args.development_manifest, args.dataset)
    seal = {
        "format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_final_inference", "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "classes": CLASSES, "final_per_class": FINAL_PER_CLASS, "evaluation": EVALUATION,
        "evaluation_device": args.device,
        "trained_config": finetuned["config"], "base_projection_sha256": patch.before_sha256,
        "edited_projection_sha256": patch.after_sha256, "runtime_versions": _runtime_versions(),
        "bindings": _bindings(args), "final_originals": len(manifest["images"]),
        "finite_stop": f"{len(manifest['images'])} originals x 3 variants; no optimizer or parameter fitting",
        "outputs": {"report": str(args.report.resolve()), "features": str((args.artifacts / "features.pt").resolve())},
        "development_dataset_id": cache["metadata"]["dataset_id"],
        "review": "Automated source labels, checksums and exact-content split audit; not human reviewed",
    }
    _write_new_json(args.seal, seal)
    return seal


def _predict(features, weight, text):
    import torch.nn.functional as functional

    return (functional.normalize(features @ weight.T, dim=-1) @ text.T).argmax(dim=-1).tolist()


def evaluate(args) -> dict:
    # Check output paths before optional model imports or loading final pixels.
    if args.report.exists() or (args.artifacts / "features.pt").exists():
        raise FileExistsError("Final outputs exist; reruns cannot overwrite recorded evidence")
    seal = load_and_validate_seal(args.seal)
    if seal["outputs"] != {"report": str(args.report.resolve()), "features": str((args.artifacts / "features.pt").resolve())}:
        raise ValueError("Final output paths differ from the sealed protocol")
    if seal["bindings"] != _bindings(args):
        raise ValueError("Requested final input paths differ from the sealed protocol")
    if _runtime_versions() != seal["runtime_versions"]:
        raise ValueError("Runtime versions changed after final protocol sealing")
    if args.device != seal["evaluation_device"]:
        raise ValueError("Evaluation device differs from the sealed protocol")
    development = json.loads(args.development_manifest.read_text(encoding="utf-8"))
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    _final_audit(manifest, development, args.dataset, per_class=FINAL_PER_CLASS)
    cache, patch, finetuned = _load_trained(args.trained_artifacts, args.development_manifest)
    if (finetuned["config"] != seal["trained_config"]
            or patch.before_sha256 != seal["base_projection_sha256"]
            or patch.after_sha256 != seal["edited_projection_sha256"]):
        raise ValueError("Trained artifact configuration or projection state differs from its seal")
    import torch
    from PIL import Image
    from .editor import apply_projection_patch_, projection_sha256, rollback_projection_patch_

    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent device substitution")
    model, _, model_load_seconds = load_clip(device=args.device)
    if projection_sha256(model.visual_projection.weight) != seal["base_projection_sha256"]:
        raise ValueError("Loaded checkpoint projection differs from the sealed parent weights")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = [{**source, "variant": variant, "position": EVALUATION["position"]}
            for source in manifest["images"] for variant in EVALUATION["variants"]]
    features, durations = [], []
    started = time.perf_counter()
    for offset in range(0, len(rows), EVALUATION["batch_size"]):
        batch = rows[offset:offset + EVALUATION["batch_size"]]
        rendered = []
        for row in batch:
            with Image.open(args.dataset / row["path"]) as source:
                image, _ = render_variant(source, row["variant"], position=row["position"], word=EVALUATION["distractor"])
                rendered.append(image)
        pixels = image_tensor(rendered).to(args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
        batch_start = time.perf_counter()
        with torch.inference_mode():
            features.append(model.vision_model(pixel_values=pixels).pooler_output.cpu())
        if args.device == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - batch_start)
    elapsed = time.perf_counter() - started
    features = torch.cat(features)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Final encoder produced non-finite features; no score will be reported")
    base, text = cache["base_weight"], cache["text_features"]
    edited = base.clone()
    apply_projection_patch_(edited, patch)
    predictions = {"base": _predict(features, base, text),
                   "projection_edit": _predict(features, edited, text),
                   "projection_finetune": _predict(features, finetuned["weight"], text)}
    rollback_projection_patch_(edited, patch)
    if not torch.equal(edited, base) or _predict(features, edited, text) != predictions["base"]:
        raise RuntimeError("Final rollback did not exactly restore weights and predictions")
    indexes = {(row["image_id"], row["variant"]): i for i, row in enumerate(rows)}
    observations = []
    for index, row in enumerate(rows):
        blank_index = indexes[(row["image_id"], "blank")]
        observations.append({"index": index, "image_id": row["image_id"], "class_id": row["class_id"],
                             "variant": row["variant"], "position": row["position"],
                             **{name: list(CLASSES)[values[index]] for name, values in predictions.items()},
                             "oracle_mask": list(CLASSES)[predictions["base"][blank_index if row["variant"] == "distractor" else index]]})
    methods = EVALUATION["methods"]
    measurements = {"device": args.device, "model_load_seconds": model_load_seconds,
                    "encoder_preparation_seconds": elapsed, "encoder_batches": len(durations),
                    "encoder_batch_seconds": durations, "encoder_batch_median_seconds": statistics.median(durations),
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
                    "limitation": "Batch encoder timings are not end-to-end single-request latency; CPU peak RAM not instrumented"}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "completed_frozen_final_evaluation",
        "training_performed": False, "selection_or_tuning_performed": False,
        "question": "How do the already-frozen projection edit and matched pilot fine-tuning baseline generalize to new source photographs?",
        "seal_sha256": file_hash(args.seal), "model": seal["model"], "config": seal["trained_config"],
        "evaluation": EVALUATION, "runtime_versions": seal["runtime_versions"],
        "dataset": {"id": manifest["dataset_id"], "final_test": True, "human_reviewed": False,
                    "original_images": len(manifest["images"]), "variants": len(rows),
                    "per_class": FINAL_PER_CLASS, "development_overlap_hashes": 0,
                    "excluded_development_originals": manifest["excluded_development_originals"],
                    "manifest_sha256": file_hash(args.dataset / "manifest.json"), "source_record": SOURCE_RECORD},
        "measurements": measurements, "exact_projection_and_prediction_rollback": True,
        "results": {method: summarize(observations, method) for method in methods},
        "generalization": {group: {method: summarize([row for row in observations if (row["class_id"] in seal["trained_config"]["support_classes"]) == seen], method)
                                   for method in methods} for group, seen in (("support_classes", True), ("other_classes", False))},
        "paired_attacked_accuracy_vs_base": {method: paired_accuracy_interval(observations, method, seed=EVALUATION["bootstrap_seed"], samples=EVALUATION["bootstrap_samples"])
                                            for method in methods if method != "base"},
        "observations": observations,
        "limitations": ["Final split is held out from development by exact content; unknown pretrained-data overlap remains",
                        "Source-label and hash audits are automated; no independent human or near-duplicate review",
                        "One printed word, one font, one final overlay location and eight fixed candidate categories",
                        "Small same-source held-out study, not a real-world security or production reliability guarantee",
                        "Fixed pilot baselines are not an exhaustive hyperparameter or method comparison",
                        "Oracle mask has privileged knowledge of the generated overlay location",
                        "No captioning, generative question answering, or novelty claim for the editing algorithm"],
    }
    args.artifacts.mkdir(parents=True, exist_ok=True)
    feature_path = args.artifacts / "features.pt"
    with feature_path.open("xb") as stream:
        torch.save({"features": features, "rows": rows, "text_features": text, "base_weight": base,
                    "metadata": {"seal_sha256": report["seal_sha256"], "labels": list(CLASSES.values()),
                                 "class_ids": list(CLASSES), "measurements": measurements}}, stream)
    report["features_sha256"] = file_hash(feature_path)
    _write_new_json(args.report, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("artifacts/visionpatch/caltech-101.zip"))
    parser.add_argument("--development-manifest", type=Path, default=Path("data/generated/visionpatch/development-v1/manifest.json"))
    parser.add_argument("--trained-artifacts", type=Path, default=Path("artifacts/visionpatch/development-v1"))
    parser.add_argument("--dataset", type=Path, default=Path("data/generated/visionpatch/final-v1"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/visionpatch/final-v1"))
    parser.add_argument("--seal", type=Path, default=Path("reports/visionpatch/final-seal-v1.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/visionpatch/final-v1.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--prepare-only", action="store_true", help="Create disjoint final images and freeze protocol; no encoder inference")
    args = parser.parse_args()
    report = prepare(args) if args.prepare_only else evaluate(args)
    print(json.dumps({"status": report["status"], "seal": str(args.seal),
                      "final_originals": report.get("final_originals", report.get("dataset", {}).get("original_images")),
                      "results": report.get("results"), "measurements": report.get("measurements")}, indent=2))


if __name__ == "__main__":
    main()
