"""Prepare fresh photo splits without consulting model predictions.

All 224 v1 development/final originals are excluded from new validation/final
selection. Only the 80 old support/retention photos may be reused for training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

from .data import CLASSES, COUNTS, SOURCE_MD5, file_hash
from .study_protocol import font_records, load_study_config

TRAIN_COUNTS = {"support": 2, "retention": 8}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_target(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    target = (root / relative_path).resolve()
    if relative_path.is_absolute() or not target.is_relative_to(root.resolve()):
        raise ValueError("Study source path escapes the dataset directory")
    return target


def _previous_audit(development: dict, previous_final: dict) -> list[dict]:
    if development.get("classes") != CLASSES or previous_final.get("classes") != CLASSES:
        raise ValueError("Previous and current candidate vocabulary must agree")
    if development.get("final_test") is not False or previous_final.get("final_test") is not True:
        raise ValueError("Expected original v1 development and final manifests")
    seen_hashes, seen_ids, counts = set(), set(), Counter()
    for previous_name, manifest in (("development", development), ("final", previous_final)):
        for row in manifest["images"]:
            category, split = row["class_id"], row["split"]
            if category not in CLASSES or row["label"] != CLASSES[category]:
                raise ValueError("Previous source label or class mismatch")
            if split not in (set(COUNTS) if previous_name == "development" else {"final"}):
                raise ValueError("Unexpected previous split")
            if row["sha256"] in seen_hashes or row["image_id"] in seen_ids:
                raise ValueError("Previous source content or identifier repeats")
            seen_hashes.add(row["sha256"])
            seen_ids.add(row["image_id"])
            counts[(category, split)] += 1
    for category in CLASSES:
        for split, expected in {**COUNTS, "final": 10}.items():
            if counts[(category, split)] != expected:
                raise ValueError("Previous manifests do not contain the complete fixed v1 study")
    return development["images"] + previous_final["images"]


def audit_study_manifest(manifest: dict, dataset_root: Path, config: dict | None = None) -> None:
    """Validate grouping, source byte hashes and fixed per-class sample counts."""
    if manifest.get("protocol_id") != "visionpatch-selectivity-v2" or manifest.get("classes") != CLASSES:
        raise ValueError("Unknown study or vocabulary")
    if manifest.get("new_per_class") != {"validation": 4, "final": 6}:
        raise ValueError("Study split sizes differ from the fixed protocol")
    if config is not None and config["new_per_class"] != manifest["new_per_class"]:
        raise ValueError("Configuration and manifest split sizes disagree")
    previous = manifest.get("previous_originals", [])
    expected_previous = len(CLASSES) * (sum(COUNTS.values()) + 10)
    if len(previous) != expected_previous:
        raise ValueError("Full v1 exclusion list is required")
    excluded_hashes = {row["sha256"] for row in previous}
    excluded_ids = {row["image_id"] for row in previous}
    if len(excluded_hashes) != len(previous) or len(excluded_ids) != len(previous):
        raise ValueError("V1 exclusion content or identifier repeats")
    reusable = {(row["image_id"], row["sha256"], row["class_id"], row["split"])
                for row in previous if row["split"] in TRAIN_COUNTS}
    hashes, identifiers, counts = set(), set(), Counter()
    for row in manifest["images"]:
        category, split, role = row["class_id"], row["split"], row["role"]
        if category not in CLASSES or row["label"] != CLASSES[category] or split not in {"train", "validation", "final"}:
            raise ValueError("Unknown source category, label or study split")
        if row["sha256"] in hashes or row["image_id"] in identifiers:
            raise ValueError("An original photograph repeats across study roles")
        hashes.add(row["sha256"])
        identifiers.add(row["image_id"])
        if split == "train":
            key = (row["image_id"], row["sha256"], category, role)
            if key not in reusable:
                raise ValueError("Only original v1 support/retention photographs may be reused for training")
        elif role != "evaluation" or row["sha256"] in excluded_hashes or row["image_id"] in excluded_ids:
            raise ValueError("New evaluation photographs overlap prior v1 content or identifiers")
        if file_hash(_safe_target(dataset_root, row["path"])) != row["sha256"]:
            raise ValueError("Study photograph differs from its frozen source hash")
        counts[(category, split, role)] += 1
    for category in CLASSES:
        for role, count in TRAIN_COUNTS.items():
            if counts[(category, "train", role)] != count:
                raise ValueError("Training source counts differ from the fixed protocol")
        for split, count in manifest["new_per_class"].items():
            if counts[(category, split, "evaluation")] != count:
                raise ValueError("Fresh evaluation counts differ from the fixed protocol")


def prepare_study_dataset(archive: Path, development_manifest: Path,
                          previous_final_manifest: Path, output: Path,
                          config_path: Path) -> dict:
    """Select by SHA256, freeze sources/fonts/config and never overwrite a split."""
    config = load_study_config(config_path)
    if file_hash(archive, "md5") != SOURCE_MD5:
        raise ValueError("Official source archive checksum mismatch")
    development, previous_final = _read_json(development_manifest), _read_json(previous_final_manifest)
    previous = _previous_audit(development, previous_final)
    archive_sha256 = file_hash(archive)
    if any(source["source"]["archive_sha256"] != archive_sha256 for source in (development, previous_final)):
        raise ValueError("Study and previous manifests must use the identical approved source archive")
    input_hashes = {"config_sha256": file_hash(config_path),
                    "previous_development_manifest_sha256": file_hash(development_manifest),
                    "previous_final_manifest_sha256": file_hash(previous_final_manifest)}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("input_hashes") != input_hashes or manifest.get("fonts") != font_records():
            raise ValueError("Frozen study inputs or font assets changed; choose a new study directory")
        audit_study_manifest(manifest, output, config)
        return manifest
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Study directory is nonempty; refusing partial-state overwrite")
    candidates = {category: {} for category in CLASSES}
    with zipfile.ZipFile(archive) as outer:
        names = [name for name in outer.namelist() if name.split("/")[-1] == "101_ObjectCategories.tar.gz"]
        if len(names) != 1:
            raise ValueError("Expected exactly one official image tarball")
        with outer.open(names[0]) as raw, tarfile.open(fileobj=raw, mode="r|gz") as inner:
            for member in inner:
                parts = member.name.replace("\\", "/").split("/")
                if not member.isfile() or len(parts) < 2 or not parts[-1].lower().endswith(".jpg"):
                    continue
                category = parts[-2].lower()
                if category not in candidates:
                    continue
                stream = inner.extractfile(member)
                if stream is None:
                    continue
                payload = stream.read()
                digest = hashlib.sha256(payload).hexdigest()
                candidates[category].setdefault(digest, (parts[-1], payload))
    excluded = {row["sha256"] for row in previous}
    excluded_ids = {row["image_id"] for row in previous}
    selected, seen = [], set()
    for row in development["images"]:
        if row["split"] not in TRAIN_COUNTS:
            continue
        source = candidates[row["class_id"]].get(row["sha256"])
        if source is None or f"{row['class_id']}/{source[0]}" != row["image_id"]:
            raise ValueError("Reusable v1 source is missing from the approved archive")
        selected.append((row["class_id"], row["sha256"], source[0], source[1], "train", row["split"]))
        seen.add(row["sha256"])
    for category in CLASSES:
        unseen = [(digest, filename, payload) for digest, (filename, payload) in sorted(candidates[category].items())
                  if digest not in excluded and digest not in seen and f"{category}/{filename}" not in excluded_ids]
        needed = sum(config["new_per_class"].values())
        if len(unseen) < needed:
            raise ValueError(f"Insufficient fresh distinct photographs for {category}: {len(unseen)}")
        offset = 0
        for split, count in config["new_per_class"].items():
            for digest, filename, payload in unseen[offset:offset + count]:
                if digest in seen:
                    raise ValueError("Source content repeats across selected categories")
                selected.append((category, digest, filename, payload, split, "evaluation"))
                seen.add(digest)
            offset += count
    images = []
    for category, digest, filename, payload, split, role in selected:
        relative = Path("images") / category / filename
        target = _safe_target(output, relative.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
        images.append({"image_id": f"{category}/{filename}", "class_id": category, "label": CLASSES[category],
                       "split": split, "role": role, "sha256": digest, "path": relative.as_posix()})
    manifest = {
        "dataset_id": "visionpatch-caltech101-selectivity-v2",
        "protocol_id": config["protocol_id"], "classes": CLASSES,
        "new_per_class": config["new_per_class"], "train_per_class": TRAIN_COUNTS,
        "source": development["source"], "input_hashes": input_hashes,
        "fonts": font_records(), "source_category_counts": {key: len(value) for key, value in candidates.items()},
        "previous_originals": [{key: row[key] for key in ("image_id", "class_id", "sha256", "split")}
                               for row in previous],
        "sampling": "Reuse only v1 support/retention for training; new SHA256-sorted originals per class after excluding every v1 development/final hash and identifier; source split before all overlays",
        "label_review": "Inherited Caltech101 category labels with automated source/content audit; no independent human review",
        "limitations": config["limitations"], "images": images,
    }
    audit_study_manifest(manifest, output, config)
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("artifacts/visionpatch/caltech-101.zip"))
    parser.add_argument("--development", type=Path, default=Path("data/generated/visionpatch/development-v1/manifest.json"))
    parser.add_argument("--previous-final", type=Path, default=Path("data/generated/visionpatch/final-v1/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/visionpatch/study-v2"))
    parser.add_argument("--config", type=Path, default=Path("configs/visionpatch-study-v2.json"))
    args = parser.parse_args()
    manifest = prepare_study_dataset(args.archive, args.development, args.previous_final, args.output, args.config)
    print(json.dumps({"dataset_id": manifest["dataset_id"], "counts": dict(Counter(row["split"] for row in manifest["images"])),
                      "excluded_v1_originals": len(manifest["previous_originals"]),
                      "manifest": str(args.output / "manifest.json"), "model_inference_ran": False}, indent=2))


if __name__ == "__main__":
    main()
