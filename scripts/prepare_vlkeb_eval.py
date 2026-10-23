"""Prepare a fixed, source-entity-disjoint 20-edit VLKEB evaluation; never run it.

Default mode prints a plan without network or writes. --prepare explicitly fetches
the 1.66 MB pinned annotation, verifies existing train/archive bytes and extracts
only selected images using the audited development preparer's safe copier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_vlkeb_assets import (download_asset, extract_fixture_images,
                                          load_manifest, relative_path,
                                          verify_asset, write_json)

REVISION = "cf2d0abe73cb638fcf368d8bc5f9ac3caa204d5a"
EVAL_ASSET = {
    "id": "data-eval.json", "path": "upstream-eval.json",
    "url": f"https://huggingface.co/datasets/HymanH/VLKEB-data/resolve/{REVISION}/eval.json",
    "revision": REVISION, "size_bytes": 1658466,
    "sha256": "72c47efe00773bb16cd83258aff394a0894b8c29a6b02e6250717a30d1b18f13",
}
EDIT_COUNT = 20
TRAIN_SHA256 = "6e40e0e9a28dc3e7523f8d48f032755fe050c5b5728344633f13113dbd6fbc01"
ARCHIVE_SHA256 = "88f774e76e51c66b1f2a6510bdc0e05cf51a55a8e45be6186feed8d2360397e4"
FIELDS = ("src", "rephrase", "pred", "alt", "image", "image_rephrase",
          "loc", "loc_ans", "m_loc", "m_loc_q", "m_loc_a")
IMAGE_FIELDS = ("image", "image_rephrase", "m_loc")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def image_entity(path):
    if not isinstance(path, str):
        raise ValueError("Image path must be text")
    parsed = PurePosixPath(path)
    if (parsed.is_absolute() or len(parsed.parts) < 2
            or ".." in parsed.parts or "\\" in path or ":" in path):
        raise ValueError("Unsafe or non-entity image path")
    return parsed.parts[0]


def validated_rows(rows):
    if not isinstance(rows, list):
        raise ValueError("Annotation must be a list")
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in FIELDS):
            raise ValueError("Annotation row lacks a required nonempty scalar field")
        for field in IMAGE_FIELDS:
            image_entity(row[field])
    return rows


def select_edits(train, evaluation):
    """Selection uses annotations only, with no model output or success filtering.

    Group by source entity + original name + counterfactual target. Sort stable
    lineage hashes, take the first lineage for each distinct eligible source
    entity, stop at exactly 20. Canonical row selection is independent of input
    order; every other row in its lineage is recorded but not separately scored.
    """
    train, evaluation = validated_rows(train), validated_rows(evaluation)
    seen_train = {image_entity(row[k]) for row in train for k in IMAGE_FIELDS}
    groups = {}
    for index, row in enumerate(evaluation):
        entity = image_entity(row["image"])
        if entity in seen_train:
            continue
        key = [entity, row["pred"], row["alt"]]
        edit_id = hashlib.sha256(canonical(key).encode()).hexdigest()
        groups.setdefault(edit_id, {"edit_id": edit_id, "source_entity": entity,
                                    "lineage": key, "members": []})["members"].append((index, row))
    selected, used_entities = [], set()
    for edit_id, group in sorted(groups.items()):
        if group["source_entity"] in used_entities:
            continue
        members = group.pop("members")
        index, row = min(members, key=lambda item: (canonical(item[1]), item[0]))
        selected.append({**group, "representative_source_index": index,
                         "all_lineage_source_indices": sorted(i for i, _ in members),
                         "record": row})
        used_entities.add(group["source_entity"])
        if len(selected) == EDIT_COUNT:
            break
    if len(selected) != EDIT_COUNT:
        raise ValueError("Fewer than 20 eligible distinct-source edit lineages; do not expand or substitute")
    return {"schema_version": 1, "purpose": "fixed_small_evaluation_not_executed",
            "source_revision": REVISION, "upstream_split": "eval",
            "source_eval_sha256": EVAL_ASSET["sha256"], "edit_count": EDIT_COUNT,
            "cases_per_edit": 5, "case_count": EDIT_COUNT * 5,
            "selection_rule": "Eligible source image-directory entity absent from every train image/image_rephrase/m_loc directory; sort SHA256(canonical JSON [entity,pred,alt]); first20 distinct source entities; canonical minimum row within each lineage",
            "selection_ids_sha256": hashlib.sha256(canonical([r["edit_id"] for r in selected]).encode()).hexdigest(),
            "train_visual_entity_count": len(seen_train), "edits": selected,
            "review": "author labels plus automated structural audit; no independent human-review claim",
            "evaluation_executed": False,
            "limitations": "Source entity disjoint from pinned train image slots; target names and locality examples may overlap train, and released checkpoint training provenance is not independently certified. This is a small nonrandom academic benchmark slice, not real-world accuracy."}


def prepare(root, output, manifest):
    if output.exists():
        raise FileExistsError("Refusing to overwrite a previously frozen selection")
    train_asset = next(a for a in manifest["assets"] if a["path"] == "upstream-train.json")
    archive_asset = next(a for a in manifest["assets"] if a["path"] == "upstream-images.tar.gz")
    if (train_asset.get("sha256") != TRAIN_SHA256
            or archive_asset.get("sha256") != ARCHIVE_SHA256):
        raise ValueError("This protocol requires the exact pinned training split and image archive")
    verify_asset(relative_path(root, train_asset["path"]), train_asset)
    verify_asset(relative_path(root, archive_asset["path"]), archive_asset)
    download_asset(root, EVAL_ASSET)
    selection = select_edits(json.loads(relative_path(root, train_asset["path"]).read_text(encoding="utf-8")),
                             json.loads(relative_path(root, EVAL_ASSET["path"]).read_text(encoding="utf-8")))
    required = sorted({edit["record"][key] for edit in selection["edits"] for key in IMAGE_FIELDS})
    # Reuse safe streaming extraction: no extractall, links, traversal or
    # duplicate suffix matches. Its archive_verified flag follows verification.
    fixture = {"required_images": required}
    selection["image_integrity"] = extract_fixture_images(root, manifest, fixture, archive_verified=True)
    selection["source_train_sha256"] = train_asset["sha256"]
    write_json(output, selection)
    return {"status": "prepared_not_evaluated", "selection_path": str(output),
            "selection_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "edit_count": EDIT_COUNT, "case_count": EDIT_COUNT * 5,
            "image_count": len(required), "models_loaded": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "artifacts/vlkeb/assets")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs/vlkeb-assets.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = (args.output or root / "eval20-selection.json").resolve()
    if not args.prepare:
        print(json.dumps({"mode": "dry_run", "network_requested": False, "files_written": False,
                          "new_annotation": EVAL_ASSET, "edit_count": EDIT_COUNT,
                          "case_count": EDIT_COUNT * 5, "output": str(output),
                          "evaluation_executed": False}, indent=2))
        return 0
    try:
        result = prepare(root, output, load_manifest(args.manifest))
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Evaluation preparation stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
