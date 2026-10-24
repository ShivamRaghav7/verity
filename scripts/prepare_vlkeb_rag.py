"""Separate pinned MiniLM assets and fixed train-row RAG validation preparation.

Default plan has no network or writes. --prepare fetches only~92MB of MiniLM
files, verifies existing train/archive bytes, and extracts selected train images.
The base manifest is never changed; no test split or model is opened here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_vlkeb_assets import (download_asset, extract_fixture_images, load_manifest,
                                         verify_asset, write_json)
from scripts.prepare_vlkeb_eval import TRAIN_SHA256, ARCHIVE_SHA256

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
INDICES = tuple(range(2, 10))
FILE_PINS = (
    ("config.json", 612, "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41"),
    ("pytorch_model.bin", 90888945, "c3a85f238711653950f6a79ece63eb0ea93d76f6a6284be04019c53733baf256"),
    ("tokenizer_config.json", 350, "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b"),
    ("special_tokens_map.json", 112, "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"),
    ("tokenizer.json", 466247, "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"),
    ("vocab.txt", 231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    ("README.md", 10502, "dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0"),
)


def minilm_assets():
    return [{"path": "minilm/" + name, "size_bytes": size, "sha256": checksum,
             "url": f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}",
             "revision": REVISION} for name, size, checksum in FILE_PINS]


def manifest_record():
    return {"schema_version": 1, "model_id": MODEL_ID, "revision": REVISION,
            "assets": minilm_assets(), "license": "Apache-2.0 per official model card",
            "primary_metadata": f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true",
            "integrity": "Weight SHA256 from official HF LFS metadata; other SHA256 computed from exact pinned small files",
            "embedding_recipe": "Official model-card AutoModel attention-mask mean pooling, L2 normalization; CPU,eval,no_grad,max_length256",
            "total_download_bytes": sum(size for _, size, _ in FILE_PINS)}


def verify_minilm(root):
    records = [verify_asset(root / asset["path"], asset) for asset in minilm_assets()]
    saved = json.loads((root / "minilm-manifest.json").read_text(encoding="utf-8"))
    if saved != manifest_record():
        raise ValueError("MiniLM manifest differs from pinned recipe")
    return records


def expected_fixture(assets):
    raw = (assets / "upstream-train.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRAIN_SHA256:
        raise ValueError("Wrong pinned training annotation")
    rows = json.loads(raw)
    records = [{"source_index": i, "record": rows[i]} for i in INDICES]
    required = sorted({item["record"][key] for item in records for key in ("image", "image_rephrase", "m_loc")})
    return {"schema_version": 1, "purpose": "rag_threshold_validation_only", "upstream_split": "train",
            "source_train_sha256": TRAIN_SHA256, "source_indices": list(INDICES), "records": records,
            "required_images": required, "independent_human_review": False,
            "limitations": "Fixed nonrandom train rows2..9, possibly repeated lineages; author labels and automated structural audit only; no test outputs used"}


def pinned_manifest(path):
    manifest = load_manifest(path)
    train = next(a for a in manifest["assets"] if a["path"] == "upstream-train.json")
    archive = next(a for a in manifest["assets"] if a["path"] == "upstream-images.tar.gz")
    if train.get("sha256") != TRAIN_SHA256 or archive.get("sha256") != ARCHIVE_SHA256:
        raise ValueError("RAG requires the pinned train split and image archive")
    return manifest, train, archive


def validate_calibration_fixture(assets, rag_root, manifest_path):
    manifest, train, archive = pinned_manifest(manifest_path)
    verify_asset(assets / train["path"], train)
    verify_asset(assets / archive["path"], archive)
    model_records = verify_minilm(rag_root)
    expected = expected_fixture(assets)
    saved_path = rag_root / "calibration-fixture.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"RAG validation fixture differs from fixed training indices: {key}")
    derived = extract_fixture_images(assets, manifest, expected, verify_only=True, archive_verified=True)
    if saved.get("image_integrity") != derived:
        raise ValueError("RAG validation images differ from source archive")
    return saved, {"source_train_sha256": TRAIN_SHA256, "source_archive_sha256": ARCHIVE_SHA256,
                   "minilm": model_records, "calibration_fixture_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(),
                   "minilm_manifest_sha256": hashlib.sha256((rag_root / "minilm-manifest.json").read_bytes()).hexdigest()}


def prepare(assets, rag_root, manifest_path):
    manifest, train, archive = pinned_manifest(manifest_path)
    verify_asset(assets / train["path"], train)
    verify_asset(assets / archive["path"], archive)
    for asset in minilm_assets():
        download_asset(rag_root, asset)
    record = manifest_record()
    manifest_output = rag_root / "minilm-manifest.json"
    if manifest_output.exists() and json.loads(manifest_output.read_text(encoding="utf-8")) != record:
        raise ValueError("Refusing to replace a different MiniLM manifest")
    write_json(manifest_output, record)
    fixture = expected_fixture(assets)
    fixture["image_integrity"] = extract_fixture_images(assets, manifest, fixture, archive_verified=True)
    output = rag_root / "calibration-fixture.json"
    if output.exists() and json.loads(output.read_text(encoding="utf-8")) != fixture:
        raise ValueError("Refusing to replace a different calibration fixture")
    write_json(output, fixture)
    return {"status": "prepared_not_calibrated", "models_loaded": False, "test_data_opened": False,
            "validation_records": len(INDICES), "image_count": len(fixture["required_images"]), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--assets", type=Path, default=ROOT / "artifacts/vlkeb/assets")
    parser.add_argument("--rag-root", type=Path, default=ROOT / "artifacts/vlkeb/rag")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/vlkeb-assets.json")
    args = parser.parse_args(argv)
    if not args.prepare:
        print(json.dumps({"mode": "plan", "network_requested": False, "files_written": False,
            "minilm": manifest_record(), "validation_indices": list(INDICES),
            "evaluation_split_opened": False, "base_manifest_changed": False}, indent=2))
        return 0
    try:
        print(json.dumps(prepare(args.assets.resolve(), args.rag_root.resolve(), args.manifest.resolve()), indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
