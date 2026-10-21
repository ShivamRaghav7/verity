"""Small, source-grouped development data for visual model editing.

All rendered variants inherit their original photograph's split. These are
controlled typographic perturbations, not collected real-world attack reports.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
SOURCE_MD5 = "3138e1922a9193bfa496528edbbc45d0"
SOURCE_RECORD = "https://data.caltech.edu/records/mzrjq-6wc02"
CLASSES = {
    "airplanes": "airplane",
    "camera": "camera",
    "cup": "cup",
    "elephant": "elephant",
    "grand_piano": "grand piano",
    "laptop": "laptop",
    "watch": "watch",
    "cellphone": "cell phone",
}
COUNTS = {"support": 2, "retention": 8, "probe": 8}


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(path: Path, *, allow_download: bool = False) -> None:
    if path.exists():
        if file_hash(path, "md5") != SOURCE_MD5:
            raise ValueError("Existing source archive fails the publisher checksum")
        return
    if not allow_download:
        raise FileNotFoundError("Source archive absent; use --allow-download once")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            output.write(chunk)
    if file_hash(partial, "md5") != SOURCE_MD5:
        raise ValueError("Downloaded source archive fails the publisher checksum")
    partial.replace(path)


def audit_manifest(manifest: dict) -> None:
    """Reject overlap, duplicate IDs, or unexpected roles before model use."""
    rows = manifest["images"]
    hashes, identifiers = set(), set()
    for row in rows:
        if row["split"] not in COUNTS:
            raise ValueError("Unknown split")
        if row["class_id"] not in CLASSES:
            raise ValueError("Unknown source category")
        if row["sha256"] in hashes or row["image_id"] in identifiers:
            raise ValueError("Original photograph repeats across the manifest")
        hashes.add(row["sha256"])
        identifiers.add(row["image_id"])


def prepare_dataset(archive: Path, output: Path) -> dict:
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit_manifest(manifest)
        for row in manifest["images"]:
            if file_hash(output / row["path"]) != row["sha256"]:
                raise ValueError("Prepared photograph differs from its frozen manifest")
        return manifest
    if file_hash(archive, "md5") != SOURCE_MD5:
        raise ValueError("Source checksum mismatch")
    candidates = {category: [] for category in CLASSES}
    # Read only whitelisted image members; never extract arbitrary archive paths.
    with zipfile.ZipFile(archive) as outer:
        members = [name for name in outer.namelist()
                   if name.split("/")[-1] == "101_ObjectCategories.tar.gz"]
        if len(members) != 1:
            raise ValueError("Expected exactly one official image tarball")
        with outer.open(members[0]) as raw, tarfile.open(fileobj=raw, mode="r|gz") as inner:
            for member in inner:
                parts = member.name.replace("\\", "/").split("/")
                if not member.isfile() or len(parts) < 2 or not member.name.lower().endswith(".jpg"):
                    continue
                category = parts[-2].lower()
                if category not in candidates:
                    continue
                stream = inner.extractfile(member)
                if stream is None:
                    continue
                payload = stream.read()
                digest = hashlib.sha256(payload).hexdigest()
                candidates[category].append((digest, parts[-1], payload))
    total = sum(COUNTS.values())
    for category, rows in candidates.items():
        if len(rows) < total:
            raise ValueError(f"Insufficient photographs in {category}: {len(rows)}")
    images, seen = [], set()
    output.mkdir(parents=True, exist_ok=True)
    for category in CLASSES:
        selected = []
        for digest, filename, payload in sorted(candidates[category]):
            if digest not in seen:
                selected.append((digest, filename, payload))
                seen.add(digest)
            if len(selected) == total:
                break
        if len(selected) != total:
            raise ValueError(f"Insufficient distinct image content in {category}")
        offset = 0
        for split, count in COUNTS.items():
            for digest, filename, payload in selected[offset:offset + count]:
                relative = Path("images") / category / filename
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                images.append({"image_id": f"{category}/{filename}", "class_id": category,
                               "label": CLASSES[category], "split": split,
                               "sha256": digest, "path": relative.as_posix()})
            offset += count
    manifest = {
        "dataset_id": "visualpatch-caltech101-development-v1",
        "purpose": "development_feasibility_only", "final_test": False,
        "source": {"record": SOURCE_RECORD, "archive_md5": SOURCE_MD5,
                   "archive_sha256": file_hash(archive), "license": "CC-BY-4.0",
                   "authors": "Li, Andreeto, Ranzato, Perona",
                   "doi": "10.22002/D1.20086"},
        "sampling": "First distinct images sorted by SHA256 within fixed classes; no model-based filtering",
        "classes": CLASSES, "per_class_counts": COUNTS,
        "label_review": "Original dataset labels; no independent human-review claim",
        "limitations": ["Exact-byte deduplication only; near-duplicate and physical-object grouping not yet audited",
                        "Public pretrained model may have seen source images; no pretraining-contamination claim"],
        "images": images,
    }
    audit_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def render_variant(image, variant: str, *, position: str = "bottom", word: str = "camera"):
    """Return deterministic 224px canvas and known patch box; no learned masking."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    if variant not in {"clean", "blank", "distractor"}:
        raise ValueError("Unknown image variant")
    if position not in {"bottom", "top"}:
        raise ValueError("Unknown patch position")
    canvas = ImageOps.fit(image.convert("RGB"), (224, 224), method=Image.Resampling.BICUBIC)
    y = 164 if position == "bottom" else 12
    box = (32, y, 192, y + 46)
    if variant != "clean":
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(box, fill="white", outline="gray", width=1)
        if variant == "distractor":
            font = ImageFont.load_default(size=34)
            draw.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), word,
                      font=font, fill="black", anchor="mm")
    return canvas, box
