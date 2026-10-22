"""Recover the published photo splits without resampling or model execution.

Tracked manifests preserve the original byte hashes. This utility copies only
their declared photographs from the checksum-approved Caltech101 archive. It
does not train, select parameters, repeat evaluation or relocate an old seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

from .data import SOURCE_MD5, download_source, file_hash

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("development-v1", "final-v1", "study-v2")


def _digest(value: str, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _target(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError("Manifest paths must be strings")
    path = PurePosixPath(relative)
    if ("\\" in relative or ":" in relative
            or path.is_absolute() or any(part in {".", ".."} for part in relative.split("/"))):
        raise ValueError("Manifest path escapes its dataset directory")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes its dataset directory")
    # A recovery must not follow an existing link, even to another local file.
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError("Recovery paths must not contain symbolic links")
        current = current.parent
    return target


def _write_new_or_matching(path: Path, payload: bytes) -> bool:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite different existing content: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
    return True


def restore_manifest(archive: Path, manifest: Path, output: Path) -> dict:
    """Restore exact listed bytes, accepting only identical existing files.

    The helper checks both archive digests declared by the manifest; the CLI
    additionally requires the publisher's official MD5. Synthetic test archives
    can therefore exercise recovery without downloading the real photographs.
    All members and existing destination files are validated before any write.
    """
    archive, manifest, output = Path(archive), Path(manifest), Path(output).absolute()
    if output.is_symlink():
        raise ValueError("Dataset output must not be a symbolic link")
    manifest_bytes = manifest.read_bytes()
    metadata = json.loads(manifest_bytes)
    source = metadata.get("source", {})
    if not _digest(source.get("archive_md5"), 32) or not _digest(source.get("archive_sha256"), 64):
        raise ValueError("Manifest must declare complete archive checksums")
    if (file_hash(archive, "md5") != source["archive_md5"]
            or file_hash(archive) != source["archive_sha256"]):
        raise ValueError("Source archive differs from the recorded checksums")
    images = metadata.get("images")
    classes = metadata.get("classes")
    if not isinstance(images, list) or not images or not isinstance(classes, dict):
        raise ValueError("Manifest must declare classes and a nonempty photo list")
    destination_manifest = _target(output, "manifest.json")
    if destination_manifest.exists() and destination_manifest.read_bytes() != manifest_bytes:
        raise ValueError("Destination manifest differs; preserve it and use a new output directory")

    declared, targets, hashes = {}, {}, set()
    for row in images:
        category, image_id, relative = row["class_id"], row["image_id"], row["path"]
        if (not isinstance(category, str) or category not in classes
                or any(c in category for c in "/\\:") or category in {".", ".."}
                or row.get("label") != classes[category]):
            raise ValueError("Manifest class or label is invalid")
        if not isinstance(image_id, str) or image_id != f"{category}/{PurePosixPath(image_id).name}":
            raise ValueError("Photo identifier must be class/filename")
        filename = PurePosixPath(image_id).name
        if not filename.lower().endswith(".jpg") or filename.startswith("._"):
            raise ValueError("Only declared original JPEG photographs can be recovered")
        if relative != "images/" + image_id:
            raise ValueError("Manifest image path differs from its source identifier")
        target = _target(output, relative)
        digest = row.get("sha256")
        if not _digest(digest, 64) or digest in hashes or image_id in declared or target in targets.values():
            raise ValueError("Manifest repeats a photo or has an invalid content hash")
        hashes.add(digest)
        declared[image_id], targets[image_id] = row, target
        if target.exists() and (not target.is_file() or file_hash(target) != digest):
            raise ValueError(f"Refusing to overwrite different existing content: {target}")

    payloads = {}
    with zipfile.ZipFile(archive) as outer:
        tar_names = [name for name in outer.namelist()
                     if name.split("/")[-1] == "101_ObjectCategories.tar.gz"]
        if len(tar_names) != 1:
            raise ValueError("Expected exactly one official image tarball")
        with outer.open(tar_names[0]) as raw, tarfile.open(fileobj=raw, mode="r|gz") as inner:
            for member in inner:
                parts = member.name.split("/")
                if (len(parts) != 3 or parts[0] != "101_ObjectCategories"
                        or not member.isfile()):
                    continue
                image_id = "/".join(parts[1:])
                if image_id not in declared:
                    continue
                if image_id in payloads:
                    raise ValueError("Declared source photograph appears more than once in archive")
                stream = inner.extractfile(member)
                if stream is None:
                    raise ValueError("Cannot read declared source photograph")
                payload = stream.read()
                if hashlib.sha256(payload).hexdigest() != declared[image_id]["sha256"]:
                    raise ValueError("Source photograph differs from the frozen manifest")
                payloads[image_id] = payload
    if set(payloads) != set(declared):
        raise ValueError("Archive is missing a declared source photograph")

    restored = sum(_write_new_or_matching(targets[image_id], payloads[image_id]) for image_id in declared)
    _write_new_or_matching(destination_manifest, manifest_bytes)
    return {"dataset_id": metadata.get("dataset_id"), "photos": len(declared),
            "restored_photos": restored, "matching_existing_photos": len(declared) - restored,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "output": str(output), "no_model_execution": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "artifacts/visionpatch/caltech-101.zip")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests/visionpatch")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/generated/visionpatch")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    download_source(args.archive, allow_download=args.allow_download)
    for name in args.datasets:
        metadata = json.loads((args.manifests / (name + ".json")).read_bytes())
        if metadata.get("source", {}).get("archive_md5") != SOURCE_MD5:
            raise ValueError("CLI recovery requires the official Caltech101 publisher checksum")
    restored = [restore_manifest(args.archive, args.manifests / (name + ".json"), args.output_root / name)
                for name in args.datasets]
    print(json.dumps({"status": "restored_exact_photo_splits", "datasets": restored,
                      "notice": "No model training or evaluation ran. Old absolute-path seals are historical evidence; a new reproduction requires its own artifacts and seal."}, indent=2))


if __name__ == "__main__":
    main()
