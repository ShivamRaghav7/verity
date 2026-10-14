"""Preserve/reconstruct exact original data bytes; never interpret question rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ORIGINAL_MANIFEST_SHA256 = "2c5882490ac5f29c45c2bf20f97128d46cc3e0974c84a4838d47bd3ac8aaa42d"
FILES = {"catalogue.json", "edits.json", "split_assignments.json", "scope_train.jsonl",
         "scope_validation.jsonl", "scope_test.jsonl", "answer_train.jsonl",
         "answer_validation.jsonl", "answer_test.jsonl", "challenge_drafts.jsonl"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_payload(payload, expected_manifest=ORIGINAL_MANIFEST_SHA256):
    if set(payload) != FILES | {"manifest.json"}:
        raise ValueError("Bundle must contain the exact original eleven-file inventory.")
    if digest(payload["manifest.json"]) != expected_manifest:
        raise ValueError("Original manifest checksum differs.")
    manifest = json.loads(payload["manifest.json"])
    if manifest.get("dataset_id") != "editguard-controlled-v1" or set(manifest["files"]) != FILES:
        raise ValueError("Unexpected original data manifest.")
    if any(digest(payload[name]) != value for name, value in manifest["files"].items()):
        raise ValueError("Original source-file checksum differs.")
    return manifest


def create_bundle(source, output, *, expected_manifest=ORIGINAL_MANIFEST_SHA256):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("Use a fresh archive outside the original source directory.")
    if {p.name for p in source.iterdir()} != FILES | {"manifest.json"}:
        raise ValueError("Source inventory must remain unchanged.")
    payload = {name: (source / name).read_bytes() for name in FILES | {"manifest.json"}}
    validate_payload(payload, expected_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(payload):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return {"archive_sha256": digest(output.read_bytes()), "files": len(payload),
            "original_manifest_sha256": expected_manifest, "question_content_deserialized": False}


def restore_bundle(archive_path, output, *, expected_manifest=ORIGINAL_MANIFEST_SHA256):
    archive_path, output = Path(archive_path).resolve(), Path(output).resolve()
    if output.exists() or archive_path == output or output in archive_path.parents:
        raise ValueError("Reconstruction requires a fresh directory; existing data is never overwritten.")
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if (len(entries) != 11 or len({item.filename for item in entries}) != 11
                or {item.filename for item in entries} != FILES | {"manifest.json"}
                or any(item.is_dir() or item.file_size > 16 * 1024**2 for item in entries)
                or sum(item.file_size for item in entries) > 128 * 1024**2):
            raise ValueError("Unexpected, duplicate, escaping or oversized archive entry.")
        payload = {item.filename: archive.read(item) for item in entries}
    validate_payload(payload, expected_manifest)
    output.mkdir(parents=True)
    for name, raw in payload.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    return {"directory": str(output), "files": len(payload), "original_manifest_sha256": expected_manifest,
            "question_content_deserialized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (create_bundle(args.source, args.output) if args.action == "create"
                  else restore_bundle(args.archive, args.output))
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
