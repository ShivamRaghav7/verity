"""Pack/unpack the pending development pilot with a manifest-bound allowlist.

No review registry, model, test or challenge file is included. This small offline
bundle makes the archived pilot and its v2 derivation reproducible from Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile

ALLOWED = {"scope_train.jsonl", "scope_validation.jsonl", "answer_train.jsonl", "answer_validation.jsonl",
           "catalogue.json", "selection.json", "split_assignments.json", "derivation.json", "manifest.json"}
MAX_BYTES = 8_000_000


def _validate(files, expected_manifest):
    if set(files) != ALLOWED or sum(len(value) for value in files.values()) > MAX_BYTES:
        raise ValueError("Only the finite development-pilot allowlist is permitted.")
    if files["manifest.json"] != expected_manifest.read_bytes():
        raise ValueError("Bundled manifest differs from the recorded source manifest.")
    manifest = json.loads(files["manifest.json"])
    if (manifest.get("pilot_version") not in (1, 2) or manifest.get("test_files_included") is not False
            or manifest.get("test_frozen") is not False or manifest.get("eligible_for_training") is not False
            or manifest.get("purpose") != "development_pilot_not_final_evaluation"
            or set(manifest.get("files", {})) != ALLOWED - {"manifest.json"}):
        raise ValueError("Bundle must be an archived pending pilot-v1/v2 development packet.")
    for name, expected in manifest["files"].items():
        if hashlib.sha256(files[name]).hexdigest() != expected:
            raise ValueError(f"Bundled source integrity failed: {name}")
    return manifest


def pack(source, output, expected_manifest):
    if output.exists():
        raise ValueError("Use a fresh bundle path; evidence is never overwritten.")
    files = {name: (source / name).read_bytes() for name in sorted(ALLOWED)}
    manifest = _validate(files, expected_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, content, compresslevel=9)
    return {"dataset_id": manifest["dataset_id"], "files": len(files), "bytes": output.stat().st_size,
            "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "test_files_included": False}


def unpack(source, output, expected_manifest):
    if output.exists():
        raise ValueError("Use a fresh output directory; source/review files are never overwritten.")
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        if (len(entries) != len(ALLOWED) or {entry.filename for entry in entries} != ALLOWED
                or sum(entry.file_size for entry in entries) > MAX_BYTES
                or any(entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1 for entry in entries)):
            raise ValueError("Archive contains unsupported names, duplicate files, links, encryption or size.")
        files = {entry.filename: archive.read(entry) for entry in entries}
    manifest = _validate(files, expected_manifest)
    output.mkdir(parents=True)
    for name, content in files.items():
        (output / name).write_bytes(content)
    return {"dataset_id": manifest["dataset_id"], "files": len(files), "output": str(output),
            "test_files_included": False, "review_decisions_created": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pack", "unpack"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (pack if args.mode == "pack" else unpack)(args.source, args.output, args.manifest)
    except (ValueError, OSError, zipfile.BadZipFile, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
