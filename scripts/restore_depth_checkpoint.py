"""Restore audited depth ledgers without changing any saved outcome or loading a model.

Only the eight canonical JSON files are accepted. A new directory is published
atomically. With --preserve-existing, matching files are retained and missing
files are created exclusively, with rollback of this restore's new files on a
publication error. Run against an idle destination, not an active evaluator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from uuid import uuid4
import zipfile

from editguard_depth.metrics import audit_and_aggregate
from editguard_depth.run import ROOT, verify_protocol


RECEIPTS = {f"{phase}-{model}.json" for phase in ("profile", "development", "final")
            for model in ("reasonke", "instruction")}
ALLOWED = RECEIPTS | {"protocol.json", "results.json"}
FROZEN_INPUTS = {f"{phase}-{model}.json" for phase in ("profile", "development")
                 for model in ("reasonke", "instruction")}
MAX_FILE_BYTES = 32 * 1024**2
MAX_TOTAL_BYTES = 64 * 1024**2


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Repeated JSON key in depth evidence: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("Non-finite JSON number in depth evidence: " + value)

    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def _read_archive(bundle):
    with zipfile.ZipFile(bundle) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if (not entries or len(names) != len(set(names)) or set(names) - ALLOWED
                or any(entry.is_dir() or entry.flag_bits & 1
                       or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                       or stat.S_IFMT(entry.external_attr >> 16) not in (0, stat.S_IFREG)
                       or entry.file_size > MAX_FILE_BYTES for entry in entries)
                or sum(entry.file_size for entry in entries) > MAX_TOTAL_BYTES):
            raise ValueError("Depth checkpoint has empty, unknown, duplicate, linked, encrypted or oversized members")
        payload = {entry.filename: archive.read(entry) for entry in entries}
    for raw in payload.values():
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("Depth checkpoint member exceeds the size bound")
        _json(raw)
    return payload


def _normalise_report(report):
    # Audit paths identify the same input on different machines. All other
    # values, including receipt hashes, outcomes, timings and CIs must match.
    report = json.loads(json.dumps(report))
    for row in report.get("receipt_audit", []):
        row["receipt"] = row["receipt"].replace("\\", "/").rsplit("/", 1)[-1]
    return report


def _validate(payload, directory, config_path):
    for name, raw in payload.items():
        if name in RECEIPTS:
            phase, model = name.removesuffix(".json").split("-", 1)
            parsed = _json(raw)
            if parsed.get("plan", {}).get("phase") != phase or parsed.get("plan", {}).get("model") != model:
                raise ValueError("Depth filename and assigned model/phase differ: " + name)
    protocol = directory / "protocol.json" if "protocol.json" in payload else None
    if protocol is not None:
        if not FROZEN_INPUTS <= payload.keys():
            raise ValueError("A depth protocol requires all four bound profile/development receipts")
        verify_protocol(protocol, config_path)
    if any(name.startswith("final-") for name in payload) and protocol is None:
        raise ValueError("Final depth ledgers require their frozen protocol and bound inputs")
    progress = {}
    for name in sorted(payload.keys() & RECEIPTS):
        report = audit_and_aggregate([directory / name], config_path=config_path, protocol=protocol)
        parsed = _json(payload[name])
        progress[name] = {"status": parsed["status"], "assigned": report["assigned_records"],
                          "recorded": sum(row["outcome"] != "unexecuted" for row in parsed["records"]),
                          "unexecuted": sum(row["outcome"] == "unexecuted" for row in parsed["records"])}
    if "results.json" in payload:
        final_names = {"final-reasonke.json", "final-instruction.json"}
        if not final_names <= payload.keys() or any(_json(payload[name])["status"] != "completed" for name in final_names):
            raise ValueError("A published depth scorecard requires both completed final ledgers")
        expected = audit_and_aggregate([directory / name for name in ("final-reasonke.json", "final-instruction.json")],
                                       config_path=config_path, protocol=protocol)
        if _normalise_report(_json(payload["results.json"])) != _normalise_report(expected):
            raise ValueError("Depth results differ from the audited final ledgers")
    return progress


def restore(bundle, directory, *, preserve_existing=False, config_path=None, recovery_note=""):
    """Validate the complete union before publishing; never overwrite a ledger."""
    bundle, directory = Path(bundle), Path(directory).absolute()
    if directory.is_symlink():
        raise ValueError("Depth restore destination must not be a symbolic link")
    directory = directory.resolve()
    config_path = Path(config_path or ROOT / "configs/depth-evaluation.json").resolve()
    destination_existed = directory.exists()
    if destination_existed and (not directory.is_dir() or not preserve_existing):
        raise ValueError("Use a new depth destination or explicitly --preserve-existing")
    if bundle.stat().st_size > MAX_TOTAL_BYTES + 2 * 1024**2:
        raise ValueError("Depth checkpoint archive exceeds the compressed size bound")
    archive_bytes = bundle.read_bytes()
    incoming = _read_archive(io.BytesIO(archive_bytes))
    existing = {}
    if directory.exists():
        for name in sorted(ALLOWED):
            path = directory / name
            if path.is_symlink():
                raise ValueError("An existing depth evidence path is a symbolic link")
            if path.exists():
                if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                    raise ValueError("Existing depth evidence is not a bounded file")
                existing[name] = path.read_bytes()
                _json(existing[name])
    if any(name in existing and raw != existing[name] for name, raw in incoming.items()):
        raise ValueError("Existing depth outcomes differ; no existing file was replaced")
    payload = {**existing, **incoming}
    if sum(map(len, payload.values())) > MAX_TOTAL_BYTES:
        raise ValueError("Combined depth evidence exceeds the total size bound")
    # This audit staging directory is outside the checkout and is never used as
    # a model/data source. The auditor still binds the actual current sources.
    with tempfile.TemporaryDirectory(prefix="editguard-depth-audit-") as temporary:
        audit_directory = Path(temporary)
        for name, raw in payload.items():
            (audit_directory / name).write_bytes(raw)
        progress = _validate(payload, audit_directory, config_path)
    directory.parent.mkdir(parents=True, exist_ok=True)
    new_names = sorted(incoming.keys() - existing.keys())
    with tempfile.TemporaryDirectory(prefix=".editguard-depth-restore-", dir=directory.parent) as temporary:
        staging_root = Path(temporary).resolve()
        if staging_root.parent != directory.parent:
            raise ValueError("Depth staging directory escaped the requested destination parent")
        staged = staging_root / "ledgers"
        staged.mkdir()
        for name, raw in payload.items():
            with (staged / name).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        if directory.exists():
            if not destination_existed:
                raise ValueError("Depth destination appeared during restore; nothing was overwritten")
            # Detect changes since the audit, then use hard links for exclusive
            # atomic file creation. A concurrent writer is never overwritten.
            if any((directory / name).is_symlink() or not (directory / name).is_file()
                   or (directory / name).read_bytes() != raw for name, raw in existing.items()):
                raise ValueError("Existing depth evidence changed during restore")
            created = []
            try:
                for name in new_names:
                    os.link(staged / name, directory / name)
                    created.append(name)
            except BaseException:
                for name in reversed(created):
                    target = directory / name
                    if target.is_file() and not target.is_symlink() and target.read_bytes() == incoming[name]:
                        target.unlink()
                raise
            publication = "exclusive_per_file_merge; validated before publication"
        else:
            # Both paths have been resolved/checked against the explicitly
            # requested parent before this directory publication.
            if staged.resolve().parent != staging_root:
                raise ValueError("Depth publication source escaped staging")
            staged.replace(directory)
            publication = "atomic_new_directory"
    receipt = {"schema": "editguard-depth-recovery-v1", "status": "restored",
               "created_utc": datetime.now(timezone.utc).isoformat(), "directory": str(directory),
               "archive_sha256": _sha(archive_bytes),
               "files": {name: _sha(raw) for name, raw in sorted(payload.items())},
               "created_files": new_names, "preserved_existing_files": sorted(existing),
               "progress": progress, "publication": publication, "model_calls": 0,
               "missing_development_receipts": sorted({"development-reasonke.json", "development-instruction.json"} - payload.keys()),
               "prior_unsaved_remote_outcomes": "Unknown; this restore cannot reconstruct missing remote outcomes",
               "missing_receipts_mean_no_generations": False, "recovery_note": recovery_note,
               "notice": "Only enclosed and already-present validated evidence is restored. Completed wrong answers, abstentions and execution errors are retained unchanged. Run exact-plan --resume only against an idle destination."}
    # Keep this note outside the ledger directory: the notebook exports *.json
    # there and checkpoint inputs are deliberately restricted to eight names.
    receipt_path = directory.parent / (directory.name + "-recovery-" + uuid4().hex + ".json")
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return {**receipt, "recovery_receipt": str(receipt_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preserve-existing", action="store_true")
    parser.add_argument("--recovery-note", default="")
    args = parser.parse_args(argv)
    print(json.dumps(restore(args.bundle, args.directory, preserve_existing=args.preserve_existing,
                             config_path=args.config, recovery_note=args.recovery_note), indent=2))


if __name__ == "__main__":
    main()
