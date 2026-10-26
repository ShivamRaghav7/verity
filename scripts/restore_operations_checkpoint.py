"""Validate a paused inference ZIP and restore it atomically, with no model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from editguard_benchmark.data import encoded
from editguard_benchmark.run import source_identity, tasks_for

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"profile-reasonke.json", "profile-instruction.json", "development-reasonke.json",
           "development-instruction.json", "final-reasonke.json", "final-instruction.json", "protocol.json"}


def restore(bundle, directory):
    bundle, directory = Path(bundle), Path(directory).resolve()
    if directory.exists():
        raise ValueError("Restore destination must be new; existing files are never overwritten")
    with zipfile.ZipFile(bundle) as archive:
        entries = archive.infolist()
        names = [row.filename for row in entries]
        if (len(names) != len(set(names)) or set(names) - (ALLOWED | {"pause-manifest.json"})
                or "pause-manifest.json" not in names or any(row.file_size > 32 * 1024**2 for row in entries)
                or sum(row.file_size for row in entries) > 64 * 1024**2):
            raise ValueError("Checkpoint has unknown, duplicate or oversized files")
        data = {name: archive.read(name) for name in names}
    manifest = json.loads(data["pause-manifest.json"])
    if manifest.get("schema") != "editguard-paused-inference-v1" or set(manifest["files"]) != set(data) - {"pause-manifest.json"}:
        raise ValueError("Checkpoint manifest does not identify all enclosed files")
    if not {"protocol.json", "profile-reasonke.json", "profile-instruction.json", "final-reasonke.json"} <= set(data):
        raise ValueError("Checkpoint omits a required frozen input or inference ledger")
    if any(hashlib.sha256(data[name]).hexdigest() != value for name, value in manifest["files"].items()):
        raise ValueError("Checkpoint file integrity mismatch")
    protocol_sha = hashlib.sha256(data["protocol.json"]).hexdigest()
    protocol = json.loads(data["protocol.json"])
    config_path = ROOT / "configs/operations-evaluation.json"
    sources = source_identity(config_path)
    if (protocol_sha != manifest["protocol_sha256"] or protocol.get("status") != "frozen"
            or protocol.get("sources") != sources):
        raise ValueError("Checkpoint source/data/configuration or frozen protocol differs; resume refused")
    config = json.loads(config_path.read_text())
    progress = {}
    for name, raw in data.items():
        if name in ("pause-manifest.json", "protocol.json"):
            continue
        phase, model = name.removesuffix(".json").split("-", 1)
        receipt = json.loads(raw)
        plan = receipt["plan"]
        dataset, _, tasks = tasks_for(config, model, phase)
        ids = [row["id"] for row in tasks]
        if (plan["model"] != model or plan["phase"] != phase or plan["sources"] != sources
                or receipt["dataset"] != dataset or [row["id"] for row in receipt["records"]] != ids
                or plan["task_ids_sha256"] != hashlib.sha256(encoded(ids)).hexdigest()
                or plan["protocol_sha256"] != (protocol_sha if phase == "final" else None)):
            raise ValueError("Checkpoint receipt input/task identity differs: " + name)
        if phase == "final":
            progress[model] = {"assigned": len(tasks), "recorded": sum(row["outcome"] != "unexecuted" for row in receipt["records"]),
                               "remaining": sum(row["outcome"] == "unexecuted" for row in receipt["records"])}
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".editguard-restore-", dir=directory.parent) as temporary:
        staged = Path(temporary) / "ledgers"
        staged.mkdir()
        for name, raw in data.items():
            (staged / name).write_bytes(raw)
        staged.replace(directory)
    return {"status": "restored", "directory": str(directory), "progress": progress,
            "protocol_sha256": protocol_sha, "model_generations": 0,
            "notice": "Only unexecuted requests resume; completed outcomes are preserved unchanged."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.bundle, args.directory), indent=2))


if __name__ == "__main__":
    main()
