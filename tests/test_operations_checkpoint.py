"""Portable inference resume contracts; fixtures contain no generated final answers."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from editguard_benchmark.data import encoded
from editguard_benchmark.run import source_identity, tasks_for
from scripts.restore_operations_checkpoint import restore

ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def bundle(self, path, mutate=None):
        config_path = ROOT / "configs/operations-evaluation.json"
        config = json.loads(config_path.read_text())
        protocol = {"status": "frozen", "sources": source_identity(config_path)}
        data = {"protocol.json": json.dumps(protocol).encode()}
        proto_sha = hashlib.sha256(data["protocol.json"]).hexdigest()
        for phase, model in (("profile", "reasonke"), ("profile", "instruction"), ("final", "reasonke")):
            dataset, _, rows = tasks_for(config, model, phase)
            receipt = {"status": "interrupted", "dataset": dataset, "records": rows,
                "plan": {"phase": phase, "model": model, "sources": protocol["sources"],
                         "task_ids_sha256": hashlib.sha256(encoded([row["id"] for row in rows])).hexdigest(),
                         "protocol_sha256": proto_sha if phase == "final" else None}}
            data[phase + "-" + model + ".json"] = json.dumps(receipt).encode()
        manifest = {"schema": "editguard-paused-inference-v1", "protocol_sha256": proto_sha,
                    "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in data.items()}}
        data["pause-manifest.json"] = json.dumps(manifest).encode()
        if mutate: mutate(data)
        with zipfile.ZipFile(path, "w") as archive:
            for name, raw in data.items(): archive.writestr(name, raw)
        return data

    def test_restore_preserves_receipt_bytes_without_loading_or_answering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data = self.bundle(root / "checkpoint.zip")
            result = restore(root / "checkpoint.zip", root / "new")
            self.assertEqual(result["model_generations"], 0)
            self.assertEqual(result["progress"]["reasonke"]["remaining"], 784)
            self.assertTrue(all((root / "new" / name).read_bytes() == raw for name, raw in data.items()))
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                restore(root / "checkpoint.zip", root / "new")

    def test_tampered_or_path_traversal_checkpoint_cannot_partially_restore(self):
        for mutate in (lambda data: data.update({"../outside.json": b"{}"}),
                       lambda data: data.update({"final-reasonke.json": b"{}"})):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); self.bundle(root / "checkpoint.zip", mutate)
                with self.assertRaises(ValueError): restore(root / "checkpoint.zip", root / "new")
                self.assertFalse((root / "new").exists())
                self.assertFalse((root / "outside.json").exists())

    def test_resealed_different_source_is_not_authorized_by_checksum(self):
        def mutate(data):
            protocol = json.loads(data["protocol.json"])
            protocol["sources"]["editguard_benchmark/run.py"] = "0" * 64
            data["protocol.json"] = json.dumps(protocol).encode()
            manifest = json.loads(data["pause-manifest.json"])
            manifest["protocol_sha256"] = hashlib.sha256(data["protocol.json"]).hexdigest()
            manifest["files"]["protocol.json"] = manifest["protocol_sha256"]
            data["pause-manifest.json"] = json.dumps(manifest).encode()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.bundle(root / "checkpoint.zip", mutate)
            with self.assertRaisesRegex(ValueError, "resume refused"):
                restore(root / "checkpoint.zip", root / "new")
            self.assertFalse((root / "new").exists())


if __name__ == "__main__": unittest.main()
