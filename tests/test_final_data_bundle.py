import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.final_data_bundle import FILES, create_bundle, digest, restore_bundle


class FinalBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "original"
        self.source.mkdir()
        # Deliberately invalid JSONL: bundling must never parse question contents.
        self.payload = {name: b"opaque question bytes\r\n" for name in FILES}
        self.payload["manifest.json"] = json.dumps({"dataset_id": "editguard-controlled-v1",
            "files": {name: digest(raw) for name, raw in self.payload.items()}}).encode()
        self.expected = digest(self.payload["manifest.json"])
        for name, raw in self.payload.items():
            (self.source / name).write_bytes(raw)

    def test_exact_bytes_reconstruct_and_deterministic_archive(self):
        first, second = self.root / "a.zip", self.root / "b.zip"
        result = create_bundle(self.source, first, expected_manifest=self.expected)
        create_bundle(self.source, second, expected_manifest=self.expected)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertFalse(result["question_content_deserialized"])
        destination = self.root / "restored"
        restore_bundle(first, destination, expected_manifest=self.expected)
        self.assertEqual(self.payload, {p.name: p.read_bytes() for p in destination.iterdir()})

    def test_tampered_source_cannot_publish(self):
        (self.source / "scope_test.jsonl").write_bytes(b"changed")
        output = self.root / "a.zip"
        with self.assertRaises(ValueError):
            create_bundle(self.source, output, expected_manifest=self.expected)
        self.assertFalse(output.exists())

    def test_existing_paths_are_preserved(self):
        output = self.root / "a.zip"
        create_bundle(self.source, output, expected_manifest=self.expected)
        before = output.read_bytes()
        with self.assertRaises(ValueError):
            create_bundle(self.source, output, expected_manifest=self.expected)
        with self.assertRaises(ValueError):
            restore_bundle(output, self.source, expected_manifest=self.expected)
        self.assertEqual(before, output.read_bytes())

    def test_unknown_duplicate_and_escaping_entries_fail_before_output(self):
        for index, extra in enumerate(("../escape", "extra", "scope_test.jsonl")):
            archive = self.root / f"invalid-{index}.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                for name, raw in self.payload.items():
                    stream.writestr(name, raw)
                stream.writestr(extra, b"malicious")
            output = self.root / f"output-{index}"
            with self.assertRaises(ValueError):
                restore_bundle(archive, output, expected_manifest=self.expected)
            self.assertFalse(output.exists())

    def test_tampered_archive_manifest_is_rejected(self):
        archive = self.root / "wrong.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            for name, raw in self.payload.items():
                stream.writestr(name, raw + b" " if name == "manifest.json" else raw)
        with self.assertRaises(ValueError):
            restore_bundle(archive, self.root / "out", expected_manifest=self.expected)


if __name__ == "__main__":
    unittest.main()
