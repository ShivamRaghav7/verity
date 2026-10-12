"""Finite archive checks on the pending development packet, never test data."""
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.pilot_bundle import unpack

ROOT = Path(__file__).resolve().parents[1]


class PilotBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = ROOT / "data/pilots/pilot-v1-source.zip"
        self.manifest = ROOT / "reports/pilot-manifest.json"

    def test_exact_bundle_reconstructs_manifest_and_preserves_existing_directory(self):
        target = self.root / "pilot"
        result = unpack(self.bundle, target, self.manifest)
        self.assertEqual(result["files"], 9)
        self.assertFalse(result["test_files_included"])
        self.assertEqual((target / "manifest.json").read_bytes(), self.manifest.read_bytes())
        with self.assertRaisesRegex(ValueError, "fresh output"):
            unpack(self.bundle, target, self.manifest)

    def test_traversal_or_duplicate_or_test_entries_fail_before_writing(self):
        with zipfile.ZipFile(self.bundle) as source:
            original = {entry.filename: source.read(entry) for entry in source.infolist()}
        for extra in ("../outside.txt", "scope_test.jsonl", "reviews.json"):
            path = self.root / "bad.zip"
            with zipfile.ZipFile(path, "w") as modified:
                for name, content in original.items():
                    modified.writestr(name, content)
                modified.writestr(extra, "fixture")
            target = self.root / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "unsupported"):
                unpack(path, target, self.manifest)
            self.assertFalse(target.exists())

    def test_rehashed_archive_does_not_bypass_manifest_file_integrity(self):
        path = self.root / "tampered.zip"
        with zipfile.ZipFile(self.bundle) as source, zipfile.ZipFile(path, "w") as modified:
            for entry in source.infolist():
                content = source.read(entry)
                modified.writestr(entry.filename, b"tampered" if entry.filename == "scope_train.jsonl" else content)
        target = self.root / "must-not-exist"
        with self.assertRaisesRegex(ValueError, "integrity failed"):
            unpack(path, target, self.manifest)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
