"""Exact data recovery checks with tiny synthetic archives; no model imports."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from visionpatch.restore_data import restore_manifest


class RestoreDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = self.root / "source.zip"
        self.manifest = self.root / "frozen.json"
        self.output = self.root / "restored"
        self.photos = {"cup/image_0001.jpg": b"first declared photograph",
                       "camera/image_0002.jpg": b"second declared photograph"}
        self.members = {"101_ObjectCategories/" + name: payload for name, payload in self.photos.items()}
        # Neither unrelated source photos nor arbitrary archive paths are extracted.
        self.members.update({"101_ObjectCategories/cup/unlisted.jpg": b"not selected",
                             "../../outside.txt": b"not extracted"})
        self.metadata = {"dataset_id": "synthetic-recovery-only", "classes": {"cup": "cup", "camera": "camera"},
                         "source": {}, "images": [
                             {"image_id": name, "class_id": name.split("/")[0], "label": name.split("/")[0],
                              "path": "images/" + name, "sha256": hashlib.sha256(payload).hexdigest()}
                             for name, payload in self.photos.items()]}
        self.write_archive()
        self.save_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def write_archive(self):
        buffer = BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, payload in self.members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, BytesIO(payload))
        with zipfile.ZipFile(self.archive, "w") as outer:
            outer.writestr("caltech-101/101_ObjectCategories.tar.gz", buffer.getvalue())
        payload = self.archive.read_bytes()
        self.metadata["source"] = {"archive_md5": hashlib.md5(payload).hexdigest(),
                                   "archive_sha256": hashlib.sha256(payload).hexdigest()}

    def save_manifest(self):
        # Exact serialized bytes matter to the published study's recorded hashes.
        self.manifest.write_bytes((json.dumps(self.metadata, indent=2) + "\n").replace("\n", "\r\n").encode())

    def restore(self):
        return restore_manifest(self.archive, self.manifest, self.output)

    def test_restores_only_listed_photographs_and_exact_manifest_bytes_idempotently(self):
        result = self.restore()
        self.assertEqual(result["restored_photos"], 2)
        self.assertTrue(result["no_model_execution"])
        self.assertEqual((self.output / "manifest.json").read_bytes(), self.manifest.read_bytes())
        self.assertEqual(result["manifest_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        for name, payload in self.photos.items():
            self.assertEqual((self.output / "images" / name).read_bytes(), payload)
        self.assertEqual(len(list(self.output.rglob("*.jpg"))), 2)
        self.assertFalse((self.root / "outside.txt").exists())
        again = self.restore()
        self.assertEqual(again["restored_photos"], 0)
        self.assertEqual(again["matching_existing_photos"], 2)

    def test_rejects_either_wrong_archive_digest_before_any_output(self):
        original = dict(self.metadata["source"])
        for key, length in (("archive_md5", 32), ("archive_sha256", 64)):
            self.metadata["source"] = {**original, key: "0" * length}
            self.save_manifest()
            with self.subTest(digest=key), self.assertRaisesRegex(ValueError, "recorded checksums"):
                self.restore()
            self.assertFalse(self.output.exists())

    def test_wrong_or_missing_source_photo_prevents_partial_restoration(self):
        self.members["101_ObjectCategories/camera/image_0002.jpg"] = b"different pixels"
        self.write_archive()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "frozen manifest"):
            self.restore()
        self.assertFalse(self.output.exists())
        self.members.pop("101_ObjectCategories/camera/image_0002.jpg")
        self.write_archive()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "missing a declared"):
            self.restore()
        self.assertFalse(self.output.exists())

    def test_existing_photo_mismatch_is_preserved_and_no_other_photo_is_written(self):
        target = self.output / self.metadata["images"][0]["path"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"preserve existing content")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.restore()
        self.assertEqual(target.read_bytes(), b"preserve existing content")
        self.assertFalse((self.output / self.metadata["images"][1]["path"]).exists())

    def test_destination_manifest_mismatch_is_preserved(self):
        self.output.mkdir()
        target = self.output / "manifest.json"
        target.write_bytes(b"an existing split")
        with self.assertRaisesRegex(ValueError, "Destination manifest differs"):
            self.restore()
        self.assertEqual(target.read_bytes(), b"an existing split")
        self.assertFalse((self.output / "images").exists())

    def test_unsafe_paths_labels_and_duplicate_sources_are_rejected(self):
        original = json.loads(json.dumps(self.metadata))
        mutations = [lambda m: m["images"][0].update(path="../../outside.jpg"),
                     lambda m: m["images"][0].update(path="images/cup/../../outside.jpg"),
                     lambda m: m["images"][0].update(image_id="cup/../../outside.jpg"),
                     lambda m: m["images"][0].update(path="C:\\outside.jpg"),
                     lambda m: m["images"][0].update(label="incorrect"),
                     lambda m: m["images"].append(dict(m["images"][0]))]
        for mutate in mutations:
            self.metadata = json.loads(json.dumps(original))
            mutate(self.metadata)
            self.save_manifest()
            with self.subTest(manifest=self.metadata), self.assertRaises(ValueError):
                self.restore()
            self.assertFalse(self.output.exists())

    def test_symlink_cannot_redirect_recovery_outside_output(self):
        self.output.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        try:
            (self.output / "images").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Host does not allow creating a directory symbolic link")
        with self.assertRaisesRegex(ValueError, "escapes|symbolic"):
            self.restore()
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
