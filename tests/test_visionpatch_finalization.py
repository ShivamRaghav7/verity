"""Final-split/seal safeguards; no model downloads or real held-out inference."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from visionpatch import finalize
from visionpatch.data import file_hash


class FinalDatasetPreparationTests(unittest.TestCase):
    def build_inputs(self, root):
        """Small archive mock tests hashing and exclusion, not image decoding."""
        archive = root / "source.zip"
        classes = {"cup": "cup", "camera": "camera"}
        payloads, development_rows = {}, []
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for category in classes:
                payloads[category] = []
                for index in range(5):
                    payload = f"mock-{category}-{index}".encode()
                    digest = hashlib.sha256(payload).hexdigest()
                    filename = f"image_{index}.jpg"
                    payloads[category].append((digest, filename, payload))
                    member = tarfile.TarInfo(f"101_ObjectCategories/{category}/{filename}")
                    member.size = len(payload)
                    tar.addfile(member, io.BytesIO(payload))
                excluded_hash, excluded_file, _ = sorted(payloads[category])[0]
                development_rows.append({"image_id": f"{category}/{excluded_file}", "class_id": category,
                                         "split": "probe", "sha256": excluded_hash})
        with zipfile.ZipFile(archive, "w") as outer:
            outer.writestr("caltech-101/101_ObjectCategories.tar.gz", buffer.getvalue())
            outer.writestr("__MACOSX/caltech-101/._101_ObjectCategories.tar.gz", b"resource fork")
        manifest = root / "development.json"
        manifest.write_text(json.dumps({"classes": classes, "final_test": False,
                                        "source": {"archive_sha256": file_hash(archive)},
                                        "images": development_rows}), encoding="utf-8")
        return archive, manifest, classes, payloads, development_rows

    def test_selection_excludes_all_development_hashes_before_choosing_next_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, development, classes, payloads, development_rows = self.build_inputs(root)
            with (patch.object(finalize, "CLASSES", classes),
                  patch.object(finalize, "SOURCE_MD5", file_hash(archive, "md5")),
                  patch.object(finalize, "load_clip") as loader):
                result = finalize.prepare_final_dataset(archive, development, root / "final", per_class=2)
                loader.assert_not_called()
                self.assertEqual(len(result["images"]), 4)
                self.assertTrue(result["final_test"])
                self.assertEqual(result["excluded_development_originals"], 2)
                excluded = {row["sha256"] for row in development_rows}
                self.assertFalse(excluded & {row["sha256"] for row in result["images"]})
                for category in classes:
                    chosen = [row["sha256"] for row in result["images"] if row["class_id"] == category]
                    self.assertEqual(chosen, [row[0] for row in sorted(payloads[category])[1:3]])
                self.assertTrue(all(row["split"] == "final" for row in result["images"]))

    def test_prepared_final_manifest_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, development, classes, _, _ = self.build_inputs(root)
            with (patch.object(finalize, "CLASSES", classes),
                  patch.object(finalize, "SOURCE_MD5", file_hash(archive, "md5"))):
                finalize.prepare_final_dataset(archive, development, root / "final", per_class=2)
                before = file_hash(root / "final" / "manifest.json")
                with self.assertRaises(FileExistsError):
                    finalize.prepare_final_dataset(archive, development, root / "final", per_class=2)
                self.assertEqual(file_hash(root / "final" / "manifest.json"), before)

    def test_frozen_final_content_overlap_tampering_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, development_path, classes, _, development_rows = self.build_inputs(root)
            development = json.loads(development_path.read_text(encoding="utf-8"))
            with (patch.object(finalize, "CLASSES", classes),
                  patch.object(finalize, "SOURCE_MD5", file_hash(archive, "md5"))):
                result = finalize.prepare_final_dataset(archive, development_path, root / "final", per_class=2)
                overlapping = copy.deepcopy(result)
                overlapping["images"][0]["sha256"] = development_rows[0]["sha256"]
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    finalize._final_audit(overlapping, development, root / "final", per_class=2)
                escaped = copy.deepcopy(result)
                escaped["images"][0]["path"] = "../outside.jpg"
                with self.assertRaisesRegex(ValueError, "escapes"):
                    finalize._final_audit(escaped, development, root / "final", per_class=2)
                photograph = root / "final" / result["images"][0]["path"]
                photograph.write_bytes(b"post-freeze modification")
                with self.assertRaisesRegex(ValueError, "sealed content"):
                    finalize._final_audit(result, development, root / "final", per_class=2)

    def test_insufficient_unseen_images_fail_before_writing_any_final_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, development, classes, _, _ = self.build_inputs(root)
            with (patch.object(finalize, "CLASSES", classes),
                  patch.object(finalize, "SOURCE_MD5", file_hash(archive, "md5"))):
                with self.assertRaisesRegex(ValueError, "Insufficient unseen"):
                    finalize.prepare_final_dataset(archive, development, root / "final", per_class=5)
                self.assertFalse((root / "final").exists())


class FinalProtocolSealTests(unittest.TestCase):
    def create_seal(self, root):
        bound = root / "input.bin"
        bound.write_bytes(b"frozen-before-evaluation")
        names = ("archive", "development_manifest", "final_manifest", "development_features",
                 "projection_patch", "projection_finetune", "code_data", "code_model",
                 "code_editor", "code_metrics", "code_finalize")
        seal = {"format_version": 1,
                "model": {"id": finalize.MODEL_ID, "revision": finalize.MODEL_REVISION},
                "evaluation": finalize.EVALUATION, "classes": finalize.CLASSES,
                "final_per_class": finalize.FINAL_PER_CLASS,
                "bindings": {name: {"path": str(bound), "sha256": file_hash(bound)} for name in names}}
        path = root / "seal.json"
        path.write_text(json.dumps(seal), encoding="utf-8")
        return path, bound, seal

    def test_seal_rejects_changed_input_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path, bound, seal = self.create_seal(Path(directory))
            with patch.object(finalize, "load_clip") as loader:
                self.assertEqual(finalize.load_and_validate_seal(path), seal)
                bound.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Sealed input changed"):
                    finalize.load_and_validate_seal(path)
                loader.assert_not_called()

    def test_seal_rejects_altered_configuration_size_or_missing_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, seal = self.create_seal(Path(directory))
            for field, value in (("final_per_class", 9), ("evaluation", {}),
                                 ("bindings", {}), ("format_version", True)):
                changed = copy.deepcopy(seal)
                changed[field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    finalize.load_and_validate_seal(path)

    def test_existing_report_is_rejected_before_loading_seal_or_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "final.json"
            report.write_text("preserved final results", encoding="utf-8")
            args = SimpleNamespace(report=report, artifacts=root / "artifacts")
            with (patch.object(finalize, "load_clip") as loader,
                  patch.object(finalize, "load_and_validate_seal") as seal_loader):
                with self.assertRaises(FileExistsError):
                    finalize.evaluate(args)
                loader.assert_not_called()
                seal_loader.assert_not_called()
            self.assertEqual(report.read_text(encoding="utf-8"), "preserved final results")


if __name__ == "__main__":
    unittest.main()
