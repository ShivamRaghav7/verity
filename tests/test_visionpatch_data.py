"""Leakage, provenance and matched-transform checks without model downloads."""

import copy
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from visionpatch import data

try:
    from PIL import Image, ImageChops
except ImportError:
    Image = None


def _row(identifier="cup/1.jpg", digest="a" * 64, split="support", category="cup"):
    return {"image_id": identifier, "sha256": digest, "split": split,
            "class_id": category, "label": data.CLASSES[category], "path": identifier}


class ManifestLeakageTests(unittest.TestCase):
    def test_exact_original_content_cannot_cross_splits(self):
        first = _row()
        second = _row("cup/2.jpg", split="probe")
        with self.assertRaisesRegex(ValueError, "repeats"):
            data.audit_manifest({"images": [first, second]})

    def test_duplicate_source_identifier_rejected_even_with_different_bytes(self):
        with self.assertRaisesRegex(ValueError, "repeats"):
            data.audit_manifest({"images": [_row(), _row(digest="b" * 64, split="retention")]})

    def test_distinct_originals_are_valid_without_mutating_manifest(self):
        manifest = {"images": [_row(), _row("cup/2.jpg", "b" * 64, "probe")]}
        before = copy.deepcopy(manifest)
        data.audit_manifest(manifest)
        self.assertEqual(manifest, before)

    def test_unknown_split_or_source_category_rejected(self):
        for field, value in (("split", "final_test"), ("class_id", "not-in-selected-classes")):
            row = _row()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                data.audit_manifest({"images": [row]})


class ArchivePreparationTests(unittest.TestCase):
    @staticmethod
    def make_archive(path, *, duplicate_across_classes=False):
        """Mock archive bytes test selection/hashing, not JPEG decoding."""
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as inner:
            for category in ("cup", "camera"):
                for index in range(4):
                    payload = f"mock-photograph-{category}-{index}".encode()
                    if duplicate_across_classes:
                        payload = f"shared-mock-photograph-{index}".encode()
                    member = tarfile.TarInfo(f"101_ObjectCategories/{category}/image_{index:04d}.jpg")
                    member.size = len(payload)
                    inner.addfile(member, io.BytesIO(payload))
            # A traversal member is never extracted into the output directory.
            member = tarfile.TarInfo("../../outside.jpg")
            member.size = 6
            inner.addfile(member, io.BytesIO(b"unsafe"))
        with zipfile.ZipFile(path, "w") as outer:
            outer.writestr("caltech-101/101_ObjectCategories.tar.gz", buffer.getvalue())
            # Official archives may contain this resource-fork pseudo-file.
            outer.writestr("__MACOSX/caltech-101/._101_ObjectCategories.tar.gz", b"not a tarball")

    def test_deterministic_hash_partition_and_frozen_manifest_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            self.make_archive(archive)
            with (patch.object(data, "SOURCE_MD5", data.file_hash(archive, "md5")),
                  patch.object(data, "CLASSES", {"cup": "cup", "camera": "camera"}),
                  patch.object(data, "COUNTS", {"support": 1, "retention": 1, "probe": 1})):
                first = data.prepare_dataset(archive, root / "first")
                second = data.prepare_dataset(archive, root / "second")
                self.assertEqual(first, second)
                self.assertEqual(len(first["images"]), 6)
                self.assertEqual(len({row["sha256"] for row in first["images"]}), 6)
                self.assertFalse(first["final_test"])
                for category in data.CLASSES:
                    rows = [row for row in first["images"] if row["class_id"] == category]
                    self.assertEqual([row["split"] for row in rows], ["support", "retention", "probe"])
                    self.assertEqual([row["sha256"] for row in rows], sorted(row["sha256"] for row in rows))
                self.assertEqual(data.prepare_dataset(archive, root / "first"), first)
                self.assertFalse((root / "outside.jpg").exists())
                photograph = root / "first" / first["images"][0]["path"]
                photograph.write_bytes(b"changed after manifest freeze")
                with self.assertRaisesRegex(ValueError, "differs"):
                    data.prepare_dataset(archive, root / "first")

    def test_cross_category_duplicate_bytes_cannot_silently_enter_multiple_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            self.make_archive(archive, duplicate_across_classes=True)
            with (patch.object(data, "SOURCE_MD5", data.file_hash(archive, "md5")),
                  patch.object(data, "CLASSES", {"cup": "cup", "camera": "camera"}),
                  patch.object(data, "COUNTS", {"support": 1, "retention": 1, "probe": 1})):
                with self.assertRaisesRegex(ValueError, "Insufficient distinct"):
                    data.prepare_dataset(archive, root / "prepared")

    def test_source_checksum_failure_prevents_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            archive.write_bytes(b"wrong source")
            with self.assertRaisesRegex(ValueError, "checksum"):
                data.prepare_dataset(archive, root / "prepared")
            self.assertFalse((root / "prepared").exists())


@unittest.skipIf(Image is None, "Optional matched-transform checks require Pillow")
class MatchedVariantTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (320, 200), "navy")

    def test_variant_is_deterministic_and_does_not_mutate_source(self):
        before = self.image.tobytes()
        first, first_box = data.render_variant(self.image, "distractor")
        second, second_box = data.render_variant(self.image, "distractor")
        self.assertEqual(first.size, (224, 224))
        self.assertEqual(first.mode, "RGB")
        self.assertEqual(first_box, second_box)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(self.image.tobytes(), before)

    def test_text_and_blank_share_exact_geometry_and_differ_only_inside_patch(self):
        for position in ("top", "bottom"):
            with self.subTest(position=position):
                blank, box = data.render_variant(self.image, "blank", position=position)
                attacked, other_box = data.render_variant(self.image, "distractor", position=position)
                clean, _ = data.render_variant(self.image, "clean", position=position)
                self.assertEqual(box, other_box)
                delta = ImageChops.difference(blank, attacked).getbbox()
                self.assertIsNotNone(delta, "printed word must change some pixels")
                x0, y0, x1, y1 = box
                # Pillow rectangle includes its maximum pixel coordinates.
                self.assertGreaterEqual(delta[0], x0)
                self.assertGreaterEqual(delta[1], y0)
                self.assertLessEqual(delta[2], x1 + 1)
                self.assertLessEqual(delta[3], y1 + 1)
                for image in (blank, attacked):
                    restored = image.copy()
                    restored.paste(clean.crop((x0, y0, x1 + 1, y1 + 1)), (x0, y0))
                    self.assertEqual(restored.tobytes(), clean.tobytes())

    def test_clean_is_position_independent_and_blank_ignores_word(self):
        clean_top, _ = data.render_variant(self.image, "clean", position="top")
        clean_bottom, _ = data.render_variant(self.image, "clean", position="bottom")
        self.assertEqual(clean_top.tobytes(), clean_bottom.tobytes())
        blank_a, _ = data.render_variant(self.image, "blank", word="camera")
        blank_b, _ = data.render_variant(self.image, "blank", word="watch")
        self.assertEqual(blank_a.tobytes(), blank_b.tobytes())

    def test_unknown_variant_and_position_rejected(self):
        for kwargs in ({"variant": "uncontrolled"}, {"variant": "blank", "position": "middle"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                data.render_variant(self.image, **kwargs)


if __name__ == "__main__":
    unittest.main()
