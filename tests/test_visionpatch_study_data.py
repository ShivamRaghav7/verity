"""Leakage, frozen-source and matched-rendering contracts for the v2 study."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from visionpatch import study_data, study_protocol

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "visionpatch-study-v2.json"

try:
    from PIL import Image, ImageChops
except ImportError:
    Image = None


class StudySourceContractTests(unittest.TestCase):
    def make_fixture(self, root):
        """Byte-level fixtures test grouping without decoding JPEGs or loading ML."""
        archive = root / "source.zip"
        buffer = io.BytesIO()
        previous_development, previous_final = [], []
        with tarfile.open(fileobj=buffer, mode="w:gz") as inner:
            for category in study_data.CLASSES:
                candidates = []
                for index in range(40):
                    filename = f"image_{index:04d}.jpg"
                    payload = f"mock-{category}-{index}".encode()
                    member = tarfile.TarInfo(f"101_ObjectCategories/{category}/{filename}")
                    member.size = len(payload)
                    inner.addfile(member, io.BytesIO(payload))
                    candidates.append((hashlib.sha256(payload).hexdigest(), filename))
                for index, (digest, filename) in enumerate(sorted(candidates)[:28]):
                    split = "support" if index < 2 else "retention" if index < 10 else "probe" if index < 18 else "final"
                    row = {"image_id": f"{category}/{filename}", "class_id": category,
                           "label": study_data.CLASSES[category], "split": split,
                           "sha256": digest, "path": f"images/{category}/{filename}"}
                    (previous_final if split == "final" else previous_development).append(row)
            member = tarfile.TarInfo("../../outside.jpg")
            member.size = 6
            inner.addfile(member, io.BytesIO(b"unsafe"))
        with zipfile.ZipFile(archive, "w") as outer:
            outer.writestr("caltech-101/101_ObjectCategories.tar.gz", buffer.getvalue())
            outer.writestr("__MACOSX/._101_ObjectCategories.tar.gz", b"resource-fork")
        source = {"archive_sha256": study_data.file_hash(archive), "license": "fixture"}
        paths = []
        for name, rows, final in (("development", previous_development, False), ("final", previous_final, True)):
            target = root / f"{name}.json"
            target.write_text(json.dumps({"classes": study_data.CLASSES, "final_test": final,
                                          "source": source, "images": rows}), encoding="utf-8")
            paths.append(target)
        return archive, paths[0], paths[1]

    def prepare(self, archive, development, final, output):
        with (patch.object(study_data, "SOURCE_MD5", study_data.file_hash(archive, "md5")),
              patch.object(study_data, "font_records", return_value={"fixture": "stable-font-hash"})):
            return study_data.prepare_study_dataset(archive, development, final, output, CONFIG_PATH)

    def test_fresh_splits_exclude_all_224_v1_sources_and_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, development, final = self.make_fixture(root)
            first = self.prepare(archive, development, final, root / "first")
            second = self.prepare(archive, development, final, root / "second")
            self.assertEqual(first, second)
            self.assertEqual(len(first["previous_originals"]), 224)
            self.assertEqual(len(first["images"]), 160)
            old = {row["sha256"] for row in first["previous_originals"]}
            for split, expected in (("train", 80), ("validation", 32), ("final", 48)):
                rows = [row for row in first["images"] if row["split"] == split]
                self.assertEqual(len(rows), expected)
                hashes = {row["sha256"] for row in rows}
                if split == "train":
                    self.assertTrue(hashes <= old)
                else:
                    self.assertTrue(hashes.isdisjoint(old))
            self.assertFalse((root / "outside.jpg").exists())
            self.assertEqual(self.prepare(archive, development, final, root / "first"), first)

    def test_repeated_original_content_is_rejected_even_under_a_new_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, development, final = self.make_fixture(root)
            manifest = self.prepare(archive, development, final, root / "dataset")
            modified = copy.deepcopy(manifest)
            source = modified["images"][0]
            modified["images"].append({**source, "image_id": "cup/new-name.jpg"})
            with self.assertRaisesRegex(ValueError, "repeats"):
                study_data.audit_study_manifest(modified, root / "dataset")

    def test_previous_probe_or_final_cannot_be_relabelled_as_fresh_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, development, final = self.make_fixture(root)
            manifest = self.prepare(archive, development, final, root / "dataset")
            modified = copy.deepcopy(manifest)
            old = next(row for row in modified["previous_originals"] if row["split"] == "final")
            evaluation = next(row for row in modified["images"] if row["split"] == "final")
            evaluation.update({"image_id": old["image_id"], "sha256": old["sha256"]})
            with self.assertRaisesRegex(ValueError, "overlap"):
                study_data.audit_study_manifest(modified, root / "dataset")

    def test_source_tampering_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, development, final = self.make_fixture(root)
            manifest = self.prepare(archive, development, final, root / "dataset")
            modified = copy.deepcopy(manifest)
            modified["images"][0]["path"] = "../outside.jpg"
            with self.assertRaisesRegex(ValueError, "escapes"):
                study_data.audit_study_manifest(modified, root / "dataset")
            (root / "dataset" / manifest["images"][0]["path"]).write_bytes(b"changed after freezing")
            with self.assertRaisesRegex(ValueError, "differs"):
                self.prepare(archive, development, final, root / "dataset")

    def test_nonempty_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, development, final = self.make_fixture(root)
            output = root / "existing"
            output.mkdir()
            (output / "keep.txt").write_text("preserved", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "nonempty"):
                self.prepare(archive, development, final, output)
            self.assertEqual((output / "keep.txt").read_text(), "preserved")


class StudyProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = study_protocol.load_study_config(CONFIG_PATH)

    def test_paraphernalia_and_counterfactual_words_never_change_group_split(self):
        manifest = {"images": [{"image_id": "cup/final.jpg", "class_id": "cup", "label": "cup",
                                 "split": "final", "role": "evaluation", "path": "images/cup/final.jpg"}]}
        rows = study_protocol.iter_photo_tasks(manifest, "final", self.config)
        self.assertEqual(len(rows), 13)
        self.assertEqual(len({row["task_id"] for row in rows}), 13)
        self.assertEqual({row["split"] for row in rows}, {"final"})
        self.assertEqual({row["image_id"] for row in rows}, {"cup/final.jpg"})
        self.assertEqual({row["label"] for row in rows}, {"cup"})
        self.assertEqual(sum(row["variant"] == "clean" for row in rows), 1)

    def test_text_templates_are_split_before_all_eight_word_variants(self):
        seen = set()
        for split, expected in (("train", 32), ("validation", 16), ("final", 24)):
            rows = study_protocol.text_task_rows(split, self.config)
            self.assertEqual(len(rows), expected)
            templates = {row["template_id"] for row in rows}
            self.assertFalse(templates & seen)
            seen.update(templates)
            self.assertEqual({row["label"] for row in rows}, set(study_data.CLASSES.values()))
            for template in templates:
                self.assertEqual(sum(row["template_id"] == template for row in rows), 8)

    def test_changed_text_grouping_or_expanded_model_grid_is_rejected(self):
        for mutation in ("template", "candidate", "edit"):
            with tempfile.TemporaryDirectory() as directory:
                config = copy.deepcopy(self.config)
                if mutation == "template":
                    config["text_template_ids"]["final"][0] = config["text_template_ids"]["train"][0]
                elif mutation == "candidate":
                    config["editing_candidates"].append(copy.deepcopy(config["editing_candidates"][0]))
                else:
                    config["edit_words"].append("cup")
                path = Path(directory) / "changed.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    study_protocol.load_study_config(path)


@unittest.skipIf(Image is None, "Pillow is an optional vision dependency")
class StudyRenderTests(unittest.TestCase):
    def setUp(self):
        self.source = Image.new("RGB", (320, 200), "navy")
        self.fonts = study_protocol.resolve_fonts()

    def test_all_fonts_and_positions_have_exactly_matched_blank_geometry(self):
        original = self.source.tobytes()
        for font in self.fonts.values():
            for position in ("top", "center", "bottom"):
                blank, box = study_protocol.render_study_variant(self.source, "blank", font_path=font, position=position)
                attacked, other_box = study_protocol.render_study_variant(self.source, "distractor", font_path=font, position=position)
                self.assertEqual(box, other_box)
                delta = ImageChops.difference(blank, attacked).getbbox()
                self.assertIsNotNone(delta)
                self.assertGreaterEqual(delta[0], box[0])
                self.assertGreaterEqual(delta[1], box[1])
                self.assertLessEqual(delta[2], box[2] + 1)
                self.assertLessEqual(delta[3], box[3] + 1)
        self.assertEqual(self.source.tobytes(), original)

    def test_controlled_text_cards_are_deterministic_and_not_identical_across_templates(self):
        kwargs = {"font_path": self.fonts["train_sans"], "position": "bottom"}
        first = study_protocol.render_text_card("camera", 0, **kwargs)
        self.assertEqual(first.tobytes(), study_protocol.render_text_card("camera", 0, **kwargs).tobytes())
        self.assertNotEqual(first.tobytes(), study_protocol.render_text_card("camera", 1, **kwargs).tobytes())
        self.assertNotEqual(first.tobytes(), study_protocol.render_text_card("watch", 0, **kwargs).tobytes())


if __name__ == "__main__":
    unittest.main()
