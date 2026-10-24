"""RAG scope/prompt/calibration contracts using tiny vectors; no model imports."""
from contextlib import redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ke_repro.rag import (VisualCorrectionRetriever, calibration_operation, cosine,
                           main, select_threshold, threshold_metrics, validate_calibration_rows,
                           calibration_base_receipt)
from ke_repro import UPSTREAM_COMMIT
from ke_repro.smoke import digest
from scripts.prepare_vlkeb_rag import main as preparation_main, manifest_record, minilm_assets


class Encoder:
    def encode(self, text):
        return [0., 1.] if text == "neighbor?" else [1., 0.]


def calibration_rows():
    rows = []
    for index in range(2, 10):
        for category in ("direct", "paraphrase", "alternate_image", "text_locality", "visual_locality"):
            value = None if category == "text_locality" else 1. if category == "direct" else .8
            rows.append({"edit_id": f"train-{index:04d}", "category": category,
                         "applicable": category in ("direct", "paraphrase", "alternate_image"),
                         "score": value, "text_cosine": value, "visual_cosine": value})
    return rows


class RagTests(unittest.TestCase):
    def test_base_receipt_requires_verified_matching_manifest_and_pins_checkpoint_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "hparams/MEND/blip2.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("release config")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "assets": [
                {"path": "model.bin", "group": "core", "size_bytes": 1,
                 "url": "https://example.com/model", "sha256": "a" * 64},
                {"path": "vision.pth", "group": "core", "size_bytes": 2,
                 "url": "https://example.com/vision", "md5": "b" * 32}]}))
            args = SimpleNamespace(manifest=manifest, upstream=root)
            verified = {"assets_verified": True, "upstream_revision": UPSTREAM_COMMIT,
                        "asset_manifest_sha256": digest(manifest)}
            receipt = calibration_base_receipt(args, verified)
            self.assertEqual(receipt["upstream_config_sha256"], digest(config))
            self.assertEqual(receipt["checkpoint_hashes"]["vision.pth"]["algorithm"], "md5")
            self.assertEqual(receipt["checkpoint_hashes"]["model.bin"]["checksum"], "a" * 64)
            for change in ({"assets_verified": False}, {"upstream_revision": "wrong"},
                           {"asset_manifest_sha256": "changed"}):
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "preflight"):
                    calibration_base_receipt(args, {**verified, **change})

    def test_receipt_rows_reject_changed_recipe_nonfinite_and_wrong_scope(self):
        rows = calibration_rows()
        validate_calibration_rows(rows)
        changes = ((0, "score", .9), (0, "text_cosine", .9),
                   (1, "score", float("nan")), (1, "visual_cosine", float("inf")),
                   (1, "text_cosine", 1.01), (1, "score", True),
                   (3, "score", .2), (3, "visual_cosine", .2), (0, "applicable", 1))
        for index, key, value in changes:
            with self.subTest(index=index, key=key, value=value):
                changed = copy.deepcopy(rows)
                changed[index][key] = value
                with self.assertRaises(ValueError):
                    validate_calibration_rows(changed)
        changed = copy.deepcopy(rows)
        changed[0].update(score=.5, text_cosine=.5, visual_cosine=.5)
        with self.assertRaisesRegex(ValueError, "itself"):
            validate_calibration_rows(changed)

    def retriever(self):
        return VisualCorrectionRetriever(None, Encoder(), .7,
            visual_encoder=lambda image: [0., 1.] if "neighbor" in str(image) else [1., 0.])

    def test_updated_memory_matches_image_and_question_and_routes_neighbor_to_base(self):
        retriever = self.retriever()
        retriever.set_correction("edit.jpg", "city?", "New City", "edit1")
        direct = retriever.retrieve("edit.jpg", "city?")
        self.assertTrue(direct["accepted"])
        self.assertEqual(direct["score"], 1.)
        self.assertFalse(direct["score_is_probability"])
        self.assertEqual(retriever.prompt("city?", direct), "Question: city? Answer: New City\nQuestion: city? Answer:")
        neighbor = retriever.retrieve("neighbor.jpg", "neighbor?")
        self.assertFalse(neighbor["accepted"])
        self.assertEqual(retriever.prompt("neighbor?", neighbor), "Question: neighbor? Answer:")
        retriever.set_correction("edit.jpg", "city?", "Updated City", "edit2")
        with self.assertRaisesRegex(ValueError, "current"):
            retriever.prompt("city?", direct)
        self.assertIn("Updated City", retriever.prompt("city?", retriever.retrieve("edit.jpg", "city?")))

    def test_text_only_query_rejected_even_when_its_question_matches(self):
        retriever = self.retriever()
        retriever.set_correction("edit.jpg", "city?", "New City", "edit")
        result = retriever.retrieve(None, "city?")
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["score"])

    def test_threshold_rule_f1_then_precision_then_higher_inclusive_cutoff(self):
        rows = [{"score": .9, "applicable": True}, {"score": .8, "applicable": True},
                {"score": .7, "applicable": False}, {"score": None, "applicable": False}]
        result = select_threshold(rows)
        self.assertEqual(result["threshold"], .8)
        self.assertEqual(result["f1"], 1.)
        self.assertEqual(result["denominator"], 4)
        self.assertEqual(threshold_metrics(rows, .9)["tp"], 1)
        negatives = [{"score": .8, "applicable": False}]
        self.assertGreater(select_threshold(negatives)["threshold"], .8)
        with self.assertRaises(ValueError):
            select_threshold([{"score": float("nan"), "applicable": True}])

    def test_finite_vector_and_memory_validation(self):
        self.assertEqual(cosine([2., 0.], [3., 0.]), 1.)
        for vectors in (([0.], [1.]), ([1.], [1., 2.]), ([math.inf], [1.])):
            with self.assertRaises(ValueError):
                cosine(*vectors)
        with self.assertRaises(ValueError):
            self.retriever().set_correction(None, "q", "a", "id")

    def test_default_plans_load_nothing_and_use_pinned_separate_assets(self):
        with patch("scripts.prepare_vlkeb_rag.prepare") as prepare, redirect_stdout(io.StringIO()):
            self.assertEqual(preparation_main([]), 0)
            self.assertEqual(main([]), 0)
        prepare.assert_not_called()
        assets = minilm_assets()
        self.assertEqual(len(assets), 7)
        self.assertTrue(all(len(a["sha256"]) == 64 and "1110a243" in a["url"] for a in assets))
        self.assertEqual(manifest_record()["total_download_bytes"], sum(a["size_bytes"] for a in assets))

    def test_calibration_is_exact_fixed40_cases_with_all_labels_retained(self):
        fixture = {"records": [{"source_index": i, "record": {"image": f"e{i}/a.jpg",
            "image_rephrase": f"e{i}/b.jpg", "m_loc": "neighbor/c.jpg", "src": "city?",
            "rephrase": "other city?", "alt": "target", "m_loc_q": "neighbor?", "loc": "text?"}}
            for i in range(2, 10)]}
        with patch("ke_repro.rag.image_feature", side_effect=lambda backend, image:
                   [0., 1.] if "neighbor" in image else [1., 0.]):
            result = calibration_operation(None, Encoder(), fixture, Path("assets"))
        self.assertEqual(result["validation_row_count"], 40)
        self.assertEqual(result["validation_source_indices"], list(range(2, 10)))
        self.assertEqual(sum(r["applicable"] for r in result["rows"]), 24)
        self.assertEqual(sum(r["score"] is None for r in result["rows"]), 8)
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["evaluation_split_opened"])
        self.assertFalse(result["probability_calibrated"])


if __name__ == "__main__":
    unittest.main()
