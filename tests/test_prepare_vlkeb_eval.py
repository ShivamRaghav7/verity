import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.prepare_vlkeb_eval import EVAL_ASSET, image_entity, main, prepare, select_edits


def row(entity, index=0):
    return {"src": f"Question {entity}?", "rephrase": f"Other {entity}?",
            "pred": entity, "alt": f"target-{entity}",
            "image": f"{entity}/a{index}.jpg", "image_rephrase": f"{entity}/b{index}.jpg",
            "loc": "Text question?", "loc_ans": "Text answer",
            "m_loc": "locality/c.jpg", "m_loc_q": "Image question?", "m_loc_a": "Image answer"}


class SelectionTests(unittest.TestCase):
    def test_fixed20_unique_source_entities_excludes_every_train_visual_slot(self):
        train = [row("trained")]
        train[0]["image_rephrase"] = "also-trained/a.jpg"
        evaluation = [row("trained"), row("also-trained"), row("locality")]
        evaluation += [row(f"entity{i:03}") for i in range(30)]
        result = select_edits(train, evaluation)
        self.assertEqual(result["edit_count"], 20)
        self.assertEqual(result["case_count"], 100)
        entities = {edit["source_entity"] for edit in result["edits"]}
        self.assertEqual(len(entities), 20)
        self.assertFalse(entities & {"trained", "also-trained", "locality"})
        self.assertFalse(result["evaluation_executed"])

    def test_order_and_duplicate_rows_do_not_select_favorable_questions(self):
        rows = [row(f"entity{i:03}") for i in range(30)] + [row("entity005", 2)]
        first, second = select_edits([row("trained")], rows), select_edits([row("trained")], list(reversed(rows)))
        self.assertEqual(first["selection_ids_sha256"], second["selection_ids_sha256"])
        self.assertEqual([e["record"] for e in first["edits"]], [e["record"] for e in second["edits"]])

    def test_insufficient_pool_or_unsafe_paths_fail(self):
        with self.assertRaisesRegex(ValueError, "Fewer than 20"):
            select_edits([row("trained")], [row("only-one")])
        for path in ("../a.jpg", "/absolute/a.jpg", "x\\a.jpg", "C:/a.jpg"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                image_entity(path)

    def test_default_plan_has_no_network_or_writes(self):
        with patch("scripts.prepare_vlkeb_eval.download_asset") as download, \
             patch("scripts.prepare_vlkeb_eval.prepare") as prepare, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 0)
        download.assert_not_called()
        prepare.assert_not_called()
        self.assertEqual(EVAL_ASSET["size_bytes"], 1658466)
        self.assertEqual(len(EVAL_ASSET["sha256"]), 64)

    def test_protocol_refuses_a_substituted_train_manifest_before_network(self):
        manifest = {"assets": [{"path": "upstream-train.json", "sha256": "wrong"},
                               {"path": "upstream-images.tar.gz", "sha256": "wrong"}]}
        with patch("scripts.prepare_vlkeb_eval.download_asset") as download, \
             self.assertRaisesRegex(ValueError, "exact pinned training"):
            prepare(Path("."), Path("a-nonexistent-frozen-selection.json"), manifest)
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
