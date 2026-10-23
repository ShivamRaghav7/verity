"""Finite evaluator contract checks with synthetic answers; no Torch or models."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ke_repro.evaluate import (CATEGORIES, METHODS, aggregate, cases_for, checked_seal,
                               evaluate_edit, fingerprint, main, placeholder, read_rows, run, seal)


def edit(index=0):
    return {"edit_id": f"{index:064x}", "record": {"src": "Which city?", "rephrase": "Name this city?",
        "pred": "Old City", "alt": "New City", "image": "entity/a.jpg", "image_rephrase": "entity/b.jpg",
        "loc": "What is two plus two?", "loc_ans": "Four", "m_loc": "neighbor/c.jpg",
        "m_loc_q": "Which mountain?", "m_loc_a": "Mountain"}}


class FakeBackend:
    def __init__(self, *, bad_restore=False, edit_error=False, generation_error=False):
        self.patches = SimpleNamespace(active=None)
        self.calls = []
        self.bad_restore, self.edit_error, self.generation_error = bad_restore, edit_error, generation_error

    def weight_hashes(self):
        return {"matrix": "edited" if self.patches.active else "base"}

    def generate(self, image, question, max_new_tokens):
        self.calls.append((image, question, max_new_tokens))
        if self.generation_error:
            raise RuntimeError("Deliberate generation problem")
        if question.endswith("What is two plus two? Answer:"):
            return "Four"
        if question.endswith("Which mountain? Answer:"):
            return "Mountain"
        return "New City" if self.patches.active else "Old City"

    def edit(self, image, question, target, *, target_alignment):
        if self.patches.active:
            raise AssertionError("Every edit method must begin from base")
        self.patches.active = True
        if self.edit_error:
            raise RuntimeError("Deliberate edit failure after update")
        return {"target_alignment": target_alignment}

    def edit_finetune(self, image, question, target):
        return {**self.edit(image, question, target, target_alignment="context"), "optimizer_steps": 15}

    def rollback(self):
        if not self.bad_restore:
            self.patches.active = None
        return {"exact_weight_restore": not self.bad_restore}


class OperationTests(unittest.TestCase):
    def test_exact_five_cases_four_methods_targetfree_edits_and_text_without_image(self):
        backend = FakeBackend()
        result = evaluate_edit(backend, edit(), Path("assets"), "seal")
        self.assertEqual(len(backend.calls), 20)
        self.assertEqual(set(result["methods"]), set(METHODS))
        self.assertTrue(result["exact_restore"])
        self.assertIsNone(backend.patches.active)
        self.assertTrue(all(call[2] == 24 for call in backend.calls))
        self.assertIsNone(backend.calls[3][0])
        self.assertIsNone(backend.calls[8][0])
        self.assertIsNone(backend.calls[13][0])
        self.assertIsNone(backend.calls[18][0])
        self.assertEqual(backend.calls[5][1], "Question: Which city? Answer: New City\nQuestion: Which city? Answer:")
        self.assertNotIn("New City", backend.calls[10][1])
        self.assertNotIn("New City", backend.calls[15][1])
        self.assertEqual(result["edit_metadata"]["target_alignment"], "context")
        self.assertTrue(result["methods"]["mend_context"][0]["strict_target_match"])
        self.assertFalse(result["methods"]["base"][0]["strict_target_match"])
        self.assertEqual(result["method_edits"]["ft_last_layer"]["metadata"]["optimizer_steps"], 15)
        self.assertTrue(all(r["base_hashes"] == result["base_hashes"] and r["exact_restore"] for r in result["method_edits"].values()))

    def test_partial_edit_error_still_restores_and_counts_all_mend_cases_failed(self):
        backend = FakeBackend(edit_error=True)
        result = evaluate_edit(backend, edit(), Path("assets"), "seal")
        self.assertTrue(result["exact_restore"])
        self.assertTrue(all(not r["strict_target_match"] and r["status"] == "edit_error" for r in result["methods"]["mend_context"]))
        self.assertEqual(len(backend.calls), 10)

    def test_generation_errors_and_restore_failure_are_explicit(self):
        result = evaluate_edit(FakeBackend(generation_error=True, bad_restore=True), edit(), Path("assets"), "seal")
        self.assertFalse(result["exact_restore"])
        self.assertIn("restore_error", result)
        self.assertTrue(all(r["status"] == "generation_error" for m in METHODS[:3] for r in result["methods"][m]))
        self.assertTrue(all(r["status"] == "not_executed" for r in result["methods"]["ft_last_layer"]))

    def test_all20_including_not_executed_keep_primary_denominators_and_retention(self):
        rows = [evaluate_edit(FakeBackend(), edit(i), Path("assets"), "seal") for i in range(2)]
        rows += [placeholder(edit(i), Path("assets"), "not run") for i in range(2, 20)]
        result = aggregate(rows)
        direct = result["scores"]["mend_context"]["direct"]
        self.assertEqual(direct["denominator"], 20)
        self.assertEqual(direct["correct"], 2)
        self.assertEqual(direct["errors_or_unexecuted"], 18)
        self.assertEqual(direct["accuracy"], .1)
        retention = result["locality_preservation"]["mend_context"]["text_locality"]
        self.assertEqual(retention["baseline_correct_denominator"], 2)
        self.assertEqual(retention["baseline_correct_answer_preserved"], 2)
        self.assertEqual(retention["unconditional_denominator"], 20)
        self.assertEqual(len(result["paired_differences"]), 30)
        self.assertTrue(all(r["unit_count"] == 20 and r["resamples"] == 10000 for r in result["paired_differences"]))
        self.assertEqual(result["query_latency"]["base"]["direct"]["sample_count"], 2)
        with self.assertRaisesRegex(ValueError, "all20"):
            aggregate(rows[:19])


class SealingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.manifest, self.selection = root / "manifest.json", root / "selection.json"
        self.manifest.write_text("{}")
        self.selection.write_text(json.dumps({"edits": [edit(i) for i in range(20)]}))
        self.args = SimpleNamespace(manifest=self.manifest, selection=self.selection, output=root / "run",
            assets=root / "assets", upstream=root / "upstream", device="cuda", timeout_seconds=1800, resume=False, fixture=None)

    def test_plan_never_validates_assets_or_imports_models(self):
        with patch("ke_repro.evaluate.validate_selection") as validation, redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 0)
        validation.assert_not_called()

    def test_seal_precedes_inference_and_rejects_modified_selection_configuration(self):
        with patch("ke_repro.evaluate.validate_selection", return_value=({}, {"verified": True})):
            result = seal(self.args)
        self.assertEqual(result["status"], "sealed_not_evaluated")
        self.assertFalse(result["models_loaded"])
        checked_seal(self.args)
        self.args.timeout_seconds = 100
        with self.assertRaisesRegex(ValueError, "changed"):
            checked_seal(self.args)
        self.args.timeout_seconds = 1800
        self.selection.write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            checked_seal(self.args)
        with patch("ke_repro.evaluate.validate_selection"), self.assertRaises(FileExistsError):
            seal(self.args)

    def test_saved_rows_must_match_seal_and_be_complete(self):
        selection = {"edits": [edit()]}
        directory = self.args.output / "edits"
        directory.mkdir(parents=True)
        path = directory / (edit()["edit_id"] + ".json")
        row = evaluate_edit(FakeBackend(), edit(), self.args.assets, "seal")
        path.write_text(json.dumps(row))
        self.assertEqual(len(read_rows(self.args, selection, "seal")), 1)
        with self.assertRaisesRegex(ValueError, "belong"):
            read_rows(self.args, selection, "another-seal")
        row["methods"]["base"].pop()
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            read_rows(self.args, selection, "seal")

    def test_resume_cannot_change_labels_or_promote_an_incorrect_answer(self):
        selection = {"edits": [edit()]}
        directory = self.args.output / "edits"
        directory.mkdir(parents=True)
        path = directory / (edit()["edit_id"] + ".json")
        row = evaluate_edit(FakeBackend(), edit(), self.args.assets, "seal")
        row["methods"]["base"][0]["strict_target_match"] = True
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "scorer"):
            read_rows(self.args, selection, "seal")
        row["methods"]["base"][0]["strict_target_match"] = False
        row["methods"]["base"][0]["expected"] = "Changed target"
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "frozen"):
            read_rows(self.args, selection, "seal")

    def test_controller_nonzero_exit_cannot_be_reported_as_success(self):
        self.args.output.mkdir()
        selection = {"edits": [edit(i) for i in range(20)]}
        def process(*args, **kwargs):
            (self.args.output / "worker.json").write_text(json.dumps({"execution_status": "completed"}))
            return SimpleNamespace(returncode=-9)
        with patch("ke_repro.evaluate.checked_seal", return_value=({}, "seal")), \
             patch("ke_repro.evaluate.validate_selection", return_value=(selection, {})), \
             patch("ke_repro.evaluate.subprocess.run", side_effect=process), redirect_stdout(io.StringIO()):
            self.assertEqual(run(self.args), 1)
        summary = json.loads((self.args.output / "evaluation.json").read_text())
        self.assertEqual(summary["execution_status"], "failed")
        self.assertEqual(summary["unexecuted_edit_units"], 20)
        self.assertEqual(summary["aggregate"]["scores"]["base"]["direct"]["denominator"], 20)


if __name__ == "__main__":
    unittest.main()
