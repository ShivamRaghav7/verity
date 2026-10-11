import json
import io
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from editguard.neural import file_sha256
from scripts.calibrate_scope import COSTS, MARGINS, TEMPERATURES, THRESHOLDS, fit_calibration, operating_metrics, reviewed_inputs, _safe_output
from scripts.prepare_data import content_hash, write_dataset


class CalibrationFitTests(unittest.TestCase):
    def test_declared_costs_and_abstentions_have_full_denominators(self):
        result = operating_metrics([1, 1, 0, 0], [.9, .3, .9, .5], .5, 0)
        self.assertEqual(result["total_cost"], 5.25)
        self.assertEqual(result["mean_cost"], {"numerator": 5.25, "denominator": 4, "value": 1.3125})
        self.assertEqual(result["coverage"]["value"], .75)
        self.assertEqual(result["accuracy_including_clarification_as_unsuccessful"]["value"], .25)
        self.assertEqual(result["recall_including_clarification_as_missed"]["denominator"], 2)
        self.assertEqual(result["false_application_rate"]["value"], .5)
        self.assertEqual(result["counts"]["clarification_negative"], 1)
        self.assertEqual(COSTS, {"false_application": 4, "missed_positive": 1, "clarification": .25, "correct": 0})

    def test_uncertainty_band_is_inclusive_like_runtime(self):
        result = operating_metrics([1, 0], [.75, .25], .5, .25)
        self.assertEqual(result["coverage"]["value"], 0)
        self.assertEqual(result["total_cost"], .5)
        self.assertIsNone(result["selective_accuracy"]["value"])
        self.assertEqual(result["recall_including_clarification_as_missed"]["value"], 0)

    def test_finite_temperature_and_operating_grids(self):
        result = fit_calibration([1, 0], [.9, .1])
        self.assertEqual(len(result["temperature_candidates"]), len(TEMPERATURES))
        self.assertEqual(len(result["operating_candidates"]), len(THRESHOLDS) * len(MARGINS))
        self.assertEqual(result["temperature"], .5)
        self.assertEqual(result["selected_operating_metrics"]["total_cost"], 0)
        self.assertEqual(result["selected_operating_metrics"]["coverage"]["value"], 1)

    def test_equal_probability_avoids_expensive_false_application(self):
        result = fit_calibration([1, 0], [.5, .5])
        self.assertEqual(result["temperature"], 1)
        self.assertEqual(result["threshold"], .5)
        self.assertEqual(result["ambiguity_margin"], 0)
        self.assertEqual(result["selected_operating_metrics"]["total_cost"], .5)
        self.assertEqual(result["selected_operating_metrics"]["counts"]["false_application"], 0)

    def test_selection_is_reproducible_and_ties_favor_coverage(self):
        labels, probabilities = [1, 0, 1, 0], [.9, .1, .7, .3]
        first = fit_calibration(labels, probabilities)
        self.assertEqual(first, fit_calibration(labels, probabilities))
        selected = first["selected_operating_metrics"]
        equal_cost = [candidate for candidate in first["operating_candidates"] if candidate["total_cost"] == selected["total_cost"]]
        self.assertEqual(selected["counts"]["false_application"], min(candidate["counts"]["false_application"] for candidate in equal_cost))
        fewest_false = [candidate for candidate in equal_cost if candidate["counts"]["false_application"] == selected["counts"]["false_application"]]
        self.assertEqual(selected["coverage"]["numerator"], max(candidate["coverage"]["numerator"] for candidate in fewest_false))

    def test_invalid_or_missing_probabilities_and_one_class_refused(self):
        cases = [([], []), ([0], [.2]), ([1, 0], [.2]), ([True, 0], [.9, .1]),
                 ([1, 0], [None, .1]), ([1, 0], [True, .1]),
                 ([1, 0], [float("nan"), .1]), ([1, 0], [1.01, .1])]
        for labels, probabilities in cases:
            with self.subTest(labels=labels, probabilities=probabilities), self.assertRaises(ValueError):
                fit_calibration(labels, probabilities)
        with self.assertRaises(ValueError):
            operating_metrics([1, 0], [.9, .1], True, .1)


class CalibrationInputGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset, self.artifact = self.root / "dataset", self.root / "scope"
        write_dataset(self.dataset, family_count=20)
        self.reviews = self.root / "reviews.json"
        raw = [json.loads(line) for line in (self.dataset / "scope_validation.jsonl").read_text().splitlines()]
        self.raw = raw
        registry = {"rows": {row["id"]: {"decision": "accept", "reviewer": "Synthetic calibration unit fixture only",
                       "reviewed_at": "2026-10-11T15:00:00+05:30", "row_sha256": content_hash(row)} for row in raw}}
        self.reviews.write_text(json.dumps(registry), encoding="utf-8")
        self.artifact.mkdir()
        for name, content in (("model/config.json", "{}"), ("model/model.safetensors", "synthetic-not-model-weights"),
                              ("tokenizer/tokenizer_config.json", "{}")):
            path = self.artifact / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.manifest = {"artifact_type": "editguard_auxiliary", "purpose": "trained", "status": "completed", "task": "scope",
                         "dataset_id": "editguard-controlled-v1", "manifest_sha256": file_sha256(self.dataset / "manifest.json"),
                         "input_format_version": "editguard-input-v1", "input_length": 192,
                         "files_sha256": {str(path.relative_to(self.artifact)).replace("\\", "/"): file_sha256(path)
                                          for path in self.artifact.rglob("*") if path.is_file()}}
        self.save_manifest()

    def save_manifest(self):
        (self.artifact / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def inputs(self, **kwargs):
        return reviewed_inputs(self.artifact, self.dataset, self.reviews, max_rows=kwargs.get("max_rows", 2000))

    def test_complete_validation_review_and_hash_bindings_without_test_reads(self):
        # Deliberately invalid test artifacts prove they are outside this reader.
        for path in self.dataset.glob("*test.jsonl"):
            path.write_text("UNREADABLE TEST DATA", encoding="utf-8")
        rows, bindings = self.inputs()
        self.assertEqual(len(rows), len(self.raw))
        self.assertEqual(bindings["accepted_validation_rows"], len(self.raw))
        self.assertFalse(bindings["test_rows_read"])
        self.assertEqual(bindings["scope_manifest_sha256"], file_sha256(self.artifact / "manifest.json"))
        self.assertEqual(bindings["accepted_row_sha256"][rows[0]["id"]], content_hash(self.raw[0]))

    def test_pending_and_stale_reviews_fail_before_model_execution(self):
        records = json.loads(self.reviews.read_text())
        records["rows"].pop(self.raw[0]["id"])
        self.reviews.write_text(json.dumps(records), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "human review"):
            self.inputs()

    def test_manifest_tampering_and_wrong_split_rejected(self):
        path = self.dataset / "scope_validation.jsonl"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content differs"):
            self.inputs()

    def test_test_tagged_row_is_refused_even_if_hashes_updated(self):
        path = self.dataset / "scope_validation.jsonl"
        altered = [dict(row) for row in self.raw]
        altered[0]["split"] = "test"
        path.write_text("".join(json.dumps(row) + "\n" for row in altered), encoding="utf-8")
        data_manifest = json.loads((self.dataset / "manifest.json").read_text())
        data_manifest["files"]["scope_validation.jsonl"] = file_sha256(path)
        (self.dataset / "manifest.json").write_text(json.dumps(data_manifest), encoding="utf-8")
        self.manifest["manifest_sha256"] = file_sha256(self.dataset / "manifest.json")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            self.inputs()

    def test_cap_never_silently_truncates_validation(self):
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            self.inputs(max_rows=2)

    def test_smoke_models_and_other_dataset_bindings_refused(self):
        self.manifest["purpose"] = "smoke_test"
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "trained auxiliary"):
            self.inputs()
        self.manifest["purpose"] = "trained"
        self.manifest["dataset_id"] = "other-data"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "declared dataset"):
            self.inputs()

    def test_rejected_rows_have_explicit_count_and_no_inference_record(self):
        records = json.loads(self.reviews.read_text())
        records["rows"][self.raw[0]["id"]].update(decision="reject", reason="Synthetic exclusion fixture")
        self.reviews.write_text(json.dumps(records), encoding="utf-8")
        rows, bindings = self.inputs()
        self.assertEqual(bindings["reviewed_rejected_rows"], 1)
        self.assertNotIn(self.raw[0]["id"], {row["id"] for row in rows})

    def test_output_cannot_overwrite_or_modify_sources(self):
        for output in (self.dataset, self.dataset / "new-results", self.artifact / "new-results", self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                _safe_output(output, self.artifact, self.dataset, self.reviews)
        _safe_output(self.root / "calibration", self.artifact, self.dataset, self.reviews)

    def test_preflight_cli_does_not_import_ml_or_create_output(self):
        output = self.root / "calibration"
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/calibrate_scope.py"),
                                    "--scope-artifact", str(self.artifact), "--dataset", str(self.dataset),
                                    "--reviews", str(self.reviews), "--output", str(output), "--preflight-only"],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertFalse(result["model_execution_requested"])
        self.assertFalse(output.exists())

    def test_stubbed_cli_publishes_bound_artifact_and_raw_denominators(self):
        from scripts.calibrate_scope import main

        class SyntheticScope:
            def __init__(self, *args):
                self.calls = 0

            def score(self, question, correction):
                self.calls += 1
                return .9 if self.calls % 2 else .1

        output = self.root / "calibration"
        argv = ["calibrate_scope.py", "--scope-artifact", str(self.artifact), "--dataset", str(self.dataset),
                "--reviews", str(self.reviews), "--output", str(output)]
        with patch.object(sys, "argv", argv), patch("editguard.neural.ScopeCheckpoint", SyntheticScope), patch("sys.stdout", io.StringIO()):
            main()
        artifact = json.loads((output / "calibration.json").read_text())
        self.assertEqual((artifact["purpose"], artifact["split"], artifact["status"]), ("validation_calibration", "validation", "frozen"))
        self.assertEqual(artifact["scope_manifest_sha256"], file_sha256(self.artifact / "manifest.json"))
        self.assertEqual(artifact["probabilities_sha256"], file_sha256(output / "validation_probabilities.jsonl"))
        self.assertEqual(artifact["validation_operating_metrics"]["counts"]["total"], len(self.raw))
        self.assertFalse(artifact["test_rows_read"])
        self.assertFalse((output / "calibration.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
