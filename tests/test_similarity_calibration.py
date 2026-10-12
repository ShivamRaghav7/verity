import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from editguard.neural import file_sha256
from editguard.formatting import INPUT_FORMAT_VERSION
from scripts.calibrate_similarity import (THRESHOLDS, fit_threshold, operating_metrics, pair_eligible,
                                         relevant_config, reviewed_inputs, safe_output, score_pairs)
from scripts.prepare_data import content_hash, write_dataset

ROOT = Path(__file__).resolve().parents[1]


class SimilarityFitTests(unittest.TestCase):
    def test_negative_cosines_are_valid_and_threshold_equality_applies(self):
        result = fit_threshold([1, 0], [-.1, -.2])
        self.assertEqual(result["threshold"], -.19)
        self.assertEqual(result["compatibility_gated_metrics"]["total_cost"], 0)
        at_boundary = operating_metrics([1, 0], [.5, .5], .5)
        self.assertEqual(at_boundary["counts"]["true_application"], 1)
        self.assertEqual(at_boundary["counts"]["false_application"], 1)
        self.assertEqual(at_boundary["counts"]["clarification"], 0)

    def test_never_apply_sentinel_can_win_when_every_threshold_is_unsafe(self):
        result = fit_threshold([1, 0], [.99999, 1.0])
        self.assertEqual(result["threshold"], 1.01)
        self.assertEqual(result["compatibility_gated_metrics"]["counts"]["false_application"], 0)
        self.assertEqual(result["compatibility_gated_metrics"]["total_cost"], 1)
        self.assertEqual(len(result["candidates"]), len(THRESHOLDS))
        self.assertEqual(len(THRESHOLDS), 202)

    def test_raw_pair_quality_is_separate_from_compatibility_gates(self):
        result = fit_threshold([1, 0, 0], [.6, .4, .95], [True, True, False])
        self.assertEqual(result["threshold"], .41)
        self.assertEqual(result["compatibility_gated_metrics"]["counts"]["false_application"], 0)
        self.assertEqual(result["raw_pair_metrics"]["counts"]["false_application"], 1)
        self.assertEqual(result["raw_pair_metrics"]["false_application_rate"]["denominator"], 2)
        self.assertEqual(result["compatibility_gated_metrics"]["counts"]["incompatible_pairs"], 1)

    def test_all_pairs_stay_in_cost_and_recall_even_when_gate_rejects_positive(self):
        result = operating_metrics([1, 0, 1, 0], [.7, .8, .2, .9], .5, [True, True, False, False])
        self.assertEqual(result["total_cost"], 5)
        self.assertEqual(result["mean_cost"]["denominator"], 4)
        self.assertEqual(result["recall"]["value"], .5)
        self.assertEqual(result["counts"]["applicable_pairs_rejected_by_gate"], 1)
        self.assertEqual(result["routing_decision_coverage"]["value"], 1)
        self.assertIsNone(result["generated_answer_coverage"])

    def test_no_probability_metrics_or_uncertainty_band_are_invented(self):
        result = fit_threshold([1, 0], [.8, .3])
        encoded = json.dumps(result)
        for field in ("temperature", "brier", "ece", "negative_log_likelihood"):
            self.assertNotIn('"' + field + '"', encoded)
        self.assertEqual(result["compatibility_gated_metrics"]["clarification_rate"]["value"], 0)

    def test_invalid_missing_scores_labels_flags_and_thresholds_fail(self):
        cases = [([], [], None), ([1], [.5], None), ([True, 0], [.5, .2], None),
                 ([1, 0], [.5], None), ([1, 0], [None, .2], None), ([1, 0], [float("nan"), .2], None),
                 ([1, 0], [1.2, .2], None), ([1, 0], [.5, .2], [True]), ([1, 0], [.5, .2], [1, 0])]
        for labels, scores, eligible in cases:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                fit_threshold(labels, scores, eligible)
        for threshold in (True, float("nan"), -1.01, 1.02):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                operating_metrics([1, 0], [.5, .2], threshold)

    def test_rounding_overshoot_is_kept_and_selection_is_deterministic(self):
        result = fit_threshold([1, 0], [1.0000001, -.5])
        self.assertEqual(result, fit_threshold([1, 0], [1.0000001, -.5]))
        self.assertEqual(result["compatibility_gated_metrics"]["total_cost"], 0)


class SimilarityInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "data"
        write_dataset(self.dataset, family_count=20)
        self.config = self.root / "inference.json"
        config = json.loads((ROOT / "configs/inference.json").read_text())
        config["input_format_version"] = INPUT_FORMAT_VERSION
        self.config.write_text(json.dumps(config), encoding="utf-8")
        self.raw = [json.loads(line) for line in (self.dataset / "scope_validation.jsonl").read_text().splitlines()]
        self.reviews = self.root / "reviews.json"
        records = {row["id"]: {"decision": "accept", "reviewer": "Synthetic similarity unit fixture only",
                   "reviewed_at": "2026-10-11T15:00:00+05:30", "row_sha256": content_hash(row)} for row in self.raw}
        self.reviews.write_text(json.dumps({"rows": records}), encoding="utf-8")

    def inputs(self, max_rows=2000):
        return reviewed_inputs(self.config, self.dataset, self.reviews, max_rows=max_rows)

    def test_validation_only_hash_review_gate_and_original_row_hashes(self):
        for path in self.dataset.glob("*test.jsonl"):
            path.write_text("UNREADABLE TEST ROWS", encoding="utf-8")
        config, rows, bindings = self.inputs()
        self.assertEqual(len(rows), len(self.raw))
        self.assertFalse(bindings["test_rows_read"])
        self.assertEqual(bindings["review_registry_sha256"], file_sha256(self.reviews))
        self.assertEqual(bindings["accepted_row_sha256"][rows[0]["id"]], content_hash(self.raw[0]))
        self.assertEqual(bindings["pooling"], "masked_mean_l2")
        self.assertEqual(bindings["embedding_max_tokens"], 256)
        self.assertEqual(bindings["routing_config_sha256"], content_hash(relevant_config(config)))

    def test_pending_review_stale_hash_and_unpinned_model_fail(self):
        records = json.loads(self.reviews.read_text())
        records["rows"][self.raw[0]["id"]]["row_sha256"] = "stale"
        self.reviews.write_text(json.dumps(records), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "human review"):
            self.inputs()
        config = json.loads(self.config.read_text())
        config["embedding_revision"] = "main"
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pinned"):
            self.inputs()

    def test_tampered_rows_test_tags_and_silent_subset_are_refused(self):
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            self.inputs(max_rows=2)
        path = self.dataset / "scope_validation.jsonl"
        rows = copy.deepcopy(self.raw)
        rows[0]["split"] = "test"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content differs"):
            self.inputs()
        manifest = json.loads((self.dataset / "manifest.json").read_text())
        manifest["files"][path.name] = file_sha256(path)
        (self.dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            self.inputs()

    def test_reviewed_rejections_are_counted_and_not_silently_pending(self):
        records = json.loads(self.reviews.read_text())
        records["rows"][self.raw[0]["id"]].update(decision="reject", reason="Synthetic rejected example")
        self.reviews.write_text(json.dumps(records), encoding="utf-8")
        _, rows, bindings = self.inputs()
        self.assertEqual(bindings["reviewed_rejected_rows"], 1)
        self.assertEqual(len(rows), len(self.raw) - 1)

    def test_pair_gate_uses_exact_scope_effective_dates_and_approval(self):
        positive = next(copy.deepcopy(row) for row in self.raw if row["label"] == 1)
        self.assertTrue(pair_eligible(positive))
        for key in ("entity_id", "plan", "version"):
            altered = copy.deepcopy(positive)
            altered["request_context"][key] = "different"
            self.assertFalse(pair_eligible(altered))
        altered = copy.deepcopy(positive)
        altered["correction"]["effective_to"] = altered["request_context"]["as_of"]
        self.assertFalse(pair_eligible(altered))
        altered = copy.deepcopy(positive)
        altered["correction"]["approval_status"] = "draft"
        self.assertFalse(pair_eligible(altered))

    def test_scoring_truth_and_label_never_enter_embedding_inference(self):
        class SyntheticRetriever:
            def __init__(self):
                self.calls = []

            def rank(self, question, corrections, formatter, k):
                self.calls.append((question, [formatter(edit) for edit in corrections], k))
                return [{"record": corrections[0], "score": .7}]

            def tokenizer(self, text):
                return {"input_ids": text.split()}

        first, second = SyntheticRetriever(), SyntheticRetriever()
        original, altered = copy.deepcopy(self.raw[0]), copy.deepcopy(self.raw[0])
        altered["expected_fact"] = {"value": "99999", "entity_id": "secret-scoring-subject"}
        altered["label"] = 1 - altered["label"]
        a, b = score_pairs(first, [original]), score_pairs(second, [altered])
        self.assertEqual(first.calls, second.calls)
        self.assertEqual(a[0]["cosine_score"], b[0]["cosine_score"])
        self.assertEqual(a[0]["compatible"], b[0]["compatible"])

    def test_no_implicit_download_output_mutation_or_ml_import_in_preflight(self):
        config = json.loads(self.config.read_text())
        config["local_files_only"] = False
        with self.assertRaisesRegex(ValueError, "cached-only"):
            relevant_config(config)
        for output in (self.dataset, self.dataset / "new-output", self.root):
            with self.assertRaises(ValueError):
                safe_output(output, self.dataset, self.reviews, self.config)
        output = self.root / "selection"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/calibrate_similarity.py"), "--config", str(self.config),
                                 "--dataset", str(self.dataset), "--reviews", str(self.reviews), "--output", str(output), "--preflight-only"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["model_execution_requested"])
        self.assertFalse(output.exists())
        lazy = subprocess.run([sys.executable, "-c", "import sys; import scripts.calibrate_similarity; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(lazy.returncode, 0, lazy.stderr)


if __name__ == "__main__":
    unittest.main()
