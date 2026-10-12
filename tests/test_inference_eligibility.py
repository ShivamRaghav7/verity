"""Eligibility boundaries with explicit synthetic artifacts; no model execution."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from editguard.comparison import ComparisonRunner, load_similarity_calibration
from editguard.formatting import INPUT_FORMAT_VERSION
from editguard.neural import ModelUnavailable, file_sha256
from scripts.prepare_data import content_hash

ROOT = Path(__file__).resolve().parents[1]


class InferenceEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / "configs/inference.json").read_text(encoding="utf-8"))
        self.config["input_format_version"] = INPUT_FORMAT_VERSION
        self.artifact_path = self.root / "calibration.json"
        routing = {key: self.config[key] for key in ("embedding_model_id", "embedding_revision", "retrieval_k", "prompt_edit_k")}
        routing.update(embedding_max_tokens=256, pooling="masked_mean_l2", input_format_version=INPUT_FORMAT_VERSION,
                       routing_policy="top_compatible_cosine_gte_threshold_v1")
        scores = self.root / "validation_cosine_scores.jsonl"
        score_rows = [{"id": row_id, "row_sha256": digest, "split": "validation", "label": label,
                       "cosine_score": score, "compatible": True}
                      for row_id, digest, label, score in (("fixture-a", "f" * 64, 0, .2), ("fixture-b", "0" * 64, 1, .8))]
        scores.write_text("".join(json.dumps(row) + "\n" for row in score_rows), encoding="utf-8")
        grid = self.root / "selection_grid.json"
        grid.write_text(json.dumps({"threshold": .5, "rows": 2}), encoding="utf-8")
        self.artifact = {**routing, "artifact_type": "editguard_similarity_threshold",
                         "purpose": "similarity_validation_calibration", "split": "validation", "status": "frozen",
                         "score_type": "cosine_similarity_not_probability", "uncertainty_band": None, "threshold": .5,
                         "population": "single_focal_question_correction_pairs", "test_rows_read": False,
                         "review_policy": "automated_development", "human_reviewed_data": False, "review_audit_sha256": "8" * 64,
                         "embedding_weights_unchanged": True, "routing_config_sha256": content_hash(routing),
                         "dataset_manifest_sha256": "a" * 64, "validation_file_sha256": "b" * 64,
                         "review_registry_sha256": "c" * 64, "tokenizer_sha256": "d" * 64,
                         "embedding_state_sha256_before": "e" * 64, "embedding_state_sha256_after": "e" * 64,
                         "pair_scores_sha256": file_sha256(scores), "selection_grid_sha256": file_sha256(grid),
                         "accepted_validation_rows": 2, "accepted_row_sha256": {"fixture-a": "f" * 64, "fixture-b": "0" * 64}}
        for field, source in (("policy_source_sha256", "editguard/comparison.py"),
                              ("embedding_source_sha256", "editguard/neural.py"),
                              ("formatter_source_sha256", "editguard/formatting.py"),
                              ("script_sha256", "scripts/calibrate_similarity.py")):
            self.artifact[field] = file_sha256(ROOT / source)
        self.update_grid(.5)
        self.save()

    def update_grid(self, threshold):
        self.artifact["threshold"] = threshold
        # Minimal linked metrics are explicit synthetic fixtures, never a production selection.
        fields = {"raw_pair_metrics": {"fixture_only": True}, "compatibility_gated_metrics": {"fixture_only": True},
                  "declared_costs": {"false_application": 4, "missed_positive": 1}}
        self.artifact.update(fields)
        grid = self.root / "selection_grid.json"
        grid.write_text(json.dumps({"threshold": threshold, "rows": 2, **fields}), encoding="utf-8")
        self.artifact["selection_grid_sha256"] = file_sha256(grid)

    def save(self):
        self.artifact_path.write_text(json.dumps(self.artifact), encoding="utf-8")

    def test_valid_evidence_supports_negative_and_never_apply_thresholds(self):
        for threshold in (-1, 0, .5, 1.01):
            self.update_grid(threshold)
            self.save()
            self.assertEqual(load_similarity_calibration(self.artifact_path, self.config)["threshold"], threshold)

    def test_wrong_identity_policy_source_or_population_is_rejected(self):
        original = copy.deepcopy(self.artifact)
        for field, value in (("purpose", "smoke_test"), ("split", "test"), ("status", "development"),
                             ("embedding_revision", "9" * 40), ("embedding_max_tokens", 192),
                             ("pooling", "cls"), ("input_format_version", "previous-format"),
                             ("routing_policy", "other"), ("retrieval_k", 99),
                             ("score_type", "probability"), ("uncertainty_band", .1),
                             ("population", "full_memory"), ("test_rows_read", True),
                             ("embedding_weights_unchanged", False), ("embedding_state_sha256_after", "9" * 64),
                             ("review_policy", "unknown"), ("human_reviewed_data", True), ("review_audit_sha256", None),
                             ("routing_config_sha256", "9" * 64), ("policy_source_sha256", "9" * 64)):
            with self.subTest(field=field):
                self.artifact = {**original, field: value}
                self.save()
                with self.assertRaises(ModelUnavailable):
                    load_similarity_calibration(self.artifact_path, self.config)

    def test_invalid_threshold_missing_review_binding_or_grid_disagreement_is_rejected(self):
        original = copy.deepcopy(self.artifact)
        for value in (True, None, float("nan"), float("inf"), -1.01, 1.02):
            self.artifact = {**original, "threshold": value}
            self.save()
            with self.assertRaises(ModelUnavailable):
                load_similarity_calibration(self.artifact_path, self.config)
        for change in ({"review_registry_sha256": None}, {"accepted_row_sha256": {}},
                       {"accepted_validation_rows": True}, {"threshold": .7}):
            self.artifact = {**original, **change}
            self.save()
            with self.assertRaises(ModelUnavailable):
                load_similarity_calibration(self.artifact_path, self.config)

    def test_tampered_or_missing_evidence_files_are_rejected(self):
        (self.root / "validation_cosine_scores.jsonl").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ModelUnavailable, "integrity"):
            load_similarity_calibration(self.artifact_path, self.config)
        (self.root / "validation_cosine_scores.jsonl").unlink()
        with self.assertRaisesRegex(ModelUnavailable, "integrity"):
            load_similarity_calibration(self.artifact_path, self.config)

    def test_config_rejects_manual_threshold_or_old_format_before_loading_models(self):
        for change in ({"similarity_threshold": .5}, {"input_format_version": "previous-format"},
                       {"retrieval_k": True}, {"prompt_edit_k": 0}, {"max_new_tokens": -1}, {"max_input_tokens": 1.5}):
            path = self.root / "inference.json"
            path.write_text(json.dumps({**self.config, **change}), encoding="utf-8")
            with patch("editguard.comparison.FrozenGenerator") as generator:
                with self.assertRaises(ModelUnavailable):
                    ComparisonRunner.from_config(path)
                generator.assert_not_called()

    def test_valid_artifact_supplies_threshold_to_runner(self):
        path = self.root / "inference.json"
        config = {**self.config, "similarity_calibration_artifact": str(self.artifact_path)}
        path.write_text(json.dumps(config), encoding="utf-8")
        with patch("editguard.comparison.FrozenGenerator"), patch("editguard.comparison.EmbeddingRetriever") as retriever:
            retriever.return_value.parameter_fingerprint.return_value = self.artifact["embedding_state_sha256_before"]
            retriever.return_value.tokenizer_fingerprint.return_value = self.artifact["tokenizer_sha256"]
            runner = ComparisonRunner.from_config(path)
        self.assertEqual(runner.similarity_threshold, .5)

    def test_loaded_embedding_or_tokenizer_disagreement_is_rejected(self):
        path = self.root / "inference.json"
        path.write_text(json.dumps({**self.config, "similarity_calibration_artifact": str(self.artifact_path)}), encoding="utf-8")
        for mismatch in ("parameters", "tokenizer"):
            with patch("editguard.comparison.FrozenGenerator") as base, patch("editguard.comparison.EmbeddingRetriever") as retriever:
                retriever.return_value.parameter_fingerprint.return_value = "9" * 64 if mismatch == "parameters" else self.artifact["embedding_state_sha256_before"]
                retriever.return_value.tokenizer_fingerprint.return_value = "9" * 64 if mismatch == "tokenizer" else self.artifact["tokenizer_sha256"]
                with self.assertRaisesRegex(ModelUnavailable, "weights/tokenizer"):
                    ComparisonRunner.from_config(path)
                base.assert_not_called()

    def test_rehashed_score_rows_cannot_change_reviewed_population(self):
        score_path = self.root / "validation_cosine_scores.jsonl"
        rows = [json.loads(line) for line in score_path.read_text(encoding="utf-8").splitlines()]
        rows[0]["row_sha256"] = "9" * 64
        score_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.artifact["pair_scores_sha256"] = file_sha256(score_path)
        self.save()
        with self.assertRaisesRegex(ModelUnavailable, "provenance"):
            load_similarity_calibration(self.artifact_path, self.config)

    def test_scope_temperature_is_required_and_checked_before_loading_models(self):
        (self.root / "manifest.json").write_text("{}", encoding="utf-8")
        calibration = {"purpose": "validation_calibration", "split": "validation", "status": "frozen",
                       "threshold": .8, "ambiguity_margin": .03,
                       "scope_manifest_sha256": file_sha256(self.root / "manifest.json")}
        path = self.root / "inference.json"
        config = {**self.config, "scope_artifact": str(self.root), "calibration_artifact": str(self.artifact_path)}
        path.write_text(json.dumps(config), encoding="utf-8")
        for value in (None, True, 0, -1, float("nan"), float("inf")):
            self.artifact_path.write_text(json.dumps({**calibration, "temperature": value}), encoding="utf-8")
            with patch("editguard.comparison.FrozenGenerator") as generator:
                with self.assertRaisesRegex(ModelUnavailable, "temperature"):
                    ComparisonRunner.from_config(path)
                generator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
