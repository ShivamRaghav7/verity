import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from editguard.comparison import ComparisonRunner
from editguard.formatting import INPUT_FORMAT_VERSION
from editguard.neural import file_sha256
from scripts.calibrate_scope import fit_calibration as raw_fit, reviewed_inputs
from scripts.calibrate_scope_runtime import (COSTS, MARGINS, TEMPERATURES, THRESHOLDS,
    ROOT, ROUTING_POLICY, fit_calibration, frozen_inputs, operating_metrics, pair_eligible,
    safe_output, select_from_frozen)
from scripts.prepare_data import content_hash, write_dataset


class RuntimeOperatingTests(unittest.TestCase):
    def test_ineligible_high_scores_do_not_force_positive_clarification(self):
        labels = [1] * 48 + [0] * 72
        probabilities = [.55] * 48 + [.55] * 60 + [.3] * 12
        eligibility = [True] * 48 + [False] * 60 + [True] * 12
        raw_selection = raw_fit(labels, probabilities)
        fit = fit_calibration(labels, probabilities, eligibility)
        self.assertEqual(raw_selection["selected_operating_metrics"]["counts"]["true_application"], 0)
        self.assertEqual(fit["compatibility_gated_metrics"]["counts"]["true_application"], 48)
        self.assertEqual(fit["compatibility_gated_metrics"]["counts"]["false_application"], 0)
        self.assertEqual(fit["compatibility_gated_metrics"]["coverage"]["numerator"], 120)
        self.assertEqual(fit["raw_pair_metrics"]["counts"]["false_application"], 60)
        self.assertEqual(fit["raw_pair_metrics"]["false_application_rate"]["denominator"], 72)
        self.assertEqual(fit["compatibility_gated_metrics"]["false_application_rate"]["denominator"], 72)
        self.assertEqual(len(fit["temperature_candidates"]), len(TEMPERATURES))
        self.assertEqual(len(fit["operating_candidates"]), len(THRESHOLDS) * len(MARGINS))
        self.assertEqual(fit["temperature"], raw_selection["temperature"])
        self.assertEqual(fit["declared_costs"], COSTS)

    def test_compatibility_precedes_inclusive_uncertainty(self):
        raw = operating_metrics([1, 0], [.8, .5], .5, .1)
        gated = operating_metrics([1, 0], [.8, .5], .5, .1, [True, False])
        self.assertEqual(raw["counts"]["clarification_negative"], 1)
        self.assertEqual(gated["counts"]["clarification_negative"], 0)
        self.assertEqual(gated["counts"]["true_rejection"], 1)
        self.assertEqual(gated["coverage"]["denominator"], 2)

    def test_incompatible_positive_is_a_miss_not_an_excluded_row(self):
        gated = operating_metrics([1, 0], [.9, .9], .5, .1, [False, False])
        self.assertEqual(gated["counts"]["missed_positive"], 1)
        self.assertEqual(gated["counts"]["true_rejection"], 1)
        self.assertEqual(gated["recall_including_clarification_as_missed"]["denominator"], 1)
        self.assertEqual(gated["total_cost"], 1)

    def test_eligibility_ignores_scoring_truth(self):
        row = {"request_context": {"entity_id": "a", "plan": "Basic", "version": "v2", "as_of": "2026-10-11"},
            "correction": {"entity_id": "a", "plan": "Basic", "version": "v2", "relation": "device_limit",
                "effective_from": "2026-09-01", "effective_to": None, "approval_status": "approved"}}
        self.assertTrue(pair_eligible(row))
        self.assertTrue(pair_eligible({**row, "label": 0, "expected_fact": {"entity_id": "wrong", "relation": "price_monthly"}}))
        self.assertFalse(pair_eligible({**row, "correction": {**row["correction"], "approval_status": "draft"}}))
        self.assertFalse(pair_eligible({**row, "request_context": {**row["request_context"], "version": "v1"}}))
        self.assertFalse(pair_eligible({**row, "correction": {**row["correction"], "effective_to": "2026-10-11"}}))

    def test_invalid_compatibility_and_scores_are_refused(self):
        for eligibility in ([True], [True, 1]):
            with self.subTest(eligibility=eligibility), self.assertRaises(ValueError):
                fit_calibration([1, 0], [.9, .1], eligibility)
        with self.assertRaises(ValueError):
            fit_calibration([1, 0], [float("nan"), .1], [True, True])


class FrozenRuntimeInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset, self.scope = self.root / "data", self.root / "scope"
        # Synthetic unit corpus only. No repository test corpus is accessed.
        write_dataset(self.dataset, family_count=20)
        self.rows = [json.loads(line) for line in (self.dataset / "scope_validation.jsonl").read_text().splitlines()]
        self.reviews = self.root / "reviews.json"
        self.write_json(self.reviews, {"rows": {row["id"]: {"decision": "accept", "reviewer": "Synthetic unit fixture",
            "reviewed_at": "2026-10-13T12:00:00+05:30", "row_sha256": content_hash(row)} for row in self.rows}})
        self.scope.mkdir()
        for name in ("model/config.json", "model/model.safetensors", "tokenizer/tokenizer_config.json"):
            path = self.scope / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("{}" if name.endswith(".json") else "synthetic metadata fixture; never loaded", encoding="utf-8")
        self.scope_manifest = {"artifact_type": "editguard_auxiliary", "purpose": "trained", "status": "completed", "task": "scope",
            "dataset_id": json.loads((self.dataset / "manifest.json").read_text())["dataset_id"],
            "manifest_sha256": file_sha256(self.dataset / "manifest.json"), "reviews_sha256": file_sha256(self.reviews),
            "review_policy": "human", "human_reviewed_data": True, "input_format_version": INPUT_FORMAT_VERSION,
            "input_length": 192, "selected_model_state_sha256": "0" * 64,
            "files_sha256": {str(path.relative_to(self.scope)).replace("\\", "/"): file_sha256(path)
                             for path in self.scope.rglob("*") if path.is_file()}}
        self.write_json(self.scope / "manifest.json", self.scope_manifest)
        self.source = self.root / "legacy"
        self.source.mkdir()
        _, bindings = reviewed_inputs(self.scope, self.dataset, self.reviews, max_rows=10000)
        self.raw = [{"id": row["id"], "family_id": row["family_id"], "split": "validation", "label": row["label"],
            "row_sha256": content_hash(row), "scope_probability": .9 if row["label"] else .1,
            "review_policy": "human", "human_reviewed_data": True} for row in self.rows]
        self.raw_path = self.source / "validation_probabilities.jsonl"
        self.raw_path.write_text("".join(json.dumps(record) + "\n" for record in self.raw), encoding="utf-8")
        fit = raw_fit([row["label"] for row in self.raw], [row["scope_probability"] for row in self.raw])
        self.write_json(self.source / "selection_grid.json", fit)
        self.source_manifest = {**bindings, "purpose": "validation_calibration", "split": "validation", "status": "frozen",
            "temperature": fit["temperature"], "threshold": fit["threshold"], "ambiguity_margin": fit["ambiguity_margin"],
            "declared_costs": fit["declared_costs"], "script_sha256": file_sha256(ROOT / "scripts/calibrate_scope.py"),
            "selection": {"temperatures": list(TEMPERATURES), "thresholds": list(THRESHOLDS), "margins": list(MARGINS)},
            "probabilities_sha256": file_sha256(self.raw_path),
            "selection_grid_sha256": file_sha256(self.source / "selection_grid.json")}
        self.source_path = self.source / "calibration.json"
        self.write_json(self.source_path, self.source_manifest)
        self.output = self.root / "new"

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def inputs(self):
        return frozen_inputs(self.scope, self.dataset, self.reviews, self.source_path, max_rows=10000)

    def select(self, **kwargs):
        return select_from_frozen(self.scope, self.dataset, self.reviews, self.source_path, self.output,
                                 max_rows=10000, **kwargs)

    def rehash_raw(self):
        self.source_manifest["probabilities_sha256"] = file_sha256(self.raw_path)
        self.write_json(self.source_path, self.source_manifest)

    def test_complete_source_integrity_and_no_test_reads(self):
        for path in self.dataset.glob("*test.jsonl"):
            path.write_text("INTENTIONALLY UNREADABLE SYNTHETIC TEST", encoding="utf-8")
        rows, raw, eligible, bindings = self.inputs()
        self.assertEqual(len(rows), len(self.rows))
        self.assertEqual(len(raw), len(eligible))
        self.assertFalse(bindings["test_rows_read"])
        self.assertFalse(bindings["model_execution_requested"])
        self.assertEqual(bindings["source_probabilities_sha256"], file_sha256(self.raw_path))

    def test_source_probabilities_and_grid_tampering_are_denied(self):
        self.raw_path.write_text(self.raw_path.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.inputs()
        self.rehash_raw()
        (self.source / "selection_grid.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.inputs()

    def test_current_registry_must_match_training_registry(self):
        self.reviews.write_text(self.reviews.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "registry bound to scope training"):
            self.inputs()

    def test_source_binds_exact_current_dataset_and_model(self):
        self.source_manifest["scope_manifest_sha256"] = "f" * 64
        self.write_json(self.source_path, self.source_manifest)
        with self.assertRaisesRegex(ValueError, "scope_manifest_sha256"):
            self.inputs()

    def test_duplicate_missing_and_wrong_split_records_are_denied_even_when_rehashed(self):
        variants = [self.raw + [self.raw[0]], self.raw[1:], [{**self.raw[0], "split": "test"}, *self.raw[1:]]]
        for records in variants:
            with self.subTest(record_count=len(records)):
                self.raw_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
                self.rehash_raw()
                with self.assertRaises(ValueError):
                    self.inputs()

    def test_fresh_outputs_preserve_raw_bytes_and_existing_loader_accepts_new_artifact(self):
        source_hash = file_sha256(self.source_path)
        artifact = self.select()
        self.assertEqual(file_sha256(self.source_path), source_hash)
        self.assertEqual((self.output / "validation_probabilities.jsonl").read_bytes(), self.raw_path.read_bytes())
        self.assertEqual(artifact["routing_policy"], ROUTING_POLICY)
        self.assertEqual(artifact["scope_manifest_sha256"], file_sha256(self.scope / "manifest.json"))
        self.assertEqual(artifact["validation_operating_metrics"], artifact["compatibility_gated_metrics"])
        self.assertEqual(artifact["raw_pair_metrics"]["counts"]["total"], len(self.rows))
        config = {"input_format_version": INPUT_FORMAT_VERSION, "scope_artifact": str(self.scope),
            "calibration_artifact": str(self.output / "calibration.json"), "answer_artifact": None,
            "retrieval_k": 3, "prompt_edit_k": 3, "max_input_tokens": 384, "max_new_tokens": 48,
            "device": "cpu", "local_files_only": True, "cache_dir": "synthetic-cache", "embedding_model_id": "fixture",
            "embedding_revision": "0" * 40, "base_model_id": "fixture", "base_revision": "0" * 40}
        path = self.root / "inference.json"
        self.write_json(path, config)
        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass
        with patch("editguard.comparison.EmbeddingRetriever", FakeModel), patch("editguard.comparison.FrozenGenerator", FakeModel), \
             patch("editguard.comparison.ScopeCheckpoint", FakeModel):
            runner = ComparisonRunner.from_config(path)
        self.assertEqual(runner.calibration["threshold"], artifact["threshold"])
        with self.assertRaisesRegex(ValueError, "fresh directory"):
            self.select()

    def test_source_mutation_during_selection_prevents_publication(self):
        original = fit_calibration
        def mutate(*args):
            fit = original(*args)
            self.raw_path.write_text(self.raw_path.read_text() + "\n", encoding="utf-8")
            return fit
        with patch("scripts.calibrate_scope_runtime.fit_calibration", mutate), self.assertRaisesRegex(ValueError, "integrity"):
            self.select()
        self.assertFalse(self.output.exists())

    def test_safety_and_preflight_never_overwrite_sources(self):
        for output in (self.dataset, self.scope / "new", self.source, self.source / "new", self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                safe_output(output, self.scope, self.dataset, self.reviews, self.source_path)
        result = self.select(preflight_only=True)
        self.assertTrue(result["preflight_valid"])
        self.assertFalse(self.output.exists())

    def test_import_and_cli_selection_do_not_import_ml(self):
        command = "import sys; import scripts.calibrate_scope_runtime; " \
            "assert not any(k == 'torch' or k.startswith('torch.') or k == 'transformers' or k.startswith('transformers.') " \
            "or k == 'numpy' for k in sys.modules)"
        completed = subprocess.run([sys.executable, "-c", command], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        command += "; from scripts.calibrate_scope_runtime import main; main()"
        completed = subprocess.run([sys.executable, "-c", command, "--scope-artifact", str(self.scope),
            "--dataset", str(self.dataset), "--reviews", str(self.reviews), "--source-calibration", str(self.source_path),
            "--output", str(self.output), "--max-rows", "10000"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((self.output / "calibration.json").is_file())


if __name__ == "__main__":
    unittest.main()
