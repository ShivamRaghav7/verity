"""Metadata-only API readiness fixtures; synthetic files are never model-loaded."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from editguard.api import ArtifactStatusCache, create_app
from editguard.formatting import INPUT_FORMAT_VERSION
from editguard.neural import file_sha256, validate_artifact

ROOT = Path(__file__).resolve().parents[1]


def runtime_policy_fixture(case, total=10, positives=4):
    """Synthetic saved probabilities only; no model or project data is loaded."""
    from scripts.calibrate_scope_runtime import ROUTING_POLICY, fit_calibration
    from scripts.prepare_data import content_hash
    labels = [1] * positives + [0] * (total - positives)
    probabilities = [.55] * (total - 1) + [.1]
    fit = fit_calibration(labels, probabilities, [True] * positives + [False] * (total - positives - 1) + [True])
    records = [{"id": f"synthetic-pair-{index}", "split": "validation", "label": label,
                "scope_probability": probability, "row_sha256": content_hash({"synthetic_index": index})}
               for index, (label, probability) in enumerate(zip(labels, probabilities))]
    raw_path, grid_path = case.root / "validation_probabilities.jsonl", case.root / "selection_grid.json"
    raw_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    grid = {**fit, "routing_policy": ROUTING_POLICY}
    grid_path.write_text(json.dumps(grid), encoding="utf-8")
    source_names = ("editguard/comparison.py", "editguard/neural.py", "editguard/formatting.py",
                    "scripts/calibrate_scope.py", "scripts/calibrate_similarity.py")
    calibration = {"purpose": "validation_calibration", "status": "frozen", "split": "validation", "test_rows_read": False,
        "dataset_id": case.manifests["scope"]["dataset_id"], "population": "single_focal_question_correction_pairs",
        "routing_policy": ROUTING_POLICY, "scope_manifest_sha256": file_sha256(case.root / "scope/manifest.json"),
        "threshold": fit["threshold"], "ambiguity_margin": fit["ambiguity_margin"], "temperature": fit["temperature"],
        "raw_pair_metrics": fit["raw_pair_metrics"], "compatibility_gated_metrics": fit["compatibility_gated_metrics"],
        "validation_operating_metrics": fit["compatibility_gated_metrics"], "declared_costs": fit["declared_costs"],
        "accepted_validation_rows": total, "accepted_row_sha256": {record["id"]: record["row_sha256"] for record in records},
        "probabilities_sha256": file_sha256(raw_path), "selection_grid_sha256": file_sha256(grid_path),
        "policy_source_sha256": {name: file_sha256(ROOT / name) for name in source_names},
        "script_sha256": file_sha256(ROOT / "scripts/calibrate_scope_runtime.py")}
    case.calibration.write_text(json.dumps(calibration), encoding="utf-8")
    return calibration


class ModelStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = json.loads((ROOT / "configs/inference.json").read_text())
        self.config["input_format_version"] = INPUT_FORMAT_VERSION
        self.config_path = self.root / "inference.json"
        self.manifests = {}
        for task, steps in (("scope", 360), ("answer", 240)):
            folder = self.root / task
            for name, value in (("model/config.json", "{}"), ("model/model.safetensors", "synthetic fixture, not weights"),
                                ("tokenizer/tokenizer_config.json", "{}")):
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value, encoding="utf-8")
            manifest = {"artifact_type": "editguard_auxiliary", "purpose": "trained", "status": "completed", "task": task,
                        "input_format_version": INPUT_FORMAT_VERSION, "input_length": 192,
                        "review_policy": "automated_development", "human_reviewed_data": False,
                        "completed_steps": steps, "selection": {"global_step": 120 if task == "scope" else 240},
                        "dataset_id": "synthetic-status-fixture", "model_id": "synthetic-model-only",
                        "files_sha256": {file.relative_to(folder).as_posix(): file_sha256(file)
                                         for file in folder.rglob("*") if file.is_file()}}
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.manifests[task] = manifest
            self.config[f"{task}_artifact"] = str(folder)
        self.calibration = self.root / "calibration.json"
        self.calibration.write_text(json.dumps({"purpose": "validation_calibration", "split": "validation", "status": "frozen",
            "threshold": .8, "ambiguity_margin": .03, "temperature": 1,
            "scope_manifest_sha256": file_sha256(self.root / "scope/manifest.json")}), encoding="utf-8")
        self.config["calibration_artifact"] = str(self.calibration)
        self.save_config()

    def save_config(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def client(self):
        client = TestClient(create_app(self.root / "status.sqlite3", "test-token", inference_config=self.config_path,
                                     final_evaluation_report=self.root / "not-yet-exported-final.json"))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_health_reports_validated_trained_artifacts_without_loading_models(self):
        client = self.client()
        with patch("editguard.comparison.ComparisonRunner.from_config", side_effect=AssertionError("No model execution")):
            health = client.get("/api/health").json()
        self.assertEqual(health["models"], {"scope": "trained", "answer": "trained"})
        self.assertEqual(health["neural_inference"], "configured_not_loaded")
        self.assertEqual(health["modes"]["learned"], {"status": "load_required", "available": None})
        for task in ("scope", "answer"):
            details = health["model_details"][task]
            self.assertFalse(details["loaded"])
            self.assertEqual(details["review_policy"], "automated_development")
            self.assertFalse(details["human_reviewed_data"])
            self.assertEqual(details["completed_steps"], self.manifests[task]["completed_steps"])
            self.assertEqual(details["artifact_manifest_sha256"], file_sha256(self.root / task / "manifest.json"))
        self.assertEqual(health["answer_mode"], "deterministic_structured")
        self.assertEqual(health["final_evaluation"], "pending")

    def test_health_hash_validation_is_cached_and_file_changes_invalidate_it(self):
        client = self.client()
        with patch("editguard.neural.validate_artifact", wraps=validate_artifact) as validate:
            self.assertEqual(client.get("/api/health").json()["models"]["scope"], "trained")
            client.get("/api/health")
            self.assertEqual(validate.call_count, 2)
            file = self.root / "scope/model/model.safetensors"
            original = file.read_bytes()
            file.write_bytes(b"corrupted synthetic fixture")
            self.assertEqual(client.get("/api/health").json()["models"]["scope"], "invalid")
            self.assertEqual(validate.call_count, 4)
            file.write_bytes(original)
            self.assertEqual(client.get("/api/health").json()["models"]["scope"], "trained")
            self.assertEqual(validate.call_count, 6)

    def test_routing_policy_metadata_checks_source_files_without_model_execution(self):
        runtime_policy_fixture(self)
        client = self.client()
        with patch("editguard.comparison.ComparisonRunner.from_config", side_effect=AssertionError("No model execution")):
            self.assertEqual(client.get("/api/health").json()["calibrations"]["scope"], "validated")
            raw = self.root / "validation_probabilities.jsonl"
            original = raw.read_bytes()
            raw.write_bytes(original + b"\n")
            self.assertEqual(client.get("/api/health").json()["calibrations"]["scope"], "invalid")
            raw.write_bytes(original)
            self.assertEqual(client.get("/api/health").json()["calibrations"]["scope"], "validated")

    def test_unknown_routing_policy_and_unbound_policy_source_are_invalid(self):
        calibration = runtime_policy_fixture(self)
        client = self.client()
        original = json.loads(json.dumps(calibration))
        for change in (lambda value: value.update(routing_policy="unknown"),
                       lambda value: value["policy_source_sha256"].update({"editguard/neural.py": "f" * 64})):
            calibration = json.loads(json.dumps(original))
            change(calibration)
            self.calibration.write_text(json.dumps(calibration), encoding="utf-8")
            self.assertEqual(client.get("/api/health").json()["calibrations"]["scope"], "invalid")

    def test_missing_invalid_and_smoke_artifacts_keep_learned_mode_unavailable(self):
        client = self.client()
        for state in ("missing", "invalid", "smoke"):
            self.config["answer_artifact"] = str(self.root / "missing-answer")
            if state != "missing":
                self.config["answer_artifact"] = str(self.root / "answer")
                manifest = {**self.manifests["answer"], "purpose": "smoke_test"} if state == "smoke" else {**self.manifests["answer"], "input_format_version": "wrong-format"}
                (self.root / "answer/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.save_config()
            with self.subTest(state=state), patch("editguard.comparison.ComparisonRunner.from_config", side_effect=AssertionError("Must refuse before model load")):
                health = client.get("/api/health").json()
                self.assertEqual(health["models"]["answer"], "missing" if state == "missing" else "invalid")
                self.assertEqual(health["modes"]["learned"]["available"], False)
                self.assertEqual(client.post("/api/ask", json={"question": "Basic devices", "mode": "learned"}).status_code, 503)
                self.assertEqual(client.post("/api/ask", json={"question": "Basic devices", "entity_id": "nimbus"}).status_code, 200)

    def test_calibration_missing_or_wrong_binding_does_not_hide_trained_artifacts(self):
        client = self.client()
        for calibration in (str(self.root / "missing-calibration"), str(self.calibration)):
            self.config["calibration_artifact"] = calibration
            self.save_config()
            if calibration == str(self.calibration):
                data = json.loads(self.calibration.read_text())
                data["scope_manifest_sha256"] = "0" * 64
                self.calibration.write_text(json.dumps(data), encoding="utf-8")
            with self.subTest(calibration=calibration):
                health = client.get("/api/health").json()
                self.assertEqual(health["models"]["scope"], "trained")
                self.assertEqual(health["modes"]["learned"]["status"], "unavailable")
                self.assertEqual(health["calibrations"]["scope"], "invalid" if calibration == str(self.calibration) else "missing")

    def test_load_failure_is_truthful_and_injected_fakes_are_not_called_trained(self):
        client = self.client()
        with patch("editguard.comparison.ComparisonRunner.from_config", side_effect=RuntimeError("Synthetic unavailable device")):
            response = client.post("/api/ask", json={"question": "Basic devices", "mode": "learned"})
        self.assertEqual(response.status_code, 503)
        health = client.get("/api/health").json()
        self.assertEqual(health["neural_inference"], "unavailable")
        self.assertEqual(health["load_error"], "Synthetic unavailable device")
        self.assertEqual(health["models"]["scope"], "trained")
        from tests.test_comparison import FakeGenerator, FakeRetriever, FakeScope
        from editguard.comparison import ComparisonRunner
        runner = ComparisonRunner(FakeGenerator(), FakeRetriever(), scope=FakeScope(.95), answer_model=FakeGenerator(),
                                  calibration={"threshold": .8, "ambiguity_margin": .03, "temperature": 1})
        app = create_app(self.root / "injected.sqlite3", "test-token", runner,
                         final_evaluation_report=self.root / "not-yet-exported-final.json")
        with TestClient(app) as injected:
            self.assertEqual(injected.get("/api/health").json()["models"]["scope"], "loaded_unvalidated")

    def test_loaded_answer_provenance_and_config_change_refusal_use_synthetic_runner(self):
        from tests.test_comparison import FakeGenerator, FakeRetriever, FakeScope
        from editguard.comparison import ComparisonRunner
        scope, answer = FakeScope(.95), FakeGenerator()
        scope.manifest, answer.manifest = self.manifests["scope"], self.manifests["answer"]
        answer.artifact_manifest_sha256 = file_sha256(self.root / "answer/manifest.json")
        runner = ComparisonRunner(FakeGenerator(), FakeRetriever(), scope=scope, answer_model=answer,
                                  calibration={"threshold": .8, "ambiguity_margin": .03, "temperature": 1})
        client = self.client()
        ask = {"question": "How many devices does Nimbus Basic version v2 support?", "entity_id": "nimbus", "plan": "Basic",
               "version": "v2", "as_of": "2026-10-12", "mode": "learned"}
        with patch("editguard.comparison.ComparisonRunner.from_config", return_value=runner):
            response = client.post("/api/ask", json=ask)
        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()
        self.assertEqual(record["answer_kind"], "model_generated")
        self.assertEqual(record["model_training"]["answer"]["review_policy"], "automated_development")
        self.assertFalse(record["model_training"]["answer"]["human_reviewed_data"])
        self.assertTrue(record["model_training"]["answer"]["loaded"])
        self.assertEqual(client.get("/api/health").json()["neural_inference"], "loaded")
        self.config["max_new_tokens"] += 1
        self.save_config()
        self.assertEqual(client.get("/api/health").json()["neural_inference"], "configuration_changed")
        self.assertEqual(client.post("/api/ask", json=ask).status_code, 503)

    def test_ui_has_dynamic_provenance_and_separate_answer_kinds(self):
        page = (ROOT / "editguard/static/index.html").read_text(encoding="utf-8")
        script = (ROOT / "editguard/static/app.js").read_text(encoding="utf-8")
        self.assertNotIn("model are untrained", page)
        for field in ("model-stage-title", "scope-model-description", "answer-model-description"):
            self.assertIn('id="' + field + '"', page)
        self.assertIn("Configured auxiliary training provenance", script)
        self.assertIn("no human review", script)
        self.assertIn("deterministic_structured", script)
        self.assertIn("model_generated", script)

    def loaded_synthetic_runner(self, client):
        from tests.test_comparison import FakeGenerator, FakeRetriever, FakeScope
        from editguard.comparison import ComparisonRunner
        scope, answer = FakeScope(.95), FakeGenerator()
        scope.manifest, answer.manifest = self.manifests["scope"], self.manifests["answer"]
        scope.artifact_manifest_sha256 = file_sha256(self.root / "scope/manifest.json")
        answer.artifact_manifest_sha256 = file_sha256(self.root / "answer/manifest.json")
        runner = ComparisonRunner(FakeGenerator(), FakeRetriever(), scope=scope, answer_model=answer,
                                  calibration={"threshold": .8, "ambiguity_margin": .03, "temperature": 1})
        ask = {"question": "How many devices does Nimbus Basic version v2 support?", "entity_id": "nimbus", "plan": "Basic",
               "version": "v2", "as_of": "2026-10-12", "mode": "learned"}
        with patch("editguard.comparison.ComparisonRunner.from_config", return_value=runner):
            response = client.post("/api/ask", json=ask)
        self.assertEqual(response.status_code, 200, response.text)
        return ask

    def test_changed_or_corrupted_artifact_does_not_get_overlaid_by_loaded_provenance(self):
        client = self.client()
        ask = self.loaded_synthetic_runner(client)
        self.assertFalse(client.get("/api/health").json()["restart_required"])
        manifest_path = self.root / "answer/manifest.json"
        original_manifest = manifest_path.read_bytes()
        changed = json.loads(original_manifest)
        changed["selection"]["global_step"] = 120
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        health = client.get("/api/health").json()
        self.assertTrue(health["restart_required"])
        self.assertEqual(health["neural_inference"], "configuration_changed")
        self.assertFalse(health["model_details"]["answer"]["loaded"])
        self.assertEqual(health["model_details"]["answer"]["selected_step"], 120)
        self.assertEqual(health["model_details"]["answer"]["artifact_manifest_sha256"], file_sha256(manifest_path))
        self.assertEqual(client.post("/api/ask", json=ask).status_code, 503)
        manifest_path.write_bytes(original_manifest)
        weights = self.root / "answer/model/model.safetensors"
        weights.write_bytes(b"corrupted fixture after load")
        health = client.get("/api/health").json()
        self.assertTrue(health["restart_required"])
        self.assertEqual(health["models"]["answer"], "invalid")
        self.assertEqual(client.post("/api/ask", json=ask).status_code, 503)
        self.assertEqual(client.post("/api/ask", json={"question": "Basic devices", "entity_id": "nimbus"}).status_code, 200)

    def test_changed_or_invalid_calibration_requires_restart_after_load(self):
        client = self.client()
        ask = self.loaded_synthetic_runner(client)
        original = self.calibration.read_bytes()
        changed = json.loads(original)
        changed["temperature"] = 2
        self.calibration.write_text(json.dumps(changed), encoding="utf-8")
        health = client.get("/api/health").json()
        self.assertTrue(health["restart_required"])
        self.assertEqual(health["calibrations"]["scope"], "validated")
        self.assertFalse(health["modes"]["learned"]["available"])
        self.assertEqual(client.post("/api/ask", json=ask).status_code, 503)
        self.calibration.write_bytes(original)
        self.calibration.unlink()
        health = client.get("/api/health").json()
        self.assertTrue(health["restart_required"])
        self.assertEqual(health["calibrations"]["scope"], "missing")
        self.assertEqual(client.post("/api/ask", json=ask).status_code, 503)

    def test_configuration_race_during_loading_is_refused(self):
        client = self.client()
        from tests.test_comparison import FakeGenerator, FakeRetriever
        from editguard.comparison import ComparisonRunner

        def changed_during_load(path):
            self.config["max_new_tokens"] += 1
            self.save_config()
            return ComparisonRunner(FakeGenerator(), FakeRetriever())

        with patch("editguard.comparison.ComparisonRunner.from_config", side_effect=changed_during_load):
            response = client.post("/api/ask", json={"question": "Basic devices", "entity_id": "nimbus", "mode": "updated_rag"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("changed while loading", response.json()["detail"])
        self.assertEqual(client.get("/api/health").json()["neural_inference"], "unavailable")


class DevelopmentReportTests(unittest.TestCase):
    """Synthetic recorded results exercise disclosure and evidence binding only."""

    save_config = ModelStatusTests.save_config
    loaded_synthetic_runner = ModelStatusTests.loaded_synthetic_runner

    def setUp(self):
        ModelStatusTests.setUp(self)
        self.raw_path = self.root / "raw/manifest.json"
        self.raw_path.parent.mkdir()
        predictions = self.raw_path.parent / "predictions.jsonl"
        predictions.write_text('{"synthetic": true}\n', encoding="utf-8")
        self.report_path = self.root / "development-summary.json"

        def rate(numerator, denominator):
            return {"numerator": numerator, "denominator": denominator,
                    "value": numerator / denominator if denominator else None}

        from editguard.comparison import CONFIGURATIONS
        self.report = {"artifact_type": "editguard_development_evaluation", "purpose": "validation_development",
            "status": "completed", "review_policy": "automated_development", "human_reviewed_data": False,
            "test_rows_read": False, "dataset_id": "synthetic-status-fixture", "questions": 10, "oracle_questions": 4,
            "model_artifacts": {task: {"manifest_sha256": file_sha256(self.root / task / "manifest.json"),
                "completed_steps": manifest["completed_steps"], "selected_step": manifest["selection"]["global_step"]}
                for task, manifest in self.manifests.items()},
            "scope": {"precision": rate(2, 3), "recall": rate(2, 4), "false_positive_rate": rate(1, 6),
                      "brier": rate(.5, 10), "ece": rate(1, 10)},
            "oracle": {"accuracy": rate(3, 4), "direct_accuracy": rate(1, 1), "paraphrase_accuracy": rate(1, 1),
                       "comparison_accuracy": rate(1, 2), "coverage": rate(4, 4)},
            "configurations": [{"configuration": name, "accuracy": rate(4, 10), "coverage": rate(8, 10),
                "out_of_scope_accuracy": rate(2, 6), "baseline_correct_retention": rate(1, 2),
                "false_application_rate": rate(1, 6), "p50_ms": 12.5, "p95_ms": 20.5,
                "pending_review": 3, "errors": 1, "unavailable": 0} for name in CONFIGURATIONS],
            "limitations": ["Synthetic fixture numbers are not measured project results.",
                "Strict full-assertion scoring is a conservative lower bound; pending fragments receive no credit."],
            "provenance": {"config_sha256": file_sha256(self.config_path), "scorer_version": "synthetic-scorer",
                "raw_manifest_path": str(self.raw_path), "predictions_sha256": file_sha256(predictions)}}
        raw = {key: self.report[key] for key in ("purpose", "status", "test_rows_read", "human_reviewed_data",
                                                "review_policy", "dataset_id", "questions")}
        raw.update(configuration_sha256=self.report["provenance"]["config_sha256"],
                   scorer_version=self.report["provenance"]["scorer_version"],
                   predictions_sha256=self.report["provenance"]["predictions_sha256"])
        self.raw_path.write_text(json.dumps(raw), encoding="utf-8")
        self.report["provenance"]["raw_manifest_sha256"] = file_sha256(self.raw_path)
        self.save_report()

    def save_report(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")

    def add_scope_policy(self):
        calibration = runtime_policy_fixture(self)
        self.report["scope_policy"] = {key: calibration[key] for key in ("temperature", "threshold", "ambiguity_margin",
            "routing_policy", "population", "scope_manifest_sha256", "raw_pair_metrics", "compatibility_gated_metrics")}
        self.report["scope_policy"].update(calibration_path=str(self.calibration), calibration_sha256=file_sha256(self.calibration))
        self.save_report()
        return calibration

    def client(self):
        client = TestClient(create_app(self.root / "report.sqlite3", "test-token", inference_config=self.config_path,
                                     evaluation_report=self.report_path,
                                     final_evaluation_report=self.root / "not-yet-exported-final.json"))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_optional_report_preserves_denominators_provenance_and_live_memory(self):
        client = self.client()
        before = client.get("/api/edits").json()
        with patch("editguard.comparison.ComparisonRunner.from_config", side_effect=AssertionError("No neural execution")):
            result = client.get("/api/evaluation").json()
            exported = client.get("/api/export").json()
        self.assertEqual(result["ml_report"]["status"], "available", result["ml_report"])
        self.assertTrue(result["ml_report"]["configuration_matches"])
        self.assertEqual(result["ml_results"], self.report)
        self.assertEqual(exported["evaluation"]["ml_results"], self.report)
        self.assertEqual(client.get("/api/edits").json(), before)
        self.assertEqual(result["final_evaluation"], "pending")
        self.assertEqual(result["ml_results"]["configurations"][0]["accuracy"]["denominator"], 10)
        self.assertEqual(result["ml_results"]["configurations"][0]["pending_review"], 3)

    def test_scope_policy_separates_raw_failures_from_deterministic_gates(self):
        self.add_scope_policy()
        client = self.client()
        with patch("editguard.comparison.ComparisonRunner.from_config", side_effect=AssertionError("No model execution")):
            result = client.get("/api/evaluation").json()
        self.assertEqual(result["ml_report"]["status"], "available", result["ml_report"])
        policy = result["ml_results"]["scope_policy"]
        self.assertEqual(policy["raw_pair_metrics"]["counts"]["false_application"], 5)
        self.assertEqual(policy["compatibility_gated_metrics"]["counts"]["false_application"], 0)
        self.assertEqual(policy["raw_pair_metrics"]["false_application_rate"]["denominator"], 6)
        self.assertEqual(policy["compatibility_gated_metrics"]["false_application_rate"]["denominator"], 6)

    def test_scope_policy_omissions_and_altered_denominators_are_rejected(self):
        self.add_scope_policy()
        client = self.client()
        original = json.loads(json.dumps(self.report))
        mutations = (lambda value: value["scope_policy"].pop("raw_pair_metrics"),
            lambda value: value["scope_policy"]["compatibility_gated_metrics"]["counts"].update(total=8),
            lambda value: value["scope_policy"]["raw_pair_metrics"]["false_application_rate"].update(denominator=10, value=.5),
            lambda value: value["scope_policy"].update(scope_manifest_sha256="f" * 64),
            lambda value: value["scope_policy"].update(calibration_sha256="f" * 64))
        for mutation in mutations:
            self.report = json.loads(json.dumps(original))
            mutation(self.report)
            self.save_report()
            self.assertIsNone(client.get("/api/evaluation").json()["ml_results"])

    def test_scope_policy_counts_and_clarifications_require_consistent_rates(self):
        from editguard.api import DevelopmentReportCache
        self.add_scope_policy()
        metrics = self.report["scope_policy"]["raw_pair_metrics"]
        for change in (lambda value: value["counts"].update(clarification_positive=1),
                       lambda value: value["precision"].update(value=1),
                       lambda value: value["false_application_rate"].update(numerator=0, denominator=6, value=0),
                       lambda value: value["coverage"].update(numerator=9, denominator=9, value=1)):
            changed = json.loads(json.dumps(metrics))
            change(changed)
            with self.assertRaises(ValueError):
                DevelopmentReportCache.scope_policy_metrics(changed, 10, 4)

    def test_scope_policy_must_match_current_configured_calibration(self):
        calibration = self.add_scope_policy()
        client = self.client()
        self.assertIsNotNone(client.get("/api/evaluation").json()["ml_results"])
        calibration["synthetic_new_snapshot"] = True
        different = self.root / "different-calibration.json"
        different.write_text(json.dumps(calibration), encoding="utf-8")
        self.config["calibration_artifact"] = str(different)
        self.save_config()
        result = client.get("/api/evaluation").json()
        self.assertIsNone(result["ml_results"])
        self.assertEqual(result["ml_report"]["status"], "incompatible")

    def test_scope_policy_evidence_changes_invalidate_cached_report(self):
        self.add_scope_policy()
        client = self.client()
        self.assertIsNotNone(client.get("/api/evaluation").json()["ml_results"])
        raw = self.root / "validation_probabilities.jsonl"
        raw.write_text(raw.read_text() + "\n", encoding="utf-8")
        result = client.get("/api/evaluation").json()
        self.assertIsNone(result["ml_results"])
        self.assertEqual(result["ml_report"]["status"], "invalid")

    def test_missing_corrupted_or_changed_model_evidence_suppresses_accuracy(self):
        client = self.client()
        self.assertIsNotNone(client.get("/api/evaluation").json()["ml_results"])
        predictions = self.raw_path.parent / "predictions.jsonl"
        original = predictions.read_bytes()
        predictions.write_bytes(b'{"synthetic": "modified evidence"}\n')
        result = client.get("/api/evaluation").json()
        self.assertIsNone(result["ml_results"])
        self.assertEqual(result["ml_report"]["status"], "invalid")
        predictions.write_bytes(original)
        self.assertIsNotNone(client.get("/api/evaluation").json()["ml_results"])
        manifest_path = self.root / "answer/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["selection"]["global_step"] = 120
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = client.get("/api/evaluation").json()
        self.assertIsNone(result["ml_results"])
        self.assertEqual(result["ml_report"]["status"], "incompatible")
        self.report_path.unlink()
        self.assertEqual(client.get("/api/evaluation").json()["ml_report"]["status"], "missing")

    def test_config_budget_change_is_disclosed_without_relabeling_recorded_results(self):
        client = self.client()
        self.config["max_new_tokens"] += 1
        self.save_config()
        result = client.get("/api/evaluation").json()
        self.assertEqual(result["ml_results"], self.report)
        self.assertFalse(result["ml_report"]["configuration_matches"])

    def test_unavailable_metric_keeps_population_and_requires_reason(self):
        client = self.client()
        metric = self.report["configurations"][0]["false_application_rate"]
        metric.update(numerator=0, value=None, reason="This generator has no explicit correction routing decision.")
        self.save_report()
        result = client.get("/api/evaluation").json()
        self.assertIsNone(result["ml_results"]["configurations"][0]["false_application_rate"]["value"])
        self.assertEqual(result["ml_results"]["configurations"][0]["false_application_rate"]["denominator"], 6)
        metric.pop("reason")
        self.save_report()
        result = client.get("/api/evaluation").json()
        self.assertIsNone(result["ml_results"])
        self.assertEqual(result["ml_report"]["status"], "invalid")

    def test_changed_loaded_calibration_suppresses_previously_matching_report(self):
        client = self.client()
        self.loaded_synthetic_runner(client)
        self.assertIsNotNone(client.get("/api/evaluation").json()["ml_results"])
        calibration = json.loads(self.calibration.read_text())
        calibration["temperature"] = 2
        self.calibration.write_text(json.dumps(calibration), encoding="utf-8")
        result = client.get("/api/evaluation").json()
        self.assertIsNone(result["ml_results"])
        self.assertEqual(result["ml_report"]["status"], "incompatible")
        self.assertTrue(result["model_readiness"]["restart_required"])

    def test_invalid_populations_and_review_claims_are_rejected(self):
        client = self.client()
        original = json.loads(json.dumps(self.report))
        cases = {
            "human_claim": lambda report: report.update(human_reviewed_data=True),
            "test_claim": lambda report: report.update(test_rows_read=True),
            "omitted_baseline": lambda report: report["configurations"].pop(),
            "hidden_abstentions": lambda report: report["configurations"][0]["accuracy"].update(numerator=4, denominator=8, value=.5),
            "false_application_denominator": lambda report: report["configurations"][0]["false_application_rate"].update(numerator=1, denominator=10, value=.1),
            "invented_rate": lambda report: report["oracle"]["accuracy"].update(value=1),
            "overlap_oracle_groups": lambda report: report["oracle"]["direct_accuracy"].update(numerator=1, denominator=2, value=.5),
            "invalid_latency": lambda report: report["configurations"][0].update(p50_ms=100),
        }
        for name, mutate in cases.items():
            self.report = json.loads(json.dumps(original))
            mutate(self.report)
            self.save_report()
            with self.subTest(name=name):
                result = client.get("/api/evaluation").json()
                self.assertIsNone(result["ml_results"])
                self.assertEqual(result["ml_report"]["status"], "invalid")

    def test_ui_explains_conservative_scoring_and_exports_recorded_evidence(self):
        page = (ROOT / "editguard/static/index.html").read_text(encoding="utf-8")
        script = (ROOT / "editguard/static/app.js").read_text(encoding="utf-8")
        self.assertIn("Strict full-assertion scoring is a conservative lower bound", page)
        self.assertIn("Incomplete generated fragments await review", page)
        self.assertIn('id="download-ml-report"', page)
        self.assertIn("Baseline-correct retention", script)
        self.assertIn("final held-out evidence appears separately", script)
        self.assertIn("configuration_matches === false", script)
        self.assertIn('id="scope-policy-panel"', page)
        self.assertIn("does not measure learned boundary accuracy", page)
        self.assertIn("full memory routing", page)
        self.assertIn("Raw classifier + uncertainty", script)
        self.assertIn("Approval & compatibility gates + classifier", script)


if __name__ == "__main__":
    unittest.main()
