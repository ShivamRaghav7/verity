"""Synthetic sealed evidence only; no final data or neural inference is read."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from editguard.api import ArtifactStatusCache, FinalReportCache, OracleRecoveryReportCache, create_app
from editguard.comparison import CONFIGURATIONS


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def population(count, applicable, names=CONFIGURATIONS):
    rows = []
    for name in names:
        latency = {"total_count": count, "sample_count": count, "missing_count": 0,
                   "complete": True, "p50_ms": 3.0, "p95_ms": 7.0}
        rows.append({"configuration": name, "questions": count, "applicable_questions": applicable,
            "accuracy": rate(0, count), "coverage": rate(count, count),
            "model_generated_answer_accuracy": rate(0, count), "model_generated_answer_coverage": rate(count, count),
            "out_of_scope_accuracy": rate(0, count - applicable), "false_application_rate": rate(0, count - applicable),
            "output_preservation": rate(0, count - applicable), "baseline_correct_retention": rate(0, 0),
            "retrieval_recall": {}, "pending_review": count, "abstentions": 0, "errors": 0, "unavailable": 0,
            "status_counts": {"answered": count}, "latency": latency, "p50_ms": 3.0, "p95_ms": 7.0})
    return {"questions": count, "applicable_questions": applicable, "configurations": rows}


def secondary_rows(count, names=CONFIGURATIONS):
    return {name: {"all_requested_qa": {"denominator": count, "correct": 0, "incorrect": 0,
                    "pending_uncredited": count, "accuracy_including_all_nonanswers_and_pending": 0.0}}
            for name in names}


class SealedFixture:
    def __init__(self, root, *, oracle_errors=0):
        self.root, self.files = root, {}
        self.scope = {key: rate(0, 1) for key in ("precision", "recall", "false_positive_rate", "brier", "ece")}
        self.combined = population(816, 384)
        self.cohorts = {"scope": population(720, 288), "challenge": population(96, 96)}
        self.oracle = population(384, 384, ("oracle_answer",))["configurations"][0]
        self.oracle["by_population"] = {name: population(count, count, ("oracle_answer",))["configurations"][0]
                                        for name, count in (("answer", 288), ("challenge", 96))}
        if oracle_errors:
            for item, errors in ((self.oracle, oracle_errors),
                                 (self.oracle["by_population"]["answer"], max(0, oracle_errors - 96)),
                                 (self.oracle["by_population"]["challenge"], min(96, oracle_errors))):
                count = item["questions"]
                item.update(errors=errors, pending_review=count - errors,
                            coverage=rate(count - errors, count), model_generated_answer_coverage=rate(count - errors, count),
                            status_counts={"answered": count - errors, "error": errors})
        self.config = self.write("config.json", {"fixture": True})
        models = {}
        for task in ("scope", "answer"):
            weights = self.write(task + "/model.bin", "synthetic weights")
            card = {"task": task, "purpose": "trained", "status": "completed", "completed_steps": 4,
                    "selection": {"global_step": 4}, "selected_model_state_sha256": "a" * 64,
                    "model_id": "synthetic-" + task, "model_revision": "fixture",
                    "files_sha256": {"model.bin": self.files[weights]}}
            manifest = self.write(task + "/manifest.json", card)
            models[task] = {"artifact_path": str(manifest.parent), "manifest_sha256": self.files[manifest],
                "completed_steps": 4, "selected_step": 4, "selected_model_state_sha256": "a" * 64,
                "model_id": card["model_id"], "model_revision": card["model_revision"]}
        counts = FinalReportCache.COUNTS
        self.protocol = self.write("protocol.json", {"purpose": "frozen_final_evaluation_protocol", "status": "frozen",
            "protocol_id": "synthetic-only", "configuration": {"sha256": self.files[self.config]},
            "review_policy": "automated_final", "human_reviewed_data": False,
            "evaluation": {"question_counts": counts, "configurations": list(CONFIGURATIONS), "max_final_runs": 1,
                           "matched_populations": ["scope", "challenge"], "oracle_populations": ["answer", "challenge"]},
            "source_sha256": {"scripts/run_final_evaluation.py": "e" * 64},
            "models": {name: {"manifest_sha256": record["manifest_sha256"]} for name, record in models.items()}})
        self.packet = self.write("packet/manifest.json", {"purpose": "frozen_final_evaluation_packet",
            "protocol_sha256": self.files[self.protocol], "counts": counts, "files": {},
            "test_frozen": True, "eligible_for_final_evaluation": True})
        self.lock = self.write("lock.json", {"purpose": "final_packet_lock_before_inference", "status": "frozen",
            "protocol_sha256": self.files[self.protocol], "counts": counts, "packet_manifest_sha256": self.files[self.packet],
            "inference_started": False, "review_counts": {"accepted": 1104, "flagged": 0},
            "review_policy": "automated_final", "human_reviewed_data": False})
        secondary = {"purpose": "locked_final_secondary_short_qa", "primary_results_replaced": False,
            "post_hoc_policy": False, "protocol_sha256": self.files[self.protocol],
            "configurations": secondary_rows(816), "by_population": {"scope": secondary_rows(720), "challenge": secondary_rows(96)},
            "oracle": secondary_rows(384, ("oracle_answer",)),
            "oracle_by_population": {"answer": secondary_rows(288, ("oracle_answer",)), "challenge": secondary_rows(96, ("oracle_answer",))}}
        self.predictions = self.write("run/predictions.jsonl", "synthetic; not real predictions")
        oracle_predictions = self.write("run/oracle_predictions.jsonl", "synthetic; not real oracle predictions")
        metrics = self.write("run/metrics.json", {**self.raw_metrics(self.combined),
            "by_population": {name: self.raw_metrics(item) for name, item in self.cohorts.items()},
            "oracle_answer_diagnostic": self.raw_metrics({"configurations": [self.oracle]}),
            "oracle_answer_by_population": {name: self.raw_metrics({"configurations": [item]}) for name, item in self.oracle["by_population"].items()}})
        secondary_file = self.write("run/short_qa_metrics.json", secondary)
        outputs = {path.name: self.files[path] for path in (self.predictions, oracle_predictions, metrics, secondary_file)}
        raw = self.write("run/manifest.json", {"purpose": "frozen_final_evaluation", "status": "completed",
            "review_policy": "automated_final", "human_reviewed_data": False, "protocol_sha256": self.files[self.protocol],
            "protocol_id": "synthetic-only", "configuration_sha256": self.files[self.config], "population_counts": counts,
            "questions": 816, "matched_prediction_records": 4080, "oracle_records": 384,
            "packet_manifest_sha256": self.files[self.packet], "final_test_tuning_performed": False,
            "fresh_final_runs": 1, "base_weights_unchanged": True,
            "base_parameter_sha256_before": "b" * 64, "base_parameter_sha256_after": "b" * 64, "outputs_sha256": outputs})
        ledger = self.write("ledger.json", {"status": "completed", "fresh_final_runs": 1,
                                          "result_manifest_sha256": self.files[raw]})
        self.summary = {"schema_version": 1, "artifact_type": "editguard_final_evaluation", "purpose": "frozen_final_evaluation",
            "status": "completed", "review_policy": "automated_final", "human_reviewed_data": False,
            "test_rows_read": True, "challenge_rows_read": True, "test_frozen": True, "final_test_tuning_performed": False,
            "dataset_id": "synthetic-only", "questions": 816, "oracle_questions": 384, "population_counts": counts,
            "matched_applicable_questions": 384, "configurations": self.combined["configurations"],
            "by_population": self.cohorts, "oracle": self.oracle, "model_artifacts": models,
            "scope": self.scope, "scope_full_metrics": self.scope, "secondary_short_qa": secondary,
            "limitations": ["Synthetic fixture only; never project measurements."],
            "provenance": {"config_path": str(self.config), "config_sha256": self.files[self.config],
                "protocol_path": str(self.protocol), "protocol_sha256": self.files[self.protocol], "protocol_id": "synthetic-only",
                "test_lock_path": str(self.lock), "test_lock_sha256": self.files[self.lock],
                "packet_manifest_path": str(self.packet), "packet_manifest_sha256": self.files[self.packet], "packet_files_sha256": {},
                "raw_manifest_path": str(raw), "raw_manifest_sha256": self.files[raw],
                "ledger_path": str(ledger), "ledger_sha256": self.files[ledger],
                **{name.removesuffix(".jsonl").removesuffix(".json") + "_sha256": digest for name, digest in outputs.items()},
                "base_weights_unchanged": True, "base_parameter_sha256": "b" * 64,
                "evidence_files_sha256": {str(path): digest for path, digest in self.files.items()}}}
        self.path = root / "summary.json"
        self.save()
        self.status = {"configuration": "valid", "configuration_sha256": self.files[self.config],
            "model_details": {task: {"status": "trained", "loaded": False, "error": None,
                                     "artifact_manifest_sha256": model["manifest_sha256"]} for task, model in models.items()},
            "calibrations": {"scope": "validated", "similarity": "validated"},
            "calibration_sha256": {"scope": "c" * 64, "similarity": "d" * 64}}

    def raw_metrics(self, collection):
        return {"configurations": {item["configuration"]: {**copy.deepcopy(item),
            "subsets": {"out_of_scope": {"accuracy": item["out_of_scope_accuracy"]}},
            "false_correction_application_rate": item["false_application_rate"],
            "scope": self.scope if item["configuration"] == "trained_scope_editor" else None}
            for item in collection["configurations"]}}

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value, str) else value, encoding="utf-8")
        self.files[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def save(self):
        self.path.write_text(json.dumps(self.summary), encoding="utf-8")

    def bind_extra(self, name, value):
        path = self.write(name, value)
        self.summary["provenance"]["evidence_files_sha256"][str(path)] = self.files[path]
        return path


class FinalDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.fixture = SealedFixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def cache(self):
        return FinalReportCache(self.fixture.path, root=self.root)

    def test_completed_hash_bound_report_is_available_without_model_loading(self):
        with patch("editguard.comparison.ComparisonRunner.from_config", side_effect=AssertionError("No model load")):
            report, descriptor = self.cache().snapshot(self.fixture.status)
        self.assertEqual(descriptor["status"], "available", descriptor)
        self.assertEqual(report["questions"], 816)
        self.assertEqual(report["oracle_questions"], 384)
        self.assertFalse(report["human_reviewed_data"])

    def test_pending_file_creation_becomes_available_without_restart(self):
        self.fixture.path.unlink()
        cache = self.cache()
        self.assertEqual(cache.snapshot(self.fixture.status)[1]["status"], "not_measured")
        self.fixture.save()
        self.assertEqual(cache.snapshot(self.fixture.status)[1]["status"], "available")

    def test_hash_change_after_cached_validation_invalidates_report(self):
        cache = self.cache()
        self.assertEqual(cache.snapshot(self.fixture.status)[1]["status"], "available")
        self.fixture.predictions.write_text("modified", encoding="utf-8")
        report, descriptor = cache.snapshot(self.fixture.status)
        self.assertIsNone(report)
        self.assertEqual(descriptor["status"], "invalid")
        self.assertIn("integrity", descriptor["error"])

    def test_model_or_configuration_change_is_incompatible(self):
        for mutation in ("configuration", "model", "restart"):
            status = copy.deepcopy(self.fixture.status)
            if mutation == "configuration":
                status["configuration_sha256"] = "f" * 64
            elif mutation == "model":
                status["model_details"]["answer"]["artifact_manifest_sha256"] = "f" * 64
            else:
                status["restart_required"] = True
            self.assertEqual(self.cache().snapshot(status)[1]["status"], "incompatible", mutation)

    def test_unsupported_final_claims_and_lost_denominators_are_rejected(self):
        original = copy.deepcopy(self.fixture.summary)
        for mutation in ("human", "incomplete", "oracle_mix", "config_missing", "qa_pending", "primary_tamper", "outside"):
            self.fixture.summary = copy.deepcopy(original)
            report = self.fixture.summary
            if mutation == "human": report["human_reviewed_data"] = True
            elif mutation == "incomplete": report["status"] = "running"
            elif mutation == "oracle_mix": report["questions"] += report["oracle_questions"]
            elif mutation == "config_missing": report["configurations"].pop()
            elif mutation == "qa_pending": report["secondary_short_qa"]["configurations"]["unchanged_base"]["all_requested_qa"]["pending_uncredited"] -= 1
            elif mutation == "primary_tamper": report["configurations"][0]["accuracy"] = rate(1, 816)
            else: report["provenance"]["evidence_files_sha256"][str(self.root.parent / "outside.bin")] = "e" * 64
            self.fixture.save()
            self.assertEqual(self.cache().snapshot(self.fixture.status)[1]["status"], "invalid", mutation)

    def test_endpoint_is_read_only_and_pending_without_neural_dependencies(self):
        missing = self.root / "not-yet-exported.json"
        with TestClient(create_app(self.root / "db.sqlite", "test", final_evaluation_report=missing)) as client:
            before = client.get("/api/edits").json()
            with patch("editguard.evaluation.run_slice_regression", side_effect=AssertionError("No evaluation run")), patch("editguard.comparison.ComparisonRunner.from_config", side_effect=AssertionError("No model load")):
                result = client.get("/api/evaluation/final")
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["status"], "not_measured")
                self.assertIsNone(result.json()["report"])
                self.assertEqual(client.get("/api/health").json()["final_evaluation"], "pending")
                self.assertEqual(client.get("/api/evaluation/final/export/summary").status_code, 409)
                self.assertEqual(client.get("/api/evaluation/final/export/unknown").status_code, 404)
            self.assertEqual(client.get("/api/edits").json(), before)

    def test_duplicate_json_keys_and_nonfinite_values_are_invalid(self):
        for text in ('{"status": "completed", "status": "running"}', '{"value": NaN}'):
            self.fixture.path.write_text(text, encoding="utf-8")
            self.assertEqual(self.cache().snapshot(self.fixture.status)[1]["status"], "invalid")

    def test_compatible_endpoint_health_and_summary_download_share_sealed_snapshot(self):
        cache = self.cache()
        with patch("editguard.api.FinalReportCache", return_value=cache), patch.object(ArtifactStatusCache, "snapshot", return_value=self.fixture.status), \
                patch("editguard.comparison.ComparisonRunner.from_config", side_effect=AssertionError("No model load")), \
                patch("editguard.evaluation.run_slice_regression", side_effect=AssertionError("No evaluation run")):
            with TestClient(create_app(self.root / "available.sqlite", "test", final_evaluation_report=self.fixture.path)) as client:
                before = client.get("/api/edits").json()
                self.assertEqual(client.get("/api/health").json()["final_evaluation"], "available")
                self.assertEqual(client.get("/api/evaluation/final").json()["report"]["questions"], 816)
                exported = client.get("/api/evaluation/final/export/summary")
                self.assertEqual(exported.status_code, 200)
                self.assertEqual(exported.content, self.fixture.path.read_bytes())
                self.assertIn("attachment", exported.headers["content-disposition"])
                self.assertEqual(client.get("/api/evaluation/final/export/csv").status_code, 404)
                self.assertEqual(client.get("/api/edits").json(), before)

    def test_cuda_disclosure_preserves_errors_and_original_fingerprint_unavailability(self):
        self.fixture = SealedFixture(self.root, oracle_errors=319)
        payload = {"schema_version": 1, "purpose": "final_runtime_failure_disclosure",
            "protocol_sha256": self.fixture.files[self.fixture.protocol], "observations_committed": 4464,
            "status_by_kind": {"matched": {"answered": 4080}, "oracle": {"answered": 65, "error": 319}},
            "original_gpu_instance_post_inference_fingerprint": "unavailable_cuda_unknown_error",
            "persistent_model_or_data_changes": False,
            "closure_recovery_policy": "Fresh closure checks do not reconstruct the original GPU memory state.",
            "oracle_measurement_policy": "Original errors remain uncredited; supplemental recovery stays separate."}
        path = self.fixture.bind_extra("gpu-disclosure.json", payload)
        disclosure = {**payload, "disclosure_path": str(path), "disclosure_sha256": self.fixture.files[path],
                      "original_results_replaced": False,
                      "limitation": "The original GPU state is unavailable; fresh closure checks do not recover it."}
        self.fixture.summary["execution_recovery"] = {"cuda_runtime_failure": disclosure}
        self.fixture.save()
        report, descriptor = self.cache().snapshot(self.fixture.status)
        self.assertEqual(descriptor["status"], "available", descriptor)
        self.assertEqual(report["oracle"]["errors"], 319)
        self.assertEqual(report["oracle"]["accuracy"]["denominator"], 384)
        self.fixture.summary["execution_recovery"]["cuda_runtime_failure"]["status_by_kind"]["oracle"]["error"] = 0
        self.fixture.save()
        self.assertEqual(self.cache().snapshot(self.fixture.status)[1]["status"], "invalid")

    def test_io_disclosure_counts_only_bound_recorded_retry_events(self):
        wrapper = self.fixture.bind_extra("wrapper.py", "# Synthetic wrapper; never executed")
        spec = {"purpose": "frozen_final_io_recovery_declaration", "protocol_sha256": self.fixture.files[self.fixture.protocol],
            "wrapper_sha256": self.fixture.files[wrapper], "frozen_evaluator_sha256": "e" * 64,
            "fresh_run_allowed": False, "inference_scoring_or_data_changes": False, "observations_rewritten": False,
            "max_write_attempts": 8, "retry_delays_seconds": [.05, .1, .2, .4, .4, .4, .4],
            "mutable_write_targets": ["run/progress.json", "ledger.json"]}
        declaration = self.fixture.bind_extra("io-declaration.json", spec)
        entry = {"declaration_sha256": self.fixture.files[declaration], "frozen_evaluator_sha256": "e" * 64,
            "inference_scoring_or_data_changes": False,
            "write_retries": [{"target": "progress.json", "failed_attempt": attempt, "delay_seconds": delay, "error_type": "PermissionError"}
                              for attempt, delay in ((1, .05), (2, .1))]}
        events = self.fixture.bind_extra("io-events.jsonl", json.dumps(entry) + "\n")
        io = {**spec, "declaration_path": str(declaration), "declaration_sha256": self.fixture.files[declaration],
            "wrapper_path": str(wrapper), "events_path": str(events), "events_sha256": self.fixture.files[events],
            "event_records": 1, "recorded_write_retries": 2, "retry_counts_by_target": {"progress.json": 2, "ledger.json": 0},
            "recorded_retry_delay_seconds": .15, "events_frozen_at_summary_export": True,
            "cause_status": "not_established", "cause_disclosure": "Underlying cause is unknown."}
        self.fixture.summary["execution_recovery"] = {"io": io}
        self.fixture.save()
        self.assertEqual(self.cache().snapshot(self.fixture.status)[1]["status"], "available")
        self.fixture.summary["execution_recovery"]["io"]["recorded_write_retries"] = 3
        self.fixture.save()
        self.assertEqual(self.cache().snapshot(self.fixture.status)[1]["status"], "invalid")

    def recovery_fixture(self):
        directory = self.root / "recovery"
        directory.mkdir()
        (directory / "source").mkdir()
        def write(path, value):
            path.write_text(json.dumps(value), encoding="utf-8")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        provenance = self.fixture.summary["provenance"]
        originals = {"manifest.json": provenance["raw_manifest_sha256"], "oracle_predictions.jsonl": provenance["oracle_predictions_sha256"],
                     "metrics.json": provenance["metrics_sha256"], "short_qa_metrics.json": provenance["short_qa_metrics_sha256"]}
        for name in originals:
            (directory / "source" / name).write_bytes((self.root / "run" / name).read_bytes())
        source = self.root / "recovery-source.py"
        source_sha = write(source, {"fixture": "Never executed"})
        ledger = self.root / "recovery-ledger.json"
        declaration = {"purpose": "frozen_oracle_runtime_recovery_declaration", "status": "frozen",
            "protocol_sha256": provenance["protocol_sha256"], "configuration_sha256": provenance["config_sha256"],
            "original_files_sha256": originals, "oracle_questions": 384, "selected_questions": 319,
            "maximum_new_generation_calls": 319, "passes": 1, "primary_results_replaced": False,
            "training_or_parameter_selection": False, "successful_original_records_reexecuted": False,
            "output_directory": str(directory), "supplement_source_path": str(source), "supplement_source_sha256": source_sha,
            "ledger_path": str(ledger), "expected_base_parameter_fingerprint_sha256": "b" * 64,
            "selected": [{"id": f"synthetic-{index}", "population": "answer" if index < 223 else "challenge"} for index in range(319)]}
        declaration_sha = write(directory / "source/declaration.json", declaration)
        def metrics_population(count, challenge_count):
            current = self.fixture.raw_metrics(population(count, count, ("oracle_answer",)))
            row = current["configurations"]["oracle_answer"]
            row.update(count=count, counts={"total": count, "missing_records": 0})
            regular = count - challenge_count
            row["subsets"].update({"direct": {"accuracy": rate(0, regular // 3)},
                                   "paraphrase": {"accuracy": rate(0, regular // 3)},
                                   "comparison": {"accuracy": rate(0, regular - regular // 3 * 2)},
                                   "challenge": {"accuracy": rate(0, challenge_count)}})
            return {**current, "question_count": count, "record_count": count}
        metrics = {**metrics_population(384, 96), "purpose": "supplemental_oracle_runtime_recovery", "primary_results_replaced": False,
            "original_denominator": 384, "selected_runtime_error_count": 319, "original_successful_or_other_unselected_records_preserved": 65,
            "supplemental_records": 319, "interrupted_reservation_count": 0, "new_runtime_error_count": 0, "not_attempted_selected_count": 0,
            "by_population": {"answer": metrics_population(288, 0), "challenge": metrics_population(96, 96)}}
        secondary = {"purpose": "supplemental_oracle_runtime_recovery_secondary", "primary_results_replaced": False,
            "configurations": secondary_rows(384, ("oracle_answer",)),
            "by_population": {"answer": secondary_rows(288, ("oracle_answer",)), "challenge": secondary_rows(96, ("oracle_answer",))}}
        outputs = {name: write(directory / name, value) for name, value in
                   (("recovery_predictions.jsonl", {"synthetic": True}), ("combined_oracle_predictions.jsonl", {"synthetic": True}),
                    ("metrics.json", metrics), ("short_qa_metrics.json", secondary))}
        manifest = {"schema_version": 1, "purpose": "supplemental_oracle_runtime_recovery", "status": "completed",
            "primary_results_replaced": False, "review_policy": "automated_final", "human_reviewed_data": False,
            "original_records_unchanged": True, "selection_used_correctness": False,
            "protocol_sha256": provenance["protocol_sha256"], "configuration_sha256": provenance["config_sha256"],
            "oracle_questions": 384, "selected_runtime_errors": 319, "maximum_new_generation_calls": 319,
            "original_files_sha256": originals, "declaration_sha256": declaration_sha, "supplement_source_sha256": source_sha,
            "outputs_sha256": outputs, "durable_attempt_reservations": 319, "interrupted_reservation_count": 0,
            "new_runtime_error_count": 0, "not_attempted_selected_count": 0,
            "sessions": [{"starting_index": 0, "completed_at": "2026-10-15T00:00:00+00:00",
                          "base_parameter_sha256_before": "b" * 64, "base_parameter_sha256_after": "b" * 64}],
            "limitations": ["Separate synthetic supplement, mixed timing, original state remains unavailable."]}
        path = directory / "manifest.json"
        manifest_sha = write(path, manifest)
        write(ledger, {"status": "completed", "result_manifest_sha256": manifest_sha,
            "binding": {"declaration_sha256": declaration_sha, "protocol_sha256": provenance["protocol_sha256"],
                        "original_oracle_sha256": provenance["oracle_predictions_sha256"]}})
        return path, manifest

    def test_recovery_is_separate_full_cohort_bound_to_original_evidence(self):
        self.fixture = SealedFixture(self.root, oracle_errors=319)
        path, _ = self.recovery_fixture()
        cache = OracleRecoveryReportCache(path, root=self.root)
        original = copy.deepcopy(self.fixture.summary)
        report, descriptor = cache.snapshot(original)
        self.assertEqual(descriptor["status"], "available", descriptor)
        self.assertEqual(report["oracle"]["accuracy"]["denominator"], 384)
        self.assertEqual(report["oracle"]["errors"], 0)
        self.assertEqual(original["oracle"]["errors"], 319)
        self.assertEqual(original, self.fixture.summary)
        self.assertFalse(report["primary_results_replaced"])

    def test_recovery_original_mismatch_or_modified_derived_file_is_rejected(self):
        self.fixture = SealedFixture(self.root, oracle_errors=319)
        path, _ = self.recovery_fixture()
        cache = OracleRecoveryReportCache(path, root=self.root)
        self.assertEqual(cache.snapshot(None)[1]["status"], "original_unavailable")
        changed = copy.deepcopy(self.fixture.summary)
        changed["provenance"]["config_sha256"] = "f" * 64
        self.assertEqual(cache.snapshot(changed)[1]["status"], "invalid")
        self.assertEqual(cache.snapshot(self.fixture.summary)[1]["status"], "available")
        (path.parent / "metrics.json").write_text("{}", encoding="utf-8")
        self.assertEqual(cache.snapshot(self.fixture.summary)[1]["status"], "invalid")

    def test_interrupted_preloading_is_visible_without_publishing_recovery_scores(self):
        self.fixture = SealedFixture(self.root, oracle_errors=319)
        path, manifest = self.recovery_fixture()
        path.unlink()
        metrics_path = path.parent / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.update(supplemental_records=0, not_attempted_selected_count=319)
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        manifest.update(status="interrupted", durable_attempt_reservations=0, not_attempted_selected_count=319,
                        sessions=[{"base_parameter_sha256_before": None, "base_parameter_sha256_after": None}])
        manifest["outputs_sha256"]["metrics.json"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        (path.parent / "interrupted_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        ledger_path = self.root / "recovery-ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger.update(status="interrupted", failure={"type": "RuntimeError", "reason": "CUDA unavailable before generation"})
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        report, descriptor = OracleRecoveryReportCache(path, root=self.root).snapshot(self.fixture.summary)
        self.assertIsNone(report)
        self.assertEqual(descriptor["status"], "interrupted", descriptor)
        self.assertEqual(descriptor["durable_attempt_reservations"], 0)
        self.assertIn("CUDA unavailable", descriptor["failure_reason"])

    def test_completed_recovery_retains_failed_preload_then_healthy_resume_sessions(self):
        self.fixture = SealedFixture(self.root, oracle_errors=319)
        path, manifest = self.recovery_fixture()
        preload = {"starting_index": 0, "base_parameter_sha256_before": None, "base_parameter_sha256_after": None,
                   "fingerprint_after_status": "unavailable_for_this_interrupted_session",
                   "failure": {"type": "RuntimeError", "reason": "CUDA unavailable before generation"}}
        manifest["sessions"].insert(0, preload)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        ledger_path = self.root / "recovery-ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["result_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        report, descriptor = OracleRecoveryReportCache(path, root=self.root).snapshot(self.fixture.summary)
        self.assertEqual(descriptor["status"], "available", descriptor)
        self.assertEqual(report["sessions"][0], preload)
        self.assertIsNone(report["sessions"][0]["base_parameter_sha256_after"])
        self.assertEqual(report["sessions"][1]["base_parameter_sha256_after"], "b" * 64)
        self.assertEqual(self.fixture.summary["oracle"]["errors"], 319)

    def test_sessions_with_reserved_work_need_before_check_and_explicit_interruption(self):
        failed = {"starting_index": 0, "base_parameter_sha256_before": None, "base_parameter_sha256_after": None,
                  "fingerprint_after_status": "unavailable_for_this_interrupted_session",
                  "failure": {"reason": "Interrupted"}}
        completed = {"starting_index": 0, "base_parameter_sha256_before": "b" * 64, "base_parameter_sha256_after": "b" * 64,
                     "completed_at": "2026-10-15T00:00:00+00:00"}
        OracleRecoveryReportCache._sessions([failed, completed], "b" * 64, 319)
        reserved_then_resumed = copy.deepcopy(completed)
        reserved_then_resumed["starting_index"] = 1
        with self.assertRaisesRegex(ValueError, "before inference"):
            OracleRecoveryReportCache._sessions([failed, reserved_then_resumed], "b" * 64, 319)
        interrupted = {**failed, "base_parameter_sha256_before": "b" * 64}
        OracleRecoveryReportCache._sessions([interrupted, reserved_then_resumed], "b" * 64, 319)
        for key in ("failure", "fingerprint_after_status"):
            malformed = copy.deepcopy(interrupted)
            malformed.pop(key)
            with self.assertRaisesRegex(ValueError, "unavailable post-run fingerprint"):
                OracleRecoveryReportCache._sessions([malformed, reserved_then_resumed], "b" * 64, 319)


if __name__ == "__main__":
    unittest.main()
