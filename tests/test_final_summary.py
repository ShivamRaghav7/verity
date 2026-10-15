"""Synthetic final exports only; no actual test content, models or inference."""

import copy
import csv
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import build_final_summary as builder
from scripts.final_protocol import canonical_hash, SOURCE_NAMES
from tests import test_final_evaluation as runner_fixtures

FakeRunner = runner_fixtures.FakeRunner


class FinalSummaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = runner_fixtures.FinalEvaluationTests("test_finite_execution_plan_has_separate_roles_and_no_missing_configs")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.original_counts = dict(builder.COUNTS)
        # A deliberately different oracle cohort size catches the assumption
        # that oracle rows are the applicable subset of matched question IDs.
        extra = copy.deepcopy(self.fixture.populations["answer"][0])
        extra["id"] = "answer-2"
        self.fixture.populations["answer"].append(extra)
        self.counts = {name: len(rows) for name, rows in self.fixture.populations.items()}
        self.enterContext(patch("scripts.build_final_summary.COUNTS", self.counts))
        protocol = self.fixture.protocol
        protocol.update(schema_version=1, purpose="frozen_final_evaluation_protocol", status="frozen")
        protocol["evaluation"]["question_counts"] = self.counts
        source_hashes = {}
        for name in SOURCE_NAMES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((builder.ROOT / name).read_bytes())
            source_hashes[name] = builder.file_hash(path)
        protocol["source_sha256"] = source_hashes
        protocol["scoring"].update(primary_sha256=source_hashes["editguard/scoring.py"],
                                   short_qa_sha256=source_hashes["editguard/short_qa.py"])
        models = {}
        for task in ("scope", "answer"):
            artifact = self.root / "models" / task
            (artifact / "model").mkdir(parents=True)
            weights = artifact / "model/model.safetensors"
            weights.write_bytes(b"synthetic weights never deserialized or loaded")
            files = {"model/model.safetensors": builder.file_hash(weights)}
            manifest = {"task": task, "purpose": "trained", "status": "completed", "input_format_version": "editguard-input-v2",
                "human_reviewed_data": False, "review_policy": "automated_development", "files_sha256": files,
                "selected_model_state_sha256": "b" * 64, "completed_steps": 12, "selection": {"global_step": 6},
                "model_id": "synthetic-model", "model_revision": "c" * 40}
            self.write(artifact / "manifest.json", manifest)
            models[task] = {"artifact": "models/" + task, "manifest_sha256": builder.file_hash(artifact / "manifest.json"),
                "files_sha256": files, "selected_model_state_sha256": "b" * 64}
        protocol["models"] = models
        calibrations = {}
        for task in ("scope", "similarity"):
            path = self.root / "calibrations" / (task + ".json")
            card = {"status": "frozen", "split": "validation", "dataset_id": "synthetic-development", "threshold": .5}
            if task == "scope":
                card.update(temperature=1., ambiguity_margin=.025, scope_manifest_sha256=models["scope"]["manifest_sha256"])
            self.write(path, card)
            calibrations[task] = {"path": "calibrations/" + task + ".json", "sha256": builder.file_hash(path)}
        protocol["calibrations"] = calibrations
        config = {task + "_artifact": "models/" + task for task in ("scope", "answer")}
        config.update(calibration_artifact="calibrations/scope.json", similarity_calibration_artifact="calibrations/similarity.json")
        self.write(self.fixture.config, config)
        protocol["configuration"]["sha256"] = builder.file_hash(self.fixture.config)
        protocol.pop("protocol_id")
        protocol["protocol_id"] = "editguard-final-" + canonical_hash(protocol)[:16]
        self.write(self.fixture.protocol_path, protocol)
        self.fixture.protocol_sha = builder.file_hash(self.fixture.protocol_path)
        packet = self.fixture.packet
        (packet / "synthetic-questions.bin").write_bytes(b"opaque fixture bytes; never parsed")
        self.fixture.packet_manifest.update(purpose="frozen_final_evaluation_packet", protocol_sha256=self.fixture.protocol_sha,
            counts=self.counts, review_policy="automated_final", human_reviewed_data=False, test_frozen=True,
            eligible_for_final_evaluation=True, all_rows_preserved=True,
            review_counts={"accepted": sum(self.counts.values()), "flagged": 0},
            files={"synthetic-questions.bin": builder.file_hash(packet / "synthetic-questions.bin")})
        self.write(packet / "manifest.json", self.fixture.packet_manifest)
        self.fixture.packet_hash = builder.file_hash(packet / "manifest.json")
        self.lock_path = self.root / "test-lock.json"
        self.lock = {"purpose": "final_packet_lock_before_inference", "status": "frozen", "review_policy": "automated_final",
            "human_reviewed_data": False, "inference_started": False, "counts": self.counts,
            "review_counts": self.fixture.packet_manifest["review_counts"], "protocol_path": "protocol.json",
            "protocol_sha256": self.fixture.protocol_sha, "packet_path": "synthetic-packet",
            "packet_manifest_sha256": self.fixture.packet_hash, "packet_files_sha256": self.fixture.packet_manifest["files"]}
        self.write(self.lock_path, self.lock)
        self.lock_sha = builder.file_hash(self.lock_path)
        self.run = self.root / "results"
        self.output = self.root / "exports/final-summary.json"

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8", newline="\n")

    def complete(self, runner=None):
        return self.fixture.run_fixture(runner or FakeRunner())

    def build(self, **kwargs):
        return builder.build(self.run, self.lock_path, self.lock_sha, self.output, root=self.root, **kwargs)

    def reseal_changed_export(self, name):
        """Simulate an internally rehashed bad export; recomputation still wins."""
        path = self.run / "manifest.json"
        manifest = builder.read_json(path)
        manifest["outputs_sha256"][name] = builder.file_hash(self.run / name)
        self.write(path, manifest)
        ledger_path = self.root / "execution-ledger.json"
        ledger = builder.read_json(ledger_path)
        ledger["result_manifest_sha256"] = builder.file_hash(path)
        self.write(ledger_path, ledger)

    def io_metadata(self, entries=None):
        """Synthetic IO log only; never execute the recovery wrapper."""
        wrapper = self.root / "scripts/resume_final_io.py"
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_bytes((builder.ROOT / "scripts/resume_final_io.py").read_bytes())
        declaration_path = self.root / "io-declaration.json"
        declaration = {"schema_version": 1, "purpose": "frozen_final_io_recovery_declaration",
            "reason": "Synthetic Windows access-denied collision; lock owner not established.",
            "protocol_path": "protocol.json", "protocol_sha256": self.fixture.protocol_sha,
            "packet_path": "synthetic-packet", "wrapper_path": "scripts/resume_final_io.py",
            "wrapper_sha256": builder.file_hash(wrapper),
            "frozen_evaluator_sha256": builder.file_hash(self.root / "scripts/run_final_evaluation.py"),
            "mutable_write_targets": ["results/progress.json", "execution-ledger.json"],
            "max_write_attempts": 8, "retry_delays_seconds": [.05, .1, .2, .4, .4, .4, .4],
            "maximum_retry_delay_seconds": 1.95, "fresh_run_allowed": False,
            "inference_scoring_or_data_changes": False, "observations_rewritten": False,
            "performance_scope": "IO retry sleeps are outside model-route request latency."}
        self.write(declaration_path, declaration)
        declaration_sha = builder.file_hash(declaration_path)
        base_event = {"declaration_sha256": declaration_sha,
            "frozen_evaluator_sha256": declaration["frozen_evaluator_sha256"],
            "inference_scoring_or_data_changes": False, "write_retries": []}
        events = self.root / "runtime/io-events.jsonl"
        events.parent.mkdir(parents=True, exist_ok=True)
        selected = [base_event] if entries is None else [{**base_event, "write_retries": rows} for rows in entries]
        events.write_text("".join(json.dumps(item) + "\n" for item in selected), encoding="utf-8", newline="\n")
        return {"io_recovery_declaration": declaration_path,
            "io_recovery_declaration_sha256": declaration_sha, "io_recovery_events": events}

    @staticmethod
    def retry(target="progress.json", attempt=1, delay=.05, error="PermissionError"):
        return {"target": target, "failed_attempt": attempt, "delay_seconds": delay, "error_type": error}

    def test_completed_evidence_preserves_cohorts_and_primary_secondary(self):
        self.assertEqual(self.original_counts, {"scope": 720, "answer": 288, "challenge": 96})
        self.complete()
        result = self.build(csv_output=self.root / "exports/final-summary.csv")
        self.assertEqual(result["artifact_type"], "editguard_final_evaluation")
        self.assertEqual(result["purpose"], "frozen_final_evaluation")
        self.assertFalse(result["human_reviewed_data"])
        self.assertEqual(result["questions"], 3)
        self.assertEqual(result["matched_applicable_questions"], 2)
        self.assertEqual(result["oracle_questions"], 4)
        self.assertEqual(result["by_population"]["scope"]["questions"], 2)
        self.assertEqual(result["by_population"]["challenge"]["questions"], 1)
        self.assertEqual(result["oracle"]["by_population"]["answer"]["questions"], 3)
        self.assertEqual(result["oracle"]["by_population"]["challenge"]["questions"], 1)
        self.assertEqual(result["oracle"]["direct_accuracy"]["denominator"], 3)
        self.assertEqual(result["oracle"]["challenge_accuracy"]["denominator"], 1)
        self.assertEqual(result["provenance"]["test_lock_sha256"], self.lock_sha)
        self.assertEqual(result["scope_policy"]["selected_on_split"], "validation")
        self.assertEqual(result["scope_policy"]["calibration_dataset_id"], "synthetic-development")
        for item in result["configurations"]:
            self.assertEqual(item["accuracy"]["denominator"], 3)
            self.assertEqual(item["out_of_scope_accuracy"]["denominator"], 1)
            self.assertEqual(item["latency"]["sample_count"], 3)
        self.assertFalse(result["secondary_short_qa"]["post_hoc_policy"])
        self.assertFalse(result["secondary_short_qa"]["primary_results_replaced"])
        self.assertNotIn("execution_recovery", result)
        with (self.root / "exports/final-summary.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 18)
        self.assertEqual(next(row for row in rows if row["population"] == "oracle_all")["denominator"], "4")
        self.assertEqual(builder.file_hash(self.root / "exports/final-summary.csv"), result["csv_export"]["sha256"])

    def test_nonanswers_remain_in_primary_and_secondary_full_denominators(self):
        self.complete(FakeRunner(status="conflict", answer_kind="deterministic_clarification",
                                 nonanswer_text="Nimbus Basic v2 supports up to 3 devices."))
        result = self.build()
        for item in result["configurations"]:
            self.assertEqual(item["accuracy"]["numerator"], 0)
            self.assertEqual(item["accuracy"]["denominator"], 3)
            self.assertEqual(item["abstention_rate"], {"numerator": 3, "denominator": 3, "value": 1.0})
            secondary = result["secondary_short_qa"]["configurations"][item["configuration"]]["all_requested_qa"]
            self.assertEqual(secondary["correct"], 0)
            self.assertEqual(secondary["denominator"], 3)

    def test_pending_outputs_keep_full_denominator_and_zero_not_factual_incorrectness_claim(self):
        runner = FakeRunner()
        runner.answer_model.generate = lambda text: "I would suggest checking the handbook."
        self.complete(runner)
        result = self.build()
        self.assertEqual(result["oracle"]["pending_review"], 4)
        self.assertEqual(result["oracle"]["pending_rate"]["denominator"], 4)
        self.assertEqual(result["oracle"]["accuracy"]["numerator"], 0)
        self.assertTrue(any("zero factual correctness" in text for text in result["limitations"]))

    def test_unfinished_run_cannot_publish(self):
        with self.assertRaises(KeyboardInterrupt):
            self.complete(FakeRunner(fail_at=1))
        with self.assertRaises(OSError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_changed_test_lock_or_source_rejects_before_publish(self):
        self.complete()
        original = self.lock_path.read_bytes()
        self.lock_path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.build()
        self.lock_path.write_bytes(original)
        (self.root / "editguard/scoring.py").write_bytes(b"changed frozen scorer")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_config_and_model_weight_drift_rejected(self):
        self.complete()
        original = self.fixture.config.read_bytes()
        self.fixture.config.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.build()
        self.fixture.config.write_bytes(original)
        (self.root / "models/answer/model/model.safetensors").write_bytes(b"other weights")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_runtime_configuration_mismatch_rejected(self):
        self.complete()
        other = self.root / "other-config.json"
        other.write_bytes(self.fixture.config.read_bytes())
        with self.assertRaisesRegex(ValueError, "Requested configuration"):
            self.build(config_path=other)

    def test_calibration_change_is_not_silently_applied_to_final_evidence(self):
        self.complete()
        (self.root / "calibrations/scope.json").write_text('{"threshold":0.1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_packet_and_output_tampering_rejected(self):
        self.complete()
        target = self.fixture.packet / "synthetic-questions.bin"
        original = target.read_bytes()
        target.write_bytes(b"changed frozen question file")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.build()
        target.write_bytes(original)
        (self.run / "predictions.jsonl").write_bytes(b"changed outputs")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.build()

    def test_rehashed_inconsistent_primary_metrics_rejected(self):
        self.complete()
        path = self.run / "metrics.json"
        value = builder.read_json(path)
        value["configurations"]["updated_rag"]["accuracy"]["numerator"] += 1
        self.write(path, value)
        self.reseal_changed_export("metrics.json")
        with self.assertRaisesRegex(ValueError, "recomputation"):
            self.build()

    def test_rehashed_secondary_denominator_drop_rejected(self):
        self.complete()
        path = self.run / "short_qa_metrics.json"
        value = builder.read_json(path)
        value["configurations"]["updated_rag"]["all_requested_qa"]["denominator"] -= 1
        self.write(path, value)
        self.reseal_changed_export("short_qa_metrics.json")
        with self.assertRaisesRegex(ValueError, "recomputation"):
            self.build()

    def test_rehashed_observation_cannot_be_credited_by_nonfrozen_scorer(self):
        self.complete()
        path = self.run / "predictions.jsonl"
        rows = builder.read_records(path)
        rows[0]["score"]["scorer_version"] = "unfrozen version"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.reseal_changed_export("predictions.jsonl")
        with self.assertRaisesRegex(ValueError, "frozen scorer"):
            self.build()

    def test_unsealed_ledger_and_repeat_export_rejected(self):
        self.complete()
        ledger = self.root / "execution-ledger.json"
        original = ledger.read_bytes()
        value = builder.read_json(ledger)
        value["status"] = "running"
        self.write(ledger, value)
        with self.assertRaisesRegex(ValueError, "ledger"):
            self.build()
        ledger.write_bytes(original)
        self.build()
        with self.assertRaisesRegex(ValueError, "fresh export"):
            self.build()

    def test_export_cannot_add_files_to_immutable_packet_or_raw_run(self):
        self.complete()
        for output in (self.fixture.packet / "new-summary.json", self.run / "new-summary.json"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "separate"):
                builder.build(self.run, self.lock_path, self.lock_sha, output, root=self.root)
            self.assertFalse(output.exists())

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self):
        path = self.root / "bad-json.json"
        for value in ('{"key":1,"key":2}', '{"number":NaN}'):
            path.write_text(value, encoding="utf-8")
            with self.subTest(value=value), self.assertRaises(ValueError):
                builder.read_json(path)

    def test_optional_io_provenance_preserves_scores_and_binds_final_event_bytes(self):
        self.complete()
        options = self.io_metadata([[self.retry(), self.retry(attempt=2, delay=.1), self.retry("execution-ledger.json")], []])
        original_primary = builder.read_json(self.run / "metrics.json")
        original_secondary = builder.read_json(self.run / "short_qa_metrics.json")
        result = self.build(**options)
        recovery = result["execution_recovery"]["io"]
        self.assertEqual(recovery["declaration_sha256"], options["io_recovery_declaration_sha256"])
        self.assertEqual(recovery["events_sha256"], builder.file_hash(options["io_recovery_events"]))
        self.assertEqual(recovery["event_records"], 2)
        self.assertEqual(recovery["recorded_write_retries"], 3)
        self.assertEqual(recovery["retry_counts_by_target"], {"progress.json": 2, "execution-ledger.json": 1})
        self.assertEqual(recovery["recorded_retry_delay_seconds"], .2)
        self.assertEqual(recovery["cause_status"], "not_established")
        self.assertIn("OneDrive involvement was not verified", recovery["cause_disclosure"])
        self.assertFalse(recovery["inference_scoring_or_data_changes"])
        self.assertTrue(recovery["events_frozen_at_summary_export"])
        self.assertEqual(builder.read_json(self.run / "metrics.json"), original_primary)
        self.assertEqual(builder.read_json(self.run / "short_qa_metrics.json"), original_secondary)
        self.assertEqual(result["questions"], 3)
        self.assertEqual(result["oracle_questions"], 4)
        for field in ("declaration", "wrapper", "events"):
            path, digest = recovery[field + "_path"], recovery[field + "_sha256"]
            self.assertEqual(result["provenance"]["evidence_files_sha256"][path], digest)

    def test_io_options_are_all_required_when_one_is_supplied(self):
        self.complete()
        options = self.io_metadata()
        for missing in options:
            selected = {key: value for key, value in options.items() if key != missing}
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "supplied together"):
                self.build(**selected)
            self.assertFalse(self.output.exists())

    def test_io_declaration_and_wrapper_tampering_rejected(self):
        self.complete()
        options = self.io_metadata()
        declaration = options["io_recovery_declaration"]
        original = declaration.read_bytes()
        declaration.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.build(**options)
        declaration.write_bytes(original)
        (self.root / "scripts/resume_final_io.py").write_bytes(b"changed wrapper")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.build(**options)
        self.assertFalse(self.output.exists())

    def test_io_declaration_cannot_authorize_new_run_other_targets_or_changed_scoring(self):
        self.complete()
        options = self.io_metadata()
        declaration_path = options["io_recovery_declaration"]
        original = builder.read_json(declaration_path)
        changes = [{"fresh_run_allowed": True}, {"inference_scoring_or_data_changes": True},
            {"observations_rewritten": True}, {"protocol_sha256": "f" * 64},
            {"frozen_evaluator_sha256": "f" * 64}, {"mutable_write_targets": ["results/predictions.jsonl", "execution-ledger.json"]},
            {"max_write_attempts": 9}, {"maximum_retry_delay_seconds": 2.0}]
        for change in changes:
            self.write(declaration_path, {**original, **change})
            selected = {**options, "io_recovery_declaration_sha256": builder.file_hash(declaration_path)}
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "write-only policy"):
                self.build(**selected)
            self.assertFalse(self.output.exists())

    def test_io_event_rejects_foreign_binding_unauthorized_target_and_invalid_retry_schedule(self):
        self.complete()
        options = self.io_metadata()
        events = options["io_recovery_events"]
        original = builder.read_records(events)[0]
        changes = [{"declaration_sha256": "f" * 64}, {"frozen_evaluator_sha256": "f" * 64},
            {"inference_scoring_or_data_changes": True}, {"write_retries": [self.retry("predictions.jsonl")]},
            {"write_retries": [self.retry(attempt=8, delay=.4)]}, {"write_retries": [self.retry(attempt=True)]},
            {"write_retries": [self.retry(attempt=2, delay=.1)]}, {"write_retries": [self.retry(error="OSError")]},
            {"write_retries": [self.retry(delay=.2)]},
            {"write_retries": [self.retry(), self.retry("execution-ledger.json", attempt=2, delay=.1)]}]
        for change in changes:
            events.write_text(json.dumps({**original, **change}) + "\n", encoding="utf-8")
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "IO event|IO retry"):
                self.build(**options)
            self.assertFalse(self.output.exists())

    def test_empty_io_log_cannot_claim_recorded_wrapper_completion(self):
        self.complete()
        options = self.io_metadata()
        options["io_recovery_events"].write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "wrapper-session record"):
            self.build(**options)
        self.assertFalse(self.output.exists())

    def test_io_events_append_during_export_prevents_publication(self):
        self.complete()
        options = self.io_metadata()
        events = options["io_recovery_events"]
        actual_hash, calls = builder.file_hash, []
        def append_on_recheck(path):
            if Path(path) == events:
                calls.append(path)
                if len(calls) == 2:
                    with events.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(builder.read_records(events)[0]) + "\n")
            return actual_hash(path)
        with patch("scripts.build_final_summary.file_hash", append_on_recheck):
            with self.assertRaisesRegex(ValueError, "changed while"):
                self.build(**options)
        self.assertFalse(self.output.exists())


class RuntimeFailureDisclosureTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "runtime-failure.json"
        self.matched = [{"status": "answered"}] * 5
        self.oracle = [{"status": "answered"}, {"status": "error"}, {"status": "error"}]
        self.disclosure = {"schema_version": 1, "purpose": "final_runtime_failure_disclosure",
            "protocol_sha256": "a" * 64, "observations_committed": 8,
            "status_by_kind": {"matched": {"answered": 5}, "oracle": {"answered": 1, "error": 2}},
            "first_error_task_index": 6, "last_error_task_index": 7,
            "original_gpu_instance_post_inference_fingerprint": "unavailable_cuda_unknown_error",
            "persistent_model_or_data_changes": False,
            "closure_recovery_policy": "Fresh closure checks do not reconstruct the lost GPU state.",
            "oracle_measurement_policy": "Keep errors in original denominators."}
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.disclosure), encoding="utf-8")
        self.checksum = builder.file_hash(self.path)

    def load(self, checksum=None):
        return builder.runtime_failure_summary(self.root, self.path, checksum or self.checksum,
            protocol_sha="a" * 64, records=self.matched, oracle_records=self.oracle, watched={})

    def test_disclosure_binds_original_errors_and_distinguishes_fresh_closure(self):
        value = self.load()
        self.assertFalse(value["original_results_replaced"])
        self.assertIn("does not establish", value["limitation"])
        self.assertEqual(value["status_by_kind"]["oracle"]["error"], 2)

    def test_tampered_disclosure_or_dropped_runtime_errors_is_refused(self):
        original = dict(self.disclosure)
        self.path.write_bytes(self.path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.load()
        for change in ({"observations_committed": 7}, {"first_error_task_index": 5},
                       {"original_gpu_instance_post_inference_fingerprint": "verified"},
                       {"persistent_model_or_data_changes": True}, {"closure_recovery_policy": "Verified original GPU"}):
            self.disclosure = {**original, **change}
            self.write()
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "preserved original"):
                self.load()

    def test_both_external_binding_arguments_are_required(self):
        with self.assertRaisesRegex(ValueError, "supplied together"):
            builder.runtime_failure_summary(self.root, self.path, None, protocol_sha="a" * 64,
                records=self.matched, oracle_records=self.oracle, watched={})


if __name__ == "__main__":
    unittest.main()
