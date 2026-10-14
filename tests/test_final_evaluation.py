"""Synthetic-only final execution tests; no real held-out data or ML imports."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from editguard.comparison import CONFIGURATIONS
from editguard.neural import file_sha256
from editguard.scoring import SCORER_VERSION
from editguard.short_qa import SHORT_QA_VERSION
from scripts.prepare_data import content_hash
from scripts.run_final_evaluation import (REQUIRED_SOURCES, atomic_json, execution_plan, locked_inputs,
                                         run_final, session_lock, workspace_path)


class FakeBase:
    device = "cpu"

    def __init__(self):
        self.fingerprint = "a" * 64

    def parameter_fingerprint(self):
        return self.fingerprint


class FakeAnswer:
    last_trace = {"fake_model": "synthetic-test-only"}

    def generate(self, text):
        return "Nimbus Basic v2 supports up to 3 devices."


class FakeRunner:
    """Prescribed fake observations test governance, never model performance."""
    retrieval_k = 3
    calibration = {"temperature": 1., "threshold": .5}

    def __init__(self, *, fail_at=None, status="answered", answer_kind="model_generated", mutate_base=False, nonanswer_text=None):
        self.base = FakeBase()
        self.answer_model = FakeAnswer()
        self.scope = SimpleNamespace(score=lambda question, correction: .1 if "price" in question else .9)
        self.retriever = SimpleNamespace(rank=lambda *args, **kwargs: [])
        self.calls = []
        self.fail_at, self.status, self.answer_kind, self.mutate_base = fail_at, status, answer_kind, mutate_base
        self.nonanswer_text = nonanswer_text

    def answer(self, question, context, facts, memory, *, configuration):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise KeyboardInterrupt("synthetic interruption")
        self.calls.append({"question": question, "context": copy.deepcopy(context), "facts": copy.deepcopy(facts),
                           "memory": copy.deepcopy(memory), "configuration": configuration})
        if self.mutate_base:
            self.base.fingerprint = "changed"
        price = "price" in question
        answer = "Nimbus Basic v2 costs 9 USD/month." if price else "Nimbus Basic v2 supports up to 3 devices."
        if configuration == "unchanged_base" and not price:
            answer = "Nimbus Basic v2 supports up to 5 devices."
        if self.status != "answered":
            answer = self.nonanswer_text
        return {"configuration": configuration, "question": question, "status": self.status,
            "answer": answer, "answer_kind": self.answer_kind, "route": "synthetic", "latency_ms": 1.,
            "selected_edit_id": "synthetic-edit" if not price and configuration.endswith("editor") else None,
            "retrieved_facts": [{"fact_id": "devices"}], "retrieval_candidates_raw": [{"edit_id": "synthetic-edit"}],
            "retrieval_candidates_eligible": [], "scope_scores": [], "input_text": "synthetic model input",
            "generation_trace": {"fake_model": True}, "error": None}


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.packet = self.root / "synthetic-packet"
        self.packet.mkdir()
        self.config = self.root / "config.json"
        self.config.write_text("{}", encoding="utf-8")
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_text("{}", encoding="utf-8")
        self.protocol_sha = file_sha256(self.protocol_path)
        common = {"entity_id": "nimbus", "entity_name": "Nimbus", "aliases": ["NMB"], "plan": "Basic", "version": "v2",
            "effective_from": "2025-01-01", "effective_to": None, "source": "controlled://synthetic/handbook", "split": "test"}
        self.facts = [{**common, "fact_id": "devices", "relation": "device_limit", "value": "5", "unit": "devices"},
                      {**common, "fact_id": "price", "relation": "price_monthly", "value": "9", "unit": "USD/month"}]
        correction = {**self.facts[0], "edit_id": "synthetic-edit", "entity_aliases": ["NMB"], "original_value": "5", "corrected_value": "3",
            "approval_status": "approved_in_simulated_scenario", "lineage_id": "synthetic-lineage", "effective_from": "2026-09-01", "supersedes": None}

        def row(identity, *, price=False, challenge=False):
            fact = self.facts[1 if price else 0]
            return {"id": identity, "family_id": "synthetic-family", "question":
                "What is the monthly price of Nimbus Basic version v2?" if price else
                "Please state Nimbus Basic version v2's device ceiling." if challenge else
                "How many devices does Nimbus Basic version v2 support?",
                "question_type": "challenge" if challenge else "direct", "label": 0 if price else 1,
                "correction": copy.deepcopy(correction), "expected_polarity": None,
                "request_context": {"entity_id": "nimbus", "plan": "Basic", "version": "v2", "as_of": "2026-10-11"},
                "expected_fact": {**fact, "value": "9" if price else "3", "as_of": "2026-10-11"}}

        self.populations = {"scope": [row("scope-positive"), row("scope-negative", price=True)],
                            "answer": [row("answer-0"), row("answer-1")], "challenge": [row("challenge", challenge=True)]}
        self.packet_manifest = {"dataset_id": "synthetic-final-fixture", "scope_normalized_overlap": {
            "scope_row_ids_sharing_train_pair": [], "scope_row_ids_sharing_validation_pair": []}}
        (self.packet / "manifest.json").write_text(json.dumps(self.packet_manifest), encoding="utf-8")
        self.packet_hash = file_sha256(self.packet / "manifest.json")
        self.protocol = {"protocol_id": "synthetic-lock", "configuration": {"path": "config.json", "sha256": file_sha256(self.config)},
            "execution": {"output_directory": "results", "ledger_path": "execution-ledger.json"},
            "source_sha256": {name: "synthetic-source-binding" for name in REQUIRED_SOURCES},
            "review_policy": "automated_final", "human_reviewed_data": False,
            "runtime_packages": {"synthetic-test-only": "1"}, "python_version": "synthetic-test-only",
            "scoring": {"primary_version": SCORER_VERSION, "primary_sha256": "synthetic-primary-sha", "short_qa_version": SHORT_QA_VERSION},
            "evaluation": {"configurations": list(CONFIGURATIONS), "matched_populations": ["scope", "challenge"],
                "oracle_populations": ["answer", "challenge"], "max_final_runs": 1,
                "bootstrap": {"paired": True, "cluster_unit": "entity_family", "seed": 42, "resamples": 2000, "confidence": .95},
                "expected_base_parameter_fingerprint_sha256": "a" * 64,
                "question_counts": {name: len(rows) for name, rows in self.populations.items()}}}
        self.protocol_path.write_text(json.dumps(self.protocol), encoding="utf-8")
        self.protocol_sha = file_sha256(self.protocol_path)

    def fake_locked_inputs(self, protocol_path, protocol_sha, packet, *, root):
        # Synthetic exact bindings reproduce the behavior expected from the
        # independently tested protocol/packet validators without ML artifacts.
        if file_sha256(protocol_path) != protocol_sha or file_sha256(self.config) != self.protocol["configuration"]["sha256"]:
            raise ValueError("Synthetic protocol/configuration binding mismatch")
        if file_sha256(packet / "manifest.json") != self.packet_hash:
            raise ValueError("Synthetic packet binding mismatch")
        return copy.deepcopy(self.protocol), copy.deepcopy(self.packet_manifest), copy.deepcopy(self.facts), copy.deepcopy(self.populations)

    def run_fixture(self, runner, *, resume=False, export_only=False):
        with patch("scripts.run_final_evaluation.locked_inputs", side_effect=self.fake_locked_inputs), patch("scripts.run_final_evaluation.resource_snapshot", return_value={"synthetic": True}):
            return run_final(self.protocol_path, self.protocol_sha, self.packet, root=self.root,
                             resume=resume, export_only=export_only, runner_factory=lambda config: runner)

    def test_complete_run_has_all_matched_denominators_and_separate_oracle(self):
        runner = FakeRunner()
        result = self.run_fixture(runner)
        self.assertEqual(result["questions"], 3)
        self.assertEqual(result["matched_prediction_records"], 15)
        self.assertEqual(result["oracle_records"], 3)
        self.assertTrue(result["base_weights_unchanged"])
        self.assertFalse(result["human_reviewed_data"])
        metrics = json.loads((self.root / "results/metrics.json").read_text())
        for config in CONFIGURATIONS:
            self.assertEqual(metrics["configurations"][config]["accuracy"]["denominator"], 3)
            self.assertEqual(metrics["by_population"]["scope"]["configurations"][config]["accuracy"]["denominator"], 2)
            self.assertEqual(metrics["by_population"]["challenge"]["configurations"][config]["accuracy"]["denominator"], 1)
        self.assertEqual(metrics["oracle_answer_by_population"]["answer"]["question_count"], 2)
        self.assertEqual(metrics["oracle_answer_by_population"]["challenge"]["question_count"], 1)
        grouped = {}
        for call in runner.calls:
            grouped.setdefault(call["question"], []).append(call)
        for calls in grouped.values():
            self.assertEqual([c["configuration"] for c in calls], list(CONFIGURATIONS))
            for field in ("facts", "memory", "context"):
                self.assertTrue(all(call[field] == calls[0][field] for call in calls))
        self.assertEqual(json.loads((self.root / "execution-ledger.json").read_text())["fresh_final_runs"], 1)

    def test_abstentions_and_unsupported_short_qa_do_not_disappear(self):
        self.run_fixture(FakeRunner(status="clarification", answer_kind="deterministic_clarification"))
        metrics = json.loads((self.root / "results/metrics.json").read_text())
        for config in CONFIGURATIONS:
            self.assertEqual(metrics["configurations"][config]["accuracy"]["denominator"], 3)
            self.assertEqual(metrics["configurations"][config]["coverage"]["value"], 0)
            self.assertEqual(metrics["configurations"][config]["abstentions"], 3)
            self.assertEqual(metrics["configurations"][config]["model_generated_answer_accuracy"]["numerator"], 0)
        secondary = json.loads((self.root / "results/short_qa_metrics.json").read_text())
        for config in CONFIGURATIONS:
            self.assertEqual(secondary["by_population"]["challenge"][config]["all_requested_qa"]["denominator"], 1)

    def test_clarification_text_containing_correct_fact_is_explicitly_uncredited(self):
        self.run_fixture(FakeRunner(status="conflict", answer_kind="deterministic_clarification",
                                   nonanswer_text="Nimbus Basic v2 supports up to 3 devices."))
        rows = [json.loads(line) for line in (self.root / "results/predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(row["answer"] == "Nimbus Basic v2 supports up to 3 devices." for row in rows))
        self.assertTrue(all(row["score"]["correct"] is False and row["short_qa"]["correct"] is False for row in rows))
        self.assertTrue(all(row["score"]["endorsed_fact"] is None and row["score"]["needs_review"] is False for row in rows))
        metrics = json.loads((self.root / "results/metrics.json").read_text())
        self.assertTrue(all(result["accuracy"]["numerator"] == 0 and result["accuracy"]["denominator"] == 3
                            for result in metrics["configurations"].values()))

    def test_error_text_containing_correct_fact_remains_nonanswer(self):
        self.run_fixture(FakeRunner(status="error", nonanswer_text="Nimbus Basic v2 supports up to 3 devices."))
        rows = [json.loads(line) for line in (self.root / "results/predictions.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["score"]["correct"] is False for row in rows))

    def test_failed_raw_scope_diagnostic_stays_in_fixed_denominator(self):
        runner = FakeRunner()
        runner.scope.score = lambda *args: (_ for _ in ()).throw(RuntimeError("synthetic raw-pair failure"))
        self.run_fixture(runner)
        rows = [json.loads(line) for line in (self.root / "results/predictions.jsonl").read_text().splitlines()]
        scope_rows = [row for row in rows if row["configuration"] == "trained_scope_editor"]
        self.assertEqual(len(scope_rows), 3)
        self.assertTrue(all(row["scope_probability"] is None and row["scope_diagnostic_error"]["type"] == "RuntimeError" for row in scope_rows))
        metrics = json.loads((self.root / "results/metrics.json").read_text())["configurations"]["trained_scope_editor"]
        self.assertEqual(metrics["accuracy"]["denominator"], 3)
        self.assertEqual(metrics["scope"]["counts"]["missing_probability"], 3)
        self.assertEqual(metrics["scope"]["recall"]["denominator"], 2)
        self.assertIsNone(metrics["scope"]["brier"]["value"])

    def test_complete_run_refuses_fresh_run_and_resume(self):
        self.run_fixture(FakeRunner())
        for resume in (False, True):
            with self.subTest(resume=resume), self.assertRaises(ValueError):
                self.run_fixture(FakeRunner(), resume=resume)

    def test_exact_resume_preserves_observations_without_replaying_committed_tasks(self):
        partial = FakeRunner(fail_at=2)
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(partial)
        journal = self.root / "results/observations"
        saved = {p.name: p.read_bytes() for p in journal.glob("*.json")}
        self.assertEqual(len(saved), 2)
        resumed = FakeRunner()
        result = self.run_fixture(resumed, resume=True)
        self.assertEqual(len(resumed.calls), 13)
        self.assertTrue(result["resumed"])
        self.assertEqual(result["matched_prediction_records"], 15)
        self.assertEqual(result["oracle_records"], 3)
        for name, data in saved.items():
            self.assertEqual((journal / name).read_bytes(), data)

    def test_resume_refuses_config_protocol_and_packet_drift(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(FakeRunner(fail_at=1))
        for path in (self.config, self.protocol_path, self.packet / "manifest.json"):
            original = path.read_bytes()
            path.write_text('{"changed":true}', encoding="utf-8")
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                self.run_fixture(FakeRunner(), resume=True)
            path.write_bytes(original)

    def test_resume_refuses_observation_tampering(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(FakeRunner(fail_at=2))
        target = self.root / "results/observations/00000000.json"
        entry = json.loads(target.read_text())
        entry["record"]["answer"] = "tampered answer"
        atomic_json(target, entry)
        with self.assertRaises(ValueError):
            self.run_fixture(FakeRunner(), resume=True)

    def test_resume_binds_family_and_reviewed_aliases_to_frozen_facts(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(FakeRunner(fail_at=1))
        target = self.root / "results/observations/00000000.json"
        original = json.loads(target.read_text())
        progress_path = self.root / "results/progress.json"
        for field in ("family_id", "aliases"):
            entry = copy.deepcopy(original)
            if field == "family_id":
                entry["record"]["family_id"] = "another-family"
            else:
                entry["record"]["expected_fact"]["aliases"] = ["unreviewed-other-product"]
            atomic_json(target, entry)
            progress = json.loads(progress_path.read_text())
            progress["last_observation_sha256"] = file_sha256(target)
            atomic_json(progress_path, progress)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "locked question"):
                self.run_fixture(FakeRunner(), resume=True)

    def test_atomic_orphan_is_recovered_once_without_new_inference(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(FakeRunner(fail_at=2))
        progress_path = self.root / "results/progress.json"
        progress = json.loads(progress_path.read_text())
        progress.update(next_task_index=1, last_observation_sha256=file_sha256(self.root / "results/observations/00000000.json"))
        atomic_json(progress_path, progress)
        runner = FakeRunner()
        result = self.run_fixture(runner, resume=True)
        self.assertEqual(len(runner.calls), 13)
        self.assertEqual(result["recovered_atomic_orphans"], 1)

    def test_missing_auxiliaries_and_structured_outputs_are_not_ml_results(self):
        runner = FakeRunner()
        runner.scope = None
        with self.assertRaisesRegex(ValueError, "both completed"):
            self.run_fixture(runner)
        runner = FakeRunner(answer_kind="deterministic_structured")
        with self.assertRaisesRegex(ValueError, "structured"):
            self.run_fixture(runner, resume=True)
        self.assertFalse((self.root / "results/manifest.json").exists())

    def test_frozen_base_change_fails_and_cannot_be_resumed_with_different_base(self):
        with self.assertRaisesRegex(RuntimeError, "weights changed"):
            self.run_fixture(FakeRunner(mutate_base=True))
        resumed = FakeRunner()
        resumed.base.fingerprint = "different-frozen-base"
        with self.assertRaisesRegex(ValueError, "unfinished"):
            self.run_fixture(resumed, resume=True)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            self.run_fixture(FakeRunner(), resume=True)
        self.assertEqual(json.loads((self.root / "execution-ledger.json").read_text())["status"], "invalidated")

    def test_finalization_recovery_seals_existing_exports_without_loading_models(self):
        result = self.run_fixture(FakeRunner())
        ledger = self.root / "execution-ledger.json"
        saved = json.loads(ledger.read_text())
        saved["status"] = "running"
        atomic_json(ledger, saved)
        with patch("scripts.run_final_evaluation.locked_inputs", side_effect=self.fake_locked_inputs):
            recovered = run_final(self.protocol_path, self.protocol_sha, self.packet, root=self.root, resume=True,
                                  runner_factory=lambda config: self.fail("Completed finalization must not reload models"))
        self.assertEqual(recovered["outputs_sha256"], result["outputs_sha256"])
        self.assertEqual(json.loads(ledger.read_text())["status"], "completed")

    def test_export_failure_recovers_complete_journal_without_model_load(self):
        def interrupted_export(path, value):
            if path.name == "metrics.json":
                raise OSError("synthetic export interruption")
            return atomic_json(path, value)
        with patch("scripts.run_final_evaluation.atomic_json", side_effect=interrupted_export):
            with self.assertRaisesRegex(OSError, "export interruption"):
                self.run_fixture(FakeRunner())
        journal = self.root / "results/observations"
        saved = {path.name: path.read_bytes() for path in journal.glob("*.json")}
        self.assertEqual(len(saved), 18)
        with patch("scripts.run_final_evaluation.locked_inputs", side_effect=self.fake_locked_inputs):
            result = run_final(self.protocol_path, self.protocol_sha, self.packet, root=self.root,
                               resume=True, export_only=True,
                               runner_factory=lambda config: self.fail("Export recovery must not load models"))
        self.assertEqual(result["matched_prediction_records"], 15)
        self.assertEqual(result["oracle_records"], 3)
        self.assertEqual({path.name: path.read_bytes() for path in journal.glob("*.json")}, saved)

    def test_export_only_refuses_partial_journal_and_fresh_run(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(FakeRunner(fail_at=1))
        with patch("scripts.run_final_evaluation.locked_inputs", side_effect=self.fake_locked_inputs):
            for resume in (False, True):
                with self.subTest(resume=resume), self.assertRaises(ValueError):
                    run_final(self.protocol_path, self.protocol_sha, self.packet, root=self.root,
                              resume=resume, export_only=True,
                              runner_factory=lambda config: self.fail("Rejected export must not load models"))

    def test_crash_after_ledger_commit_recovers_empty_initialization(self):
        with patch("scripts.run_final_evaluation.initialize_progress", side_effect=KeyboardInterrupt("initialization crash")):
            with self.assertRaises(KeyboardInterrupt):
                self.run_fixture(FakeRunner())
        self.assertFalse((self.root / "results").exists())
        self.assertEqual(json.loads((self.root / "execution-ledger.json").read_text())["status"], "initializing")
        result = self.run_fixture(FakeRunner(), resume=True)
        self.assertEqual(result["fresh_final_runs"], 1)
        self.assertEqual(result["matched_prediction_records"], 15)

    def test_crash_after_output_creation_recovers_zero_observations_only(self):
        def interrupted_creation(output, binding):
            output.mkdir()
            (output / "observations").mkdir()
            (output / "progress.json.tmp-crashed").write_text("partial initial write")
            raise KeyboardInterrupt("initialization crash")
        with patch("scripts.run_final_evaluation.initialize_progress", side_effect=interrupted_creation):
            with self.assertRaises(KeyboardInterrupt):
                self.run_fixture(FakeRunner())
        self.assertEqual(self.run_fixture(FakeRunner(), resume=True)["matched_prediction_records"], 15)

    def test_missing_initial_progress_cannot_adopt_existing_observations(self):
        def interrupted_creation(output, binding):
            output.mkdir()
            (output / "observations").mkdir()
            (output / "observations/00000000.json").write_text("{}")
            raise KeyboardInterrupt("initialization crash")
        with patch("scripts.run_final_evaluation.initialize_progress", side_effect=interrupted_creation):
            with self.assertRaises(KeyboardInterrupt):
                self.run_fixture(FakeRunner())
        with self.assertRaisesRegex(ValueError, "already committed"):
            self.run_fixture(FakeRunner(), resume=True)

    def test_missing_progress_from_started_run_cannot_restart(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(FakeRunner(fail_at=1))
        (self.root / "results/progress.json").unlink()
        with self.assertRaisesRegex(ValueError, "initialization ledger"):
            self.run_fixture(FakeRunner(), resume=True)

    def test_os_lock_rejects_concurrent_session_and_is_released_after_crash(self):
        code = "from pathlib import Path; import sys; from scripts.run_final_evaluation import session_lock;\nwith session_lock(Path(sys.argv[1])):\n print('locked', flush=True)\n sys.stdin.read()"
        process = subprocess.Popen([sys.executable, "-u", "-c", code, str(self.root / "execution-ledger.json")],
                                   cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), "locked")
            with patch("scripts.run_final_evaluation.locked_inputs", side_effect=self.fail):
                with self.assertRaisesRegex(ValueError, "execution lock"):
                    run_final(self.protocol_path, self.protocol_sha, self.packet, root=self.root,
                              runner_factory=lambda config: self.fail("Concurrent session must not load models"))
            self.assertFalse((self.root / "execution-ledger.json").exists())
            process.kill()
            process.wait(timeout=10)
            self.assertEqual(self.run_fixture(FakeRunner())["matched_prediction_records"], 15)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()

    def test_protocol_requires_actual_nested_uncertainty_base_and_source_bindings(self):
        def checked(protocol):
            with patch("scripts.final_protocol.validate_protocol", return_value=protocol), patch(
                    "scripts.prepare_final_packet.validate_packet", return_value=(
                        self.packet_manifest, self.facts, self.populations["scope"], self.populations["answer"], self.populations["challenge"])):
                return locked_inputs(self.protocol_path, self.protocol_sha, self.packet, root=self.root)
        checked(copy.deepcopy(self.protocol))
        for mutation in ("bootstrap", "base", "packages", "transitive_source"):
            protocol = copy.deepcopy(self.protocol)
            if mutation == "bootstrap":
                protocol["evaluation"].pop("bootstrap")
                protocol["bootstrap"] = self.protocol["evaluation"]["bootstrap"]
            elif mutation == "base":
                protocol["evaluation"].pop("expected_base_parameter_fingerprint_sha256")
            elif mutation == "packages":
                protocol["runtime_packages"] = {}
            else:
                protocol["source_sha256"].pop("scripts/review_dataset.py")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                checked(protocol)

    def test_finite_execution_plan_has_separate_roles_and_no_missing_configs(self):
        plan = execution_plan(self.populations)
        self.assertEqual(len(plan), 18)
        self.assertEqual(sum(p["kind"] == "matched" for p in plan), 15)
        self.assertEqual(sum(p["kind"] == "oracle" for p in plan), 3)
        self.assertFalse(any(p["population"] == "answer" and p["kind"] == "matched" for p in plan))

    def test_workspace_paths_cannot_escape(self):
        for path in ("../escape", str(self.root / "absolute"), "", None):
            with self.subTest(path=path), self.assertRaises(ValueError):
                workspace_path(self.root, path)


if __name__ == "__main__":
    unittest.main()
