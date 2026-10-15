"""Finite supplemental recovery tests using synthetic exports and fake models."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import recover_oracle_runtime as recovery
from scripts.final_protocol import read_json
from tests import test_final_summary as summary_fixtures
from tests.test_final_evaluation import FakeRunner


class OracleRuntimeRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = summary_fixtures.FinalSummaryTests("test_completed_evidence_preserves_cohorts_and_primary_secondary")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.protocol = self.fixture.fixture.protocol_path
        self.protocol_sha = self.fixture.fixture.protocol_sha
        self.packet = self.fixture.fixture.packet
        self.original_run = self.fixture.run
        self.output = self.root / "supplement"
        self.declaration = self.root / "supplement-declaration.json"
        self.enterContext(patch("scripts.recover_oracle_runtime.locked_inputs", side_effect=self.fixture.fixture.fake_locked_inputs))
        initial = FakeRunner()
        calls = []
        def failing_generator(text):
            calls.append(text)
            if len(calls) <= 2:
                raise RuntimeError("CUDA error: unknown error\nsynthetic runtime failure")
            return "Nimbus Basic v2 supports up to 3 devices."
        initial.answer_model.generate = failing_generator
        self.fixture.complete(initial)
        self.originals = recovery.read_records(self.original_run / "oracle_predictions.jsonl")
        self.assertEqual(sum(recovery.cuda_unknown(row) for row in self.originals), 2)
        self.original_bytes = {name: (self.original_run / name).read_bytes() for name in
                               ("manifest.json", "predictions.jsonl", "oracle_predictions.jsonl", "metrics.json", "short_qa_metrics.json")}

    def prepare(self):
        result = recovery.prepare(self.protocol, self.protocol_sha, self.packet, self.original_run,
                                  self.output, self.declaration, root=self.root)
        self.declaration_sha = recovery.file_sha256(self.declaration)
        return result

    def execute(self, runner=None, *, resume=False):
        return recovery.run(self.declaration, self.declaration_sha, root=self.root,
                            resume=resume, runner_factory=lambda path: runner or FakeRunner())

    def assert_originals_unchanged(self):
        self.assertEqual(self.original_bytes, {name: (self.original_run / name).read_bytes() for name in self.original_bytes})

    def test_status_only_selection_and_successes_preserved_without_reexecution(self):
        declaration = self.prepare()
        self.assertEqual(declaration["selected_questions"], 2)
        self.assertEqual(declaration["maximum_new_generation_calls"], 2)
        self.assertFalse(declaration["training_or_parameter_selection"])
        self.assertFalse(declaration["primary_results_replaced"])
        runner = FakeRunner()
        calls = []
        runner.answer_model.generate = lambda text: calls.append(text) or "Nimbus Basic v2 supports up to 3 devices."
        metadata = self.execute(runner)
        self.assertEqual(len(calls), 2)
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["oracle_questions"], 4)
        self.assertEqual(metadata["not_attempted_selected_count"], 0)
        combined = recovery.read_records(self.output / "combined_oracle_predictions.jsonl")
        untouched = {row["id"]: row for row in self.originals if not recovery.cuda_unknown(row)}
        self.assertTrue(all(row == untouched[row["id"]] for row in combined if row["id"] in untouched))
        metrics = read_json(self.output / "metrics.json")
        self.assertEqual(metrics["configurations"]["oracle_answer"]["accuracy"]["denominator"], 4)
        self.assertEqual(metrics["by_population"]["answer"]["question_count"], 3)
        self.assertEqual(metrics["by_population"]["challenge"]["question_count"], 1)
        self.assertEqual(metadata["sessions"][0]["base_parameter_sha256_before"], metadata["sessions"][0]["base_parameter_sha256_after"])
        self.assert_originals_unchanged()

    def test_wrong_answers_are_not_selected_merely_because_incorrect(self):
        failed = {"status": "answered", "answer": "Wrong answer", "error": None, "score": {"correct": False}}
        self.assertFalse(recovery.cuda_unknown(failed))
        self.assertFalse(recovery.cuda_unknown({"status": "error", "error": "ordinary generation error"}))
        self.assertFalse(recovery.cuda_unknown({"status": "error", "error": "CUDA out of memory"}))
        self.assertTrue(recovery.cuda_unknown({"status": "error", "error": "CUDA error: unknown error"}))

    def test_new_cuda_error_aborts_immediately_and_full_denominator_is_retained(self):
        self.prepare()
        runner = FakeRunner()
        calls = []
        def failing(text):
            calls.append(text)
            raise RuntimeError("CUDA error: unknown error")
        runner.answer_model.generate = failing
        with self.assertRaisesRegex(RuntimeError, "stop supplemental"):
            self.execute(runner)
        self.assertEqual(len(calls), 1)
        interrupted = read_json(self.output / "interrupted_manifest.json")
        self.assertEqual(interrupted["oracle_questions"], 4)
        self.assertEqual(interrupted["not_attempted_selected_count"], 1)
        self.assertEqual(interrupted["durable_attempt_reservations"], 1)
        metrics = read_json(self.output / "metrics.json")
        self.assertEqual(metrics["configurations"]["oracle_answer"]["accuracy"]["denominator"], 4)
        resumed = FakeRunner()
        resumed_calls = []
        resumed.answer_model.generate = lambda text: resumed_calls.append(text) or "Nimbus Basic v2 supports up to 3 devices."
        result = self.execute(resumed, resume=True)
        self.assertEqual(len(resumed_calls), 1)
        self.assertEqual(result["durable_attempt_reservations"], 2)
        self.assertEqual(result["sessions"][0]["fingerprint_after_status"], "unavailable_for_this_interrupted_session")
        self.assert_originals_unchanged()

    def test_input_mismatch_rejected_before_generator_call(self):
        self.prepare()
        runner = FakeRunner()
        calls = []
        runner.answer_model.generate = lambda text: calls.append(text) or "answer"
        def mismatched_input(proxy, row, facts):
            proxy.answer_model.generate("changed input")
            self.fail("Input mismatch should have raised before result construction")
        with patch("scripts.recover_oracle_runtime.oracle_answer_result", side_effect=mismatched_input):
            with self.assertRaisesRegex(ValueError, "byte-for-byte"):
                self.execute(runner)
        self.assertEqual(calls, [])
        self.assert_originals_unchanged()

    def test_crash_after_reservation_cannot_repeat_that_generation(self):
        self.prepare()
        runner = FakeRunner()
        calls = []
        def interrupted(text):
            calls.append(text)
            raise KeyboardInterrupt("synthetic crash after generator reservation")
        runner.answer_model.generate = interrupted
        with self.assertRaises(KeyboardInterrupt):
            self.execute(runner)
        self.assertEqual(len(calls), 1)
        resumed = FakeRunner()
        remaining_calls = []
        resumed.answer_model.generate = lambda text: remaining_calls.append(text) or "Nimbus Basic v2 supports up to 3 devices."
        result = self.execute(resumed, resume=True)
        self.assertEqual(len(remaining_calls), 1)
        self.assertEqual(result["durable_attempt_reservations"], 2)
        recovered = recovery.read_records(self.output / "recovery_predictions.jsonl")
        interrupted_record = next(row for row in recovered if row["runtime_recovery"].get("interrupted_reservation"))
        self.assertIs(interrupted_record["score"]["correct"], False)
        self.assertIsNone(interrupted_record["latency_ms"])
        self.assertEqual(read_json(self.output / "metrics.json")["configurations"]["oracle_answer"]["accuracy"]["denominator"], 4)
        self.assert_originals_unchanged()

    def test_later_interrupted_reservation_preserves_previous_hash_and_is_not_replayed(self):
        self.prepare()
        runner = FakeRunner()
        calls = []
        def interrupted_second(text):
            calls.append(text)
            if len(calls) == 2:
                raise KeyboardInterrupt("synthetic second-call crash")
            return "Nimbus Basic v2 supports up to 3 devices."
        runner.answer_model.generate = interrupted_second
        with self.assertRaises(KeyboardInterrupt):
            self.execute(runner)
        first = (self.output / "observations/00000000.json").read_bytes()
        resumed = FakeRunner()
        resumed.answer_model.generate = lambda text: self.fail("All selected calls were already durably reserved")
        result = self.execute(resumed, resume=True)
        self.assertEqual(result["interrupted_reservation_count"], 1)
        self.assertEqual(result["durable_attempt_reservations"], 2)
        self.assertEqual((self.output / "observations/00000000.json").read_bytes(), first)
        self.assert_originals_unchanged()

    def test_completed_supplement_refuses_fresh_run_and_resume(self):
        self.prepare()
        self.execute()
        for resume in (False, True):
            with self.subTest(resume=resume), self.assertRaises(ValueError):
                self.execute(resume=resume)

    def test_declaration_original_exports_and_source_snapshots_cannot_drift(self):
        self.prepare()
        original_declaration = self.declaration.read_bytes()
        self.declaration.write_bytes(original_declaration + b" ")
        with self.assertRaisesRegex(ValueError, "external checksum"):
            self.execute()
        self.declaration.write_bytes(original_declaration)
        target = self.original_run / "oracle_predictions.jsonl"
        original = target.read_bytes()
        target.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.execute()
        target.write_bytes(original)
        runner = FakeRunner()
        runner.answer_model.generate = lambda text: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.execute(runner)
        (self.output / "source/oracle_predictions.jsonl").write_bytes(b"altered immutable source")
        with self.assertRaisesRegex(ValueError, "source snapshot"):
            self.execute(resume=True)

    def test_different_new_process_base_rejected_before_generation(self):
        self.prepare()
        runner = FakeRunner()
        runner.base.fingerprint = "f" * 64
        calls = []
        runner.answer_model.generate = lambda text: calls.append(text) or "answer"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.execute(runner)
        self.assertEqual(calls, [])
        self.assert_originals_unchanged()

    def test_new_process_base_mutation_permanently_invalidates_supplement(self):
        self.prepare()
        runner = FakeRunner()
        def mutated(text):
            runner.base.fingerprint = "c" * 64
            return "Nimbus Basic v2 supports up to 3 devices."
        runner.answer_model.generate = mutated
        with self.assertRaisesRegex(ValueError, "invalidated"):
            self.execute(runner)
        self.assertFalse((self.output / "manifest.json").exists())
        with self.assertRaisesRegex(ValueError, "cannot rerun"):
            self.execute(resume=True)
        self.assert_originals_unchanged()

    def test_original_run_argument_must_match_frozen_execution_path(self):
        with self.assertRaisesRegex(ValueError, "Requested original run"):
            recovery.prepare(self.protocol, self.protocol_sha, self.packet, self.root / "another-run",
                             self.output, self.declaration, root=self.root)
        self.assertFalse(self.declaration.exists())


if __name__ == "__main__":
    unittest.main()
