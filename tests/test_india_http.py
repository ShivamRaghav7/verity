"""Software-only India workload checks. Fixtures are not model quality evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from editguard_chat.server import ChatApplication
from scripts.verify_india_http import case_plan, run, validate_answer, MODEL_ID, MODEL_REVISION


class ContractFixtureBackend:
    """Explicit non-neural fixture, with deliberately wrong endorsed answers."""
    model = MODEL_ID

    def __init__(self, *, fail_on=None, interrupt_on=None, fixture=True):
        self.calls = 0
        self.fail_on = fail_on
        self.interrupt_on = interrupt_on
        self.fixture = fixture
        self.content = "Deliberately wrong software fixture"

    def status(self):
        return {"ready": True, "name": MODEL_ID, "revision": MODEL_REVISION,
                "supports_images": False, "backend": "Unit-test contract fixture; no model loaded"}

    def answer_with_facts(self, question, facts, history):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("Injected worker transport failure")
        if self.calls == self.interrupt_on:
            raise KeyboardInterrupt("Injected interruption after the request intent checkpoint")
        return {"content": self.content, "raw_output": "SOFTWARE FIXTURE ONLY: no neural generation",
                "generation": {"model": MODEL_ID, "revision": MODEL_REVISION, "fixture": self.fixture}}


class FixtureClient:
    def __init__(self, *, existing=None, policy="iterative", **kwargs):
        self.temporary = tempfile.TemporaryDirectory()
        self.backend = ContractFixtureBackend(**kwargs)
        self.app = ChatApplication(self.temporary.name, backend=self.backend, context_policy=policy)
        self.token = None
        self.calls = []
        if existing == "chat":
            self.app.store.create_chat()
        elif existing == "history":
            row = self.app.store.publish_correction("User record?", "Preserve this", "Fixture", "User scope")
            self.app.store.withdraw(row["id"])

    def close(self):
        self.app.store.close()
        self.temporary.cleanup()

    def request(self, path, payload=None):
        self.calls.append((path, payload))
        if path == "/api/status":
            return self.app.status()
        if path == "/api/corrections" and payload is None:
            return {"revision": self.app.store.get_revision(), "corrections": self.app.store.list_corrections(),
                    "events": self.app.store.list_events()}
        if path == "/api/chats" and payload is None:
            return {"chats": self.app.store.list_chats()}
        return self.app.mutate(path, payload)


class IndiaHTTPTests(unittest.TestCase):
    def execute(self, client):
        self.addCleanup(client.close)
        with tempfile.TemporaryDirectory() as temporary, patch("scripts.verify_india_http.Client", return_value=client):
            output = Path(temporary) / "software-fixture.json"
            receipt = run("http://fixture.invalid", output)
            text = output.read_text()
            self.assertEqual(json.loads(text), receipt)
            self.assertNotIn(client.app.token, text)
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                run("http://fixture.invalid", output)
        return receipt

    def test_plan_has_exactly_fourteen_fixed_slots_and_revisions(self):
        rows = case_plan()
        self.assertEqual(len(rows), 14)
        self.assertEqual(len({row["id"] for row in rows}), 14)
        self.assertEqual([row["expected_revision"] for row in rows], [9] * 3 + [10] * 6 + [9, 10] + [11] * 3)
        self.assertEqual([row["depth"] for row in rows[:3]], [1, 2, 3])
        self.assertEqual([row["question"]["expected_answer"] for row in rows[:3]],
                         ["Bengaluru Platform Team", "Meera Iyer", "Chennai"])
        self.assertEqual([row["question"]["expected_answer"] for row in rows[9:11]], ["Chennai", "Hyderabad"])
        self.assertTrue(all(row["outcome"] == "unexecuted" for row in rows))

    def test_explicit_fixture_is_never_accepted_as_fourteen_neural_responses(self):
        client = FixtureClient()
        receipt = self.execute(client)
        self.assertEqual(client.backend.calls, 14)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["outcome_counts"], {"correct": 0, "wrong": 0, "abstain": 0,
                                                   "execution_error": 14, "unexecuted": 0})
        self.assertEqual(receipt["verified_neural_response_count"], 0)
        self.assertEqual(receipt["neural_accuracy_denominator"], 14)
        self.assertTrue(receipt["all_consistency_checks_passed"])
        self.assertTrue(receipt["initial_handbook_restored"])
        self.assertEqual(receipt["latency"]["request_n"], 13)
        self.assertEqual(len(receipt["owned_chat_ids"]), 12)
        active = [row for row in client.app.store.list_corrections() if row["status"] == "active"]
        self.assertEqual(len(active), 9)
        self.assertEqual(client.app.store.get_revision(), 11)
        self.assertFalse(any(path.endswith("/withdraw") for path, _ in client.calls))
        self.assertEqual(sum(path == "/api/compare" for path, _ in client.calls), 1)

    def test_wrong_and_abstained_contract_answers_remain_in_denominator_without_retry(self):
        # This mocks the accepted metadata shape to exercise accounting only.
        # It makes no assertion about actual model quality.
        for content, expected in (("Deliberately wrong", "wrong"), ("UNKNOWN", "abstain")):
            with self.subTest(content=content):
                client = FixtureClient(fixture=False)
                client.backend.content = content
                receipt = self.execute(client)
                self.assertEqual(client.backend.calls, 14)
                self.assertEqual(receipt["outcome_counts"][expected], 14)
                self.assertEqual(receipt["outcome_counts"]["correct"], 0)
                self.assertEqual(sum(receipt["outcome_counts"].values()), 14)

    def test_transport_failure_in_comparison_records_both_errors_and_never_replays(self):
        client = FixtureClient(fail_on=11, fixture=False)
        receipt = self.execute(client)
        self.assertEqual(client.backend.calls, 14)
        self.assertEqual(receipt["outcome_counts"]["execution_error"], 2)
        self.assertEqual(receipt["outcome_counts"]["wrong"], 12)
        self.assertTrue(receipt["initial_handbook_restored"])
        self.assertEqual(sum(path == "/api/compare" for path, _ in client.calls), 1)

    def test_failed_three_hop_keeps_comparison_unexecuted_and_denominator_fixed(self):
        client = FixtureClient(fail_on=6, fixture=False)
        receipt = self.execute(client)
        self.assertEqual(client.backend.calls, 12)
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(receipt["outcome_counts"]["unexecuted"], 2)
        self.assertEqual(receipt["outcome_counts"]["execution_error"], 1)
        self.assertEqual(receipt["outcome_counts"]["wrong"], 11)
        self.assertFalse(any(path == "/api/compare" for path, _ in client.calls))
        self.assertTrue(receipt["initial_handbook_restored"])

    def test_existing_history_or_chats_are_never_mutated(self):
        for existing in ("history", "chat"):
            with self.subTest(existing=existing):
                client = FixtureClient(existing=existing)
                before = client.app.store.export_snapshot()["tables"]
                receipt = self.execute(client)
                self.assertEqual(client.backend.calls, 0)
                self.assertEqual(receipt["status"], "interrupted")
                self.assertEqual(receipt["outcome_counts"]["unexecuted"], 14)
                self.assertEqual(client.app.store.export_snapshot()["tables"], before)
                self.assertFalse(any(payload is not None for _, payload in client.calls))

    def test_wrong_policy_refused_before_approval_or_generation(self):
        client = FixtureClient(policy="retrieved")
        receipt = self.execute(client)
        self.assertEqual(client.backend.calls, 0)
        self.assertFalse(receipt["consistency_checks"]["iterative_serving_policy"])
        self.assertEqual(client.app.store.get_revision(), 0)

    def test_interruption_preserves_inflight_intent_and_refuses_a_rerun(self):
        client = FixtureClient(interrupt_on=5, fixture=False)
        self.addCleanup(client.close)
        with tempfile.TemporaryDirectory() as temporary, patch("scripts.verify_india_http.Client", return_value=client):
            output = Path(temporary) / "interrupted-software-fixture.json"
            with self.assertRaises(KeyboardInterrupt):
                run("http://fixture.invalid", output)
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["status"], "interrupted")
            self.assertEqual(receipt["cases"][4]["request_state"], "in_flight")
            self.assertEqual(receipt["cases"][4]["outcome"], "unexecuted")
            self.assertEqual(client.backend.calls, 5)
            self.assertTrue(receipt["initial_handbook_restored"])
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                run("http://fixture.invalid", output)
            self.assertEqual(client.backend.calls, 5)
            self.assertEqual(client.app.store.get_revision(), 11)

    def test_missing_revision_policy_or_model_provenance_cannot_earn_a_correct_score(self):
        case = case_plan()[0]
        answer = {"content": case["question"]["expected_answer"], "evidence": {
            "generation_kind": "model", "generation": {"model": MODEL_ID, "revision": MODEL_REVISION},
            "context_policy": "iterative", "configured_context_policy": "iterative",
            "knowledge_revision": 9, "raw_model_output": "Contract-shaped software fixture"}}
        self.assertEqual(validate_answer(case, answer, 9)["outcome"], "correct")
        changes = [("generation_kind", "deterministic_clarification"), ("knowledge_revision", 10),
                   ("context_policy", "retrieved"), ("raw_model_output", ""),
                   ("generation", {"model": MODEL_ID, "revision": MODEL_REVISION, "fixture": True}),
                   ("generation", {"model": "different-model", "revision": MODEL_REVISION})]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(answer)
                changed["evidence"][key] = value
                with self.assertRaisesRegex(ValueError, "provenance"):
                    validate_answer(case, changed, 9)


if __name__ == "__main__":
    unittest.main()
