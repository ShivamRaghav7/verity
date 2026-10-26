"""History-workload accounting/safety fixtures; no model is loaded or evaluated."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from editguard_chat.server import ChatApplication
from scripts.verify_operations_history import run, MODEL_ID, MODEL_REVISION


class NoNeuralModelFixture:
    """Deliberately wrong contract responses, never model-performance evidence."""
    model = MODEL_ID

    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on
        self.content = "Deliberately wrong fixture"

    def status(self):
        return {"ready": True, "name": MODEL_ID, "revision": MODEL_REVISION, "supports_images": False,
                "backend": "explicit unit-test fixture; no model loaded"}

    def answer_with_facts(self, question, facts, history):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("Injected neural-worker transport failure fixture")
        return {"content": self.content, "raw_output": "UNIT TEST ONLY: no actual neural generation",
                "generation": {"model": MODEL_ID, "revision": MODEL_REVISION, "fixture": True},
                "format": {"verified": False}}


class FixtureClient:
    def __init__(self, *, fail_on=None, fail_rollback=False, context_policy="retrieved", existing=None):
        self.temporary = tempfile.TemporaryDirectory()
        self.backend = NoNeuralModelFixture(fail_on=fail_on)
        self.app = ChatApplication(self.temporary.name, backend=self.backend, context_policy=context_policy)
        self.token = None
        self.calls = []
        self.fail_rollback = fail_rollback
        if existing == "history":
            fact = self.app.store.publish_correction("Keep user history?", "Yes", "Fixture", "User scope")
            self.app.store.withdraw(fact["id"])
        elif existing == "chat":
            self.app.store.create_chat()

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
        if path.endswith("/rollback") and self.fail_rollback:
            raise RuntimeError("Injected atomic rollback request failure fixture")
        return self.app.mutate(path, payload)


class OperationsHistoryTests(unittest.TestCase):
    def execute(self, client):
        self.addCleanup(client.close)
        with tempfile.TemporaryDirectory() as temporary, patch("scripts.verify_operations_history.Client", return_value=client):
            path = Path(temporary) / "history-fixture.json"
            receipt = run("http://fixture.invalid", path)
            saved = path.read_text()
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                run("http://fixture.invalid", path)
        self.assertNotIn(client.app.token, saved, "Never save website tokens in evidence")
        return receipt

    def test_four_wrong_answers_preserved_without_retry_and_atomic_rollback_observed(self):
        client = FixtureClient()
        receipt = self.execute(client)
        self.assertEqual(client.backend.calls, 4)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["outcome_counts"], {"correct": 0, "wrong": 4, "abstain": 0, "execution_error": 0, "unexecuted": 0})
        self.assertEqual(receipt["neural_accuracy_denominator"], 4)
        self.assertEqual(len(receipt["owned_fact_ids"]), 10)
        self.assertTrue(receipt["all_consistency_checks_passed"])
        self.assertTrue(receipt["owned_facts_inactive_at_end"])
        self.assertEqual(receipt["cleanup_errors"], [])
        self.assertEqual(sum(path == "/api/compare" for path, _ in client.calls), 1)
        self.assertEqual([case["revision"] for case in receipt["cases"]], [9, 9, 10, 11])
        self.assertEqual([case["question"]["expected_answer"] for case in receipt["cases"]],
                         ["Mara Voss", "Mara Voss", "Orin Dane", "Mara Voss"])
        self.assertTrue(all(case["response"]["evidence"]["raw_model_output"].startswith("UNIT TEST ONLY")
                            for case in receipt["cases"]))
        self.assertFalse(any(row["status"] == "active" for row in client.app.store.list_corrections()))

    def test_comparison_transport_failure_retains_two_errors_without_recovery_generation(self):
        client = FixtureClient(fail_on=3)
        receipt = self.execute(client)
        self.assertEqual(client.backend.calls, 4, "Initial + two comparison attempts + one restored request, no retry")
        self.assertEqual(receipt["outcome_counts"]["execution_error"], 2)
        self.assertEqual(receipt["outcome_counts"]["wrong"], 2)
        self.assertEqual(sum(receipt["outcome_counts"].values()), 4)
        self.assertEqual(sum(path == "/api/compare" for path, _ in client.calls), 1)
        self.assertTrue(receipt["owned_facts_inactive_at_end"])
        self.assertEqual(receipt["cleanup_errors"], [])

    def test_abstentions_remain_in_all_four_assigned_denominators(self):
        client = FixtureClient()
        client.backend.content = "UNKNOWN"
        receipt = self.execute(client)
        self.assertEqual(client.backend.calls, 4)
        self.assertEqual(receipt["outcome_counts"]["abstain"], 4)
        self.assertEqual(receipt["outcome_counts"]["correct"], 0)
        self.assertEqual(receipt["neural_accuracy_denominator"], 4)
        self.assertTrue(receipt["owned_facts_inactive_at_end"])

    def test_first_generation_failure_keeps_unexecutable_comparison_in_denominator(self):
        client = FixtureClient(fail_on=1)
        receipt = self.execute(client)
        self.assertEqual(client.backend.calls, 2)
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(receipt["outcome_counts"]["execution_error"], 1)
        self.assertEqual(receipt["outcome_counts"]["unexecuted"], 2)
        self.assertEqual(receipt["outcome_counts"]["wrong"], 1)
        self.assertFalse(any(path == "/api/compare" for path, _ in client.calls))
        self.assertTrue(receipt["owned_facts_inactive_at_end"])

    def test_rollback_operation_failure_cleans_only_created_fact_ids(self):
        client = FixtureClient(fail_rollback=True)
        receipt = self.execute(client)
        self.assertEqual(client.backend.calls, 3)
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(receipt["outcome_counts"]["wrong"], 3)
        self.assertEqual(receipt["outcome_counts"]["unexecuted"], 1)
        self.assertTrue(receipt["owned_facts_inactive_at_end"])
        withdrawn_ids = {path.split("/")[-2] for path, _ in client.calls if path.endswith("/withdraw")}
        self.assertEqual(withdrawn_ids, set(receipt["owned_fact_ids"]))

    def test_existing_user_history_or_conversations_are_preserved_without_generation(self):
        for existing in ("history", "chat"):
            with self.subTest(existing=existing):
                client = FixtureClient(existing=existing)
                before = client.app.store.export_snapshot()["tables"]
                receipt = self.execute(client)
                self.assertEqual(client.backend.calls, 0)
                self.assertEqual(receipt["status"], "interrupted")
                self.assertEqual(receipt["outcome_counts"]["unexecuted"], 4)
                self.assertEqual(receipt["owned_fact_ids"], [])
                self.assertEqual(client.app.store.export_snapshot()["tables"], before)
                self.assertFalse(any(payload is not None for _, payload in client.calls))

    def test_wrong_context_policy_refused_before_approval(self):
        client = FixtureClient(context_policy="all")
        receipt = self.execute(client)
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(client.backend.calls, 0)
        self.assertFalse(receipt["consistency_checks"]["retrieved_policy"])
        self.assertEqual(client.app.store.get_revision(), 0)


if __name__ == "__main__":
    unittest.main()
