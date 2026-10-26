"""Operational workload safety/accounting fixtures, never neural quality evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from editguard_chat.store import ChatStore
from scripts.verify_operations_http import run, MODEL_ID, MODEL_REVISION


class FixtureClient:
    def __init__(self, existing=False):
        self.token = None
        self.store = ChatStore(":memory:")
        self.calls = []
        self.generated = 0
        if existing:
            self.store.publish_correction("Existing user fact?", "Preserve it", "Fixture", "User scope")

    def request(self, path, payload=None):
        self.calls.append((path, payload))
        if path == "/api/status":
            return {"token": "private-fixture-token-not-a-real-credential",
                    "model": {"ready": True, "name": MODEL_ID, "revision": MODEL_REVISION}}
        if path == "/api/corrections" and payload is None:
            return {"corrections": self.store.list_corrections()}
        if path == "/api/corrections":
            return {"correction": self.store.publish_correction(**payload)}
        if path == "/api/chats":
            return {"chat": self.store.create_chat()}
        if path.endswith("/withdraw"):
            return {"correction": self.store.withdraw(path.split("/")[-2])}
        if path.endswith("/restore"):
            return {"correction": self.store.restore(path.split("/")[-2])}
        if path == "/api/chat":
            self.generated += 1
            # Deliberately wrong: the runner must retain all wrong outcomes and
            # perform no answer-based retry. Not an answer-model simulation.
            return {"revision": self.store.get_revision(), "assistant": {"content": "Deliberately wrong fixture",
                    "evidence": {"generation_kind": "model", "generation": {"model": MODEL_ID, "revision": MODEL_REVISION}}}}
        raise AssertionError(path)


class OperationsHTTPTests(unittest.TestCase):
    def test_twelve_assigned_wrong_outcomes_are_never_retried_and_own_facts_cleaned(self):
        client = FixtureClient()
        try:
            with tempfile.TemporaryDirectory() as temporary, patch("scripts.verify_operations_http.Client", return_value=client):
                output = Path(temporary) / "fixture-receipt.json"
                receipt = run("http://fixture.invalid", output)
                self.assertEqual(receipt["status"], "completed")
                self.assertEqual(client.generated, 12)
                self.assertEqual(receipt["outcome_counts"]["wrong"], 12)
                self.assertEqual(receipt["neural_accuracy_denominator"], 12)
                self.assertEqual(receipt["cleanup_errors"], [])
                self.assertFalse(any(row["status"] == "active" for row in client.store.list_corrections()))
                self.assertEqual(len(receipt["owned_fact_ids"]), 30)
                self.assertNotIn("private-fixture-token-not-a-real-credential", output.read_text())
                self.assertEqual(json.loads(output.read_text())["outcome_counts"], receipt["outcome_counts"])
                with self.assertRaisesRegex(ValueError, "never overwritten"):
                    run("http://fixture.invalid", output)
                self.assertEqual(client.generated, 12)
        finally:
            client.store.close()

    def test_existing_active_user_memory_is_never_modified(self):
        client = FixtureClient(existing=True)
        before = client.store.export_snapshot()
        try:
            with tempfile.TemporaryDirectory() as temporary, patch("scripts.verify_operations_http.Client", return_value=client):
                receipt = run("http://fixture.invalid", Path(temporary) / "fixture-receipt.json")
            self.assertEqual(receipt["status"], "interrupted")
            self.assertEqual(receipt["outcome_counts"]["unexecuted"], 12)
            self.assertEqual(receipt["owned_fact_ids"], [])
            self.assertEqual(client.generated, 0)
            self.assertEqual(client.store.export_snapshot()["tables"], before["tables"])
        finally:
            client.store.close()


if __name__ == "__main__":
    unittest.main()
