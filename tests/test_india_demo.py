"""India display workflow contracts; no model calls or answer-quality claims."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import urlopen

from editguard_chat.examples import catalogue
from editguard_chat.server import ChatApplication, handler_for
from editguard_chat.store import StoreError
from tests.test_chat_server import FakeBackend, FakeRetriever

ROOT = Path(__file__).resolve().parents[1]


class IndiaDemoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.backend = FakeBackend()
        self.app = ChatApplication(self.temporary.name, backend=self.backend, retriever=FakeRetriever())

    def tearDown(self):
        self.app.store.close()
        self.temporary.cleanup()

    def test_display_graph_has_three_distinct_hops_without_changing_legacy_catalogue(self):
        example = catalogue(ROOT, locale="india")
        self.assertTrue(example["display_only"])
        self.assertEqual(example["world"]["split"], "display_only")
        facts = {row["id"]: row for row in example["world"]["current_facts"]}
        self.assertEqual(len(facts), 9)
        owner, contact, office = [facts["india-demo-001-f" + str(number)] for number in (1, 3, 9)]
        self.assertIn(owner["answer"], contact["question"])
        self.assertIn(contact["answer"], office["question"])
        self.assertEqual(office["answer"], "Hyderabad")
        self.assertEqual(example["world"]["ground_truth"]["old_contact_office"], "Chennai")
        self.assertTrue(all("fictional" in row["source"].lower() for row in facts.values()))
        self.assertEqual(catalogue(ROOT)["world"]["entity"], "Dahlia Relay")
        self.assertEqual(catalogue(ROOT)["dataset_id"], "editguard-operations-v1")

    def test_explicit_approval_replacement_history_and_rollback_preserve_other_eight_facts(self):
        chat = self.app.mutate("/api/chats", {})["chat"]
        self.assertEqual(self.app.store.get_revision(), 0)
        result = self.app.mutate("/api/examples/india/approve", {})
        self.assertEqual(result["revision"], 9)
        self.assertEqual(len(self.app.store.list_corrections()), 9)
        self.assertEqual(self.app.store.list_chats()[0]["id"], chat["id"])
        before = {row["id"]: row for row in self.app.store.list_corrections()}
        suggestion = result["suggested_edit"]
        self.assertEqual(suggestion["answer"], "Pune Reliability Team")
        changed = self.app.mutate("/api/corrections", suggestion)
        self.assertEqual(changed["revision"], 10)
        historical_owner = next(row for row in self.app.store.memory_at_revision(9)
                                if row["identity"]["entity_id"] == "india-demo-001-service"
                                and row["identity"]["relation_id"] == "owner"
                                and row["identity"]["scope_id"] == "release-2")
        self.assertEqual(historical_owner["answer"], "Bengaluru Platform Team")
        rolled_back = self.app.mutate("/api/corrections/" + changed["correction"]["id"] + "/rollback", {})
        self.assertEqual(rolled_back["revision"], 11)
        active = [row for row in self.app.store.list_corrections() if row["status"] == "active"]
        self.assertEqual({row["id"] for row in active}, set(before))
        self.assertTrue(any(row["answer"] == "Bengaluru Platform Team" for row in active))
        for row in active:
            if row["id"] != suggestion["supersedes"]:
                self.assertEqual(row, before[row["id"]])
        self.assertEqual(self.backend.calls, [])

    def test_legacy_or_custom_history_cannot_be_silently_replaced(self):
        self.app.mutate("/api/examples/operations/approve", {})
        snapshot = self.app.store.export_snapshot()
        with self.assertRaises(StoreError) as caught:
            self.app.mutate("/api/examples/india/approve", {})
        self.assertEqual(caught.exception.status, 409)
        after = self.app.store.export_snapshot()
        self.assertEqual(after["revision"], snapshot["revision"])
        self.assertEqual(after["tables"], snapshot["tables"])
        self.assertEqual(self.backend.calls, [])

    def test_custom_customer_knowledge_cannot_be_replaced_by_display_example(self):
        self.app.mutate("/api/corrections", {"question": "Owner of my real service?", "answer": "Customer team",
                        "scope": "Customer workspace", "source": "Administrator-approved local fact"})
        snapshot = self.app.store.export_snapshot()
        with self.assertRaises(StoreError) as caught:
            self.app.mutate("/api/examples/india/approve", {})
        self.assertEqual(caught.exception.status, 409)
        after = self.app.store.export_snapshot()
        self.assertEqual(after["revision"], snapshot["revision"])
        self.assertEqual(after["tables"], snapshot["tables"])
        self.assertEqual(self.backend.calls, [])

    def test_both_read_only_catalogue_routes_are_available(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.app))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            base = "http://127.0.0.1:" + str(server.server_address[1])
            for route, entity in (("india", "Kaveri Payment Gateway"), ("operations", "Dahlia Relay")):
                with urlopen(base + "/api/examples/" + route, timeout=5) as response:
                    self.assertEqual(json.load(response)["world"]["entity"], entity)
            self.assertEqual(self.app.store.get_revision(), 0)
        finally:
            server.shutdown(); server.server_close(); worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
