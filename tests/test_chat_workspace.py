"""Real SQLite identity/backup/restore contracts; no model-quality assertions."""
import copy
from pathlib import Path
import tempfile
import unittest

from editguard_chat.store import ChatStore, StoreError
from editguard_chat.workspace import seal


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.source = ChatStore(Path(self.temp.name) / "source.sqlite3")
        self.target = ChatStore(Path(self.temp.name) / "target.sqlite3")

    def tearDown(self):
        self.source.close()
        self.target.close()
        self.temp.cleanup()

    def publish(self, **changes):
        args = dict(question="Who owns the Atlas service?", answer="Amber team", source="Fictional handbook",
                    scope="Atlas, release 2", identity={"entity_id": "atlas", "relation_id": "owner",
                    "scope_id": "release-2", "aliases": ["Atlas service"]})
        args.update(changes)
        return self.source.publish_correction(**args)

    def test_canonical_identity_unifies_different_wording_and_preserves_versions(self):
        first = self.publish()
        with self.assertRaises(StoreError) as caught:
            self.publish(question="Which team maintains Atlas?", answer="Indigo team")
        self.assertEqual(caught.exception.status, 409)
        second = self.publish(question="Which team maintains Atlas?", answer="Indigo team", supersedes=first["id"])
        self.assertEqual(second["version"], 2)
        self.assertEqual(second["lineage_id"], first["id"])
        self.assertEqual(second["identity"]["entity_id"], "atlas")
        self.assertEqual(self.source.get_revision(), 2)

    def test_canonical_relation_and_version_boundaries(self):
        self.publish()
        self.publish(identity={"entity_id": "atlas", "relation_id": "owner", "scope_id": "release-1"},
                     answer="Historic team")
        self.publish(identity={"entity_id": "atlas", "relation_id": "region", "scope_id": "release-2"},
                     answer="West")
        self.assertEqual(len(self.source.list_corrections()), 3)

    def populated(self):
        first = self.publish()
        second = self.publish(answer="Indigo team", supersedes=first["id"])
        chat = self.source.create_chat()
        self.source.add_message(chat["id"], "user", "Who owns Atlas?")
        self.source.add_message(chat["id"], "assistant", "Indigo team", evidence={"kind": "model", "fact_id": second["id"]})
        self.source.withdraw(second["id"])
        self.source.restore(first["id"])
        return self.source.export_snapshot()

    def test_roundtrip_preserves_state_messages_and_audit_after_reopen(self):
        snapshot = self.populated()
        result = self.target.import_snapshot(snapshot)
        self.assertEqual(result["revision"], 4)
        self.assertEqual(self.source.list_corrections(), self.target.list_corrections())
        self.assertEqual(self.source.list_events(), self.target.list_events())
        self.assertEqual(self.source.list_chats(), self.target.list_chats())
        for chat in self.source.list_chats():
            self.assertEqual(self.source.get_chat(chat["id"]), self.target.get_chat(chat["id"]))
        reopened = ChatStore(Path(self.temp.name) / "target.sqlite3")
        try:
            self.assertEqual(reopened.get_revision(), 4)
            self.assertEqual(len(reopened.list_events()), 5)
        finally:
            reopened.close()

    def test_restore_never_overwrites_nonempty_workspace(self):
        snapshot = self.populated()
        self.target.create_chat()
        before = self.target.export_snapshot()
        with self.assertRaises(StoreError) as caught:
            self.target.import_snapshot(snapshot)
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(before["tables"], self.target.export_snapshot()["tables"])
        self.assertEqual(self.target.get_revision(), 0)

    def test_corruption_and_broken_references_cannot_partially_import(self):
        original = self.populated()
        corruptions = [lambda s: s.update(sha256="bad"),
                       lambda s: s["tables"]["messages"][0].update(chat_id="a" * 32),
                       lambda s: s["tables"]["corrections"][0].update(question_key="forged"),
                       lambda s: s["tables"]["events"].clear(),
                       lambda s: s["tables"]["messages"][0].update(image_path="C:/private/file"),
                       lambda s: s["tables"]["messages"][0].update(evidence_json='{"x":NaN}'),
                       lambda s: s["tables"]["chats"][0].update(title=None)]
        for index, mutate in enumerate(corruptions):
            changed = copy.deepcopy(original)
            mutate(changed)
            if index:
                seal(changed)
            with self.subTest(index=index), self.assertRaises(StoreError):
                self.target.import_snapshot(changed)
            self.assertEqual(self.target.list_chats(), [])
            self.assertEqual(self.target.list_corrections(), [])
            self.assertEqual(self.target.get_revision(), 0)

    def test_partial_identity_is_rejected_without_mutation(self):
        for identity in ({"entity_id": "atlas"}, {"entity_id": "a", "relation_id": "r", "scope_id": "bad scope"},
                         {"entity_id": "a", "relation_id": "r", "scope_id": "v2", "aliases": "wrong"}):
            with self.assertRaises(StoreError):
                self.publish(identity=identity)
        self.assertEqual(self.source.get_revision(), 0)

    def test_historical_memory_replays_each_state_without_mutating_current(self):
        first = self.publish()
        second = self.publish(answer="Indigo team", supersedes=first["id"])
        self.source.withdraw(second["id"])
        self.source.restore(first["id"])
        expected = {0: [], 1: [first["id"]], 2: [second["id"]], 3: [], 4: [first["id"]]}
        for revision, ids in expected.items():
            rows = self.source.memory_at_revision(revision)
            self.assertEqual([row["id"] for row in rows], ids)
            self.assertTrue(all(row["status"] == "active" and row["snapshot_revision"] == revision for row in rows))
        self.assertEqual(self.source.get_revision(), 4)
        self.assertEqual(len(self.source.list_events()), 5)
        for revision in (True, -1, 1.5, "1", 5):
            with self.assertRaises(StoreError):
                self.source.memory_at_revision(revision)

    def test_import_rejects_a_rehashed_fact_state_that_disagrees_with_history(self):
        snapshot = self.populated()
        active = next(row for row in snapshot["tables"]["corrections"] if row["status"] == "active")
        active["status"] = "withdrawn"
        seal(snapshot)
        with self.assertRaises(StoreError):
            self.target.import_snapshot(snapshot)
        self.assertEqual(self.target.get_revision(), 0)


if __name__ == "__main__":
    unittest.main()
