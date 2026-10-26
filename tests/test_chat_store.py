"""Database lifecycle tests, independent of generated-answer quality."""
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from editguard_chat.store import ChatStore, StoreError


class ChatStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "chat.sqlite3"
        self.store = ChatStore(self.path)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def publish(self, **overrides):
        values = dict(question="How many devices does Orion Basic support?",
                      answer="Three devices.", source="Controlled support handbook, revision 2",
                      scope="Orion Basic / current")
        values.update(overrides)
        return self.store.publish_correction(**values)

    def test_history_and_messages_persist_without_publishing_chat_facts(self):
        chat = self.store.create_chat()
        user = self.store.add_message(chat["id"], "user", "  Please remember that my plan allows seven devices.  ")
        assistant = self.store.add_message(chat["id"], "assistant", "I cannot approve product corrections.",
                                           evidence=[{"source": "example", "id": "x"}], image_path=None)
        self.assertEqual(self.store.get_revision(), 0)
        self.assertEqual(self.store.list_corrections(), [])
        reopened = ChatStore(self.path)
        try:
            loaded = reopened.get_chat(chat["id"])
            self.assertTrue(loaded["title"].startswith("Please remember"))
            self.assertEqual([m["id"] for m in loaded["messages"]], [user["id"], assistant["id"]])
            self.assertEqual(loaded["messages"][1]["evidence"], [{"source": "example", "id": "x"}])
            self.assertEqual(loaded["messages"][0]["knowledge_revision"], 0)
        finally:
            reopened.close()

    def test_conversations_are_isolated_and_title_comes_from_first_user(self):
        a, b = self.store.create_chat(), self.store.create_chat()
        self.store.add_message(a["id"], "assistant", "Hello")
        self.store.add_message(a["id"], "user", "My first question")
        self.store.add_message(a["id"], "user", "A second question")
        self.assertEqual(self.store.get_chat(a["id"])["title"], "My first question")
        self.assertEqual(len(self.store.list_messages(a["id"])), 3)
        self.assertEqual(self.store.list_messages(b["id"]), [])
        with self.assertRaises(StoreError) as caught:
            self.store.add_message("missing", "user", "Hello")
        self.assertEqual(caught.exception.status, 404)

    def test_generated_turn_stores_both_messages_with_one_revision_and_title(self):
        chat = self.store.create_chat()
        self.publish()
        user, assistant = self.store.add_turn(chat["id"], "  A new question  ", "Generated answer",
                                              evidence={"generation_kind": "model"})
        self.assertEqual([row["role"] for row in self.store.list_messages(chat["id"])], ["user", "assistant"])
        self.assertEqual((user["position"], assistant["position"]), (1, 2))
        self.assertEqual((user["knowledge_revision"], assistant["knowledge_revision"]), (1, 1))
        self.assertEqual(self.store.get_chat(chat["id"])["title"], "A new question")
        self.assertEqual(assistant["evidence"], {"generation_kind": "model"})

    def test_late_second_turn_write_failure_rolls_back_both_messages_and_chat_metadata(self):
        chat = self.store.create_chat()
        before = self.store.export_snapshot()
        append = self.store._append_message
        def fail_after_assistant_write(db, chat_id, prepared):
            result = append(db, chat_id, prepared)
            if prepared["role"] == "assistant":
                raise RuntimeError("Injected failure after second message and metadata writes")
            return result
        with patch.object(self.store, "_append_message", side_effect=fail_after_assistant_write):
            with self.assertRaisesRegex(RuntimeError, "after second message"):
                self.store.add_turn(chat["id"], "Question that would change the title", "Generated response")
        self.assertEqual(self.store.export_snapshot()["tables"], before["tables"])
        self.assertEqual(self.store.get_revision(), before["revision"])
        user, assistant = self.store.add_turn(chat["id"], "Retry after storage recovery", "Fresh response")
        self.assertEqual((user["position"], assistant["position"]), (1, 2))

    def test_invalid_assistant_evidence_never_stores_first_half_of_turn(self):
        chat = self.store.create_chat()
        before = self.store.get_chat(chat["id"])
        with self.assertRaises(StoreError):
            self.store.add_turn(chat["id"], "Question", "Answer", evidence={"invalid": float("nan")})
        self.assertEqual(self.store.get_chat(chat["id"]), before)

    def test_explicit_replacement_preserves_lineage_and_audit(self):
        old = self.publish()
        with self.assertRaises(StoreError) as caught:
            self.publish(question="  HOW many   devices does ORION Basic support?  ",
                         scope="  orion basic / CURRENT ", answer="Four devices.")
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(caught.exception.conflicts[0]["id"], old["id"])
        self.assertEqual(self.store.get_revision(), 1)
        current = self.publish(answer="Four devices.", supersedes=old["id"])
        self.assertEqual((current["version"], current["lineage_id"], current["supersedes"]),
                         (2, old["lineage_id"], old["id"]))
        by_id = {row["id"]: row for row in self.store.list_corrections()}
        self.assertEqual(by_id[old["id"]]["status"], "superseded")
        self.assertEqual(by_id[old["id"]]["superseded_by"], current["id"])
        self.assertEqual(by_id[current["id"]]["status"], "active")
        self.assertEqual(self.store.get_revision(), 2)
        self.assertEqual([event["action"] for event in self.store.list_events()],
                         ["published", "superseded", "published"])

    def test_withdraw_restore_and_cache_revision_are_durable(self):
        old = self.publish()
        current = self.publish(answer="Four devices.", supersedes=old["id"])
        with self.assertRaises(StoreError) as caught:
            self.store.restore(old["id"])
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(self.store.get_revision(), 2)
        self.assertEqual(self.store.withdraw(current["id"])["status"], "withdrawn")
        self.assertEqual(self.store.get_revision(), 3)
        self.store.withdraw(current["id"])
        self.assertEqual(self.store.get_revision(), 3, "Idempotent withdrawal is not another mutation")
        restored = self.store.restore(old["id"])
        self.assertEqual(restored["status"], "active")
        self.assertEqual(restored["answer"], "Three devices.")
        self.assertIsNone(restored["superseded_by"])
        self.assertEqual(self.store.get_revision(), 4)
        self.store.restore(old["id"])
        self.assertEqual(self.store.get_revision(), 4)
        reopened = ChatStore(self.path)
        try:
            self.assertEqual(reopened.get_revision(), 4)
            self.assertEqual([r["id"] for r in reopened.list_corrections() if r["status"] == "active"], [old["id"]])
            self.assertEqual(reopened.list_events(old["id"])[-1]["action"], "restored")
        finally:
            reopened.close()

    def test_atomic_rollback_restores_previous_fact_without_an_empty_revision(self):
        old = self.publish()
        current = self.publish(answer="Four devices.", supersedes=old["id"])
        restored = self.store.rollback(current["id"])
        self.assertEqual(restored["id"], old["id"])
        self.assertEqual(self.store.get_revision(), 3)
        self.assertEqual([row["id"] for row in self.store.memory_at_revision(2)], [current["id"]])
        self.assertEqual([row["id"] for row in self.store.memory_at_revision(3)], [old["id"]])
        events = self.store.list_events()[-2:]
        self.assertEqual([(row["action"], row["revision"]) for row in events], [("withdrawn", 3), ("restored", 3)])
        self.store.rollback(current["id"])
        self.assertEqual(self.store.get_revision(), 3, "Repeated rollback must be idempotent")

    def test_failed_atomic_rollback_preserves_both_states_revision_and_events(self):
        old = self.publish()
        current = self.publish(answer="Four devices.", supersedes=old["id"])
        before = self.store.export_snapshot()
        original_event = self.store._event
        def fail_restore_event(db, correction_id, action, *rest):
            if action == "restored":
                raise RuntimeError("Injected second audit write failure")
            return original_event(db, correction_id, action, *rest)
        with patch.object(self.store, "_event", side_effect=fail_restore_event):
            with self.assertRaises(RuntimeError):
                self.store.rollback(current["id"])
        self.assertEqual(self.store.export_snapshot()["tables"], before["tables"])
        self.assertEqual(self.store.get_revision(), before["revision"])
        latest = self.publish(answer="Five devices.", supersedes=current["id"])
        with self.assertRaises(StoreError):
            self.store.rollback(current["id"])
        with self.assertRaises(StoreError):
            self.store.rollback(old["id"])
        self.assertEqual([row["id"] for row in self.store.list_corrections() if row["status"] == "active"], [latest["id"]])

    def test_scope_and_question_boundaries_and_wrong_replacement(self):
        old = self.publish()
        pro = self.publish(scope="Orion Pro / current", answer="Ten devices.")
        price = self.publish(question="What is Orion Basic's price?", answer="$10.")
        self.assertEqual(len([r for r in self.store.list_corrections() if r["status"] == "active"]), 3)
        for replacement in ("nonexistent", pro["id"], price["id"]):
            with self.assertRaises(StoreError):
                self.publish(answer="Wrong replacement", supersedes=replacement)
        self.assertEqual(self.store.get_revision(), 3)
        self.assertEqual(len(self.store.list_events()), 3)
        self.store.withdraw(old["id"])
        with self.assertRaises(StoreError):
            self.publish(answer="No current version", supersedes=old["id"])

    def test_required_fields_and_finite_evidence(self):
        for field in ("question", "answer", "source", "scope"):
            with self.assertRaises(StoreError):
                self.publish(**{field: "  "})
        chat = self.store.create_chat()
        with self.assertRaises(StoreError):
            self.store.add_message(chat["id"], "user", "No", evidence={"bad": float("nan")})
        with self.assertRaises(StoreError):
            self.store.add_message(chat["id"], "user", "No", knowledge_revision=True)
        empty = self.store.add_message(chat["id"], "assistant", "")
        self.assertEqual(empty["content"], "", "An empty model answer remains an auditable record")
        self.assertEqual(self.store.get_revision(), 0)

    def test_concurrent_publish_cannot_silently_replace(self):
        def attempt(index):
            try:
                return self.publish(answer=f"Value {index}")["id"]
            except StoreError as exc:
                return exc.status
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(attempt, range(8)))
        self.assertEqual(sum(isinstance(outcome, str) for outcome in outcomes), 1)
        self.assertEqual(outcomes.count(409), 7)
        self.assertEqual(self.store.get_revision(), 1)
        self.assertEqual(len(self.store.list_events()), 1)

    def test_two_store_instances_serialize_competing_publish_transactions(self):
        other = ChatStore(self.path)
        try:
            def attempt(index):
                store = self.store if index % 2 == 0 else other
                try:
                    return store.publish_correction("Same question", f"Value {index}",
                                                    "Reviewed source", "Same scope")["id"]
                except StoreError as exc:
                    return exc.status
            with ThreadPoolExecutor(max_workers=8) as pool:
                outcomes = list(pool.map(attempt, range(12)))
            self.assertEqual(sum(isinstance(outcome, str) for outcome in outcomes), 1)
            self.assertEqual(outcomes.count(409), 11)
            self.assertEqual(other.get_revision(), 1)
            active = [row for row in self.store.list_corrections() if row["status"] == "active"]
            self.assertEqual(len(active), 1)
            self.assertEqual(len(other.list_events()), 1)
        finally:
            other.close()

    def test_withdrawn_lineage_cannot_restore_over_a_later_active_version(self):
        old = self.publish()
        self.store.withdraw(old["id"])
        newer = self.publish(answer="Four devices.")
        self.assertEqual(newer["lineage_id"], old["lineage_id"])
        self.assertEqual(newer["version"], 2)
        self.assertEqual(self.store.get_revision(), 3)
        with self.assertRaises(StoreError) as caught:
            self.store.restore(old["id"])
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(caught.exception.conflicts[0]["id"], newer["id"])
        self.assertEqual(self.store.get_revision(), 3)
        self.assertEqual(len(self.store.list_events()), 3)
        self.store.withdraw(newer["id"])
        self.store.restore(old["id"])
        with self.assertRaises(StoreError):
            self.store.restore(newer["id"])
        active = [row for row in self.store.list_corrections() if row["status"] == "active"]
        self.assertEqual([row["id"] for row in active], [old["id"]])

    def test_failed_audit_write_rolls_back_replacement_revision_and_both_states(self):
        old = self.publish()
        with patch.object(self.store, "_event", side_effect=OSError("Simulated audit write failure")):
            with self.assertRaises(OSError):
                self.publish(answer="Four devices.", supersedes=old["id"])
        reopened = ChatStore(self.path)
        try:
            rows = reopened.list_corrections()
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]["id"], rows[0]["status"], rows[0]["superseded_by"]),
                             (old["id"], "active", None))
            self.assertEqual(reopened.get_revision(), 1)
            self.assertEqual(len(reopened.list_events()), 1)
        finally:
            reopened.close()


if __name__ == "__main__":
    unittest.main()
