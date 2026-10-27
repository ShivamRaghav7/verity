"""Optional iterative serving contracts; fake outputs are not neural evidence."""
import tempfile
import unittest
from pathlib import Path

from editguard_benchmark.retrieval import retrieve
from editguard_chat.backend import ModelError
from editguard_chat.reasonke_backend import ApprovedFactContext, IterativeFactContext, RetrievedFactContext
from editguard_chat.server import ChatApplication
from editguard_depth.retrieval import DEFAULT_K, EXPANSION_ROUNDS, SEED_COUNT, retrieve_iterative
from tests.test_chat_reasonke import FakeFactBackend
from tests.test_chat_server import FakeBackend


class IterativeServingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.backend = FakeFactBackend()
        self.app = ChatApplication(self.temporary.name, backend=self.backend, context_policy="iterative")
        self.example = self.app.mutate("/api/examples/india/approve", {})
        self.chat = self.app.store.create_chat()["id"]

    def tearDown(self):
        self.app.store.close()
        self.temporary.cleanup()

    def test_iterative_policy_supplies_selected_approved_facts_and_keeps_model_output(self):
        question = ("In which Indian city is the office of the on-call contact for the team "
                    "that owns Kaveri Payment Gateway in release 2?")
        response = self.app.ask({"chat_id": self.chat, "message": question})["assistant"]
        _, facts, _ = self.backend.calls[-1]
        active = self.app.store.list_corrections()
        expected = retrieve_iterative(question, active, k=DEFAULT_K)
        self.assertEqual([row["id"] for row in facts], [row["id"] for row in expected])
        self.assertLessEqual(len(facts), DEFAULT_K)
        # The database's target must never substitute for actual backend output.
        self.assertEqual(response["content"], "FAKE_CONTEXT_PRESENT")
        self.assertNotEqual(response["content"], "Chennai")
        evidence = response["evidence"]
        self.assertEqual(evidence["context_policy"], "iterative")
        self.assertEqual(evidence["selection_k"], DEFAULT_K)
        self.assertEqual(evidence["retrieval_seed_count"], SEED_COUNT)
        self.assertEqual(evidence["retrieval_expansion_rounds"], EXPANSION_ROUNDS)
        self.assertEqual(evidence["generation_kind"], "model")
        self.assertFalse(evidence["score_is_probability"])
        self.assertFalse(evidence["correction_application_verified"])
        self.assertIn("current-revision", evidence["history_policy"])
        status = self.app.status()
        self.assertEqual(status["retrieval"]["context_policy"], "iterative")
        self.assertEqual(status["retrieval"]["method"], IterativeFactContext.method)

    def test_original_mode_supplies_no_memory_despite_configured_iterative_policy(self):
        response = self.app.ask({"chat_id": self.chat, "message": "Who owns Kaveri?", "mode": "original"})["assistant"]
        self.assertEqual(self.backend.calls[-1][1], [])
        self.assertEqual(response["content"], "FAKE_NO_CONTEXT")
        self.assertEqual(response["evidence"]["context_policy"], "without_correction_context")
        self.assertEqual(response["evidence"]["configured_context_policy"], "iterative")

    def test_three_hop_display_support_follows_edits_and_rollback_with_real_store_ids(self):
        question = ("In which Indian city is the office of the on-call contact for the team "
                    "that owns Kaveri Payment Gateway in release 2?")
        replacement = None
        for state, team, contact, city in (("initial", "Bengaluru Platform Team", "Meera Iyer", "Chennai"),
                                          ("updated", "Pune Reliability Team", "Arjun Deshmukh", "Hyderabad"),
                                          ("rolled_back", "Bengaluru Platform Team", "Meera Iyer", "Chennai")):
            if state == "updated":
                replacement = self.app.mutate("/api/corrections", self.example["suggested_edit"])["correction"]
            elif state == "rolled_back":
                self.app.mutate("/api/corrections/" + replacement["id"] + "/rollback", {})
            selected = self.app.retriever.rank(question, self.app.store.list_corrections())["selected"]
            pairs = {(row["question"], row["answer"]) for row in selected}
            with self.subTest(state=state):
                self.assertIn(("Which team owns Kaveri Payment Gateway in release 2?", team), pairs)
                self.assertIn(("Who is the on-call contact for " + team + "?", contact), pairs)
                self.assertIn(("In which Indian city is " + contact + "'s office?", city), pairs)
                self.assertLessEqual(len(selected), DEFAULT_K)
                self.assertTrue(all(len(row["id"]) == 32 for row in selected))
        self.assertEqual(self.backend.calls, [], "Retrieval development checks generate no model answers")

    def test_iterative_selection_never_bypasses_memory_conflict_or_size_limits(self):
        rows = self.app.store.list_corrections()
        first = rows[0]
        conflict = {**first, "id": "different-id", "answer": "Conflicting value"}
        context = IterativeFactContext()
        result = context.rank("An unrelated question", [first, conflict])
        self.assertTrue(result["clarification_required"])
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["context_policy"], "iterative")
        too_many = [{**first, "id": str(i), "question": "Unique question " + str(i)} for i in range(201)]
        with self.assertRaisesRegex(ModelError, "at most 200"):
            context.rank("Any question", too_many)

    def test_legacy_full_memory_and_two_stage_selection_are_unchanged(self):
        rows = self.app.store.list_corrections()
        question = "Which Indian city hosts Kaveri Payment Gateway in release 2?"
        legacy = RetrievedFactContext().rank(question, rows)
        self.assertEqual([row["id"] for row in legacy["selected"]],
                         [row["id"] for row in retrieve(question, rows, k=4)])
        self.assertEqual(legacy["method"], "two_stage_lexical_approved_memory")
        self.assertEqual(len(legacy["selected"]), 4)
        self.assertNotIn("context_policy", legacy)
        self.assertEqual(len(ApprovedFactContext().rank(question, rows)["selected"]), 9)
        with tempfile.TemporaryDirectory() as directory:
            default_app = ChatApplication(directory, backend=FakeFactBackend())
            try:
                self.assertIsInstance(default_app.retriever, ApprovedFactContext)
                self.assertNotIsInstance(default_app.retriever, IterativeFactContext)
            finally:
                default_app.store.close()

    def test_iterative_option_rejects_a_backend_without_fact_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "fact-conditioned"):
                ChatApplication(directory, backend=FakeBackend(), context_policy="iterative")
            self.assertFalse((Path(directory) / "chat.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
