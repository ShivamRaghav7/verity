"""Adversarial, independently specified answer-scoring and human-review fixtures."""

import json
import tempfile
import unittest
from pathlib import Path

from editguard.scoring import score_answer
from scripts.aggregate_evaluation import (aggregate_file, apply_adjudications,
                                          review_hash, review_queue, score_records)


FACT = {"entity_id": "nimbus", "entity_name": "Nimbus",
        "aliases": ["Nimbus Cloud"], "plan": "Basic", "version": "v2",
        "relation": "device_limit", "value": "3", "unit": "devices", "as_of": "2026-10-11"}


class EndorsementScoringTests(unittest.TestCase):
    def test_correct_assertion_and_decimal_word_normalization(self):
        for answer in ("Nimbus Basic v2 supports up to 3 devices.",
                       "Nimbus Cloud Basic version v2 supports three devices.",
                       "Nimbus Basic version 2 allows 3.0 devices.",
                       "  NIMBUS  Basic v2 supports 3 devices.  "):
            with self.subTest(answer=answer):
                score = score_answer(answer, FACT)
                self.assertIs(score["correct"], True)
                self.assertEqual(score["endorsed_fact"]["entity_id"], "nimbus")
                self.assertEqual(score["endorsed_fact"]["value"], "3")

    def test_old_value_and_substring_collision_fail(self):
        for answer in ("Nimbus Basic v2 supports 5 devices.",
                       "Nimbus Basic v2 supports 13 devices."):
            self.assertIs(score_answer(answer, FACT)["correct"], False)

    def test_alias_boundaries_and_wrong_scope_fail(self):
        for answer in ("Nimbus Mini Basic v2 supports 3 devices.",
                       "Other Basic v2 supports 3 devices.",
                       "Nimbus Pro v2 supports 3 devices.",
                       "Nimbus Basic v1 supports 3 devices."):
            with self.subTest(answer=answer):
                self.assertIs(score_answer(answer, FACT)["correct"], False)
        plus = {**FACT, "entity_id": "aster-plus", "entity_name": "Aster Plus", "aliases": ["A001+"]}
        self.assertIs(score_answer("A001+ Basic v2 supports 3 devices.", plus)["correct"], True)
        self.assertIs(score_answer("A001 Basic v2 supports 3 devices.", plus)["correct"], False)

    def test_missing_subject_relation_or_scope_requires_review(self):
        for answer in ("Three.", "3 devices", "Nimbus supports 3 devices.",
                       "Nimbus Basic supports 3 devices.", "Nimbus v2 supports 3 devices.",
                       "Nimbus Basic v2: 3"):
            with self.subTest(answer=answer):
                score = score_answer(answer, FACT)
                self.assertIsNone(score["correct"])
                self.assertTrue(score["needs_review"])

    def test_same_value_wrong_relation_does_not_pass(self):
        score = score_answer("Nimbus Basic v2 costs 3 USD per month.", FACT)
        self.assertIs(score["correct"], False)
        self.assertEqual(score["endorsed_fact"]["relation"], "price_monthly")

    def test_all_relation_target_forms_and_wrong_units(self):
        fixtures = [
            ({**FACT, "relation": "price_monthly", "value": "9", "unit": "USD/month"},
             "Nimbus Basic v2 costs 9 USD per month.", "Nimbus Basic v2 costs 9 EUR per month."),
            ({**FACT, "relation": "storage_gb", "value": "50", "unit": "GB"},
             "Nimbus Basic v2 includes 50 GB of storage.", "Nimbus Basic v2 includes 50 MB of storage."),
            ({**FACT, "relation": "support_hours", "value": "8", "unit": "hours/day"},
             "Nimbus Basic v2 includes 8 support hours per day.", "Nimbus Basic v2 includes 8 support hours per week."),
        ]
        for fact, correct, wrong in fixtures:
            with self.subTest(relation=fact["relation"]):
                self.assertIs(score_answer(correct, fact)["correct"], True)
                self.assertIs(score_answer(wrong, fact)["correct"], False)
        price = {**FACT, "relation": "price_monthly", "unit": "USD/month"}
        self.assertIsNone(score_answer("Nimbus Basic v2 costs 3 dollars per month.", price)["correct"])

    def test_explicit_polarity_must_match_comparison_truth(self):
        statement = "Nimbus Basic v2 supports 3 devices."
        self.assertIs(score_answer("No. " + statement, FACT, expected_polarity="no")["correct"], True)
        self.assertIs(score_answer("Yes. " + statement, FACT, expected_polarity="no")["correct"], False)
        self.assertIs(score_answer(statement, FACT, expected_polarity="no")["correct"], False)
        self.assertIsNone(score_answer("Yes. " + statement, FACT)["correct"])

    def test_rejected_target_example_and_conflicting_prose_never_receive_credit(self):
        outputs = [
            "Five was the old limit; three is wrong.",
            "Nimbus Basic v2 does not support 3 devices.",
            "For example, Nimbus Basic v2 supports 3 devices, but the actual limit is 5.",
            "Nimbus Basic v2 supports 3 devices. Nimbus Basic v2 supports 5 devices.",
            "Nimbus Basic v2 supports 3 devices, or perhaps 5.",
            "If upgraded, Nimbus Basic v2 supports 3 devices.",
        ]
        for answer in outputs:
            with self.subTest(answer=answer):
                self.assertIsNot(score_answer(answer, FACT)["correct"], True)
        self.assertIs(score_answer(outputs[0], FACT)["correct"], False)
        self.assertIs(score_answer(outputs[3], FACT)["correct"], False)

    def test_unfamiliar_correct_prose_remains_pending(self):
        answer = "The present allowance for Nimbus Basic v2 totals three devices."
        score = score_answer(answer, FACT)
        self.assertIsNone(score["correct"])
        self.assertTrue(score["needs_review"])
        self.assertIsNone(score["endorsed_fact"])

    def test_decimal_assertion_and_opposite_yes_no_endorsements(self):
        self.assertIs(score_answer("Nimbus Basic v2 supports 3.0 devices.", FACT)["correct"], True)
        answer = "No. Nimbus Basic v2 supports 3 devices. Yes. Nimbus Basic v2 supports 3 devices."
        score = score_answer(answer, FACT, expected_polarity="no")
        self.assertIs(score["correct"], False)
        self.assertIn("contradictory", score["reason"])
        answer = "Nimbus Basic v2 supports 3.0 devices; Nimbus Basic v2 supports 5.0 devices."
        self.assertIs(score_answer(answer, FACT)["correct"], False)

    def test_empty_answer_and_structured_evidence_label(self):
        self.assertIs(score_answer("", FACT)["correct"], False)
        score = score_answer("Nimbus Basic v2 supports 3 devices.", FACT,
                             answer_kind="deterministic_structured")
        self.assertIs(score["correct"], True)
        self.assertEqual(score["answer_kind"], "deterministic_structured")

    def test_ground_truth_entity_names_can_be_passed_without_catalogue_leak_to_model(self):
        fact = {key: value for key, value in FACT.items() if key not in ("entity_name", "aliases")}
        self.assertIs(score_answer("Nimbus Basic v2 supports 3 devices.", fact, aliases=("Nimbus",))["correct"], True)

    def test_invalid_polarity_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            score_answer("Nimbus Basic v2 supports 3 devices.", FACT, expected_polarity="maybe")


class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "q-1", "family_id": "f-1", "configuration": "trained_scope_editor",
                    "applicable": True, "question_type": "direct", "expected_fact": FACT,
                    "question": "How many devices are allowed on Nimbus Basic v2?",
                    "answer": "The present allowance for Nimbus Basic v2 totals three devices.",
                    "answer_kind": "model_generated", "status": "answered", "latency_ms": 10,
                    "retrieval_hits": {"1": True}}
        self.scored = score_records([self.row])

    def registry(self, verdicts=("correct", "correct")):
        digest = review_hash(self.scored[0])
        return {"schema_version": 1, "decisions": [
            {"review_id": f"review:{digest}", "row_sha256": digest,
             "reviewer": f"reviewer-{index}", "decision": verdict,
             "reason": "The output endorses the requested device allowance and scope.",
             "reviewed_at": "2026-10-11T13:00:00+05:30"}
            for index, verdict in enumerate(verdicts)]}

    def test_queue_blinds_configuration_and_routing(self):
        queue = review_queue(self.scored)
        self.assertEqual(len(queue), 1)
        self.assertNotIn("configuration", queue[0])
        self.assertNotIn("retrieval_hits", queue[0])
        self.assertEqual(queue[0]["answer"], self.row["answer"])

    def test_two_independent_agreeing_reviews_can_credit_exact_unchanged_output(self):
        rows = apply_adjudications(self.scored, self.registry())
        self.assertEqual(rows[0]["answer"], self.row["answer"])
        self.assertIs(rows[0]["score"]["correct"], True)
        self.assertTrue(rows[0]["score"]["human_adjudicated"])
        self.assertIsNone(rows[0]["automatic_score"]["correct"])
        self.assertFalse(rows[0]["score"]["needs_review"])

    def test_human_endorsement_uses_the_automatic_canonical_form(self):
        row = {**self.row, "expected_fact": {**FACT, "value": 3.0, "unit": "device"}}
        scored = score_records([row])
        self.scored = scored
        result = apply_adjudications(scored, self.registry())[0]
        self.assertEqual(result["score"]["endorsed_fact"]["value"], "3")
        self.assertEqual(result["score"]["endorsed_fact"]["unit"], "devices")

    def test_single_vote_or_disagreement_remains_uncredited(self):
        for verdicts in (("correct",), ("correct", "incorrect")):
            rows = apply_adjudications(self.scored, self.registry(verdicts))
            self.assertIsNone(rows[0]["score"]["correct"])
            self.assertTrue(rows[0]["score"]["needs_review"])
        disagree = apply_adjudications(self.scored, self.registry(("correct", "incorrect")))
        self.assertEqual(disagree[0]["human_review"]["status"], "disagreement")

    def test_changed_output_or_truth_invalidates_review(self):
        registry = self.registry()
        changed = score_records([{**self.row, "answer": "The allowance is two devices."}])
        with self.assertRaises(ValueError):
            apply_adjudications(changed, registry)
        changed_truth = score_records([{**self.row, "expected_fact": {**FACT, "value": "4"}}])
        with self.assertRaises(ValueError):
            apply_adjudications(changed_truth, registry)

    def test_duplicate_identity_and_naive_timestamps_are_rejected(self):
        registry = self.registry()
        registry["decisions"][1]["reviewer"] = " REVIEWER-0 "
        with self.assertRaises(ValueError):
            apply_adjudications(self.scored, registry)
        registry = self.registry()
        registry["decisions"][0]["reviewed_at"] = "2026-10-11T13:00:00"
        with self.assertRaises(ValueError):
            apply_adjudications(self.scored, registry)

    def test_aggregation_cli_exports_raw_denominators_and_review_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "raw.jsonl"
            unavailable = {**self.row, "id": "q-2", "family_id": "f-2",
                           "answer": None, "status": "unavailable"}
            input_path.write_text("\n".join(json.dumps(row) for row in (self.row, unavailable)), encoding="utf-8")
            report = aggregate_file(input_path, directory / "exports")
            config = report["configurations"]["trained_scope_editor"]
            self.assertEqual(config["accuracy"], {"numerator": 0, "denominator": 2, "value": 0.0})
            self.assertEqual(config["counts"]["unavailable"], 1)
            self.assertEqual(config["pending_review"], 1)
            self.assertTrue((directory / "exports" / "metrics.csv").exists())
            queue = (directory / "exports" / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(queue), 1)
            self.assertFalse(report["aggregation"]["inference_executed"])
            self.assertEqual(input_path.read_text(encoding="utf-8").count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
