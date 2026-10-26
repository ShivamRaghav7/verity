"""Narrow-format explanation support checks, never a full reasoning certificate."""
import copy
import json
from pathlib import Path
import unittest

from editguard_benchmark.data import load_dataset
from scripts.audit_operations_explanations import assess, audit_receipt, audit_trace, resolve_query

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "reports/reasonke/operations-development-v1/development-reasonke.json"


class ExplanationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        _, cls.worlds, _ = load_dataset(ROOT / "data/reasonke/operations-v1", "development")

    def record(self, key):
        return next(row for row in self.receipt["records"] if row["id"] == key)

    def test_correct_indexed_subanswers_do_not_certify_surrounding_prose(self):
        row = self.record("R-RAG:ops-003-two_hop")
        result = audit_trace(row, self.worlds["ops-003"]["current_facts"])
        self.assertEqual([claim["current_fact_correctness"]["label"] for claim in result["claims"]], ["supported", "supported"])
        self.assertEqual([claim["supplied_fact_support"]["label"] for claim in result["claims"]], ["supported", "supported"])
        self.assertFalse(result["trace_fully_verified"])
        self.assertEqual(result["raw_output"], row["response"]["raw_output"])
        self.assertGreaterEqual(len(result["unassessable_segments"]), 3)
        self.assertIn("<acknowledge>", result["unassessable_segments"][-1]["text"])

    def test_old_contact_join_error_is_a_contradiction_despite_correct_first_hop(self):
        row = self.record("R-ALL:ops-004-two_hop")
        result = audit_trace(row, self.worlds["ops-004"]["current_facts"])
        self.assertEqual(result["claims"][0]["current_fact_correctness"]["label"], "supported")
        second = result["claims"][1]
        self.assertEqual(second["resolved_question"], "Who is the on-call contact for Willow unit 75?")
        self.assertEqual(second["boxed_answer"], "Ilan Voss")
        self.assertEqual(second["current_fact_correctness"]["label"], "contradicted")
        self.assertEqual(second["supplied_fact_support"]["label"], "contradicted")
        self.assertEqual(second["current_fact_correctness"]["fact_answers"], ["Lina Dane"])

    def test_missing_supplied_support_is_separate_from_current_truth(self):
        row = copy.deepcopy(self.record("R-RAG:ops-003-two_hop"))
        row["supplied_facts"] = [fact for fact in row["supplied_facts"] if fact["id"] != "ops-003-f3"]
        result = audit_trace(row, self.worlds["ops-003"]["current_facts"])
        self.assertEqual(result["claims"][1]["supplied_fact_support"]["label"], "unsupported")
        self.assertEqual(result["claims"][1]["current_fact_correctness"]["label"], "supported")

    def test_unknown_wording_and_unknown_entity_are_not_silently_supported(self):
        facts = self.worlds["ops-003"]["current_facts"]
        self.assertEqual(assess("How do we feel about this team?", "Orin Dane", facts)["label"], "unassessable")
        self.assertEqual(assess("Who is the on-call contact for Imaginary team?", "Orin Dane", facts)["label"], "unsupported")

    def test_negation_multiple_boxes_and_missing_format_preserve_unassessable_material(self):
        original = self.record("R-RAG:ops-003-two_hop")
        for replacement in (r"not \boxed{Copper unit 74}", r"\boxed{Copper unit 74} or \boxed{Saffron unit 14}"):
            row = copy.deepcopy(original)
            row["response"]["raw_output"] = row["response"]["raw_output"].replace(r"\boxed{Copper unit 74}", replacement)
            result = audit_trace(row, self.worlds["ops-003"]["current_facts"])
            self.assertEqual(result["claims"][0]["current_fact_correctness"]["label"], "unassessable")
            self.assertIsNone(result["claims"][1]["resolved_question"])
        row = copy.deepcopy(original)
        row["response"]["raw_output"] = "A claim without the declared trace format."
        result = audit_trace(row, self.worlds["ops-003"]["current_facts"])
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["unassessable_segments"][0]["text"], row["response"]["raw_output"])

    def test_reference_resolution_is_finite_and_does_not_guess_unknown_pronouns(self):
        self.assertEqual(resolve_query("Who is the on-call contact for that team?", {1: "Copper unit 74"}, 2)[0], "Who is the on-call contact for Copper unit 74?")
        self.assertIsNone(resolve_query("Who is the on-call contact for [sub answer 2]?", {1: "Copper unit 74"}, 2)[0])
        self.assertIsNone(resolve_query("Who is the on-call contact for that team?", {}, 2)[0])

    def test_execution_errors_and_unexecuted_traces_remain_in_trace_denominator(self):
        for outcome in ("execution_error", "unexecuted"):
            row = copy.deepcopy(self.record("R-RAG:ops-003-two_hop"))
            row.pop("response")
            row["outcome"] = outcome
            result = audit_trace(row, self.worlds["ops-003"]["current_facts"])
            self.assertEqual(result["trace_status"], outcome)
            self.assertEqual(result["claims"], [])

    def test_development_parser_diagnostic_keeps_its_small_population_and_failures(self):
        result = audit_receipt(RECEIPT)
        self.assertEqual(result["phase"], "development")
        self.assertEqual(result["assigned_traces"], 12)
        self.assertEqual(result["model_generations"], 0)
        self.assertFalse(result["human_reviewed"])
        self.assertFalse(result["primary_scores_modified"])
        self.assertEqual(result["by_condition"]["R-ALL"]["traces_with_current_fact_contradictions"], 2)
        self.assertEqual(result["by_condition"]["R-RAG"]["traces_with_current_fact_contradictions"], 0)
        self.assertEqual(result["by_condition"]["R-ALL"]["fully_verified_traces"], 0)


if __name__ == "__main__":
    unittest.main()
