"""Guard descriptive pairing against hidden exclusions or secondary rescoring."""

import copy
import unittest

from scripts.paired_final_short_qa import CONFIGURATIONS, SECONDARY_VERSION, summarize_pairs


def observations():
    rows = []
    for name in CONFIGURATIONS:
        for number in range(1, 5):
            correct = number == 1 if name == "updated_rag" else number in (1, 2, 4)
            status = "clarification" if name == "trained_scope_editor" and number == 3 else "answered"
            if status != "answered":
                correct = False
            # Intentionally unsupported wording: this aggregation must consume
            # stored verdicts, never rerun a question parser or the primary scorer.
            rows.append({"id": f"q{number}", "family_id": "family-a" if number <= 3 else "family-b",
                "population": "scope", "evaluation_kind": "matched", "configuration": name,
                "question": f"A sealed synthetic phrasing {number}", "expected_fact": {"value": str(number)},
                "request_context": {}, "applicable": True, "status": status,
                "answer": "Saved output" if status == "answered" else "Please clarify.",
                "answer_kind": "model_generated" if status == "answered" else "deterministic_clarification",
                "review_policy": "automated_final", "human_reviewed_data": False,
                "score": {"correct": not correct},
                "short_qa": {"correct": correct, "verdict": "correct" if correct else "incorrect",
                             "diagnostic_version": SECONDARY_VERSION, "human_semantic_review": False}})
    return rows


def report(rows):
    return summarize_pairs(rows, expected_questions=4, expected_families=2)


class SecondaryPairingTests(unittest.TestCase):
    def test_pairs_saved_secondary_not_primary_and_keeps_family_correlation(self):
        rows = observations()
        original = copy.deepcopy(rows)
        result = report(rows)
        pair = next(p for p in result["paired_comparisons"]
                    if p["left"] == "updated_rag" and p["right"] == "trained_scope_editor")
        self.assertEqual(pair["direction"], "right_minus_left")
        self.assertEqual(pair["difference"], .5)
        # The families have sizes 3 and 1; resampling correlated families gives
        # endpoints 1/3 and 1, rather than an independent-question interval.
        self.assertAlmostEqual(pair["ci95"][0], 1 / 3)
        self.assertEqual(pair["ci95"][1], 1)
        self.assertEqual(pair["matched_answerable"], 4)
        self.assertEqual(pair["family_count"], 2)
        self.assertEqual((pair["seed"], pair["resamples"]), (42, 2000))
        self.assertEqual(result["configurations"]["updated_rag"]["correct"], 1)
        self.assertEqual(result["configurations"]["trained_scope_editor"]["correct"], 3)
        self.assertEqual(rows, original)

    def test_pending_error_and_clarification_stay_in_denominator(self):
        rows = observations()
        row = next(r for r in rows if r["configuration"] == "trained_scope_editor" and r["id"] == "q4")
        row["status"] = "error"
        row["short_qa"].update(correct=None, verdict="pending_uncredited")
        result = report(rows)["configurations"]["trained_scope_editor"]
        self.assertEqual(result["denominator"], 4)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["pending_uncredited"], 1)
        self.assertEqual(result["status_counts"], {"answered": 2, "clarification": 1, "error": 1})
        self.assertEqual(result["accuracy"], .5)

    def test_missing_or_duplicate_configuration_question_refuses_inner_join(self):
        rows = observations()
        with self.assertRaisesRegex(ValueError, "complete question population"):
            report(rows[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            report(rows + [copy.deepcopy(rows[0])])

    def test_different_question_truth_or_family_cannot_be_paired(self):
        for field, value in [("expected_fact", {"value": "99"}), ("question", "different question"),
                             ("family_id", "family-b")]:
            with self.subTest(field=field):
                rows = observations()
                rows[-4][field] = value
                with self.assertRaises(ValueError):
                    report(rows)

    def test_stale_positive_on_nonanswer_or_deterministic_output_is_rejected(self):
        for change in [{"status": "error"}, {"answer": ""}, {"answer_kind": "deterministic_structured"}]:
            with self.subTest(change=change):
                rows = observations()
                rows[0].update(change)
                with self.assertRaisesRegex(ValueError, "cannot receive learned-answer credit"):
                    report(rows)

    def test_challenge_or_nonfrozen_verdicts_cannot_enter_standard_interval(self):
        for mutate in [lambda r: r.update(population="challenge"),
                       lambda r: r.update(review_policy="automated_development"),
                       lambda r: r["short_qa"].update(diagnostic_version="new-rule"),
                       lambda r: r["short_qa"].update(correct=1),
                       lambda r: r["short_qa"].update(verdict="incorrect")]:
            rows = observations()
            mutate(rows[0])
            with self.assertRaises(ValueError):
                report(rows)

    def test_pairing_is_reproducible_and_row_order_independent(self):
        rows = observations()
        self.assertEqual(report(rows), report(list(reversed(rows))))


if __name__ == "__main__":
    unittest.main()
