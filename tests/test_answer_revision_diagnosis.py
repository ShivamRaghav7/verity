"""Read-only answer revision diagnostics separate objectives and paired decisions."""

import unittest

from scripts.diagnose_answer_revision import (first_weight, grouped_summary,
                                              paired_changes, selection_reproduction)


def reviewed(kind="comparison", polarity="yes"):
    return {"reviewed": True, "review_status": "accepted", "question_type": kind,
            "expected_polarity": polarity, "target_text": (polarity.title() + ". fact.") if polarity else "fact."}


def record(identifier="a", correct=True, weight=8):
    return {"id": identifier, "partition": "validation", "edit_id": "edit-" + identifier,
        "subset": "comparison_yes", "target_token_ids": [1, 2], "first_token_weight": weight,
        "target_tokens": 2, "loss_weight_sum": 2 + weight - 1,
        "total_nll": 3., "weighted_total_nll": 3. + weight - 1,
        "first_token_nll": 1., "first_token_top1_correct": correct,
        "first_probability_yes": .7, "first_probability_no": .2, "first_top1_text": "Yes"}


class AnswerRevisionDiagnosisTests(unittest.TestCase):
    def test_equal_yes_no_objective_weights_and_noncomparison_unity(self):
        self.assertEqual(first_weight(reviewed()), 8)
        self.assertEqual(first_weight(reviewed(polarity="no")), 8)
        self.assertEqual(first_weight(reviewed("direct", None)), 1)
        self.assertEqual(first_weight(reviewed("paraphrase", None)), 1)

    def test_unreviewed_unsupported_and_inconsistent_targets_rejected(self):
        for alteration in ({"reviewed": False}, {"review_status": "pending"},
                           {"expected_polarity": "unknown"}, {"target_text": "No, fact."},
                           {"question_type": "direct"}, {"question_type": "dialogue", "expected_polarity": None}):
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                first_weight({**reviewed(), **alteration})

    def test_aggregate_normalizes_by_real_tokens_or_weights_not_rows(self):
        short, long = record(), {**record("b", False, 1), "target_tokens": 4, "loss_weight_sum": 4,
            "total_nll": 2., "weighted_total_nll": 2., "first_token_nll": .5}
        result = grouped_summary([short, long])
        self.assertEqual(result["target_tokens"], 6)
        self.assertEqual(result["comparison_first_token_8x_weight_sum"], 13)
        self.assertEqual(result["unweighted_token_nll"], 5 / 6)
        self.assertEqual(result["comparison_first_token_8x_token_nll"], 12 / 13)
        self.assertEqual(result["first_token_top1_correct"], {"numerator": 1, "denominator": 2, "value": .5})

    def test_selection_reproduces_own_objective_and_rejects_mismatch(self):
        values = {"unweighted_token_nll": .1, "comparison_first_token_8x_token_nll": .3}
        original = {"selection": {"metric": "validation_loss", "value": .1}}
        revision = {"loss_objective": {"policy": "comparison_first_token_8x"},
                    "selection": {"metric": "validation_weighted_token_loss", "value": .3}}
        self.assertEqual(selection_reproduction(original, values)["reproduced_value"], .1)
        self.assertEqual(selection_reproduction(revision, values)["reproduced_value"], .3)
        with self.assertRaisesRegex(ValueError, "do not agree"):
            selection_reproduction({**revision, "selection": original["selection"]}, values)
        with self.assertRaisesRegex(ValueError, "differs"):
            selection_reproduction(original, {**values, "unweighted_token_nll": .2})

    def test_paired_changes_include_regressions_and_dont_claim_full_answer_accuracy(self):
        old = [record("a", False), record("b", True), record("c", True)]
        new = [record("c", True), record("a", True), record("b", False)]
        result = paired_changes(old, new)
        self.assertEqual(result["first_token_top1_improved"], 1)
        self.assertEqual(result["first_token_top1_regressed"], 1)
        self.assertEqual(result["first_token_top1_unchanged"], 1)
        self.assertIsNone(result["free_running_answer_accuracy"])

    def test_pairing_refuses_changed_target_weights_identifiers_and_duplicates(self):
        for changed in ([record("b")], [record(), record()],
                        [{**record(), "target_token_ids": [2, 1]}],
                        [{**record(), "first_token_weight": 1}]):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                paired_changes([record()], changed)


if __name__ == "__main__":
    unittest.main()
