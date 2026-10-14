"""Diagnostic scoring preserves counts and explicitly separates policy gates."""

import math
import unittest

from scripts.diagnose_scope_revision import (apply_policy, auroc, distribution,
                                             metrics, validate_partitions)


def pair(label, probability, compatible=True):
    return {"label": label, "raw_probability": probability, "compatible": compatible}


class ScopeDiagnosticTests(unittest.TestCase):
    def test_probability_loss_and_confusion_have_all_denominators(self):
        result = metrics([pair(1, .8), pair(1, .3), pair(0, .6), pair(0, .1)])
        self.assertAlmostEqual(result["binary_negative_log_likelihood"],
                               -sum(math.log(p) for p in (.8, .3, .4, .9)) / 4)
        self.assertAlmostEqual(result["brier_score"], (.2 ** 2 + .7 ** 2 + .6 ** 2 + .1 ** 2) / 4)
        self.assertEqual(result["threshold_0_5"]["accuracy"], {"numerator": 2, "denominator": 4, "value": .5})
        self.assertEqual(result["auroc"]["value"], .75)

    def test_auroc_ties_half_credit_and_single_class_undefined(self):
        self.assertEqual(auroc([0, 1], [.4, .4])["value"], .5)
        self.assertEqual(auroc([1, 1], [.1, .9])["denominator"], 0)
        self.assertIsNone(auroc([1, 1], [.1, .9])["value"])

    def test_probability_distribution_even_and_odd(self):
        self.assertEqual(distribution([.3, .1, .9, .5])["median"], .4)
        self.assertEqual(distribution([.3, .1, .9])["median"], .3)
        self.assertIsNone(distribution([])["mean"])

    def test_invalid_labels_or_nonfinite_values_rejected(self):
        for record in (pair(True, .2), pair(2, .2), pair(1, float("nan")), pair(1, 1.1), pair(1, True)):
            with self.assertRaises(ValueError):
                metrics([record])

    def test_gates_reject_incompatible_without_omitting_rows(self):
        records = [pair(1, .9), pair(1, .5), pair(0, .9, False), pair(0, .1)]
        policy = {"temperature": 1., "threshold": .5, "ambiguity_margin": .05}
        raw, gated = (apply_policy(records, policy, gated=flag) for flag in (False, True))
        self.assertEqual(raw["counts"]["false_application"], 1)
        self.assertEqual(gated["counts"]["false_application"], 0)
        self.assertEqual(gated["counts"]["clarification_positive"], 1)
        self.assertEqual(gated["coverage"]["denominator"], 4)
        self.assertEqual(gated["precision"]["value"], 1.)
        self.assertEqual(gated["recall_including_clarification_as_missed"]["value"], .5)
        self.assertEqual(gated["accuracy_including_clarification_as_unsuccessful"]["value"], .75)

    def test_raw_all_positive_clarification_is_zero_recall(self):
        records = [pair(1, .5), pair(1, .5), pair(0, .1), pair(0, .1)]
        result = apply_policy(records, {"temperature": 1., "threshold": .5, "ambiguity_margin": .1})
        self.assertEqual(result["recall_including_clarification_as_missed"]["value"], 0.)
        self.assertIsNone(result["precision"]["value"])
        self.assertEqual(result["coverage"]["value"], .5)

    @staticmethod
    def row(suffix):
        return {"family_id": "family-" + suffix, "lineage_id": "lineage-" + suffix,
                "edit_id": "edit-" + suffix, "question": "question-" + suffix,
                "correction_text": "correction-" + suffix,
                "expected_fact": {"entity_id": "entity-" + suffix, "fact_id": "fact-" + suffix}}

    def test_partition_overlap_independently_detected(self):
        train, validation = self.row("a"), self.row("b")
        self.assertFalse(any(validate_partitions({"train": [train], "validation": [validation]}).values()))
        for field in ("family_id", "lineage_id", "edit_id"):
            changed = {**validation, field: train[field]}
            with self.assertRaises(ValueError):
                validate_partitions({"train": [train], "validation": [changed]})
        changed = {**validation, "question": train["question"], "correction_text": train["correction_text"]}
        with self.assertRaises(ValueError):
            validate_partitions({"train": [train], "validation": [changed]})


if __name__ == "__main__":
    unittest.main()
