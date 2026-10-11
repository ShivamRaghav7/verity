"""Metrics fixtures target denominator inflation, unfair pairing and wrong retention."""

import unittest

from editguard.metrics import aggregate_predictions, paired_cluster_bootstrap, scope_metrics


FACT = {"entity_id": "nimbus", "plan": "Basic", "version": "v2",
        "relation": "device_limit", "value": "3", "unit": "devices"}


def record(identity, config="trained_scope_editor", *, family="family-a",
           correct=False, applicable=True, kind="model_generated",
           question_type="direct", status="answered", value="3"):
    return {"id": identity, "family_id": family, "configuration": config,
            "applicable": applicable, "question_type": question_type,
            "expected_fact": FACT, "answer": "An independently labeled synthetic answer.",
            "answer_kind": kind, "status": status, "latency_ms": 10,
            "score": {"correct": correct, "needs_review": correct is None,
                      "endorsed_fact": {**FACT, "value": value}},
            "retrieval_hits": {"1": applicable}}


class FixedDenominatorTests(unittest.TestCase):
    def test_abstention_error_unavailable_and_review_all_remain_in_accuracy(self):
        rows = [record("q1", correct=True), record("q2", status="clarification"),
                record("q3", status="error"), record("q4", status="unavailable"),
                record("q5", correct=None)]
        metrics = aggregate_predictions(rows)["configurations"]["trained_scope_editor"]
        self.assertEqual(metrics["accuracy"]["numerator"], 1)
        self.assertEqual(metrics["accuracy"]["denominator"], 5)
        self.assertEqual(metrics["coverage"]["value"], .4)
        self.assertEqual(metrics["selective_accuracy"]["value"], .5)
        self.assertEqual(metrics["pending_review"], 1)
        self.assertEqual(metrics["counts"]["abstentions"], 1)
        self.assertEqual(metrics["counts"]["errors"], 1)
        self.assertEqual(metrics["counts"]["unavailable"], 1)

    def test_stale_correct_score_cannot_credit_error_or_empty_answer(self):
        error = record("q1", correct=True, status="error")
        empty = {**record("q2", correct=True), "answer": ""}
        metrics = aggregate_predictions([error, empty])["configurations"]["trained_scope_editor"]
        self.assertEqual(metrics["accuracy"]["value"], 0)
        self.assertEqual(metrics["coverage"]["value"], 0)
        self.assertIsNone(metrics["selective_accuracy"]["value"])

    def test_missing_answer_field_never_credits_a_stale_score(self):
        row = record("q1", correct=True)
        del row["answer"]
        metrics = aggregate_predictions([row])["configurations"]["trained_scope_editor"]
        self.assertEqual(metrics["accuracy"]["value"], 0)
        self.assertEqual(metrics["coverage"]["value"], 0)

    def test_structured_correctness_never_becomes_neural_generator_evidence(self):
        rows = [record("q1", correct=True, kind="deterministic_structured"),
                record("q2", correct=False, kind="model_generated")]
        metrics = aggregate_predictions(rows)["configurations"]["trained_scope_editor"]
        self.assertEqual(metrics["accuracy"]["value"], .5)
        self.assertEqual(metrics["by_answer_kind"]["deterministic_structured"]["accuracy"]["value"], 1)
        self.assertEqual(metrics["by_answer_kind"]["model_generated"]["accuracy"]["value"], 0)
        self.assertEqual(metrics["model_generated_answer_accuracy"]["denominator"], 2)
        self.assertEqual(metrics["model_generated_answer_accuracy"]["value"], 0)
        self.assertEqual(metrics["model_generated_answer_coverage"]["value"], .5)

    def test_preservation_normalizes_reviewed_numeric_representations(self):
        baseline = record("q1", "unchanged_base", applicable=False, correct=True)
        edited = record("q1", applicable=False, correct=True)
        edited["score"]["endorsed_fact"] = {**FACT, "value": 3.0, "unit": "device"}
        metrics = aggregate_predictions([baseline, edited])["configurations"]["trained_scope_editor"]
        self.assertEqual(metrics["output_preservation"]["value"], 1)

    def test_generative_baselines_do_not_claim_zero_editor_false_application(self):
        rows = [record("q1", config, applicable=False, correct=False)
                for config in ("unchanged_base", "updated_rag", "prompt_only")]
        for summary in aggregate_predictions(rows)["configurations"].values():
            self.assertIsNone(summary["false_correction_application_rate"]["value"])
            self.assertIn("no explicit editor", summary["false_correction_application_rate"]["reason"])
            self.assertEqual(summary["subsets"]["out_of_scope"]["accuracy"]["denominator"], 1)

    def test_preserving_wrong_base_answers_is_not_factual_retention(self):
        baseline = [record("good", "unchanged_base", applicable=False, correct=True, question_type="out_of_scope"),
                    record("wrong", "unchanged_base", applicable=False, correct=False, value="5", question_type="out_of_scope")]
        edited = [record("good", applicable=False, status="unavailable", question_type="out_of_scope"),
                  record("wrong", applicable=False, correct=False, value="5", question_type="out_of_scope")]
        edited[1]["selected_edit_id"] = "inapplicable-edit"
        metrics = aggregate_predictions(baseline + edited)["configurations"]["trained_scope_editor"]
        self.assertEqual(metrics["baseline_correct_retention"]["denominator"], 1)
        self.assertEqual(metrics["baseline_correct_retention"]["value"], 0)
        self.assertEqual(metrics["output_preservation"]["value"], .5)
        self.assertEqual(metrics["subsets"]["out_of_scope"]["accuracy"]["value"], 0)
        self.assertEqual(metrics["false_correction_application_rate"]["value"], .5)

    def test_missing_configuration_rows_are_padded_instead_of_dropped(self):
        rows = [record("q1", "unchanged_base", correct=True),
                record("q2", "unchanged_base", correct=True, family="family-b"),
                record("q1", correct=True)]
        report = aggregate_predictions(rows)
        edited = report["configurations"]["trained_scope_editor"]
        self.assertEqual(edited["accuracy"]["denominator"], 2)
        self.assertEqual(edited["accuracy"]["value"], .5)
        self.assertEqual(edited["counts"]["missing_records"], 1)
        self.assertEqual(edited["latency"]["sample_count"], 1)
        self.assertIsNone(edited["latency"]["p95_ms"])
        self.assertEqual(report["paired_comparisons"][0]["matched_answerable"], 2)

    def test_retrieval_recall_counts_missing_hits_as_misses(self):
        rows = [record("q1", correct=True), record("q2", status="error"),
                record("q3", applicable=False)]
        rows[1]["retrieval_hits"] = {}
        metrics = aggregate_predictions(rows)["configurations"]["trained_scope_editor"]
        self.assertEqual(metrics["retrieval_recall"]["1"], {"numerator": 1, "denominator": 2, "value": .5})

    def test_latency_includes_abstentions_and_failures_when_measured(self):
        rows = [record("q1", correct=True), record("q2", status="clarification"),
                record("q3", status="error")]
        for row, latency in zip(rows, (10, 20, 100)):
            row["latency_ms"] = latency
        latency = aggregate_predictions(rows)["configurations"]["trained_scope_editor"]["latency"]
        self.assertEqual(latency["sample_count"], 3)
        self.assertEqual(latency["p50_ms"], 20)
        self.assertEqual(latency["p95_ms"], 92)

    def test_duplicate_ids_or_different_truth_fail_instead_of_deduplicating(self):
        with self.assertRaises(ValueError):
            aggregate_predictions([record("q1"), record("q1")])
        wrong = {**record("q1", "updated_rag"), "expected_fact": {**FACT, "value": "5"}}
        with self.assertRaises(ValueError):
            aggregate_predictions([record("q1"), wrong])
        left = {**record("q1", "unchanged_base"), "expected_status": "clarification"}
        right = {**record("q1"), "expected_status": "conflict"}
        with self.assertRaises(ValueError):
            aggregate_predictions([left, right])

    def test_unanswerable_handling_is_separate_from_factual_accuracy(self):
        row = {**record("q1", status="clarification"), "expected_fact": None,
               "answerable": False, "handling_correct": True}
        metrics = aggregate_predictions([row])["configurations"]["trained_scope_editor"]
        self.assertIsNone(metrics["accuracy"]["value"])
        self.assertEqual(metrics["handling_accuracy"]["value"], 1)

    def test_absent_baseline_and_empty_populations_are_not_reported_as_perfect(self):
        metrics = aggregate_predictions([record("q1")])["configurations"]["trained_scope_editor"]
        self.assertIsNone(metrics["baseline_correct_retention"]["value"])
        self.assertIsNone(metrics["subsets"]["out_of_scope"]["accuracy"]["value"])
        self.assertFalse(metrics["baseline_available"])
        self.assertEqual(aggregate_predictions([])["configurations"], {})


class RawScopeTests(unittest.TestCase):
    def test_known_confusion_and_calibration_values(self):
        result = scope_metrics([1, 0, 1, 0], [.9, .8, .6, .1], threshold=.7, bins=2)
        self.assertEqual(result["precision"]["value"], .5)
        self.assertEqual(result["recall"]["value"], .5)
        self.assertEqual(result["false_positive_rate"]["value"], .5)
        self.assertAlmostEqual(result["brier"]["value"], .205)
        self.assertAlmostEqual(result["ece"]["value"], .1)
        self.assertEqual(sum(row["count"] for row in result["reliability_bins"]), 4)

    def test_missing_pair_probability_penalizes_recall_and_discloses_incomplete_calibration(self):
        result = scope_metrics([1, 0, 1, 0], [.9, .8, .1, None], bins=2)
        self.assertEqual(result["recall"]["denominator"], 2)
        self.assertEqual(result["false_positive_rate"]["denominator"], 2)
        self.assertEqual(result["accuracy"]["denominator"], 4)
        self.assertEqual(result["probability_coverage"]["value"], .75)
        self.assertIsNone(result["brier"]["value"])
        self.assertIsNone(result["ece"]["value"])
        self.assertAlmostEqual(result["observed_only_brier"]["value"], 1.46 / 3)
        missing_positive = scope_metrics([1, 0], [None, .1])
        self.assertEqual(missing_positive["recall"]["value"], 0)

    def test_invalid_probabilities_labels_or_mixed_thresholds_are_rejected(self):
        for labels, probabilities in (([1], [1.1]), ([1], [float("nan")]),
                                      (["yes"], [.5]), ([1, 0], [.5])):
            with self.assertRaises(ValueError):
                scope_metrics(labels, probabilities)
        rows = [record("q1"), record("q2")]
        for row, threshold in zip(rows, (.5, .7)):
            row.update(scope_probability=.8, scope_label=True, scope_threshold=threshold)
        with self.assertRaises(ValueError):
            aggregate_predictions(rows)
        with self.assertRaises(ValueError):
            scope_metrics([1], [.8], threshold=True)

    def test_missing_rows_keep_explicit_pair_truth_and_threshold_without_mutation(self):
        left = [record("q1", "scope_a"), record("q2", "scope_a", family="family-b")]
        right = [record("q1", "scope_b")]
        for row in left + right:
            row.update(scope_label=1, scope_probability=.9, scope_threshold=.8)
        report = aggregate_predictions(left + right)
        metrics = report["configurations"]["scope_b"]
        self.assertEqual(metrics["scope"]["recall"], {"numerator": 1, "denominator": 2, "value": .5})
        self.assertEqual(metrics["scope"]["threshold"], .8)
        self.assertIsNone(metrics["scope"]["brier"]["value"])
        self.assertEqual(len(right), 1)
        self.assertEqual(right[0]["scope_probability"], .9)

    def test_pair_truth_is_not_inferred_from_query_applicability(self):
        rows = [record("q1"), record("q2")]
        rows[0].update(scope_label=False, scope_probability=.1)
        result = aggregate_predictions(rows)["configurations"]["trained_scope_editor"]
        self.assertEqual(result["scope"]["counts"]["total"], 1)
        self.assertEqual(result["scope_label_coverage"]["value"], .5)
        incompatible = {**record("q1", "updated_rag"), "scope_label": True}
        with self.assertRaises(ValueError):
            aggregate_predictions(rows + [incompatible])
        with self.assertRaises(ValueError):
            aggregate_predictions([{**record("unlabeled"), "scope_probability": .9}])


class PairedClusterTests(unittest.TestCase):
    def test_pairing_is_by_id_and_family_and_resamples_whole_unequal_clusters(self):
        left = [record(f"large-{i}", "left", family="large", correct=False) for i in range(3)]
        left += [record("small", "left", family="small", correct=True)]
        right = [record(f"large-{i}", "right", family="large", correct=True) for i in range(3)]
        right += [record("small", "right", family="small", correct=False)]
        right.reverse()
        result = paired_cluster_bootstrap(left, right, resamples=500)
        self.assertEqual(result["difference"], .5)
        self.assertEqual(result["ci95"], [-1.0, 1.0])
        self.assertEqual(result["family_count"], 2)
        self.assertEqual(result, paired_cluster_bootstrap(left, right, resamples=500))

    def test_unmatched_ids_and_single_family_interval_are_explicit(self):
        left = [record("q1", "left"), record("left-only", "left")]
        right = [record("q1", "right", correct=True)]
        result = paired_cluster_bootstrap(left, right)
        self.assertEqual(result["matched_ids"], 1)
        self.assertEqual(result["left_unmatched"], 1)
        self.assertIsNone(result["ci95"])
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap(left, [{**right[0], "family_id": "different"}])


if __name__ == "__main__":
    unittest.main()
