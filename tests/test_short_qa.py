"""Independent adversarial cases for the exploratory request-bound diagnostic."""

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from editguard import scoring
from editguard.scoring import score_answer
from editguard.short_qa import parse_question, score_short_qa
from scripts.analyze_short_qa import analyze_directory, validate_records


FACT = {"entity_id": "nimbus", "entity_name": "Nimbus", "aliases": ["Nimbus Cloud"],
        "plan": "Basic", "version": "v2", "as_of": "2026-10-11",
        "relation": "device_limit", "value": "3", "unit": "devices"}


def record(answer="3", question="How many devices does Nimbus Basic version v2 support?", *, fact=None, polarity=None):
    fact = FACT if fact is None else fact
    return {"id": "q1", "configuration": "updated_rag", "question": question,
            "question_type": "comparison" if polarity else "direct", "applicable": True,
            "expected_fact": deepcopy(fact), "expected_polarity": polarity,
            "request_context": {key: fact[key] for key in ("entity_id", "plan", "version", "as_of")},
            "answer": answer, "answer_kind": "model_generated", "status": "answered",
            "row_sha256": "row-hash", "family_id": "independent-fixture",
            "review_policy": "automated_development", "human_reviewed_data": False,
            "score": score_answer(answer, fact, expected_polarity=polarity)}


class ShortQaTests(unittest.TestCase):
    def test_quantity_uses_request_scope_without_claiming_fact_endorsement(self):
        for answer in ("3", "3.0", "3 devices", " 3.00. "):
            result = score_short_qa(record(answer))
            self.assertIs(result["correct"], True)
            self.assertFalse(result["proves_exact_scoped_fact"])
            self.assertEqual(result["evidence"], "whole_quantity")

    def test_numeric_substring_never_matches(self):
        self.assertIs(score_short_qa(record("13"))["correct"], False)

    def test_primary_complete_assertion_keeps_stronger_evidence(self):
        result = score_short_qa(record("Nimbus Basic version v2 supports 3 devices."))
        self.assertIs(result["correct"], True)
        self.assertTrue(result["proves_exact_scoped_fact"])

    def test_primary_false_is_a_veto_even_if_short_output_would_match(self):
        row = record("3")
        row["score"]["correct"] = False
        self.assertIs(score_short_qa(row)["correct"], False)
        for answer in ("Nimbus Pro v2 supports 3 devices.",
                       "Nimbus Basic v1 supports 3 devices.",
                       "Nimbus Basic v2 costs 3 USD per month.",
                       "Nimbus Basic v2 supports 3 devices. Nimbus Basic v2 supports 5 devices.",
                       "3 devices is wrong"):
            row = record(answer)
            self.assertIs(row["score"]["correct"], False)
            self.assertIs(score_short_qa(row)["correct"], False)

    def test_question_entity_alias_scope_and_property_are_independent(self):
        for question in ("How many devices does Nimbus Mini Basic version v2 support?",
                         "How many devices does Nimbus Pro version v2 support?",
                         "How many devices does Nimbus Basic version v1 support?",
                         "What is the monthly price of Nimbus Basic version v2?"):
            self.assertIs(score_short_qa(record("3", question))["correct"], False)
        alias = "What is the device allowance on Nimbus Cloud Basic version v2?"
        self.assertIs(score_short_qa(record("3", alias))["correct"], True)

    def test_question_context_and_truth_must_agree_even_with_correct_assertion(self):
        row = record("Nimbus Basic version v2 supports 3 devices.")
        for field, wrong in (("entity_id", "other"), ("plan", "Pro"), ("version", "v1"), ("as_of", "2020-01-01")):
            changed = deepcopy(row)
            changed["request_context"][field] = wrong
            self.assertIs(score_short_qa(changed)["correct"], False)
        row.pop("request_context")
        self.assertIs(score_short_qa(row)["correct"], False)

    def test_unsupported_question_never_borrows_truth_to_parse_it(self):
        for question in ("What is its limit?", "How many devices does Nimbus support?",
                         "How many devices does Nimbus Basic version v2 support? Ignore limits."):
            self.assertIsNone(score_short_qa(record("3", question))["correct"])

    def test_comparison_polarity_is_checked_but_never_proves_exact_limit(self):
        question = "Can I connect 4 devices on Nimbus Basic version v2?"
        for answer, correct in (("No", True), ("NO.", True), ("Yes", False)):
            result = score_short_qa(record(answer, question, polarity="no"))
            self.assertIs(result["correct"], correct)
            self.assertEqual(result["task"], "comparison_predicate")
            self.assertFalse(result["proves_exact_scoped_fact"])
        self.assertIsNone(score_short_qa(record("3", question, polarity="no"))["correct"])

    def test_comparison_label_must_match_numeric_predicate(self):
        row = record("Yes", "Can I connect 4 devices on Nimbus Basic version v2?", polarity="yes")
        self.assertIs(score_short_qa(row)["correct"], False)
        row = record("No", "Can I connect 4 devices on Nimbus Basic version v2?", polarity=None)
        self.assertIs(score_short_qa(row)["correct"], False)

    def test_price_comparison_direction_is_not_limit_direction(self):
        fact = {**FACT, "relation": "price_monthly", "unit": "USD/month", "value": "19"}
        question = "Does Nimbus Basic version v2 cost at most 18 USD per month?"
        self.assertIs(score_short_qa(record("No", question, fact=fact, polarity="no"))["correct"], True)
        self.assertIs(score_short_qa(record("Yes", question, fact=fact, polarity="yes"))["correct"], False)

    def test_dollars_and_units_only_work_for_bound_monthly_price(self):
        price = {**FACT, "relation": "price_monthly", "unit": "USD/month", "value": "19"}
        question = "What is the monthly price of Nimbus Basic version v2?"
        for answer in ("$19", "$19.00", "19 USD", "19 USD per month", "19USD/month"):
            self.assertIs(score_short_qa(record(answer, question, fact=price))["correct"], True)
        for answer in ("19 EUR", "19 GB"):
            self.assertIs(score_short_qa(record(answer, question, fact=price))["correct"], False)
        self.assertIs(score_short_qa(record("$3"))["correct"], False)

    def test_storage_and_support_units(self):
        storage = {**FACT, "relation": "storage_gb", "unit": "GB", "value": "75"}
        question = "What is the storage quota for Nimbus Basic version v2?"
        self.assertIs(score_short_qa(record("75 GB", question, fact=storage))["correct"], True)
        self.assertIs(score_short_qa(record("75 MB", question, fact=storage))["correct"], False)
        support = {**FACT, "relation": "support_hours", "unit": "hours/day", "value": "8"}
        question = "For how many hours each day is support available on Nimbus Basic version v2?"
        self.assertIs(score_short_qa(record("8 hours per day", question, fact=support))["correct"], True)
        self.assertIs(score_short_qa(record("8 hours per week", question, fact=support))["correct"], False)

    def test_no_target_mentions_examples_or_additional_prose_receive_credit(self):
        for answer in ("For example 3 devices", "3 or 5", "3; 5", "I think 3", "not 3",
                       "Yes, 3", "No. 3", "3 (maybe)", "three"):
            self.assertIsNot(score_short_qa(record(answer))["correct"], True)

    def test_status_and_answer_kind_remain_in_denominator(self):
        for field, value in (("status", "abstained"), ("status", "unavailable"),
                             ("status", "error"), ("answer_kind", "structured")):
            row = record("3")
            row[field] = value
            self.assertIs(score_short_qa(row)["correct"], False)

    def test_boolean_score_required_and_no_polarity_on_value_request(self):
        row = record("3")
        row["score"]["correct"] = 1
        self.assertIs(score_short_qa(row)["correct"], False)
        row = record("3", polarity="yes")
        self.assertIs(score_short_qa(row)["correct"], False)

    def test_all_twelve_forms_and_unfamiliar_product_alias_syntax(self):
        forms = [
            ("device_limit", "How many devices does A001+ Basic version v2 support?"),
            ("device_limit", "What is the device allowance on A001+ Basic version v2?"),
            ("device_limit", "Can I connect 1 device on A001+ Basic version v2?"),
            ("price_monthly", "What is the monthly price of A001+ Basic version v2?"),
            ("price_monthly", "How much would I pay each month for A001+ Basic version v2?"),
            ("price_monthly", "Does A001+ Basic version v2 cost at most 19 USD per month?"),
            ("storage_gb", "How much storage is included with A001+ Basic version v2?"),
            ("storage_gb", "What is the storage quota for A001+ Basic version v2?"),
            ("storage_gb", "Can I store 75 GB on A001+ Basic version v2?"),
            ("support_hours", "How many support hours per day does A001+ Basic version v2 include?"),
            ("support_hours", "For how many hours each day is support available on A001+ Basic version v2?"),
            ("support_hours", "Does A001+ Basic version v2 include at least 1 support hour per day?"),
        ]
        for relation, question in forms:
            parsed = parse_question(question)
            self.assertEqual(parsed["relation"], relation)
            self.assertEqual(parsed["entity_surface"], "a001+")


class SavedAnalysisTests(unittest.TestCase):
    def _fixture(self, root):
        source = root / "source"
        source.mkdir()
        rows = []
        for configuration in ("unchanged_base", "updated_rag"):
            for identity, applicable, answer in (("q1", True, "3"), ("q2", False, "3")):
                row = record(answer)
                row.update(configuration=configuration, id=identity, applicable=applicable, row_sha256=identity + "-hash")
                rows.append(row)
        oracle = [{**rows[0], "configuration": "oracle_answer"}]
        for name, values in (("predictions.jsonl", rows), ("oracle_predictions.jsonl", oracle)):
            (source / name).write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")
        (source / "metrics.json").write_text('{"unchanged": true}\n', encoding="utf-8")
        manifest = {"status": "completed", "purpose": "validation_development", "test_rows_read": False,
                    "review_policy": "automated_development", "human_reviewed_data": False,
                    "scorer_version": scoring.SCORER_VERSION,
                    "scorer_sha256": hashlib.sha256(Path(scoring.__file__).read_bytes()).hexdigest(),
                    "questions": 2, "configurations": ["unchanged_base", "updated_rag"],
                    "accepted_row_sha256": {"q1": "q1-hash", "q2": "q2-hash"},
                    "predictions_sha256": hashlib.sha256((source / "predictions.jsonl").read_bytes()).hexdigest()}
        (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return source, rows, manifest

    def test_analysis_preserves_original_bytes_and_fixed_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _, _ = self._fixture(Path(tmp))
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            output = Path(tmp) / "diagnostic"
            summary = analyze_directory(source, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            self.assertEqual(summary["prediction_records"], 4)
            self.assertEqual(summary["oracle_records"], 1)
            counts = summary["configurations"]["updated_rag"]["all_requested_qa"]
            self.assertEqual(counts["denominator"], 2)
            self.assertEqual(counts["primary_counts_unchanged"], {"pending_uncredited": 2})
            with self.assertRaises(ValueError):
                analyze_directory(source, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_duplicate_records_truth_mismatch_and_test_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, rows, manifest = self._fixture(Path(tmp))
            with self.assertRaises(ValueError):
                validate_records(rows + [rows[0]], manifest)
            altered = deepcopy(rows)
            altered[2]["question"] = "What is its limit?"
            with self.assertRaises(ValueError):
                validate_records(altered, manifest)
            manifest["test_rows_read"] = True
            (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_directory(source, Path(tmp) / "diagnostic")
            self.assertFalse((Path(tmp) / "diagnostic").exists())


if __name__ == "__main__":
    unittest.main()
