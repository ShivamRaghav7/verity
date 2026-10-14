"""Bounded storage-capacity alias and adversarial endorsement regressions."""

import importlib.util
import json
from pathlib import Path
import unittest

from editguard import scoring


FACT = {"entity_id": "fixture-store", "entity_name": "Fixture Store",
        "aliases": ["FS+"], "plan": "Basic", "version": "v2",
        "relation": "storage_gb", "value": "20", "unit": "GB", "as_of": "2026-10-11"}


class StorageEndorsementAliasTests(unittest.TestCase):
    def test_complete_direct_assertions_and_reviewed_entity_aliases(self):
        for answer in ("Fixture Store Basic version v2 supports 20 GB of storage.",
                       "FS+ Basic v2 supports twenty gigabytes of storage.",
                       "The Fixture Store Basic plan version 2 supports 20.0 GB of storage"):
            with self.subTest(answer=answer):
                result = scoring.score_answer(answer, FACT)
                self.assertIs(result["correct"], True)
                self.assertEqual(result["endorsed_fact"]["relation"], "storage_gb")
                self.assertEqual(result["endorsed_fact"]["value"], "20")
                self.assertEqual(result["endorsed_fact"]["unit"], "GB")

    def test_no_and_yes_comparisons_require_matching_endorsed_polarity(self):
        statement = "Fixture Store Basic v2 supports 20 GB of storage."
        for expected in ("yes", "no"):
            self.assertIs(scoring.score_answer(expected.title() + ". " + statement, FACT,
                                               expected_polarity=expected)["correct"], True)
        for answer in ("Yes. " + statement, statement):
            self.assertIs(scoring.score_answer(answer, FACT, expected_polarity="no")["correct"], False)
        self.assertIsNone(scoring.score_answer("No. " + statement, FACT)["correct"])

    def test_wrong_entity_name_tier_version_value_and_unit_are_explicit_failures(self):
        for answer in ("Fixture Store Mini Basic v2 supports 20 GB of storage.",
                       "FS Basic v2 supports 20 GB of storage.",
                       "Other Basic v2 supports 20 GB of storage.",
                       "Fixture Store Pro v2 supports 20 GB of storage.",
                       "Fixture Store Basic v1 supports 20 GB of storage.",
                       "Fixture Store Basic v2 supports 200 GB of storage.",
                       "Fixture Store Basic v2 supports 20 MB of storage.",
                       "Fixture Store Basic v2 supports 20 TB of storage."):
            with self.subTest(answer=answer):
                self.assertIs(scoring.score_answer(answer, FACT)["correct"], False)

    def test_missing_scope_relation_and_unbounded_capacity_language_remain_uncredited(self):
        for answer in ("20 GB of storage.", "Fixture Store supports 20 GB of storage.",
                       "Fixture Store Basic supports 20 GB of storage.",
                       "Fixture Store v2 supports 20 GB of storage.",
                       "Fixture Store Basic v2 supports 20 GB.",
                       "Fixture Store Basic v2 supports 20 GB storage.",
                       "Fixture Store Basic v2 supports up to 20 GB of storage.",
                       "Fixture Store Basic v2 supports at least 20 GB of storage.",
                       "Fixture Store Basic v2 supports 20 GB of RAM.",
                       "Fixture Store Basic v2 supports 20 GB of storage per device."):
            with self.subTest(answer=answer):
                self.assertIsNot(scoring.score_answer(answer, FACT)["correct"], True)

    def test_negations_examples_rejected_possibilities_and_conditionals_never_pass(self):
        for answer in ("Fixture Store Basic v2 does not support 20 GB of storage.",
                       "Fixture Store Basic v2 cannot support 20 GB of storage.",
                       "Fixture Store Basic v2 no longer supports 20 GB of storage.",
                       "Fixture Store Basic v2 supports 20 GB of storage is false.",
                       "For example, Fixture Store Basic v2 supports 20 GB of storage.",
                       "If upgraded, Fixture Store Basic v2 supports 20 GB of storage.",
                       "Perhaps Fixture Store Basic v2 supports 20 GB of storage.",
                       "Fixture Store Basic v2 supports 20 GB of storage, or perhaps 40.",
                       "The claim 'Fixture Store Basic v2 supports 20 GB of storage' is wrong.",
                       "Fixture Store Basic v2 supports 20 GB of storage. This is only an example."):
            with self.subTest(answer=answer):
                self.assertIsNot(scoring.score_answer(answer, FACT)["correct"], True)

    def test_storage_assertion_never_answers_device_price_or_support_relation(self):
        answer = "Fixture Store Basic v2 supports 20 GB of storage."
        for relation, unit in (("device_limit", "devices"), ("price_monthly", "USD/month"),
                               ("support_hours", "hours/day")):
            with self.subTest(relation=relation):
                self.assertIs(scoring.score_answer(answer, {**FACT, "relation": relation, "unit": unit})["correct"], False)

    def test_multiple_incompatible_values_and_polarities_are_failures(self):
        for answer in ("Fixture Store Basic v2 supports 20 GB of storage. Fixture Store Basic v2 includes 40 GB of storage.",
                       "No. Fixture Store Basic v2 supports 20 GB of storage. Yes. Fixture Store Basic v2 includes 20 GB of storage."):
            with self.subTest(answer=answer):
                self.assertIs(scoring.score_answer(answer, FACT, expected_polarity="no")["correct"], False)

    def test_existing_other_relation_assertion_behaviour_is_preserved(self):
        device = {**FACT, "relation": "device_limit", "value": "3", "unit": "devices"}
        self.assertIs(scoring.score_answer("Fixture Store Basic v2 supports 3 devices.", device)["correct"], True)
        self.assertIs(scoring.score_answer("Fixture Store Basic v2 supports 30 devices.", device)["correct"], False)

    def test_archive_is_exact_hash_bound_v2_source_and_new_rule_is_versioned(self):
        archive = Path(__file__).resolve().parents[1] / "reports/scoring-source-v2"
        manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
        import hashlib
        self.assertEqual(hashlib.sha256((archive / "scoring.py").read_bytes()).hexdigest(),
                         manifest["files_sha256"]["scoring.py"])
        specification = importlib.util.spec_from_file_location("editguard_archived_scorer_v2", archive / "scoring.py")
        historical = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(historical)
        self.assertEqual(historical.SCORER_VERSION, "editguard-short-facts-v2")
        self.assertEqual(scoring.SCORER_VERSION, "editguard-short-facts-v3")
        answer = "No. Fixture Store Basic v2 supports 20 GB of storage."
        self.assertIsNone(historical.score_answer(answer, FACT, expected_polarity="no")["correct"])
        self.assertIs(scoring.score_answer(answer, FACT, expected_polarity="no")["correct"], True)
        # Historical statements keep their exact verdict apart from the explicit version label.
        for sentence in ("Fixture Store Basic v2 includes 20 GB of storage.",
                         "Fixture Store Basic v2 includes 200 GB of storage.",
                         "Fixture Store Pro v2 includes 20 GB of storage.",
                         "If upgraded, Fixture Store Basic v2 includes 20 GB of storage."):
            before = historical.score_answer(sentence, FACT)
            after = scoring.score_answer(sentence, FACT)
            before.pop("scorer_version"); after.pop("scorer_version")
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
