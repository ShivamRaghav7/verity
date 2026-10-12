"""Curated handbook tests for bounded scope forms; no models or test dataset."""

import json
import unittest
from pathlib import Path

from editguard.comparison import resolve_context


ROOT = Path(__file__).resolve().parents[1]


class BoundedContextResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = json.loads((ROOT / "data/catalogue.json").read_text(encoding="utf-8"))["facts"]

    def setUp(self):
        self.selected = {"entity_id": "nimbus", "plan": "Basic", "version": "v2", "as_of": "2026-10-11"}

    def test_unsupported_explicit_product_cannot_inherit_selected_scope(self):
        for question in (
            "How many devices does UnknownCo support?",
            "What is the monthly price of UnknownCo?",
            "How much storage is included with UnknownCo Basic version v2?",
            "Can I connect four devices on UnknownCo?",
            "What is UnknownCo's price?",
            "How many devices does UnknownCo support as of 2026-10-11?",
        ):
            with self.subTest(question=question), self.assertRaises(ValueError):
                resolve_context(question, self.selected, self.facts)

    def test_unsupported_explicit_tier_cannot_inherit_basic_selector(self):
        for question in (
            "How many devices are permitted with Enterprise?",
            "How many devices are allowed for Enterprise?",
            "What is the device limit for Nimbus Enterprise?",
            "What is the monthly price of Nimbus Enterprise?",
            "Can I store 50 GB on Nimbus Enterprise?",
            "Does Nimbus Enterprise include at least 8 support hours per day?",
        ):
            with self.subTest(question=question), self.assertRaises(ValueError):
                resolve_context(question, self.selected, self.facts)

    def test_known_canonical_names_and_aliases_resolve_in_supported_forms(self):
        for question in (
            "How many devices does Nimbus support?",
            "How many devices does Nimbus Basic version v2 support?",
            "What is the device limit for Nimbus Basic?",
            "Can I connect four devices on the Nimbus Basic plan v2?",
            "How much storage is included with Nimbus Basic v2?",
            "Does Nimbus Basic version 2 cost at most 9 USD per month?",
            "What is Nimbus Basic's price?",
        ):
            with self.subTest(question=question):
                self.assertEqual(resolve_context(question, self.selected, self.facts), self.selected)
        # Derive aliases from curated facts instead of assuming a hidden catalogue.
        fact = next(fact for fact in self.facts if fact["entity_id"] == "nimbus")
        for alias in fact.get("aliases", []):
            with self.subTest(alias=alias):
                self.assertEqual(resolve_context(f"What is the monthly price of {alias} Basic v2?", self.selected, self.facts), self.selected)

    def test_bare_property_and_known_plan_questions_preserve_selectors(self):
        for question in (
            "What is the monthly price?",
            "How many devices are supported?",
            "How much storage is included?",
            "What does this plan cost?",
            "How many devices does Basic support?",
            "What is the monthly price of Basic?",
            "What is Basic's price?",
            "What is Basic’s price?",
            "What is the price of the Basic plan?",
            "Basic price?",
        ):
            with self.subTest(question=question):
                self.assertEqual(resolve_context(question, self.selected, self.facts), self.selected)

    def test_near_name_known_product_and_selected_scope_conflicts(self):
        mini = {**self.selected, "entity_id": "nimbus-mini"}
        self.assertEqual(resolve_context("How many devices does Nimbus Mini Basic v2 support?", mini, self.facts), mini)
        with self.assertRaises(ValueError):
            resolve_context("How many devices does Nimbus Mini Basic v2 support?", self.selected, self.facts)
        for question in ("How many devices does Nimbus Pro v2 support?",
                         "How many devices does Nimbus Basic v1 support?",
                         "How many devices does Nimbus Basic version 2.1 support?"):
            with self.subTest(question=question), self.assertRaises(ValueError):
                resolve_context(question, self.selected, self.facts)


if __name__ == "__main__":
    unittest.main()
