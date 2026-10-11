import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_data import (NEGATIVE_TYPES, VALUE_POOLS, build_dataset, content_hash,
                                  freeze_test, load_reviewed_rows, write_dataset)


class DataPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = build_dataset()
        cls.temp_root = Path(__file__).resolve().parents[1] / "runtime/test-data"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def test_family_lineage_and_negative_entities_are_split_disjoint(self):
        seen = {}
        for task in ("scope", "answer"):
            for row in self.dataset[task]:
                family = row["family_id"]
                self.assertEqual(row["split"], self.dataset["family_splits"][family])
                for key in (row["lineage_id"], row["correction"]["entity_id"],
                            row["expected_fact"]["entity_id"]):
                    self.assertEqual(seen.setdefault(key, row["split"]), row["split"])
                if "question_family_id" in row:
                    self.assertEqual(row["split"], self.dataset["family_splits"][row["question_family_id"]])

    def test_counts_and_unreviewed_status(self):
        self.assertEqual(len(self.dataset["scope"]), 4800)
        self.assertEqual(len(self.dataset["answer"]), 1920)
        self.assertEqual(len(self.dataset["facts"]), 1280)
        self.assertEqual(len(self.dataset["edits"]), 480)
        self.assertEqual(len(self.dataset["challenges"]), 96)
        self.assertTrue(all(not row["reviewed"] for row in self.dataset["scope"]))

    def test_negative_taxonomy_matches_explicit_facts(self):
        self.assertEqual({row["negative_type"] for row in self.dataset["scope"] if row["label"] == 0}, set(NEGATIVE_TYPES))
        for row in self.dataset["scope"]:
            if row["label"]:
                self.assertEqual(row["expected_fact"]["value"], row["correction"]["corrected_value"])
                continue
            edit, fact = row["correction"], row["expected_fact"]
            kind = row["negative_type"]
            if kind == "same_entity_different_property":
                self.assertEqual(edit["entity_id"], fact["entity_id"])
                self.assertNotEqual(edit["relation"], fact["relation"])
            if kind in ("different_entity_same_property", "similar_name"):
                self.assertNotEqual(edit["entity_id"], fact["entity_id"])
                self.assertEqual(edit["relation"], fact["relation"])
            if kind == "wrong_tier":
                self.assertEqual(edit["entity_id"], fact["entity_id"])
                self.assertNotEqual(edit["plan"], fact["plan"])
            if kind == "wrong_version":
                self.assertEqual(edit["entity_id"], fact["entity_id"])
                self.assertNotEqual(edit["version"], fact["version"])

    def test_counterfactual_variants_remain_together_and_support_excludes_target(self):
        scenarios = {}
        for row in self.dataset["answer"]:
            self.assertTrue(all(fact["relation"] != row["expected_fact"]["relation"]
                                for fact in row["supporting_context"]))
            key = (row["lineage_id"], row["question"])
            scenarios.setdefault(key, set()).add(row["expected_fact"]["value"])
        self.assertTrue(any(len(values) == 2 for values in scenarios.values()))

    def test_shared_value_support_and_per_edit_polarity_balance(self):
        comparisons = {}
        for edit in self.dataset["edits"]:
            pool = {str(value) for value in VALUE_POOLS[edit["relation"]]}
            self.assertIn(edit["original_value"], pool)
            self.assertIn(edit["corrected_value"], pool)
            self.assertNotEqual(edit["original_value"], edit["corrected_value"])
        for row in self.dataset["answer"]:
            if row["expected_polarity"]:
                comparisons.setdefault(row["edit_id"], set()).add(row["expected_polarity"])
        self.assertEqual(len(comparisons), len(self.dataset["edits"]))
        self.assertTrue(all(labels == {"yes", "no"} for labels in comparisons.values()))

    def test_unseen_value_challenges_have_explicit_corrections_and_novel_values(self):
        novel = [row for row in self.dataset["challenges"] if row["challenge_type"] == "unseen_value_draft"]
        self.assertEqual(len(novel), 48)
        for row in novel:
            expected = row["expected_fact"]
            self.assertNotIn(int(expected["value"]), VALUE_POOLS[expected["relation"]])
            self.assertEqual(expected["value"], row["correction"]["corrected_value"])
            self.assertEqual(row["split"], "test")

    def test_preparation_is_reproducible_and_preflights_overwrites(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            output = Path(directory)
            first = write_dataset(output)
            self.assertEqual(first, write_dataset(output))
            before = (output / "scope_test.jsonl").read_bytes()
            with self.assertRaises(ValueError):
                write_dataset(output, seed=999)
            self.assertEqual(before, (output / "scope_test.jsonl").read_bytes())

    def test_review_gate_rejects_pending_and_stale_rows(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            path = Path(directory) / "rows.jsonl"
            registry = Path(directory) / "reviews.json"
            row = self.dataset["scope"][0]
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            registry.write_text(json.dumps({"rows": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_reviewed_rows(path, registry)
            record = {"decision": "accept", "reviewer": "example reviewer",
                      "reviewed_at": "2026-10-11T12:00:00+05:30", "row_sha256": content_hash(row)}
            registry.write_text(json.dumps({"rows": {row["id"]: record}}), encoding="utf-8")
            self.assertTrue(load_reviewed_rows(path, registry)[0]["reviewed"])
            path.write_text(json.dumps({**row, "label": 1 - row["label"]}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_reviewed_rows(path, registry)

    def test_test_lock_requires_review_and_frozen_configuration(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            output = Path(directory)
            write_dataset(output, family_count=20)
            registry = output / "reviews.json"
            config = output / "evaluation.json"
            config.write_text('{"status":"development"}', encoding="utf-8")
            registry.write_text('{"rows":{}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                freeze_test(output, registry, config)
            records = {}
            for name in ("scope_test.jsonl", "answer_test.jsonl", "challenge_drafts.jsonl"):
                for line in (output / name).read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    records[row["id"]] = {"decision": "accept", "reviewer": "example reviewer",
                                          "reviewed_at": "2026-10-11T12:00:00+05:30", "row_sha256": content_hash(row)}
            registry.write_text(json.dumps({"rows": records}), encoding="utf-8")
            with self.assertRaises(ValueError):
                freeze_test(output, registry, config)
            config.write_text('{"status":"frozen"}', encoding="utf-8")
            lock = freeze_test(output, registry, config)
            self.assertIn("scope_test.jsonl", lock["files"])
            with self.assertRaises(ValueError):
                write_dataset(output, family_count=20, seed=3, overwrite=True)

    def test_rejected_exclusions_require_review_reason(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            path = Path(directory) / "rows.jsonl"
            registry = Path(directory) / "reviews.json"
            first, second = self.dataset["scope"][:2]
            path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
            records = {
                row["id"]: {"decision": "accept", "reviewer": "example reviewer",
                            "reviewed_at": "2026-10-11T12:00:00+05:30", "row_sha256": content_hash(row)}
                for row in (first, second)
            }
            records[second["id"]]["decision"] = "reject"
            registry.write_text(json.dumps({"rows": records}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_reviewed_rows(path, registry)
            records[second["id"]]["reason"] = "Question requires a clearer scope."
            registry.write_text(json.dumps({"rows": records}), encoding="utf-8")
            self.assertEqual(len(load_reviewed_rows(path, registry)), 1)


if __name__ == "__main__":
    unittest.main()
