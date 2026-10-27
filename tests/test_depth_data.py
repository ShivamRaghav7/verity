"""Ground truth, scope and leakage checks for the separate depth challenge."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from editguard_depth.data import audit, build, CATEGORIES, load_dataset, TEMPLATES, write_dataset


ROOT = Path(__file__).resolve().parents[1]


class DepthDatasetTests(unittest.TestCase):
    def test_three_hop_chain_changes_only_through_owner_edit(self):
        worlds, questions = build()
        result = audit(worlds, questions)
        self.assertEqual(result["checked_current_and_initial_paths"], 768)
        world = worlds[0]
        question = next(row for row in questions if row["lineage_id"] == world["id"] and row["category"] == "three_hop")
        current = {row["id"]: row for row in world["current_facts"]}
        initial = {row["id"]: row for row in world["initial_facts"]}
        path = [current[key] for key in question["required_fact_ids"]]
        self.assertEqual([row["relation_id"] for row in path], ["owner", "on_call_contact", "office_city"])
        self.assertEqual(path[0]["answer"], path[1]["entity_aliases"][0])
        self.assertEqual(path[1]["answer"], path[2]["entity_aliases"][0])
        self.assertEqual(path[-1]["answer"], question["expected_answer"])
        self.assertNotEqual(question["initial_answer"], question["expected_answer"])
        self.assertEqual(sum(initial[key] != current[key] for key in initial), 1)

    def test_lineages_and_wording_banks_are_separate(self):
        worlds, questions = build()
        self.assertEqual({split: sum(row["split"] == split for row in worlds) for split in TEMPLATES}, {"development": 16, "final": 32})
        self.assertEqual({split: sum(row["split"] == split for row in questions) for split in TEMPLATES}, {"development": 128, "final": 256})
        for world in worlds:
            cases = [row for row in questions if row["lineage_id"] == world["id"]]
            self.assertEqual({row["category"] for row in cases}, set(CATEGORIES))
            self.assertTrue(all(row["split"] == row["wording_family"] == world["split"] for row in cases))
        banks = [{template for values in bank.values() for template in values} for bank in TEMPLATES.values()]
        self.assertTrue(banks[0].isdisjoint(banks[1]))

    def test_display_and_original_study_subjects_do_not_leak(self):
        worlds, _ = build()
        names = {alias for world in worlds for fact in world["current_facts"] for alias in fact["entity_aliases"]}
        reserved = {"Kaveri Payment Gateway", "Kaveri Payment Gateway Pro", "Bengaluru Platform Team", "Pune Reliability Team", "Hyderabad Integration Team", "Chennai Legacy Team", "Meera Iyer", "Arjun Deshmukh", "Konkan Records Office"}
        self.assertTrue(names.isdisjoint(reserved))
        from editguard_benchmark.data import build as build_original
        original, _ = build_original()
        original_names = {row["entity"] for row in original}
        self.assertTrue(names.isdisjoint(original_names))

    def test_bad_label_scope_join_and_branch_are_rejected(self):
        worlds, questions = build()
        broken = copy.deepcopy(questions)
        broken[0]["expected_answer"] = "Invented answer"
        with self.assertRaises(ValueError): audit(worlds, broken)
        broken = copy.deepcopy(questions)
        broken[0]["requested_scope"] = "release-1"
        with self.assertRaises(ValueError): audit(worlds, broken)
        broken_worlds = copy.deepcopy(worlds)
        for view in ("initial_facts", "current_facts"):
            next(row for row in broken_worlds[0][view] if row["id"].endswith("-f5"))["subject_entity_id"] = "wrong-person"
        with self.assertRaises(ValueError): audit(broken_worlds, questions)
        broken = copy.deepcopy(questions)
        question = next(row for row in broken if row["category"] == "three_hop")
        question["required_fact_ids"] = [question["lineage_id"] + suffix for suffix in ("-f7", "-f9", "-f10")]
        with self.assertRaises(ValueError): audit(worlds, broken)

    def test_packet_round_trip_refuses_overwrite_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "packet"
            manifest = write_dataset(target)
            loaded, worlds, questions = load_dataset(target, "final")
            self.assertEqual(loaded, manifest)
            self.assertEqual(len(worlds), 32)
            self.assertEqual(len(questions), 256)
            self.assertEqual(len(manifest["labels_sha256"]), 64)
            with self.assertRaises(ValueError): write_dataset(target)
            path = target / "questions.jsonl"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaises(ValueError): load_dataset(target, "final")

    def test_checked_in_packet_matches_deterministic_authoring(self):
        root = ROOT / "data/reasonke/depth-v1"
        manifest, worlds, questions = load_dataset(root, "development")
        expected_worlds, expected_questions = build()
        self.assertEqual(list(worlds.values()), [row for row in expected_worlds if row["split"] == "development"])
        self.assertEqual(questions, [row for row in expected_questions if row["split"] == "development"])
        self.assertEqual(manifest["generator_sha256"], hashlib.sha256((ROOT / "editguard_depth/data.py").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
