"""Development-only recall and bounded retrieval contracts; no neural calls."""
import copy
import hashlib
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from editguard_benchmark.retrieval import rank
from editguard_benchmark.run import source_identity
from editguard_chat.examples import india_catalogue
from editguard_depth import data, retrieval


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/reasonke/depth-v1"
REPORT = ROOT / "reports/reasonke/depth-development/retrieval-selection.json"


class _NoGoldFact(dict):
    """Any use of hidden fact annotations fails instead of silently helping."""
    allowed = {"id", "question", "scope", "answer"}

    def __getitem__(self, key):
        if key not in self.allowed:
            raise AssertionError("Retriever inspected hidden annotation: " + str(key))
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in self.allowed:
            raise AssertionError("Retriever inspected hidden annotation: " + str(key))
        return super().get(key, default)

    def items(self):
        raise AssertionError("Retriever must not inspect every annotation")


class IterativeDepthRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The loader audits the entire packet, then returns only this partition.
        # No final question is selected or evaluated by these tests.
        _, cls.worlds, cls.questions = data.load_dataset(DATASET, "development")

    def test_selected_policy_completes_each_development_depth_and_control(self):
        self.assertEqual((retrieval.DEFAULT_K, retrieval.SEED_COUNT, retrieval.EXPANSION_ROUNDS), (4, 2, 3))
        depth_counts = {depth: [0, 0] for depth in (0, 1, 2, 3)}
        category_counts = {}
        for question in self.questions:
            self.assertEqual(question["split"], "development")
            selected = retrieval.retrieve_iterative(question["question"],
                self.worlds[question["lineage_id"]]["current_facts"])
            complete = set(question["required_fact_ids"]).issubset(row["id"] for row in selected)
            depth_counts[question["depth"]][0] += int(complete)
            depth_counts[question["depth"]][1] += 1
            group = category_counts.setdefault(question["category"], [0, 0])
            group[0] += int(complete); group[1] += 1
        self.assertEqual(depth_counts, {depth: [32, 32] for depth in (0, 1, 2, 3)})
        self.assertEqual(category_counts["wrong_version_or_tier"], [16, 16])
        self.assertEqual(category_counts["same_entity_other_relation"], [16, 16])

    def test_public_retriever_cannot_read_ground_truth_or_structured_scope_hints(self):
        question = next(row for row in self.questions if row["depth"] == 3)
        facts = self.worlds[question["lineage_id"]]["current_facts"]
        guarded = [_NoGoldFact(row) for row in facts]
        plain = retrieval.retrieve_iterative(question["question"], facts)
        checked = retrieval.retrieve_iterative(question["question"], guarded)
        self.assertEqual([row["id"] for row in plain], [row["id"] for row in checked])
        with self.assertRaises(ValueError):
            retrieval.retrieve_iterative(question, facts)  # Question metadata is not a valid API input.

    def test_bounds_no_duplicates_no_mutation_and_original_records_returned(self):
        question = self.questions[0]
        facts = self.worlds[question["lineage_id"]]["current_facts"]
        before = copy.deepcopy(facts)
        for cap in range(1, 7):
            selected = retrieval.retrieve_iterative(question["question"], facts, k=cap)
            self.assertEqual(len(selected), cap)
            self.assertEqual(len({row["id"] for row in selected}), cap)
            self.assertTrue(all(any(row is original for original in facts) for row in selected))
        self.assertEqual(facts, before)
        self.assertEqual(retrieval.retrieve_iterative("A valid query", []), [])
        self.assertEqual(len(retrieval.retrieve_iterative("A valid query", facts[:1], k=6)), 1)
        for cap in (0, 7, -1, True, 4.0, "4"):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                retrieval.retrieve_iterative(question["question"], facts, k=cap)
        for query in ("", " ", None, 123):
            with self.subTest(query=query), self.assertRaises(ValueError):
                retrieval.retrieve_iterative(query, facts)
        with self.assertRaises(ValueError):
            retrieval.retrieve_iterative(question["question"], facts + [dict(facts[0])])
        with self.assertRaises(ValueError):
            retrieval.retrieve_iterative("Query", [{"id": "missing"}])
        too_many = [{"id": str(index), "question": "Q", "answer": "A", "scope": ""}
                    for index in range(201)]
        with self.assertRaises(ValueError): retrieval.retrieve_iterative("Query", too_many)
        huge = [{"id": "oversized", "question": "Q", "answer": "x" * 1_000_001, "scope": ""}]
        with self.assertRaises(ValueError): retrieval.retrieve_iterative("Query", huge)

    def test_fact_order_and_ties_are_deterministic(self):
        tied = [{"id": name, "question": "Equal words", "answer": "Equal answer", "scope": "Equal scope"}
                for name in ("c", "a", "d", "b")]
        self.assertEqual([row["id"] for row in retrieval.retrieve_iterative("Unmatched query", tied)], ["a", "b", "c", "d"])
        for question in self.questions:
            facts = self.worlds[question["lineage_id"]]["current_facts"]
            shuffled = list(facts)
            random.Random(43).shuffle(shuffled)
            one = retrieval.retrieve_iterative(question["question"], facts)
            two = retrieval.retrieve_iterative(question["question"], shuffled)
            self.assertEqual([row["id"] for row in one], [row["id"] for row in two])

    def test_expansion_budget_is_finite_and_uses_lexical_answer_frontiers(self):
        question = next(row for row in self.questions if row["depth"] == 3)
        facts = self.worlds[question["lineage_id"]]["current_facts"]
        seeds = rank(question["question"], facts)[:retrieval.SEED_COUNT]
        with patch.object(retrieval, "rank", wraps=rank) as ranking:
            selected = retrieval.retrieve_iterative(question["question"], facts, k=6)
        self.assertEqual(len(selected), 6)
        self.assertLessEqual(ranking.call_count, retrieval.EXPANSION_ROUNDS + 2)
        self.assertEqual(ranking.call_args_list[0].args[0], question["question"])
        self.assertEqual(ranking.call_args_list[1].args[0], " ".join(row["answer"] for row in seeds))
        allowed_answers = {row["answer"] for row in facts}
        self.assertTrue(all(call.args[0] in allowed_answers for call in ranking.call_args_list[2:]))

    def test_both_display_development_states_work_with_unrelated_uuid_ids(self):
        display = india_catalogue()
        query = display["questions"][2]
        for view, suffixes in (("initial_facts", (1, 2, 8)), ("current_facts", (1, 3, 9))):
            facts = copy.deepcopy(display["world"][view])
            required = {"india-demo-001-f" + str(index) for index in suffixes}
            expected_questions = {row["question"] for row in facts if row["id"] in required}
            for index, row in enumerate(facts):
                row["id"] = "independent-memory-" + hashlib.sha256(str(index).encode()).hexdigest()
            selected = retrieval.retrieve_iterative(query, facts)
            self.assertTrue(expected_questions.issubset(row["question"] for row in selected))

    def test_predeclared_selection_is_reproducible_and_never_requests_final_split(self):
        calls = []
        original = data.load_dataset
        def checked(path, split):
            calls.append(split)
            return original(path, split)
        with patch.object(data, "load_dataset", side_effect=checked):
            report = retrieval.development_comparison(DATASET)
        self.assertEqual(calls, ["development"])
        self.assertEqual(report, json.loads(REPORT.read_text(encoding="utf-8")))
        self.assertEqual(report["selected"], {"seed_count": 2, "expansion_rounds": 3, "cap": 4})
        self.assertEqual(len(report["candidate_family"]), 6)
        self.assertEqual({(row["seed_count"], row["cap"]) for row in report["candidate_family"]}, set(retrieval.CANDIDATES))
        self.assertEqual(report["selection_populations"]["selection_paths"], 130)
        self.assertEqual(report["neural_model_calls"], 0)
        self.assertFalse(report["used_final_labels_for_selection"])
        self.assertFalse(report["used_final_model_outputs"])
        selected = next(row for row in report["candidate_family"] if row["seed_count"] == 2 and row["cap"] == 4)
        self.assertEqual(selected["all_development"], {"assigned": 130, "complete": 130, "found": 230, "required": 230})
        one_seed = next(row for row in report["candidate_family"] if row["seed_count"] == 1 and row["cap"] == 4)
        self.assertEqual(one_seed["complete"], 128)
        self.assertEqual(one_seed["display_development"]["complete"], 0)
        baseline = report["baseline_two_stage_k4"]
        self.assertEqual((baseline["complete"], baseline["assigned"], baseline["found"], baseline["required"]), (96, 128, 192, 224))
        self.assertEqual(baseline["by_depth"]["3"]["complete"], 0)

    def test_selection_provenance_and_completed_study_hashes_remain_intact(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.assertEqual(report["dataset_manifest_sha256"], sha(DATASET / "manifest.json"))
        self.assertEqual(report["method_source_sha256"], sha(ROOT / "editguard_depth/retrieval.py"))
        self.assertEqual(report["shared_rank_source_sha256"], sha(ROOT / "editguard_benchmark/retrieval.py"))
        prior = json.loads(REPORT.with_name("retrieval-selection-packet-only.json").read_text(encoding="utf-8"))
        self.assertEqual(prior["selected"], {"seed_count": 1, "expansion_rounds": 3, "cap": 4})
        frozen = json.loads((ROOT / "reports/reasonke/operations-final/protocol.json").read_text())
        self.assertEqual(source_identity(ROOT / "configs/operations-evaluation.json"), frozen["sources"])


if __name__ == "__main__":
    unittest.main()
