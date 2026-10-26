"""Data/scoring/retrieval and finite-plan contracts, with no neural inference."""
import copy
from pathlib import Path
import tempfile
import unittest

from editguard_benchmark.data import audit, build, load_dataset
from editguard_benchmark.metrics import paired_bootstrap, summarize
from editguard_benchmark.retrieval import retrieve
from editguard_benchmark.run import request_for, source_identity, tasks_for, verify_protocol
from editguard_benchmark.scoring import score
from editguard_chat.results import current_results
import json

ROOT = Path(__file__).resolve().parents[1]


class OperationsTests(unittest.TestCase):
    def test_graph_ground_truth_and_lineage_first_split(self):
        lineages, questions = build()
        self.assertEqual(audit(lineages, questions)["checked_question_paths"], 384)
        broken = copy.deepcopy(questions)
        broken[0]["expected_answer"] = "Wrong"
        with self.assertRaises(AssertionError):
            audit(lineages, broken)

    def test_checked_in_packet_matches_generator(self):
        _, worlds, questions = load_dataset(ROOT / "data/reasonke/operations-v1", "development")
        expected_worlds, expected_questions = build()
        self.assertEqual(questions, [row for row in expected_questions if row["split"] == "development"])
        self.assertEqual(list(worlds.values()), [row for row in expected_worlds if row["split"] == "development"])

    def test_scoring_requires_endorsed_whole_final_span_and_preserves_outcomes(self):
        _, questions = build()
        question = questions[0]
        expected = question["expected_answer"]
        self.assertEqual(score(question, {"content": expected})["outcome"], "correct")
        for text in (f"Not {expected}", f"For example {expected}", f"{expected} or another team"):
            self.assertEqual(score(question, {"content": text})["outcome"], "wrong")
        self.assertEqual(score(question, {"content": "UNKNOWN"})["outcome"], "abstain")
        self.assertEqual(score(question, error={"message": "failed"})["outcome"], "execution_error")
        self.assertEqual(score(question)["outcome"], "unexecuted")

    def test_shared_retrieval_never_reads_gold_question_fields(self):
        _, worlds, questions = load_dataset(ROOT / "data/reasonke/operations-v1", "development")
        row = next(row for row in questions if row["category"] == "two_hop")
        facts = worlds[row["lineage_id"]]["current_facts"]
        selected = retrieve(row["question"], facts, 4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({fact["id"] for fact in selected}), 4)
        config = json.loads((ROOT / "configs/operations-evaluation.json").read_text())
        one, _ = request_for({"question": row, "condition": "G-RAG"}, worlds, config)
        two, _ = request_for({"question": row, "condition": "R-RAG"}, worlds, config)
        self.assertEqual(one, two)
        self.assertNotIn("expected_answer", one)

    def test_all_assigned_denominator_and_grouped_pairing(self):
        _, questions = build()
        rows = []
        for index, outcome in enumerate(("correct", "wrong", "abstain", "execution_error", "unexecuted")):
            rows.append({"question": questions[index], "outcome": outcome, "correct": outcome == "correct"})
        summary = summarize(rows)
        self.assertEqual(summary["assigned"], 5)
        self.assertEqual(summary["accuracy"], .2)
        self.assertEqual(summary["coverage"], .4)
        self.assertEqual(summary["answered_only_precision"], .5)
        paired = paired_bootstrap(rows, copy.deepcopy(rows), resamples=20)
        self.assertEqual(paired["left_minus_right"], 0)
        self.assertEqual(paired["ci95"], [0, 0])
        self.assertEqual(paired["independent_lineages"], 1)

    def test_finite_job_sizes_and_profile_category_coverage(self):
        config = json.loads((ROOT / "configs/operations-evaluation.json").read_text())
        totals = {}
        for model in ("reasonke", "instruction"):
            for phase in ("profile", "development", "final"):
                _, _, tasks = tasks_for(config, model, phase)
                totals[model, phase] = len(tasks)
                if phase == "profile":
                    self.assertEqual(len({row["question"]["category"] for row in tasks}), 8)
        self.assertEqual(totals, {("reasonke", "profile"): 12, ("instruction", "profile"): 12,
            ("reasonke", "development"): 72, ("instruction", "development"): 48,
            ("reasonke", "final"): 784, ("instruction", "final"): 528})
        self.assertEqual(sum(totals.values()), 1456)

    def test_final_requires_source_bound_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps({"status": "frozen", "sources": {"changed": "bad"}}))
            with self.assertRaises(ValueError):
                verify_protocol(path, ROOT / "configs/operations-evaluation.json")

    def test_current_dashboard_does_not_relabel_visual_or_development_scores(self):
        report = current_results(ROOT)
        self.assertEqual(report["kind"], "development_acceptance")
        self.assertEqual(report["workflow"]["factual_matches"], 6)
        self.assertEqual(report["workflow"]["assigned"], 7)
        self.assertEqual(report["profile"]["assigned"], 5)
        self.assertIn("not a general accuracy benchmark", report["notice"])


if __name__ == "__main__":
    unittest.main()
