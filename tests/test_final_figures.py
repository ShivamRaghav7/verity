"""Synthetic exported-result validation; no real final inputs or inference."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from editguard.metrics import aggregate_predictions
from scripts.analyze_short_qa import summarize
from scripts import plot_final_results as plots


class FinalFigureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evaluation = self.root / "evaluation"
        self.evaluation.mkdir()
        self.protocol = self.root / "protocol.json"
        calibration = self.root / "calibration.json"
        self.write(calibration, {"threshold": .97, "ambiguity_margin": .025})
        self.write(self.protocol, {"status": "frozen", "protocol_id": "synthetic-only", "review_policy": "automated_final",
            "human_reviewed_data": False, "scoring": {"primary_version": "synthetic-scorer"},
            "source_sha256": {name: plots.digest(plots.ROOT / name) for name in ("editguard/metrics.py", "scripts/analyze_short_qa.py")},
            "calibrations": {"scope": {"path": "calibration.json", "sha256": plots.digest(calibration)}}})
        protocol_sha = plots.digest(self.protocol)
        matched, oracle = [], []
        for family in range(12):
            for kind in plots.COHORTS:
                identity = f"synthetic:{family}:{kind}"
                for config_index, config in enumerate(plots.CONFIGS):
                    applicable = kind != "out_of_scope"
                    score_correct = None if config_index < 3 else family % 3 != 0
                    status = "clarification" if config_index == 4 and family == 0 else "answered"
                    row = {"id": identity, "family_id": f"synthetic-family-{family}", "question": f"Synthetic fixture {identity}?",
                        "population": "challenge" if kind == "challenge" else "scope", "question_type": kind,
                        "request_context": {"entity_id": str(family), "plan": "Basic", "version": "v2"},
                        "expected_fact": {"value": "3", "relation": "device_limit", "unit": "devices"},
                        "expected_polarity": None, "applicable": applicable, "configuration": config,
                        "status": status, "answer": "Synthetic prescribed answer" if status == "answered" else None,
                        "answer_kind": "model_generated" if status == "answered" else "deterministic_clarification",
                        "selected_edit_id": "edit" if applicable and config_index >= 3 and family > 0 else None,
                        "focal_edit_id": "edit", "retrieval_candidates_raw": [{"edit_id": "edit"}],
                        "retrieval_hits": {"1": True, "3": True} if applicable and config_index else {},
                        "latency_ms": 10.+family+config_index*5,
                        "row_sha256": "synthetic-row-"+identity, "final_protocol_sha256": protocol_sha,
                        "review_policy": "automated_final", "human_reviewed_data": False,
                        "normalized_scope_input_seen_in_train": family == 1 and kind == "direct",
                        "normalized_scope_input_seen_in_validation": family == 1 and kind == "direct",
                        "score": {"correct": score_correct if status == "answered" else False,
                                  "needs_review": score_correct is None and status == "answered", "scorer_version": "synthetic-scorer"},
                        "short_qa": {"verdict": "correct" if family % 2 and status == "answered" else "pending_uncredited" if kind == "challenge" else "incorrect",
                            "task": "requested_value", "proves_exact_scoped_fact": False, "evidence": "synthetic-only"}}
                    row["short_qa"]["correct"] = True if row["short_qa"]["verdict"] == "correct" else False if row["short_qa"]["verdict"] == "incorrect" else None
                    if config_index == 4:
                        row.update(scope_label=int(applicable), scope_probability=.997 if applicable else .001,
                                   scope_threshold=.97)
                    matched.append(row)
            exemplar = copy.deepcopy(next(r for r in matched if r["family_id"] == f"synthetic-family-{family}" and r["configuration"] == "trained_scope_editor" and r["question_type"] == "direct"))
            exemplar.update(id=f"synthetic-oracle:{family}", population="answer", configuration="oracle_answer")
            exemplar["row_sha256"] = "synthetic-oracle-row:"+str(family)
            oracle.append(exemplar)
        oracle.extend([{**copy.deepcopy(row), "configuration": "oracle_answer"} for row in matched
                       if row["configuration"] == "trained_scope_editor" and row["population"] == "challenge"])
        self.matched, self.oracle = matched, oracle
        metrics = aggregate_predictions(matched, bootstrap_seed=42, bootstrap_resamples=2000)
        metrics.update(protocol_sha256=protocol_sha, scope_normalized_overlap={
            "scope_row_ids_sharing_train_pair": ["synthetic:1:direct"],
            "scope_row_ids_sharing_validation_pair": ["synthetic:1:direct"],
            "same_label_pairs_shared_with_train": 1})
        metrics["by_population"] = {p: aggregate_predictions([r for r in matched if r["population"] == p], bootstrap_seed=42, bootstrap_resamples=2000) for p in ("scope", "challenge")}
        secondary = {"protocol_sha256": protocol_sha, "configurations": summarize(matched), "oracle": summarize(oracle),
            "by_population": {p: summarize([r for r in matched if r["population"] == p]) for p in ("scope", "challenge")}}
        self.write(self.evaluation / "metrics.json", metrics)
        self.write(self.evaluation / "short_qa_metrics.json", secondary)
        self.rows(self.evaluation / "predictions.jsonl", matched)
        self.rows(self.evaluation / "oracle_predictions.jsonl", oracle)
        self.write(self.evaluation / "progress.json", {"status": "completed"})
        manifest = {"status": "completed", "purpose": "frozen_final_evaluation", "review_policy": "automated_final",
            "human_reviewed_data": False, "base_weights_unchanged": True, "final_test_tuning_performed": False,
            "configurations": list(plots.CONFIGS), "protocol_id": "synthetic-only", "protocol_sha256": protocol_sha,
            "scorer_version": "synthetic-scorer", "matched_prediction_records": len(matched), "oracle_records": len(oracle),
            "questions": len(matched)//5, "population_counts": {"scope": 48, "answer": 12, "challenge": 12},
            "accepted_row_sha256": {r["id"]: r["row_sha256"] for r in [*matched, *oracle]},
            "outputs_sha256": {name: plots.digest(self.evaluation / name) for name in plots.OUTPUT_NAMES},
            "sessions": [{"model_construction_seconds": .01}]}
        self.write(self.evaluation / "manifest.json", manifest)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True)+"\n", encoding="utf-8")

    @staticmethod
    def rows(path, rows):
        path.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in rows), encoding="utf-8")

    def validate(self):
        return plots.evidence(self.evaluation, self.protocol, root=self.root)

    def rehash(self, name):
        manifest = plots.read_json(self.evaluation / "manifest.json")
        manifest["outputs_sha256"][name] = plots.digest(self.evaluation / name)
        self.write(self.evaluation / "manifest.json", manifest)

    def test_completed_exports_reproduce_all_denominators_and_paired_intervals(self):
        ev = self.validate()
        self.assertEqual(len(ev["matched"]), 300)
        self.assertEqual(ev["metrics"]["configurations"]["unchanged_base"]["accuracy"]["denominator"], 60)
        self.assertEqual(ev["metrics"]["configurations"]["unchanged_base"]["counts"]["pending_review"], 60)
        self.assertEqual(ev["metrics"]["paired_comparisons"][0]["family_count"], 12)

    def test_partial_run_refused_before_any_prediction_read(self):
        self.write(self.evaluation / "progress.json", {"status": "running"})
        with patch("scripts.plot_final_results.read_rows") as reader:
            with self.assertRaisesRegex(ValueError, "command completion"):
                self.validate()
            reader.assert_not_called()

    def test_corrupt_export_hash_and_rehashed_metrics_cannot_pass(self):
        path = self.evaluation / "predictions.jsonl"
        path.write_text(path.read_text()+" ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            self.validate()
        self.rehash("predictions.jsonl")
        metrics = plots.read_json(self.evaluation / "metrics.json")
        metrics["configurations"]["unchanged_base"]["accuracy"]["numerator"] = 999
        self.write(self.evaluation / "metrics.json", metrics)
        self.rehash("metrics.json")
        with self.assertRaisesRegex(ValueError, "cannot be reproduced"):
            self.validate()

    def test_rehashed_changed_truth_and_missing_matched_record_refused(self):
        rows = copy.deepcopy(self.matched)
        rows[0]["expected_fact"]["value"] = "999"
        self.rows(self.evaluation / "predictions.jsonl", rows)
        self.rehash("predictions.jsonl")
        with self.assertRaisesRegex(ValueError, "different questions/truth"):
            self.validate()
        self.rows(self.evaluation / "predictions.jsonl", self.matched[:-1])
        self.rehash("predictions.jsonl")
        with self.assertRaisesRegex(ValueError, "must remain present"):
            self.validate()

    def test_pending_and_nonanswer_never_disappear_or_gain_credit(self):
        rows = [r for r in self.matched if r["configuration"] == "trained_scope_editor"]
        counts = plots.outcome_counts(rows)
        self.assertEqual(sum(counts.values()), 60)
        self.assertEqual(counts[plots.OUTCOMES[3]], 5)
        row = copy.deepcopy(rows[0])
        row["score"]["correct"] = True
        row["short_qa"]["verdict"] = "correct"
        self.assertEqual(plots.outcome(row), "Clarification / nonanswer")
        self.assertEqual(plots.outcome(row, "qa"), "Clarification / nonanswer")

    def test_confidence_ties_full_denominator_and_fixed_policy_preserved(self):
        rows = [{"scope_label": 1, "scope_probability": .999}, {"scope_label": 0, "scope_probability": .999},
                {"scope_label": 1, "scope_probability": .97}, {"scope_label": 0, "scope_probability": None}]
        curve = plots.descriptive_risk_coverage(rows, .97, .025)
        self.assertEqual(curve["points"][0], {"coverage": .5, "risk": .5, "covered": 2, "errors": 1})
        self.assertEqual(curve["fixed"]["covered"], 2)
        self.assertEqual(curve["fixed"]["denominator"], 4)
        self.assertEqual(curve["points"][-1]["coverage"], .75)
        self.assertEqual(curve["threshold"], .97)

    def test_failure_categories_are_observational_and_exclusive(self):
        row = copy.deepcopy(next(r for r in self.matched if r["configuration"] == "trained_scope_editor" and r["question_type"] == "out_of_scope" and r["family_id"] == "synthetic-family-3"))
        row["selected_edit_id"] = "edit"
        self.assertEqual(plots.observed_category(row), "Observed false application")
        row.update(applicable=True, selected_edit_id=None, retrieval_candidates_raw=[])
        self.assertEqual(plots.observed_category(row), "Focal edit absent in retrieval")
        row["retrieval_candidates_raw"] = [{"edit_id": "edit"}]
        self.assertEqual(plots.observed_category(row), "Retrieved edit not applied")
        row["score"]["correct"] = None
        self.assertEqual(plots.observed_category(row), "Pending semantics")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional plotting dependency")
    def test_standalone_figures_are_reproducible_and_preserve_inputs(self):
        sources = {path.name: plots.digest(path) for path in self.evaluation.iterdir()}
        first, second = self.root / "figures-a", self.root / "figures-b"
        one = plots.generate(self.evaluation, self.protocol, first, root=self.root)
        two = plots.generate(self.evaluation, self.protocol, second, root=self.root)
        self.assertEqual(one["outputs_sha256"], two["outputs_sha256"])
        self.assertEqual(len(one["outputs_sha256"]), 12)
        self.assertFalse(one["inference_executed"])
        self.assertFalse(one["human_reviewed_data"])
        self.assertEqual(sources, {path.name: plots.digest(path) for path in self.evaluation.iterdir()})
        with self.assertRaisesRegex(ValueError, "fresh figure"):
            plots.generate(self.evaluation, self.protocol, first, root=self.root)


if __name__ == "__main__":
    unittest.main()
