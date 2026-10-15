import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_project_report as report


class ProjectReportEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.enterContext(patch.object(report, "ROOT", self.root))
        self.run = self.root / "reports/final/evaluation"
        self.figures = self.root / "reports/final/figures"
        self.run.mkdir(parents=True)
        self.figures.mkdir()
        self.protocol = self.root / "reports/final/protocol-v2.json"
        self.lock = self.root / "reports/final/test_lock-v2.json"
        self.write(self.protocol, {"scoring": {"primary_version": "fixture", "primary_sha256": "a" * 64}})
        self.write(self.lock, {"protocol_sha256": report.sha(self.protocol), "packet_manifest_sha256": "b" * 64})
        self.write(self.run / "metrics.json", {"configurations": {name: {"count": 816} for name in report.NAMES}})
        self.write(self.run / "short_qa_metrics.json", {})
        for name in ("predictions.jsonl", "oracle_predictions.jsonl"):
            (self.run / name).write_bytes(b"synthetic source not model evidence\n")
        self.manifest = {"status": "completed", "purpose": "frozen_final_evaluation", "human_reviewed_data": False,
            "review_policy": "automated_final", "base_weights_unchanged": True, "fresh_final_runs": 1,
            "final_test_tuning_performed": False, "protocol_sha256": report.sha(self.protocol),
            "packet_manifest_sha256": "b" * 64, "questions": 816, "oracle_records": 384,
            "matched_prediction_records": 4080, "population_counts": {"scope": 720, "answer": 288, "challenge": 96},
            "scorer_version": "fixture", "scorer_sha256": "a" * 64,
            "outputs_sha256": {p.name: report.sha(p) for p in self.run.iterdir()}}
        self.write(self.run / "manifest.json", self.manifest)
        (self.figures / "example.png").write_bytes(b"synthetic figure bytes")
        self.figure_manifest = {"protocol_sha256": report.sha(self.protocol),
            "outputs_sha256": {"example.png": report.sha(self.figures / "example.png")},
            "source_files_sha256": {str(i): report.sha(p) for i, p in enumerate((self.run / "manifest.json",
                self.run / "metrics.json", self.run / "short_qa_metrics.json", self.protocol))}}
        self.write(self.figures / "manifest.json", self.figure_manifest)
        self.write(self.root / "reports/targeted-training-audit.json", {"purpose": "synthetic test fixture"})

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def load(self):
        return report.load_evidence(self.run, self.protocol, self.lock, self.figures)

    def test_complete_bound_evidence_loads_without_pdf_or_ml_import(self):
        evidence = self.load()
        self.assertEqual(evidence[0]["questions"], 816)
        self.assertTrue(evidence[-1])

    def test_partial_or_false_human_claim_is_not_a_final_report(self):
        for field, value in (("status", "running"), ("human_reviewed_data", True)):
            self.write(self.run / "manifest.json", {**self.manifest, field: value})
            with self.assertRaises(ValueError):
                self.load()

    def test_dropped_denominator_or_changed_scoring_is_refused(self):
        for field, value in (("questions", 815), ("scorer_version", "different")):
            self.write(self.run / "manifest.json", {**self.manifest, field: value})
            with self.assertRaises(ValueError):
                self.load()

    def test_prediction_or_figure_byte_tampering_is_refused(self):
        (self.figures / "example.png").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.load()
        (self.figures / "example.png").write_bytes(b"synthetic figure bytes")
        (self.run / "predictions.jsonl").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.load()

    def test_different_figure_source_or_protocol_is_refused(self):
        for change in ({"protocol_sha256": "c" * 64}, {"source_files_sha256": {}}):
            self.write(self.figures / "manifest.json", {**self.figure_manifest, **change})
            with self.assertRaises(ValueError):
                self.load()


if __name__ == "__main__":
    unittest.main()
