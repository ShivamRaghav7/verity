import json
import copy
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


class HttpReportEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.models = {"scope": "a" * 64, "answer": "b" * 64}
        row = {"http_status": 200, "error": None, "model_manifests": self.models,
               "revision": 3, "http_roundtrip_ms": 100.}
        self.value = {"purpose": "isolated_neural_http_latency_v1", "status": "completed", "ml_accuracy": None,
            "live_application_mutated": False, "source_snapshot_unchanged": True,
            "isolated_fixture_domain_state_unchanged": True, "fixture_expected_revision": 3,
            "configuration": {"samples": 30, "warmups": 3, "concurrency": 1, "mode": "learned", "maximum_question_requests": 34},
            "samples": [copy.deepcopy(row) for _ in range(30)], "warmup_samples": [copy.deepcopy(row) for _ in range(3)],
            "cold_sample": copy.deepcopy(row), "expected_model_manifests": self.models,
            "source_sha256": {"configs/inference-final.json": "c" * 64},
            "counts": {"attempted": 30, "valid_responses": 30, "errors": 0},
            "latency": {"denominator": 30, "p50_ms": 100., "p95_ms": 100.}}

    def validate(self, value=None):
        report.validate_http(value or self.value, config_sha="c" * 64, model_manifests=self.models)

    def test_complete_stable_latency_only_evidence_is_accepted(self):
        self.validate()

    def test_partial_denominator_accuracy_claim_or_wrong_models_is_refused(self):
        for change in ({"status": "incomplete"}, {"ml_accuracy": 1.}, {"samples": self.value["samples"][:-1]},
                       {"expected_model_manifests": {"scope": "wrong"}}, {"live_application_mutated": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate({**self.value, **change})

    def test_changed_sample_identity_failure_or_invented_percentile_is_refused(self):
        for change in ({"revision": 4}, {"model_manifests": {}}, {"http_status": 503},
                       {"error": "CUDA failure"}, {"http_roundtrip_ms": float("nan")}):
            value = copy.deepcopy(self.value)
            value["samples"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(value)
        value = copy.deepcopy(self.value)
        value["latency"]["p95_ms"] = 1.
        with self.assertRaisesRegex(ValueError, "percentile"):
            self.validate(value)


if __name__ == "__main__":
    unittest.main()
