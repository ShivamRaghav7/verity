import unittest
from io import BytesIO
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from urllib.error import HTTPError

from scripts import benchmark_neural_http as benchmark
from scripts.benchmark_neural_http import bind_sample, payload, sample, summarize


class NeuralHttpBenchmarkTests(unittest.TestCase):
    def row(self, latency=100, **changes):
        return {"http_status": 200, "error": None, "http_roundtrip_ms": latency,
                "revision": 2, "model_manifests": {"scope": "a" * 64, "answer": "b" * 64},
                "answer_kind": "model_generated", "status": "answered", **changes}

    def test_errors_and_clarifications_remain_in_attempted_denominators(self):
        value = summarize([self.row(100), self.row(200, status="clarification", answer_kind="deterministic_clarification"),
                           self.row(1000, http_status=503, error="unavailable")])
        self.assertEqual(value["counts"], {"attempted": 3, "valid_responses": 2, "errors": 1, "answered": 1, "abstentions": 1})
        self.assertEqual(value["latency"]["denominator"], 3)
        self.assertEqual(value["latency"]["p50_ms"], 200)
        self.assertEqual(value["latency"]["valid_response_p50_ms"], 150)
        self.assertEqual(value["status"], "incomplete")

    def test_stable_responses_use_interpolated_percentiles(self):
        value = summarize([self.row(10), self.row(20), self.row(30)])
        self.assertEqual(value["status"], "completed")
        self.assertEqual(value["latency"]["p95_ms"], 29)

    def test_changed_revision_or_models_cannot_claim_stable_run(self):
        for change in ({"revision": 3}, {"model_manifests": {"scope": "changed"}}):
            with self.subTest(change=change):
                self.assertEqual(summarize([self.row(), self.row(**change)])["status"], "incomplete")

    def test_payloads_are_fixed_development_questions_and_learned_mode(self):
        self.assertEqual(payload(0), payload(3))
        self.assertEqual(payload(0)["mode"], "learned")
        self.assertEqual(payload(0)["entity_id"], "nimbus")

    def test_missing_model_identities_cannot_count_as_valid_stable_measurements(self):
        report = summarize([self.row(model_manifests={})])
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["counts"]["errors"], 1)
        self.assertEqual(report["latency"]["denominator"], 1)
        self.assertEqual(report["latency"]["valid_denominator"], 0)

    def test_each_phase_is_bound_to_expected_revision_and_real_manifest_hashes(self):
        expected = {"scope": "a" * 64, "answer": "b" * 64}
        self.assertIsNone(bind_sample(self.row(), 2, expected)["error"])
        for row in (self.row(revision=3), self.row(model_manifests={"scope": "c" * 64, "answer": "b" * 64})):
            self.assertIn("differ", bind_sample(row, 2, expected)["error"])

    def test_sample_extracts_actual_api_provenance_key_and_rejects_nonobject_json(self):
        class Response(BytesIO):
            status = 200
        body = {"status": "answered", "answer_kind": "model_generated", "revision": 2,
                "model_training": {task: {"artifact_manifest_sha256": letter * 64} for task, letter in (("scope", "a"), ("answer", "b"))}}
        with patch.object(benchmark, "urlopen", return_value=Response(json.dumps(body).encode())):
            row = sample("http://127.0.0.1:12345", 0, "cold")
        self.assertIsNone(row["error"])
        self.assertEqual(row["model_manifests"], {"scope": "a" * 64, "answer": "b" * 64})
        with patch.object(benchmark, "urlopen", return_value=Response(b"[]")):
            malformed = sample("http://127.0.0.1:12345", 0, "measured")
        self.assertIn("object", malformed["error"])
        self.assertGreaterEqual(malformed["http_roundtrip_ms"], 0)

    def test_http_failures_retain_measured_roundtrip_and_error_detail(self):
        failure = HTTPError("http://127.0.0.1:12345/api/ask", 503, "Unavailable", {}, BytesIO(b"CUDA unavailable"))
        with patch.object(benchmark, "urlopen", side_effect=failure):
            row = sample("http://127.0.0.1:12345", 0, "cold")
        self.assertEqual(row["http_status"], 503)
        self.assertIn("CUDA", row["error"])
        self.assertGreaterEqual(row["http_roundtrip_ms"], 0)

    def test_cold_gpu_failure_exports_zero_attempted_warm_measurements_without_inference(self):
        class Process:
            terminated = False
            def poll(self): return 0 if self.terminated else None
            def terminate(self): self.terminated = True
            def wait(self, timeout): return 0
        class Health(BytesIO):
            status = 200
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "editguard").mkdir()
            (root / "editguard/api.py").write_text("# synthetic API identity", encoding="utf-8")
            own_source = root / "benchmark.py"
            own_source.write_text("# synthetic benchmark identity", encoding="utf-8")
            for task in ("scope", "answer"):
                (root / task).mkdir()
                (root / task / "manifest.json").write_text('{"synthetic":true}', encoding="utf-8")
            config = root / "config.json"
            config.write_text(json.dumps({"scope_artifact": "scope", "answer_artifact": "answer"}), encoding="utf-8")
            output = root / "report.json"
            cold_failure = self.row(http_status=503, error="CUDA unavailable", phase="cold_lazy_model_loading", index=0)
            process = Process()
            with patch.object(benchmark, "ROOT", root), patch.object(benchmark, "__file__", str(own_source)), \
                    patch.object(benchmark.subprocess, "Popen", return_value=process), \
                    patch.object(benchmark, "urlopen", return_value=Health()), \
                    patch.object(benchmark, "sample", return_value=cold_failure) as sampled:
                report = benchmark.run(config, output)
            self.assertTrue(output.is_file())
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["counts"]["attempted"], 0)
            self.assertEqual(report["phase_counts"]["measured_not_attempted"], 30)
            self.assertEqual(report["phase_counts"]["warmup_not_attempted"], 3)
            self.assertEqual(report["phase_counts"]["cold_errors"], 1)
            self.assertIsNone(report["latency"]["p50_ms"])
            self.assertIsNotNone(report["execution_failure"])
            self.assertTrue(report["source_snapshot_unchanged"])
            self.assertTrue(report["isolated_fixture_domain_state_unchanged"])
            self.assertTrue(process.terminated)
            sampled.assert_called_once()


if __name__ == "__main__":
    unittest.main()
