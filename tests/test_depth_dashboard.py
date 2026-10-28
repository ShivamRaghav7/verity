"""Saved-evidence dashboard contracts using explicit non-neural fixtures."""
import copy
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

from editguard_chat.results import DEPTH_INPUTS, depth_artifact, depth_results
from editguard_chat.server import ChatApplication, handler_for
from tests.test_chat_server import FakeBackend, FakeRetriever


class DepthDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "reports/reasonke/depth-final"

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((json.dumps(value, sort_keys=True) + "\n").encode())

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def metric(n):
        # Deliberately includes wrong, abstained and error outcomes. These are
        # software values, never model measurements or labels for inference.
        counts = {"correct": n // 2, "wrong": n // 4, "abstain": n // 8,
                  "execution_error": n // 8, "unexecuted": 0}
        return {"assigned": n, "counts": counts, "accuracy": .5, "coverage": .75,
                "generated_latency": {"n": n * 7 // 8, "p50_seconds": 2, "p95_seconds": 3}}

    def packet(self):
        report = {"schema": "editguard-depth-results-v1", "phase": "final", "execution_status": "completed",
                  "assigned_records": 544, "independent_lineages": 32, "conditions": {},
                  "explicit_non_neural_test_fixture": True}
        for prefix in ("R", "G"):
            for suffix in ("RAG", "ORACLE"):
                oracle = suffix == "ORACLE"
                report["conditions"][prefix + "-" + suffix] = {
                    "overall": self.metric(16 if oracle else 256),
                    "by_depth": {str(d): self.metric(16 if oracle else 64) for d in ([3] if oracle else [0, 1, 2, 3])},
                    "retrieval_by_depth": {str(d): {"assigned": 16 if oracle else 64,
                        "required": (16 if oracle else 64) * max(1, d),
                        "found": (16 if oracle else 64) * max(1, d), "complete": 16 if oracle else 64}
                        for d in ([3] if oracle else [0, 1, 2, 3])}}
        for name in DEPTH_INPUTS - {"results.json"}:
            self.write(self.directory / name, {"explicit_non_neural_test_fixture": name})
        self.write(self.directory / "results.json", report)
        self.reseal()
        return report

    def reseal(self, **extra):
        files = {name: self.sha(self.directory / name) for name in DEPTH_INPUTS}
        self.write(self.directory / "host-audit.json", {
            "schema": "editguard-depth-host-audit-v1", "status": "passed", "model_calls": 0,
            "files": files, "results_sha256": files["results.json"],
            "protocol_sha256": files["protocol.json"], **extra})

    def test_absent_and_partial_final_evidence_never_exposes_a_score(self):
        absent = depth_results(self.root)
        self.assertEqual(absent["status"], "unmeasured")
        self.assertNotIn("report", absent)
        self.write(self.directory / "results.json", {"explicit_non_neural_test_fixture": True})
        partial = depth_results(self.root)
        self.assertEqual(partial["status"], "unavailable")
        self.assertFalse(partial["available"])
        with self.assertRaises(FileNotFoundError):
            depth_artifact(self.root, "results.json")

    def test_complete_bound_packet_keeps_all_outcomes_and_is_a_separate_study(self):
        report = self.packet()
        result = depth_results(self.root)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["report"], report)
        self.assertEqual(result["report"]["conditions"]["R-RAG"]["overall"]["counts"]["execution_error"], 32)
        self.assertIn("not pooled", result["notice"])
        self.assertEqual(depth_artifact(self.root, "results.json"), self.directory / "results.json")
        with self.assertRaises(ValueError):
            depth_artifact(self.root, "../../secret")

    def test_changed_raw_or_resealed_wrong_population_invalidates_cached_display(self):
        original = self.packet()
        self.assertTrue(depth_results(self.root)["available"])
        self.write(self.directory / "final-reasonke.json", {"changed_fixture": True})
        self.assertFalse(depth_results(self.root)["available"])
        self.reseal()
        self.assertTrue(depth_results(self.root)["available"])
        wrong = copy.deepcopy(original)
        wrong["conditions"]["R-RAG"]["by_depth"]["3"]["assigned"] = 63
        self.write(self.directory / "results.json", wrong)
        self.reseal()
        self.assertFalse(depth_results(self.root)["available"])

    def test_cached_returns_cannot_mutate_later_responses(self):
        self.packet()
        received = depth_results(self.root)
        received["report"]["assigned_records"] = 999
        self.assertEqual(depth_results(self.root)["report"]["assigned_records"], 544)

    def test_figures_are_bound_to_the_audit_and_scorecard(self):
        self.packet()
        derived = self.directory / "derived"
        self.write(derived / "REPORT.md", {"explicit_non_neural_test_fixture": "report"})
        self.write(derived / "figure-manifest.json", {
            "schema": "editguard-depth-figures-v1", "protocol_sha256": self.sha(self.directory / "protocol.json"),
            "files": {"results.json": self.sha(self.directory / "results.json"), "REPORT.md": self.sha(derived / "REPORT.md")}})
        self.reseal(figure_manifest_sha256=self.sha(derived / "figure-manifest.json"))
        self.assertIn("REPORT.md", depth_results(self.root)["artifacts"])
        self.assertEqual(depth_artifact(self.root, "REPORT.md"), derived / "REPORT.md")
        self.write(derived / "REPORT.md", {"changed_fixture": True})
        self.assertFalse(depth_results(self.root)["available"])

    def test_http_route_and_missing_export_preserve_unmeasured_state(self):
        app = ChatApplication(self.root / "runtime", backend=FakeBackend(), retriever=FakeRetriever())
        self.addCleanup(app.store.close)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = "http://127.0.0.1:" + str(server.server_port)
        with patch("editguard_chat.server.ROOT", self.root):
            with urlopen(base + "/api/experiments/depth") as response:
                self.assertEqual(json.load(response)["status"], "unmeasured")
            with self.assertRaises(HTTPError) as error:
                urlopen(base + "/api/experiments/depth/artifacts/results.json")
            self.assertEqual(error.exception.code, 404)
            error.exception.close()


if __name__ == "__main__":
    unittest.main()
