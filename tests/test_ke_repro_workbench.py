"""Tiny application contract tests; the fake is never model-quality evidence."""
import json
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from ke_repro.workbench import Workbench, WorkbenchError, handler_for


class FakeBackend:
    def __init__(self):
        self.active = False
        self.fail_save = False
        self.last_question = None
        self.last_method = None

    def weight_hashes(self):
        return {"matrix": "edited" if self.active else "original"}

    def generate(self, image, question, max_new_tokens):
        self.last_question = question
        assert max_new_tokens == 24
        return "new answer" if self.active else "old answer"

    def edit(self, image, question, target, target_alignment):
        assert target_alignment == "context"
        self.last_method = "mend"
        self.active = True
        return {"changed_parameters": ["matrix"]}

    def edit_finetune(self, image, question, target):
        self.last_method = "ft_last_layer"
        self.active = True
        return {"changed_parameters": ["matrix"], "steps": 15}

    def save_patch(self, path):
        if self.fail_save:
            raise OSError("disk full")
        path.write_bytes(b"fake parameter snapshot")
        return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def load_patch(self, path):
        assert path.read_bytes() == b"fake parameter snapshot"
        self.active = True
        return {"loaded": True}

    def rollback(self):
        if not self.active:
            raise RuntimeError("No patch")
        self.active = False
        return {"exact_weight_restore": True}


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / "image.jpg"
        self.image.write_bytes(b"trusted fixture supplied after preflight")
        self.backend = FakeBackend()
        self.cases = [{"id": "direct", "image": str(self.image), "question": "What is this?"},
                      {"id": "neighbor", "image": str(self.image), "question": "Unrelated question?"}]
        self.app = Workbench(self.root / "history", self.cases, backend=self.backend)
        self.app.evaluation_dir = self.root / "evaluation"
        self.payload = {"fixture_id": "direct", "question": "What is this?", "target": "new answer"}

    def tearDown(self):
        self.temp.cleanup()

    def test_generated_before_after_saved_patch_and_exact_rollback(self):
        current = self.app.run("generate", self.payload)
        self.assertEqual(current["answer_kind"], "model_generated_unverified")
        self.assertEqual(current["answer"], "old answer")
        result = self.app.run("edit", self.payload)["edit"]
        self.assertEqual(result["before_answer"], "old answer")
        self.assertEqual(result["after_answer"], "new answer")
        self.assertEqual(result["state"], "candidate")
        self.assertTrue(result["preview_exact_restoration"])
        self.assertTrue(result["direct_whole_answer_match"])
        self.assertTrue(result["unrelated_probe"]["changed"])
        self.assertFalse(self.backend.active)
        self.assertIsNone(self.app.active_id)
        self.assertEqual(len(self.app.history()), 1)
        self.assertTrue((self.app.storage / result["id"] / "patch").is_file())
        with self.assertRaisesRegex(WorkbenchError, "approval"):
            self.app.run("activate", {"edit_id": result["id"]})
        self.app.run("activate", {"edit_id": result["id"], "approve": True})
        self.assertTrue(self.backend.active)
        with self.assertRaises(WorkbenchError):
            self.app.run("activate", {"edit_id": result["id"], "approve": True})
        restored = self.app.run("rollback", {})["edit"]
        self.assertTrue(restored["exact_restoration"])
        self.assertEqual(restored["state"], "rolled_back")
        self.assertIsNone(self.app.active_id)

    def test_save_failure_restores_model_and_preserves_failure_record(self):
        self.backend.fail_save = True
        with self.assertRaisesRegex(WorkbenchError, "disk full"):
            self.app.run("edit", self.payload)
        self.assertFalse(self.backend.active)
        self.assertIsNone(self.app.active_id)
        row = self.app.history()[0]
        self.assertEqual(row["state"], "failed")
        self.assertTrue(row["exact_restoration"])
        self.assertEqual(self.app.phase, "ready")

    def test_method_dispatch_and_generated_active_provenance(self):
        row = self.app.run("edit", {**self.payload, "method": "ft_last_layer"})["edit"]
        self.assertEqual(self.backend.last_method, "ft_last_layer")
        self.assertEqual(row["method"], "ft_last_layer")
        self.assertIn("15 steps", row["method_label"])
        self.assertFalse(self.backend.active)
        self.app.run("activate", {"edit_id": row["id"], "approve": True})
        generated = self.app.run("generate", self.payload)
        self.assertEqual(generated["active_edit_id"], row["id"])
        self.assertEqual(generated["active_method"], "ft_last_layer")
        self.app.run("rollback", {})
        self.assertIsNone(self.app.status()["active_method"])

    def test_unknown_method_rejected_before_model_execution(self):
        for invalid in ("anything", {"method": "mend"}, None):
            with self.assertRaisesRegex(WorkbenchError, "supported edit method"):
                self.app.run("edit", {**self.payload, "method": invalid})
        self.assertIsNone(self.backend.last_question)
        self.assertIsNone(self.backend.last_method)
        self.assertFalse(self.app.history())

    def test_serialized_jobs_and_image_paths_cannot_be_supplied(self):
        self.app.job_lock.acquire()
        try:
            with self.assertRaises(WorkbenchError) as caught:
                self.app.run("generate", self.payload)
            self.assertEqual(caught.exception.status, 409)
        finally:
            self.app.job_lock.release()
        with self.assertRaisesRegex(WorkbenchError, "Choose exactly"):
            self.app.run("generate", {"question": "question", "image": "/secret"})
        with self.assertRaisesRegex(WorkbenchError, "Unknown fixture"):
            self.app.run("generate", {"question": "question", "fixture_id": "../../secret"})
        with self.assertRaisesRegex(WorkbenchError, "base64"):
            self.app.run("generate", {"question": "question", "image_base64": "not-base64!"})

    def test_zero_delta_edit_is_cleared_after_export_failure(self):
        self.backend.weight_hashes = lambda: {"matrix": "unchanged"}
        self.backend.fail_save = True
        with self.assertRaises(WorkbenchError):
            self.app.run("edit", self.payload)
        self.assertFalse(self.backend.active)
        self.assertTrue(self.app.history()[0]["exact_restoration"])

    def test_candidate_integrity_and_owned_id_required_for_activation(self):
        row = self.app.run("edit", self.payload)["edit"]
        with self.assertRaisesRegex(WorkbenchError, "saved candidate"):
            self.app.run("activate", {"edit_id": "../../anything", "approve": True})
        patch = self.app.storage / row["id"] / "patch"
        patch.write_bytes(b"tampered")
        with self.assertRaisesRegex(WorkbenchError, "integrity"):
            self.app.run("activate", {"edit_id": row["id"], "approve": True})
        self.assertFalse(self.backend.active)
        self.assertIsNone(self.app.active_id)

    def test_preview_probe_and_target_failure_are_recorded_without_blocking_operator_choice(self):
        row = self.app.run("edit", {**self.payload, "target": "different intended target"})["edit"]
        self.assertFalse(row["direct_whole_answer_match"])
        self.assertEqual(row["unrelated_probe"]["before_answer"], "old answer")
        self.assertEqual(row["unrelated_probe"]["after_answer"], "new answer")
        self.assertTrue(row["unrelated_probe"]["changed"])
        self.assertFalse(self.backend.active)
        active = self.app.run("activate", {"edit_id": row["id"], "approve": True})["edit"]
        self.assertEqual(active["approved_by"], "local_research_operator")
        self.assertEqual(active["state"], "active")

    def test_restart_does_not_silently_reapply_historical_patch(self):
        row = self.app.run("edit", self.payload)["edit"]
        self.app.run("activate", {"edit_id": row["id"], "approve": True})
        fresh = Workbench(self.app.storage, self.cases, backend=FakeBackend())
        self.assertIsNone(fresh.active_id)
        self.assertEqual(fresh.history()[0]["state"], "inactive_after_restart")
        self.assertEqual(fresh.run("generate", self.payload)["answer"], "old answer")

    def test_evaluation_is_read_only_and_absent_report_has_no_numerical_scores(self):
        absent = self.app.evaluation()
        self.assertEqual(absent["execution_status"], "not_executed")
        self.assertNotIn("aggregate", absent)
        # Deliberately tiny API fixture; this is not an actual benchmark result.
        report = {"execution_status": "completed", "aggregate": {
            "provenance": "own_measured_with_unexecuted_failures_in_denominators",
            "scores": {"base": {"direct": {"correct": 1, "denominator": 2}}}}}
        self.app.evaluation_dir.mkdir()
        path = self.app.evaluation_dir / "evaluation.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        original = path.read_bytes()
        self.assertEqual(self.app.evaluation(), report)
        self.assertEqual(path.read_bytes(), original)

    def test_http_requires_session_token_and_serves_escaped_client(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            status = json.load(urlopen(base + "/api/status", timeout=5))
            absent = json.load(urlopen(base + "/api/evaluation", timeout=5))
            self.assertEqual(absent["execution_status"], "not_executed")
            self.assertNotIn("aggregate", absent)
            self.app.evaluation_dir.mkdir()
            local_report = {"execution_status": "completed", "aggregate": {"scores": {}}}
            (self.app.evaluation_dir / "evaluation.json").write_text(json.dumps(local_report), encoding="utf-8")
            self.assertEqual(json.load(urlopen(base + "/api/evaluation", timeout=5)), local_report)
            with urlopen(base + "/api/evaluation-download", timeout=5) as response:
                self.assertIn("attachment", response.headers["Content-Disposition"])
                self.assertEqual(json.load(response), local_report)
            request = Request(base + "/api/generate", json.dumps(self.payload).encode(),
                              {"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()
            request.add_header("X-EditGuard-Token", status["csrf_token"])
            self.assertEqual(json.load(urlopen(request, timeout=5))["answer"], "old answer")
            unknown = Request(base + "/api/edit", json.dumps({**self.payload, "method": "unknown"}).encode(),
                              {"Content-Type": "application/json", "X-EditGuard-Token": status["csrf_token"]})
            with self.assertRaises(HTTPError) as caught:
                urlopen(unknown, timeout=5)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
            self.assertFalse(self.backend.active)
            row = self.app.run("edit", self.payload)["edit"]
            approval = Request(base + "/api/activate", json.dumps({"edit_id": row["id"], "approve": True}).encode(),
                               {"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(approval, timeout=5)
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()
            approval.add_header("X-EditGuard-Token", status["csrf_token"])
            self.assertEqual(json.load(urlopen(approval, timeout=5))["edit"]["state"], "active")
            evidence = json.load(urlopen(base + "/api/export/" + row["id"], timeout=5))
            self.assertEqual(evidence["id"], row["id"])
            with urlopen(base + "/api/patch/" + row["id"], timeout=5) as response:
                self.assertIn("attachment", response.headers["Content-Disposition"])
                self.assertEqual(response.read(), b"fake parameter snapshot")
            html = urlopen(base + "/", timeout=5).read().decode("utf-8")
            self.assertIn("Visual Model Editing Workbench", html)
            script = urlopen(base + "/app.js", timeout=5).read().decode("utf-8")
            self.assertNotIn("innerHTML", script)
            with self.assertRaises(HTTPError) as caught:
                urlopen(base + "/../../ke_repro/bridge.py", timeout=5)
            self.assertEqual(caught.exception.code, 404)
            caught.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
