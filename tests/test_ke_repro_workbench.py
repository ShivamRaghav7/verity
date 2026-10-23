"""Tiny application contract tests; the fake is never model-quality evidence."""
import json
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

    def weight_hashes(self):
        return {"matrix": "edited" if self.active else "original"}

    def generate(self, image, question, max_new_tokens):
        self.last_question = question
        assert max_new_tokens == 24
        return "new answer" if self.active else "old answer"

    def edit(self, image, question, target, target_alignment):
        assert target_alignment == "context"
        self.active = True
        return {"changed_parameters": ["matrix"]}

    def save_patch(self, path):
        if self.fail_save:
            raise OSError("disk full")
        path.write_bytes(b"fake parameter snapshot")
        return {"bytes": path.stat().st_size, "sha256": "fake"}

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
        self.cases = [{"id": "direct", "image": str(self.image), "question": "What is this?"}]
        self.app = Workbench(self.root / "history", self.cases, backend=self.backend)
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
        self.assertEqual(self.backend.last_question, "Question: What is this? Answer:")
        self.assertEqual(len(self.app.history()), 1)
        self.assertTrue((self.app.storage / result["id"] / "patch").is_file())
        with self.assertRaises(WorkbenchError):
            self.app.run("edit", self.payload)
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

    def test_restart_does_not_silently_reapply_historical_patch(self):
        self.app.run("edit", self.payload)
        fresh = Workbench(self.app.storage, self.cases, backend=FakeBackend())
        self.assertIsNone(fresh.active_id)
        self.assertEqual(fresh.history()[0]["state"], "inactive_after_restart")
        self.assertEqual(fresh.run("generate", self.payload)["answer"], "old answer")

    def test_http_requires_session_token_and_serves_escaped_client(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            status = json.load(urlopen(base + "/api/status", timeout=5))
            request = Request(base + "/api/generate", json.dumps(self.payload).encode(),
                              {"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()
            request.add_header("X-EditGuard-Token", status["csrf_token"])
            self.assertEqual(json.load(urlopen(request, timeout=5))["answer"], "old answer")
            row = self.app.run("edit", self.payload)["edit"]
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
