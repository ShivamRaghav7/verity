"""Boundary checks for the isolated actual-checkpoint verification command."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import verify_learned_workflow as workflow


class WorkflowBoundaryTests(unittest.TestCase):
    def test_live_and_external_database_paths_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(workflow, "ROOT", root):
                for database in (root / "runtime/editguard.sqlite3", root / "other.sqlite3"):
                    with self.assertRaises(ValueError):
                        workflow.protect_paths(database, root / "report.json")
                database, output = workflow.protect_paths(root / "runtime/isolated.sqlite3", root / "report.json")
                self.assertEqual(database, (root / "runtime/isolated.sqlite3").resolve())
                self.assertEqual(output, (root / "report.json").resolve())
                with self.assertRaises(ValueError):
                    workflow.protect_paths(database, database)

    def test_existing_fixture_report_and_sqlite_sidecar_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runtime").mkdir()
            database, output = root / "runtime/isolated.sqlite3", root / "report.json"
            with patch.object(workflow, "ROOT", root):
                for occupied in (database, Path(str(database) + "-wal"), Path(str(database) + "-shm"), output):
                    occupied.write_text("earlier evidence", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        workflow.protect_paths(database, output)
                    self.assertEqual(occupied.read_text(encoding="utf-8"), "earlier evidence")
                    occupied.unlink()

    def test_request_cap_prevents_any_client_call(self):
        class RefusingClient:
            def request(self, *args, **kwargs):
                raise AssertionError("No client request is permitted after the cap.")

        report = {"checks": [], "requests": [], "model_answers": []}
        with patch.object(workflow, "MAX_REQUESTS", 0):
            with self.assertRaisesRegex(RuntimeError, "request cap"):
                workflow.verify_workflow(RefusingClient(), report, "test-token")
        self.assertEqual(report["requests"], [])

    def test_ineligible_checkpoint_precondition_never_mutates_fixture(self):
        class Response:
            status_code = 200

            def __init__(self, body):
                self.body = body

            def json(self):
                return self.body

        class IneligibleClient:
            def __init__(self):
                self.methods = []

            def request(self, method, path, **kwargs):
                self.methods.append(method)
                if path == "/api/health":
                    return Response({"revision": 1, "models": {"scope": "not_configured", "answer": "not_configured"},
                                     "modes": {"learned": {"status": "unavailable"}}})
                return Response({"revision": 1, "edits": []})

        client = IneligibleClient()
        report = {"checks": [], "requests": [], "model_answers": []}
        with self.assertRaisesRegex(RuntimeError, "no mutation attempted"):
            workflow.verify_workflow(client, report, "test-token")
        self.assertEqual(client.methods, ["GET", "GET"])
        self.assertEqual(report["model_answers"], [])


if __name__ == "__main__":
    unittest.main()
