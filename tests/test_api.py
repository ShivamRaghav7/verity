import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from editguard.api import create_app


class APITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.client = TestClient(create_app(Path(self.temporary.name) / "test.sqlite3", "test-token"))
        self.client.__enter__()
        self.headers = {"X-Admin-Token": "test-token"}
        self.payload = {"fact_id": "nimbus-basic-v2-devices", "original_value": "5",
                        "corrected_value": "3", "source": "Fictional release EG-1",
                        "reason": "Approved policy change", "effective_from": "2026-10-01"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temporary.cleanup()

    def ask(self, question="How many devices does Basic support?", **scope):
        return self.client.post("/api/ask", json={"question": question, "entity_id": "nimbus",
                                                "as_of": "2026-10-11", **scope})

    def test_admin_boundary_and_full_workflow(self):
        self.assertEqual(self.client.post("/api/edits", json=self.payload).status_code, 401)
        self.assertEqual(self.ask().json()["answer"], "Nimbus Basic v2 supports up to 5 devices.")
        draft = self.client.post("/api/edits", json=self.payload, headers=self.headers)
        self.assertEqual(draft.status_code, 201)
        edit_id = draft.json()["edit_id"]
        self.assertEqual(self.ask().json()["route"], "initial_catalogue")
        self.assertEqual(self.client.post(f"/api/edits/{edit_id}/approve").status_code, 401)
        self.assertEqual(self.client.post(f"/api/edits/{edit_id}/approve", headers=self.headers).status_code, 200)
        self.assertEqual(self.ask().json()["answer"], "Nimbus Basic v2 supports up to 3 devices.")
        self.assertEqual(self.client.post(f"/api/edits/{edit_id}/withdraw", headers=self.headers).status_code, 200)
        self.assertEqual(self.ask().json()["answer"], "Nimbus Basic v2 supports up to 5 devices.")
        self.assertGreater(len(self.client.get("/api/edits").json()["events"]), 2)

    def test_learned_mode_is_unavailable_and_not_mislabeled(self):
        self.assertEqual(self.ask(mode="learned").status_code, 503)
        self.assertEqual(self.ask().json()["answer_kind"], "deterministic_structured")
        self.assertEqual(self.client.get("/api/health").json()["models"]["scope"], "not_trained")

    def test_unknown_and_conflicting_scope_clarifies(self):
        for question in ["What is the warranty?", "Compare Basic and Pro device limits", "Nimbus and Nimbus Mini Basic devices"]:
            self.assertEqual(self.ask(question).json()["status"], "clarification", question)
        self.assertEqual(self.ask("Basic v1 device limit", version="v2").json()["status"], "clarification")
        self.assertEqual(self.client.post("/api/ask", json={"question": "What did Basic previously support on devices?", "entity_id": "nimbus"}).json()["status"], "clarification")
        for question in ["Can I connect more than three devices on Basic?", "Can I connect fewer than four devices on Basic?", "Can I connect four devices on Enterprise?", "Can I connect four or five devices on Basic?"]:
            self.assertEqual(self.ask(question).json()["status"], "clarification", question)

    def test_similar_product_and_historical_date_are_scoped(self):
        response = self.client.post("/api/ask", json={"question": "Nimbus Mini Basic device limit", "as_of": "2026-10-11"}).json()
        self.assertEqual(response["answer"], "Nimbus Mini Basic v2 supports up to 2 devices.")
        draft = self.client.post("/api/edits", json=self.payload, headers=self.headers).json()
        self.client.post(f"/api/edits/{draft['edit_id']}/approve", headers=self.headers)
        self.assertEqual(self.ask(as_of="2026-09-01").json()["answer"], "Nimbus Basic v2 supports up to 5 devices.")

    def test_regression_does_not_mutate_live_state(self):
        before = self.client.get("/api/edits").json()
        report = self.client.get("/api/evaluation").json()
        self.assertEqual(report["status"], "passed", report)
        self.assertIsNone(report["ml_results"])
        self.assertEqual(self.client.get("/api/edits").json(), before)

    def test_invalid_values_dates_and_authority_are_rejected(self):
        for value in ["nan", "-1", "three", "3.5"]:
            result = self.client.post("/api/edits", headers=self.headers, json={**self.payload, "corrected_value": value})
            self.assertIn(result.status_code, [400, 422], value)
        self.assertEqual(self.ask(as_of="2026-99-99").status_code, 422)
        self.assertEqual(self.client.post("/api/ask", json={"question": "Basic devices", "approved_correction": self.payload}).status_code, 422)
