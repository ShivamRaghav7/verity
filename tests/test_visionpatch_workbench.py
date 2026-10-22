"""Exercise real authoring, approval and inference transitions with tiny tensors.

The injected two-dimensional encoder is deliberately simplistic: a blue image
maps to camera and a red image maps to cup. These are lifecycle/API checks, not
evidence about pretrained CLIP accuracy. No model download or GPU is required.
"""

import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Workbench tests require the isolated ML environment") from exc

from fastapi.testclient import TestClient
from PIL import Image

from visionpatch.app import create_app
from visionpatch.editor import fit_projection_patch, save_projection_patch


def image_payload(color, *, image_format="PNG"):
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format=image_format)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def tiny_encoder(image):
    red, _, blue = image.getpixel((0, 0))
    return torch.tensor([[1., 0.]]) if red > blue else torch.tensor([[0., 1.]])


class WorkbenchAPITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "data"
        self.dataset.mkdir()
        Image.new("RGB", (32, 32), "red").save(self.dataset / "cup.png")
        Image.new("RGB", (32, 32), "blue").save(self.dataset / "camera.png")
        self.base = torch.eye(2)
        patch = fit_projection_patch(self.base, torch.tensor([[0., 1.]]),
                                     torch.tensor([[1., 0.]]), torch.eye(2))
        self.patch_path = self.root / "published-patch.pt"
        save_projection_patch(patch, self.patch_path)
        self.feature_path = self.root / "features.pt"
        rows = [{"image_id": label, "class_id": label, "label": label,
                 "split": "retention", "variant": "clean", "position": "top",
                 "path": f"{label}.png"} for label in ("cup", "camera")]
        torch.save({"features": torch.eye(2), "text_features": torch.eye(2),
                    "base_weight": self.base, "rows": rows,
                    "metadata": {"labels": ["cup", "camera"], "purpose": "synthetic_workbench_test"}},
                   self.feature_path)
        self.report_path = self.root / "report.json"
        self.report_path.write_text(json.dumps({"purpose": "synthetic_workbench_test"}), encoding="utf-8")
        self.directory = self.root / "workbench"
        self.source = image_payload("blue")
        self.reference = image_payload("red")
        self.identity = {"actor": "Test reviewer", "reason": "Reviewed synthetic correction"}
        self.app, self.client = self.open_client()

    def open_client(self):
        app = create_app(feature_path=self.feature_path, patch_path=self.patch_path,
                         dataset_root=self.dataset, report_path=self.report_path,
                         encoder=tiny_encoder, workbench_directory=self.directory,
                         workbench_text_reference=(torch.eye(2), torch.eye(2), [0, 1]))
        return app, TestClient(app)

    def tearDown(self):
        self.client.close()
        self.temporary.cleanup()

    def preview(self, **changes):
        payload = {**self.identity, "title": "Repair the supplied blue-image association",
                   "target_word": "camera", "expected_label": "cup",
                   "source": "Synthetic fixture, not a real photograph",
                   "source_image_base64": self.source,
                   "reference_image_base64": self.reference}
        payload.update(changes)
        return self.client.post("/api/workbench/preview", json=payload)

    def state(self):
        response = self.client.get("/api/workbench/status")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["registry"]

    def predict(self):
        response = self.client.post("/api/workbench/predict", json={"image_base64": self.source})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def approve(self, patch_id):
        response = self.client.post(f"/api/workbench/patches/{patch_id}/approve",
                                    json={**self.identity, "approved": True})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def activate(self, patch_id, revision=None):
        if revision is None:
            revision = self.state()["revision"]
        return self.client.post(f"/api/workbench/patches/{patch_id}/activate",
                                json={**self.identity, "expected_revision": revision})

    def test_preview_saves_measured_candidate_without_changing_active_state(self):
        before = self.state()
        prediction = self.predict()
        response = self.preview()
        self.assertEqual(response.status_code, 200, response.text)
        candidate = response.json()
        self.assertEqual(candidate["status"], "candidate")
        self.assertEqual(candidate["preview"]["support"]["before_label"], "camera")
        self.assertEqual(candidate["preview"]["support"]["after_label"], "cup")
        self.assertEqual(candidate["preview"]["metrics"]["clean_accuracy"]["before"],
                         {"correct": 2, "total": 2})
        # This intentionally harsh fixture damages the camera retention case.
        # Reporting that damage is essential, even though the support is repaired.
        self.assertEqual(candidate["preview"]["new_errors"], {"clean": 1, "text": 1})
        self.assertEqual(candidate["preview"]["metrics"]["text_accuracy"]["after"],
                         {"correct": 1, "total": 2})
        self.assertEqual(self.state(), before)
        self.assertEqual(self.predict(), prediction)
        records = self.client.get("/api/workbench/patches").json()["patches"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["patch_id"], candidate["patch_id"])
        self.assertEqual(records[0]["metadata"]["source"], "Synthetic fixture, not a real photograph")

    def test_approval_is_explicit_and_does_not_itself_activate(self):
        candidate = self.preview().json()
        patch_id = candidate["patch_id"]
        self.assertEqual(self.activate(patch_id).status_code, 409)
        path = f"/api/workbench/patches/{patch_id}/approve"
        self.assertEqual(self.client.post(path, json={**self.identity, "approved": False}).status_code, 400)
        self.assertEqual(self.client.post(path, json={**self.identity, "approved": True, "actor": ""}).status_code, 422)
        approved = self.approve(patch_id)
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(self.predict()["current"]["label"], "camera")
        self.assertEqual(self.state()["active_patch_ids"], [])

    def test_activation_changes_real_inference_but_keeps_original_reference(self):
        patch_id = self.preview().json()["patch_id"]
        self.approve(patch_id)
        response = self.activate(patch_id)
        self.assertEqual(response.status_code, 200, response.text)
        prediction = self.predict()
        self.assertEqual(prediction["before"]["label"], "camera")
        self.assertEqual(prediction["current"]["label"], "cup")
        self.assertEqual(prediction["registry"]["active_patch_ids"], [patch_id])
        self.assertEqual(prediction["registry"]["revision"], 1)
        self.assertTrue(torch.equal(self.app.state.demo.base_weight, self.base))
        self.assertTrue(torch.equal(self.app.state.demo.weight, self.base))

    def test_stale_revision_and_competing_parent_cannot_replace_current_patch(self):
        first = self.preview().json()["patch_id"]
        competing = self.preview(title="Competing base-state correction").json()["patch_id"]
        self.approve(first)
        self.approve(competing)
        self.assertEqual(self.activate(first, revision=0).status_code, 200)
        active = self.state()
        self.assertEqual(self.activate(competing, revision=0).status_code, 409)
        # A fresh revision cannot override a mismatched parent, either.
        self.assertEqual(self.activate(competing, revision=active["revision"]).status_code, 409)
        self.assertEqual(self.state(), active)
        self.assertEqual(self.predict()["current"]["label"], "cup")

    def test_rollback_restores_exact_parameters_and_prediction_with_audit_trail(self):
        before = self.state()
        prediction_before = self.predict()
        patch_id = self.preview().json()["patch_id"]
        self.approve(patch_id)
        self.assertEqual(self.activate(patch_id).status_code, 200)
        stale = self.client.post("/api/workbench/rollback", json={**self.identity, "expected_revision": 0})
        self.assertEqual(stale.status_code, 409)
        response = self.client.post("/api/workbench/rollback", json={**self.identity,
                                    "expected_revision": self.state()["revision"], "to_patch_id": "base"})
        self.assertEqual(response.status_code, 200, response.text)
        after = self.state()
        self.assertEqual(after["current_sha256"], before["current_sha256"])
        self.assertEqual(after["active_patch_ids"], [])
        self.assertTrue(torch.equal(self.app.state.workbench.registry.get_current_weight(), self.base))
        self.assertEqual(self.predict()["current"], prediction_before["current"])
        actions = [event["action"] for event in self.client.get("/api/workbench/history").json()["events"]]
        self.assertEqual(actions, ["initialize", "propose", "approve", "activate", "rollback"])

    def test_restart_recovers_active_projection_candidates_and_audit_history(self):
        patch_id = self.preview().json()["patch_id"]
        self.approve(patch_id)
        self.assertEqual(self.activate(patch_id).status_code, 200)
        before = self.state()
        events = self.client.get("/api/workbench/history").json()
        app, restarted = self.open_client()
        try:
            state = restarted.get("/api/workbench/status").json()["registry"]
            self.assertEqual(state, before)
            self.assertEqual(restarted.get("/api/workbench/history").json(), events)
            prediction = restarted.post("/api/workbench/predict", json={"image_base64": self.source}).json()
            self.assertEqual(prediction["current"]["label"], "cup")
            self.assertEqual(len(restarted.get("/api/workbench/patches").json()["patches"]), 1)
            self.assertTrue(torch.equal(app.state.demo.base_weight, self.base))
        finally:
            restarted.close()

    def test_declared_expected_label_is_scoring_metadata_not_edit_routing(self):
        cup = self.preview(expected_label="cup").json()
        camera = self.preview(expected_label="camera").json()
        self.assertEqual(cup["record"]["after_sha256"], camera["record"]["after_sha256"])
        self.assertEqual(camera["preview"]["support"]["after_label"], "cup")
        self.assertTrue(any("fails the declared correction" in text for text in camera["warnings"]))
        self.assertEqual(self.state()["active_patch_ids"], [])

    def test_malformed_unsupported_or_identical_images_never_publish_candidate(self):
        self.state()
        for payload in ("%%%invalid-base64%%%", "data:image/png,raw-text",
                        base64.b64encode(b"this is not an image").decode("ascii"),
                        image_payload("red", image_format="BMP")):
            with self.subTest(payload=payload[:35]):
                self.assertEqual(self.preview(source_image_base64=payload).status_code, 400)
                self.assertEqual(self.client.post("/api/workbench/predict", json={"image_base64": payload}).status_code, 400)
        self.assertEqual(self.preview(reference_image_base64=self.source).status_code, 400)
        self.assertEqual(self.preview(expected_label="unsupported object").status_code, 400)
        self.assertEqual(self.client.get("/api/workbench/patches").json()["patches"], [])
        self.assertEqual(self.state()["revision"], 0)

    def test_provided_examples_use_declared_support_sources_and_reject_unknown_files(self):
        payload = torch.load(self.feature_path, weights_only=True)
        for image_id, label in [("cup/image_0011.jpg", "cup"), ("camera/image_0050.jpg", "camera")]:
            payload["rows"].append({"image_id": image_id, "class_id": label, "label": label,
                                    "split": "support", "variant": "clean", "position": "bottom",
                                    "path": f"{label}.png"})
        payload["features"] = torch.cat([payload["features"], torch.eye(2)])
        torch.save(payload, self.feature_path)
        response = self.client.get("/api/workbench/examples")
        self.assertEqual(response.status_code, 200, response.text)
        examples = response.json()["examples"]
        self.assertEqual(len(examples), 3)
        for example in examples:
            self.assertIn("training-support", example["source"])
            source = self.client.get(example["source_url"])
            reference = self.client.get(example["reference_url"])
            self.assertEqual(source.status_code, 200)
            self.assertEqual(reference.status_code, 200)
            self.assertEqual(source.headers["content-type"], "image/png")
            self.assertNotEqual(source.content, reference.content)
        self.assertEqual(self.client.get("/api/workbench/examples/unknown/source").status_code, 404)
        self.assertEqual(self.client.get("/api/workbench/examples/camera-cup/unknown").status_code, 404)

    def test_withdrawal_is_terminal_and_does_not_remove_evidence(self):
        patch_id = self.preview().json()["patch_id"]
        self.approve(patch_id)
        response = self.client.post(f"/api/workbench/patches/{patch_id}/withdraw", json=self.identity)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "withdrawn")
        refused = self.client.post(f"/api/workbench/patches/{patch_id}/approve",
                                   json={**self.identity, "approved": True})
        self.assertEqual(refused.status_code, 409)
        self.assertEqual(self.activate(patch_id).status_code, 409)
        records = self.client.get("/api/workbench/patches").json()["patches"]
        self.assertEqual(len(records), 1)
        self.assertIsNotNone(records[0]["preview"])
        self.assertEqual(self.state()["active_patch_ids"], [])

    def test_current_site_has_one_studio_and_does_not_publish_archived_interface(self):
        home = self.client.get("/")
        alias = self.client.get("/workbench")
        self.assertEqual(home.status_code, 200)
        self.assertEqual(home.content, alias.content)
        self.assertIn("Correction studio", home.text)
        self.assertNotIn("Pilot experiment", home.text)
        for path in ("/api/status", "/api/gallery", "/api/history", "/api/predict",
                     "/api/patch/apply", "/api/figures/accuracy.png",
                     "/static/index.html", "/static/app.js", "/static/style.css"):
            response = self.client.post(path, json={}) if path in {"/api/predict", "/api/patch/apply"} else self.client.get(path)
            self.assertEqual(response.status_code, 404, path)
        public_paths = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/api/workbench/predict", public_paths)
        self.assertNotIn("/api/patch/apply", public_paths)


if __name__ == "__main__":
    unittest.main()
