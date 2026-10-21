"""API checks use tiny synthetic features, not pretrained-model accuracy claims."""

import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("VisionPatch API tests require the isolated ML environment") from exc
from fastapi.testclient import TestClient
from PIL import Image

from visionpatch.app import create_app
from visionpatch.editor import fit_projection_patch, save_projection_patch


class VisionPatchAPITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        dataset = self.root / "data"
        dataset.mkdir()
        Image.new("RGB", (240, 240), "red").save(dataset / "image.png")
        weight = torch.eye(2)
        patch = fit_projection_patch(weight, torch.tensor([[0., 1.]]),
                                     torch.tensor([[1., 0.]]), torch.tensor([[1., 0.]]))
        patch_path = self.root / "patch.pt"
        save_projection_patch(patch, patch_path)
        features = self.root / "features.pt"
        rows = [{"image_id": "source-one", "class_id": "cup", "label": "cup", "split": "probe",
                 "variant": variant, "position": "top", "path": "image.png"}
                for variant in ("clean", "blank", "distractor")]
        torch.save({"features": torch.tensor([[1., 0.], [1., 0.], [0., 1.]]),
                    "text_features": torch.eye(2), "base_weight": weight, "rows": rows,
                    "metadata": {"labels": ["cup", "camera"], "purpose": "synthetic_api_test"}}, features)
        report = self.root / "report.json"
        report.write_text(json.dumps({"purpose": "synthetic_api_test"}), encoding="utf-8")
        self.app = create_app(feature_path=features, patch_path=patch_path, dataset_root=dataset,
                              report_path=report, encoder=lambda image: torch.tensor([[0., 1.]]))
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temporary.cleanup()

    def test_apply_requires_approval_and_rollback_restores_exact_state(self):
        before = self.client.get("/api/status").json()
        self.assertFalse(before["active"])
        self.assertEqual(self.client.post("/api/patch/apply", json={"approved": False}).status_code, 400)
        self.assertTrue(self.client.post("/api/patch/apply", json={"approved": True}).json()["active"])
        attacked = self.client.get("/api/gallery").json()["rows"][2]
        self.assertEqual(attacked["before"]["label"], "camera")
        self.assertEqual(attacked["active"]["label"], "cup")
        self.assertEqual(self.client.post("/api/patch/apply", json={"approved": True}).status_code, 200)
        restored = self.client.post("/api/patch/rollback").json()
        self.assertEqual(restored["projection_sha256"], before["base_sha256"])
        self.assertTrue(torch.equal(self.app.state.demo.weight, self.app.state.demo.base_weight))
        self.assertEqual([event["action"] for event in self.client.get("/api/history").json()["events"]],
                         ["apply", "rollback"])

    def test_gallery_renders_variants_and_upload_uses_encoder(self):
        clean = self.client.get("/api/image/0")
        attacked = self.client.get("/api/image/2")
        self.assertEqual(clean.status_code, 200)
        self.assertEqual(attacked.status_code, 200)
        self.assertEqual(clean.headers["content-type"], "image/png")
        self.assertNotEqual(clean.content, attacked.content)
        self.assertEqual(self.client.get("/api/image/-1").status_code, 404)
        stream = BytesIO()
        Image.new("RGB", (100, 80), "blue").save(stream, format="PNG")
        payload = base64.b64encode(stream.getvalue()).decode("ascii")
        response = self.client.post("/api/predict", json={"image_base64": payload})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["before"]["label"], "camera")
        self.assertIn("not calibrated", response.json()["score_disclaimer"])
        self.assertEqual(self.client.post("/api/predict", json={"image_base64": "not base64"}).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", json={"image_base64": payload, "variant": "unknown"}).status_code, 422)

    def test_missing_artifacts_fail_clearly_without_model_download(self):
        client = TestClient(create_app(feature_path=self.root / "absent.pt"))
        response = client.get("/api/status")
        self.assertEqual(response.status_code, 503)
        self.assertIn("preparation", response.json()["detail"])
        client.close()

    def test_mismatched_parent_patch_is_rejected(self):
        payload = torch.load(self.app.state.demo.feature_path, weights_only=True)
        payload["base_weight"] *= 2
        torch.save(payload, self.app.state.demo.feature_path)
        self.assertEqual(self.client.get("/api/status").status_code, 503)

    def test_measured_report_cannot_describe_a_different_patch(self):
        self.app.state.demo.report_path.write_text(
            json.dumps({"patch_sha256": "0" * 64}), encoding="utf-8")
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 503)
        self.assertIn("does not match", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
