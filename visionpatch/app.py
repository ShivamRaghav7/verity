"""Loopback demonstration API for a measured visual projection patch.

Gallery inference uses cached frozen-encoder features. Uploads use the real,
pinned CLIP encoder, loaded lazily from its local cache. This is not an
authenticated production service; launch on 127.0.0.1 only.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
import threading
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as functional

from .data import CLASSES, file_hash, render_variant
from .editor import (
    apply_projection_patch_, load_projection_patch, projection_sha256,
    rollback_projection_patch_,
)


ROOT = Path(__file__).resolve().parent.parent
SCORE_DISCLAIMER = "Cosine similarity scores rank the fixed candidates; they are not calibrated probabilities."
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class ApplyRequest(BaseModel):
    approved: bool = False
    reason: str = Field(default="Approved local demonstration patch", max_length=1000)


class PredictRequest(BaseModel):
    image_base64: str = Field(min_length=1, max_length=7_000_512)
    variant: Literal["clean", "blank", "distractor"] = "clean"
    position: Literal["top", "bottom"] = "top"


class DemoState:
    def __init__(self, feature_path: Path, patch_path: Path, dataset_root: Path,
                 report_path: Path, encoder=None):
        self.feature_path = feature_path
        self.patch_path = patch_path
        self.dataset_root = dataset_root.resolve()
        self.report_path = report_path
        self.lock = threading.RLock()
        self.loaded = False
        self.encoder = encoder
        self.model = None
        self.events = []

    def ensure_loaded(self):
        if self.loaded:
            return
        try:
            payload = torch.load(self.feature_path, map_location="cpu", weights_only=True)
            patch = load_projection_patch(self.patch_path)
        except FileNotFoundError as exc:
            raise HTTPException(503, "Development artifacts are absent. Run data preparation and the pilot first.") from exc
        except (ValueError, RuntimeError, TypeError, KeyError, pickle.UnpicklingError, EOFError) as exc:
            raise HTTPException(503, "Development artifacts failed validation; regenerate or restore the recorded run.") from exc
        if not isinstance(payload, dict):
            raise HTTPException(503, "Feature artifact must contain a dictionary.")
        try:
            features = payload["features"].float()
            texts = payload["text_features"].float()
            weight = payload["base_weight"].clone()
            rows = payload["rows"]
            metadata = payload.get("metadata", {})
            labels = metadata.get("labels", list(CLASSES.values()))
            if isinstance(labels, dict):
                labels = list(labels.values())
            if (features.ndim != 2 or texts.ndim != 2 or weight.ndim != 2
                    or features.shape[0] != len(rows) or features.shape[1] != weight.shape[1]
                    or texts.shape[1] != weight.shape[0] or len(labels) != len(texts)
                    or not all(isinstance(label, str) for label in labels)
                    or not all(torch.isfinite(tensor).all() for tensor in (features, texts, weight))):
                raise ValueError("Feature dimensions or values are invalid")
            if projection_sha256(weight) != patch.before_sha256:
                raise ValueError("Feature projection differs from patch parent")
            json.dumps({"metadata": metadata, "rows": rows}, allow_nan=False)
            for row in rows:
                if row["variant"] not in {"clean", "blank", "distractor"}:
                    raise ValueError("Unknown variant")
                target = (self.dataset_root / row["path"]).resolve()
                if not target.is_relative_to(self.dataset_root):
                    raise ValueError("Image path escapes dataset root")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise HTTPException(503, "Incompatible feature or patch artifact.") from exc
        self.features, self.texts = features, functional.normalize(texts, dim=-1)
        self.base_weight, self.weight = weight.clone(), weight.clone()
        self.rows, self.metadata, self.labels, self.patch = rows, metadata, labels, patch
        self.active = False
        self.loaded = True

    def report(self):
        if not self.report_path.exists():
            return None
        try:
            return json.loads(self.report_path.read_text(encoding="utf-8"),
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (OSError, ValueError) as exc:
            raise HTTPException(503, "Pilot report is unreadable or contains non-finite values.") from exc

    def status(self):
        self.ensure_loaded()
        recorded = self.report()
        if recorded and "patch_sha256" in recorded and file_hash(self.patch_path) != recorded["patch_sha256"]:
            raise HTTPException(503, "The measured report does not match this patch artifact.")
        return {"ready": True, "active": self.active, "labels": self.labels,
                "base_sha256": self.patch.before_sha256,
                "projection_sha256": projection_sha256(self.weight),
                "metadata": self.metadata, "report": recorded,
                "final_report": json.loads((ROOT / "reports/visionpatch/final-v1.json").read_text(encoding="utf-8"))
                if (ROOT / "reports/visionpatch/final-v1.json").exists() else None,
                "encoder_device": str(self.model.device) if self.model is not None else None,
                "history_persistence": "in_memory_only; resets when this demo process restarts",
                "disclaimer": "Local development demonstration; fixed candidate labels; no final-test or deployment reliability claim.",
                "score_disclaimer": SCORE_DISCLAIMER}

    def predictions(self, features, weight):
        scores = functional.normalize(features.float() @ weight.float().T, dim=-1) @ self.texts.T
        result = []
        for sample in scores:
            result.append({"label": self.labels[int(sample.argmax())],
                           "scores": [{"label": label, "score": float(score)}
                                      for label, score in zip(self.labels, sample)]})
        return result

    def encode(self, image):
        if self.encoder is not None:
            return self.encoder(image)
        from .model import image_tensor, load_clip
        if self.model is None:
            device = os.environ.get("VISIONPATCH_DEVICE", "cpu")
            if device not in {"cpu", "cuda"}:
                raise ValueError("VISIONPATCH_DEVICE must be cpu or cuda")
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA explicitly requested but unavailable")
            model, _, _ = load_clip(device=device, allow_download=False)
            if projection_sha256(model.visual_projection.weight) != self.patch.before_sha256:
                raise RuntimeError("Pinned encoder projection does not match this patch's parent")
            self.model = model
        with torch.inference_mode():
            pixels = image_tensor([image]).to(self.model.device)
            return self.model.vision_model(pixel_values=pixels).pooler_output.float().cpu()


def create_app(*, feature_path=ROOT / "artifacts/visionpatch/development-v1/features.pt",
               patch_path=ROOT / "artifacts/visionpatch/development-v1/projection-patch.pt",
               dataset_root=ROOT / "data/generated/visionpatch/development-v1",
               report_path=None, encoder=None):
    if report_path is None:
        adjacent = Path(patch_path).parent / "run-report.json"
        report_path = adjacent if adjacent.exists() else ROOT / "reports/visionpatch/pilot-v1.json"
    app = FastAPI(title="VisionPatch local demonstration", version="0.1.0")
    state = DemoState(Path(feature_path), Path(patch_path), Path(dataset_root), Path(report_path), encoder)
    app.state.demo = state

    @app.get("/api/status")
    def status():
        with state.lock:
            return state.status()

    @app.get("/api/gallery")
    def gallery():
        with state.lock:
            state.ensure_loaded()
            before = state.predictions(state.features, state.base_weight)
            active = state.predictions(state.features, state.weight)
            rows = [{**row, "index": index, "image_url": f"/api/image/{index}",
                     "before": before[index], "active": active[index]}
                    for index, row in enumerate(state.rows) if row["split"] == "probe"]
            return {"active": state.active, "projection_sha256": projection_sha256(state.weight),
                    "rows": rows, "score_disclaimer": SCORE_DISCLAIMER}

    @app.get("/api/image/{rowindex}")
    def image(rowindex: int):
        with state.lock:
            state.ensure_loaded()
            if not 0 <= rowindex < len(state.rows):
                raise HTTPException(404, "Image index is outside the prepared gallery.")
            row = state.rows[rowindex]
            path = (state.dataset_root / row["path"]).resolve()
            if not path.is_relative_to(state.dataset_root):
                raise HTTPException(503, "Invalid prepared image path.")
            try:
                with Image.open(path) as source:
                    rendered, _ = render_variant(source, row["variant"], position=row["position"],
                                                 word=state.metadata.get("distractor_word", "camera"))
            except (OSError, ValueError) as exc:
                raise HTTPException(503, "Prepared source image is absent or unreadable.") from exc
            output = BytesIO()
            rendered.save(output, format="PNG")
            return Response(output.getvalue(), media_type="image/png")

    def transition(action, reason):
        with state.lock:
            state.ensure_loaded()
            desired = action == "apply"
            if state.active != desired:
                if desired:
                    apply_projection_patch_(state.weight, state.patch)
                else:
                    rollback_projection_patch_(state.weight, state.patch)
                state.active = desired
                state.events.append({"action": action, "reason": reason,
                                     "at": datetime.now(timezone.utc).isoformat(),
                                     "projection_sha256": projection_sha256(state.weight)})
            return state.status()

    @app.post("/api/patch/apply")
    def apply(body: ApplyRequest):
        if not body.approved:
            raise HTTPException(400, "Set approved=true after explicitly approving the local patch.")
        return transition("apply", body.reason)

    @app.post("/api/patch/rollback")
    def rollback():
        return transition("rollback", "User requested exact parent restoration")

    @app.get("/api/history")
    def history():
        with state.lock:
            return {"events": list(state.events), "persistence": "in_memory_only"}

    @app.get("/api/figures/{name}")
    def figure(name: str):
        if name not in {"accuracy.png", "paired-gains.png", "categories.png"}:
            raise HTTPException(404, "Unknown scientific figure")
        path = ROOT / "output/visionpatch/figures" / name
        if not path.exists():
            raise HTTPException(404, "Generate the measured result figures first")
        return FileResponse(path)

    @app.post("/api/predict")
    def predict(body: PredictRequest):
        raw = body.image_base64
        if raw.startswith("data:"):
            header, separator, raw = raw.partition(",")
            if not separator or ";base64" not in header:
                raise HTTPException(400, "Expected a base64 image data URL.")
        try:
            payload = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(400, "Invalid base64 image.") from exc
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Decoded image exceeds the 5 MiB upload limit.")
        try:
            with Image.open(BytesIO(payload)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Unsupported image format")
                if source.width * source.height > 20_000_000:
                    raise ValueError("Image exceeds the 20 megapixel decode limit")
                image, _ = render_variant(ImageOps.exif_transpose(source), body.variant,
                                          position=body.position, word="camera")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, "Use a valid JPEG, PNG, or WebP image of at most 20 megapixels.") from exc
        with state.lock:
            state.ensure_loaded()
            started = time.perf_counter()
            try:
                features = state.encode(image)
                before = state.predictions(features, state.base_weight)[0]
                active = state.predictions(features, state.weight)[0]
            except (OSError, RuntimeError, ValueError) as exc:
                raise HTTPException(503, "Pinned local CLIP inference is unavailable; check the local model cache and artifact parent.") from exc
            return {"variant": body.variant, "position": body.position,
                    "before": before, "active": active, "active_patch": state.active,
                    "inference_seconds": time.perf_counter() - started,
                    "device": str(state.model.device) if state.model is not None else "test_encoder",
                    "score_disclaimer": SCORE_DISCLAIMER,
                    "ground_truth": "No ground truth supplied; correctness is not established for this upload."}

    static = Path(__file__).resolve().parent / "static"
    if static.exists():
        app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        path = static / "index.html"
        if not path.exists():
            raise HTTPException(503, "Demo interface is not installed; API documentation is at /docs.")
        return FileResponse(path)

    return app


app = create_app()
