"""Create, inspect and persist actual projection edits from administrator pairs.

The registry is authoritative for the projection used by this workbench's forward
pass. The published v1 demonstration and its frozen measurements remain separate.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as functional

from .data import CLASSES, file_hash
from .editor import fit_projection_patch, projection_sha256
from .model import MODEL_ID, MODEL_REVISION
from .registry import PatchRegistry, RegistryConflictError, RegistryIntegrityError
from .study_protocol import render_study_variant, render_text_card, resolve_fonts

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 5 * 1024 * 1024


class Identity(BaseModel):
    actor: str = Field(min_length=1, max_length=80)
    reason: str = Field(min_length=1, max_length=1000)


class PreviewRequest(Identity):
    title: str = Field(min_length=1, max_length=200)
    target_word: str = Field(min_length=1, max_length=80)
    expected_label: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=2000)
    source_image_base64: str = Field(min_length=1, max_length=7_000_512)
    reference_image_base64: str = Field(min_length=1, max_length=7_000_512)


class Approval(Identity):
    approved: bool = False


class Activation(Identity):
    expected_revision: int = Field(ge=0)


class Rollback(Activation):
    to_patch_id: str | None = Field(default=None, max_length=80)


class PredictImage(BaseModel):
    image_base64: str = Field(min_length=1, max_length=7_000_512)


def decode_image(raw: str):
    if raw.startswith("data:"):
        header, separator, raw = raw.partition(",")
        if not separator or ";base64" not in header:
            raise HTTPException(400, "Expected a base64 image data URL")
    try:
        payload = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(400, "Invalid base64 image") from exc
    if len(payload) > MAX_BYTES:
        raise HTTPException(413, "Each decoded image must be at most 5 MiB")
    try:
        with Image.open(BytesIO(payload)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"} or source.width * source.height > 20_000_000:
                raise ValueError("Unsupported image format or dimensions")
            image = ImageOps.fit(ImageOps.exif_transpose(source).convert("RGB"), (224, 224),
                                 method=Image.Resampling.BICUBIC)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "Use a valid JPEG, PNG or WebP of at most 20 megapixels") from exc
    return image, hashlib.sha256(payload).hexdigest()


def weighted_references(objects, text, weight=1.0):
    """Rows whose second moment is mean(objects) + weight * mean(text)."""
    count = len(objects) + len(text)
    return torch.cat([objects * (count / len(objects)) ** .5,
                      text * (weight * count / len(text)) ** .5])


class Workbench:
    def __init__(self, demo, directory: Path, *, text_reference=None):
        self.demo, self.directory = demo, Path(directory)
        self.lock = threading.RLock()
        self.registry = None
        self.text_reference = text_reference

    def ensure(self):
        self.demo.ensure_loaded()
        if self.registry is None:
            self.registry = PatchRegistry(self.directory / "workbench.sqlite3", self.demo.base_weight,
                                          self.directory / "patches")
        indexes = [i for i, row in enumerate(self.demo.rows)
                   if row["split"] == "retention" and row["variant"] == "clean"]
        if not indexes:
            raise ValueError("A separate clean retention reference set is required")
        self.object_features = self.demo.features[indexes]
        self.object_truth = [self.demo.labels.index(self.demo.rows[i]["label"]) for i in indexes]

    def protected_text(self):
        if self.text_reference is not None:
            return self.text_reference
        fonts = resolve_fonts()
        key = {"base_projection": projection_sha256(self.demo.base_weight),
               "base_features_sha256": file_hash(self.demo.feature_path),
               "font_sha256": file_hash(fonts["train_sans"]),
               "protocol_sha256": file_hash(Path(__file__).with_name("study_protocol.py")),
               "model_revision": MODEL_REVISION, "templates": [0, 1, 2, 3],
               "labels": self.demo.labels, "text_prompt": 'a photo of the word "{label}".'}
        path = self.directory / "text-references.pt"
        if path.exists():
            cache = torch.load(path, weights_only=True, map_location="cpu")
            if cache["key"] != key:
                raise ValueError("Text-reference cache differs from this runtime; preserve it and use a new workbench directory")
            self.text_reference = (cache["features"], cache["texts"], cache["truth"])
            return self.text_reference
        features, truth = [], []
        for template in key["templates"]:
            for index, word in enumerate(self.demo.labels):
                card = render_text_card(word, template, font_path=fonts["train_sans"], position="bottom")
                features.append(self.demo.encode(card).detach().float().cpu())
                truth.append(index)
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        model = self.demo.model
        tokens = tokenizer([key["text_prompt"].format(label=label) for label in self.demo.labels],
                           padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            pooled = model.text_model(**tokens).pooler_output
            texts = functional.normalize(model.text_projection(pooled), dim=-1).cpu()
        cache = {"key": key, "features": torch.cat(features), "texts": texts, "truth": truth}
        temporary = path.with_suffix(".part")
        torch.save(cache, temporary)
        temporary.replace(path)
        self.text_reference = (cache["features"], texts, truth)
        return self.text_reference

    @staticmethod
    def indexes(features, weight, texts):
        return (functional.normalize(features @ weight.T, dim=-1) @ texts.T).argmax(-1).tolist()

    @staticmethod
    def counter(predictions, truth):
        return {"correct": sum(p == t for p, t in zip(predictions, truth)), "total": len(truth)}

    def measures(self, weight, *, text=None):
        clean = self.indexes(self.object_features, weight, self.demo.texts)
        metrics = {"clean_accuracy": self.counter(clean, self.object_truth)}
        if text is not None:
            features, texts, truth = text
            metrics["text_accuracy"] = self.counter(self.indexes(features, weight, texts), truth)
        return metrics

    def flatten(self, record):
        metadata = record.get("metadata", {})
        return {**record, "title": metadata.get("title", record["patch_id"]),
                "parent_sha256": record["before_sha256"],
                "preview": metadata.get("preview"), "warnings": metadata.get("warnings", [])}

    def study(self):
        path = ROOT / "reports/visionpatch/study-v2/final.json"
        if not path.exists():
            path = ROOT / "reports/visionpatch/study-v2/development.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "status": "not_run", "available": False}

    def status(self):
        self.ensure()
        weight = self.registry.get_current_weight()
        return {"registry": self.registry.state(), "labels": self.demo.labels,
                "patches": [self.flatten(record) for record in self.registry.list_patches()],
                "reference_count": len(self.object_truth),
                "reference_metrics": self.measures(weight, text=self.text_reference),
                "study": self.study(), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "scope": "Local administrator experiment; eight fixed object candidates. Reference previews are development evidence.",
                "identity": "Actor identifiers are supplied by the local administrator, not authenticated identities"}

    def preview(self, body: PreviewRequest):
        self.ensure()
        label = next((label for label in self.demo.labels if label.casefold() == body.expected_label.casefold()), None)
        if label is None and body.expected_label in CLASSES:
            label = CLASSES[body.expected_label]
        if label not in self.demo.labels:
            raise ValueError("Expected label must belong to the fixed candidate vocabulary")
        source, source_hash = decode_image(body.source_image_base64)
        reference, reference_hash = decode_image(body.reference_image_base64)
        if source_hash == reference_hash:
            raise ValueError("Supply distinct source and matched reference images")
        current = self.registry.get_current_weight()
        parent_revision = self.registry.state()["revision"]
        started = time.perf_counter()
        keys = self.demo.encode(source).detach().cpu().float()
        reference_features = self.demo.encode(reference).detach().cpu().float()
        text = self.protected_text()
        patch = fit_projection_patch(current, keys, reference_features @ current.T,
                                     weighted_references(self.object_features, text[0]),
                                     covariance_ridge=.001, fit_ridge=.000001,
                                     metadata={"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                                               "source_sha256": source_hash, "reference_sha256": reference_hash,
                                               "parent_revision": parent_revision,
                                               "objective": "paired image representation fit with object and legitimate text retention"})
        edited = patch.edited_weight()
        before = self.demo.predictions(keys, current)[0]
        after = self.demo.predictions(keys, edited)[0]
        reference_prediction = self.demo.predictions(reference_features, current)[0]
        before_clean = self.indexes(self.object_features, current, self.demo.texts)
        after_clean = self.indexes(self.object_features, edited, self.demo.texts)
        before_text = self.indexes(text[0], current, text[1])
        after_text = self.indexes(text[0], edited, text[1])
        new_clean = sum(b == t and a != t for b, a, t in zip(before_clean, after_clean, self.object_truth))
        new_text = sum(b == t and a != t for b, a, t in zip(before_text, after_text, text[2]))
        warnings = ["One supplied pair fits this candidate; unseen-image generalization is not established by its support prediction.",
                    "Reference-set metrics are development evidence; this candidate has no independently sealed final evaluation."]
        if reference_prediction["label"] != label:
            warnings.append("The active model's reference prediction disagrees with the declared label; the paired target may not teach the intended correction.")
        if after["label"] != label:
            warnings.append("The fitted candidate still fails the declared correction on its source image.")
        if new_clean or new_text:
            warnings.append("This candidate introduces reference-set errors; review the explicit counts before approval.")
        elapsed = time.perf_counter() - started
        evidence = {
            "support": {"before_label": before["label"], "after_label": after["label"],
                        "expected_label": label, "reference_label": reference_prediction["label"]},
            "metrics": {"clean_accuracy": {"before": self.counter(before_clean, self.object_truth),
                                               "after": self.counter(after_clean, self.object_truth)},
                        "text_accuracy": {"before": self.counter(before_text, text[2]),
                                          "after": self.counter(after_text, text[2])}},
            "new_errors": {"clean": new_clean, "text": new_text},
            "repairs": {"clean": sum(b != t and a == t for b, a, t in zip(before_clean, after_clean, self.object_truth)),
                        "text": sum(b != t and a == t for b, a, t in zip(before_text, after_text, text[2]))},
            "seconds_including_reference_preparation": elapsed, "support_pairs": 1,
            "reference_objects": len(self.object_truth), "reference_text_cards": len(text[2]),
            "parent_revision": parent_revision, "warnings": warnings,
        }
        metadata = {"title": body.title, "target_word": body.target_word, "expected_label": label,
                    "actor": body.actor, "source": body.source, "reason": body.reason,
                    "source_sha256": source_hash, "reference_sha256": reference_hash,
                    "parent_revision": parent_revision, "preview": evidence, "warnings": warnings,
                    "fit": patch.metadata["fit"]}
        record = self.registry.propose(patch, actor=body.actor, reason=body.reason, metadata=metadata)
        # Preview never changes active parameters. Saving a candidate is explicit.
        return {"patch_id": record["patch_id"], "record": self.flatten(record),
                "status": "candidate", "preview": evidence, "warnings": warnings}


def create_workbench_router(demo, directory: Path, *, text_reference=None):
    router = APIRouter(prefix="/api/workbench", tags=["Correction workbench"])
    service = Workbench(demo, directory, text_reference=text_reference)

    def execute(operation):
        with service.lock:
            try:
                return operation()
            except HTTPException:
                raise
            except KeyError as exc:
                raise HTTPException(404, "Unknown correction candidate") from exc
            except RegistryConflictError as exc:
                raise HTTPException(409, str(exc)) from exc
            except RegistryIntegrityError as exc:
                raise HTTPException(503, str(exc)) from exc
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, str(exc)) from exc
            except (OSError, RuntimeError) as exc:
                raise HTTPException(503, f"Local correction preparation is unavailable: {exc}") from exc

    @router.get("/status")
    def status():
        return execute(service.status)

    @router.get("/patches")
    def patches():
        def read():
            service.ensure()
            return {"patches": [service.flatten(record) for record in service.registry.list_patches()]}
        return execute(read)

    @router.get("/history")
    def history():
        def read():
            service.ensure()
            return {"events": service.registry.events(), "persistence": "sqlite_append_only"}
        return execute(read)

    @router.post("/preview")
    def preview(body: PreviewRequest):
        return execute(lambda: service.preview(body))

    @router.post("/patches/{patch_id}/approve")
    def approve(patch_id: str, body: Approval):
        if not body.approved:
            raise HTTPException(400, "Explicit administrator approval is required")
        def change():
            service.ensure()
            return service.flatten(service.registry.approve(patch_id, actor=body.actor, reason=body.reason))
        return execute(change)

    @router.post("/patches/{patch_id}/activate")
    def activate(patch_id: str, body: Activation):
        def change():
            service.ensure()
            service.registry.activate(patch_id, actor=body.actor, reason=body.reason, expected_revision=body.expected_revision)
            return service.status()
        return execute(change)

    @router.post("/patches/{patch_id}/withdraw")
    def withdraw(patch_id: str, body: Identity):
        def change():
            service.ensure()
            return service.flatten(service.registry.withdraw(patch_id, actor=body.actor, reason=body.reason))
        return execute(change)

    @router.post("/rollback")
    def rollback(body: Rollback):
        def change():
            service.ensure()
            service.registry.rollback(actor=body.actor, reason=body.reason, to_patch_id=body.to_patch_id,
                                      expected_revision=body.expected_revision)
            return service.status()
        return execute(change)

    @router.post("/predict")
    def predict(body: PredictImage):
        image, _ = decode_image(body.image_base64)
        def read():
            service.ensure()
            features = demo.encode(image).detach().cpu().float()
            weight = service.registry.get_current_weight()
            return {"before": demo.predictions(features, demo.base_weight)[0],
                    "current": demo.predictions(features, weight)[0], "registry": service.registry.state(),
                    "device": str(demo.model.device) if demo.model is not None else "test_encoder",
                    "score_disclaimer": "Fixed-candidate similarity scores are uncalibrated; uploaded-image correctness is not established"}
        return execute(read)

    def examples():
        service.ensure()
        result = []
        # Observed training-support failures, deliberately chosen for illustration.
        # Neither the sealed study nor its final inputs depend on these examples.
        for word, image_id in [("camera", "cup/image_0011.jpg"),
                               ("laptop", "camera/image_0050.jpg"),
                               ("watch", "cup/image_0011.jpg")]:
            row = next((row for row in demo.rows if row["split"] == "support"
                        and row["image_id"] == image_id), None)
            if row is None:
                raise HTTPException(503, "The recorded development support examples are unavailable")
            category = row["class_id"]
            result.append({"id": f"{word}-{category}", "label": row["label"], "target_word": word,
                           "image_id": image_id,
                           "source_url": f"/api/workbench/examples/{word}-{category}/source",
                           "reference_url": f"/api/workbench/examples/{word}-{category}/reference",
                           "source": f"Caltech101 {image_id}; chosen training-support failure, controlled overlay; CC BY 4.0"})
        return result

    @router.get("/examples")
    def example_list():
        return execute(lambda: {"examples": examples()})

    @router.get("/examples/{example_id}/{kind}")
    def example_image(example_id: str, kind: str):
        def read():
            selected = next((item for item in examples() if item["id"] == example_id), None)
            if selected is None or kind not in {"source", "reference"}:
                raise HTTPException(404, "Unknown support example")
            row = next(row for row in demo.rows if row["split"] == "support"
                       and row["image_id"] == selected["image_id"])
            with Image.open(demo.dataset_root / row["path"]) as source:
                image, _ = render_study_variant(source, "distractor" if kind == "source" else "blank",
                                                 font_path=resolve_fonts()["train_sans"], position="bottom",
                                                 word=selected["target_word"])
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return Response(buffer.getvalue(), media_type="image/png")
        return execute(read)

    @router.get("/study")
    def study():
        return execute(service.study)

    @router.get("/figures/{name}")
    def study_figure(name: str):
        if name not in {"selectivity.png", "interference.png", "paired_uncertainty.png"}:
            raise HTTPException(404, "Unknown study figure")
        path = ROOT / "output/visionpatch/study-v2" / name
        if not path.exists():
            raise HTTPException(404, "Measured study figure is unavailable")
        return FileResponse(path)

    return router, service
