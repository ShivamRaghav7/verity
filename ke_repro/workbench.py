"""Loopback workbench for the pinned research backend, with no download endpoint.

Run with the isolated VLKEB Python: ``python -m ke_repro.workbench``.
The web server is application plumbing, not an accuracy evaluation. One model
operation runs at a time and only one parameter patch may remain active.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import secrets
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit
import uuid

from . import UPSTREAM_COMMIT
from .smoke import digest, load_cases, preflight, write_json

STATIC = Path(__file__).with_name("static")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BODY_BYTES = 7 * 1024 * 1024


class WorkbenchError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def required_text(payload, key, limit):
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise WorkbenchError(f"{key} must contain 1–{limit} characters")
    return value.strip()


class Workbench:
    def __init__(self, storage, cases, *, backend=None, verification=None):
        self.storage = Path(storage)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.cases = {case["id"]: dict(case) for case in cases}
        self.backend = backend
        self.verification = verification or {}
        self.job_lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)
        self.phase = "ready" if backend is not None else "loading"
        self.error = None
        self.active_id = None
        self.job = None
        if backend is not None:
            self._reconcile_history()

    def _reconcile_history(self):
        # A successful fresh load uses original weights. Historical patches
        # are never silently reapplied or accepted from caller-supplied files.
        for row in self.history():
            if row.get("state") == "active":
                row.update(state="inactive_after_restart", restored_by="fresh_original_model_load")
                self._save_record(row)

    def load(self, factory):
        try:
            self.backend = factory()
            self._reconcile_history()
            self.phase = "ready"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.phase = "failed"

    def status(self):
        return {"phase": self.phase, "error": self.error,
                "busy": self.job_lock.locked(), "job": self.job,
                "active_edit_id": self.active_id, "csrf_token": self.token,
                "provenance": {"upstream": "VLKEB / MEND / BLIP-2 OPT 2.7B",
                               "upstream_commit": UPSTREAM_COMMIT,
                               "editor": "Published pretrained MEND checkpoint; not trained by this application",
                               "adaptation": "VQA question scaffold and experimental context-aligned target tokens",
                               "notice": "Development preview. Reliability is being evaluated. Model-generated answers are not verified facts."},
                "verification": self.verification}

    def fixture_list(self):
        # Expected benchmark labels are intentionally not shown as generated answers.
        return [{"id": key, "question": case["question"],
                 "image_url": "api/fixture-image/" + key,
                 "source": "Author-provided training fixture; development only"}
                for key, case in self.cases.items()]

    def history(self):
        rows = []
        for path in self.storage.glob("*/record.json"):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(rows, key=lambda row: row.get("created_at_unix", 0), reverse=True)

    def _save_record(self, record):
        write_json(self.storage / record["id"] / "record.json", record)

    def _image(self, payload):
        fixture_id = payload.get("fixture_id")
        encoded = payload.get("image_base64")
        if bool(fixture_id) == bool(encoded):
            raise WorkbenchError("Choose exactly one fixture image or uploaded image")
        if fixture_id:
            if not isinstance(fixture_id, str) or fixture_id not in self.cases:
                raise WorkbenchError("Unknown fixture image")
            path = Path(self.cases[fixture_id]["image"])
            return path, {"kind": "fixture", "fixture_id": fixture_id, "sha256": digest(path)}, None
        if not isinstance(encoded, str) or len(encoded) > (MAX_UPLOAD_BYTES * 4 // 3 + 8):
            raise WorkbenchError("Uploaded image exceeds 5 MiB")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise WorkbenchError("Image must be valid base64") from exc
        if not raw or len(raw) > MAX_UPLOAD_BYTES:
            raise WorkbenchError("Uploaded image must be between 1 byte and 5 MiB")
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(io.BytesIO(raw)) as probe:
                if probe.format not in ("JPEG", "PNG", "WEBP") or probe.width * probe.height > 20_000_000:
                    raise WorkbenchError("Use JPEG, PNG or WebP up to 20 megapixels")
                probe.verify()
            with Image.open(io.BytesIO(raw)) as decoded:
                image = decoded.convert("RGB")
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise WorkbenchError("Uploaded file is not a supported valid image") from exc
        return image, {"kind": "upload", "sha256": hashlib.sha256(raw).hexdigest(),
                       "bytes": len(raw)}, raw

    def run(self, operation, payload):
        if self.phase != "ready":
            raise WorkbenchError("Model is not ready: " + self.phase, 503)
        if not self.job_lock.acquire(blocking=False):
            raise WorkbenchError("Another model operation is running; wait for it to finish", 409)
        self.job = operation
        started = time.perf_counter()
        try:
            if operation == "rollback":
                if self.active_id is None:
                    raise WorkbenchError("There is no active edit to roll back", 409)
                row = next(row for row in self.history() if row["id"] == self.active_id)
                result = self.backend.rollback()
                actual = self.backend.weight_hashes()
                restored = actual == row["before_weight_hashes"]
                row.update(state="rolled_back" if restored else "restoration_failed",
                           rollback=result, restored_weight_hashes=actual,
                           exact_restoration=restored, rolled_back_at_unix=time.time())
                self._save_record(row)
                self.active_id = None
                if not restored:
                    self.phase = "failed"
                    raise WorkbenchError("Restoration hash check failed. Restart before any further work", 500)
                return {"edit": row, "seconds": time.perf_counter() - started}
            question = required_text(payload, "question", 256)
            formatted = "Question: " + question + " Answer:"
            image, provenance, raw_image = self._image(payload)
            if operation == "generate":
                answer = self.backend.generate(image, formatted, max_new_tokens=24)
                return {"answer": answer, "answer_kind": "model_generated_unverified",
                        "question": question, "model_question": formatted,
                        "image": provenance, "active_edit_id": self.active_id,
                        "seconds": time.perf_counter() - started}
            if operation != "edit":
                raise WorkbenchError("Unknown model operation", 404)
            if self.active_id is not None:
                raise WorkbenchError("Roll back the active patch before applying another edit", 409)
            target = required_text(payload, "target", 64)
            edit_id = uuid.uuid4().hex
            edit_dir = self.storage / edit_id
            edit_dir.mkdir()
            row = {"id": edit_id, "state": "preparing", "created_at_unix": time.time(),
                   "question": question, "model_question": formatted, "target": target,
                   "image": provenance, "target_alignment": "context",
                   "question_format": "vqa", "answer_kind": "model_generated_unverified",
                   "upstream_commit": UPSTREAM_COMMIT,
                   "limitations": "One development edit. No benchmark accuracy or factual verification is implied."}
            self._save_record(row)
            edit_applied = False
            try:
                if raw_image is not None:
                    # Bytes have already passed image validation; never caller paths.
                    (edit_dir / "source-image").write_bytes(raw_image)
                row["before_weight_hashes"] = self.backend.weight_hashes()
                row["before_answer"] = self.backend.generate(image, formatted, max_new_tokens=24)
                row["edit_result"] = self.backend.edit(image, formatted, target, target_alignment="context")
                edit_applied = True
                row["patch"] = self.backend.save_patch(edit_dir / "patch")
                row["after_answer"] = self.backend.generate(image, formatted, max_new_tokens=24)
                row["after_weight_hashes"] = self.backend.weight_hashes()
                row.update(state="active", seconds=time.perf_counter() - started)
                self._save_record(row)
                self.active_id = edit_id
                return {"edit": row}
            except Exception as exc:
                # Save/export/answer failure must never leave a silently active edit.
                try:
                    current = self.backend.weight_hashes()
                    if edit_applied or (row.get("before_weight_hashes") is not None and current != row["before_weight_hashes"]):
                        row["failure_rollback"] = self.backend.rollback()
                    row["restored_weight_hashes"] = self.backend.weight_hashes()
                    row["exact_restoration"] = row.get("before_weight_hashes") == row["restored_weight_hashes"]
                    if not row["exact_restoration"]:
                        self.phase = "failed"
                except Exception as restoration:
                    row["rollback_error"] = str(restoration)
                    self.phase = "failed"
                row.update(state="failed", error=f"{type(exc).__name__}: {exc}")
                self._save_record(row)
                raise WorkbenchError("Edit failed; inspect its history record. " + row["error"], 500) from exc
        finally:
            self.job = None
            self.job_lock.release()


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EditGuardLocal/1"

        def _reply(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ("/api/health", "/api/status"):
                return self._reply(200, app.status())
            if path == "/api/fixtures":
                return self._reply(200, {"cases": app.fixture_list()})
            if path == "/api/history":
                return self._reply(200, {"edits": app.history()})
            if path.startswith(("/api/export/", "/api/patch/")):
                edit_id = path.rsplit("/", 1)[-1]
                if not re.fullmatch(r"[a-f0-9]{32}", edit_id):
                    return self._reply(404, {"error": "Unknown saved edit"})
                record = app.storage / edit_id / "record.json"
                is_patch = path.startswith("/api/patch/")
                artifact = record.with_name("patch") if is_patch else record
                if (not record.is_file() or not artifact.is_file()
                        or not artifact.resolve().is_relative_to(app.storage.resolve())):
                    return self._reply(404, {"error": "Saved artifact not available"})
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream" if is_patch else "application/json")
                self.send_header("Content-Length", str(artifact.stat().st_size))
                self.send_header("Content-Disposition", f'attachment; filename="editguard-{edit_id}.{"patch" if is_patch else "json"}"')
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                # Parameter snapshots can be hundreds of MiB; stream instead
                # of duplicating them in the RAM-constrained model process.
                with artifact.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        self.wfile.write(block)
                return
            if path.startswith("/api/fixture-image/"):
                case = app.cases.get(path.rsplit("/", 1)[-1])
                if case is None:
                    return self._reply(404, {"error": "Unknown fixture image"})
                image = Path(case["image"])
                suffix = image.suffix.lower()
                return self._reply(200, image.read_bytes(), "image/png" if suffix == ".png" else "image/jpeg")
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in files:
                name, content_type = files[path]
                return self._reply(200, (STATIC / name).read_bytes(), content_type)
            return self._reply(404, {"error": "Not found"})

        def do_POST(self):
            if self.headers.get("X-EditGuard-Token") != app.token:
                return self._reply(403, {"error": "Missing local session token; refresh the workbench"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._reply(415, {"error": "Use application/json"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 2 or length > MAX_BODY_BYTES:
                    raise WorkbenchError("Request body exceeds the permitted size", 413)
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise WorkbenchError("JSON body must be an object")
                operation = {"/api/generate": "generate", "/api/edit": "edit", "/api/rollback": "rollback"}.get(urlsplit(self.path).path)
                if operation is None:
                    raise WorkbenchError("Not found", 404)
                return self._reply(200, app.run(operation, payload))
            except (ValueError, UnicodeDecodeError) as exc:
                return self._reply(400, {"error": "Invalid JSON request"})
            except WorkbenchError as exc:
                return self._reply(exc.status, {"error": str(exc)})
            except Exception as exc:
                return self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("tmp/vlkeb-upstream"))
    parser.add_argument("--assets", type=Path, default=Path("artifacts/vlkeb/assets"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/vlkeb-assets.json"))
    parser.add_argument("--storage", type=Path, default=Path("artifacts/vlkeb/workbench"))
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    verification = preflight(SimpleNamespace(upstream=args.upstream, assets=args.assets,
                                            manifest=args.manifest, fixture=None))
    _, cases = load_cases(args.assets, args.assets / "train-fixture.json")
    app = Workbench(args.storage, cases, verification=verification)
    def factory():
        from .bridge import build_backend
        return build_backend(args.upstream, args.assets, device="cuda")
    loader = threading.Thread(target=app.load, args=(factory,), daemon=True)
    loader.start()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(app))
    print(f"EditGuard research workbench: http://{args.host}:{args.port} (model loading)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if app.backend is not None:
            app.backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
