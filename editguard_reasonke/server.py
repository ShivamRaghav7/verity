"""Small authenticated worker; model loading happens explicitly in the CLI."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import traceback
from urllib.parse import urlsplit

from .runtime import MAX_BODY_BYTES, RuntimeErrorBase


def make_server(runtime, token, host="127.0.0.1", port=8010):
    if not isinstance(token, str) or len(token) < 16 or "\r" in token or "\n" in token:
        raise ValueError("EDITGUARD_REASONKE_TOKEN must contain at least 16 characters and no line breaks")

    class Handler(BaseHTTPRequestHandler):
        server_version = "EditGuardReasonKE/0.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, format, *args):
            # Never log Authorization headers, request bodies, or tunnel tokens.
            pass

        def send_json(self, value, status=200):
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
            self.close_connection = True

        def authorized(self):
            auth = self.headers.get("Authorization", "")
            if not secrets.compare_digest(auth.encode("utf-8"), ("Bearer " + token).encode("utf-8")):
                self.send_json({"error": "Unauthorized"}, 401)
                return False
            # Python backend-to-worker requests do not have Origin. Reject
            # direct browser calls; the private worker is not a browser API.
            if self.headers.get("Origin"):
                self.send_json({"error": "Browser requests are not accepted by this worker"}, 403)
                return False
            return True

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/health":
                self.send_json({"ok": True, "service": "editguard-reasonke"})
            elif path == "/status":
                if self.authorized():
                    self.send_json(runtime.status())
            else:
                self.send_json({"error": "Not found"}, 404)

        def do_POST(self):
            # Drain a valid bounded request body before rejecting a request.
            # Closing a Windows TCP socket with unread inbound bytes can reset
            # it before the client receives the intended JSON error response.
            # This consumes bytes only; unauthenticated JSON is never parsed
            # and can never reach generation.
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise RuntimeErrorBase("Transfer-Encoding is not supported")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise RuntimeErrorBase("Request body must be from 1 to 65536 bytes", 413)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise RuntimeErrorBase("Incomplete request body")
            except ValueError:
                self.send_json({"error": "Invalid Content-Length"}, 400)
                return
            except RuntimeErrorBase as exc:
                self.send_json({"error": str(exc)}, exc.status)
                return
            except (TimeoutError, OSError):
                self.close_connection = True
                return
            if not self.authorized():
                return
            if urlsplit(self.path).path != "/generate":
                self.send_json({"error": "Not found"}, 404)
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                self.send_json({"error": "Use application/json"}, 415)
                return
            try:
                payload = json.loads(raw)
                self.send_json(runtime.generate(payload))
            except (ValueError, UnicodeError):
                self.send_json({"error": "Invalid JSON request"}, 400)
            except RuntimeErrorBase as exc:
                if exc.status >= 500:
                    traceback.print_exc()
                self.send_json({"error": str(exc)}, exc.status)
            except (TimeoutError, OSError):
                self.close_connection = True
            except Exception:
                traceback.print_exc()
                self.send_json({"error": "Worker request failed; no model answer is available"}, 500)

    class WorkerServer(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    return WorkerServer((host, port), Handler)
