"""Local Ollama inference. No simulated answers and no model downloads here."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request


class ModelError(Exception):
    pass


class OllamaClient:
    def __init__(self, model="qwen3-vl:2b-instruct", embedding_model="all-minilm:22m"):
        self.model = model
        self.embedding_model = embedding_model
        self.base = "http://127.0.0.1:11434"
        self._status = None
        self._checked = 0
        self.config = json.loads((Path(__file__).resolve().parents[1] / "configs/chat-models.json").read_text(encoding="utf-8"))

    def request(self, path, payload=None, timeout=180):
        request = urllib.request.Request(self.base + path,
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise ModelError("Local model request failed. Check that Ollama is running and the model is installed. " + str(exc)) from exc
        if not isinstance(result, dict):
            raise ModelError("The local model server returned an invalid JSON object.")
        if result.get("error"):
            raise ModelError(result["error"])
        return result

    def status(self):
        if self._status is not None and time.monotonic() - self._checked < 3:
            return dict(self._status)
        result = {"name": self.model, "backend": "Local Ollama", "ready": False,
                  "supports_images": False, "error": None, "embedding_ready": False}
        try:
            tags = self.request("/api/tags", timeout=5)
            models = tags.get("models", [])
            if not isinstance(models, list) or any(not isinstance(item, dict) or "name" not in item for item in models):
                raise ModelError("The model registry response is malformed.")
            installed = {item["name"]: item for item in models}
            if self.model not in installed:
                raise ModelError("Run the setup script to download " + self.model + ".")
            for name, expected in ((self.config["chat_model"], self.config["chat_digest"]),
                                   (self.config["embedding_model"], self.config["embedding_digest"])):
                if name in installed and installed[name].get("digest") != expected:
                    raise ModelError("Installed " + name + " differs from the recorded model digest. Review the model configuration before running.")
            shown = self.request("/api/show", {"model": self.model}, timeout=10)
            capabilities = shown.get("capabilities", [])
            if not isinstance(capabilities, list):
                raise ModelError("The model capability response is malformed.")
            result.update(ready=True, supports_images="vision" in capabilities,
                digest=installed[self.model].get("digest"), details=installed[self.model].get("details"),
                embedding_ready=self.embedding_model in installed,
                embedding_digest=installed.get(self.embedding_model, {}).get("digest"))
        except ModelError as exc:
            result["error"] = str(exc)
        self._status, self._checked = result, time.monotonic()
        return dict(result)

    def generate(self, messages):
        started = time.perf_counter()
        result = self.request("/api/chat", {"model": self.model, "messages": messages,
            "stream": False, "think": False, "keep_alive": "20m", "options": {
                "temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 384}}, timeout=240)
        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ModelError("The local model returned no text response.")
        return {"content": content.strip(), "generation": {
            "model": self.model, "seconds": time.perf_counter() - started,
            "eval_tokens": result.get("eval_count"), "prompt_tokens": result.get("prompt_eval_count"),
            "total_duration_ns": result.get("total_duration"), "done_reason": result.get("done_reason"),
            "temperature": 0, "seed": 42, "num_ctx": 8192, "max_new_tokens": 384}}


class OllamaEncoder:
    def __init__(self, client):
        self.client = client

    def encode(self, text):
        result = self.client.request("/api/embed", {"model": self.client.embedding_model,
            "input": text, "truncate": False, "keep_alive": "20m", "options": {"num_gpu": 0}}, timeout=60)
        vectors = result.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != 1:
            raise ModelError("The embedding model did not return one vector.")
        if (not isinstance(vectors[0], list) or not vectors[0]
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vectors[0])):
            raise ModelError("The embedding model returned an invalid or non-finite vector.")
        return vectors[0]
