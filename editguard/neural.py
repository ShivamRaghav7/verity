"""Pinned, frozen neural inference; heavyweight packages are imported lazily."""

import hashlib
import json
import math
import re
from pathlib import Path
from threading import RLock

from .formatting import INPUT_FORMAT_VERSION
from .scope_inputs import DEFAULT_SCOPE_INPUT_POLICY, scope_pair, validate_scope_policy


class ModelUnavailable(RuntimeError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(path, task):
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("artifact_type") != "editguard_auxiliary" or manifest.get("purpose") != "trained"
            or manifest.get("status") != "completed" or manifest.get("task") != task):
        raise ModelUnavailable("A completed trained auxiliary artifact is required; smoke/pretrained weights are ineligible.")
    hashes = manifest.get("files_sha256", manifest.get("file_hashes"))
    if not isinstance(hashes, dict) or not hashes:
        raise ModelUnavailable("Trained artifact lacks file integrity hashes.")
    if manifest.get("input_format_version") != INPUT_FORMAT_VERSION:
        raise ModelUnavailable("Artifact input format differs from the inference formatter.")
    if (type(manifest.get("input_length")) is not int or not 1 <= manifest["input_length"] <= 4096):
        raise ModelUnavailable("Artifact input length is invalid.")
    if task == "scope":
        try:
            validate_scope_policy(manifest)
        except ValueError as exc:
            raise ModelUnavailable("Scope artifact input policy version/source differs from preprocessing.") from exc
    for name, expected in hashes.items():
        target = (root / name).resolve()
        if root not in target.parents or not target.is_file() or file_sha256(target) != expected:
            raise ModelUnavailable(f"Artifact integrity failed: {name}")
    for key, default in [("model_path", "model"), ("tokenizer_path", "tokenizer")]:
        target = (root / manifest.get(key, default)).resolve()
        if root not in target.parents or not target.is_dir():
            raise ModelUnavailable(f"Invalid artifact {key}")
        required = "config.json" if key == "model_path" else "tokenizer_config.json"
        if not any((root / name).resolve() == target / required for name in hashes):
            raise ModelUnavailable(f"Missing integrity record for {required}")
        recorded = {(root / name).resolve() for name in hashes}
        if any(file.resolve() not in recorded for file in target.rglob("*") if file.is_file()):
            raise ModelUnavailable(f"Unhashed files in artifact {key}")
        if key == "model_path" and not any(file.suffix == ".safetensors" for file in target.rglob("*") if file.is_file()):
            raise ModelUnavailable("Artifact requires integrity-checked safetensors weights.")
    return manifest


class FrozenGenerator:
    def __init__(self, model_id, revision=None, device="cpu", local_files_only=True,
                 max_input_tokens=384, max_new_tokens=48, cache_dir=None, tokenizer_path=None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            raise ModelUnavailable("Configured CUDA device is unavailable; choose CPU explicitly.")
        self.model_id, self.revision = str(model_id), revision
        options = {"local_files_only": local_files_only, "cache_dir": cache_dir}
        if revision:
            options["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_id, **options)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **options).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        self.last_trace = {}
        self._lock = RLock()

    def generate(self, input_text):
        with self._lock, self.torch.inference_mode():
            untruncated = self.tokenizer(input_text, add_special_tokens=True)["input_ids"]
            inputs = self.tokenizer(input_text, max_length=self.max_input_tokens,
                                    truncation=True, return_tensors="pt").to(self.device)
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                         max_new_tokens=self.max_new_tokens)
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            self.last_trace = {"model_id": self.model_id, "revision": self.revision,
                               "input_tokens": len(untruncated),
                               "consumed_input_tokens": int(inputs["input_ids"].shape[1]),
                               "input_truncated": len(untruncated) > self.max_input_tokens,
                               "output_tokens": int(output.shape[1]),
                               "max_new_tokens": self.max_new_tokens,
                               "all_parameters_frozen": all(not p.requires_grad for p in self.model.parameters())}
            return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def parameter_fingerprint(self):
        digest = hashlib.sha256()
        with self._lock:
            for name, tensor in sorted(self.model.state_dict().items()):
                digest.update(name.encode())
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()


class EmbeddingRetriever:
    """Official MiniLM masked mean pooling with unit-normalized embeddings."""

    def __init__(self, model_id, revision, device="cpu", local_files_only=True, cache_dir=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.model_id, self.revision = model_id, revision
        options = {"revision": revision, "local_files_only": local_files_only, "cache_dir": cache_dir}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
        self.model = AutoModel.from_pretrained(model_id, **options).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._cache = {}
        self._lock = RLock()
        self.last_trace = {}

    def parameter_fingerprint(self):
        digest = hashlib.sha256()
        with self._lock:
            for name, tensor in sorted(self.model.state_dict().items()):
                digest.update(name.encode())
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def tokenizer_fingerprint(self):
        state = {"vocabulary": self.tokenizer.get_vocab(), "special_tokens": self.tokenizer.special_tokens_map,
                 "padding_side": self.tokenizer.padding_side, "truncation_side": self.tokenizer.truncation_side}
        encoded = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def encode(self, texts):
        with self._lock, self.torch.inference_mode():
            results = []
            truncated = 0
            for text in texts:
                if text not in self._cache:
                    tokens = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(self.device)
                    truncated += len(self.tokenizer(text)["input_ids"]) > 256
                    hidden = self.model(**tokens).last_hidden_state
                    mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    vector = self.torch.nn.functional.normalize(pooled, p=2, dim=1)[0].cpu()
                    if len(self._cache) >= 4096:
                        self._cache.clear()
                    self._cache[text] = vector
                results.append(self._cache[text])
            self.last_trace = {"embedding_model_id": self.model_id, "embedding_revision": self.revision,
                               "input_truncation_count": truncated, "max_tokens": 256}
            return self.torch.stack(results) if results else self.torch.empty((0, self.model.config.hidden_size))

    def rank(self, question, records, text_fn, k=3):
        if not records:
            return []
        vectors = self.encode([question] + [text_fn(record) for record in records])
        scores = (vectors[1:] @ vectors[0]).tolist()
        indexed = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))[:k]
        return [{"record": records[index], "score": float(score)} for index, score in indexed]


class ScopeCheckpoint:
    def __init__(self, path, device="cpu"):
        root = Path(path).resolve()
        self.manifest = validate_artifact(root, "scope")
        self.scope_input_policy = self.manifest.get("scope_input_policy", DEFAULT_SCOPE_INPUT_POLICY)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(root / self.manifest.get("tokenizer_path", "tokenizer"), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(root / self.manifest.get("model_path", "model"), local_files_only=True).to(device).eval()
        if self.model.config.num_labels != 2 or self.manifest.get("positive_class_index", 1) != 1:
            raise ModelUnavailable("Scope model must use binary labels and positive class index 1.")
        self.input_length = self.manifest.get("input_length", 192)
        self._lock = RLock()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def score(self, question, correction):
        question, correction = scope_pair(question, correction, self.scope_input_policy)
        with self._lock, self.torch.inference_mode():
            inputs = self.tokenizer(question, correction, truncation=True, max_length=self.input_length, return_tensors="pt").to(self.device)
            return float(self.model(**inputs).logits.softmax(-1)[0, 1].cpu())


class AnswerCheckpoint(FrozenGenerator):
    def __init__(self, path, device="cpu", max_new_tokens=48):
        root = Path(path).resolve()
        manifest = validate_artifact(root, "answer")
        super().__init__(root / manifest.get("model_path", "model"), device=device,
                         max_input_tokens=manifest["input_length"], max_new_tokens=max_new_tokens,
                         tokenizer_path=root / manifest.get("tokenizer_path", "tokenizer"))
        self.manifest = manifest
        self.artifact_manifest_sha256 = file_sha256(root / "manifest.json")

    def generate(self, input_text):
        answer = super().generate(input_text)
        self.last_trace.update(artifact_purpose=self.manifest["purpose"],
                               training_review_policy=self.manifest.get("review_policy", "unspecified"),
                               human_reviewed_training_data=self.manifest.get("human_reviewed_data"),
                               artifact_manifest_sha256=self.artifact_manifest_sha256)
        return answer


def calibrated_probability(probability, temperature):
    if not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be finite and between zero and one")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    p = min(max(probability, 1e-8), 1 - 1e-8)
    logit = (math.log(p) - math.log1p(-p)) / temperature
    return 1 / (1 + math.exp(-max(min(logit, 700), -700)))
