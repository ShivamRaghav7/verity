"""Bounded inference; this project did not train the upstream checkpoint.

Prompt structure adapted from YukinoshitaKaren/Reason-KE at
d515c6ac5ec747029484af65103a6270fa693058, ReasonKE++/eval/eval_rasoning.py.
Upstream code: Apache-2.0. Quantization and serving are our adaptations.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import threading
import time

MODEL_ID = "YukinoKaren/ReasonKE_plus"
MODEL_REVISION = "ceaedc2ae7c8fed6257953587d991ce550db6106"
MAX_BODY_BYTES = 64 * 1024


class RuntimeErrorBase(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _text(value, name, limit, *, optional=False):
    if optional and (value is None or value == ""):
        return ""
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise RuntimeErrorBase(f"{name} must contain 1–{limit} characters")
    return value.strip()


def read_manifest(path):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    revision = manifest.get("revision", manifest.get("model_revision"))
    if manifest.get("model_id") != MODEL_ID or revision != MODEL_REVISION:
        raise RuntimeErrorBase("The manifest must identify the inspected author checkpoint and pinned revision")
    for key, supported in (("quantization", "nf4"), ("compute_dtype", "float16")):
        if manifest.get(key, supported) != supported:
            raise RuntimeErrorBase(f"Unsupported manifest {key}; this loader requires {supported}")
        manifest[key] = supported
    for name, default, cap in (("max_input_tokens", 4096, 4096), ("max_new_tokens", 512, 512)):
        value = manifest.get(name, default)
        if type(value) is not int or not 1 <= value <= cap:
            raise RuntimeErrorBase(f"Invalid manifest {name}")
        manifest[name] = value
    manifest["revision"] = revision
    manifest["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    return manifest


def validate_request(payload, max_new_tokens=512):
    if not isinstance(payload, dict):
        raise RuntimeErrorBase("Request must be a JSON object")
    if set(payload) - {"question", "facts", "history", "max_new_tokens"}:
        raise RuntimeErrorBase("Unsupported request fields")
    question = _text(payload.get("question"), "Question", 4000)
    facts = payload.get("facts", [])
    if not isinstance(facts, list) or len(facts) > 20:
        raise RuntimeErrorBase("Supply at most 20 approved facts")
    checked_facts = []
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            raise RuntimeErrorBase("Each fact must be an object")
        checked_facts.append({
            "id": _text(fact.get("id", str(index + 1)), "Fact ID", 128),
            "question": _text(fact.get("question"), "Fact question", 1000),
            "answer": _text(fact.get("answer"), "Fact answer", 2000),
            "scope": _text(fact.get("scope"), "Fact scope", 1000, optional=True),
            "source": _text(fact.get("source"), "Fact source", 2000, optional=True),
        })
    if sum(len(value) for fact in checked_facts for value in fact.values()) > 12000:
        raise RuntimeErrorBase("Approved fact values exceed the fixed 12000-character budget", 413)
    history = payload.get("history", [])
    if not isinstance(history, list) or len(history) > 8:
        raise RuntimeErrorBase("Supply at most 8 preceding history messages")
    checked_history = []
    for item in history:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            raise RuntimeErrorBase("History roles must be user or assistant")
        checked_history.append({"role": item["role"], "content": _text(item.get("content"), "History content", 3000)})
    tokens = payload.get("max_new_tokens", max_new_tokens)
    if type(tokens) is not int or not 1 <= tokens <= max_new_tokens:
        raise RuntimeErrorBase(f"max_new_tokens must be an integer from 1 to {max_new_tokens}")
    return {"question": question, "facts": checked_facts, "history": checked_history, "max_new_tokens": tokens}


def build_messages(request):
    """Retain upstream edit prompt in the latest user turn, not an old system turn.

    JSON serialization delimits facts; it does not guarantee prompt-injection
    resistance or factual adherence. Facts must arrive from the admin workflow.
    """
    messages = [dict(item) for item in request["history"]]
    if request["facts"]:
        facts = json.dumps(request["facts"], ensure_ascii=False)
        question = (
            "Please acknowledge the updated information provided below and respond to the subsequent query."
            "\n\n[Updated Information]:\n" + facts + "\n\n[Query]:\n" + request["question"]
        )
    else:
        question = request["question"]
    messages.append({"role": "user", "content": question})
    return messages


def extract_answer(raw):
    """Separate the model's final answer from its generated explanation.

    Multiple answer blocks or an incomplete block are disclosed as malformed;
    plain conversational text is returned as raw fallback, never as verified.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeErrorBase("The model generated no text", 502)
    blocks = re.findall(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.I | re.S)
    if len(blocks) == 1 and blocks[0].strip():
        return blocks[0].strip(), {"answer_tag_found": True, "multiple_answer_tags": False,
                                  "raw_fallback": False, "verified": False}
    return raw.strip(), {"answer_tag_found": bool(blocks), "multiple_answer_tags": len(blocks) > 1,
                         "raw_fallback": True, "verified": False}


class ReasonKERuntime:
    def __init__(self, manifest_path):
        self.manifest = read_manifest(manifest_path)
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.lock = threading.Lock()
        self.load_seconds = None
        self.hardware = {}

    def load(self):
        """Explicit network/model initialization. Never called at import time."""
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeErrorBase("Install the isolated ReasonKE inference requirements before loading", 503) from exc
        if not torch.cuda.is_available():
            raise RuntimeErrorBase("CUDA is required for this fixed 4-bit deployment configuration", 503)
        capability = torch.cuda.get_device_capability(0)
        if capability < (6, 0):
            raise RuntimeErrorBase("The fixed bitsandbytes configuration needs a supported NVIDIA GPU", 503)
        self.torch = torch
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                       trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise RuntimeErrorBase("The pinned checkpoint does not provide a chat template", 503)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
            trust_remote_code=False, use_safetensors=True, quantization_config=quantization,
            torch_dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True,
            attn_implementation="sdpa")
        self.model.eval()
        torch.cuda.synchronize(0)
        self.load_seconds = time.perf_counter() - started
        properties = torch.cuda.get_device_properties(0)
        self.hardware = {"name": properties.name, "total_vram_bytes": properties.total_memory,
            "compute_capability": list(capability), "load_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "load_peak_reserved_bytes": torch.cuda.max_memory_reserved(0), "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("transformers", "accelerate", "bitsandbytes")}}

    def status(self):
        return {"ready": self.model is not None, "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "backend": "ReasonKE author checkpoint, 4-bit inference adaptation", "supports_images": False,
            "quantization": "NF4 double quantization, FP16 compute", "load_seconds": self.load_seconds,
            "hardware": self.hardware, "busy": self.lock.locked(),
            "manifest_sha256": self.manifest["manifest_sha256"],
            "checkpoint_matches_published_95_48_verified": False,
            "weights_changed_by_new_facts": False, "checkpoint_trained_by_this_project": False}

    def generate(self, payload):
        request = validate_request(payload, self.manifest["max_new_tokens"])
        if self.model is None:
            raise RuntimeErrorBase("The checkpoint has not been loaded", 503)
        if not self.lock.acquire(blocking=False):
            raise RuntimeErrorBase("The model is busy; retry after the current request", 409)
        try:
            torch = self.torch
            prompt = self.tokenizer.apply_chat_template(build_messages(request), tokenize=False,
                                                       add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_tokens = int(inputs["input_ids"].shape[-1])
            if input_tokens > self.manifest["max_input_tokens"]:
                raise RuntimeErrorBase("Context exceeds the fixed token budget; reduce history or facts", 413)
            inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                    max_new_tokens=request["max_new_tokens"], max_time=120.0,
                    pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                                  else self.tokenizer.eos_token_id),
                    return_dict_in_generate=False, output_scores=False, output_hidden_states=False)
            torch.cuda.synchronize(0)
            seconds = time.perf_counter() - started
            new_tokens = output[0, input_tokens:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            content, format_info = extract_answer(raw)
            count = int(new_tokens.shape[-1])
            return {"content": content, "raw_output": raw, "format": format_info,
                "generation": {"model": MODEL_ID, "revision": MODEL_REVISION, "seconds": seconds,
                    "input_tokens": input_tokens, "output_tokens": count,
                    "max_new_tokens": request["max_new_tokens"], "max_input_tokens": self.manifest["max_input_tokens"],
                    "quantization": "NF4 double quantization, FP16 compute", "do_sample": False,
                    "token_budget_reached": count >= request["max_new_tokens"],
                    "time_budget_seconds": 120.0, "time_budget_reached": seconds >= 120.0,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                    "fact_ids": [item["id"] for item in request["facts"]], "weights_changed_by_facts": False}}
        except RuntimeErrorBase:
            raise
        except Exception as exc:
            # No synthetic fallback may turn model/runtime failures into answers.
            raise RuntimeErrorBase("Model inference failed; inspect the worker console and saved diagnostic", 502) from exc
        finally:
            self.lock.release()
