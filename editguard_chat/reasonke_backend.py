"""Explicit client for the separately hosted, pinned ReasonKE_plus worker.

This adapter never downloads or substitutes a model. Facts supplied to the
worker are correction context, not evidence of successful factual endorsement.
"""
from __future__ import annotations

import json
import os
import time
import unicodedata
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .backend import ModelError

MODEL_ID = "YukinoKaren/ReasonKE_plus"
MODEL_REVISION = "ceaedc2ae7c8fed6257953587d991ce550db6106"
MAX_FACTS = 20
MAX_FACT_CHARACTERS = 12000


class ApprovedFactContext:
    """Bound the full approved memory; applicability is decided by the model."""

    method = "all_approved_facts_learned_applicability"
    threshold = None
    max_memory_records = MAX_FACTS
    max_memory_characters = MAX_FACT_CHARACTERS

    def clear_cache(self):
        # No embeddings, scores, or cached correction contexts exist here.
        pass

    def rank(self, question, corrections):
        rows = []
        groups = {}
        for correction in corrections:
            if correction.get("status") != "active":
                continue
            row = {key: correction.get(key) for key in
                   ("id", "question", "answer", "scope", "source", "version")}
            if any(not isinstance(row.get(key), str) or not row[key].strip()
                   for key in ("id", "question", "answer", "scope", "source")):
                raise ModelError("An approved correction is missing its identity, fact, scope or source.")
            rows.append(row)
            key = tuple(" ".join(unicodedata.normalize("NFKC", row[field]).casefold().split())
                        for field in ("question", "scope"))
            groups.setdefault(key, set()).add(row["answer"])
        if len(rows) > self.max_memory_records or sum(len(str(value)) for row in rows for value in row.values()) > self.max_memory_characters:
            raise ModelError(f"This context policy supports at most {self.max_memory_records} active facts and "
                             f"{self.max_memory_characters} memory characters. Export or withdraw unused records before continuing.")
        conflict = any(len(answers) > 1 for answers in groups.values())
        return {"accepted": bool(rows) and not conflict, "selected": [] if conflict else rows,
                "candidates": rows, "method": self.method, "threshold": None,
                "score_is_probability": False, "clarification_required": conflict,
                "reason": "ambiguous_corrections" if conflict else
                          "approved_context_supplied" if rows else "no_active_corrections",
                "notice": "All active approved facts were supplied as context. The trained model decides "
                          "applicability; this does not establish that it used them correctly.",
                "limitation": "Conflict checks cover matching question/scope keys; semantic overlap is not certified.",
                "active_correction_count": len(rows)}


class RetrievedFactContext(ApprovedFactContext):
    """Same declared question-only two-stage retriever as the matched experiment."""
    method = "two_stage_lexical_approved_memory"
    max_memory_records = 200
    max_memory_characters = 1000000

    def rank(self, question, corrections):
        from editguard_benchmark.retrieval import retrieve
        result = super().rank(question, corrections)
        if result.get("clarification_required"):
            return result
        selected = retrieve(question, result["selected"], k=4) if result["selected"] else []
        if sum(len(str(value)) for row in selected for value in row.values()) > MAX_FACT_CHARACTERS:
            raise ModelError("The selected facts exceed the model's 12,000-character context limit; shorten their source or answer fields.")
        return {**result, "method": self.method, "selected": selected, "candidates": selected,
                "accepted": bool(selected), "reason": "question_only_approved_fact_retrieval" if selected else "no_active_corrections",
                "notice": "Up to four approved facts were selected by question-only two-stage lexical retrieval. "
                          "Selection scores are not confidence probabilities; retrieved sources do not verify generated prose.",
                "memory_record_limit": self.max_memory_records, "supplied_fact_count": len(selected),
                "limitation": "The model benchmark uses nine-fact worlds. The 200-record storage/retrieval bound is not a measured neural accuracy guarantee."}


class IterativeFactContext(ApprovedFactContext):
    """Optional bounded expansion policy, separate from the frozen study's RAG."""

    method = "bounded_iterative_lexical_approved_memory"
    max_memory_records = 200
    max_memory_characters = 1000000

    def __init__(self):
        # Import only when requested: the legacy policies keep their dependency
        # and retrieval behavior, including their exact four-fact selection.
        from editguard_depth.retrieval import DEFAULT_K, SEED_COUNT, EXPANSION_ROUNDS
        self.fact_limit = DEFAULT_K
        self.seed_count = SEED_COUNT
        self.expansion_rounds = EXPANSION_ROUNDS

    def rank(self, question, corrections):
        from editguard_depth.retrieval import retrieve_iterative
        result = super().rank(question, corrections)
        if result.get("clarification_required"):
            return {**result, "context_policy": "iterative", "selection_k": self.fact_limit}
        selected = retrieve_iterative(question, result["selected"], k=self.fact_limit) if result["selected"] else []
        if sum(len(str(value)) for row in selected for value in row.values()) > MAX_FACT_CHARACTERS:
            raise ModelError("The selected facts exceed the model's 12,000-character context limit; shorten their source or answer fields.")
        return {**result, "method": self.method, "selected": selected, "candidates": selected,
                "accepted": bool(selected), "reason": "bounded_iterative_approved_fact_retrieval" if selected else "no_active_corrections",
                "context_policy": "iterative", "selection_k": self.fact_limit,
                "retrieval_seed_count": self.seed_count,
                "retrieval_expansion_rounds": self.expansion_rounds,
                "notice": f"Up to {self.fact_limit} approved facts were selected by bounded iterative lexical expansion from the question. "
                          "Only approved question/answer text drives retrieval; no expected answer or gold support is used. "
                          "The neural model generates the answer; retrieved sources do not verify its prose.",
                "memory_record_limit": self.max_memory_records, "supplied_fact_count": len(selected),
                "limitation": "This optional policy is separate from the original frozen two-stage study. "
                              "A storage or retrieval bound is not a neural accuracy guarantee or calibrated confidence."}


class ReasonKEClient:
    """Authenticated HTTP connection to a separately loaded inference worker."""

    def __init__(self, base_url=None, token=None):
        self.base = (base_url or os.environ.get("EDITGUARD_REASONKE_URL", "http://127.0.0.1:8010")).rstrip("/")
        parsed = urlsplit(self.base)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment
                or (parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost"))):
            raise ValueError("Use an HTTPS worker origin or a loopback HTTP worker origin.")
        self.token = token if token is not None else os.environ.get("EDITGUARD_REASONKE_TOKEN", "")
        if self.token and (len(self.token) < 16 or any(char.isspace() for char in self.token)):
            raise ValueError("The Reason-KE worker token must contain at least 16 non-whitespace characters.")
        self.model = MODEL_ID
        self._status = None
        self._checked = 0.

    def request(self, path, payload=None, timeout=180):
        if not self.token:
            raise ModelError("Set EDITGUARD_REASONKE_TOKEN to the token used by the Reason-KE worker.")
        request = urllib.request.Request(self.base + path,
            data=None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ModelError("The Reason-KE worker response exceeds the 2 MiB limit.")
            result = json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                rejected = json.loads(exc.read(4096))
                if isinstance(rejected, dict) and isinstance(rejected.get("error"), str):
                    detail = ": " + rejected["error"][:500]
            except (ValueError, OSError):
                pass
            finally:
                exc.close()
            raise ModelError(f"Reason-KE worker rejected the request ({exc.code})" + detail) from exc
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise ModelError("Reason-KE worker request failed. Check its launch log and connection. " + str(exc)) from exc
        if not isinstance(result, dict):
            raise ModelError("The Reason-KE worker returned an invalid JSON object.")
        if result.get("error"):
            raise ModelError(str(result["error"]))
        return result

    def status(self):
        if self._status is not None and time.monotonic() - self._checked < 3:
            return dict(self._status)
        status = {"name": MODEL_ID, "backend": "Reason-KE trained checkpoint worker", "ready": False,
                  "supports_images": False, "embedding_ready": False, "error": None,
                  "revision": MODEL_REVISION, "weights_changed_by_corrections": False,
                  "published_metric_reproduced": False}
        try:
            result = self.request("/status", timeout=10)
            if result.get("model_id") != MODEL_ID or result.get("revision") != MODEL_REVISION:
                raise ModelError("The Reason-KE worker checkpoint does not match the pinned model and revision.")
            if result.get("ready") is not True:
                raise ModelError("The Reason-KE worker has not finished loading its checkpoint.")
            status.update(ready=True, hardware=result.get("hardware"), quantization=result.get("quantization"))
        except ModelError as exc:
            status["error"] = str(exc)
        self._status, self._checked = status, time.monotonic()
        return dict(status)

    def answer_with_facts(self, question, facts, history):
        payload = {"question": question,
                   "facts": [{key: fact[key] for key in ("id", "question", "answer", "scope", "source")}
                             for fact in facts],
                   "history": [{"role": item["role"], "content": item["content"]} for item in history],
                   "max_new_tokens": 512}
        result = self.request("/generate", payload, timeout=240)
        if not isinstance(result.get("content"), str) or not result["content"].strip():
            raise ModelError("The Reason-KE worker generated no answer text.")
        generation = result.get("generation")
        if (not isinstance(generation, dict) or generation.get("model") != MODEL_ID
                or generation.get("revision") != MODEL_REVISION):
            raise ModelError("The generated answer does not identify the pinned Reason-KE checkpoint.")
        raw = result.get("raw_output")
        if not isinstance(raw, str) or len(raw) > 32000 or len(result["content"]) > 16000:
            raise ModelError("The Reason-KE worker returned invalid or oversized generated text.")
        return {"content": result["content"].strip(), "raw_output": raw,
                "generation": generation, "format": result.get("format", {})}
