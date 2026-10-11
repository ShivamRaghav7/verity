"""Fair scenario inference; ground-truth values/labels are never inputs here."""

import copy
import json
import math
import re
from datetime import date
from pathlib import Path
from threading import RLock
from time import perf_counter

from .formatting import conditional_input, correction_text, fact_text
from .neural import (AnswerCheckpoint, EmbeddingRetriever, FrozenGenerator,
                     ModelUnavailable, ScopeCheckpoint, calibrated_probability, file_sha256)

CONFIGURATIONS = ("unchanged_base", "updated_rag", "prompt_only", "similarity_editor", "trained_scope_editor")


def effective(record, as_of):
    return record["effective_from"] <= as_of and (not record.get("effective_to") or as_of < record["effective_to"])


def compatible(record, context):
    return all(record.get(key) == context.get(key) for key in ("entity_id", "plan", "version")) and effective(record, context["as_of"])


def resolve_context(question, requested, facts):
    """Resolve product/tier/version/date only; relation remains a retrieval task."""
    context = {key: str(value) for key, value in requested.items() if value is not None}
    text = question.lower()
    hits = {(m.start(), m.end(), f["entity_id"]) for f in facts
            for alias in [f["entity_name"], f["entity_id"], *f.get("aliases", [])]
            for m in re.finditer(r"(?<!\w)" + re.escape(alias.lower()) + r"(?!\w)", text)}
    products = {entity for a, b, entity in hits
                if not any(c <= a and b <= d and (a, b) != (c, d) for c, d, _ in hits)}
    plans = {f["plan"] for f in facts if re.search(r"(?<!\w)" + re.escape(f["plan"].lower()) + r"(?!\w)", text)}
    version_tokens = [first or second for first, second in re.findall(
        r"\b(?:version\s+(v?[a-z0-9][a-z0-9_.-]*)|(v\d[\w.-]*))", text)]
    versions = {token.rstrip(".") if token.startswith("v") else "v" + token.rstrip(".") for token in version_tokens}
    dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
    known_plans = {f["plan"].lower() for f in facts}
    # Selectors cannot authorize silently replacing an explicit unknown tier.
    explicit_tiers = re.findall(r"\b([a-z][a-z0-9_-]*)\s+(?:subscription\s+)?(?:plan|tier)\b", text)
    version_tiers = re.findall(r"\b([a-z][a-z0-9_-]*)\s+(?:version\s+)?v\d", text)
    ignored = {"the", "my", "our", "a", "which", "this", "that", "version"}
    product_words = {word for fact in facts for alias in [fact["entity_name"], fact["entity_id"], *fact.get("aliases", [])]
                     for word in alias.lower().split()}
    if (any(token not in known_plans | ignored for token in explicit_tiers)
            or any(token not in known_plans | ignored | product_words for token in version_tiers)):
        raise ValueError("Unknown subscription tier; select or name a supported plan.")
    if not products and plans:
        # Bounded support forms: a named product before a known tier must be a
        # catalogue alias. Bare tier questions (e.g. 'on Basic') remain valid.
        first_plan = min(re.finditer(r"\b(?:" + "|".join(map(re.escape, known_plans)) + r")\b", text), key=lambda m: m.start())
        prefix = text[:first_plan.start()].strip()
        separators = list(re.finditer(r"\b(?:does|do|on|for|of|with|under|about|is)\b", prefix))
        candidate = prefix[separators[-1].end():].strip() if separators else prefix
        candidate = re.sub(r"^(?:the|my|our)\s+", "", candidate)
        if candidate and candidate not in {"what's", "whats", "what", "the", "my", "our"}:
            raise ValueError("Unrecognized product name before the subscription tier.")
    for key, found in [("entity_id", products), ("plan", plans), ("version", versions), ("as_of", dates)]:
        if len(found) > 1:
            raise ValueError("Multiple scopes mentioned; ask about one product/plan/version/date.")
        if context.get(key) and found and context[key] not in found:
            raise ValueError(f"Selected {key} conflicts with the question.")
        if not context.get(key) and found:
            context[key] = next(iter(found))
    if any(word in text for word in ["older version", "used to", "previously", "originally"]) and not context.get("as_of") and not versions:
        raise ValueError("Historical requests need a version or date.")
    context.setdefault("as_of", date.today().isoformat())
    if date.fromisoformat(context["as_of"]).isoformat() != context["as_of"]:
        raise ValueError("Invalid ISO date.")
    for key in ("entity_id", "plan"):
        if key not in context:
            candidates = {f[key] for f in facts if all(f[k] == context[k] for k in ("entity_id", "plan") if k in context)}
            if len(candidates) != 1:
                raise ValueError(f"Please specify the {key.replace('_', ' ')}.")
            context[key] = next(iter(candidates))
    scoped = [f for f in facts if f["entity_id"] == context["entity_id"] and f["plan"] == context["plan"] and effective(f, context["as_of"])]
    if not scoped:
        raise ValueError("Unknown or date-incompatible product/plan.")
    if "version" not in context:
        context["version"] = max(scoped, key=lambda f: (f["effective_from"], int(re.sub(r"\D", "", f["version"]) or 0)))["version"]
    if not any(f["version"] == context["version"] for f in scoped):
        raise ValueError("Unknown or incompatible product version.")
    # Explicit unknown tier after common plan prepositions must not be ignored.
    known = {f["plan"].lower() for f in facts} | {"the", "my"}
    for match in re.finditer(r"\b(?:on|under) (?:the )?([a-z]+)(?: plan)?\b", text):
        token = match.group(1)
        aliases = {word for f in facts for a in [f["entity_name"], *f.get("aliases", [])] for word in a.lower().split()}
        if token not in known | aliases:
            raise ValueError("Unknown plan/product in the question.")
    return context


def active_memory(edits, as_of):
    active = [copy.deepcopy(edit) for edit in edits
              if edit.get("status", edit.get("approval_status")) in ("approved", "approved_in_simulated_scenario")
              and effective(edit, as_of)]
    by_id = {edit["edit_id"]: edit for edit in edits}

    def has_active_descendant(edit):
        for candidate in active:
            predecessor = candidate.get("supersedes")
            seen = set()
            while predecessor and predecessor not in seen:
                if predecessor == edit["edit_id"]:
                    return True
                seen.add(predecessor)
                predecessor = by_id.get(predecessor, {}).get("supersedes")
        return False

    return [edit for edit in active if not has_active_descendant(edit)]


def updated_snapshot(facts, edits, as_of):
    updated = copy.deepcopy(facts)
    values = {}
    for edit in active_memory(edits, as_of):
        key = (edit["entity_id"], edit["relation"], edit["plan"], edit["version"])
        if key in values and values[key]["corrected_value"] != edit["corrected_value"]:
            raise ValueError("Unresolved approved correction conflict.")
        values[key] = edit
    for fact in updated:
        edit = values.get((fact["entity_id"], fact["relation"], fact["plan"], fact["version"]))
        if edit and effective(fact, as_of):
            fact.update(value=edit["corrected_value"], source=edit["source"], applied_edit_id=edit["edit_id"])
    return updated


def base_prompt(question, context, retrieved_facts, edits=()):
    facts_text = "\n".join(fact_text(item["record"]) for item in retrieved_facts)
    prompt = ("Answer the support question using the supplied facts. State the product, plan, version, property and answer. "
              "For a yes/no question start Yes or No. Do not guess.\n"
              f"Request scope: {context['entity_id']}, {context['plan']}, {context['version']}, as of {context['as_of']}.\n"
              f"Support facts:\n{facts_text}\n")
    if edits:
        prompt += "Approved corrections override initial facts only within their scope:\n" + "\n".join(correction_text(item["record"]) for item in edits) + "\n"
    return prompt + f"Question: {question}\nAnswer:"


class ComparisonRunner:
    def __init__(self, base, retriever, *, scope=None, answer_model=None, calibration=None,
                 retrieval_k=3, prompt_edit_k=3, similarity_threshold=None):
        self.base, self.retriever = base, retriever
        self.scope, self.answer_model, self.calibration = scope, answer_model, calibration
        self.retrieval_k, self.prompt_edit_k = retrieval_k, prompt_edit_k
        self.similarity_threshold = similarity_threshold
        self._lock = RLock()

    @classmethod
    def from_config(cls, path):
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        options = {key: config[key] for key in ("device", "local_files_only", "cache_dir")}
        base = FrozenGenerator(config["base_model_id"], config["base_revision"], **options,
                               max_input_tokens=config["max_input_tokens"], max_new_tokens=config["max_new_tokens"])
        retriever = EmbeddingRetriever(config["embedding_model_id"], config["embedding_revision"], **options)
        scope = ScopeCheckpoint(config["scope_artifact"], config["device"]) if config.get("scope_artifact") else None
        answer = AnswerCheckpoint(config["answer_artifact"], config["device"], config["max_new_tokens"]) if config.get("answer_artifact") else None
        calibration = None
        if config.get("calibration_artifact"):
            calibration = json.loads(Path(config["calibration_artifact"]).read_text(encoding="utf-8"))
            if (not scope or calibration.get("purpose") != "validation_calibration"
                    or calibration.get("split") != "validation" or calibration.get("status") != "frozen"
                    or calibration.get("scope_manifest_sha256") != file_sha256(Path(config["scope_artifact"]) / "manifest.json")):
                raise ModelUnavailable("Calibration must be validation-only, frozen, and bound to this scope artifact.")
            for key in ("threshold", "ambiguity_margin"):
                if not isinstance(calibration.get(key), (int, float)) or not 0 <= calibration[key] <= 1:
                    raise ModelUnavailable("Invalid calibration operating point.")
            calibrated_probability(.5, calibration.get("temperature", 1))
        return cls(base, retriever, scope=scope, answer_model=answer, calibration=calibration,
                   retrieval_k=config["retrieval_k"], prompt_edit_k=config["prompt_edit_k"],
                   similarity_threshold=config.get("similarity_threshold"))

    def answer(self, question, requested_context, facts, edits, *, configuration, revision=0, conflicts=()):
        if configuration not in CONFIGURATIONS:
            raise ValueError("Unknown comparison configuration")
        started = perf_counter()
        with self._lock:
            try:
                return self._answer(question, requested_context, facts, edits, configuration, revision, conflicts)
            except Exception as exc:
                return {"configuration": configuration, "question": question, "status": "error", "answer": None,
                        "answer_kind": "model_generated", "route": "inference_error", "revision": revision,
                        "selected_edit_id": None, "retrieval_candidates_raw": [], "retrieval_candidates_eligible": [],
                        "retrieved_facts": [], "scope_scores": [], "generation_trace": None, "input_text": None,
                        "error": str(exc), "latency_ms": round((perf_counter() - started) * 1000, 3)}

    def _answer(self, question, requested_context, facts, edits, configuration, revision, conflicts):
        started = perf_counter()
        result = {"configuration": configuration, "question": question, "status": "clarification",
                  "answer": None, "answer_kind": "model_generated", "route": "clarification",
                  "revision": revision, "selected_edit_id": None, "retrieval_candidates_raw": [],
                  "retrieval_candidates_eligible": [], "retrieved_facts": [], "scope_scores": [],
                  "generation_trace": None, "input_text": None, "error": None}

        def finish(**changes):
            result.update(changes)
            result["latency_ms"] = round((perf_counter() - started) * 1000, 3)
            return result

        if configuration not in CONFIGURATIONS:
            raise ValueError("Unknown comparison configuration")
        if configuration in ("similarity_editor", "trained_scope_editor") and not self.answer_model:
            return finish(status="unavailable", route="unavailable", error="Trained answer artifact is missing.")
        if configuration == "trained_scope_editor" and (not self.scope or not self.calibration):
            return finish(status="unavailable", route="unavailable", error="Trained scope artifact or frozen calibration is missing.")
        if configuration == "similarity_editor" and self.similarity_threshold is None:
            return finish(status="unavailable", route="unavailable", error="Validation-selected similarity threshold is missing.")
        try:
            context = resolve_context(question, requested_context, facts)
        except ValueError as exc:
            return finish(answer=str(exc), answer_kind="deterministic_clarification")
        result["request_context"] = context
        scoped_conflicts = [conflict for conflict in conflicts
                            if compatible({**conflict["scope"], "effective_from": conflict["effective_from"],
                                           "effective_to": conflict.get("effective_to")}, context)]
        if scoped_conflicts and configuration != "unchanged_base":
            return finish(status="conflict", route="conflict", answer_kind="deterministic_clarification",
                          answer="An overlapping correction in this product/plan/version requires administrator resolution.",
                          conflicts=scoped_conflicts)
        relevant_facts = [fact for fact in facts if compatible(fact, context)]
        memory = active_memory(edits, context["as_of"])
        eligible = [edit for edit in memory if compatible(edit, context)]
        if configuration != "unchanged_base":
            try:
                updated_snapshot(relevant_facts, eligible, context["as_of"])
            except ValueError as exc:
                return finish(status="conflict", route="conflict", answer=str(exc), answer_kind="deterministic_clarification")
        # Base and updated RAG do not retrieve corrections: their costs reflect
        # their actual routes. Updated RAG retrieves the corrected fact snapshot.
        correction_route = configuration in ("prompt_only", "similarity_editor", "trained_scope_editor")
        raw = self.retriever.rank(question, memory, correction_text, k=self.retrieval_k) if correction_route else []
        ranked = self.retriever.rank(question, eligible, correction_text, k=max(self.retrieval_k, self.prompt_edit_k)) if correction_route else []
        result["retrieval_candidates_raw"] = [{"edit_id": hit["record"]["edit_id"], "score": hit["score"]} for hit in raw]
        result["retrieval_candidates_eligible"] = [{"edit_id": hit["record"]["edit_id"], "score": hit["score"]} for hit in ranked]
        chosen = None
        if configuration == "similarity_editor" and ranked and ranked[0]["score"] >= self.similarity_threshold:
            chosen = ranked[0]["record"]
        if configuration == "trained_scope_editor" and ranked:
            scores = [{"edit": hit["record"], "probability": calibrated_probability(
                       self.scope.score(question, correction_text(hit["record"])), self.calibration["temperature"])} for hit in ranked[:self.retrieval_k]]
            scores.sort(key=lambda score: -score["probability"])
            result["scope_scores"] = [{"edit_id": score["edit"]["edit_id"], "probability": score["probability"]} for score in scores]
            best = scores[0]
            threshold = self.calibration["threshold"]
            # An explicit uncertainty band around the selected operating point.
            if abs(best["probability"] - threshold) <= self.calibration["ambiguity_margin"]:
                return finish(answer="Correction applicability is uncertain; please clarify the requested property.", answer_kind="deterministic_clarification")
            if best["probability"] >= threshold:
                if len(scores) > 1 and best["probability"] - scores[1]["probability"] < self.calibration["ambiguity_margin"]:
                    return finish(answer="Multiple corrections are similarly applicable; please clarify the property.", answer_kind="deterministic_clarification")
                chosen = best["edit"]
        if chosen:
            support = [fact for fact in relevant_facts if fact["relation"] != chosen["relation"]]
            support = [hit["record"] for hit in self.retriever.rank(question, support, fact_text, k=2)]
            result["supporting_context"] = copy.deepcopy(support)
            input_text = conditional_input(question, chosen, support, context["as_of"])
            generator = self.answer_model
            result.update(selected_edit_id=chosen["edit_id"], route="approved_correction")
        else:
            snapshot = updated_snapshot(relevant_facts, eligible, context["as_of"]) if configuration == "updated_rag" else relevant_facts
            retrieved = self.retriever.rank(question, snapshot, fact_text, k=self.retrieval_k)
            result["retrieved_facts"] = [{"fact_id": hit["record"]["fact_id"], "value": hit["record"]["value"],
                                           "source": hit["record"]["source"], "score": hit["score"],
                                           "applied_edit_id": hit["record"].get("applied_edit_id"),
                                           "fact": copy.deepcopy(hit["record"])} for hit in retrieved]
            input_text = base_prompt(question, context, retrieved, ranked[:self.prompt_edit_k] if configuration == "prompt_only" else ())
            generator = self.base
            result["route"] = "updated_rag" if configuration == "updated_rag" else "prompt_only" if configuration == "prompt_only" else "frozen_base"
        result["input_text"] = input_text
        try:
            answer = generator.generate(input_text)
        except Exception as exc:
            return finish(status="error", route="generation_error", error=str(exc))
        return finish(status="answered" if answer else "abstention", answer=answer,
                      generation_trace=copy.deepcopy(getattr(generator, "last_trace", {})))
