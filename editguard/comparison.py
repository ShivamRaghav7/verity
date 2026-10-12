"""Fair scenario inference; ground-truth values/labels are never inputs here."""

import copy
import json
import math
import re
from datetime import date
from pathlib import Path
from threading import RLock
from time import perf_counter

from .formatting import INPUT_FORMAT_VERSION, conditional_input, correction_text, fact_text
from .neural import (AnswerCheckpoint, EmbeddingRetriever, FrozenGenerator,
                     ModelUnavailable, ScopeCheckpoint, calibrated_probability, file_sha256)

CONFIGURATIONS = ("unchanged_base", "updated_rag", "prompt_only", "similarity_editor", "trained_scope_editor")


def load_similarity_calibration(path, config):
    """Validate recorded validation selection before any heavyweight model load."""
    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    routing = {
        "embedding_model_id": config["embedding_model_id"],
        "embedding_revision": config["embedding_revision"],
        "embedding_max_tokens": 256, "pooling": "masked_mean_l2",
        "input_format_version": INPUT_FORMAT_VERSION,
        "routing_policy": "top_compatible_cosine_gte_threshold_v1",
        "retrieval_k": config["retrieval_k"], "prompt_edit_k": config["prompt_edit_k"],
    }
    required = {"artifact_type": "editguard_similarity_threshold",
                "purpose": "similarity_validation_calibration", "split": "validation", "status": "frozen",
                "score_type": "cosine_similarity_not_probability", "uncertainty_band": None,
                "population": "single_focal_question_correction_pairs", **routing}
    if any(key not in artifact or artifact[key] != value for key, value in required.items()):
        raise ModelUnavailable("Similarity selection must be frozen, validation-only and bound to this embedding and routing policy.")
    threshold = artifact.get("threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not -1 <= threshold <= 1.01):
        raise ModelUnavailable("Similarity threshold must be finite in [-1, 1.01].")
    if artifact.get("test_rows_read") is not False or artifact.get("embedding_weights_unchanged") is not True:
        raise ModelUnavailable("Similarity evidence must exclude test data and preserve the frozen embedding weights.")
    review_policy = artifact.get("review_policy")
    if review_policy not in ("human", "automated_development") or artifact.get("human_reviewed_data") is not (review_policy == "human"):
        raise ModelUnavailable("Similarity selection must disclose its validation review policy.")
    if review_policy == "automated_development" and not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("review_audit_sha256", ""))):
        raise ModelUnavailable("Automated similarity selection requires its independent review audit hash.")
    digest_fields = ("routing_config_sha256", "dataset_manifest_sha256", "validation_file_sha256",
                     "review_registry_sha256", "tokenizer_sha256", "embedding_state_sha256_before",
                     "embedding_state_sha256_after", "pair_scores_sha256", "selection_grid_sha256")
    if any(not isinstance(artifact.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", artifact[key])
           for key in digest_fields):
        raise ModelUnavailable("Similarity selection lacks valid integrity/provenance hashes.")
    if artifact["embedding_state_sha256_before"] != artifact["embedding_state_sha256_after"]:
        raise ModelUnavailable("Embedding state changed during similarity selection.")
    import hashlib
    routing_bytes = json.dumps(routing, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if artifact["routing_config_sha256"] != hashlib.sha256(routing_bytes).hexdigest():
        raise ModelUnavailable("Similarity routing configuration hash differs from inference.")
    source_root = Path(__file__).resolve().parents[1]
    for field, relative in (("policy_source_sha256", "editguard/comparison.py"),
                            ("embedding_source_sha256", "editguard/neural.py"),
                            ("formatter_source_sha256", "editguard/formatting.py"),
                            ("script_sha256", "scripts/calibrate_similarity.py")):
        if artifact.get(field) != file_sha256(source_root / relative):
            raise ModelUnavailable(f"Similarity evidence was selected with a different source: {relative}.")
    for name, field in (("validation_cosine_scores.jsonl", "pair_scores_sha256"),
                        ("selection_grid.json", "selection_grid_sha256")):
        sibling = artifact_path.parent / name
        if not sibling.is_file() or file_sha256(sibling) != artifact[field]:
            raise ModelUnavailable(f"Similarity selection evidence integrity failed: {name}.")
    grid = json.loads((artifact_path.parent / "selection_grid.json").read_text(encoding="utf-8"))
    row_hashes = artifact.get("accepted_row_sha256")
    accepted = artifact.get("accepted_validation_rows")
    if (type(accepted) is not int or accepted < 2 or not isinstance(row_hashes, dict)
            or len(row_hashes) != accepted
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in row_hashes.values())
            or grid.get("threshold") != threshold or grid.get("rows") != accepted):
        raise ModelUnavailable("Similarity selection lacks consistent reviewed-row or operating-point evidence.")
    records = [json.loads(line) for line in (artifact_path.parent / "validation_cosine_scores.jsonl").read_text(encoding="utf-8").splitlines() if line]
    seen = set()
    for record in records:
        row_id, score = record.get("id"), record.get("cosine_score")
        if (row_id in seen or row_id not in row_hashes or record.get("row_sha256") != row_hashes[row_id]
                or record.get("split") != "validation" or type(record.get("label")) is not int or record["label"] not in (0, 1)
                or type(record.get("compatible")) is not bool or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score) or not -1.000001 <= score <= 1.000001):
            raise ModelUnavailable("Similarity score rows disagree with reviewed validation provenance.")
        seen.add(row_id)
    if len(records) != accepted or seen != set(row_hashes) or {row["label"] for row in records} != {0, 1}:
        raise ModelUnavailable("Similarity score population differs from accepted validation rows.")
    for field in ("raw_pair_metrics", "compatibility_gated_metrics", "declared_costs"):
        if not isinstance(artifact.get(field), dict) or artifact[field] != grid.get(field):
            raise ModelUnavailable("Similarity operating metrics disagree with the hashed selection grid.")
    return artifact


def effective(record, as_of):
    return record["effective_from"] <= as_of and (not record.get("effective_to") or as_of < record["effective_to"])


def compatible(record, context):
    return all(record.get(key) == context.get(key) for key in ("entity_id", "plan", "version")) and effective(record, context["as_of"])


def resolve_context(question, requested, facts):
    """Resolve product/tier/version/date only; relation remains a retrieval task."""
    context = {key: str(value) for key, value in requested.items() if value is not None}
    text = question.lower().replace("’", "'")
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
    ignored = {"the", "my", "our", "a", "which", "this", "that", "version", "plan", "tier", "subscription"}
    product_words = {word for fact in facts for alias in [fact["entity_name"], fact["entity_id"], *fact.get("aliases", [])]
                     for word in alias.lower().split()}
    if (any(token not in known_plans | ignored for token in explicit_tiers)
            or any(token not in known_plans | ignored | product_words for token in version_tiers)):
        raise ValueError("Unknown subscription tier; select or name a supported plan.")
    # Explicit subjects in the bounded support forms need a known catalogue
    # alias or a known bare tier. Selectors cannot turn an unsupported name in
    # one of these forms into an authoritative question about another scope.
    subject_forms = (
        r"how many devices does (?P<subject>.+?) support",
        r"how many support hours per day does (?P<subject>.+?) include",
        r"what is (?:the )?(?:monthly price|price|device limit|device allowance|storage quota|storage limit|support hours) (?:of|for|on|with) (?P<subject>.+)",
        r"what(?: is|'s) (?P<subject>.+?)'s (?:monthly price|price|device limit|device allowance|storage quota|storage limit|support hours)",
        r"what is (?:the )?(?:device allowance|storage quota) (?:on|for) (?P<subject>.+)",
        r"how much storage is included with (?P<subject>.+)",
        r"how much would i pay each month for (?P<subject>.+)",
        r"for how many hours each day is support available on (?P<subject>.+)",
        r"can i connect [\w.-]+ devices on (?P<subject>.+)",
        r"can i store [\w.-]+ gb on (?P<subject>.+)",
        r"does (?P<subject>.+?) cost at most [\w.-]+ usd per month",
        r"does (?P<subject>.+?) include at least [\w.-]+ support hours per day",
        r"how many devices (?:are (?:allowed|permitted|supported)|can i connect) (?:on|with|under|for) (?P<subject>.+)",
    )
    bounded_question = re.sub(r"\s+as of \d{4}-\d{2}-\d{2}(?=[?.]?$)", "", text).strip().rstrip("?.")
    explicit_subjects = {match["subject"].strip() for pattern in subject_forms
                         if (match := re.fullmatch(pattern, bounded_question))}
    plan_pattern = "(?:" + "|".join(re.escape(plan) for plan in sorted(known_plans, key=len, reverse=True)) + ")"
    version_pattern = r"(?:version\s+v?[a-z0-9][a-z0-9_.-]*|v\d[\w.-]*)"
    aliases = sorted({alias.lower() for fact in facts
                      for alias in [fact["entity_name"], fact["entity_id"], *fact.get("aliases", [])]}, key=len, reverse=True)
    for subject in explicit_subjects:
        subject = re.sub(r"^(?:the|my|our)\s+", "", subject)
        if re.fullmatch(plan_pattern + rf"(?:\s+(?:plan|tier))?(?:\s+{version_pattern})?", subject):
            continue
        alias = next((alias for alias in aliases if re.match(re.escape(alias) + r"(?=$|\s)", subject)), None)
        if alias is None:
            raise ValueError("Unrecognized product or subscription tier in the question; select or name a supported scope.")
        suffix = subject[len(alias):].strip()
        if suffix and not re.fullmatch(rf"(?:{plan_pattern}(?:\s+(?:plan|tier))?(?:\s+{version_pattern})?|{version_pattern})", suffix):
            raise ValueError("Unknown subscription tier or unsupported product scope in the question.")
    if not products and plans and not explicit_subjects:
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
        if config.get("input_format_version") != INPUT_FORMAT_VERSION:
            raise ModelUnavailable("Inference configuration differs from the versioned input formatter.")
        for key in ("retrieval_k", "prompt_edit_k", "max_input_tokens", "max_new_tokens"):
            if type(config.get(key)) is not int or config[key] <= 0:
                raise ModelUnavailable("Inference retrieval and token budgets must be positive integers.")
        if config.get("similarity_threshold") is not None:
            raise ModelUnavailable("Configure a reviewed validation selection artifact; a bare similarity threshold is ineligible.")
        similarity = (load_similarity_calibration(config["similarity_calibration_artifact"], config)
                      if config.get("similarity_calibration_artifact") else None)
        calibration = None
        if config.get("calibration_artifact"):
            calibration = json.loads(Path(config["calibration_artifact"]).read_text(encoding="utf-8"))
            if (not config.get("scope_artifact") or calibration.get("purpose") != "validation_calibration"
                    or calibration.get("split") != "validation" or calibration.get("status") != "frozen"
                    or calibration.get("scope_manifest_sha256") != file_sha256(Path(config["scope_artifact"]) / "manifest.json")):
                raise ModelUnavailable("Calibration must be validation-only, frozen, and bound to this scope artifact.")
            for key in ("threshold", "ambiguity_margin"):
                value = calibration.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ModelUnavailable("Invalid calibration operating point.")
            temperature = calibration.get("temperature")
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
                raise ModelUnavailable("Calibration requires an explicit finite positive temperature.")
            calibrated_probability(.5, temperature)
        options = {key: config[key] for key in ("device", "local_files_only", "cache_dir")}
        retriever = EmbeddingRetriever(config["embedding_model_id"], config["embedding_revision"], **options)
        if similarity and (retriever.parameter_fingerprint() != similarity["embedding_state_sha256_before"]
                           or retriever.tokenizer_fingerprint() != similarity["tokenizer_sha256"]):
            raise ModelUnavailable("Loaded embedding weights/tokenizer differ from validation selection.")
        base = FrozenGenerator(config["base_model_id"], config["base_revision"], **options,
                               max_input_tokens=config["max_input_tokens"], max_new_tokens=config["max_new_tokens"])
        scope = ScopeCheckpoint(config["scope_artifact"], config["device"]) if config.get("scope_artifact") else None
        answer = AnswerCheckpoint(config["answer_artifact"], config["device"], config["max_new_tokens"]) if config.get("answer_artifact") else None
        return cls(base, retriever, scope=scope, answer_model=answer, calibration=calibration,
                   retrieval_k=config["retrieval_k"], prompt_edit_k=config["prompt_edit_k"],
                   similarity_threshold=similarity["threshold"] if similarity else None)

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
