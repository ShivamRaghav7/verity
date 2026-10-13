"""Exploratory request-bound QA scoring, separate from full-fact endorsement.

Only the controlled dataset's twelve question forms and whole short outputs are
supported. This module does not alter the primary scorer or adjudicate prose.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .scoring import normalize_text

SHORT_QA_VERSION = "editguard-request-bound-short-qa-v1"
_N = r"\d+(?:\.\d+)?"
_S = r"(?P<subject>.+?)"
_FORMS = (
    ("device_limit", None, rf"how many devices does {_S} support"),
    ("device_limit", None, rf"what is the device allowance on {_S}"),
    ("price_monthly", None, rf"what is the monthly price of {_S}"),
    ("price_monthly", None, rf"how much would i pay each month for {_S}"),
    ("storage_gb", None, rf"how much storage is included with {_S}"),
    ("storage_gb", None, rf"what is the storage quota for {_S}"),
    ("support_hours", None, rf"how many support hours per day does {_S} include"),
    ("support_hours", None, rf"for how many hours each day is support available on {_S}"),
    ("device_limit", "requested_le_limit", rf"can i connect (?P<quantity>{_N}) devices? on {_S}"),
    ("storage_gb", "requested_le_limit", rf"can i store (?P<quantity>{_N}) gb on {_S}"),
    ("price_monthly", "limit_le_requested", rf"does {_S} cost at most (?P<quantity>{_N}) usd per month"),
    ("support_hours", "requested_le_limit", rf"does {_S} include at least (?P<quantity>{_N}) support hours? per day"),
)
_QUESTION_PATTERNS = [(relation, predicate, re.compile(rf"^{form}\?$"))
                      for relation, predicate, form in _FORMS]
_SCOPE = re.compile(r"^(?P<entity>.+?) (?P<plan>basic|pro)(?: plan)? (?:version )?(?P<version>v\d+)$")
_OUTPUT = re.compile(rf"^(?P<currency>\$)?(?P<value>{_N})(?:\s*(?P<unit>devices?|gb|mb|tb|gigabytes?|usd(?:\s*(?:per month|/month))?|eur(?:\s*(?:per month|/month))?|gbp(?:\s*(?:per month|/month))?|(?:support )?hours?(?:\s*(?:per day|/day|per week|/week))?))?\.?$")
_UNITS = {"device_limit": "devices", "price_monthly": "USD/month",
          "storage_gb": "GB", "support_hours": "hours/day"}


def _numeric(value):
    if not isinstance(value, str) or not re.fullmatch(_N, value):
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def parse_question(question: str) -> dict | None:
    """Parse relation and explicit scope independently of expected truth."""
    if not isinstance(question, str):
        return None
    text = normalize_text(question)
    for relation, predicate, pattern in _QUESTION_PATTERNS:
        match = pattern.fullmatch(text)
        if not match:
            continue
        scope = _SCOPE.fullmatch(match["subject"])
        if scope is None:
            return None
        return {"entity_surface": scope["entity"], "plan": scope["plan"],
                "version": scope["version"], "relation": relation,
                "task": "comparison_predicate" if predicate else "requested_value",
                "predicate": predicate,
                "quantity": match.groupdict().get("quantity")}
    return None


def _bound_request(record: dict, parsed: dict) -> tuple[bool, str, str | None]:
    fact, context = record.get("expected_fact"), record.get("request_context")
    if not isinstance(fact, dict) or not isinstance(context, dict):
        return False, "Explicit fact and request context are required.", None
    for field in ("entity_id", "plan", "version", "as_of"):
        if not isinstance(fact.get(field), str) or not fact[field] or context.get(field) != fact[field]:
            return False, f"Request context and expected {field} disagree or are missing.", None
    aliases = [fact.get("entity_name"), fact.get("entity_id")]
    for values in (fact.get("aliases", []), fact.get("entity_aliases", []),
                   record.get("entity_aliases", record.get("aliases", []))):
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (tuple, list)) or any(not isinstance(v, str) for v in values):
            return False, "Entity aliases are malformed.", None
        aliases.extend(values)
    known = {normalize_text(v) for v in aliases if isinstance(v, str) and v.strip()}
    if parsed["entity_surface"] not in known:
        return False, "The independently parsed question subject is not an exact bound entity alias.", None
    if (parsed["plan"] != normalize_text(fact["plan"]) or
            parsed["version"] != normalize_text(fact["version"]) or
            parsed["relation"] != fact.get("relation")):
        return False, "The question plan, version or relation disagrees with ground truth.", None
    if fact.get("unit") != _UNITS[parsed["relation"]] or _numeric(fact.get("value")) is None:
        return False, "Ground truth has an unsupported quantity or unit.", None
    polarity = record.get("expected_polarity")
    if parsed["task"] == "requested_value":
        return (False, "A value question has an unexpected polarity label.", None) if polarity is not None else (True, "Request bound.", None)
    quantity, value = _numeric(parsed["quantity"]), _numeric(fact["value"])
    truth = value <= quantity if parsed["predicate"] == "limit_le_requested" else quantity <= value
    expected = "yes" if truth else "no"
    if polarity != expected:
        return False, "Comparison polarity disagrees with the independently parsed predicate and quantity.", None
    return True, "Request and comparison predicate bound.", expected


def _output_unit(raw: str | None, currency: str | None) -> str | None:
    if currency:
        return "USD/month" if raw is None else "contradictory_currency_and_unit"
    if raw is None:
        return None
    raw = normalize_text(raw)
    if raw in ("device", "devices"):
        return "devices"
    if raw in ("gb", "gigabyte", "gigabytes"):
        return "GB"
    if raw in ("mb", "tb"):
        return raw.upper()
    for currency_name in ("usd", "eur", "gbp"):
        if raw == currency_name or raw.startswith(currency_name + " ") or raw == currency_name + "/month":
            return currency_name.upper() + "/month"
    if "week" in raw:
        return "hours/week"
    if re.fullmatch(r"(?:support )?hours?(?:\s*(?:per day|/day))?", raw):
        return "hours/day"
    return "unsupported_unit"


def _result(correct, reason, *, task=None, evidence=None, exact_fact=False, parsed=None):
    return {"correct": correct,
            "verdict": "correct" if correct is True else "incorrect" if correct is False else "pending_uncredited",
            "reason": reason, "task": task, "evidence": evidence,
            "proves_exact_scoped_fact": exact_fact,
            "parsed_request": parsed, "diagnostic_version": SHORT_QA_VERSION,
            "human_semantic_review": False}


def score_short_qa(record: dict) -> dict:
    """Score one preserved record, keeping all nonanswers and pending cases.

    Primary incorrect verdicts are authoritative vetoes. A bare answer can use
    a bound question's scope, but cannot establish a complete fact endorsement.
    """
    parsed = parse_question(record.get("question"))
    task = parsed["task"] if parsed else None
    if record.get("status", "answered") not in {"answered", "ok", "success", "generated"}:
        return _result(False, "A failed, unavailable or abstained result receives no QA credit.", task=task, parsed=parsed)
    if record.get("answer_kind") != "model_generated":
        return _result(False, "Structured or unspecified answers do not count as model-generated QA evidence.", task=task, parsed=parsed)
    primary = record.get("score")
    if (not isinstance(primary, dict) or "correct" not in primary or
            not any(primary["correct"] is value for value in (True, False, None))):
        return _result(False, "The original primary score is missing or malformed.", task=task, parsed=parsed)
    if primary["correct"] is False:
        return _result(False, "The primary scorer found an incorrect endorsement; short-answer fallback is forbidden.", task=task, parsed=parsed)
    if parsed is None:
        return _result(None, "Question wording or scope is outside the twelve locked controlled question forms.")
    valid, reason, polarity = _bound_request(record, parsed)
    if not valid:
        return _result(False, reason, task=task, parsed=parsed)
    if primary["correct"] is True:
        return _result(True, "The primary full-fact scorer credited a complete assertion and the request is independently bound.",
                       task=task, evidence="primary_complete_assertion", exact_fact=True, parsed=parsed)
    text = record.get("answer")
    if not isinstance(text, str) or not text.strip():
        return _result(False, "No generated answer.", task=task, parsed=parsed)
    text = normalize_text(text)
    if task == "comparison_predicate":
        match = re.fullmatch(r"(yes|no)\.?", text)
        if match:
            return _result(match[1] == polarity, "Whole Yes/No output checked against the bound comparison predicate; it does not establish the exact limit.",
                           task=task, evidence="whole_polarity", parsed=parsed)
        return _result(None, "A comparison requires a complete credited assertion or a whole Yes/No output; a quantity alone receives no predicate credit.", task=task, parsed=parsed)
    match = _OUTPUT.fullmatch(text)
    if match is None:
        return _result(None, "Additional prose or an unsupported output form remains pending and uncredited.", task=task, parsed=parsed)
    observed_unit = _output_unit(match["unit"], match["currency"])
    expected = record["expected_fact"]
    if observed_unit is not None and observed_unit != expected["unit"]:
        return _result(False, "The whole short output uses an incompatible explicit unit.", task=task, evidence="whole_quantity", parsed=parsed)
    correct = _numeric(match["value"]) == _numeric(expected["value"])
    return _result(correct, "Whole quantity checked against the requested value and bound scope; omitted units use the explicit question's unit.",
                   task=task, evidence="whole_quantity", parsed=parsed)
