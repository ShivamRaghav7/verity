"""Conservative, auditable scoring of short factual support answers.

No model, model download, substring target match, or LLM judge is used here.
Unrecognized prose is deliberately uncredited until independent human review.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Iterable

SCORER_VERSION = "editguard-short-facts-v3"

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
         70: "seventy", 80: "eighty", 90: "ninety"}
_WORD_VALUES = {word: str(value) for value, word in enumerate(_ONES)}
for _value, _word in _TENS.items():
    _WORD_VALUES[_word] = str(_value)
    for _one in range(1, 10):
        _WORD_VALUES[f"{_word} {_ONES[_one]}"] = str(_value + _one)
        _WORD_VALUES[f"{_word}-{_ONES[_one]}"] = str(_value + _one)
_WORDS = "|".join(re.escape(word) for word in sorted(_WORD_VALUES, key=len, reverse=True))
_NUMBER = rf"(?P<value>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?|{_WORDS})"
_SUBJECT = r"(?P<subject>.+?)"

# Complete assertion grammars, not patterns searched inside arbitrary prose.
# The unit is parsed independently so a recognized wrong unit is a clear failure.
_GRAMMARS = (
    ("device_limit", rf"{_SUBJECT}\s+(?:supports|allows|permits)(?:\s+(?:up to|at most|a maximum of))?\s+{_NUMBER}\s+(?P<unit>devices?)"),
    ("device_limit", rf"{_SUBJECT}(?:'s)?\s+(?:device limit|device allowance)\s+(?:is|=)\s+{_NUMBER}\s+(?P<unit>devices?)"),
    ("price_monthly", rf"{_SUBJECT}\s+(?:costs|is priced at|has a monthly price of)\s+{_NUMBER}\s+(?P<unit>(?:usd|eur|gbp|us dollars|dollars)\s*(?:/\s*month|per month|monthly))"),
    ("price_monthly", rf"{_SUBJECT}(?:'s)?\s+monthly price\s+(?:is|=)\s+{_NUMBER}\s+(?P<unit>(?:usd|eur|gbp|us dollars|dollars)\s*(?:/\s*month|per month|monthly))"),
    ("storage_gb", rf"{_SUBJECT}\s+(?:includes|provides|offers|has)\s+{_NUMBER}\s+(?P<unit>gb|gigabytes?|mb|megabytes?|tb|terabytes?)\s+(?:of\s+)?storage"),
    # Controlled plan storage capacity: require the entire 'of storage' phrase.
    # This does not match device limits, RAM, bare GB mentions or hedged capacity.
    ("storage_gb", rf"{_SUBJECT}\s+supports\s+{_NUMBER}\s+(?P<unit>gb|gigabytes?|mb|megabytes?|tb|terabytes?)\s+of\s+storage"),
    ("storage_gb", rf"{_SUBJECT}(?:'s)?\s+(?:storage allowance|storage quota)\s+(?:is|=)\s+{_NUMBER}\s+(?P<unit>gb|gigabytes?|mb|megabytes?|tb|terabytes?)"),
    ("support_hours", rf"{_SUBJECT}\s+(?:includes|provides|offers|has)\s+{_NUMBER}\s+(?P<unit>support hours?\s+(?:per|each)\s+(?:day|week)|hours?\s*/\s*(?:day|week)\s+(?:of\s+)?support|hours?\s+of\s+support\s+(?:per|each)\s+(?:day|week))"),
)
_PATTERNS = [(relation, re.compile(rf"^{grammar}\.?$", re.IGNORECASE))
             for relation, grammar in _GRAMMARS]


def normalize_text(text: str) -> str:
    """Normalize typography and whitespace; preserve negation and all numbers."""
    return " ".join(unicodedata.normalize("NFKC", text).replace("’", "'").split()).casefold()


def _number(value: str) -> str | None:
    token = normalize_text(str(value))
    if token in _WORD_VALUES:
        return _WORD_VALUES[token]
    if not re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", token):
        return None
    try:
        numeric = Decimal(token.replace(",", ""))
        return format(numeric.normalize(), "f")
    except InvalidOperation:
        return None


def _unit(raw: str) -> str | None:
    token = normalize_text(raw)
    if token in ("device", "devices"):
        return "devices"
    if token in ("gb", "gigabyte", "gigabytes"):
        return "GB"
    if token in ("mb", "megabyte", "megabytes"):
        return "MB"
    if token in ("tb", "terabyte", "terabytes"):
        return "TB"
    if re.fullmatch(r"(?:usd|us dollars)\s*(?:/\s*month|per month|monthly)", token):
        return "USD/month"
    for currency in ("eur", "gbp"):
        if re.fullmatch(rf"{currency}\s*(?:/\s*month|per month|monthly)", token):
            return f"{currency.upper()}/month"
    if "support" in token and re.search(r"\bhours?\b", token):
        if re.search(r"\bday\b", token):
            return "hours/day"
        if re.search(r"\bweek\b", token):
            return "hours/week"
    # Truth fields use canonical units, including hours/day without prose.
    if re.fullmatch(r"hours?\s*/\s*(?:day|week)", token):
        return "hours/day" if "day" in token else "hours/week"
    return None


def _aliases(fact: dict, aliases: Iterable[str]) -> tuple[str, ...]:
    if isinstance(aliases, str):
        aliases = (aliases,)
    values = [fact.get("entity_name"), fact.get("entity_id")]
    for field in ("aliases", "entity_aliases"):
        additions = fact.get(field) or []
        values.extend((additions,) if isinstance(additions, str) else additions)
    values.extend(aliases)
    if any(value is not None and not isinstance(value, str) for value in values):
        raise TypeError("Entity aliases must be strings bound to the expected entity.")
    return tuple(sorted({normalize_text(value) for value in values if value and value.strip()}, key=len, reverse=True))


def _subject_verdict(subject: str, fact: dict, aliases: tuple[str, ...]) -> tuple[str, str]:
    """Validate the complete subject; near-name prefixes never identify an entity."""
    subject = normalize_text(subject)
    if subject.startswith("the "):
        subject = subject[4:]
    plan = normalize_text(str(fact.get("plan") or ""))
    version = normalize_text(str(fact.get("version") or ""))
    version_digits = re.fullmatch(r"v(\d+)", version)
    version_pattern = (rf"(?:v{version_digits[1]}|version\s+(?:v)?{version_digits[1]})"
                       if version_digits else re.escape(version))
    scope = (rf"\s+{re.escape(plan)}(?:\s+plan)?" if plan else "")
    scope += (rf"\s+{version_pattern}" if version else "")
    for alias in aliases:
        if re.fullmatch(re.escape(alias) + scope, subject):
            return "valid", "The endorsed subject matches the entity, plan and version."
    for alias in aliases:
        prefix = re.match(re.escape(alias) + r"(?=$|\s)", subject)
        if not prefix:
            continue
        suffix = subject[prefix.end():].strip()
        if not suffix or (plan and suffix in (plan, plan + " plan")):
            return "review", "The answer omits the requested plan or version scope."
        if plan and version and re.fullmatch(version_pattern, suffix):
            return "review", "The answer omits the requested plan scope."
        observed_version = re.search(r"\b(?:v|version\s+v?)(\d+)\b", suffix)
        if observed_version and version_digits and observed_version[1] != version_digits[1]:
            return "wrong", "The answer endorses an incompatible product version."
        if plan and re.match(re.escape(plan) + r"(?:\s+plan)?(?:\s|$)", suffix):
            return "review", "The complete endorsed scope is not an accepted form."
        return "wrong", "The answer endorses a different plan or a similarly named entity."
    return "wrong", "The answer endorses a different or unrecognized entity."


def _parse(text: str) -> dict | None:
    for relation, pattern in _PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            return {"subject": match["subject"], "relation": relation,
                    "value": _number(match["value"]), "unit": _unit(match["unit"])}
    return None


def _prefix(text: str) -> tuple[str | None, str]:
    match = re.match(r"^(yes|no)\s*[.,:]\s+(.+)$", text)
    return (match[1], match[2]) if match else (None, text)


def canonical_endorsement(fact: dict, *, polarity=None) -> dict:
    """Canonicalize an already established endorsement, without judging prose.

    Human adjudication can use the same numeric/unit representation as automatic
    scoring. Unknown values retain their literal representation; normalization
    never creates correctness or invents an endorsed fact.
    """
    result = {key: fact.get(key) for key in ("entity_id", "plan", "version", "as_of")
              if key in fact}
    value = _number(str(fact.get("value", "")))
    unit = _unit(str(fact.get("unit", "")))
    result.update(relation=fact.get("relation"),
                  value=value if value is not None else fact.get("value"),
                  unit=unit if unit is not None else fact.get("unit"))
    if polarity is not None:
        result["polarity"] = "yes" if polarity is True else "no" if polarity is False else normalize_text(str(polarity))
    return result


def _result(correct, reason: str, *, endorsed=None, answer_kind="model_generated") -> dict:
    return {"correct": correct, "verdict": "correct" if correct is True else "incorrect" if correct is False else "review_required",
            "reason": reason, "endorsed_fact": endorsed, "needs_review": correct is None,
            "answer_kind": answer_kind, "scorer_version": SCORER_VERSION}


def score_answer(text, expected_fact, *, aliases=(), expected_polarity=None,
                 answer_kind="model_generated") -> dict:
    """Score the unchanged output against an explicit, reviewed fact.

    Entity aliases must identify this fact's entity only. Exact short assertion
    grammars require all supplied scope fields. ``None`` is pending human review,
    and must count as uncredited in fixed-denominator accuracy.
    """
    if not isinstance(expected_fact, dict):
        raise TypeError("expected_fact must be an explicit fact dictionary.")
    if expected_polarity is True:
        expected_polarity = "yes"
    elif expected_polarity is False:
        expected_polarity = "no"
    if expected_polarity is not None:
        expected_polarity = normalize_text(str(expected_polarity))
        if expected_polarity not in ("yes", "no"):
            raise ValueError("expected_polarity must be yes, no, or None.")
    if not isinstance(text, str) or not text.strip():
        return _result(False, "No answer was generated.", answer_kind=answer_kind)
    entity_aliases = _aliases(expected_fact, aliases)
    expected_value = _number(str(expected_fact.get("value", "")))
    expected_unit = _unit(str(expected_fact.get("unit", "")))
    relation = expected_fact.get("relation")
    if not entity_aliases or expected_value is None or expected_unit is None or relation not in {row[0] for row in _PATTERNS}:
        return _result(None, "Ground truth is outside the locked short-fact grammar or lacks reviewed entity aliases.", answer_kind=answer_kind)
    normalized = normalize_text(text)
    polarity, body = _prefix(normalized)
    # Scan complete assertions before the whole-answer grammar. Its broad subject
    # capture can otherwise absorb an earlier sentence as part of the subject.
    # Periods inside decimals do not split the numeric token.
    fragments = re.split(r"[.!?;]\s+", body)
    observed = []
    index = 0
    while index < len(fragments):
        fragment = fragments[index]
        index += 1
        if fragment in ("yes", "no") and index < len(fragments):
            fragment += ". " + fragments[index]
            index += 1
        fragment_polarity, fragment = _prefix(fragment)
        candidate = _parse(fragment)
        if candidate and _subject_verdict(candidate["subject"], expected_fact, entity_aliases)[0] == "valid" and candidate["relation"] == relation:
            observed.append((candidate["value"], candidate["unit"], fragment_polarity))
    if len({(value, unit) for value, unit, _ in observed}) > 1:
        return _result(False, "The answer endorses incompatible values for the requested fact.", answer_kind=answer_kind)
    polarities = {value for _, _, value in observed if value is not None}
    if polarity is not None:
        polarities.add(polarity)
    if len(polarities) > 1:
        return _result(False, "The answer contains contradictory yes/no endorsements.", answer_kind=answer_kind)
    assertion = _parse(body)
    if assertion is None:
        # Explicitly rejecting the target is a failure even if it appears in prose.
        target_values = [re.escape(expected_value)] + [re.escape(word) for word, value in _WORD_VALUES.items() if value == expected_value]
        target = "(?:" + "|".join(target_values) + ")"
        reject_before = re.search(rf"\b(?:does not support|doesn't support|cannot support|not|no longer supports)\s+(?:up to\s+)?{target}\s+(?:devices?|gb|gigabytes?|support hours?)\b", body)
        reject_after = re.search(rf"\b{target}(?:\s+(?:devices?|gb|gigabytes?|support hours?))?\s+(?:is|are)\s+(?:wrong|incorrect|false)\b", body)
        if reject_before or reject_after:
            return _result(False, "The target value is explicitly rejected rather than endorsed.", answer_kind=answer_kind)
        return _result(None, "Unrecognized or additional prose requires blinded human semantic review; no target mention receives automatic credit.", answer_kind=answer_kind)
    subject_status, subject_reason = _subject_verdict(assertion["subject"], expected_fact, entity_aliases)
    if subject_status != "valid":
        return _result(None if subject_status == "review" else False, subject_reason, answer_kind=answer_kind)
    endorsed = canonical_endorsement({**expected_fact, **assertion}, polarity=polarity)
    if assertion["relation"] != relation:
        return _result(False, "The endorsed relation differs from the requested property.", endorsed=endorsed, answer_kind=answer_kind)
    if assertion["unit"] is None:
        return _result(None, "The unit is ambiguous and requires review.", endorsed=endorsed, answer_kind=answer_kind)
    if assertion["unit"] != expected_unit:
        return _result(False, "The endorsed unit differs from ground truth.", endorsed=endorsed, answer_kind=answer_kind)
    if assertion["value"] != expected_value:
        return _result(False, "The endorsed value differs from ground truth.", endorsed=endorsed, answer_kind=answer_kind)
    if expected_polarity is not None and polarity != expected_polarity:
        return _result(False, "The comparison polarity is missing or incorrect.", endorsed=endorsed, answer_kind=answer_kind)
    if expected_polarity is None and polarity is not None:
        return _result(None, "An unsolicited yes/no endorsement needs question-aware semantic review.", endorsed=endorsed, answer_kind=answer_kind)
    return _result(True, "One complete short assertion endorses the correct subject, scope, relation, value, unit and required polarity.", endorsed=endorsed, answer_kind=answer_kind)
