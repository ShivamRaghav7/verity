"""Explicit, versioned scope-pair normalization using correction text only.

Entity normalization is deterministic preprocessing, not learned entity or date
verification. Original correction metadata remains authoritative for routing,
approval, dates and provenance. No gold facts, labels or request truth enter here.
"""

from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path
import re

SCOPE_INPUT_POLICIES = ("raw_v1", "entity_neutral_v1")
SCOPE_INPUT_POLICY_VERSION = "editguard-scope-input-v1"
DEFAULT_SCOPE_INPUT_POLICY = "raw_v1"
SCOPE_INPUT_IDENTITY_FIELDS = ("scope_input_policy", "scope_input_policy_version", "scope_input_source_sha256")

_UNITS = {"device_limit": {"device", "devices"}, "price_monthly": {"USD/month"},
          "storage_gb": {"GB"}, "support_hours": {"hour/day", "hours/day"}}
_DESCRIPTOR = re.compile(
    r"\ACorrection for (?P<entity>[^\r\n]+?) \(aliases: (?P<aliases>[^()\r\n]*)\), "
    r"(?P<plan>[^,:()\r\n]+), version (?P<version>[^\s,:()]+): "
    r"(?P<relation>[a-z][a-z0-9_]+) changed from (?P<old>[+-]?\d+(?:\.\d+)?) to "
    r"(?P<new>[+-]?\d+(?:\.\d+)?) (?P<unit>[^,\r\n]+), effective from "
    r"(?P<effective>\d{4}-\d{2}-\d{2})\. Source: (?P<source>[^\r\n]+)\.\Z")


def _policy(policy):
    if not isinstance(policy, str) or policy not in SCOPE_INPUT_POLICIES:
        raise ValueError("Scope input policy must be raw_v1 or entity_neutral_v1.")
    return policy


def scope_policy_identity(policy=DEFAULT_SCOPE_INPUT_POLICY):
    """Bind exact preprocessing source bytes separately from answer formatting."""
    _policy(policy)
    return {"scope_input_policy": policy, "scope_input_policy_version": SCOPE_INPUT_POLICY_VERSION,
            "scope_input_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def validate_scope_policy(record):
    """Resolve legacy raw input or validate a complete opt-in metadata identity.

    Cards, profiles and manifests use the same contract. Scope metadata is not
    an answer-task setting, including when the explicitly supplied policy is raw.
    """
    if not isinstance(record, dict):
        raise ValueError("Scope policy metadata must be a mapping.")
    present = {key for key in SCOPE_INPUT_IDENTITY_FIELDS if key in record}
    if not present:
        return DEFAULT_SCOPE_INPUT_POLICY
    if record.get("task", record.get("component")) == "answer":
        raise ValueError("Scope input policy metadata is supported only for the scope task.")
    if present != set(SCOPE_INPUT_IDENTITY_FIELDS):
        raise ValueError("Scope input policy metadata requires the complete policy, version and source identity.")
    identity = scope_policy_identity(record["scope_input_policy"])
    if any(record[key] != value for key, value in identity.items()):
        raise ValueError("Scope input policy version/source differs from preprocessing.")
    return identity["scope_input_policy"]


def _descriptor(text):
    matched = _DESCRIPTOR.fullmatch(text)
    if matched is None:
        raise ValueError("entity_neutral_v1 requires the standard structured correction descriptor.")
    fields = matched.groupdict()
    entity, plan, version = fields["entity"], fields["plan"], fields["version"]
    if any(value != value.strip() or not value for value in (entity, plan, version)):
        raise ValueError("Correction subjects must be explicit nonempty canonical fields.")
    relation, unit = fields["relation"], fields["unit"]
    if relation not in _UNITS or unit not in _UNITS[relation]:
        raise ValueError("entity_neutral_v1 supports the four declared numeric support relations and units.")
    aliases = fields["aliases"].split(", ") if fields["aliases"] else []
    if any(not alias or alias != alias.strip() or "," in alias for alias in aliases):
        raise ValueError("Correction aliases must use the standard comma-space delimiter without empty aliases.")
    if any(alias.casefold() in {plan.casefold(), version.casefold()} for alias in [entity, *aliases]):
        raise ValueError("Product names and aliases cannot overlap the plan or version selector.")
    if any(token in name for name in [entity, *aliases] for token in (" (aliases:", "changed from", "effective from", "Source:")):
        raise ValueError("Product names or aliases contain ambiguous descriptor delimiters.")
    try:
        date.fromisoformat(fields["effective"])
        if not all(Decimal(fields[key]).is_finite() for key in ("old", "new")):
            raise ValueError("Correction values must be finite quantities.")
    except (ValueError, InvalidOperation) as exc:
        raise ValueError("Correction descriptor requires a valid ISO effective date and finite quantities.") from exc
    return fields, aliases


def _literal_pattern(literals):
    """Protect identifier continuations, including A015+ and v2.1, from masking."""
    alternatives = "|".join(re.escape(value) for value in sorted(set(literals), key=lambda value: (-len(value), value.casefold())))
    # A sentence-ending period is permitted; a dot followed by an identifier
    # character is a continuation. Hyphens, slashes and pluses can be part of a
    # product/version token and must not turn a different token into a match.
    return re.compile(r"(?<![\w+/-])(?<!\w\.)(?:" + alternatives + r")(?![\w+/-]|\.[\w])", re.IGNORECASE)


def scope_pair(question, correction_descriptor, policy=DEFAULT_SCOPE_INPUT_POLICY):
    """Return a pair for the classifier; raw_v1 preserves both strings exactly.

    The opt-in policy canonicalizes only descriptor-referenced literal selectors.
    Unmatched product, plan and version text and requested properties/quantities
    remain visible. Eligibility is still decided elsewhere from original metadata.
    """
    _policy(policy)
    if not isinstance(question, str) or not isinstance(correction_descriptor, str):
        raise ValueError("Scope question and correction descriptor must be strings.")
    if policy == "raw_v1":
        return question, correction_descriptor
    if not question.strip():
        raise ValueError("entity_neutral_v1 requires a nonempty question.")
    fields, aliases = _descriptor(correction_descriptor)
    normalized = _literal_pattern([fields["entity"], *aliases]).sub("this product", question)
    normalized = _literal_pattern([fields["plan"]]).sub("this plan", normalized)
    # Match the explicit 'version V' phrase first so it is not rendered as
    # 'version this version'; bare V has the same protected token boundaries.
    normalized = _literal_pattern(["version " + fields["version"], fields["version"]]).sub("this version", normalized)
    descriptor = (f"Correction for this product, this plan, this version: {fields['relation'].replace('_', ' ')} "
                  f"changed from {fields['old']} to {fields['new']} {fields['unit']}.")
    return normalized, descriptor
