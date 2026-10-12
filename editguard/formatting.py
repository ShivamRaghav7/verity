"""Versioned model input formatting; no scoring truth enters inference input."""

from decimal import Decimal, InvalidOperation


INPUT_FORMAT_VERSION = "editguard-input-v2"


def quantity_word(value, singular, plural):
    """Render a numeric quantity without changing the canonical fact value."""
    try:
        is_one = Decimal(str(value)) == 1
    except (InvalidOperation, ValueError):
        is_one = False
    return singular if is_one else plural


def display_unit(unit, value):
    """Inflect display text; structured unit fields remain canonical."""
    if unit == "devices":
        return quantity_word(value, "device", "devices")
    if unit == "hours/day":
        return quantity_word(value, "hour/day", "hours/day")
    return unit


def fact_text(fact):
    subject = f"{fact['entity_name']} {fact['plan']} version {fact['version']}"
    value = fact["value"]
    forms = {
        "device_limit": f"{subject} supports {value} {quantity_word(value, 'device', 'devices')}.",
        "price_monthly": f"{subject} costs {value} USD per month.",
        "storage_gb": f"{subject} includes {value} GB of storage.",
        "support_hours": f"{subject} includes {value} support {quantity_word(value, 'hour', 'hours')} per day.",
    }
    return forms[fact["relation"]]


def correction_text(edit):
    aliases = edit.get("entity_aliases", edit.get("aliases", []))
    return (f"Correction for {edit['entity_name']} (aliases: {', '.join(aliases)}), "
            f"{edit['plan']}, version {edit['version']}: {edit['relation']} changed from "
            f"{edit['original_value']} to {edit['corrected_value']} {display_unit(edit['unit'], edit['corrected_value'])}, "
            f"effective from {edit['effective_from']}. Source: {edit['source']}.")


def conditional_input(question, edit, context, as_of):
    return (f"answer correction: {correction_text(edit)} Question: {question} "
            f"As of: {as_of}. Unchanged context: " + " ".join(fact_text(f) for f in context))
