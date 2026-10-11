"""Versioned model input formatting; no scoring truth enters inference input."""

INPUT_FORMAT_VERSION = "editguard-input-v1"


def fact_text(fact):
    subject = f"{fact['entity_name']} {fact['plan']} version {fact['version']}"
    value = fact["value"]
    forms = {
        "device_limit": f"{subject} supports {value} devices.",
        "price_monthly": f"{subject} costs {value} USD per month.",
        "storage_gb": f"{subject} includes {value} GB of storage.",
        "support_hours": f"{subject} includes {value} support hours per day.",
    }
    return forms[fact["relation"]]


def correction_text(edit):
    aliases = edit.get("entity_aliases", edit.get("aliases", []))
    return (f"Correction for {edit['entity_name']} (aliases: {', '.join(aliases)}), "
            f"{edit['plan']}, version {edit['version']}: {edit['relation']} changed from "
            f"{edit['original_value']} to {edit['corrected_value']} {edit['unit']}, "
            f"effective from {edit['effective_from']}. Source: {edit['source']}.")


def conditional_input(question, edit, context, as_of):
    return (f"answer correction: {correction_text(edit)} Question: {question} "
            f"As of: {as_of}. Unchanged context: " + " ".join(fact_text(f) for f in context))
