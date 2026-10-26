"""Strict endorsed final-answer matching, with all outcomes retained."""
import re
import unicodedata

ABSTAIN = re.compile(r"^(unknown|i (?:do not|don't) know|not (?:enough|specified|provided)|cannot determine)[.!\s]*$", re.I)


def normalize(text):
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"^\*\*(.*?)\*\*$", r"\1", text)
    return " ".join(text.casefold().rstrip(". ").split())


def score(question, response=None, error=None):
    if error is not None:
        return {"outcome": "execution_error", "correct": False, "answered": False, "false_correction_application": False}
    if response is None:
        return {"outcome": "unexecuted", "correct": False, "answered": False, "false_correction_application": False}
    text = response.get("content", "")
    if not isinstance(text, str) or not text.strip() or ABSTAIN.fullmatch(text.strip()):
        return {"outcome": "abstain", "correct": False, "answered": False, "false_correction_application": False}
    final = normalize(text)
    correct = final in {normalize(alias) for alias in question["expected_aliases"]}
    # Whole endorsed final spans only; mentioning a value in a rejected sentence earns no match.
    false_application = (not question["applicable"] and final in
                         {normalize(alias) for alias in question.get("inapplicable_edit_aliases", [])})
    return {"outcome": "correct" if correct else "wrong", "correct": correct,
            "answered": True, "false_correction_application": false_application,
            "initial_truth_match": final == normalize(question["initial_answer"]),
            "semantic_review": "Not performed; strict frozen whole-final-span aliases"}
