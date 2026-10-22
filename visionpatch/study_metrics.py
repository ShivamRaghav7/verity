"""Explicit denominators and photo-grouped paired uncertainty for study v2."""

from __future__ import annotations

import random

from .metrics import ratio


def _validate(rows: list[dict]) -> None:
    identifiers = [row["task_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Repeated task identifiers would duplicate evaluation observations")


def _accuracy(rows: list[dict], method: str) -> dict:
    return ratio(sum(row[method] == row["class_id"] for row in rows), len(rows))


def _preservation(rows: list[dict], method: str) -> dict:
    correct = [row for row in rows if row["base"] == row["class_id"]]
    preserved = sum(row[method] == row["class_id"] for row in correct)
    return {"baseline_correct_preserved": ratio(preserved, len(correct)),
            "new_errors": ratio(len(correct) - preserved, len(correct))}


def summarize_study(object_rows: list[dict], text_rows: list[dict], method: str,
                    target_word: str) -> dict:
    """Score every attempted question; missing/invalid predictions are not dropped.

    Augmentations remain question-level accuracy observations. Uncertainty must
    instead use ``grouped_accuracy_interval`` to resample their shared sources.
    """
    _validate(object_rows)
    _validate(text_rows)
    clean = [row for row in object_rows if row["variant"] == "clean"]
    target = [row for row in object_rows if row["variant"] == "distractor"
              and row["word"] == target_word]
    blank = [row for row in object_rows if row["variant"] == "blank" and row["word"] == target_word]
    other = [row for row in object_rows if row["variant"] == "distractor" and row["word"] != target_word]
    genuine = [row for row in clean if row["class_id"] == target_word]
    genuine_attacked = [row for row in target if row["class_id"] == target_word]
    clean_by_source = {row["image_id"]: row for row in clean}
    if len(clean_by_source) != len(clean):
        raise ValueError("Every source must have exactly one clean task")
    if any(row["image_id"] not in clean_by_source for row in object_rows):
        raise ValueError("Every augmented source needs its clean task")
    eligible = [row for row in target if row["class_id"] != target_word
                and clean_by_source[row["image_id"]]["base"] == row["class_id"]]
    induced = [row for row in eligible if row["base"] != row["class_id"]]
    groups = {"clean_object": clean, "target_distractor": target, "matched_blank": blank,
              "other_word_distractor": other, "genuine_target_clean": genuine,
              "genuine_target_distractor": genuine_attacked, "legitimate_text": text_rows}
    result = {f"{name}_accuracy": _accuracy(rows, method) for name, rows in groups.items()}
    result["preservation"] = {name: _preservation(rows, method) for name, rows in groups.items()}
    result["baseline_attack_induced_errors"] = ratio(len(induced), len(eligible))
    result["repair_rate"] = ratio(sum(row[method] == row["class_id"] for row in induced), len(induced))
    all_wrong = [row for row in target if row["base"] != row["class_id"]]
    result["all_baseline_wrong_target_repair_rate"] = ratio(
        sum(row[method] == row["class_id"] for row in all_wrong), len(all_wrong))
    result["repairs_per_all_target_questions"] = ratio(
        sum(row[method] == row["class_id"] for row in induced), len(target))
    result["original_photos"] = len(clean)
    result["object_questions"] = len(object_rows)
    result["text_questions"] = len(text_rows)
    result["text_macro_by_word"] = {
        class_id: _accuracy([row for row in text_rows if row["class_id"] == class_id], method)
        for class_id in sorted({row["class_id"] for row in text_rows})}
    result["target_distractor_breakdowns"] = {
        field: {value: _accuracy([row for row in target if row.get(field, "unspecified") == value], method)
                for value in sorted({row.get(field, "unspecified") for row in target})}
        for field in ("class_id", "font_id", "position")}
    result["abstentions_scored_as_incorrect"] = sum(row[method] is None for row in object_rows + text_rows)
    return result


def select_global_candidate(candidate_ids: list[str], results: dict[str, list[dict]]) -> dict:
    """Select once across words using the predeclared equally weighted objective."""
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate order must be nonempty and unique")
    ranked = []
    for index, candidate in enumerate(candidate_ids):
        summaries = results[candidate]
        if not summaries:
            raise ValueError("Every candidate needs validation summaries")
        values = [summary[key]["rate"] for summary in summaries for key in (
            "target_distractor_accuracy", "clean_object_accuracy", "legitimate_text_accuracy")]
        if any(value is None for value in values):
            raise ValueError("Selection components must all have positive denominators")
        score = sum(values) / len(values)
        new_errors = sum(summary["preservation"]["clean_object"]["new_errors"]["numerator"]
                         for summary in summaries)
        ranked.append({"candidate_id": candidate, "score": score, "new_clean_errors": new_errors,
                       "declared_index": index})
    winner = min(ranked, key=lambda row: (-row["score"], row["new_clean_errors"], row["declared_index"]))
    return {"selected": winner["candidate_id"], "candidates": ranked,
            "objective": "equal mean of target distractor, clean object and legitimate text accuracy across all edit words",
            "tie_break": "fewest newly introduced clean errors, then declared index"}


def grouped_accuracy_interval(rows: list[dict], method: str, *, group_key: str = "image_id",
                              seed: int = 42, samples: int = 2000) -> dict:
    """Paired percentile bootstrap of equal-source mean accuracy differences.

    Photos, rather than generated overlays, are independent resampling units.
    For controlled text cards use class_id: eight word groups, with an explicit
    small-group limitation. No claim is made that these groups cover real OCR.
    """
    _validate(rows)
    if type(samples) is not int or samples <= 0:
        raise ValueError("bootstrap samples must be a positive integer")
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(
            int(row[method] == row["class_id"]) - int(row["base"] == row["class_id"]))
    values = [sum(value) / len(value) for value in grouped.values()]
    if not values:
        return {"groups": 0, "questions": 0, "difference": None, "interval_95": None}
    generator = random.Random(seed)
    estimates = sorted(sum(generator.choices(values, k=len(values))) / len(values) for _ in range(samples))
    return {"groups": len(values), "questions": len(rows), "group_key": group_key,
            "difference": sum(values) / len(values), "interval_95": [
                estimates[int(samples * .025)], estimates[min(samples - 1, int(samples * .975))]],
            "bootstrap_samples": samples, "seed": seed,
            "limitation": "Paired grouped interval within this small controlled dataset; not a deployment guarantee"}
