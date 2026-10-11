"""Finite validation-only temperature and scope operating-point selection."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from editguard.formatting import INPUT_FORMAT_VERSION, correction_text
from editguard.neural import calibrated_probability, file_sha256, validate_artifact
from scripts.prepare_data import content_hash, load_reviewed_rows
from scripts.review_dataset import checked_rows, read_json

CALIBRATION_VERSION = "editguard-validation-calibration-v1"
TEMPERATURES = (.5, .75, 1.0, 1.5, 2.0, 3.0)
THRESHOLDS = tuple(index / 100 for index in range(101))
MARGINS = (0.0, .025, .05, .1)
COSTS = {"false_application": 4.0, "missed_positive": 1.0, "clarification": .25, "correct": 0.0}


def _validated_pairs(labels, probabilities):
    labels, probabilities = list(labels), list(probabilities)
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Calibration requires nonempty equally sized labels and probabilities.")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("Calibration labels must be explicit integer zero or one.")
    if set(labels) != {0, 1}:
        raise ValueError("Validation calibration requires positive and negative pairs.")
    if any(isinstance(p, bool) or not isinstance(p, (int, float))
           or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Every validation pair requires a finite probability in [0,1].")
    return labels, [float(p) for p in probabilities]


def _rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def operating_metrics(labels, probabilities, threshold, margin):
    """Mirror runtime's inclusive uncertainty band; every pair stays counted."""
    labels, probabilities = _validated_pairs(labels, probabilities)
    for name, value in (("threshold", threshold), ("margin", margin)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite in [0,1].")
    counts = {"total": len(labels), "positive": sum(labels), "negative": len(labels) - sum(labels),
              "true_application": 0, "false_application": 0, "true_rejection": 0,
              "missed_positive": 0, "clarification_positive": 0, "clarification_negative": 0}
    for label, probability in zip(labels, probabilities):
        if abs(probability - threshold) <= margin:
            counts["clarification_positive" if label else "clarification_negative"] += 1
        elif probability >= threshold:
            counts["true_application" if label else "false_application"] += 1
        else:
            counts["missed_positive" if label else "true_rejection"] += 1
    clarification = counts["clarification_positive"] + counts["clarification_negative"]
    covered = counts["total"] - clarification
    correct = counts["true_application"] + counts["true_rejection"]
    cost = (COSTS["false_application"] * counts["false_application"]
            + COSTS["missed_positive"] * counts["missed_positive"]
            + COSTS["clarification"] * clarification)
    return {"threshold": float(threshold), "ambiguity_margin": float(margin), "counts": counts,
            "total_cost": cost, "mean_cost": _rate(cost, counts["total"]),
            "coverage": _rate(covered, counts["total"]), "clarification_rate": _rate(clarification, counts["total"]),
            "accuracy_including_clarification_as_unsuccessful": _rate(correct, counts["total"]),
            "selective_accuracy": _rate(correct, covered),
            "precision": _rate(counts["true_application"], counts["true_application"] + counts["false_application"]),
            "recall_including_clarification_as_missed": _rate(counts["true_application"], counts["positive"]),
            "false_application_rate": _rate(counts["false_application"], counts["negative"])}


def fit_calibration(labels, probabilities):
    """Pure finite grids: temperature minimizes NLL, then operating cost."""
    labels, probabilities = _validated_pairs(labels, probabilities)
    temperatures = []
    for temperature in TEMPERATURES:
        transformed = [calibrated_probability(p, temperature) for p in probabilities]
        nll = -sum(math.log(max(min(p if label else 1 - p, 1 - 1e-15), 1e-15))
                   for label, p in zip(labels, transformed)) / len(labels)
        temperatures.append({"temperature": temperature, "negative_log_likelihood": nll})
    # A completely tied NLL prefers the unscaled probability; remaining ties
    # prefer the smaller temperature, making selection deterministic.
    best_temperature = min(temperatures, key=lambda item: (item["negative_log_likelihood"],
                                                           abs(item["temperature"] - 1), item["temperature"]))
    calibrated = [calibrated_probability(p, best_temperature["temperature"]) for p in probabilities]
    candidates = [operating_metrics(labels, calibrated, threshold, margin)
                  for threshold in THRESHOLDS for margin in MARGINS]
    # Integer counts determine the exact declared quarter-unit cost. Equal-cost
    # choices first avoid false applications, then retain more answer coverage.
    selected = min(candidates, key=lambda item: (item["total_cost"], item["counts"]["false_application"],
                                                -item["coverage"]["numerator"], item["ambiguity_margin"], item["threshold"]))
    return {"temperature": best_temperature["temperature"], "threshold": selected["threshold"],
            "ambiguity_margin": selected["ambiguity_margin"], "selected_operating_metrics": selected,
            "temperature_candidates": temperatures, "operating_candidates": candidates,
            "declared_costs": dict(COSTS), "rows": len(labels)}


def reviewed_inputs(scope_artifact: Path, dataset: Path, reviews: Path, *, max_rows: int):
    """Validate before model import; only the fixed validation filename is read."""
    if type(max_rows) is not int or not 2 <= max_rows <= 10000:
        raise ValueError("max_rows must be an integer between 2 and 10000.")
    scope_manifest = validate_artifact(scope_artifact, "scope")
    manifest = read_json(dataset / "manifest.json")
    manifest_hash = file_sha256(dataset / "manifest.json")
    if scope_manifest.get("dataset_id") != manifest.get("dataset_id") or scope_manifest.get("manifest_sha256") != manifest_hash:
        raise ValueError("Calibration must use the scope artifact's declared dataset and validation partition.")
    # checked_rows enforces manifest integrity, duplicate-ID/key checks and exact
    # validation split tags before the existing complete human-review gate.
    original = checked_rows(dataset, "scope", "validation")
    read_json(reviews)  # Reject duplicate registry keys before the legacy loader.
    accepted = load_reviewed_rows(dataset / "scope_validation.jsonl", reviews, require_all=True)
    accepted_ids = {row["id"] for row in accepted}
    rows = [row for row in original if row["id"] in accepted_ids]
    if len(rows) > max_rows:
        raise ValueError(f"{len(rows)} accepted validation rows exceeds the declared {max_rows}-row cap; no silent truncation.")
    for row in rows:
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError("Scope validation rows require explicit binary integer labels.")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Scope validation questions must be nonempty strings.")
        if row.get("correction_text") != correction_text(row["correction"]):
            raise ValueError("Validation correction text differs from the runtime/training input formatter.")
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Accepted validation rows must contain positive and negative pairs.")
    bindings = {"dataset_id": manifest["dataset_id"], "dataset_manifest_sha256": manifest_hash,
                "validation_file_sha256": file_sha256(dataset / "scope_validation.jsonl"),
                "review_registry_sha256": file_sha256(reviews),
                "scope_manifest_sha256": file_sha256(scope_artifact / "manifest.json"),
                "scope_selected_model_state_sha256": scope_manifest.get("selected_model_state_sha256"),
                "accepted_row_sha256": {row["id"]: content_hash(row) for row in rows},
                "source_validation_rows": len(original), "accepted_validation_rows": len(rows),
                "reviewed_rejected_rows": len(original) - len(rows), "max_rows": max_rows,
                "test_rows_read": False, "input_format_version": INPUT_FORMAT_VERSION}
    return rows, bindings


def _safe_output(output, scope_artifact, dataset, reviews):
    target = output.resolve()
    if output.exists():
        raise ValueError("Calibration output must be a fresh directory; no result overwrite.")
    for source in (scope_artifact.resolve(), dataset.resolve()):
        if target == source or target in source.parents or source in target.parents:
            raise ValueError("Calibration output must be separate from source dataset and model directories.")
    review_path = reviews.resolve()
    if target == review_path or target in review_path.parents:
        raise ValueError("Calibration output must not contain or replace the review registry.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=2000,
                        help="Hard cap on the complete accepted validation partition; no truncation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--preflight-only", action="store_true", help="Check inputs without loading models or writing output")
    args = parser.parse_args()
    try:
        _safe_output(args.output, args.scope_artifact, args.dataset, args.reviews)
        rows, bindings = reviewed_inputs(args.scope_artifact, args.dataset, args.reviews, max_rows=args.max_rows)
        if args.preflight_only:
            print(json.dumps({"preflight_valid": True, "model_execution_requested": False, **bindings}, indent=2))
            return
        # Heavyweight model packages load only after artifact/data/review preflight.
        from editguard.neural import ScopeCheckpoint
        scope = ScopeCheckpoint(args.scope_artifact, args.device)
        started = perf_counter()
        raw = [{"id": row["id"], "family_id": row["family_id"], "split": "validation", "label": row["label"],
                "scope_probability": scope.score(row["question"], correction_text(row["correction"])),
                "row_sha256": content_hash(row)} for row in rows]
        fit = fit_calibration([row["label"] for row in raw], [row["scope_probability"] for row in raw])
        # Prevent committing a calibration if on-disk inputs changed mid-run.
        current = {"scope_manifest_sha256": file_sha256(args.scope_artifact / "manifest.json"),
                   "dataset_manifest_sha256": file_sha256(args.dataset / "manifest.json"),
                   "validation_file_sha256": file_sha256(args.dataset / "scope_validation.jsonl"),
                   "review_registry_sha256": file_sha256(args.reviews)}
        if any(bindings[key] != value for key, value in current.items()):
            raise ValueError("Calibration inputs changed during inference; use the exact original snapshot and a fresh run.")
        validate_artifact(args.scope_artifact, "scope")
        artifact = {"schema_version": 1, "calibration_version": CALIBRATION_VERSION,
                    "purpose": "validation_calibration", "split": "validation", "status": "frozen",
                    "created_at": datetime.now(timezone.utc).isoformat(), **bindings,
                    "temperature": fit["temperature"], "threshold": fit["threshold"], "ambiguity_margin": fit["ambiguity_margin"],
                    "declared_costs": fit["declared_costs"], "validation_operating_metrics": fit["selected_operating_metrics"],
                    "selection": {"temperature_objective": "minimum_binary_negative_log_likelihood",
                                  "operating_objective": "minimum_declared_per_row_cost",
                                  "tie_break": "fewer_false_applications_then_higher_coverage_then_smaller_margin_then_lower_threshold",
                                  "temperatures": list(TEMPERATURES), "thresholds": list(THRESHOLDS), "margins": list(MARGINS)},
                    "device": args.device, "inference_and_selection_seconds": perf_counter() - started,
                    "script_sha256": file_sha256(Path(__file__)),
                    "disclaimer": "Development selection on validation data; this artifact does not establish test accuracy or guarantee reliability. Clarifications remain in denominators."}
        args.output.mkdir(parents=True)
        probability_path = args.output / "validation_probabilities.jsonl"
        probability_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in raw), encoding="utf-8")
        candidate_path = args.output / "selection_grid.json"
        candidate_path.write_text(json.dumps(fit, indent=2) + "\n", encoding="utf-8")
        artifact.update(probabilities_sha256=file_sha256(probability_path), selection_grid_sha256=file_sha256(candidate_path))
        temporary = args.output / "calibration.json.tmp"
        temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, args.output / "calibration.json")
        print(json.dumps({"output": str(args.output), "accepted_validation_rows": len(rows),
                          "temperature": artifact["temperature"], "threshold": artifact["threshold"],
                          "ambiguity_margin": artifact["ambiguity_margin"], "test_rows_read": False}, indent=2))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
