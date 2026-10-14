"""Select a scope operating point from frozen validation scores after routing gates.

No model executes here. Temperature still fits raw-pair NLL; operating-point
selection mirrors the product/plan/version/date/approval gate before the scope
uncertainty decision. The archived raw-pair selector and records stay unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.neural import calibrated_probability, file_sha256, validate_artifact
from scripts.calibrate_scope import (COSTS, MARGINS, TEMPERATURES, THRESHOLDS,
    _rate, _safe_output, _validated_pairs, operating_metrics as raw_operating_metrics,
    reviewed_inputs as reviewed_scope_inputs)
from scripts.calibrate_similarity import pair_eligible
from scripts.prepare_data import REVIEW_POLICIES, content_hash
from scripts.review_dataset import _unique_object, read_json

VERSION = "editguard-scope-runtime-validation-v1"
ROUTING_POLICY = "approved_compatible_then_scope_uncertainty_v1"
SOURCE_NAMES = ("calibration.json", "selection_grid.json", "validation_probabilities.jsonl")
POLICY_SOURCES = ("editguard/comparison.py", "editguard/neural.py", "editguard/formatting.py",
                  "scripts/calibrate_scope.py", "scripts/calibrate_similarity.py")
BINDING_FIELDS = ("dataset_id", "dataset_manifest_sha256", "validation_file_sha256",
    "review_registry_sha256", "review_policy", "human_reviewed_data", "review_audit_sha256",
    "scope_manifest_sha256", "scope_selected_model_state_sha256", "scope_training_review_policy",
    "scope_training_human_reviewed_data", "accepted_row_sha256", "source_validation_rows",
    "accepted_validation_rows", "reviewed_rejected_rows", "test_rows_read", "input_format_version")


def operating_metrics(labels, probabilities, threshold, margin, eligibility=None):
    """Count every pair; an incompatible pair is rejected before uncertainty."""
    raw = raw_operating_metrics(labels, probabilities, threshold, margin)
    if eligibility is None:
        return raw
    labels, probabilities = _validated_pairs(labels, probabilities)
    eligibility = list(eligibility)
    if len(eligibility) != len(labels) or any(type(flag) is not bool for flag in eligibility):
        raise ValueError("Every pair requires an explicit boolean compatibility flag.")
    counts = {key: value if key in ("total", "positive", "negative") else 0
              for key, value in raw["counts"].items()}
    for label, probability, eligible in zip(labels, probabilities, eligibility):
        if not eligible:
            counts["missed_positive" if label else "true_rejection"] += 1
        elif abs(probability - threshold) <= margin:
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
        "eligible_pairs": sum(eligibility), "ineligible_pairs": len(eligibility) - sum(eligibility),
        "total_cost": cost, "mean_cost": _rate(cost, counts["total"]),
        "coverage": _rate(covered, counts["total"]), "clarification_rate": _rate(clarification, counts["total"]),
        "accuracy_including_clarification_as_unsuccessful": _rate(correct, counts["total"]),
        "selective_accuracy": _rate(correct, covered),
        "precision": _rate(counts["true_application"], counts["true_application"] + counts["false_application"]),
        "recall_including_clarification_as_missed": _rate(counts["true_application"], counts["positive"]),
        "false_application_rate": _rate(counts["false_application"], counts["negative"])}


def fit_calibration(labels, probabilities, eligibility):
    """Reuse the original finite grids/costs; no tuning of the search space."""
    labels, probabilities = _validated_pairs(labels, probabilities)
    eligibility = list(eligibility)
    if len(eligibility) != len(labels) or any(type(flag) is not bool for flag in eligibility):
        raise ValueError("Every pair requires an explicit boolean compatibility flag.")
    temperatures = []
    for temperature in TEMPERATURES:
        transformed = [calibrated_probability(p, temperature) for p in probabilities]
        nll = -sum(math.log(max(min(p if label else 1 - p, 1 - 1e-15), 1e-15))
                   for label, p in zip(labels, transformed)) / len(labels)
        temperatures.append({"temperature": temperature, "negative_log_likelihood": nll})
    selected_temperature = min(temperatures, key=lambda item: (item["negative_log_likelihood"],
        abs(item["temperature"] - 1), item["temperature"]))
    calibrated = [calibrated_probability(p, selected_temperature["temperature"]) for p in probabilities]
    candidates = [operating_metrics(labels, calibrated, threshold, margin, eligibility)
                  for threshold in THRESHOLDS for margin in MARGINS]
    selected = min(candidates, key=lambda item: (item["total_cost"], item["counts"]["false_application"],
        -item["coverage"]["numerator"], item["ambiguity_margin"], item["threshold"]))
    raw = raw_operating_metrics(labels, calibrated, selected["threshold"], selected["ambiguity_margin"])
    return {"temperature": selected_temperature["temperature"], "threshold": selected["threshold"],
        "ambiguity_margin": selected["ambiguity_margin"], "selected_operating_metrics": selected,
        "compatibility_gated_metrics": selected, "raw_pair_metrics": raw,
        "temperature_candidates": temperatures, "operating_candidates": candidates,
        "declared_costs": dict(COSTS), "rows": len(labels)}


def frozen_inputs(scope_artifact, dataset, reviews, source_calibration, *, max_rows, review_policy="human"):
    """Validate reviewed rows and all frozen score provenance without ML imports."""
    scope_artifact, dataset, reviews, source_calibration = map(Path,
        (scope_artifact, dataset, reviews, source_calibration))
    if source_calibration.name != "calibration.json":
        raise ValueError("Source must be the archived calibration.json with its hashed sibling records.")
    rows, bindings = reviewed_scope_inputs(scope_artifact, dataset, reviews,
                                          max_rows=max_rows, review_policy=review_policy)
    scope_manifest = read_json(scope_artifact / "manifest.json")
    if scope_manifest.get("reviews_sha256") != bindings["review_registry_sha256"]:
        raise ValueError("Current review registry differs from the registry bound to scope training.")
    if scope_manifest.get("review_policy", "human") != review_policy:
        raise ValueError("Scope training and calibration review policies must agree.")
    source = read_json(source_calibration)
    if (source.get("purpose") != "validation_calibration" or source.get("split") != "validation"
            or source.get("status") != "frozen" or source.get("test_rows_read") is not False):
        raise ValueError("Source scope probabilities must be frozen validation-only evidence.")
    if source.get("script_sha256") != file_sha256(ROOT / "scripts/calibrate_scope.py"):
        raise ValueError("Archived score source differs from the unchanged calibration implementation.")
    for field in BINDING_FIELDS:
        if source.get(field) != bindings.get(field):
            raise ValueError(f"Archived score provenance differs from current reviewed inputs: {field}.")
    probability_path = source_calibration.parent / "validation_probabilities.jsonl"
    grid_path = source_calibration.parent / "selection_grid.json"
    for field, path in (("probabilities_sha256", probability_path), ("selection_grid_sha256", grid_path)):
        if not path.is_file() or source.get(field) != file_sha256(path):
            raise ValueError(f"Archived score evidence integrity failed: {path.name}.")
    grid = read_json(grid_path)
    if (grid.get("rows") != len(rows) or grid.get("declared_costs") != COSTS
            or len(grid.get("operating_candidates", [])) != len(THRESHOLDS) * len(MARGINS)
            or len(grid.get("temperature_candidates", [])) != len(TEMPERATURES)
            or any(grid.get(key) != source.get(key) for key in ("temperature", "threshold", "ambiguity_margin"))):
        raise ValueError("Archived calibration grid does not match the fixed population or declared selection.")
    if source.get("selection", {}).get("temperatures") != list(TEMPERATURES) or (
            source.get("selection", {}).get("thresholds") != list(THRESHOLDS)) or (
            source.get("selection", {}).get("margins") != list(MARGINS)):
        raise ValueError("Archived calibration uses a different declared finite search space.")
    raw = [json.loads(line, object_pairs_hook=_unique_object)
           for line in probability_path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["id"]: row for row in rows}
    seen = set()
    for record in raw:
        row_id = record.get("id")
        if row_id in seen or row_id not in by_id:
            raise ValueError("Archived probabilities contain duplicate or unknown validation row IDs.")
        row = by_id[row_id]
        if (record.get("split") != "validation" or record.get("row_sha256") != content_hash(row)
                or type(record.get("label")) is not int or record["label"] != row["label"]
                or record.get("family_id") != row["family_id"] or record.get("review_policy") != review_policy
                or record.get("human_reviewed_data") is not (review_policy == "human")):
            raise ValueError("Archived probability row disagrees with exact reviewed validation provenance.")
        seen.add(row_id)
    if len(raw) != len(rows) or seen != set(by_id):
        raise ValueError("Archived probabilities must cover every accepted validation pair exactly once.")
    _validated_pairs([record["label"] for record in raw], [record.get("scope_probability") for record in raw])
    eligible = [pair_eligible(by_id[record["id"]]) for record in raw]
    bindings.update(source_calibration_sha256=file_sha256(source_calibration),
        source_probabilities_sha256=file_sha256(probability_path), source_selection_grid_sha256=file_sha256(grid_path),
        policy_source_sha256={name: file_sha256(ROOT / name) for name in POLICY_SOURCES},
        population="single_focal_question_correction_pairs", routing_policy=ROUTING_POLICY,
        model_execution_requested=False, optimizer_updates=0, eligible_pairs=sum(eligible),
        ineligible_pairs=len(eligible) - sum(eligible))
    return rows, raw, eligible, bindings


def safe_output(output, scope_artifact, dataset, reviews, source_calibration):
    _safe_output(Path(output), Path(scope_artifact), Path(dataset), Path(reviews))
    target, source = Path(output).resolve(), Path(source_calibration).parent.resolve()
    if target == source or target in source.parents or source in target.parents:
        raise ValueError("New calibration must remain separate from the immutable source calibration.")


def select_from_frozen(scope_artifact, dataset, reviews, source_calibration, output, *, max_rows=120,
                       review_policy="human", preflight_only=False):
    output = Path(output)
    safe_output(output, scope_artifact, dataset, reviews, source_calibration)
    rows, raw, eligible, bindings = frozen_inputs(scope_artifact, dataset, reviews, source_calibration,
        max_rows=max_rows, review_policy=review_policy)
    if preflight_only:
        return {"preflight_valid": True, **bindings}
    started = perf_counter()
    fit = fit_calibration([record["label"] for record in raw],
                         [record["scope_probability"] for record in raw], eligible)
    # Re-read all bound sources before publication; no changed data/source is accepted.
    _, current_raw, current_eligible, current = frozen_inputs(scope_artifact, dataset, reviews,
        source_calibration, max_rows=max_rows, review_policy=review_policy)
    if current != bindings or current_raw != raw or current_eligible != eligible:
        raise ValueError("Frozen selection inputs changed during selection; no artifact published.")
    validate_artifact(Path(scope_artifact), "scope")
    artifact = {"schema_version": 1, "calibration_version": VERSION, "purpose": "validation_calibration",
        "split": "validation", "status": "frozen", "created_at": datetime.now(timezone.utc).isoformat(), **bindings,
        "temperature": fit["temperature"], "threshold": fit["threshold"], "ambiguity_margin": fit["ambiguity_margin"],
        "declared_costs": fit["declared_costs"], "validation_operating_metrics": fit["compatibility_gated_metrics"],
        "compatibility_gated_metrics": fit["compatibility_gated_metrics"], "raw_pair_metrics": fit["raw_pair_metrics"],
        "selection": {"temperature_objective": "minimum_raw_pair_binary_negative_log_likelihood",
            "operating_objective": "minimum_compatibility_gated_declared_per_row_cost",
            "tie_break": "fewer_false_applications_then_higher_coverage_then_smaller_margin_then_lower_threshold",
            "temperatures": list(TEMPERATURES), "thresholds": list(THRESHOLDS), "margins": list(MARGINS)},
        "selection_seconds": perf_counter() - started, "script_sha256": file_sha256(Path(__file__)),
        "disclaimer": "Validation-informed correction of the operating selector to match existing routing gates. "
            "Temperature fits raw pairs; raw failures stay reported separately. Single focal pairs do not measure "
            "retrieval, multi-correction competition, conflicts, generated answers or final-test accuracy. "
            "Same validation rows select and report the point; clarify/abstain outcomes remain in every denominator."}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name("." + output.name + ".pending-" + uuid4().hex)
    staging.mkdir()
    try:
        # Preserve source raw probabilities byte-for-byte, including their original review provenance.
        (staging / "validation_probabilities.jsonl").write_bytes(
            (Path(source_calibration).parent / "validation_probabilities.jsonl").read_bytes())
        grid = {**fit, "routing_policy": ROUTING_POLICY, "review_policy": review_policy,
            "human_reviewed_data": review_policy == "human", "review_registry_sha256": bindings["review_registry_sha256"]}
        (staging / "selection_grid.json").write_text(json.dumps(grid, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        artifact.update(probabilities_sha256=file_sha256(staging / "validation_probabilities.jsonl"),
                        selection_grid_sha256=file_sha256(staging / "selection_grid.json"))
        (staging / "calibration.json").write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Catch late source changes as well as races on the fresh output directory.
        _, final_raw, final_eligible, final = frozen_inputs(scope_artifact, dataset, reviews, source_calibration,
            max_rows=max_rows, review_policy=review_policy)
        if final != bindings or final_raw != raw or final_eligible != eligible or output.exists():
            raise ValueError("Sources or output changed before publication; no overwrite is permitted.")
        if artifact["probabilities_sha256"] != bindings["source_probabilities_sha256"]:
            raise ValueError("Source probability bytes changed during copying.")
        os.rename(staging, output)
    finally:
        if staging.exists():
            resolved = staging.resolve()
            if resolved.parent != output.parent.resolve() or resolved.name != staging.name:
                raise ValueError("Staging cleanup escaped the intended output parent.")
            shutil.rmtree(resolved)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--source-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-policy", choices=REVIEW_POLICIES, default="human")
    parser.add_argument("--max-rows", type=int, default=120)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        artifact = select_from_frozen(args.scope_artifact, args.dataset, args.reviews, args.source_calibration,
            args.output, max_rows=args.max_rows, review_policy=args.review_policy, preflight_only=args.preflight_only)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({key: artifact[key] for key in ("dataset_id", "accepted_validation_rows", "review_policy",
        "human_reviewed_data", "test_rows_read", "model_execution_requested", "eligible_pairs", "ineligible_pairs")}, indent=2))
    if not args.preflight_only:
        print(json.dumps({key: artifact[key] for key in ("temperature", "threshold", "ambiguity_margin",
            "raw_pair_metrics", "compatibility_gated_metrics")}, indent=2))


if __name__ == "__main__":
    main()
