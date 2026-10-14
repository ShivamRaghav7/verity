"""Export a hash-bound development dashboard summary without model execution."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from editguard.comparison import CONFIGURATIONS
from editguard.neural import file_sha256, validate_artifact


def build(run, config_path, output):
    if output.exists():
        raise ValueError("Use a fresh summary path; recorded evidence is not overwritten.")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed" or manifest.get("purpose") != "validation_development"
            or manifest.get("human_reviewed_data") is not False or manifest.get("review_policy") != "automated_development"
            or manifest.get("test_rows_read") is not False or manifest.get("base_weights_unchanged") is not True
            or manifest.get("configuration_sha256") != file_sha256(config_path)
            or manifest.get("predictions_sha256") != file_sha256(run / "predictions.jsonl")):
        raise ValueError("Only completed immutable development evidence is eligible.")
    records = [json.loads(line) for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    questions = manifest["questions"]
    if (set(metrics["configurations"]) != set(CONFIGURATIONS)
            or len(records) != questions * len(CONFIGURATIONS) or metrics["record_count"] != len(records)):
        raise ValueError("All five matched configurations must retain the full population.")
    artifacts = {}
    for task in ("scope", "answer"):
        root = Path(config[f"{task}_artifact"])
        model = validate_artifact(root, task)
        artifacts[task] = {"manifest_sha256": file_sha256(root / "manifest.json"),
                           "completed_steps": model["completed_steps"], "selected_step": model["selection"]["global_step"]}
    table = []
    for name in CONFIGURATIONS:
        current = metrics["configurations"][name]
        if current["count"] != questions:
            raise ValueError("A configuration has a different question denominator.")
        table.append({"configuration": name, "accuracy": current["accuracy"], "coverage": current["coverage"],
                      "out_of_scope_accuracy": current["subsets"]["out_of_scope"]["accuracy"],
                      "baseline_correct_retention": current["baseline_correct_retention"],
                      "false_application_rate": current["false_correction_application_rate"],
                      "p50_ms": current["latency"]["p50_ms"], "p95_ms": current["latency"]["p95_ms"],
                      "pending_review": current["pending_review"], "errors": current["errors"], "unavailable": current["unavailable"]})
    oracle = metrics["oracle_answer_diagnostic"]["configurations"]["oracle_answer"]
    scope = metrics["configurations"]["trained_scope_editor"]["scope"]
    calibration_path = Path(config["calibration_artifact"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    scope_policy = None
    if calibration.get("routing_policy"):
        if calibration.get("scope_manifest_sha256") != artifacts["scope"]["manifest_sha256"]:
            raise ValueError("Routing policy is bound to a different scope checkpoint.")
        scope_policy = {key: calibration[key] for key in (
            "temperature", "threshold", "ambiguity_margin", "routing_policy", "population",
            "raw_pair_metrics", "compatibility_gated_metrics", "scope_manifest_sha256")}
        scope_policy.update(calibration_path=calibration_path.as_posix(),
                            calibration_sha256=file_sha256(calibration_path))
    summary = {"schema_version": 1, "artifact_type": "editguard_development_evaluation", "purpose": "validation_development",
               "status": "completed", "review_policy": "automated_development", "human_reviewed_data": False,
               "test_rows_read": False, "dataset_id": manifest["dataset_id"], "questions": questions,
               "oracle_questions": oracle["count"], "model_artifacts": artifacts,
               "scope": {key: scope[key] for key in ("precision", "recall", "false_positive_rate", "brier", "ece")},
               "configurations": table,
               "oracle": {"accuracy": oracle["accuracy"], "coverage": oracle["coverage"],
                          "direct_accuracy": oracle["subsets"]["direct"]["accuracy"],
                          "paraphrase_accuracy": oracle["subsets"]["paraphrase"]["accuracy"],
                          "comparison_accuracy": oracle["subsets"]["comparison"]["accuracy"]},
               "provenance": {"config_sha256": file_sha256(config_path), "scorer_version": manifest["scorer_version"],
                              "raw_manifest_path": (run / "manifest.json").as_posix(), "raw_manifest_sha256": file_sha256(run / "manifest.json"),
                              "predictions_sha256": file_sha256(run / "predictions.jsonl"), "metrics_sha256": file_sha256(run / "metrics.json"),
                              "base_weights_unchanged": manifest["base_weights_unchanged"]},
               "limitations": ["Development validation on two entity families and shared question forms; no final-test claim.",
                  "Labels have automated development review, not independent human approval.",
                  "Strict scores require complete scoped fact endorsements. Pending fragments remain uncredited; zero credited is not zero factual correctness.",
                  "Short-answer QA is a separate post-hoc diagnostic and cannot replace these primary scores.",
                  "Scope temperature/threshold were selected on the same validation population; estimates are optimistic for generalization. Raw threshold cards omit the uncertainty band and deterministic guards; the separate policy panel includes them.",
                  "CLI latency excludes model loading, scoring, extra scope diagnostics and HTTP. Fixed order/shared caches prevent a controlled independent latency comparison.",
                  "Baseline-correct retention is undefined when no baseline complete assertions are automatically credited.",
                  f"Oracle diagnostic: {oracle['accuracy']['numerator']}/{oracle['count']} credited complete assertions; {oracle['pending_review']} outputs pending review and {oracle['abstentions']} abstentions remain in the denominator."]}
    if scope_policy:
        summary["scope_policy"] = scope_policy
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = build(args.run, args.config, args.output)
        print(json.dumps({"output": str(args.output), "questions": summary["questions"],
                          "oracle_questions": summary["oracle_questions"], "human_reviewed_data": False}, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
