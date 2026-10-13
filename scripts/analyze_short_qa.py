"""Analyze saved validation outputs once; never runs inference or replaces scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard import scoring, short_qa
from editguard.short_qa import SHORT_QA_VERSION, score_short_qa


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def validate_records(rows, manifest, *, oracle=False):
    """Validate the matched saved comparison; no inference or truth inference."""
    configurations = ["oracle_answer"] if oracle else manifest["configurations"]
    if not isinstance(configurations, list) or len(set(configurations)) != len(configurations):
        raise ValueError("Unique configuration names are required.")
    identities, truth_by_id, sets = set(), {}, {c: set() for c in configurations}
    for record in rows:
        identity = (record.get("configuration"), record.get("id"))
        if identity[0] not in sets or not isinstance(identity[1], str) or not identity[1] or identity in identities:
            raise ValueError("Unknown configuration or duplicate/missing prediction identity.")
        identities.add(identity)
        sets[identity[0]].add(identity[1])
        if record.get("review_policy") != "automated_development" or record.get("human_reviewed_data") is not False:
            raise ValueError("This report is restricted to explicitly automated development records.")
        if (record.get("score") or {}).get("scorer_version") != manifest["scorer_version"]:
            raise ValueError("Original scorer identity is inconsistent.")
        if record.get("row_sha256") != manifest.get("accepted_row_sha256", {}).get(record["id"]):
            raise ValueError("Prediction is not bound to the accepted validation row hash.")
        truth = _canonical({key: record.get(key) for key in
                            ("question", "request_context", "expected_fact", "expected_polarity",
                             "question_type", "applicable", "row_sha256")})
        if record["id"] in truth_by_id and truth_by_id[record["id"]] != truth:
            raise ValueError("Configurations disagree about question truth or request context.")
        truth_by_id[record["id"]] = truth
    if not rows:
        raise ValueError("No predictions.")
    if not oracle and (len(truth_by_id) != manifest["questions"] or
                       any(values != next(iter(sets.values())) for values in sets.values())):
        raise ValueError("Every configuration must have all recorded questions.")
    if oracle:
        expected_ids = {r["id"] for r in rows if r.get("applicable") is True}
        if expected_ids != sets["oracle_answer"]:
            raise ValueError("Oracle diagnostic contains a nonapplicable case.")
    return truth_by_id


def _counts(rows):
    verdicts = Counter(row["short_qa"]["verdict"] for row in rows)
    primary = Counter("correct" if row["score"]["correct"] is True else
                      "incorrect" if row["score"]["correct"] is False else "pending_uncredited"
                      for row in rows)
    n = len(rows)
    return {"denominator": n, "correct": verdicts["correct"],
            "incorrect": verdicts["incorrect"], "pending_uncredited": verdicts["pending_uncredited"],
            "accuracy_including_all_nonanswers_and_pending": verdicts["correct"] / n if n else None,
            "primary_counts_unchanged": dict(primary),
            "proves_exact_scoped_fact": sum(row["short_qa"]["proves_exact_scoped_fact"] for row in rows),
            "evidence_counts": dict(Counter(row["short_qa"]["evidence"] or "none" for row in rows)),
            "answer_kind_counts": dict(Counter(row.get("answer_kind", "unspecified") for row in rows)),
            "status_counts": dict(Counter(row.get("status", "answered") for row in rows))}


def summarize(rows):
    result = {}
    for configuration in sorted({r["configuration"] for r in rows}):
        selected = [r for r in rows if r["configuration"] == configuration]
        slices = {
            "all_requested_qa": selected,
            "requested_value": [r for r in selected if r["short_qa"]["task"] == "requested_value"],
            "comparison_predicate": [r for r in selected if r["short_qa"]["task"] == "comparison_predicate"],
            "applicable_direct": [r for r in selected if r.get("applicable") and r.get("question_type") == "direct"],
            "applicable_paraphrase": [r for r in selected if r.get("applicable") and r.get("question_type") == "paraphrase"],
            "applicable_comparison": [r for r in selected if r.get("applicable") and r.get("question_type") == "comparison"],
            "out_of_scope": [r for r in selected if r.get("applicable") is False],
        }
        result[configuration] = {name: _counts(values) for name, values in slices.items()}
    baseline = {r["id"]: r for r in rows if r["configuration"] == "unchanged_base" and
                r.get("applicable") is False and r["short_qa"]["correct"] is True}
    for configuration in result:
        selected = [r for r in rows if r["configuration"] == configuration and r["id"] in baseline]
        result[configuration]["retention_of_baseline_qa_correct_out_of_scope"] = {
            "denominator": len(selected), "correct": sum(r["short_qa"]["correct"] is True for r in selected),
            "definition": "QA correctness on the subset the unchanged base already answered correctly; all selected outcomes remain in the denominator."}
    return result


def analyze_directory(input_dir: Path, output_dir: Path):
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if output_dir.exists() or output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output must be a fresh directory separate from the immutable source.")
    inputs = {name: input_dir / name for name in
              ("manifest.json", "metrics.json", "predictions.jsonl", "oracle_predictions.jsonl")}
    source_hashes = {name: _hash(path) for name, path in inputs.items()}
    manifest = json.loads(inputs["manifest.json"].read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed" or manifest.get("purpose") != "validation_development" or
            manifest.get("test_rows_read") is not False or
            manifest.get("review_policy") != "automated_development" or
            manifest.get("human_reviewed_data") is not False):
        raise ValueError("Only completed automated development validation runs are permitted.")
    if manifest.get("predictions_sha256") != source_hashes["predictions.jsonl"]:
        raise ValueError("Original predictions hash does not match the recorded manifest.")
    if manifest.get("scorer_version") != scoring.SCORER_VERSION or manifest.get("scorer_sha256") != _hash(Path(scoring.__file__)):
        raise ValueError("Original scorer source must remain byte-identical for this analysis.")
    rows = _read_rows(inputs["predictions.jsonl"])
    oracle = _read_rows(inputs["oracle_predictions.jsonl"])
    validate_records(rows, manifest)
    validate_records(oracle, manifest, oracle=True)
    applicable_ids = {r["id"] for r in rows if r.get("applicable") is True}
    if {r["id"] for r in oracle} != applicable_ids:
        raise ValueError("The oracle diagnostic must include every applicable comparison case exactly once.")
    scored = [{**row, "short_qa": score_short_qa(row)} for row in rows]
    scored_oracle = [{**row, "short_qa": score_short_qa(row)} for row in oracle]
    summary = {"schema_version": 1, "purpose": "exploratory_validation_short_qa",
               "diagnostic_version": SHORT_QA_VERSION,
               "primary_results_replaced": False, "post_hoc_policy": True,
               "human_semantic_review": False, "review_policy": "automated_development",
               "test_rows_read": False, "unique_questions": manifest["questions"],
               "unique_question_families": sorted({r.get("family_id") for r in rows}),
               "prediction_records": len(rows), "oracle_records": len(oracle),
               "configurations": summarize(scored), "oracle": summarize(scored_oracle),
               "source_sha256": source_hashes,
               "limitations": ["Post-hoc development diagnostic; not a frozen test result or replacement for the primary full-fact scorer.",
                               "Question scope supplies omitted subject, property and unit; bare outputs do not prove a complete scoped fact endorsement.",
                               "A bare Yes/No establishes only the requested comparison predicate, never the exact limit.",
                               "Unsupported prose and question forms remain pending and uncredited; failures and abstentions stay in all-case denominators.",
                               "Two held-out entity families with shared templates do not establish unfamiliar-wording or general-dialogue reliability.",
                               "Training and evaluation labels received automated development review, not independent human review.",
                               "Preservation is reported only on baseline-QA-correct out-of-scope cases; it is separate from overall out-of-scope correctness."]}
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, values in (("predictions.jsonl", scored), ("oracle_predictions.jsonl", scored_oracle)):
        with (output_dir / name).open("x", encoding="utf-8", newline="\n") as handle:
            for row in values:
                handle.write(_canonical(row) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    with (output_dir / "summary.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["population", "configuration", "slice", "denominator", "correct", "incorrect", "pending_uncredited", "accuracy"])
        for population, data in (("end_to_end", summary["configurations"]), ("oracle", summary["oracle"])):
            for configuration, slices in data.items():
                for slice_name, counts in slices.items():
                    if "accuracy_including_all_nonanswers_and_pending" in counts:
                        writer.writerow([population, configuration, slice_name, counts["denominator"], counts["correct"], counts["incorrect"], counts["pending_uncredited"], counts["accuracy_including_all_nonanswers_and_pending"]])
    if source_hashes != {name: _hash(path) for name, path in inputs.items()}:
        raise RuntimeError("Original run files changed during read-only analysis.")
    derived = {name: _hash(output_dir / name) for name in
               ("predictions.jsonl", "oracle_predictions.jsonl", "summary.json", "summary.csv")}
    report_manifest = {"schema_version": 1, "status": "completed", "purpose": "exploratory_validation_short_qa",
                       "generated_at": datetime.now(timezone.utc).isoformat(),
                       "diagnostic_version": SHORT_QA_VERSION, "diagnostic_source_sha256": _hash(Path(short_qa.__file__)),
                       "analysis_source_sha256": _hash(Path(__file__)), "primary_scorer_version": manifest["scorer_version"],
                       "primary_scorer_sha256": manifest["scorer_sha256"],
                       "input_directory": str(input_dir), "source_sha256": source_hashes,
                       "output_sha256": derived, "source_files_unchanged": True,
                       "training_or_inference_executed": False, "human_semantic_review": False,
                       "test_rows_read": False, "post_hoc_policy": True,
                       "prediction_records": len(rows), "oracle_records": len(oracle)}
    (output_dir / "manifest.json").write_text(json.dumps(report_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze_directory(args.input, args.output)
    print(json.dumps({"output": str(args.output), "questions": summary["unique_questions"],
                      "records": summary["prediction_records"], "oracle": summary["oracle_records"],
                      "primary_results_replaced": False}))


if __name__ == "__main__":
    main()
