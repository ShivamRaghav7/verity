"""Rescore saved outputs and export honest metrics, CSV and blinded review queues.

This command runs no inference, training, threshold selection or model downloads.
Optional human decisions require two independent agreeing reviewers and are bound
to the exact input record, answer, truth and scorer source hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard import scoring
from editguard.metrics import aggregate_predictions
from editguard.scoring import SCORER_VERSION, canonical_endorsement, score_answer


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def scorer_hash():
    return hashlib.sha256(Path(scoring.__file__).read_bytes()).hexdigest()


def review_hash(record):
    original = {key: value for key, value in record.items()
                if key not in {"score", "automatic_score", "human_review"}}
    return hashlib.sha256(canonical_bytes({"record": original,
                                         "scorer_version": SCORER_VERSION,
                                         "scorer_sha256": scorer_hash()})).hexdigest()


def score_records(records):
    """Preserve text/inputs and recompute scores under the current locked grammar."""
    scored = []
    for record in records:
        result = dict(record)
        if result.get("status", "answered") not in {"answered", "ok", "success", "generated"}:
            result["score"] = {"correct": False, "verdict": "not_answered",
                               "reason": "Unavailable, error or abstained result receives no factual-answer credit.",
                               "endorsed_fact": None, "needs_review": False,
                               "answer_kind": result.get("answer_kind", "unspecified"),
                               "scorer_version": SCORER_VERSION}
        elif result.get("expected_fact"):
            result["score"] = score_answer(result.get("answer"), result["expected_fact"],
                                           aliases=result.get("entity_aliases", result.get("aliases", ())),
                                           expected_polarity=result.get("expected_polarity"),
                                           answer_kind=result.get("answer_kind", "unspecified"))
        else:
            result["score"] = {"correct": False, "verdict": "not_factual",
                               "reason": "No answerable fact; handling outcomes require separate reviewed labels.",
                               "endorsed_fact": None, "needs_review": False,
                               "answer_kind": result.get("answer_kind", "unspecified"),
                               "scorer_version": SCORER_VERSION}
        scored.append(result)
    return scored


def review_queue(records):
    """Blind configuration and routing evidence; keep exact question and truth."""
    queue = []
    for record in records:
        if not (record.get("score") or {}).get("needs_review"):
            continue
        digest = review_hash(record)
        queue.append({"review_id": f"review:{digest}", "row_sha256": digest,
                      "scorer_version": SCORER_VERSION, "scorer_sha256": scorer_hash(),
                      "question": record.get("question"),
                      "expected_fact": record.get("expected_fact"),
                      "expected_entity_aliases": record.get("entity_aliases", record.get("aliases", ())),
                      "expected_polarity": record.get("expected_polarity"),
                      "answer": record.get("answer"),
                      "automatic_review_reason": record["score"]["reason"]})
    return queue


def apply_adjudications(records, registry):
    """Accept only current-hash, timestamped, reasoned, independent agreement.

    A single decision or any disagreement stays pending. This validates provenance
    structure; it does not authenticate a person's identity or certify expertise.
    """
    if not isinstance(registry, dict) or registry.get("schema_version") != 1 or not isinstance(registry.get("decisions"), list):
        raise ValueError("Adjudication registry requires schema_version=1 and a decisions list.")
    eligible = {f"review:{review_hash(row)}": row for row in records
                if (row.get("score") or {}).get("needs_review")}
    decisions = {}
    reviewers = set()
    for decision in registry["decisions"]:
        if not isinstance(decision, dict):
            raise ValueError("Each human decision must be a dictionary.")
        identity = decision.get("review_id")
        if identity not in eligible or decision.get("row_sha256") != review_hash(eligible[identity]):
            raise ValueError("Human decision is stale, unmatched or addresses an automatically scored row.")
        reviewer = decision.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("A nonempty independent human reviewer identifier is required.")
        reviewer_key = (identity, reviewer.strip().casefold())
        if reviewer_key in reviewers:
            raise ValueError("Duplicate reviewer decisions are not independent votes.")
        reviewers.add(reviewer_key)
        verdict = decision.get("decision")
        if verdict not in ("correct", "incorrect") or not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
            raise ValueError("Human decisions need a correct/incorrect verdict and substantive reason.")
        try:
            timestamp = datetime.fromisoformat(decision["reviewed_at"].replace("Z", "+00:00"))
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Human review needs a timezone-aware ISO timestamp.") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Human review needs a timezone-aware ISO timestamp.")
        decisions.setdefault(identity, []).append(decision)
    result = []
    for record in records:
        row = dict(record)
        identity = f"review:{review_hash(record)}"
        votes = decisions.get(identity, [])
        if votes:
            agree = len({vote["decision"] for vote in votes}) == 1
            complete = len(votes) >= 2 and agree
            row["human_review"] = {"review_id": identity, "row_sha256": review_hash(record),
                                   "status": "adjudicated" if complete else "disagreement" if not agree else "pending_second_reviewer",
                                   "decisions": votes}
            if complete:
                correct = votes[0]["decision"] == "correct"
                row["automatic_score"] = dict(row["score"])
                row["score"] = {**row["score"], "correct": correct,
                                "verdict": "correct" if correct else "incorrect",
                                "needs_review": False, "human_adjudicated": True,
                                "reason": "Two or more independent, agreeing human reviewers adjudicated the exact hash-bound output."}
                if correct:
                    row["score"]["endorsed_fact"] = canonical_endorsement(
                        row["expected_fact"], polarity=row.get("expected_polarity"))
        result.append(row)
    return result


def metric_rows(report):
    """Long-form CSV rows expose every nested rate's numerator and denominator."""
    flattened = []

    def walk(value, prefix, config):
        if isinstance(value, dict):
            if {"numerator", "denominator", "value"} <= value.keys():
                flattened.append({"configuration": config, "metric": prefix,
                                  "numerator": value["numerator"], "denominator": value["denominator"],
                                  "value": value["value"]})
            else:
                for key, child in value.items():
                    walk(child, f"{prefix}.{key}" if prefix else key, config)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened.append({"configuration": config, "metric": prefix,
                              "numerator": "", "denominator": "", "value": value})

    for configuration, metrics in report["configurations"].items():
        walk(metrics, "", configuration)
    return flattened


def aggregate_file(input_path, output_dir, *, adjudication_path=None,
                   bootstrap_seed=42, bootstrap_resamples=2000):
    input_path, output_dir = Path(input_path), Path(output_dir)
    raw = input_path.read_bytes()
    records = []
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed prediction JSONL at line {line_number}.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Prediction line {line_number} must be an object.")
        records.append(row)
    scored = score_records(records)
    if adjudication_path:
        registry = json.loads(Path(adjudication_path).read_text(encoding="utf-8-sig"))
        scored = apply_adjudications(scored, registry)
    report = aggregate_predictions(scored, bootstrap_seed=bootstrap_seed,
                                   bootstrap_resamples=bootstrap_resamples)
    report["aggregation"] = {"input_path": str(input_path.resolve()),
                             "input_sha256": hashlib.sha256(raw).hexdigest(),
                             "scorer_version": SCORER_VERSION, "scorer_sha256": scorer_hash(),
                             "generated_at": datetime.now(timezone.utc).isoformat(),
                             "inference_executed": False, "model_selection_executed": False,
                             "human_adjudications": sum(bool((row.get("score") or {}).get("human_adjudicated")) for row in scored),
                             "human_disagreements": sum((row.get("human_review") or {}).get("status") == "disagreement" for row in scored)}
    outputs = {name: output_dir / name for name in ("report.json", "metrics.csv", "scored_predictions.jsonl", "review_queue.jsonl")}
    if input_path.resolve() in {path.resolve() for path in outputs.values()}:
        raise ValueError("Derived exports cannot overwrite the raw prediction input.")
    if adjudication_path and Path(adjudication_path).resolve() in {path.resolve() for path in outputs.values()}:
        raise ValueError("Derived exports cannot overwrite human review evidence.")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs["report.json"].write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, values in (("scored_predictions.jsonl", scored), ("review_queue.jsonl", review_queue(scored))):
        outputs[name].write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in values))
    with outputs["metrics.csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("configuration", "metric", "numerator", "denominator", "value"))
        writer.writeheader()
        writer.writerows(metric_rows(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Saved per-question predictions JSONL.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--adjudications", type=Path, help="Optional hash-bound independent human decisions registry.")
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    try:
        report = aggregate_file(args.input, args.output_dir,
                                adjudication_path=args.adjudications,
                                bootstrap_seed=args.bootstrap_seed,
                                bootstrap_resamples=args.bootstrap_resamples)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "questions": report["question_count"], "records": report["record_count"],
                      "configurations": list(report["configurations"]),
                      "pending_reviews": sum(value["pending_review"] for value in report["configurations"].values()),
                      "note": "Aggregation only; no inference or model selection executed."}, indent=2))


if __name__ == "__main__":
    main()
