"""Create a provenance-bound development rescore without rerunning inference."""

from __future__ import annotations

import argparse
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
from scripts.aggregate_evaluation import score_records
from scripts.analyze_short_qa import validate_records


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def rescore(source, output, source_scorer):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("Use a fresh output outside the immutable source run.")
    names = ("manifest.json", "metrics.json", "predictions.jsonl", "oracle_predictions.jsonl")
    hashes = {name: file_hash(source / name) for name in names}
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("purpose") != "validation_development" or manifest.get("status") != "completed"
            or manifest.get("review_policy") != "automated_development"
            or manifest.get("human_reviewed_data") is not False or manifest.get("test_rows_read") is not False
            or manifest.get("base_weights_unchanged") is not True
            or manifest.get("predictions_sha256") != hashes["predictions.jsonl"]
            or manifest.get("scorer_sha256") != file_hash(source_scorer)):
        raise ValueError("Completed development run and exact archived source scorer are required.")
    original, oracle = rows(source / "predictions.jsonl"), rows(source / "oracle_predictions.jsonl")
    validate_records(original, manifest)
    validate_records(oracle, manifest, oracle=True)
    if {r["id"] for r in oracle} != {r["id"] for r in original if r.get("applicable") is True}:
        raise ValueError("Oracle must retain every applicable case exactly once.")

    def scored(records):
        result = score_records(records)
        for old, new in zip(records, result):
            new["source_score"] = old["score"]
            if {k: v for k, v in new.items() if k not in ("score", "source_score")} != {
                k: v for k, v in old.items() if k not in ("score", "source_score")
            }:
                raise ValueError("Rescoring altered inference text, inputs, truth or routing.")
        return result

    rescored, rescored_oracle = scored(original), scored(oracle)
    metrics = aggregate_predictions(rescored)
    metrics["oracle_answer_diagnostic"] = aggregate_predictions(rescored_oracle)
    metrics.update(review_policy="automated_development", human_reviewed_data=False,
                   reviewed_data=True, purpose="validation_development")
    derived = {**manifest, "scorer_version": scoring.SCORER_VERSION,
               "scorer_sha256": file_hash(scoring.__file__),
               "derivation": {"operation": "offline_rescore", "inference_executed": False,
                              "training_executed": False, "source_run": source.as_posix(),
                              "source_sha256": hashes, "source_scorer_version": manifest["scorer_version"],
                              "source_scorer_sha256": manifest["scorer_sha256"],
                              "rescore_script_sha256": file_hash(__file__),
                              "rescored_at": datetime.now(timezone.utc).isoformat(),
                              "original_scores_preserved_field": "source_score"}}
    output.mkdir(parents=True)
    for name, records in (("predictions.jsonl", rescored), ("oracle_predictions.jsonl", rescored_oracle)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    derived.update(predictions_sha256=file_hash(output / "predictions.jsonl"),
                   oracle_predictions_sha256=file_hash(output / "oracle_predictions.jsonl"))
    for name, value in (("manifest.json", derived), ("metrics.json", metrics)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-scorer", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = rescore(args.source, args.output, args.source_scorer)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"records": report["record_count"], "questions": report["question_count"],
                      "scorer_version": scoring.SCORER_VERSION, "inference_executed": False,
                      "oracle_accuracy": report["oracle_answer_diagnostic"]["configurations"]["oracle_answer"]["accuracy"]}))


if __name__ == "__main__":
    main()
