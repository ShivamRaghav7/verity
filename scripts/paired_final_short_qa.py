"""Aggregate frozen secondary verdicts; no scoring, inference or tuning."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.metrics import paired_cluster_bootstrap

CONFIGURATIONS = ("unchanged_base", "updated_rag", "prompt_only",
                  "similarity_editor", "trained_scope_editor")
SECONDARY_VERSION = "editguard-request-bound-short-qa-v1"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_pairs(rows, *, expected_questions=720, expected_families=12):
    """Pair complete fixed cohorts using only their saved secondary verdicts."""
    groups = {name: [] for name in CONFIGURATIONS}
    identities = {name: set() for name in CONFIGURATIONS}
    counts = {name: Counter() for name in CONFIGURATIONS}
    for row in rows:
        if row.get("population") != "scope" or row.get("evaluation_kind") != "matched":
            raise ValueError("Only standard-scope matched observations belong in this aggregation.")
        name, identity = row.get("configuration"), row.get("id")
        if name not in groups or not isinstance(identity, str) or not identity or identity in identities[name]:
            raise ValueError("Unknown configuration or missing/duplicate observation identity.")
        if row.get("review_policy") != "automated_final" or row.get("human_reviewed_data") is not False:
            raise ValueError("Frozen final observations must disclose automated-only review.")
        score = row.get("short_qa")
        if not isinstance(score, dict) or score.get("diagnostic_version") != SECONDARY_VERSION:
            raise ValueError("A saved frozen secondary verdict is required; rescoring is forbidden.")
        correct = score.get("correct")
        if correct is not None and type(correct) is not bool:
            raise ValueError("Secondary correctness must be an explicit bool or None.")
        verdict = "correct" if correct is True else "incorrect" if correct is False else "pending_uncredited"
        if score.get("verdict") != verdict or score.get("human_semantic_review") is not False:
            raise ValueError("Secondary verdict/review provenance is inconsistent.")
        if correct is True and (row.get("status") != "answered" or row.get("answer_kind") != "model_generated"
                                or not isinstance(row.get("answer"), str) or not row["answer"].strip()):
            raise ValueError("Nonanswers or deterministic answers cannot receive learned-answer credit.")
        if not isinstance(row.get("expected_fact"), dict) or row.get("answerable") is False:
            raise ValueError("Every standard question must have explicit answerable ground truth.")
        identities[name].add(identity)
        # A shallow copy changes only the analysis view consumed by the existing
        # bootstrap. The original primary score and observation are never edited.
        groups[name].append({**row, "score": {"correct": correct}})
        counts[name][verdict] += 1
        counts[name]["status:" + str(row.get("status"))] += 1
    first_ids = identities[CONFIGURATIONS[0]]
    if len(first_ids) != expected_questions or any(ids != first_ids for ids in identities.values()):
        raise ValueError("All five configurations must contain the same complete question population.")
    families = {row.get("family_id") for row in groups[CONFIGURATIONS[0]]}
    if len(families) != expected_families or any(not isinstance(f, str) or not f for f in families):
        raise ValueError("The declared entity-family population is incomplete.")
    comparisons = []
    for left, right in itertools.combinations(CONFIGURATIONS, 2):
        pair = paired_cluster_bootstrap(groups[left], groups[right], seed=42, resamples=2000)
        if (pair["matched_answerable"] != expected_questions or pair["family_count"] != expected_families
                or pair["left_unmatched"] or pair["right_unmatched"]):
            raise ValueError("Pairing changed the fixed denominator or family partition.")
        pair.update(left=left, right=right, metric="frozen_request_bound_short_qa_accuracy")
        comparisons.append(pair)
    configurations = {}
    for name in CONFIGURATIONS:
        c = counts[name]
        configurations[name] = {"denominator": expected_questions, "correct": c["correct"],
            "incorrect": c["incorrect"], "pending_uncredited": c["pending_uncredited"],
            "accuracy": c["correct"] / expected_questions,
            "status_counts": {key.removeprefix("status:"): value for key, value in c.items() if key.startswith("status:")}}
    return {"question_count": expected_questions, "family_count": expected_families,
            "record_count": expected_questions * len(CONFIGURATIONS),
            "configurations": configurations, "paired_comparisons": comparisons}


def aggregate_final(run_directory, protocol_path, protocol_sha256, output):
    run_directory, protocol_path, output = map(lambda p: Path(p).resolve(),
                                               (run_directory, protocol_path, output))
    if output.exists() or output == run_directory or run_directory in output.parents:
        raise ValueError("Derived output must be new and separate from the sealed final run.")
    protocol = read_json(protocol_path)
    if file_hash(protocol_path) != protocol_sha256 or protocol.get("status") != "frozen":
        raise ValueError("An exact frozen protocol binding is required.")
    if protocol.get("evaluation", {}).get("bootstrap") != {
            "cluster_unit": "entity_family", "confidence": .95, "paired": True,
            "resamples": 2000, "seed": 42}:
        raise ValueError("Use only the already frozen bootstrap settings.")
    if protocol.get("scoring", {}).get("short_qa_version") != SECONDARY_VERSION:
        raise ValueError("Unsupported frozen secondary scorer.")
    sources = {"protocol": protocol_path,
               "manifest": run_directory / "manifest.json",
               "predictions": run_directory / "predictions.jsonl",
               "secondary_metrics": run_directory / "short_qa_metrics.json",
               "aggregation_script": Path(__file__).resolve(),
               "paired_metrics_implementation": ROOT / "editguard/metrics.py"}
    before = {name: file_hash(path) for name, path in sources.items()}
    if before["paired_metrics_implementation"] != protocol.get("source_sha256", {}).get("editguard/metrics.py"):
        raise ValueError("The existing bootstrap implementation differs from the frozen source.")
    manifest = read_json(sources["manifest"])
    if (manifest.get("status") != "completed" or manifest.get("purpose") != "frozen_final_evaluation"
            or manifest.get("protocol_sha256") != protocol_sha256
            or manifest.get("review_policy") != "automated_final" or manifest.get("human_reviewed_data") is not False
            or manifest.get("fresh_final_runs") != 1 or manifest.get("final_test_tuning_performed") is not False
            or manifest.get("configurations") != list(CONFIGURATIONS)):
        raise ValueError("Only the completed original frozen final execution is eligible.")
    if (manifest.get("outputs_sha256", {}).get("predictions.jsonl") != before["predictions"]
            or manifest.get("outputs_sha256", {}).get("short_qa_metrics.json") != before["secondary_metrics"]):
        raise ValueError("Sealed final output hashes disagree with the manifest.")
    rows = [json.loads(line) for line in sources["predictions"].read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 4080:
        raise ValueError("The complete sealed matched output must contain 4,080 records.")
    for row in rows:
        if (row.get("final_protocol_sha256") != protocol_sha256
                or row.get("row_sha256") != manifest.get("accepted_row_sha256", {}).get(row.get("id"))):
            raise ValueError("An observation is not bound to the final protocol and reviewed row.")
    standard = [row for row in rows if row.get("population") == "scope"]
    result = summarize_pairs(standard)
    saved_secondary = read_json(sources["secondary_metrics"])
    for name, result_counts in result["configurations"].items():
        saved = saved_secondary["by_population"]["scope"][name]["all_requested_qa"]
        if any(result_counts[key] != saved[key] for key in ("denominator", "correct", "incorrect", "pending_uncredited", "status_counts")):
            raise ValueError("Derived counts disagree with the sealed secondary metric.")
    after = {name: file_hash(path) for name, path in sources.items()}
    if before != after:
        raise ValueError("An input changed during aggregation.")
    result.update(schema_version=1, purpose="frozen_final_secondary_paired_uncertainty",
        created_at=datetime.now(timezone.utc).isoformat(), protocol_sha256=protocol_sha256,
        secondary_scorer_version=SECONDARY_VERSION, review_policy="automated_final",
        human_reviewed_data=False, original_results_replaced=False,
        new_scoring_or_inference_performed=False, final_test_tuning_performed=False,
        population="720_standard_scope_questions_excluding_the_96_unsupported_challenge_forms",
        source_sha256=before,
        source_paths={name: str(path) for name, path in sources.items()},
        limitations=["Paired descriptive aggregation of existing frozen secondary verdicts; no new metric, scoring rule or test tuning.",
            "All 720 standard questions remain in denominators, including pending semantic cases and clarifications.",
            "The request-bound metric uses question context and does not establish a complete scoped assertion for bare outputs.",
            "Only 12 controlled entity-family clusters and shared templates; intervals do not establish unrestricted-language performance.",
            "Primary full-assertion baseline zeros are format-limited uncredited cases, not evidence of zero factual accuracy.",
            "The 96 challenge questions are excluded from this standard-population interval and remain separately reported."])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate_final(args.run, args.protocol, args.protocol_sha256, args.output)
    print(json.dumps({key: result[key] for key in ("purpose", "question_count", "family_count", "configurations", "paired_comparisons")}, indent=2))


if __name__ == "__main__":
    main()
