"""Fixed-denominator factual, locality, retrieval and paired-lineage statistics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from editguard_reasonke.__main__ import write_receipt
from .scoring import score
from .data import digest


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    fraction = position - lower
    return values[lower] + fraction * (values[min(lower + 1, len(values) - 1)] - values[lower])


def summarize(rows):
    counts = Counter(row["outcome"] for row in rows)
    denominator = len(rows)
    answered = counts["correct"] + counts["wrong"]
    generated = [row for row in rows if isinstance(row.get("response"), dict)]
    latency = [row["pipeline_seconds"] for row in generated if isinstance(row.get("pipeline_seconds"), (int, float))]
    errors = [row["pipeline_seconds"] for row in rows if row["outcome"] == "execution_error" and "pipeline_seconds" in row]
    return {"assigned": denominator, "counts": {key: counts[key] for key in
            ("correct", "wrong", "abstain", "execution_error", "unexecuted")},
            "accuracy": counts["correct"] / denominator if denominator else None,
            "coverage": answered / denominator if denominator else None,
            "answered_only_precision": counts["correct"] / answered if answered else None,
            "false_correction_applications": sum(bool(row.get("false_correction_application")) for row in rows),
            "generated_latency": {"n": len(latency), "p50_seconds": quantile(latency, .5),
                                  "p95_seconds": quantile(latency, .95), "definition": "context selection + local IPC + generation; excludes model load and browser network"},
            "failed_request_latency": {"n": len(errors), "p50_seconds": quantile(errors, .5), "p95_seconds": quantile(errors, .95)}}


def paired_bootstrap(left, right, resamples=10000, seed=42):
    l = {row["question"]["id"]: row for row in left}
    r = {row["question"]["id"]: row for row in right}
    if set(l) != set(r):
        raise ValueError("Paired comparison requires identical assigned questions")
    grouped = {}
    for key in sorted(l):
        grouped.setdefault(l[key]["question"]["lineage_id"], []).append(int(l[key]["correct"]) - int(r[key]["correct"]))
    groups = list(grouped.values())
    if not groups:
        return None
    rng = random.Random(seed)
    sampled = []
    for _ in range(resamples):
        selected = [groups[rng.randrange(len(groups))] for _ in groups]
        sampled.append(sum(sum(group) for group in selected) / sum(map(len, selected)))
    return {"left_minus_right": sum(sum(group) for group in groups) / sum(map(len, groups)),
            "ci95": [quantile(sampled, .025), quantile(sampled, .975)], "independent_lineages": len(groups),
            "paired_questions": len(l), "bootstrap_resamples": resamples, "seed": seed}


def aggregate(paths):
    receipts = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not receipts or any(receipt.get("schema") != "editguard-operations-inference-v1" for receipt in receipts):
        raise ValueError("Choose controlled operations inference receipts")
    phases = {receipt["plan"]["phase"] for receipt in receipts}
    identities = [receipt["plan"]["sources"] for receipt in receipts]
    if len(phases) != 1 or any(identity != identities[0] for identity in identities):
        raise ValueError("Do not combine different phases or source/configuration identities")
    rows = [row for receipt in receipts for row in receipt["records"]]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Repeated outcomes cannot inflate the denominator")
    for row in rows:
        checked = score(row["question"], row.get("response"), row.get("error"))
        if row["outcome"] != checked["outcome"]:
            raise ValueError("Recorded verdict differs from fixed scorer")
        row.update(checked)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)
    report = {"schema": "editguard-operations-results-v1", "phase": next(iter(phases)),
              "execution_status": "completed" if all(receipt["status"] == "completed" for receipt in receipts) else "incomplete",
              "source_receipts": {str(path): digest(path) for path in paths},
              "dataset_id": receipts[0]["dataset"]["dataset_id"], "model_measurements": {}, "conditions": {},
              "assigned_records": len(rows), "independent_lineages": len({row["question"]["lineage_id"] for row in rows}),
              "notes": ["Strict whole endorsed final answer matching; no sole LLM judge",
                        "All assigned outcomes retained; abstentions/errors/unexecuted receive zero primary credit",
                        "Both checkpoints are upstream trained; no causal attribution to ReasonKE training",
                        "Fictional controlled graph with shared wording templates and automated audit"]}
    for receipt in receipts:
        report["model_measurements"][receipt["plan"]["model"]] = [session.get("runtime") for session in receipt["sessions"] if session.get("runtime")]
    for name, records in grouped.items():
        categories = sorted({row["question"]["category"] for row in records})
        report["conditions"][name] = {"overall": summarize(records),
            "by_category": {category: summarize([row for row in records if row["question"]["category"] == category]) for category in categories},
            "applicable": summarize([row for row in records if row["question"]["applicable"]]),
            "locality": summarize([row for row in records if not row["question"]["applicable"]])}
        if name.endswith("RAG"):
            retrieval = [row["retrieval"] for row in records if "retrieval" in row]
            hop = [row["retrieval"] for row in records if "retrieval" in row and "two_hop" in row["question"]["category"]]
            required = sum(len(row["question"]["required_fact_ids"]) for row in records)
            report["conditions"][name]["retrieval"] = {
                "opportunities": len(records), "recorded": len(retrieval),
                "fact_recall_numerator": sum(item["found"] for item in retrieval), "fact_recall_denominator": required,
                "fact_recall": sum(item["found"] for item in retrieval) / required if required else None,
                "two_hop_complete_numerator": sum(item["complete"] for item in hop),
                "two_hop_complete_denominator": sum("two_hop" in row["question"]["category"] for row in records),
                "note": "Shared question-only retrieval; duplicated generator use is not an independent retrieval experiment"}
    reference = {row["question"]["id"]: row for row in grouped.get("R0", []) if not row["question"]["applicable"] and row["correct"]}
    for name in ("R-ALL", "R-RAG"):
        if name in grouped and "R0" in grouped:
            matches = {row["question"]["id"]: row for row in grouped[name]}
            eligible = [key for key in reference if key in matches]
            retained = sum(matches[key]["correct"] for key in eligible)
            report["conditions"][name]["baseline_correct_retention"] = {
                "correct": retained, "denominator": len(eligible),
                "rate": retained / len(eligible) if eligible else None, "reference": "R0"}
    report["paired_differences"] = {}
    for left, right in (("R-ALL", "G-ALL"), ("R-RAG", "G-RAG"), ("R-ALL", "R-RAG")):
        if left in grouped and right in grouped:
            report["paired_differences"][left + " minus " + right] = paired_bootstrap(grouped[left], grouped[right])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = aggregate(args.receipts)
    write_receipt(args.output, report)
    print(json.dumps({"output": args.output, "phase": report["phase"], "assigned": report["assigned_records"],
                      "conditions": list(report["conditions"])}))


if __name__ == "__main__":
    main()
