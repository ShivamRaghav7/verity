"""Post-hoc descriptive audit of released failures. No inference or rescoring.

This supplementary analysis never changes frozen labels, aliases or outcomes.
Exact value associations describe saved text, not a model-internal causal claim.
Run from the repository root with a fresh output directory.
"""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import re
import statistics

from editguard_benchmark.scoring import normalize

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "operations_reasonke": "reports/reasonke/operations-final/final-reasonke.json",
    "operations_qwen": "reports/reasonke/operations-final/final-instruction.json",
    "depth_reasonke": "reports/reasonke/depth-final/final-reasonke.json",
    "depth_qwen": "reports/reasonke/depth-final/final-instruction.json",
}
FACT_ROLES = {
    "f1": "release_2_owner", "f2": "previous_owner_contact",
    "f3": "current_owner_contact", "f4": "hosting_region",
    "f5": "pro_service_owner", "f6": "release_1_owner",
    "f7": "unrelated_office_city", "f8": "pro_team_function",
    "f9": "legacy_team_function",
}


def analyze(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh output directory; released evidence is preserved")
    hashes, records, groups = {}, [], defaultdict(list)
    for key, name in INPUTS.items():
        path = ROOT / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        study = key.split("_")[0]
        for value in json.loads(path.read_text(encoding="utf-8"))["records"]:
            groups[(study, value["condition"])].append(value)
            if value["condition"] == "R0" or value["outcome"] == "correct":
                continue
            question, response = value["question"], value.get("response", {})
            endorsed = response.get("content", "")
            exact = [f for f in value["supplied_facts"]
                     if normalize(f["answer"]) == normalize(endorsed)]
            mentioned = [f for f in value["supplied_facts"]
                         if re.search(r"(?<!\w)" + re.escape(normalize(f["answer"])) + r"(?!\w)", normalize(endorsed))]
            action = re.search(r"<action>(.*?)</action>", response.get("raw_output", ""), re.S | re.I)
            boxed = re.findall(r"\\boxed\{([^{}]*)\}", action.group(1)) if action else []
            if value["outcome"] == "abstain":
                diagnosis = "literal_unknown" if endorsed == "UNKNOWN" else "other_abstention"
            elif exact:
                diagnosis = "off_target_supplied_value"
            elif value["id"] in {"G-ALL:ops-045-two_hop_paraphrase", "G-ALL:ops-047-two_hop_paraphrase"}:
                diagnosis = "correct_owner_contact_chain_but_extra_text"
            elif value["id"] == "G-ALL:ops-005-two_hop_paraphrase":
                diagnosis = "pro_owner_plus_correct_contact_contamination"
            else:
                diagnosis = "requires_semantic_inspection"
            records.append({
                "study": study, "id": value["id"], "condition": value["condition"],
                "question_id": question["id"], "lineage_id": question["lineage_id"],
                "category": question["category"], "template_id": question.get("template_id"),
                "question": question["question"], "expected": question["expected_answer"],
                "endorsed": endorsed, "frozen_outcome": value["outcome"],
                "diagnosis": diagnosis, "support_complete": value["retrieval"]["complete"],
                "exact_other_fact_ids": [f["id"] for f in exact],
                "exact_other_fact_roles": [FACT_ROLES.get(f["id"].rsplit("-", 1)[-1], "not_operations_role") for f in exact],
                "mentioned_fact_ids": [f["id"] for f in mentioned],
                "boxed_answers": boxed,
                "expected_in_boxed_answers": normalize(question["expected_answer"]) in {normalize(v) for v in boxed},
                "output_tokens": response.get("generation", {}).get("output_tokens"),
                "token_budget_reached": response.get("generation", {}).get("token_budget_reached"),
                "time_budget_reached": response.get("generation", {}).get("time_budget_reached"),
                "required_fact_ids": question["required_fact_ids"],
                "supplied_facts": value["supplied_facts"], "raw_output": response.get("raw_output", ""),
            })
    summary = {}
    for (study, condition), values in groups.items():
        failures = [v for v in records if (v["study"], v["condition"]) == (study, condition)]
        tokens = [v["response"]["generation"]["output_tokens"] for v in values if v.get("response")]
        summary[study + ":" + condition] = {
            "assigned": len(values), "outcomes": dict(Counter(v["outcome"] for v in values)),
            "median_output_tokens": statistics.median(tokens) if tokens else None,
            "diagnoses": dict(Counter(v["diagnosis"] for v in failures)),
            "failure_categories": dict(Counter(v["category"] for v in failures)),
            "all_failure_support_complete": all(v["support_complete"] for v in failures),
            "failure_budget_hits": sum(bool(v["token_budget_reached"] or v["time_budget_reached"]) for v in failures),
            "exact_off_target_fact_roles": dict(Counter(role for v in failures for role in v["exact_other_fact_roles"])),
            "wrong_final_with_expected_boxed_subanswer": sum(v["expected_in_boxed_answers"] for v in failures if v["frozen_outcome"] == "wrong"),
        }
    assert len(records) == 63
    assert all(v["support_complete"] for v in records)
    assert all(not v["token_budget_reached"] and not v["time_budget_reached"] for v in records)
    assert not any(v["diagnosis"] == "requires_semantic_inspection" for v in records)
    assert summary["operations:R-ALL"]["exact_off_target_fact_roles"]["pro_service_owner"] == 15
    templates = {}
    for (study, condition), values in groups.items():
        if study == "depth" and condition == "G-RAG":
            for tid in sorted({v["question"].get("template_id") for v in values}):
                rows = [v for v in values if v["question"].get("template_id") == tid]
                templates[tid] = {"assigned": len(rows), "outcomes": dict(Counter(v["outcome"] for v in rows))}
    result = {"schema": "verity-presentation-failure-audit-v1", "analysis_date": "2026-10-30",
              "purpose": "Post-hoc description of frozen outputs, not a new experiment",
              "input_sha256": hashes, "model_calls": 0, "scores_changed": False,
              "R0_excluded_reason": "Staleness diagnostic against updated truth, not an updated-method failure cohort",
              "updated_noncorrect_records": len(records), "summary": summary,
              "depth_qwen_templates": templates, "records": records,
              "limitations": ["Exact value matching does not reveal model-internal causes",
                              "Template concentration is descriptive, not a randomized wording ablation",
                              "Three concatenation diagnoses use inspected fixed record IDs and are not a general semantic judge",
                              "Boxed text checks are parser diagnostics, not verification of all reasoning prose",
                              "No aliases, outcomes, prompts, checkpoints or denominators changed"]}
    output.mkdir(parents=True)
    (output / "failure-audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    fields = ["study", "id", "condition", "category", "question", "expected", "endorsed", "frozen_outcome", "diagnosis", "support_complete", "exact_other_fact_ids", "exact_other_fact_roles", "expected_in_boxed_answers"]
    with (output / "failure-ledger.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore", lineterminator="\n");writer.writeheader()
        for row in records:
            writer.writerow({k: "; ".join(v) if isinstance(v, list) else v for k, v in row.items() if k in fields})
    print(json.dumps({"updated_noncorrect_records": len(records), "model_calls": 0, "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output)
