"""Replay the finite India website receipt on CPU; never loads or calls a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from editguard_benchmark.scoring import score
from editguard_reasonke.runtime import extract_answer
from scripts import verify_india_http as workload

CHECKS = {"ready_pinned_model", "iterative_serving_policy", "fresh_separate_workspace",
    "exact_nine_fictional_initial_facts", "update_replaces_owned_predecessor",
    "conflicting_approval_rejected", "rejected_conflict_preserves_revision_and_ids",
    "replacement_is_revision_ten", "comparison_preserves_live_revision",
    "rollback_restores_predecessor_in_one_revision", "initial_facts_restored_with_history_retained",
    "rollback_audit_events_share_revision"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reject_fixtures(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"fixture", "deterministic", "software_fixture", "explicit_software_fixture"} and item:
                raise ValueError("Explicit fixture or deterministic output is not neural evidence")
            _reject_fixtures(item)
    elif isinstance(value, list):
        for item in value:
            _reject_fixtures(item)


def audit(path):
    path = Path(path)
    raw = path.read_bytes()
    receipt = json.loads(raw)
    _reject_fixtures(receipt)
    planned = workload.case_plan()
    plan_hash = hashlib.sha256(json.dumps(planned, sort_keys=True).encode()).hexdigest()
    if (receipt.get("schema") != "editguard-india-http-v1" or receipt.get("status") != "completed"
            or receipt.get("assigned_model_generations") != 14
            or receipt.get("neural_accuracy_denominator") != 14
            or receipt.get("independent_display_lineages") != 1
            or receipt.get("held_out_benchmark_generations") != 0
            or receipt.get("model_id") != workload.MODEL_ID
            or receipt.get("model_revision") != workload.MODEL_REVISION
            or receipt.get("context_policy") != "iterative"
            or receipt.get("plan_sha256") != plan_hash
            or receipt.get("script_sha256") != digest(workload.__file__)
            or receipt.get("authorization_headers_recorded") is not False):
        raise ValueError("Completed India receipt, source identity or assigned population differs")
    cases = receipt.get("cases", [])
    if len(cases) != 14 or [row.get("id") for row in cases] != [row["id"] for row in planned]:
        raise ValueError("The fourteen assigned case identities differ")
    verified = 0
    outcomes = {key: 0 for key in workload.OUTCOMES}
    for case, expected in zip(cases, planned):
        if any(case.get(key) != expected[key] for key in ("id", "question", "expected_revision", "depth")):
            raise ValueError("Fixed India question, label or revision differs: " + expected["id"])
        response, error = case.get("response"), case.get("error")
        if error is None:
            if not isinstance(response, dict) or case.get("request_state") != "returned":
                raise ValueError("A completed generated case lacks its returned response")
            recomputed = workload.validate_answer(case, response, case.get("revision"),
                historical=case["id"] == "historical_office_comparison")
            content, _ = extract_answer(response["evidence"]["raw_model_output"])
            if content != response["content"] or case.get("neural_provenance_accepted") is not True:
                raise ValueError("Raw extraction or neural provenance differs: " + case["id"])
            verified += 1
        else:
            if not isinstance(error, dict) or case.get("request_state") != "error":
                raise ValueError("Malformed retained execution error")
            recomputed = score(case["question"], error=error)
            if case.get("neural_provenance_accepted") is not False:
                raise ValueError("An execution error cannot claim accepted neural provenance")
        if any(case.get(key) != value for key, value in recomputed.items()):
            raise ValueError("Recorded score differs from fixed whole-answer scoring: " + case["id"])
        outcomes[recomputed["outcome"]] += 1
    if (receipt.get("outcome_counts") != outcomes
            or receipt.get("verified_neural_response_count") != verified):
        raise ValueError("Saved outcome or neural-provenance denominators differ")

    requests = receipt.get("generation_requests", [])
    expected_requests = [[row["id"]] for row in planned[:9]] + [
        [row["id"] for row in planned[9:11]]] + [[row["id"]] for row in planned[11:]]
    if (len(requests) != 13 or [row.get("case_ids") for row in requests] != expected_requests
            or any(row.get("path") != ("/api/compare" if len(row["case_ids"]) == 2 else "/api/chat")
                   or row.get("state") not in {"returned", "error"} for row in requests)):
        raise ValueError("Finite generation request accounting differs or a request was replayed")

    checks = receipt.get("consistency_checks", {})
    if set(checks) != CHECKS or any(type(value) is not bool for value in checks.values()):
        raise ValueError("Declared transaction-check population differs")
    if receipt.get("all_consistency_checks_passed") != all(checks.values()):
        raise ValueError("Transaction summary differs from the saved checks")
    if receipt.get("initial_handbook_restored") is not True:
        raise ValueError("Receipt does not record restoration of the original handbook")
    operations = receipt.get("operations", [])
    names = [row.get("name") for row in operations]
    if len(names) != len(set(names)):
        raise ValueError("An operation was repeated")
    by_name = {row["name"]: row for row in operations}
    conflict = by_name.get("reject_unresolved_owner_conflict", {})
    if conflict.get("state") != "error" or conflict.get("error", {}).get("http_status") != 409:
        raise ValueError("Expected conflict rejection is absent")
    final = by_name.get("inspect_restored_nine_facts", {}).get("response", {})
    corrections = final.get("corrections", [])
    active = {row["id"] for row in corrections if row.get("status") == "active"}
    initial = by_name.get("inspect_nine_approved_facts", {}).get("response", {})
    initial_ids = {row["id"] for row in initial.get("corrections", [])}
    if (len(initial_ids) != 9 or initial.get("revision") != 9 or final.get("revision") != 11
            or active != initial_ids or len(corrections) != 10
            or [(row["action"], row["revision"]) for row in final.get("events", [])[-2:]]
               != [("withdrawn", 11), ("restored", 11)]):
        raise ValueError("Saved transaction snapshots do not reproduce rollback")
    expected_facts = {(row["entity_id"], row["relation_id"], row["scope_id"], row["answer"])
        for row in workload.india_catalogue()["world"]["initial_facts"]}
    initial_facts = {(row.get("identity", {}).get("entity_id"), row.get("identity", {}).get("relation_id"),
        row.get("identity", {}).get("scope_id"), row.get("answer")) for row in initial["corrections"]}
    initial_by_id = {row["id"]: row for row in initial["corrections"]}
    if initial_facts != expected_facts or any(
            any(row.get(key) != initial_by_id[row["id"]].get(key) for key in workload.FIELDS)
            for row in corrections if row.get("status") == "active"):
        raise ValueError("The restored handbook differs from its original nine facts")
    if path.read_bytes() != raw:
        raise ValueError("Input changed during audit")
    return {"schema": "editguard-india-http-host-audit-v1", "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(), "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "workload_script_sha256": receipt["script_sha256"], "plan_sha256": plan_hash,
        "audit_script_sha256": digest(__file__), "assigned_model_generations": 14,
        "verified_neural_response_count": verified, "outcome_counts": outcomes,
        "primary_accuracy": outcomes["correct"] / 14,
        "generation_requests": 13, "independent_display_lineages": 1,
        "held_out_benchmark_generations": 0, "model_calls": 0, "human_reviewed": False,
        "transaction_checks": checks, "transaction_checks_all_passed": all(checks.values()),
        "initial_handbook_restored": True,
        "verified": ["fixed fourteen-case identities and labels", "workload source and plan hashes",
            "raw-output extraction", "pinned neural and memory provenance", "whole-final-answer scoring",
            "all outcome denominators", "finite request accounting", "conflict rejection and rollback snapshots"],
        "notice": "Mechanical audit of one display handbook; no independent semantic review or generated-prose verification. Transaction correctness is separate from neural answer accuracy."}


def publish(path, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Existing audit is preserved; choose a new output path")
    result = audit(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.receipt, args.output), indent=2))


if __name__ == "__main__":
    main()
