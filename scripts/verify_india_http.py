"""Fourteen finite website generations on the fictional India display handbook.

Run only after the frozen study finishes, on a separate fresh website workspace.
This is demonstration acceptance, not additional held-out benchmark evidence.
Every request intent and returned outcome is saved. Existing receipts are never
overwritten or retried, including an in-flight call after an interruption.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from editguard_benchmark.metrics import quantile
from editguard_benchmark.scoring import score
from editguard_chat.examples import india_catalogue
from editguard_chat.reasonke_backend import IterativeFactContext
from scripts.verify_reasonke_live import Client, MODEL_ID, MODEL_REVISION, atomic_write

OUTCOMES = ("correct", "wrong", "abstain", "execution_error", "unexecuted")
FIELDS = ("question", "answer", "scope", "source", "identity", "supersedes")


def case_plan():
    example = india_catalogue()
    world = example["world"]
    facts = world["initial_facts"]
    truth = world["ground_truth"]
    direct = facts[0]["question"]
    two_hop, three_hop = example["questions"][1:3]
    rows = []

    def add(uid, question, target, revision, depth, applicable=True):
        rows.append({"id": uid, "expected_revision": revision, "depth": depth,
            "outcome": "unexecuted", "request_state": "not_started",
            "question": {"question": question, "expected_answer": target,
                "expected_aliases": [target], "initial_answer": target,
                "applicable": applicable, "inapplicable_edit_aliases": [truth["new_owner"]]}})

    for phase, revision, prefix in (("initial", 9, "old"), ("rollback", 11, "old")):
        for kind, question, target, depth in (
                ("direct", direct, truth[prefix + "_owner"], 1),
                ("two_hop", two_hop, truth[prefix + "_contact"], 2),
                ("three_hop", three_hop, truth[prefix + "_contact_office"], 3)):
            add(phase + "_" + kind, question, target, revision, depth)
    # Insert edited and comparison slots before the rollback slots.
    rollback = rows[3:]
    rows = rows[:3]
    add("edited_paraphrase", "For release 2, which team is responsible for Kaveri Payment Gateway?",
        truth["new_owner"], 10, 1)
    add("edited_two_hop", two_hop, truth["new_contact"], 10, 2)
    add("edited_three_hop", three_hop, truth["new_contact_office"], 10, 3)
    for uid, index in (("edited_hosting", 3), ("edited_release_1", 5), ("edited_pro", 4)):
        add(uid, facts[index]["question"], facts[index]["answer"], 10, 0, applicable=False)
    add("historical_office_comparison", three_hop, truth["old_contact_office"], 9, 3)
    add("current_office_comparison", three_hop, truth["new_contact_office"], 10, 3)
    rows.extend(rollback)
    return rows


def validate_answer(case, answer, revision, *, historical=False):
    """Require model provenance before scoring the endorsed final answer."""
    evidence = answer.get("evidence", {})
    generation = evidence.get("generation", {})
    if (not isinstance(answer.get("content"), str)
            or evidence.get("generation_kind") != "model"
            or generation.get("model") != MODEL_ID or generation.get("revision") != MODEL_REVISION
            or generation.get("fixture") or generation.get("deterministic")
            or evidence.get("fixture") or evidence.get("deterministic")
            or evidence.get("context_policy") != "iterative"
            or evidence.get("configured_context_policy") != "iterative"
            or evidence.get("knowledge_revision") != case["expected_revision"]
            or revision != case["expected_revision"]
            or not isinstance(evidence.get("raw_model_output"), str)
            or not evidence["raw_model_output"].strip()):
        raise ValueError("Response lacks pinned neural generation and expected iterative memory provenance")
    if historical and (evidence.get("historical_snapshot") is not True
                       or evidence.get("snapshot_revision") != case["expected_revision"]):
        raise ValueError("Comparison lacks the requested historical approved-memory snapshot")
    return score(case["question"], {"content": answer["content"]})


def run(base, output):
    output = Path(output)
    if output.exists():
        raise ValueError("India receipt already exists; it is never overwritten or answer-retried")
    client = Client(base)
    cases = case_plan()
    receipt = {"schema": "editguard-india-http-v1", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "cases": cases,
        "assigned_model_generations": 14, "neural_accuracy_denominator": 14,
        "independent_display_lineages": 1, "held_out_benchmark_generations": 0,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "context_policy": "iterative", "operations": [], "generation_requests": [],
        "consistency_checks": {}, "owned_fact_ids": [], "owned_chat_ids": [],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "authorization_headers_recorded": False,
        "notice": "Fourteen assigned generations on one fictional display handbook. Automated strict aliases, no independent human review. Separate from both frozen studies; no answer-based retries.",
        "interruption_policy": "Each request intent is saved before transmission. An existing receipt refuses rerun, including unknown in-flight outcomes. Do not delete it to obtain better answers.",
        "comparison_failure_policy": "A failed two-generation comparison records both unavailable outcomes as execution errors; partial backend output may exist and is not replayed.",
        "final_state_policy": "Keep the nine approved initial facts and all chat/history records. Roll back only this workload's known replacement; never clear other memory."}
    started = time.perf_counter()
    replacement = None
    predecessor = None
    initial_ids = set()
    rollback_done = False

    def save():
        atomic_write(output, receipt)

    def check(name, condition):
        receipt["consistency_checks"][name] = bool(condition)
        save()
        if not condition:
            raise ValueError("India acceptance consistency check failed: " + name)

    def operation(name, path, payload=None):
        row = {"name": name, "path": path, "state": "in_flight"}
        receipt["operations"].append(row)
        save()
        began = time.perf_counter()
        try:
            result = client.request(path, payload)
            row.update(state="completed", response={key: value for key, value in result.items() if key != "token"})
            return result
        except Exception as exc:
            row.update(state="error", error={"type": type(exc).__name__, "message": str(exc),
                       "http_status": getattr(exc, "status", None)})
            raise
        finally:
            row["http_seconds"] = time.perf_counter() - began
            save()

    def finish(case, answer, revision, historical=False):
        case.update(response=answer, revision=revision, request_state="returned")
        try:
            case.update(validate_answer(case, answer, revision, historical=historical))
            case["neural_provenance_accepted"] = True
        except Exception as exc:
            fail(case, exc)
        save()

    def fail(case, exc):
        case.update(error={"type": type(exc).__name__, "message": str(exc)}, request_state="error",
                    neural_provenance_accepted=False)
        case.update(score(case["question"], error=case["error"]))
        save()

    def generate(case):
        # A fresh conversation prevents previous correct answers providing a shortcut.
        try:
            chat = operation("create_" + case["id"] + "_chat", "/api/chats", {})["chat"]
            receipt["owned_chat_ids"].append(chat["id"])
            case.update(chat_id=chat["id"], request_state="in_flight")
            request = {"case_ids": [case["id"]], "path": "/api/chat", "state": "in_flight"}
            receipt["generation_requests"].append(request)
            save()
            began = time.perf_counter()
            try:
                result = operation("generate_" + case["id"], "/api/chat",
                    {"chat_id": chat["id"], "message": case["question"]["question"], "mode": "edited"})
                request["state"] = "returned"
                case["user_message_id"] = result["user"]["id"]
                finish(case, result["assistant"], result["revision"])
            finally:
                request["http_seconds"] = time.perf_counter() - began
        except Exception as exc:
            if case["outcome"] == "unexecuted":
                fail(case, exc)
            if "request" in locals():
                request["state"] = "error"
        save()

    def rollback():
        nonlocal rollback_done
        rolled = operation("atomic_rollback_owned_update", f"/api/corrections/{replacement}/rollback", {})
        check("rollback_restores_predecessor_in_one_revision", rolled.get("revision") == 11
              and rolled.get("correction", {}).get("id") == predecessor
              and rolled["correction"].get("status") == "active")
        rollback_done = True

    save()
    try:
        status = operation("inspect_ready_model", "/api/status")
        client.token = status.get("token")
        model = status.get("model", {})
        check("ready_pinned_model", model.get("ready") is True and model.get("name") == MODEL_ID
              and model.get("revision") == MODEL_REVISION and not status.get("busy"))
        check("iterative_serving_policy", status.get("retrieval", {}).get("context_policy") == "iterative"
              and status["retrieval"].get("method") == IterativeFactContext.method)
        memory = operation("inspect_empty_knowledge_history", "/api/corrections")
        chats = operation("inspect_empty_conversations", "/api/chats")
        check("fresh_separate_workspace", memory.get("revision") == 0 and not memory.get("corrections")
              and not memory.get("events") and not chats.get("chats"))
        approved = operation("approve_fictional_india_handbook", "/api/examples/india/approve", {})
        initial = operation("inspect_nine_approved_facts", "/api/corrections")
        initial_ids = {row["id"] for row in initial["corrections"]}
        receipt["owned_fact_ids"] = sorted(initial_ids)
        expected = {(row["entity_id"], row["relation_id"], row["scope_id"], row["answer"])
                    for row in india_catalogue()["world"]["initial_facts"]}
        actual = {(row.get("identity", {}).get("entity_id"), row.get("identity", {}).get("relation_id"),
                   row.get("identity", {}).get("scope_id"), row["answer"]) for row in initial["corrections"]}
        check("exact_nine_fictional_initial_facts", approved.get("revision") == initial.get("revision") == 9
              and len(initial_ids) == 9 and actual == expected
              and all(row["status"] == "active" for row in initial["corrections"]))
        for case in cases[:3]:
            generate(case)
        payload = {key: approved["suggested_edit"][key] for key in FIELDS}
        predecessor = payload["supersedes"]
        check("update_replaces_owned_predecessor", predecessor in initial_ids)
        # A competing approval must be rejected until the administrator resolves it explicitly.
        rejected = False
        try:
            operation("reject_unresolved_owner_conflict", "/api/corrections",
                      {key: value for key, value in payload.items() if key != "supersedes"})
        except Exception as exc:
            rejected = getattr(exc, "status", None) == 409
        check("conflicting_approval_rejected", rejected)
        unchanged = operation("inspect_conflict_left_memory_unchanged", "/api/corrections")
        check("rejected_conflict_preserves_revision_and_ids", unchanged.get("revision") == 9
              and {row["id"] for row in unchanged["corrections"]} == initial_ids)
        updated = operation("approve_explicit_owner_replacement", "/api/corrections", payload)
        replacement = updated["correction"]["id"]
        receipt["owned_fact_ids"].append(replacement)
        check("replacement_is_revision_ten", updated["revision"] == 10
              and updated["correction"]["supersedes"] == predecessor)
        for case in cases[3:9]:
            generate(case)
        comparison_source = cases[5]  # edited three-hop question, fresh conversation
        if comparison_source.get("user_message_id"):
            request = {"case_ids": [row["id"] for row in cases[9:11]], "path": "/api/compare", "state": "in_flight"}
            receipt["generation_requests"].append(request)
            for case in cases[9:11]:
                case["request_state"] = "in_flight"
            save()
            began = time.perf_counter()
            try:
                compared = operation("compare_historical_and_current_office", "/api/compare",
                    {"chat_id": comparison_source["chat_id"], "message_id": comparison_source["user_message_id"],
                     "baseline_revision": 9})
                request["state"] = "returned"
                finish(cases[9], compared["original"], 9, historical=True)
                finish(cases[10], compared["edited"], compared["revision"])
                check("comparison_preserves_live_revision", compared["revision"] == 10
                      and compared.get("baseline_label") == "APPROVED MEMORY r9")
            except Exception as exc:
                request["state"] = "error"
                for case in cases[9:11]:
                    if case["outcome"] == "unexecuted":
                        fail(case, exc)
            finally:
                request["http_seconds"] = time.perf_counter() - began
                save()
        else:
            for case in cases[9:11]:
                case["unexecuted_reason"] = "Three-hop request returned no saved user message; no comparison retry."
        rollback()
        for case in cases[11:]:
            generate(case)
        final = operation("inspect_restored_nine_facts", "/api/corrections")
        check("initial_facts_restored_with_history_retained", final.get("revision") == 11
              and {row["id"] for row in final["corrections"] if row["status"] == "active"} == initial_ids
              and len(final["corrections"]) == 10)
        check("rollback_audit_events_share_revision", [(row["action"], row["revision"]) for row in final["events"][-2:]]
              == [("withdrawn", 11), ("restored", 11)])
        receipt["status"] = "completed" if all(case["outcome"] != "unexecuted" for case in cases) else "interrupted"
    except Exception as exc:
        receipt.update(status="interrupted", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        if receipt["status"] == "running":
            receipt["status"] = "interrupted"
        # Keep a usable initial demo. No deletion/withdrawal of the handbook or chats.
        if replacement and not rollback_done:
            try:
                rollback()
            except Exception as exc:
                receipt["restoration_error"] = {"type": type(exc).__name__, "message": str(exc)}
        requests = receipt["generation_requests"]
        timings = [row["http_seconds"] for row in requests if "http_seconds" in row]
        receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(), total_seconds=time.perf_counter() - started,
            outcome_counts={name: sum(case["outcome"] == name for case in cases) for name in OUTCOMES},
            verified_neural_response_count=sum(case.get("neural_provenance_accepted", False) for case in cases),
            all_consistency_checks_passed=bool(receipt["consistency_checks"]) and all(receipt["consistency_checks"].values()),
            initial_handbook_restored=rollback_done,
            latency={"request_n": len(timings), "p50_request_seconds": quantile(timings, .5),
                "p95_request_seconds": quantile(timings, .95),
                "scope": "Loopback website wall time. Twelve single-generation requests and one two-generation comparison; do not interpret request quantiles as per-generation latency."})
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base_url, args.output)
    print(json.dumps({key: result[key] for key in ("status", "outcome_counts", "verified_neural_response_count",
        "consistency_checks", "initial_handbook_restored", "latency")}, indent=2))
    if (result["status"] != "completed" or not result["all_consistency_checks_passed"]
            or not result["initial_handbook_restored"] or result["outcome_counts"]["execution_error"]
            or result["verified_neural_response_count"] != 14):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
