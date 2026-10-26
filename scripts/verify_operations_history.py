"""Four real neural generations for historical comparison and atomic rollback.

This is a separate development-world operational check, not an extension of the
frozen held-out study. A fresh website workspace is required. Wrong answers are
never retried, and cleanup changes only fact IDs created by this workload.
"""
from datetime import datetime, timezone
import argparse
import copy
import json
from pathlib import Path
import time

from scripts.verify_reasonke_live import Client, atomic_write, MODEL_ID, MODEL_REVISION
from editguard_benchmark.data import load_dataset, digest
from editguard_benchmark.scoring import score

RETRIEVAL_METHOD = "two_stage_lexical_approved_memory"


def run(base, output, dataset="data/reasonke/operations-v1"):
    output = Path(output)
    if output.exists():
        raise ValueError("History receipt already exists; it is never overwritten or answer-retried")
    manifest, worlds, questions = load_dataset(Path(dataset), "development")
    world_id = sorted(worlds)[0]
    world = worlds[world_id]
    question = next(row for row in questions if row["lineage_id"] == world_id and row["category"] == "two_hop")
    cases = []
    for name, current in (("initial_contact", False), ("historical_comparison", False),
                          ("current_comparison", True), ("after_atomic_rollback", False)):
        target = copy.deepcopy(question)
        if not current:
            target.update(expected_answer=target["initial_answer"], expected_aliases=[target["initial_answer"]])
        cases.append({"id": name, "question": target, "outcome": "unexecuted"})
    receipt = {"schema": "editguard-operations-history-v1", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "dataset": manifest,
        "assigned_model_generations": 4, "neural_accuracy_denominator": 4,
        "independent_development_lineages": 1, "cases": cases, "operations": [],
        "consistency_checks": {}, "owned_fact_ids": [], "owned_chat_ids": [],
        "script_sha256": digest(__file__), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "authorization_headers_recorded": False,
        "notice": "Four assigned generations on one previously used development-world question, not four new independent held-out facts. Automated strict alias scoring; no human-review claim.",
        "comparison_failure_policy": "If the two-generation comparison request fails, both unavailable results are execution errors. A partial generation may have occurred but is not recovered or retried.",
        "cleanup_policy": "Withdraw only explicitly recorded fact IDs created by this workload; preserve chats, audit events and every model answer."}
    client = Client(base)
    started = time.perf_counter()
    atomic_write(output, receipt)

    def save():
        atomic_write(output, receipt)

    def operation(name, path, payload=None):
        began = time.perf_counter()
        record = {"name": name, "path": path}
        receipt["operations"].append(record)
        try:
            result = client.request(path, payload)
            # Never write the website session token or authorization headers.
            record.update(status="completed", response={key: value for key, value in result.items() if key != "token"})
            return result
        except Exception as exc:
            record.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            record["seconds"] = time.perf_counter() - began
            save()

    def check(name, condition):
        receipt["consistency_checks"][name] = bool(condition)
        save()
        if not condition:
            raise ValueError("Operational consistency check failed: " + name)

    def record_answer(case, answer, revision, *, expected_revision, historical=False):
        case.update(response=answer, revision=revision)
        try:
            evidence = answer.get("evidence", {})
            generation = evidence.get("generation", {})
            if (not isinstance(answer.get("content"), str)
                    or evidence.get("generation_kind") != "model"
                    or generation.get("model") != MODEL_ID or generation.get("revision") != MODEL_REVISION
                    or evidence.get("knowledge_revision") != expected_revision or revision != expected_revision):
                raise ValueError("Response does not identify the pinned model and expected approved-memory revision")
            if historical and (evidence.get("historical_snapshot") is not True
                               or evidence.get("snapshot_revision") != expected_revision):
                raise ValueError("Historical comparison does not identify the requested memory snapshot")
            case.update(score(case["question"], {"content": answer["content"]}))
        except Exception as exc:
            case.update(error={"type": type(exc).__name__, "message": str(exc)})
            case.update(score(case["question"], error=case["error"]))
        save()

    def failed_case(case, exc):
        case.update(error={"type": type(exc).__name__, "message": str(exc)})
        case.update(score(case["question"], error=case["error"]))
        save()

    try:
        status = operation("inspect_ready_pinned_model", "/api/status")
        client.token = status.get("token")
        model = status.get("model", {})
        check("pinned_ready_model", model.get("ready") is True and model.get("name") == MODEL_ID
              and model.get("revision") == MODEL_REVISION)
        check("retrieved_policy", status.get("retrieval", {}).get("method") == RETRIEVAL_METHOD)
        memory = operation("inspect_empty_knowledge_history", "/api/corrections")
        chats = operation("inspect_empty_conversation_history", "/api/chats")
        check("fresh_workspace", memory.get("revision") == 0 and not memory.get("corrections")
              and not memory.get("events") and not chats.get("chats"))

        approved = operation("approve_fictional_handbook", "/api/examples/operations/approve", {})
        initial = operation("record_created_fact_ids", "/api/corrections")
        facts = initial["corrections"]
        expected_keys = {(row["entity_id"], row["relation_id"], row["scope_id"]) for row in world["initial_facts"]}
        actual_keys = {(row.get("identity", {}).get("entity_id"), row.get("identity", {}).get("relation_id"),
                        row.get("identity", {}).get("scope_id")) for row in facts}
        check("exact_approved_example", approved.get("revision") == initial.get("revision") == 9
              and len(facts) == 9 and actual_keys == expected_keys
              and all(row["status"] == "active" for row in facts))
        receipt["owned_fact_ids"] = [row["id"] for row in facts]
        check("initial_example_entity", approved.get("example", {}).get("entity") == world["entity"])
        chat = operation("create_history_measurement_chat", "/api/chats", {})["chat"]
        receipt["owned_chat_ids"].append(chat["id"])
        user_message_id = None
        try:
            result = operation("generate_initial_contact", "/api/chat",
                               {"chat_id": chat["id"], "message": question["question"], "mode": "edited"})
            user_message_id = result["user"]["id"]
            record_answer(cases[0], result["assistant"], result["revision"], expected_revision=9)
        except Exception as exc:
            failed_case(cases[0], exc)

        suggestion = approved["suggested_edit"]
        payload = {key: suggestion[key] for key in ("question", "answer", "source", "scope", "identity", "supersedes")}
        check("replace_owned_initial_fact", payload["supersedes"] in receipt["owned_fact_ids"])
        updated = operation("approve_ownership_revision", "/api/corrections", payload)
        new_id = updated["correction"]["id"]
        receipt["owned_fact_ids"].append(new_id)
        check("ownership_revision_is_ten", updated["revision"] == 10
              and updated["correction"]["supersedes"] == payload["supersedes"])
        if user_message_id:
            try:
                comparison = operation("generate_historical_and_current_comparison", "/api/compare",
                    {"chat_id": chat["id"], "message_id": user_message_id, "baseline_revision": 9})
                record_answer(cases[1], comparison["original"], 9, expected_revision=9, historical=True)
                record_answer(cases[2], comparison["edited"], comparison["revision"], expected_revision=10)
                check("comparison_preserves_current_revision", comparison["revision"] == 10
                      and comparison.get("baseline_label") == "APPROVED MEMORY r9")
            except Exception as exc:
                for case in cases[1:3]:
                    if case["outcome"] == "unexecuted":
                        failed_case(case, exc)
        else:
            for case in cases[1:3]:
                case["unexecuted_reason"] = "Initial generation failed, leaving no persisted user message for comparison. No retry."
            save()

        rolled = operation("atomic_rollback_ownership", f"/api/corrections/{new_id}/rollback", {})
        check("rollback_single_revision_restores_predecessor", rolled["revision"] == 11
              and rolled["correction"]["id"] == payload["supersedes"] and rolled["correction"]["status"] == "active")
        after = operation("inspect_atomic_rollback_events", "/api/corrections")
        by_id = {row["id"]: row for row in after["corrections"]}
        check("rollback_memory_consistent", after["revision"] == 11 and len(by_id) == 10
              and sum(row["status"] == "active" for row in by_id.values()) == 9
              and by_id[new_id]["status"] == "withdrawn" and by_id[payload["supersedes"]]["status"] == "active")
        check("rollback_audit_events_same_revision", [(row["action"], row["revision"]) for row in after["events"][-2:]]
              == [("withdrawn", 11), ("restored", 11)])
        try:
            restored = operation("generate_restored_contact", "/api/chat",
                                 {"chat_id": chat["id"], "message": question["question"], "mode": "edited"})
            record_answer(cases[3], restored["assistant"], restored["revision"], expected_revision=11)
        except Exception as exc:
            failed_case(cases[3], exc)
        receipt["status"] = "completed" if all(row["outcome"] != "unexecuted" for row in cases) else "interrupted"
    except Exception as exc:
        receipt.update(status="interrupted", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        cleanup_errors = []
        if client.token:
            for fact_id in receipt["owned_fact_ids"]:
                try:
                    operation("withdraw_owned_fact", f"/api/corrections/{fact_id}/withdraw", {})
                except Exception as exc:
                    cleanup_errors.append({"fact_id": fact_id, "error": str(exc)})
        # Read-only confirmation must not try to acquire or delete unknown IDs.
        if receipt["owned_fact_ids"] and client.token:
            try:
                final_memory = operation("confirm_owned_facts_inactive", "/api/corrections")
                own = set(receipt["owned_fact_ids"])
                receipt["owned_facts_inactive_at_end"] = not any(row["id"] in own and row["status"] == "active"
                                                               for row in final_memory["corrections"])
                if not receipt["owned_facts_inactive_at_end"]:
                    cleanup_errors.append({"error": "A recorded owned fact remains active"})
            except Exception as exc:
                cleanup_errors.append({"error": "Could not confirm cleanup: " + str(exc)})
        receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(), total_seconds=time.perf_counter() - started,
            outcome_counts={outcome: sum(row["outcome"] == outcome for row in cases) for outcome in
                            ("correct", "wrong", "abstain", "execution_error", "unexecuted")},
            cleanup_errors=cleanup_errors,
            all_consistency_checks_passed=all(receipt["consistency_checks"].values()))
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base_url, args.output)
    print(json.dumps({key: result[key] for key in ("status", "outcome_counts", "consistency_checks", "cleanup_errors")}))
    if result["status"] != "completed" or result["cleanup_errors"] or not result["all_consistency_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
