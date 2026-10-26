"""Twelve real website requests, including neural answers after historical restoration.

Separate operational supplement, not extra independent held-out benchmark cases.
Only this workload's own records are changed; active user memory is never cleared.
"""
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from scripts.verify_reasonke_live import Client, atomic_write, MODEL_ID, MODEL_REVISION
from editguard_benchmark.data import load_dataset, digest
from editguard_benchmark.metrics import quantile
from editguard_benchmark.scoring import score


def run(base, output, dataset="data/reasonke/operations-v1"):
    output = Path(output)
    if output.exists():
        raise ValueError("Operational receipt already exists; it is never overwritten or answer-retried")
    manifest, worlds, questions = load_dataset(Path(dataset), "development")
    selected_worlds = sorted(worlds)[:3]
    cases = []
    for uid in selected_worlds:
        group = [row for row in questions if row["lineage_id"] == uid]
        for category in ("direct", "direct_paraphrase", "two_hop"):
            question = next(row for row in group if row["category"] == category)
            cases.append({"id": uid + ":" + category, "world": uid, "question": question, "outcome": "unexecuted"})
        question = copy.deepcopy(next(row for row in group if row["category"] == "direct"))
        question.update(expected_answer=question["initial_answer"], expected_aliases=[question["initial_answer"]])
        cases.append({"id": uid + ":restored_initial_fact", "world": uid, "question": question, "outcome": "unexecuted"})
    receipt = {"schema": "editguard-operations-http-v1", "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "assigned_model_requests": 12, "dataset": manifest, "cases": cases, "operations": [], "owned_fact_ids": [],
        "script_sha256": digest(__file__), "notice": "Development-world operational supplement; 12 real HTTP requests, not 12 independent new test facts",
        "latency_scope": "Client wall time includes website request, retrieval, model-worker HTTP, generation, SQLite writes and response parsing. Model already loaded; no browser rendering.",
        "authorization_headers_recorded": False, "model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    client = Client(base)
    started = time.perf_counter()
    atomic_write(output, receipt)
    def operation(name, path, payload=None):
        began = time.perf_counter()
        row = {"name": name, "path": path}
        receipt["operations"].append(row)
        try:
            result = client.request(path, payload)
            row.update(status="completed", response={key: value for key, value in result.items() if key != "token"})
            return result
        except Exception as exc:
            row.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            row["seconds"] = time.perf_counter() - began
            atomic_write(output, receipt)
    try:
        status = operation("inspect_ready_model", "/api/status")
        client.token = status.get("token")
        model = status.get("model", {})
        if not (model.get("ready") and model.get("name") == MODEL_ID and model.get("revision") == MODEL_REVISION):
            raise ValueError("The ready pinned ReasonKE website is required")
        memory = operation("inspect_empty_active_memory", "/api/corrections")
        if any(row["status"] == "active" for row in memory["corrections"]):
            raise ValueError("Use a separate empty workspace; existing active facts were not changed")
        for uid in selected_worlds:
            world = worlds[uid]
            original_owner = None
            owned_in_world = []
            for fact in world["initial_facts"]:
                payload = {key: fact[key] for key in ("question", "answer", "scope", "source")}
                payload["identity"] = {key: fact[key] for key in ("entity_id", "relation_id", "scope_id")}
                result = operation("approve_initial_fact", "/api/corrections", payload)
                fact_id = result["correction"]["id"]
                receipt["owned_fact_ids"].append(fact_id); owned_in_world.append(fact_id)
                if fact["id"] == world["edit"]["fact_id"]:
                    original_owner = fact_id
            updated = next(row for row in world["current_facts"] if row["id"] == world["edit"]["fact_id"])
            payload = {key: updated[key] for key in ("question", "answer", "scope", "source")}
            payload.update(identity={key: updated[key] for key in ("entity_id", "relation_id", "scope_id")}, supersedes=original_owner)
            result = operation("approve_ownership_revision", "/api/corrections", payload)
            new_id = result["correction"]["id"]
            receipt["owned_fact_ids"].append(new_id); owned_in_world.append(new_id)
            for case in [row for row in cases if row["world"] == uid]:
                if case["id"].endswith("restored_initial_fact"):
                    operation("withdraw_ownership_revision", f"/api/corrections/{new_id}/withdraw", {})
                    operation("restore_initial_owner", f"/api/corrections/{original_owner}/restore", {})
                chat = operation("create_fresh_measurement_chat", "/api/chats", {})
                began = time.perf_counter()
                try:
                    result = client.request("/api/chat", {"chat_id": chat["chat"]["id"], "message": case["question"]["question"], "mode": "edited"})
                    answer = result["assistant"]
                    evidence = answer.get("evidence", {})
                    generation = evidence.get("generation", {})
                    if (generation.get("model") != MODEL_ID or generation.get("revision") != MODEL_REVISION
                            or evidence.get("generation_kind") != "model"):
                        raise ValueError("Response is not a real pinned-model generation")
                    case.update(response=answer, revision=result["revision"], http_seconds=time.perf_counter() - began)
                    case.update(score(case["question"], {"content": answer["content"]}))
                except Exception as exc:
                    case.update(error={"type": type(exc).__name__, "message": str(exc)}, http_seconds=time.perf_counter() - began)
                    case.update(score(case["question"], error=case["error"]))
                atomic_write(output, receipt)
                print(json.dumps({"id": case["id"], "outcome": case["outcome"], "http_seconds": case["http_seconds"]}), flush=True)
            for fact_id in owned_in_world:
                operation("withdraw_owned_world_fact", f"/api/corrections/{fact_id}/withdraw", {})
        receipt["status"] = "completed"
    except Exception as exc:
        receipt.update(status="interrupted", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        # Only own IDs are eligible for cleanup, including after an interrupted call.
        cleanup_errors = []
        if client.token:
            for fact_id in receipt["owned_fact_ids"]:
                try:
                    client.request(f"/api/corrections/{fact_id}/withdraw", {})
                except Exception as exc:
                    cleanup_errors.append({"fact_id": fact_id, "error": str(exc)})
        times = [row["http_seconds"] for row in cases if row.get("response")]
        receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(), total_seconds=time.perf_counter() - started,
            outcome_counts={outcome: sum(row["outcome"] == outcome for row in cases) for outcome in
                            ("correct", "wrong", "abstain", "execution_error", "unexecuted")},
            latency={"generated_n": len(times), "p50_seconds": quantile(times, .5), "p95_seconds": quantile(times, .95)},
            cleanup_errors=cleanup_errors, neural_accuracy_denominator=12)
        atomic_write(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.base_url, args.output)
    print(json.dumps({key: result[key] for key in ("status", "outcome_counts", "latency", "cleanup_errors")}))
    if result["status"] != "completed" or result["cleanup_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
