"""Finite, isolated API lifecycle demonstration with actual trained checkpoints.

Governance/integration checks and generated-answer scoring are distinct. No
training, tuning, final-test reads or connection to the live HTTP server occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_REQUESTS = 40
SCOPE = {"entity_id": "nimbus", "plan": "Basic", "version": "v2", "as_of": "2026-10-13"}
DIRECT = "How many devices does Nimbus Basic version v2 support?"
PARAPHRASE = "What is the device allowance on Nimbus Cloud Basic version v2?"
SOURCE = "controlled://isolated-learned-workflow/approved-3"
CONFLICT_SOURCE = "controlled://isolated-learned-workflow/conflicting-4"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protect_paths(database: Path, output: Path):
    database, output = database.resolve(), output.resolve()
    runtime = (ROOT / "runtime").resolve()
    if runtime not in database.parents or database == runtime / "editguard.sqlite3":
        raise ValueError("Verification database must be a new isolated fixture under runtime/.")
    if database.exists() or Path(str(database) + "-wal").exists() or Path(str(database) + "-shm").exists():
        raise ValueError("Fixture exists; use a fresh database and preserve earlier evidence.")
    if output.exists() or output == database:
        raise ValueError("Report already exists or overlaps the fixture; choose a fresh report.")
    return database, output


def verify_workflow(client, report, token):
    """Run at most forty in-process API requests; no retry or live server access."""
    from editguard.scoring import score_answer
    from editguard.short_qa import score_short_qa

    headers = {"X-Admin-Token": token}
    checks, requests, answers = report["checks"], report["requests"], report["model_answers"]
    facts = {}

    def request(name, method, path, payload=None, administrator=False):
        if len(requests) >= MAX_REQUESTS:
            raise RuntimeError("Finite workflow request cap exceeded.")
        started = perf_counter()
        response = client.request(method, path, json=payload, headers=headers if administrator else {})
        elapsed = round((perf_counter() - started) * 1000, 3)
        body = response.json()
        row = {"phase": name, "method": method, "path": path,
               "payload": payload, "http_status": response.status_code,
               "wall_ms": elapsed, "body": body}
        requests.append(row)
        return row

    def check(name, condition, category="governance", detail=None):
        checks.append({"name": name, "category": category, "passed": bool(condition), "detail": detail})

    def ask(name, question=DIRECT, *, mode="structured", scope=None, expected=None,
            expected_polarity=None, factual=True, model_phase="warm_resident_models"):
        context = {**SCOPE, **(scope or {})}
        response = request(name, "POST", "/api/ask", {**context, "question": question, "mode": mode})
        if mode != "structured":
            body = response["body"]
            truth = None
            if expected is not None:
                fact_id, value = expected
                truth = {**facts[fact_id], "value": value, "as_of": context["as_of"]}
            row = {**body, "id": name, "expected_fact": truth,
                   "expected_polarity": expected_polarity, "request_context": context,
                   "phase": name, "timing_phase": model_phase, "wall_ms": response["wall_ms"],
                   "factual_answer_case": factual, "review_policy": "automated_development",
                   "human_reviewed_data": False}
            if factual and truth:
                if body.get("status") == "answered":
                    row["score"] = score_answer(body.get("answer"), truth, expected_polarity=expected_polarity,
                                                answer_kind=body.get("answer_kind", "unspecified"))
                else:
                    row["score"] = {"correct": False, "verdict": "not_answered",
                                    "reason": "All failed, unavailable or abstained factual cases receive no credit.",
                                    "answer_kind": body.get("answer_kind"), "scorer_version": __import__("editguard.scoring", fromlist=["SCORER_VERSION"]).SCORER_VERSION}
                row["short_qa"] = score_short_qa(row)
            answers.append(row)
        return response

    def structured_correct(row, value, route=None):
        body = row["body"]
        return (row["http_status"] == 200 and body.get("status") == "answered"
                and body.get("answer_kind") == "deterministic_structured"
                and body.get("fact", {}).get("fact_id") == "nimbus-basic-v2-devices"
                and body.get("answer") == f"Nimbus Basic v2 supports up to {value} devices."
                and (route is None or body.get("route") == route))

    health = request("initial_health", "GET", "/api/health")
    history = request("initial_history", "GET", "/api/edits")
    check("fresh_seeded_isolated_memory", health["body"].get("revision") == 1 and history["body"].get("edits") == [])
    check("actual_trained_artifacts_eligible", health["body"].get("models") == {"scope": "trained", "answer": "trained"}, "integration")
    check("learned_mode_requires_real_loading", health["body"].get("modes", {}).get("learned", {}).get("status") == "load_required", "integration")
    if not all(row["passed"] for row in checks):
        raise RuntimeError("Fresh-instance or actual trained-artifact precondition failed; no mutation attempted.")
    initial_facts = request("initial_facts", "GET", "/api/facts")
    facts.update({fact["fact_id"]: fact for fact in initial_facts["body"]["facts"]})
    original = ask("initial_structured")
    check("initial_fact_is_five", structured_correct(original, "5", "initial_catalogue"))
    cold = ask("initial_learned_cold", mode="learned", expected=("nimbus-basic-v2-devices", "5"),
               model_phase="cold_checkpoint_loading_and_first_inference")
    check("actual_neural_api_loaded", cold["http_status"] == 200 and cold["body"].get("status") == "answered", "integration")
    loaded_health = request("loaded_health", "GET", "/api/health")
    check("both_auxiliary_checkpoints_loaded", all(loaded_health["body"].get("model_details", {}).get(task, {}).get("loaded")
          and loaded_health["body"]["model_details"][task].get("status") == "trained" for task in ("scope", "answer")), "integration")
    ask("initial_learned_warm", mode="learned", expected=("nimbus-basic-v2-devices", "5"))
    correction = {"fact_id": "nimbus-basic-v2-devices", "original_value": "5", "corrected_value": "3",
                  "source": SOURCE, "reason": "Finite isolated actual-checkpoint lifecycle verification",
                  "effective_from": "2026-10-01"}
    unauthorized = request("unauthorized_correction_rejected", "POST", "/api/edits", correction)
    check("administrator_boundary", unauthorized["http_status"] == 401)
    draft = request("create_draft", "POST", "/api/edits", correction, True)
    check("draft_created", draft["http_status"] == 201 and draft["body"].get("status") == "draft")
    edit_id = draft["body"]["edit_id"]
    edit_path = f"/api/edits/{edit_id}"
    draft_answer = ask("draft_structured")
    check("draft_has_no_authority", structured_correct(draft_answer, "5", "initial_catalogue"))
    approved = request("approve_correction", "POST", edit_path + "/approve", administrator=True)
    check("approved_correction_authoritative", approved["http_status"] == 200 and approved["body"].get("status") == "approved")
    corrected = ask("corrected_structured")
    check("approved_fact_is_three_with_source", structured_correct(corrected, "3", "approved_correction")
          and any(edit.get("source") == SOURCE and edit.get("edit_id") == edit_id for edit in corrected["body"].get("applicable_corrections", [])))
    corrected_requests = [
        ("corrected_direct_learned", DIRECT, None),
        ("corrected_paraphrase_learned", PARAPHRASE, None),
        ("corrected_capacity_yes_learned", "Can I connect 3 devices on Nimbus Basic version v2?", "yes"),
        ("corrected_capacity_no_learned", "Can I connect 4 devices on Nimbus Basic version v2?", "no"),
    ]
    for name, question, polarity in corrected_requests:
        answer = ask(name, question, mode="learned", expected=("nimbus-basic-v2-devices", "3"), expected_polarity=polarity)
        check(name + "_real_scope_call", answer["http_status"] == 200 and bool(answer["body"].get("scope_scores")), "integration")
    neighbors = [
        ("price_out_of_scope", "What is the monthly price of Nimbus Basic version v2?", {}, "nimbus-basic-v2-price", "9"),
        ("pro_out_of_scope", "How many devices does Nimbus Pro version v2 support?", {"plan": "Pro"}, "nimbus-pro-v2-devices", "12"),
        ("version_out_of_scope", "How many devices does Nimbus Basic version v1 support?", {"version": "v1"}, "nimbus-basic-v1-devices", "5"),
        ("similar_name_out_of_scope", "How many devices does Nimbus Mini Basic version v2 support?", {"entity_id": "nimbus-mini"}, "nimbus-mini-basic-v2-devices", "2"),
        ("historical_date_out_of_scope", DIRECT, {"as_of": "2026-09-01"}, "nimbus-basic-v2-devices", "5"),
    ]
    for name, question, scope, fact_id, value in neighbors:
        answer = ask(name, question, mode="learned", scope=scope, expected=(fact_id, value))
        if name != "price_out_of_scope":
            check(name + "_compatibility_guard", answer["http_status"] == 200 and answer["body"].get("selected_edit_id") is None)
    ambiguous = ask("ambiguous_plan_request", "How many devices do Nimbus Basic and Pro version v2 support?", mode="learned", factual=False)
    check("explicit_ambiguity_clarifies", ambiguous["http_status"] == 200 and ambiguous["body"].get("status") == "clarification"
          and ambiguous["body"].get("answer_kind") == "deterministic_clarification")
    conflicting = {**correction, "original_value": "3", "corrected_value": "4", "source": CONFLICT_SOURCE}
    competitor = request("create_conflicting_draft", "POST", "/api/edits", conflicting, True)
    check("conflicting_draft_recorded", competitor["http_status"] == 201 and competitor["body"].get("status") == "draft")
    competitor_id = competitor["body"]["edit_id"]
    competitor_path = f"/api/edits/{competitor_id}"
    rejected = request("conflicting_approval_rejected", "POST", competitor_path + "/approve", administrator=True)
    check("contradictory_approval_rejected", rejected["http_status"] == 409)
    blocked = ask("conflict_blocks_learned", mode="learned", factual=False)
    check("conflict_withholds_model_answer", blocked["http_status"] == 200 and blocked["body"].get("status") == "conflict"
          and blocked["body"].get("answer_kind") == "deterministic_clarification" and blocked["body"].get("generation_trace") is None)
    conflict_list = request("visible_conflict", "GET", "/api/conflicts")
    check("administrator_can_inspect_conflict", conflict_list["http_status"] == 200 and bool(conflict_list["body"].get("conflicts")))
    withdrawal = request("withdraw_conflicting_draft", "POST", competitor_path + "/withdraw", administrator=True)
    check("conflict_withdrawal_recorded", withdrawal["http_status"] == 200 and withdrawal["body"].get("status") == "withdrawn")
    resolved = ask("resolution_structured")
    check("resolution_restores_three", structured_correct(resolved, "3", "approved_correction"))
    ask("resolved_learned", PARAPHRASE, mode="learned", expected=("nimbus-basic-v2-devices", "3"))
    rollback = request("rollback_correction", "POST", edit_path + "/withdraw", administrator=True)
    check("rollback_withdrawal_recorded", rollback["http_status"] == 200 and rollback["body"].get("status") == "withdrawn")
    restored = ask("rollback_structured")
    check("rollback_restores_original_five", structured_correct(restored, "5", "initial_catalogue"))
    restored_model = ask("rollback_learned", mode="learned", expected=("nimbus-basic-v2-devices", "5"))
    check("rollback_removes_edit_from_model_memory", restored_model["http_status"] == 200
          and restored_model["body"].get("selected_edit_id") is None
          and restored_model["body"].get("retrieval_candidates_eligible") == []
          and "supports 5 devices" in (restored_model["body"].get("input_text") or ""))
    history = request("final_history", "GET", "/api/edits")
    states = {edit["edit_id"]: edit for edit in history["body"]["edits"]}
    check("history_retains_both_withdrawn_sources", set(states) == {edit_id, competitor_id}
          and all(edit.get("status") == "withdrawn" for edit in states.values())
          and states[edit_id].get("source") == SOURCE and states[competitor_id].get("source") == CONFLICT_SOURCE)
    report["fixture"] = {"approved_edit": approved["body"], "conflicting_edit": competitor["body"],
                         "final_history": history["body"], "facts": list(facts.values())}
    return report["fixture"]


def run(config: Path, database: Path, output: Path):
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from the repository root so configured local paths resolve consistently.")
    database, output = protect_paths(database, output)
    config = config.resolve()
    configured = json.loads(config.read_text(encoding="utf-8"))
    if configured.get("local_files_only") is not True or configured.get("device") != "cuda":
        raise ValueError("This finite verification requires existing local checkpoints on the profiled CUDA device.")
    watched = {"inference_configuration": config, "catalogue": ROOT / "data/catalogue.json",
               "api_source": ROOT / "editguard/api.py", "comparison_source": ROOT / "editguard/comparison.py",
               "neural_source": ROOT / "editguard/neural.py", "formatting_source": ROOT / "editguard/formatting.py",
               "primary_scorer_source": ROOT / "editguard/scoring.py", "short_qa_source": ROOT / "editguard/short_qa.py",
               "workflow_source": Path(__file__)}
    for task in ("scope", "answer"):
        artifact_root = Path(configured[task + "_artifact"])
        watched[task + "_manifest"] = artifact_root / "manifest.json"
        manifest = json.loads(watched[task + "_manifest"].read_text(encoding="utf-8"))
        for name in manifest.get("files_sha256", manifest.get("file_hashes", {})):
            watched[task + "_file:" + name] = artifact_root / name
    for key in ("calibration_artifact", "similarity_calibration_artifact"):
        watched[key] = Path(configured[key])
    hashes = {name: file_hash(path) for name, path in watched.items()}
    report = {"schema_version": 1, "purpose": "isolated_actual_trained_workflow_engineering",
              "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "database": str(database), "inference_configuration": str(config),
              "live_application_accessed": False, "training_executed": False, "test_rows_read": False,
              "review_policy": "automated_development", "human_reviewed_data": False,
              "max_api_requests": MAX_REQUESTS, "oracle_answer_calls": 2,
              "checks": [], "requests": [], "model_answers": [], "oracle_answers": [],
              "source_and_artifact_sha256_before": hashes, "error": None}
    started = perf_counter()
    try:
        os.environ.setdefault("HF_HOME", str(ROOT / "runtime/hf-cache"))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("The configured CUDA device is unavailable.")
        torch.cuda.reset_peak_memory_stats()
        from fastapi.testclient import TestClient
        from editguard.api import create_app
        token = secrets.token_urlsafe(32)
        with TestClient(create_app(database, token, inference_config=config)) as client:
            fixture = verify_workflow(client, report, token)
        # Read the same closed fixture back from SQLite; no live-app state is used.
        from editguard.store import Store
        store = Store(database)
        try:
            final_value = store.resolve_fact("nimbus-basic-v2-devices", SCOPE["as_of"])
            persisted = store.edits()
            report["checks"].append({"name": "sqlite_reopen_preserves_rollback", "category": "governance",
                "passed": final_value["value"] == "5" and not final_value["conflict"]
                and len(persisted) == 2 and all(edit["status"] == "withdrawn" for edit in persisted), "detail": None})
        finally:
            store.close()
        # Explicit oracle probes guarantee answer-checkpoint execution even if
        # the learned scope gate declined this unfamiliar demonstration entity.
        from editguard.neural import AnswerCheckpoint
        from editguard.formatting import conditional_input
        from editguard.scoring import score_answer
        from editguard.short_qa import score_short_qa
        answer_model = AnswerCheckpoint(configured["answer_artifact"], configured["device"], configured["max_new_tokens"])
        correction = fixture["approved_edit"]
        support = [fact for fact in fixture["facts"] if fact["entity_id"] == "nimbus"
                   and fact["plan"] == "Basic" and fact["version"] == "v2" and fact["relation"] != "device_limit"][:2]
        truth = {**next(fact for fact in fixture["facts"] if fact["fact_id"] == "nimbus-basic-v2-devices"),
                 "value": "3", "as_of": SCOPE["as_of"]}
        for name, question, polarity in (("oracle_direct", DIRECT, None),
                                        ("oracle_capacity_no", "Can I connect 4 devices on Nimbus Basic version v2?", "no")):
            prompt = conditional_input(question, correction, support, SCOPE["as_of"])
            inference_started = perf_counter()
            answer = answer_model.generate(prompt)
            row = {"id": name, "configuration": "oracle_answer", "question": question,
                   "request_context": SCOPE, "expected_fact": truth, "expected_polarity": polarity,
                   "status": "answered" if answer else "abstention", "answer": answer,
                   "answer_kind": "model_generated", "route": "oracle_correct_routing_diagnostic",
                   "input_text": prompt, "generation_trace": answer_model.last_trace,
                   "wall_ms": round((perf_counter() - inference_started) * 1000, 3),
                   "training_manifest_sha256": file_hash(Path(configured["answer_artifact"]) / "manifest.json")}
            row["score"] = score_answer(answer, truth, expected_polarity=polarity)
            row["short_qa"] = score_short_qa(row)
            report["oracle_answers"].append(row)
        report["resource_usage"] = {"torch_version": torch.__version__, "device": torch.cuda.get_device_name(),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2,
            "second_answer_checkpoint_for_oracle": True}
        report["status"] = "passed" if all(check["passed"] for check in report["checks"]) else "failed"
    except Exception as error:
        report["status"], report["error"] = "failed", f"{type(error).__name__}: {error}"
    finally:
        report["source_and_artifact_sha256_after"] = {name: file_hash(path) for name, path in watched.items()}
        report["source_and_artifact_unchanged"] = hashes == report["source_and_artifact_sha256_after"]
        if not report["source_and_artifact_unchanged"]:
            report["status"] = "failed"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = round(perf_counter() - started, 3)
        factual = [row for row in report["model_answers"] if row.get("factual_answer_case")]
        report["generated_answer_diagnostic"] = {"denominator": len(factual),
            "primary_correct": sum((row.get("score") or {}).get("correct") is True for row in factual),
            "primary_pending_uncredited": sum((row.get("score") or {}).get("correct") is None for row in factual),
            "short_qa_correct": sum((row.get("short_qa") or {}).get("correct") is True for row in factual),
            "short_qa_pending_uncredited": sum((row.get("short_qa") or {}).get("correct") is None for row in factual),
            "definition": "Handwritten engineering demonstration cases including repeated questions; not a representative accuracy estimate or final evaluation."}
        report["checks_passed"] = sum(row["passed"] for row in report["checks"])
        report["checks_denominator"] = len(report["checks"])
        report["api_requests_executed"] = len(report["requests"])
        report["limitations"] = ["Governance checks never require model-generated factual correctness.",
            "Cold timing includes checkpoint validation/loading and first inference; warm timings include API before-answer generation.",
            "The fixed demonstration cases contain repetition and are not a statistical evaluation population.",
            "Oracle probes use supplied correct routing and never count as end-to-end learned routing success.",
            "Model artifacts and labels have disclosed automated development review, without independent human review.",
            "A source, model trace or successful workflow does not verify arbitrary generated prose."]
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/inference-pilot-v2.json")
    parser.add_argument("--database", type=Path, default=ROOT / f"runtime/learned-workflow-{uuid.uuid4().hex}.sqlite3")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pilot-v2-learned-workflow.json")
    args = parser.parse_args()
    report = run(args.config, args.database, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "checks": [report["checks_passed"], report["checks_denominator"]],
                      "requests": report["api_requests_executed"], "generated_answers": report["generated_answer_diagnostic"],
                      "error": report["error"]}))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
