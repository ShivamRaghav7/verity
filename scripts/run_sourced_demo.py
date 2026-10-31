"""Finite seven-generation sourced website demonstration; never a held-out benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from editguard_benchmark.scoring import normalize
from scripts.prepare_sourced_demo import FIXTURE, load_fixture


def run(base_url, manifest, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Existing demonstration evidence is preserved; choose a fresh planned run")
    fixture = load_fixture()
    prepared = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if prepared["fixture_sha256"] != hashlib.sha256(FIXTURE.read_bytes()).hexdigest():
        raise ValueError("The prepared workspace and fixture must match")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "verity-sourced-demo-execution-v1", "purpose": fixture["purpose"],
               "fixture_sha256": prepared["fixture_sha256"], "notice": fixture["notice"],
               "planned_model_calls": 7, "benchmark_observations": 0,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "status": "running", "operations": [], "cases": [], "snapshots": {}}
    def save():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(output)
    token = None
    def request(path, payload=None):
        headers = {} if payload is None else {"Content-Type": "application/json", "X-EditGuard-Token": token}
        body = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(base_url.rstrip("/") + path, data=body, headers=headers), timeout=300) as response:
            return json.load(response)
    status = request("/api/status")
    token = status["token"]
    if not status["model"]["ready"] or status["revision"] != prepared["initial_revision"]:
        raise ValueError("Use the prepared fresh workspace and a ready model")
    receipt["model"] = {key: status["model"].get(key) for key in ("name", "revision", "quantization", "hardware")}
    def snapshot(label):
        receipt["snapshots"][label] = request("/api/workspace/export")
        save()
    def ask(label, question, aliases):
        chat = request("/api/chats", {})["chat"]
        row = {"id": label, "question": question, "expected_aliases": aliases,
               "chat_id": chat["id"], "request_state": "intent_saved"}
        receipt["cases"].append(row)
        save()
        start = time.perf_counter()
        try:
            response = request("/api/chat", {"chat_id": chat["id"], "message": question, "mode": "edited"})
            assistant = response["assistant"]
            if assistant.get("evidence", {}).get("generation_kind") != "model":
                raise ValueError("A deterministic response cannot be recorded as a neural generation")
            row.update(request_state="returned", response=response, http_seconds=time.perf_counter()-start,
                       final_span_matches_target=normalize(assistant["content"]) in {normalize(a) for a in aliases})
            save()
            print(label + ": " + assistant["content"], flush=True)
        except Exception as exc:
            row.update(request_state="error_or_unavailable", error_type=type(exc).__name__)
            save()
            raise
    save()
    try:
        snapshot("initial")
        ask("initial_direct", fixture["questions"]["direct"], ["Government of India", "Indian government"])
        replacement = request("/api/corrections", prepared["suggested_replacement"])["correction"]
        receipt["operations"].append({"action": "publish", "correction": replacement})
        revision = request("/api/status")["revision"]
        attempted = dict(prepared["suggested_replacement"])
        attempted.pop("supersedes")
        try:
            request("/api/corrections", attempted)
            raise AssertionError("Competing approval was accepted")
        except HTTPError as exc:
            if exc.code != 409:
                raise
            receipt["operations"].append({"action": "competing_approval", "http_status": 409,
                                           "revision_unchanged": request("/api/status")["revision"] == revision})
        for label in ("direct", "connected", "three_hop", "historical_control"):
            ask("updated_" + label, fixture["questions"][label], fixture["expected_after"][label])
        snapshot("updated")
        restored = request("/api/corrections/" + replacement["id"] + "/rollback", {})["correction"]
        receipt["operations"].append({"action": "rollback", "restored": restored,
                                       "restored_owned_predecessor": restored["id"] == prepared["fact_ids"]["selected_owner"]})
        ask("rollback_direct", fixture["questions"]["direct"], ["Government of India", "Indian government"])
        snapshot("rollback")
        final_payload = dict(prepared["suggested_replacement"], supersedes=restored["id"])
        final = request("/api/corrections", final_payload)["correction"]
        receipt["operations"].append({"action": "reapprove_handover_for_active_demo", "correction": final})
        ask("active_connected", fixture["questions"]["connected"], fixture["expected_after"]["connected"])
        snapshot("active")
        receipt.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                       completed_model_calls=len(receipt["cases"]),
                       target_matches=sum(row.get("final_span_matches_target", False) for row in receipt["cases"]))
        save()
    except Exception as exc:
        receipt.update(status="interrupted_preserved", error_type=type(exc).__name__)
        save()
        raise
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.base_url, args.manifest, args.output)
    print(json.dumps({key: report[key] for key in ("status", "completed_model_calls", "target_matches", "benchmark_observations")}, indent=2))
