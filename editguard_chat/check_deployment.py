"""Real HTTP/SQLite deployment contracts. No model is loaded or asked to answer."""
import argparse
import copy
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .server import ChatApplication, handler_for
from .store import StoreError


class NoModel:
    def status(self):
        return {"ready": False, "name": "explicit deployment contract: no model loaded"}

    def generate(self, *args):
        raise AssertionError("Deployment checks must not generate or substitute model answers")


class NoRetriever:
    method = "deployment contract fixture, no retrieval"
    threshold = None

    def clear_cache(self):
        pass


def verify():
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)
    with tempfile.TemporaryDirectory() as directory:
        app = ChatApplication(Path(directory) / "source", backend=NoModel(), retriever=NoRetriever())
        class Handler(handler_for(app)):
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        def request(path, payload=None, token=True):
            headers = {"Content-Type": "application/json"}
            if token: headers["X-EditGuard-Token"] = app.token
            req = Request(base + path, None if payload is None else json.dumps(payload).encode(), headers=headers)
            try: response = urlopen(req, timeout=5)
            except HTTPError as exc: response = exc
            with response: return response.status, json.loads(response.read())
        try:
            code, status = request("/api/status")
            check("status_explicitly_has_no_loaded_model", code == 200 and status["model"]["ready"] is False)
            code, _ = request("/api/chats", {}, token=False)
            check("mutation_requires_session_token", code == 403)
            code, chat = request("/api/chats", {})
            chat_id = chat["chat"]["id"]
            check("create_chat_http", code == 200)
            code, _ = request("/api/chat", {"chat_id": chat_id, "message": "Hello"})
            check("no_model_never_substitutes_an_answer", code == 503 and app.store.list_messages(chat_id) == [])
            payload = {"question": "Who owns Atlas?", "answer": "Amber", "scope": "Atlas r2",
                       "source": "Fictional deployment fixture", "identity": {"entity_id": "atlas", "relation_id": "owner", "scope_id": "r2"}}
            code, first = request("/api/corrections", payload)
            first_id = first["correction"]["id"]
            check("publish_canonical_fact", code == 200 and first["revision"] == 1)
            code, conflict = request("/api/corrections", {**payload, "question": "Which team maintains Atlas?", "answer": "Indigo"})
            check("paraphrased_competing_fact_blocked", code == 409 and conflict["conflicts"][0]["id"] == first_id)
            check("rejected_conflict_preserves_revision", app.store.get_revision() == 1)
            code, replacement = request("/api/corrections", {**payload, "question": "Which team maintains Atlas?", "answer": "Indigo", "supersedes": first_id})
            current_id = replacement["correction"]["id"]
            check("explicit_replacement_keeps_lineage", code == 200 and replacement["correction"]["lineage_id"] == first_id and replacement["correction"]["version"] == 2)
            code, _ = request(f"/api/corrections/{first_id}/restore", {})
            check("restore_cannot_overlap_active_fact", code == 409)
            code, result = request(f"/api/corrections/{current_id}/withdraw", {})
            check("withdraw_removes_current_fact", code == 200 and result["correction"]["status"] == "withdrawn")
            before = app.store.get_revision()
            request(f"/api/corrections/{current_id}/withdraw", {})
            check("withdraw_is_idempotent", app.store.get_revision() == before)
            code, result = request(f"/api/corrections/{first_id}/restore", {})
            check("historical_version_restores", code == 200 and result["correction"]["status"] == "active")
            code, replacement = request("/api/corrections", {**payload, "answer": "Violet", "supersedes": first_id})
            active_id, before = replacement["correction"]["id"], replacement["revision"]
            code, result = request(f"/api/corrections/{active_id}/rollback", {})
            check("atomic_rollback_one_revision", code == 200 and result["revision"] == before + 1 and result["correction"]["id"] == first_id)
            last_events = app.store.list_events()[-2:]
            check("rollback_two_audit_events_same_revision", [(row["action"], row["revision"]) for row in last_events] == [("withdrawn", before + 1), ("restored", before + 1)])
            code, result = request(f"/api/corrections/{active_id}/rollback", {})
            check("rollback_is_idempotent", code == 200 and result["revision"] == before + 1)
            code, snapshot = request("/api/workspace/export")
            check("portable_text_snapshot_export", code == 200 and snapshot["schema"] == "editguard-workspace-v1")
            code, _ = request("/api/workspace/restore", {"snapshot": snapshot})
            check("restore_refuses_existing_work", code == 409)
            target = ChatApplication(Path(directory) / "restored", backend=NoModel(), retriever=NoRetriever())
            try:
                result = target.mutate("/api/workspace/restore", {"snapshot": snapshot})
                check("restore_exact_revision", result["revision"] == app.store.get_revision())
                check("restore_exact_facts", target.store.list_corrections() == app.store.list_corrections())
                check("restore_exact_audit_history", target.store.list_events() == app.store.list_events())
                check("restore_exact_conversations", target.store.list_chats() == app.store.list_chats())
            finally:
                target.store.close()
            corrupt = copy.deepcopy(snapshot); corrupt["sha256"] = "bad"
            empty = ChatApplication(Path(directory) / "empty", backend=NoModel(), retriever=NoRetriever())
            try:
                try: empty.mutate("/api/workspace/restore", {"snapshot": corrupt})
                except StoreError: rejected = True
                else: rejected = False
                check("corrupt_snapshot_rejected_atomically", rejected and empty.store.list_chats() == [] and empty.store.get_revision() == 0)
            finally:
                empty.store.close()
            code, evidence = request("/api/experiments")
            check("current_evidence_files_packaged", code == 200 and evidence["model"] == "YukinoKaren/ReasonKE_plus")
            check("evidence_kind_explicit", evidence["kind"] in ("development_acceptance", "controlled_development_benchmark", "controlled_final_benchmark"))
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5); app.store.close()
    return {"schema": "editguard-chat-deployment-check-v1", "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
            "checks": checks, "passed": sum(checks.values()), "assigned": len(checks),
            "model_generations": 0, "model_accuracy": None,
            "notice": "Real HTTP/SQLite software contracts using an explicit unavailable-model fixture; not neural answer-quality evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
