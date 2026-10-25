"""Seven finite, real ReasonKE generations through the website API.

This is a controlled development acceptance check, not a general accuracy
benchmark. The Mumbai correction is explicitly counterfactual. No training,
retry loop, simulated answer, or modification of another user's facts occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

MODEL_ID = "YukinoKaren/ReasonKE_plus"
MODEL_REVISION = "ceaedc2ae7c8fed6257953587d991ce550db6106"
QUESTION = "What is the capital of India?"
PARAPHRASE = "Which city serves as India's capital?"
ORDINARY = "Explain overfitting in two short sentences."
CORRECTION = {
    "question": QUESTION,
    "answer": "Mumbai",
    "scope": "India; capital; controlled counterfactual demonstration",
    "source": "Controlled counterfactual demonstration. This record defines the approved answer within the demonstration.",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".reasonke-live-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RequestFailure(Exception):
    def __init__(self, message, *, status=None, body=None):
        super().__init__(message)
        self.status, self.body = status, body


class Client:
    def __init__(self, base_url):
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
                or parsed.username or parsed.password or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment):
            raise ValueError("Run inside the serving environment against its loopback HTTP website origin.")
        self.base = base_url.rstrip("/")
        self.token = None

    def request(self, path, payload=None):
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            if not self.token:
                raise RequestFailure("No website session token has been obtained")
            headers["X-EditGuard-Token"] = self.token
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            with exc:
                raw = exc.read(65536)
                try:
                    body = json.loads(raw)
                except (ValueError, UnicodeError):
                    body = {"unparsed_error": raw.decode("utf-8", errors="replace")}
                raise RequestFailure("Website rejected the request", status=exc.code, body=body) from exc
        except OSError as exc:
            raise RequestFailure("Website request failed: " + str(exc)) from exc
        if len(raw) > 2 * 1024 * 1024:
            raise RequestFailure("Website response exceeded the 2 MiB evidence budget")
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise RequestFailure("Website did not return valid JSON") from exc
        if not isinstance(result, dict):
            raise RequestFailure("Website response must be a JSON object")
        return result


def case_plan():
    return [
        {"id": "original", "question": QUESTION, "expected": "New Delhi"},
        {"id": "corrected-direct", "question": QUESTION, "expected": "Mumbai"},
        {"id": "fresh-paraphrase", "question": PARAPHRASE, "expected": "Mumbai"},
        {"id": "compare-without-context", "question": PARAPHRASE, "expected": "New Delhi"},
        {"id": "compare-with-context", "question": PARAPHRASE, "expected": "Mumbai"},
        {"id": "after-withdrawal", "question": PARAPHRASE, "expected": "New Delhi"},
        {"id": "ordinary-conversation", "question": ORDINARY, "expected": None,
         "scoring": "Nonempty real model response only; factual quality requires qualitative review"},
    ]


def run(base_url, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Receipt already exists; choose a new output to preserve the previous evidence.")
    client = Client(base_url)
    cases = [{**item, "status": "unexecuted", "passed": False} for item in case_plan()]
    receipt = {
        "schema": "editguard-reasonke-live-v1", "started_utc": utc_now(), "status": "running",
        "base_url": client.base, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "planned_model_generations": 7, "cases": cases, "operations": [], "owned_chat_ids": [],
        "owned_correction_id": None, "cleanup": {"needed": False, "attempted": False},
        "disclosures": [
            "Controlled development acceptance; not a held-out benchmark or general accuracy estimate",
            "Author-trained checkpoint; its exact association with the published 95.48% score is unverified",
            "Mumbai is a synthetic counterfactual target, not the real-world capital",
            "Both comparison sides use the same trained checkpoint; only correction context changes",
            "Ordinary conversation is checked only for nonempty generation, not factual correctness",
            "All assigned cases, including failures and unexecuted cases, remain in this receipt",
            "Website CSRF token and request authorization headers are never written to this receipt",
            "Memory-inspection responses retain counts and this run's records; unrelated records are omitted",
        ],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    by_id = {row["id"]: row for row in cases}
    owned_correction = None
    withdrawn = False
    started = time.perf_counter()
    atomic_write(output, receipt)

    def operation(name, path, payload=None):
        record = {"name": name, "path": path, "started_utc": utc_now(), "status": "running"}
        if payload is not None:
            record["request"] = payload
        receipt["operations"].append(record)
        atomic_write(output, receipt)
        began = time.perf_counter()
        try:
            response = client.request(path, payload)
            # The /status response contains a CSRF token. Keep it private.
            recorded = {key: value for key, value in response.items() if key != "token"}
            if path == "/api/corrections" and payload is None:
                recorded = {
                    "revision": response.get("revision"),
                    "active_correction_count": sum(row.get("status") == "active" for row in response.get("corrections", [])),
                    "corrections": [row for row in response.get("corrections", []) if row.get("id") == owned_correction],
                    "events": [row for row in response.get("events", []) if row.get("correction_id") == owned_correction],
                    "unrelated_records_omitted": True,
                }
            record.update(status="completed", response=recorded)
            return response
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, RequestFailure):
                error.update(http_status=exc.status, response=exc.body)
            record.update(status="error", error=error)
            raise
        finally:
            record.update(seconds=time.perf_counter() - began, completed_utc=utc_now())
            atomic_write(output, receipt)

    def score(case_id, answer):
        row = by_id[case_id]
        text = answer.get("content")
        evidence = answer.get("evidence", {})
        generation = evidence.get("generation", {}) if isinstance(evidence, dict) else {}
        provenance = (isinstance(generation, dict) and generation.get("model") == MODEL_ID
                      and generation.get("revision") == MODEL_REVISION
                      and evidence.get("generation_kind") == "model")
        nonempty = isinstance(text, str) and bool(text.strip())
        expected = row["expected"]
        exact = None if expected is None else nonempty and text.strip() == expected
        row.update(status="generated", actual=text, response=answer, valid_model_provenance=provenance,
                   exact_answer_match=exact, nonempty_response=nonempty,
                   passed=bool(provenance and nonempty and (expected is None or exact)),
                   factual_quality_evaluated=expected is not None)
        atomic_write(output, receipt)
        print(json.dumps({"case": case_id, "passed": row["passed"],
                          "exact_answer_match": exact, "factual_quality_evaluated": expected is not None}), flush=True)

    def create_chat(name):
        response = operation(name, "/api/chats", {})
        chat_id = response["chat"]["id"]
        if not isinstance(chat_id, str) or not chat_id:
            raise RequestFailure("Website did not return a conversation identifier")
        receipt["owned_chat_ids"].append(chat_id)
        atomic_write(output, receipt)
        return chat_id

    try:
        status = operation("inspect-model", "/api/status")
        client.token = status.get("token")
        model = status.get("model", {})
        if (not isinstance(client.token, str) or not client.token or model.get("ready") is not True
                or model.get("name") != MODEL_ID or model.get("revision") != MODEL_REVISION):
            raise RequestFailure("The website must report the pinned, ready ReasonKE model")
        memory = operation("check-empty-approved-memory", "/api/corrections")
        if any(row.get("status") == "active" for row in memory.get("corrections", [])):
            raise RequestFailure("Existing approved facts are active; use an isolated storage directory. Nothing was cleared.")
        first_chat = create_chat("create-original-conversation")
        response = operation("original-answer", "/api/chat", {"chat_id": first_chat, "message": QUESTION, "mode": "original"})
        score("original", response["assistant"])
        published = operation("approve-owned-counterfactual", "/api/corrections", CORRECTION)
        owned_correction = published["correction"]["id"]
        receipt["owned_correction_id"] = owned_correction
        receipt["cleanup"]["needed"] = True
        atomic_write(output, receipt)
        response = operation("corrected-direct-answer", "/api/chat", {"chat_id": first_chat, "message": QUESTION, "mode": "edited"})
        score("corrected-direct", response["assistant"])
        second_chat = create_chat("create-fresh-paraphrase-conversation")
        response = operation("independent-paraphrase-answer", "/api/chat", {"chat_id": second_chat, "message": PARAPHRASE, "mode": "edited"})
        score("fresh-paraphrase", response["assistant"])
        compared = operation("matched-fresh-comparison", "/api/compare", {"chat_id": second_chat, "message_id": response["user"]["id"]})
        score("compare-without-context", compared["original"])
        score("compare-with-context", compared["edited"])
        result = operation("withdraw-owned-counterfactual", f"/api/corrections/{owned_correction}/withdraw", {})
        withdrawn = result.get("correction", {}).get("status") == "withdrawn"
        receipt["cleanup"].update(needed=not withdrawn, withdrawal_response_confirmed=withdrawn)
        atomic_write(output, receipt)
        if not withdrawn:
            raise RequestFailure("The website did not confirm withdrawal of the owned correction")
        response = operation("restored-answer", "/api/chat", {"chat_id": second_chat, "message": PARAPHRASE, "mode": "edited"})
        score("after-withdrawal", response["assistant"])
        response = operation("ordinary-conversation-answer", "/api/chat", {"chat_id": second_chat, "message": ORDINARY, "mode": "edited"})
        score("ordinary-conversation", response["assistant"])
        operation("final-owned-conversation-one", "/api/chats/" + first_chat)
        operation("final-owned-conversation-two", "/api/chats/" + second_chat)
        final_memory = operation("confirm-owned-fact-is-inactive", "/api/corrections")
        owned_rows = [row for row in final_memory.get("corrections", []) if row.get("id") == owned_correction]
        receipt["owned_fact_inactive_at_end"] = bool(owned_rows and all(row.get("status") != "active" for row in owned_rows))
        if not receipt["owned_fact_inactive_at_end"]:
            withdrawn = False
            raise RequestFailure("The owned correction did not remain inactive at the end")
        receipt["status"] = "completed"
    except Exception as exc:
        receipt.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        if owned_correction and not withdrawn:
            receipt["cleanup"]["attempted"] = True
            try:
                result = operation("cleanup-withdraw-owned-correction", f"/api/corrections/{owned_correction}/withdraw", {})
                receipt["cleanup"].update(succeeded=result.get("correction", {}).get("status") == "withdrawn")
            except Exception as exc:
                receipt["cleanup"].update(succeeded=False, error={"type": type(exc).__name__, "message": str(exc)})
        receipt.update(completed_utc=utc_now(), total_seconds=time.perf_counter() - started,
                       generated_count=sum(row["status"] == "generated" for row in cases),
                       unexecuted_count=sum(row["status"] == "unexecuted" for row in cases),
                       matched_factual_cases=sum(row["passed"] for row in cases if row["expected"] is not None),
                       assigned_factual_cases=6,
                       all_assigned_checks_passed=all(row["passed"] for row in cases))
        atomic_write(output, receipt)
    print(json.dumps({"receipt": str(output.resolve()), "status": receipt["status"],
                      "generated": receipt["generated_count"], "unexecuted": receipt["unexecuted_count"],
                      "all_assigned_checks_passed": receipt["all_assigned_checks_passed"],
                      "notice": "Development acceptance only; not a general accuracy estimate"}), flush=True)
    return 0 if receipt["status"] == "completed" and receipt["all_assigned_checks_passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.base_url, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, "Cannot start live acceptance: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
