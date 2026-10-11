"""Finite structured HTTP lifecycle smoke for a fresh isolated loopback instance."""

from __future__ import annotations

import argparse
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
from time import perf_counter, sleep
from urllib.parse import quote, urlsplit

PHASES = ("fresh_instance", "initial_answer", "create_draft", "draft_has_no_authority", "approve_correction",
          "corrected_direct_and_source", "corrected_paraphrase", "capacity_question", "price_retention",
          "pro_retention", "older_version_retention", "similar_product_retention", "historical_date_retention",
          "create_contradictory_draft", "reject_conflicting_approval", "conflict_blocks_answer",
          "withdraw_conflicting_draft", "resolution_restores_correction", "rollback_correction",
          "rollback_restores_initial", "history_and_provenance", "conflicts_cleared", "export")
MAX_REQUESTS = 32
MAX_STARTUP_ATTEMPTS = 30
MAX_RESPONSE_BYTES = 1_048_576
SOURCE = "controlled://deployment-smoke/device-limit"
CONFLICT_SOURCE = "controlled://deployment-smoke/contradictory-draft"
SCOPE = {"entity_id": "nimbus", "plan": "Basic", "version": "v2", "as_of": "2026-10-11", "mode": "structured"}


def loopback_target(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("Use an HTTP loopback origin with no credentials, path, query or fragment.")
    hostname = parsed.hostname
    if hostname == "localhost":
        hostname = "127.0.0.1"  # Avoid DNS resolution or proxy routing.
    try:
        if not hostname or not ipaddress.ip_address(hostname).is_loopback:
            raise ValueError("Deployment verification refuses a non-loopback target.")
        port = parsed.port or 80
    except ValueError as error:
        raise ValueError("Deployment verification requires a literal loopback address or localhost and valid port.") from error
    display = f"[{hostname}]" if ":" in hostname else hostname
    return hostname, port, f"http://{display}:{port}"


def redact(value, token):
    if isinstance(value, str):
        return value.replace(token, "[REDACTED]") if token else value
    if isinstance(value, dict):
        return {redact(key, token): redact(item, token) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, token) for item in value]
    return value


class Client:
    def __init__(self, host, port, token, timeout):
        self.host, self.port, self.token, self.timeout = host, port, token, timeout
        self.requests = []
        self.workflow_requests, self.startup_requests = 0, 0

    def request(self, method, path, payload=None, *, administrator=False, startup=False, timeout=None):
        if (self.startup_requests >= MAX_STARTUP_ATTEMPTS if startup else self.workflow_requests >= MAX_REQUESTS):
            raise RuntimeError("Finite HTTP request cap reached.")
        if startup:
            self.startup_requests += 1
        else:
            self.workflow_requests += 1
        request_timeout = self.timeout if timeout is None else min(self.timeout, timeout)
        started = perf_counter()
        record = {"phase": "startup_readiness" if startup else "workflow", "method": method, "path": path,
                  "http_status": None, "response_status": None, "request_timeout_seconds": request_timeout,
                  "latency_ms": None, "body": None, "error": None}
        connection = http.client.HTTPConnection(self.host, self.port, timeout=request_timeout)
        try:
            headers = {"Accept": "application/json"}
            data = json.dumps(payload).encode("utf-8") if payload is not None else None
            if data is not None:
                headers["Content-Type"] = "application/json"
            if administrator:
                headers["X-Admin-Token"] = self.token
            # HTTPConnection bypasses environment proxies and never follows redirects.
            connection.request(method, path, body=data, headers=headers)
            socket = connection.sock
            remaining = request_timeout - (perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("Request deadline expired before response headers.")
            socket.settimeout(remaining)
            response = connection.getresponse()
            record["http_status"] = response.status
            chunks, size = [], 0
            while True:
                remaining = request_timeout - (perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("Request deadline expired while reading the response.")
                socket.settimeout(remaining)
                chunk = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise ValueError("Response exceeds the 1 MiB verification limit.")
            body = json.loads(b"".join(chunks).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("API response must be a JSON object.")
            record["body"] = body
            record["response_status"] = body.get("status")
            if 300 <= response.status < 400:
                raise ValueError("Redirect refused; verification stays on the explicit loopback origin.")
        except (OSError, http.client.HTTPException, ValueError) as error:
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            connection.close()
            record["latency_ms"] = round((perf_counter() - started) * 1000, 3)
            self.requests.append(redact(record, self.token))
        return record


def await_readiness(client, timeout):
    """Only startup health polling retries; workflow failures are never retried."""
    started, attempts = perf_counter(), 0
    ready = False
    while attempts < MAX_STARTUP_ATTEMPTS:
        remaining = timeout - (perf_counter() - started)
        if remaining <= 0:
            break
        response = client.request("GET", "/api/health", startup=True, timeout=remaining)
        attempts += 1
        if not response["error"] and response["http_status"] == 200 and (response.get("body") or {}).get("status") == "ok":
            ready = True
            break
        remaining = timeout - (perf_counter() - started)
        if remaining > 0 and attempts < MAX_STARTUP_ATTEMPTS:
            sleep(min(1, remaining))
    return {"status": "passed" if ready else "failed", "attempts": attempts,
            "max_attempts": MAX_STARTUP_ATTEMPTS, "timeout_seconds": timeout,
            "elapsed_seconds": round(perf_counter() - started, 3),
            "reason": None if ready else "Isolated server was not ready within the finite startup window."}


def verify(client):
    cases, state = [], {}

    def phase(name, operation, assertion, *, dependency=True):
        if not dependency:
            case = {"phase": name, "status": "not_run", "passed": False,
                    "reason": "Required earlier state was not established.", "http_status": None}
            cases.append(case)
            return None
        started = perf_counter()
        response = None
        try:
            response = operation()
            if response.get("error"):
                raise ValueError(response["error"])
            if not assertion(response):
                raise ValueError("Response did not satisfy the declared engineering check.")
            passed, reason = True, None
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
            passed, reason = False, f"{type(error).__name__}: {error}"
        cases.append({"phase": name, "status": "passed" if passed else "failed", "passed": passed,
                      "reason": reason, "http_status": response.get("http_status") if response else None,
                      "response_status": response.get("response_status") if response else None,
                      "latency_ms": round((perf_counter() - started) * 1000, 3)})
        return response

    def ask(question="How many devices does Nimbus Basic v2 support?", **scope):
        return client.request("POST", "/api/ask", {**SCOPE, **scope, "question": question})

    def answered(response, text, route=None):
        body = response.get("body") or {}
        return (response["http_status"] == 200 and body.get("status") == "answered"
                and body.get("answer_kind") == "deterministic_structured" and body.get("answer") == text
                and (route is None or body.get("route") == route))

    def fresh():
        health = client.request("GET", "/api/health")
        history = client.request("GET", "/api/edits")
        state["initial_health"], state["initial_history"] = health, history
        if health.get("error") or history.get("error"):
            return {"error": "Fresh-instance health/history could not be read.", "http_status": None}
        return {"http_status": 200 if health["http_status"] == history["http_status"] == 200 else None,
                "body": {"health": health["body"], "history": history["body"]}}

    phase("fresh_instance", fresh, lambda r: r["http_status"] == 200
          and r["body"]["health"].get("status") == "ok"
          and type(r["body"]["health"].get("revision")) is int and r["body"]["health"]["revision"] == 1
          and r["body"]["history"].get("revision") == 1 and r["body"]["history"].get("edits") == []
          and all(event.get("event_type") == "catalogue_seeded" for event in r["body"]["history"].get("events", [])))
    if not cases[-1]["passed"]:
        cases.extend({"phase": name, "status": "not_run", "passed": False,
                      "reason": "Fresh memory precondition failed; no mutation attempted.", "http_status": None} for name in PHASES[1:])
        return cases, {"refused": True, "mutation_attempted": False}
    phase("initial_answer", ask, lambda r: answered(r, "Nimbus Basic v2 supports up to 5 devices.", "initial_catalogue"))
    if not cases[-1]["passed"]:
        cases.extend({"phase": name, "status": "not_run", "passed": False,
                      "reason": "Expected controlled initial fact failed; no mutation attempted.", "http_status": None} for name in PHASES[2:])
        return cases, {"refused": True, "mutation_attempted": False}
    correction = {"fact_id": "nimbus-basic-v2-devices", "original_value": "5", "corrected_value": "3",
                  "source": SOURCE, "reason": "Controlled isolated deployment verification", "effective_from": "2026-10-01"}
    draft = phase("create_draft", lambda: client.request("POST", "/api/edits", correction, administrator=True),
                  lambda r: r["http_status"] == 201 and r["body"].get("status") == "draft"
                  and r["body"].get("source") == SOURCE and r["body"].get("corrected_value") == "3")
    edit_id = (draft.get("body") or {}).get("edit_id") if draft else None
    usable_edit = isinstance(edit_id, str) and bool(edit_id)
    edit_path = f"/api/edits/{quote(edit_id, safe='')}" if usable_edit else ""
    phase("draft_has_no_authority", ask, lambda r: answered(r, "Nimbus Basic v2 supports up to 5 devices.", "initial_catalogue"), dependency=usable_edit)
    approved_response = phase("approve_correction", lambda: client.request("POST", edit_path + "/approve", administrator=True),
                              lambda r: r["http_status"] == 200 and r["body"].get("status") == "approved", dependency=usable_edit)
    approved = bool(approved_response and approved_response["http_status"] == 200
                    and (approved_response.get("body") or {}).get("status") == "approved")
    phase("corrected_direct_and_source", ask,
          lambda r: answered(r, "Nimbus Basic v2 supports up to 3 devices.", "approved_correction")
          and r["body"].get("before_answer") == "Nimbus Basic v2 supports up to 5 devices."
          and any(edit.get("edit_id") == edit_id and edit.get("source") == SOURCE and edit.get("status") == "approved"
                  for edit in r["body"].get("applicable_corrections", [])), dependency=approved)
    phase("corrected_paraphrase", lambda: ask("What is the device limit for Nimbus Basic v2?"),
          lambda r: answered(r, "Nimbus Basic v2 supports up to 3 devices.", "approved_correction"), dependency=approved)
    phase("capacity_question", lambda: ask("Can I connect four devices on Nimbus Basic v2?"),
          lambda r: answered(r, "No. Nimbus Basic v2 supports up to 3 devices."), dependency=approved)
    neighbors = (("price_retention", "What is Nimbus Basic v2 monthly price?", {}, "Nimbus Basic v2 costs 9 USD/month."),
                 ("pro_retention", "How many devices does Nimbus Pro v2 support?", {"plan": "Pro"}, "Nimbus Pro v2 supports up to 12 devices."),
                 ("older_version_retention", "How many devices does Nimbus Basic v1 support?", {"version": "v1"}, "Nimbus Basic v1 supports up to 5 devices."),
                 ("similar_product_retention", "How many devices does Nimbus Mini Basic v2 support?", {"entity_id": "nimbus-mini"}, "Nimbus Mini Basic v2 supports up to 2 devices."),
                 ("historical_date_retention", "How many devices does Nimbus Basic v2 support?", {"as_of": "2026-09-01"}, "Nimbus Basic v2 supports up to 5 devices."))
    for name, question, scope, expected in neighbors:
        phase(name, lambda q=question, s=scope: ask(q, **s), lambda r, expected=expected: answered(r, expected, "initial_catalogue"), dependency=approved)
    conflicting = {**correction, "original_value": "3", "corrected_value": "4", "source": CONFLICT_SOURCE}
    competitor = phase("create_contradictory_draft", lambda: client.request("POST", "/api/edits", conflicting, administrator=True),
                       lambda r: r["http_status"] == 201 and r["body"].get("status") == "draft", dependency=approved)
    competitor_id = (competitor.get("body") or {}).get("edit_id") if competitor else None
    usable_competitor = isinstance(competitor_id, str) and bool(competitor_id) and competitor_id != edit_id
    competitor_path = f"/api/edits/{quote(competitor_id, safe='')}" if usable_competitor else ""
    phase("reject_conflicting_approval", lambda: client.request("POST", competitor_path + "/approve", administrator=True),
          lambda r: r["http_status"] == 409, dependency=usable_competitor)
    phase("conflict_blocks_answer", ask,
          lambda r: r["http_status"] == 200 and r["body"].get("status") == "conflict"
          and r["body"].get("route") == "conflict" and r["body"].get("answer_kind") == "deterministic_structured", dependency=usable_competitor)
    phase("withdraw_conflicting_draft", lambda: client.request("POST", competitor_path + "/withdraw", administrator=True),
          lambda r: r["http_status"] == 200 and r["body"].get("status") == "withdrawn", dependency=usable_competitor)
    phase("resolution_restores_correction", ask,
          lambda r: answered(r, "Nimbus Basic v2 supports up to 3 devices.", "approved_correction"), dependency=approved)
    # Attempt the known owned edit's withdrawal even if a later assertion failed.
    phase("rollback_correction", lambda: client.request("POST", edit_path + "/withdraw", administrator=True),
          lambda r: r["http_status"] == 200 and r["body"].get("status") == "withdrawn", dependency=usable_edit)
    phase("rollback_restores_initial", ask, lambda r: answered(r, "Nimbus Basic v2 supports up to 5 devices.", "initial_catalogue"), dependency=usable_edit)

    def history_matches(response):
        body = response.get("body") or {}
        edits = {edit["edit_id"]: edit for edit in body.get("edits", [])}
        if response["http_status"] != 200 or set(edits) != {edit_id, competitor_id}:
            return False
        if any(edit.get("status") != "withdrawn" for edit in edits.values()):
            return False
        if edits[edit_id].get("source") != SOURCE or edits[competitor_id].get("source") != CONFLICT_SOURCE:
            return False
        events = body.get("events", [])
        revisions = [event.get("revision") for event in events]
        if revisions != list(range(1, body.get("revision", 0) + 1)):
            return False
        expected_events = (("correction_created", edit_id), ("correction_approved", edit_id),
                           ("correction_created", competitor_id), ("approval_rejected", competitor_id),
                           ("correction_withdrawn", competitor_id), ("correction_withdrawn", edit_id))
        return all(any(event.get("event_type") == action and event.get("details", {}).get("edit_id") == identity
                       for event in events) for action, identity in expected_events)

    history = phase("history_and_provenance", lambda: client.request("GET", "/api/edits"), history_matches, dependency=usable_edit)
    phase("conflicts_cleared", lambda: client.request("GET", "/api/conflicts"),
          lambda r: r["http_status"] == 200 and r["body"].get("conflicts") == [], dependency=usable_edit)
    phase("export", lambda: client.request("GET", "/api/export"),
          lambda r: history_matches(r) and r["body"].get("dataset") == "controlled_fictional_demonstration"
          and r["body"].get("evaluation", {}).get("status") == "passed", dependency=usable_edit)
    return cases, {"refused": False, "mutation_attempted": True, "created_edit_id": edit_id,
                   "created_competitor_id": competitor_id, "final_revision": (history.get("body") or {}).get("revision") if history else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--admin-token-env", required=True, help="Environment-variable name containing the administrator token")
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSON report path; existing reports are never overwritten")
    parser.add_argument("--request-timeout-seconds", type=float, default=5)
    parser.add_argument("--startup-timeout", type=float, default=30,
                        help="Readiness-only health polling window, at most 30 seconds")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh report path; deployment evidence is not overwritten.")
    token, client, origin = "", None, None
    started = perf_counter()
    report = {"schema_version": 1, "purpose": "structured_deployment_engineering_smoke",
              "started_at": datetime.now(timezone.utc).isoformat(), "answer_kind": "deterministic_structured",
              "disclaimer": "Controlled HTTP workflow checks; no neural model or ML performance claim.",
              "fixed_phases": list(PHASES), "max_workflow_http_requests": MAX_REQUESTS,
              "max_startup_attempts": MAX_STARTUP_ATTEMPTS, "ml_results": None}
    try:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.admin_token_env):
            raise ValueError("Administrator token environment-variable name is invalid.")
        token = os.environ.get(args.admin_token_env, "")
        if not token:
            raise ValueError("The explicit administrator token environment variable is missing or empty.")
        if not math.isfinite(args.request_timeout_seconds) or not .1 <= args.request_timeout_seconds <= 30:
            raise ValueError("Request timeout must be finite between 0.1 and 30 seconds.")
        if not math.isfinite(args.startup_timeout) or not .1 <= args.startup_timeout <= 30:
            raise ValueError("Startup timeout must be finite between 0.1 and 30 seconds.")
        host, port, origin = loopback_target(args.base_url)
        client = Client(host, port, token, args.request_timeout_seconds)
        report["startup_readiness"] = await_readiness(client, args.startup_timeout)
        if report["startup_readiness"]["status"] != "passed":
            raise RuntimeError(report["startup_readiness"]["reason"])
        cases, state = verify(client)
        report.update(cases=cases, **state)
    except (ValueError, OSError, RuntimeError) as error:
        report.update(refused=True, mutation_attempted=bool(client and any(item["method"] == "POST" and item["path"].startswith("/api/edits") for item in client.requests)),
                      initialization_error=f"{type(error).__name__}: {error}",
                      cases=[{"phase": name, "status": "not_run", "passed": False,
                              "reason": "Initialization refused; no workflow result credited.", "http_status": None} for name in PHASES])
    passed = sum(case["passed"] for case in report["cases"])
    report.update(base_url=origin, request_timeout_seconds=args.request_timeout_seconds if math.isfinite(args.request_timeout_seconds) else None,
                  startup_timeout_seconds=args.startup_timeout if math.isfinite(args.startup_timeout) else None,
                  requests=client.requests if client else [], completed_at=datetime.now(timezone.utc).isoformat(),
                  request_counts={"startup": client.startup_requests if client else 0,
                                  "workflow": client.workflow_requests if client else 0},
                  elapsed_seconds=round(perf_counter() - started, 3),
                  counts={"passed": passed, "failed": sum(case["status"] == "failed" for case in report["cases"]),
                          "not_run": sum(case["status"] == "not_run" for case in report["cases"]), "total": len(PHASES)},
                  pass_rate={"numerator": passed, "denominator": len(PHASES), "value": passed / len(PHASES)},
                  status="refused" if report["refused"] else "passed" if passed == len(PHASES) else "failed")
    report = redact(report, token)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(redact({"status": report["status"], "counts": report["counts"], "output": str(args.output),
                             "mutation_attempted": report["mutation_attempted"]}, token)))
    return 0 if report["status"] == "passed" else 2 if report["status"] == "refused" else 1


if __name__ == "__main__":
    sys.exit(main())
