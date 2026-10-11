"""Read-only restart verification against a successful isolated deployment smoke."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from verify_deployment import Client, SCOPE, await_readiness, loopback_target


PHASES = ("ready_after_restart", "authoritative_snapshot_preserved", "rollback_stays_restored")
SNAPSHOT_KEYS = ("revision", "facts", "edits", "events", "dataset")


def snapshot(body):
    return {key: body[key] for key in SNAPSHOT_KEYS}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh report path.")
    started = perf_counter()
    report = {"schema_version": 1, "purpose": "structured_restart_persistence_check",
              "started_at": datetime.now(timezone.utc).isoformat(), "ml_results": None,
              "cases": [{"phase": phase, "passed": False, "status": "not_run"} for phase in PHASES],
              "disclaimer": "Read-only engineering check; restart must be performed separately."}
    client = None
    try:
        host, port, origin = loopback_target(args.base_url)
        reference_bytes = args.reference.read_bytes()
        reference = json.loads(reference_bytes)
        if (reference.get("purpose") != "structured_deployment_engineering_smoke"
                or reference.get("status") != "passed" or reference.get("base_url") != origin):
            raise ValueError("Reference must be a successful deployment smoke on the same loopback origin.")
        exports = [record["body"] for record in reference["requests"]
                   if record["method"] == "GET" and record["path"] == "/api/export"
                   and record["http_status"] == 200 and not record["error"]]
        if len(exports) != 1 or len(exports[0]["edits"]) != 2 or any(
                edit["status"] != "withdrawn" for edit in exports[0]["edits"]):
            raise ValueError("Reference must contain exactly one export with both smoke edits withdrawn.")
        expected = snapshot(exports[0])
        report.update(base_url=origin, reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                      before_snapshot_sha256=digest(expected), before_revision=expected["revision"])
        client = Client(host, port, "", 5)
        report["readiness"] = await_readiness(client, 30)
        ready = report["readiness"]["status"] == "passed"
        report["cases"][0].update(passed=ready, status="passed" if ready else "failed")
        if ready:
            exported = client.request("GET", "/api/export")
            preserved = False
            if not exported["error"] and exported["http_status"] == 200:
                actual = snapshot(exported["body"])
                preserved = actual == expected
                report.update(after_snapshot_sha256=digest(actual), after_revision=actual["revision"])
            report["cases"][1].update(passed=preserved, status="passed" if preserved else "failed")
            answer = client.request("POST", "/api/ask", {**SCOPE, "question": "How many devices does Basic support?"})
            body = answer.get("body") or {}
            restored = (not answer["error"] and answer["http_status"] == 200
                        and body.get("status") == "answered"
                        and body.get("answer_kind") == "deterministic_structured"
                        and body.get("route") == "initial_catalogue"
                        and body.get("answer") == "Nimbus Basic v2 supports up to 5 devices.")
            report["cases"][2].update(passed=restored, status="passed" if restored else "failed")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    passed = sum(case["passed"] for case in report["cases"])
    report.update(status="passed" if passed == len(PHASES) else "failed",
                  counts={"passed": passed, "total": len(PHASES)},
                  requests=client.requests if client else [], mutation_attempted=False,
                  completed_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=round(perf_counter() - started, 3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
