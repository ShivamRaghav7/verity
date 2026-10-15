"""Finite HTTP engineering measurement on a new isolated approved-edit fixture.

No benchmark question comes from the held-out set. This is latency/availability
evidence, not model accuracy. The first request includes lazy model loading.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import secrets
import re
import socket
import subprocess
import sys
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_http import percentile

QUESTIONS = ("How many devices does Nimbus Basic version v2 support?",
             "What is the device allowance on Nimbus Cloud Basic version v2?",
             "Can I connect four devices on Nimbus Basic version v2?")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload(index):
    return {"question": QUESTIONS[index % len(QUESTIONS)], "entity_id": "nimbus", "plan": "Basic",
            "version": "v2", "as_of": "2026-10-13", "mode": "learned"}


def sample(url, index, phase, timeout=120):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise ValueError("Request timeout must be finite and within 0..120 seconds")
    record = {"phase": phase, "index": index, "payload": payload(index), "http_status": None,
              "status": None, "answer_kind": None, "revision": None, "error": None}
    started = perf_counter()
    try:
        request = Request(url + "/api/ask", json.dumps(record["payload"]).encode("utf-8"),
                          {"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=timeout) as response:
            record["http_status"] = response.status
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("Response exceeds finite size limit")
        body = json.loads(raw)
        if not isinstance(body, dict) or not isinstance(body.get("model_training"), dict):
            raise ValueError("Learned response must include object-valued model training provenance")
        for key in ("status", "answer_kind", "revision", "route", "answer", "api_latency_ms", "latency_ms"):
            record[key] = body.get(key)
        record["model_manifests"] = {task: info.get("artifact_manifest_sha256")
                                     for task, info in body["model_training"].items() if isinstance(info, dict)}
        if (record["status"] not in ("answered", "clarification", "conflict")
                or type(record["revision"]) is not int
                or record["answer_kind"] not in ("model_generated", "deterministic_clarification")):
            raise ValueError("Unexpected learned response contract")
        if not valid_model_manifests(record["model_manifests"]):
            raise ValueError("Learned response lacks both validated auxiliary manifest identities")
    except HTTPError as exc:
        try:
            detail = exc.read(500).decode("utf-8", errors="replace")
        except (OSError, HTTPException) as read_error:
            detail = "Unable to read error detail: " + str(read_error)
        record.update(http_status=exc.code, error="HTTP " + str(exc.code) + ": " + detail)
        exc.close()
    except (URLError, OSError, HTTPException, ValueError, TypeError) as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)
    record["http_roundtrip_ms"] = (perf_counter() - started) * 1000
    return record


def valid_model_manifests(value):
    return isinstance(value, dict) and set(value) == {"scope", "answer"} and all(
        re.fullmatch(r"[0-9a-f]{64}", str(item)) is not None for item in value.values())


def bind_sample(record, expected_revision, expected_models):
    if record["error"] is None and (record.get("revision") != expected_revision or record.get("model_manifests") != expected_models):
        record["error"] = "Response revision or trained manifest identities differ from the benchmark fixture/configuration"
    return record


def summarize(rows):
    valid = [row for row in rows if row["http_status"] == 200 and row["error"] is None
             and type(row.get("revision")) is int and valid_model_manifests(row.get("model_manifests"))]
    timings = [row["http_roundtrip_ms"] for row in rows]
    revisions = sorted({row["revision"] for row in valid})
    snapshots = {json.dumps(row["model_manifests"], sort_keys=True) for row in valid}
    stable = len(revisions) == 1 and len(snapshots) == 1
    return {"status": "completed" if len(valid) == len(rows) and stable else "incomplete",
            "counts": {"attempted": len(rows), "valid_responses": len(valid), "errors": len(rows) - len(valid),
                       "answered": sum(row["status"] == "answered" for row in valid),
                       "abstentions": sum(row["status"] != "answered" for row in valid)},
            "status_counts": dict(Counter(row["status"] for row in valid)),
            "answer_kind_counts": dict(Counter(row["answer_kind"] for row in valid)),
            "observed_revisions": revisions, "stable_observed_model_and_revision": stable,
            "latency": {"denominator": len(rows), "p50_ms": percentile(timings, .5), "p95_ms": percentile(timings, .95),
                        "includes_errors": True, "valid_denominator": len(valid),
                        "valid_response_p50_ms": percentile([row["http_roundtrip_ms"] for row in valid], .5),
                        "valid_response_p95_ms": percentile([row["http_roundtrip_ms"] for row in valid], .95)}}


def run(config, output, *, samples=30, warmups=3):
    from editguard.store import Store
    config, output = config.resolve(), output.resolve()
    if not 1 <= samples <= 60 or not 0 <= warmups <= 5 or output.exists():
        raise ValueError("Use a fresh output and a finite 1..60 sample / 0..5 warmup plan")
    if ROOT not in config.parents or ROOT not in output.parents:
        raise ValueError("All benchmark paths must remain within the workspace")
    watched = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
               for path in (config, ROOT / "editguard/api.py", Path(__file__))}
    settings = json.loads(config.read_text(encoding="utf-8"))
    expected_models = {}
    for task in ("scope", "answer"):
        path = ROOT / settings[task + "_artifact"] / "manifest.json"
        watched[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
        expected_models[task] = sha(path)
    identifier = uuid4().hex[:12]
    fixture = ROOT / "runtime" / ("http-neural-" + identifier + ".sqlite3")
    store = Store(fixture)
    store.seed()
    edit = store.create_correction({"fact_id": "nimbus-basic-v2-devices", "original_value": "5", "corrected_value": "3",
        "source": "controlled://http-latency-fixture", "reason": "Isolated latency fixture, not a scored test example.",
        "effective_from": "2026-10-12"})
    store.approve(edit["edit_id"])
    expected_revision = store.revision()
    fixture_before = {"revision": expected_revision, "facts": store.facts(), "edits": store.edits(), "events": store.events()}
    store.close()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = "http://127.0.0.1:" + str(port)
    env = {**os.environ, "EDITGUARD_DB": str(fixture), "EDITGUARD_ADMIN_TOKEN": secrets.token_urlsafe(24),
           "EDITGUARD_INFERENCE_CONFIG": str(config), "HF_HOME": str(ROOT / "runtime/hf-cache"), "HF_HUB_OFFLINE": "1"}
    log_path = ROOT / "runtime" / ("http-neural-" + identifier + ".log")
    started = datetime.now(timezone.utc).isoformat()
    cold, warmup_rows, rows, failure, process = None, [], [], None, None
    startup_started, startup_attempts = perf_counter(), 0
    startup_elapsed_ms = None
    with log_path.open("wb") as log:
        try:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "editguard.api:create_app", "--factory",
                "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, env=env, stdout=log, stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            ready = False
            for _ in range(160):
                startup_attempts += 1
                if process.poll() is not None:
                    raise RuntimeError("Isolated server exited; inspect " + str(log_path))
                try:
                    with urlopen(url + "/api/health", timeout=.5) as response:
                        ready = response.status == 200
                    if ready:
                        break
                except (URLError, OSError):
                    sleep(.25)
            if not ready:
                raise RuntimeError("Isolated server startup exceeded its finite readiness plan")
            startup_elapsed_ms = (perf_counter() - startup_started) * 1000
            cold = bind_sample(sample(url, 0, "cold_lazy_model_loading"), expected_revision, expected_models)
            if cold["error"]:
                raise RuntimeError("Cold inference failed: " + cold["error"])
            for phase, count, collection in (("warmup", warmups, warmup_rows), ("measured", samples, rows)):
                for index in range(count):
                    record = bind_sample(sample(url, index, phase), expected_revision, expected_models)
                    collection.append(record)
                    if record["error"] and "cuda" in record["error"].lower():
                        raise RuntimeError("CUDA failure; remaining benchmark requests were not attempted")
        except (RuntimeError, OSError) as exc:
            failure = {"type": type(exc).__name__, "reason": str(exc)}
        finally:
            if startup_elapsed_ms is None:
                startup_elapsed_ms = (perf_counter() - startup_started) * 1000
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    snapshot_unchanged = watched == {name: sha(ROOT / name) for name in watched}
    store = Store(fixture)
    fixture_after = {"revision": store.revision(), "facts": store.facts(), "edits": store.edits(), "events": store.events()}
    store.close()
    report = {"schema_version": 1, "purpose": "isolated_neural_http_latency_v1", "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(), "source_sha256": watched,
        "samples": rows, "warmup_samples": warmup_rows, "cold_sample": cold,
        "fixture_database": str(fixture), "fixture_expected_revision": expected_revision,
        "expected_model_manifests": expected_models, "source_snapshot_unchanged": snapshot_unchanged,
        "isolated_fixture_domain_state_unchanged": fixture_before == fixture_after,
        "server_log": {"path": str(log_path), "sha256": sha(log_path)}, "execution_failure": failure,
        "startup": {"readiness_attempts": startup_attempts, "maximum_readiness_attempts": 160,
                    "elapsed_ms": startup_elapsed_ms},
        "configuration": {"samples": samples, "warmups": warmups, "concurrency": 1, "mode": "learned",
                          "new_http_connection_per_request": True, "maximum_question_requests": 1 + warmups + samples},
        "ml_accuracy": None, "live_application_mutated": False, **summarize(rows),
        "phase_counts": {"cold_attempted": int(cold is not None), "cold_errors": int(cold is not None and cold["error"] is not None),
            "warmup_attempted": len(warmup_rows), "warmup_errors": sum(row["error"] is not None for row in warmup_rows),
            "warmup_not_attempted": warmups - len(warmup_rows), "measured_not_attempted": samples - len(rows)},
        "limitations": ["Small repeated demonstration-question engineering workload, not the held-out accuracy population or a load test.",
            "HTTP client timings include transport, JSON decoding, retrieval, learned routing/generation, and API before-answer generation.",
            "Cold checkpoint loading is separate; warmups are excluded from measured percentiles. Reused model/retrieval caches affect warm timings.",
            "All attempted measured requests, including failures and clarifications, remain in latency and availability denominators; requests not attempted have no invented latency and are counted separately.",
            "Startup and cold inference failures stop later requests and retain a diagnostic report. A new CUDA failure stops the finite workload to avoid an execution-error cascade."]}
    if (failure or len(rows) != samples or not snapshot_unchanged or fixture_before != fixture_after
            or report["observed_revisions"] != [expected_revision] or any(row["error"] for row in warmup_rows)):
        report["status"] = "incomplete"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/inference-final.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/final/neural-http.json")
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({key: result[key] for key in ("status", "counts", "latency")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
