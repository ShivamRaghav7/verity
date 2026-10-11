"""Finite, sequential HTTP latency measurement for the structured vertical slice.

This sends read-only question requests. It measures client-observed elapsed time,
including HTTP transport, response reading and JSON decoding. It does not measure
learned-model accuracy, score factual answers, or mutate corrections.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.client import HTTPException
import json
import math
from pathlib import Path
from time import perf_counter_ns
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_KIND = "deterministic_structured"
QUESTIONS = (
    "How many devices does Nimbus Basic v2 support?",
    "What is the device limit for Nimbus Basic v2?",
    "Can I connect four devices on Nimbus Basic v2?",
)


def percentile(values, quantile):
    """Linear interpolation between observed sorted samples (inclusive ends)."""
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def payload_for(index):
    return {"question": QUESTIONS[index % len(QUESTIONS)], "entity_id": "nimbus",
            "plan": "Basic", "version": "v2", "as_of": "2026-10-11", "mode": "structured"}


def request_sample(endpoint, index, phase, timeout):
    payload = payload_for(index)
    record = {"phase": phase, "index": index, "question": payload["question"],
              "latency_ms": None, "http_status": None, "response_status": None,
              "answer_kind": None, "revision": None, "route": None, "error": None}
    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json", "Accept": "application/json"},
                      method="POST")
    started = perf_counter_ns()
    try:
        with urlopen(request, timeout=timeout) as response:
            record["http_status"] = response.status
            data = response.read(1_048_577)
            if len(data) > 1_048_576:
                raise ValueError("Response exceeds the benchmark's 1 MiB size limit")
            body = json.loads(data.decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("JSON response is not an object")
        for field in ("status", "answer_kind", "revision", "route"):
            record["response_status" if field == "status" else field] = body.get(field)
        if record["response_status"] not in ("answered", "clarification", "conflict"):
            raise ValueError("Response has an unsupported assistant status")
        if not isinstance(record["answer_kind"], str):
            raise ValueError("Response is missing answer_kind")
        if type(record["revision"]) is not int or record["revision"] < 0:
            raise ValueError("Response is missing a nonnegative integer revision")
    except HTTPError as exc:
        record["http_status"] = exc.code
        try:
            detail = exc.read(501).decode("utf-8", errors="replace")[:500]
        except (OSError, HTTPException) as read_error:
            detail = f"Unable to read error detail: {read_error}"
        record["error"] = f"HTTP {exc.code}: {detail}"
        exc.close()
    except (URLError, OSError, HTTPException, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["latency_ms"] = round((perf_counter_ns() - started) / 1_000_000, 6)
    return record


def summarise(records, warmup_records=()):
    """Retain failed requests and abstentions in explicit denominators."""
    latencies = [record["latency_ms"] for record in records]
    valid = [record for record in records if record["error"] is None and record["http_status"] == 200]
    valid_latencies = [record["latency_ms"] for record in valid]
    observed = [record for record in [*warmup_records, *records] if record["error"] is None]
    revisions = sorted({record["revision"] for record in observed})
    kinds = sorted({record["answer_kind"] for record in observed})
    mismatched = [kind for kind in kinds if kind != EXPECTED_KIND]
    mixed = len(revisions) > 1 or len(kinds) > 1 or bool(mismatched)
    errors = sum(record["error"] is not None for record in records)
    warnings = []
    if len(revisions) > 1:
        warnings.append("The authoritative revision changed during the run; results mix product states.")
    if mismatched or len(kinds) > 1:
        warnings.append("Response answer kinds did not remain deterministic_structured; results mix or mismatch answer modes.")
    if errors:
        warnings.append("Failed requests remain in attempted-roundtrip percentiles; valid-response percentiles are reported separately.")
    if not valid:
        warnings.append("No valid measured responses; latency is not evidence of working inference.")
    warmup_errors = sum(record["error"] is not None for record in warmup_records)
    if warmup_errors:
        warnings.append("Warmup requests had errors; inspect warmup_samples before interpreting the run.")
    counts = {
        "attempted": len(records), "http_200": sum(record["http_status"] == 200 for record in records),
        "valid_responses": len(valid), "answered": sum(record["response_status"] == "answered" for record in valid),
        "clarifications": sum(record["response_status"] == "clarification" for record in valid),
        "conflicts": sum(record["response_status"] == "conflict" for record in valid),
        "abstentions": sum(record["response_status"] in ("clarification", "conflict") for record in valid),
        "errors": errors,
    }
    return {
        "status": "mixed_state" if mixed else ("incomplete" if errors else "completed"),
        "mixed_state": mixed,
        "state_consistency": "mixed" if mixed else ("stable_observed" if observed else "unknown"),
        "observed_revisions": revisions, "observed_answer_kinds": kinds,
        "stable_revision": revisions[0] if len(revisions) == 1 else None,
        "answer_kind": kinds[0] if len(kinds) == 1 else None,
        "counts": counts, "warmup_counts": {"attempted": len(warmup_records), "errors": warmup_errors},
        "warnings": warnings,
        "metrics": {
            "http_roundtrip_p50_ms": percentile(latencies, .5),
            "http_roundtrip_p95_ms": percentile(latencies, .95),
            "valid_response_roundtrip_p50_ms": percentile(valid_latencies, .5),
            "valid_response_roundtrip_p95_ms": percentile(valid_latencies, .95),
            "latency_denominator": len(latencies), "valid_latency_denominator": len(valid_latencies),
        },
        "latencies_ms": latencies, "ml_accuracy": None,
    }


def run_benchmark(base_url, samples=30, warmup=3, timeout=5.0):
    if not 1 <= samples <= 200:
        raise ValueError("samples must be between 1 and 200")
    if not 0 <= warmup <= 20:
        raise ValueError("warmup must be between 0 and 20")
    if not 0 < timeout <= 30:
        raise ValueError("timeout must be greater than 0 and at most 30 seconds")
    parsed = urlsplit(base_url)
    if (parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment
            or parsed.username is not None or parsed.password is not None):
        raise ValueError("url must be an HTTP(S) base URL without credentials, query or fragment")
    endpoint = base_url.rstrip("/") + "/api/ask"
    started_at = datetime.now(timezone.utc).isoformat()
    warmup_records = [request_sample(endpoint, index, "warmup", timeout) for index in range(warmup)]
    records = [request_sample(endpoint, index, "measurement", timeout) for index in range(samples)]
    return {
        "schema_version": 1, "benchmark": "structured_http_latency_v1",
        "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Engineering measurement of the deterministic structured slice. No trained-model accuracy is scored. Sequential client timings include HTTP transport and JSON decoding; warmups are excluded from measured percentiles.",
        "configuration": {"url": base_url, "endpoint": endpoint, "samples": samples,
                          "warmup": warmup, "timeout_seconds": timeout, "concurrency": 1,
                          "question_payloads": [payload_for(index) for index in range(len(QUESTIONS))],
                          "expected_answer_kind": EXPECTED_KIND,
                          "percentile_method": "linear_interpolation_observed_order_statistics",
                          "timer": "perf_counter_ns", "new_http_connection_per_request": True},
        **summarise(records, warmup_records), "samples": records, "warmup_samples": warmup_records,
    }


def write_report(report, output):
    output = Path(output)
    if output.suffix.lower() != ".json":
        raise ValueError("output must be a .json path; companion CSV uses the same stem")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["phase", "index", "question", "latency_ms", "http_status", "response_status",
                  "answer_kind", "revision", "route", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([*report["warmup_samples"], *report["samples"]])
    return output, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=5.0, help="Per-request socket timeout; at most 30 seconds")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "structured-http.json")
    args = parser.parse_args()
    try:
        if args.output.suffix.lower() != ".json":
            raise ValueError("output must be a .json path")
        report = run_benchmark(args.url, args.samples, args.warmup, args.timeout)
        output, csv_path = write_report(report, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Benchmark status: {report['status']}; {report['counts']['answered']}/{report['counts']['attempted']} answered, "
          f"{report['counts']['abstentions']} abstentions, {report['counts']['errors']} errors.")
    print(f"HTTP roundtrip p50={report['metrics']['http_roundtrip_p50_ms']:.3f} ms; "
          f"p95={report['metrics']['http_roundtrip_p95_ms']:.3f} ms.")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Saved {output} and {csv_path}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
