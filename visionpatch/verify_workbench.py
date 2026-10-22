"""Finite real-HTTP authoring/inference/rollback check against a local service.

This creates an auditable candidate, activates it, and restores the exact initial
state. It does not establish accuracy on unseen images or benchmark concurrency.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    with urlopen(Request(base + path, data=data, headers=headers), timeout=180) as response:
        return json.load(response)


def run(base):
    prefix = "/api/workbench"
    initial = request(base, prefix + "/status")["registry"]
    example = request(base, prefix + "/examples")["examples"][0]
    pair = {}
    for key in ("source", "reference"):
        with urlopen(base + example[key + "_url"], timeout=30) as response:
            pair[key] = base64.b64encode(response.read()).decode("ascii")
    identity = {"actor": "automated-local-workflow-check",
                "reason": "Exercise authored candidate, explicit approval, activation and exact rollback"}
    started = time.perf_counter()
    candidate = request(base, prefix + "/preview", {
        **identity, "title": "Measured HTTP workflow check", "target_word": example["target_word"],
        "expected_label": example["label"], "source": example["source"],
        "source_image_base64": pair["source"], "reference_image_base64": pair["reference"],
    })
    preview_seconds = time.perf_counter() - started
    patch_id = candidate["patch_id"]
    if request(base, prefix + "/status")["registry"] != initial:
        raise RuntimeError("Preview changed the active parameter state")
    request(base, prefix + f"/patches/{patch_id}/approve", {**identity, "approved": True})
    if request(base, prefix + "/status")["registry"] != initial:
        raise RuntimeError("Approval changed the active parameter state")
    activated = False
    try:
        active = request(base, prefix + f"/patches/{patch_id}/activate",
                         {**identity, "expected_revision": initial["revision"]})["registry"]
        activated = True
        if active["current_sha256"] == initial["current_sha256"]:
            raise RuntimeError("Activation did not change the projection")
        samples, inference = [], None
        for _ in range(12):
            started = time.perf_counter()
            inference = request(base, prefix + "/predict", {"image_base64": pair["source"]})
            samples.append(time.perf_counter() - started)
            if inference["registry"]["current_sha256"] != active["current_sha256"]:
                raise RuntimeError("Inference did not use the activated registry state")
        try:
            request(base, prefix + "/rollback", {**identity, "expected_revision": initial["revision"]})
            raise RuntimeError("Stale revision was accepted")
        except HTTPError as error:
            if error.code != 409:
                raise
    finally:
        if activated:
            current = request(base, prefix + "/status")["registry"]
            restored = request(base, prefix + "/rollback", {
                **identity, "expected_revision": current["revision"],
                "to_patch_id": initial["active_patch_ids"][-1] if initial["active_patch_ids"] else "base",
            })["registry"]
            if (restored["current_sha256"] != initial["current_sha256"]
                    or restored["active_patch_ids"] != initial["active_patch_ids"]):
                raise RuntimeError("Rollback did not restore the exact initial parameter chain")
    ordered = sorted(samples)
    return {"status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
            "base_url": base, "candidate_id": patch_id, "example": example,
            "initial": initial, "active": active, "restored": restored,
            "preview": candidate["preview"], "preview_seconds": preview_seconds,
            "inference": inference, "latency_seconds": {"samples": samples,
                "p50": statistics.median(samples), "p95": ordered[11], "count": len(samples)},
            "stale_revision_rejected": True, "exact_rollback": True,
            "limitations": ["Twelve serial requests for one support image, not a deployment latency benchmark",
                            "Support-image repair does not measure unseen-image accuracy",
                            "Actor identity is local user-supplied metadata, not authentication",
                            "Candidate and audit history intentionally persist after rollback"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--output", type=Path, default=Path("reports/visionpatch/workbench-http-v2.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve measured checks; choose a fresh output path")
    result = run(args.base_url.rstrip("/"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "preview_seconds", "latency_seconds", "exact_rollback")}, indent=2))


if __name__ == "__main__":
    main()
