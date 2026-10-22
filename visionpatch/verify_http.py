"""Archived V1 HTTP check; requires a service launched with --legacy-api.

Use visionpatch.verify_workbench for the current website instead.
"""

import argparse
import base64
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--output", type=Path, default=Path("reports/visionpatch/http-smoke-v2.json"))
    args = parser.parse_args()
    parsed = urlparse(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("This smoke check activates a demo patch; only loopback is permitted")
    if args.output.exists():
        raise FileExistsError("Preserve prior measurements; choose another output path")

    def call(path, payload=None):
        request = urllib.request.Request(args.base_url.rstrip("/") + path,
                                         data=json.dumps(payload).encode() if payload is not None else None,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    state = call("/api/patch/rollback", {})
    original = state["projection_sha256"]
    gallery = call("/api/gallery")["rows"]
    example = next(row for row in gallery if row["variant"] == "distractor" and row["before"]["label"] != row["label"])
    path = Path("data/generated/visionpatch/development-v1") / example["path"]
    payload = {"image_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
               "variant": "distractor", "position": "top"}
    start = time.perf_counter()
    before = call("/api/predict", payload)
    first_seconds = time.perf_counter() - start
    call("/api/patch/apply", {"approved": True, "reason": "Finite local workflow regression check"})
    after = call("/api/predict", payload)
    restored = call("/api/patch/rollback", {})
    rolled = call("/api/predict", payload)
    if restored["projection_sha256"] != original or before["active"] != rolled["active"]:
        raise RuntimeError("Original projection or upload scores did not restore")
    if before["before"]["label"] != example["before"]["label"]:
        raise RuntimeError("Real upload encoder and gallery disagree")
    call("/api/patch/apply", {"approved": True, "reason": "Activate measured patch for demonstration"})
    durations = []
    for _ in range(12):
        start = time.perf_counter()
        call("/api/predict", payload)
        durations.append(time.perf_counter() - start)
    report = {
        "status": "passed", "image_id": example["image_id"], "source_label": example["label"],
        "original_prediction": before["active"]["label"], "patched_prediction": after["active"]["label"],
        "rollback_prediction": rolled["active"]["label"], "exact_projection_hash_restored": True,
        "device": after["device"], "encoder_was_loaded_before_check": state["encoder_device"] is not None,
        "first_http_seconds": first_seconds, "warm_requests": len(durations), "warm_http_seconds": durations,
        "warm_http_p50_seconds": statistics.median(durations),
        "warm_http_p95_seconds": sorted(durations)[math.ceil(.95 * len(durations)) - 1],
        "p95_method": "nearest rank ceil(.95*n)",
        "limitations": "One development image repeated; workflow and serial latency evidence, not recognition accuracy or a concurrency benchmark",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
