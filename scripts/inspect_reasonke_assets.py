"""Audit the pinned public checkpoint metadata without downloading model weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def read_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "EditGuard-asset-audit/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Metadata exceeds the inspection budget")
    return json.loads(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/reasonke/asset-audit.json")
    args = parser.parse_args(argv)
    path = ROOT / "configs/reasonke-model.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    model_id, revision = manifest["model_id"], manifest["revision"]
    base = "https://huggingface.co/" + model_id
    metadata = read_json("https://huggingface.co/api/models/" + model_id + "/revision/" + revision)
    config = read_json(base + "/resolve/" + revision + "/config.json")
    index = read_json(base + "/resolve/" + revision + "/model.safetensors.index.json")
    files = sorted(row["rfilename"] for row in metadata["siblings"])
    total_bytes = index["metadata"]["total_size"]
    checks = {
        "revision_matches": metadata["sha"] == revision,
        "public": metadata.get("private") is False,
        "ungated": metadata.get("gated") is False,
        "qwen2_architecture": config.get("architectures") == ["Qwen2ForCausalLM"],
        "source_tensor_bytes_match": total_bytes == manifest["source_tensor_bytes"],
        "all_indexed_shards_listed": set(index["weight_map"].values()).issubset(files),
        "tokenizer_present": "tokenizer.json" in files and "chat_template.jinja" in files,
    }
    report = {
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id, "revision": revision,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": ["https://huggingface.co/api/models/" + model_id + "/revision/" + revision,
                    base + "/resolve/" + revision + "/config.json",
                    base + "/resolve/" + revision + "/model.safetensors.index.json"],
        "checks": checks, "metadata_checks_passed": all(checks.values()),
        "files": files, "source_tensor_bytes": total_bytes,
        "config": config,
        "model_card_present": "README.md" in files,
        "checkpoint_license_metadata": metadata.get("cardData", {}).get("license"),
        "exact_paper_checkpoint_identity_verified": False,
        "weights_downloaded": False, "gpu_inference_ran": False,
        "our_accuracy_measured": False,
        "notice": "Asset availability audit only. No model loading, accuracy, T4 fit, or exact paper-score reproduction has been established.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model_id", "revision", "checks", "notice")}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
