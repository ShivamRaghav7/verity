"""Explicit local Ollama downloads; never called by a website request."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/chat-models.json").read_text(encoding="utf-8"))
MODELS = (CONFIG["chat_model"], CONFIG["embedding_model"])
BASE = "http://127.0.0.1:11434"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/chat/model-receipt.json"))
    args = parser.parse_args()
    if not args.download:
        print(json.dumps({"models": MODELS, "download_requested": False,
            "notice": "Ollama must be running locally. Run with --download to fetch official registry assets; this performs no training."}, indent=2))
        return
    records = []
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/chat-models.json").read_text(encoding="utf-8"))
    for model in MODELS:
        print("Preparing " + model, flush=True)
        request = urllib.request.Request(BASE + "/api/pull", data=json.dumps({"model": model, "stream": True}).encode(),
            headers={"Content-Type": "application/json"})
        last = None
        with urllib.request.urlopen(request, timeout=1800) as response:
            for line in response:
                row = json.loads(line)
                if row.get("error"):
                    raise RuntimeError(row["error"])
                progress = int(100 * row.get("completed", 0) / row["total"]) // 10 * 10 if row.get("total") else None
                state = (row.get("status"), progress)
                if state != last:
                    print(json.dumps({"model": model, "status": state[0], "percent_bucket": state[1]}), flush=True)
                    last = state
        with urllib.request.urlopen(BASE + "/api/tags", timeout=10) as response:
            tags = json.load(response)
        installed = next(item for item in tags["models"] if item["name"] == model)
        expected = config["chat_digest"] if model == config["chat_model"] else config["embedding_digest"]
        if installed["digest"] != expected:
            raise RuntimeError("The official tag now differs from the recorded digest; preserve the existing evidence and review the model pin before proceeding.")
        request = urllib.request.Request(BASE + "/api/show", data=json.dumps({"model": model}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            shown = json.load(response)
        records.append({"name": model, "digest": installed["digest"], "size_bytes": installed["size"],
            "details": installed.get("details"), "capabilities": shown.get("capabilities", []),
            "license": shown.get("license", ""), "parameters": shown.get("parameters", "")})
    receipt = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "source": "Official Ollama model registry",
        "training_executed": False, "models": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "receipt": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
