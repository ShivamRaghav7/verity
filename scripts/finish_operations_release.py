"""After completed final inference, run 16 real website generations and leave the demo ready.

Never starts while a final receipt is incomplete. No training, benchmark retry,
token disclosure or modification of an existing user workspace occurs.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from scripts.audit_operations_evidence import audit_receipts
from scripts.verify_reasonke_live import atomic_write
from editguard_chat.server import validate_trusted_origin

ROOT = Path(__file__).resolve().parents[1]


def require_completed(directory):
    directory = Path(directory)
    paths = [directory / ("final-" + model + ".json") for model in ("reasonke", "instruction")]
    for path in paths:
        receipt = json.loads(path.read_text())
        if receipt.get("status") != "completed" or any(row["outcome"] == "unexecuted" for row in receipt["records"]):
            raise ValueError("Final inference must finish before loading another model; ledger preserved: " + str(path))
    return audit_receipts(paths, protocol=directory / "protocol.json")


def wait_ready(process, url, *, token=None, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Owned service stopped before readiness; inspect its saved log")
        try:
            request = Request(url, headers={"Authorization": "Bearer " + token} if token else {})
            with urlopen(request, timeout=5) as response:
                result = json.load(response)
            if result.get("ready") or result.get("model", {}).get("ready"):
                return result
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError("Finite service-readiness deadline reached; logs preserved")


def stop_owned(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=10)


def launch(directory, benchmark_directory, *, trusted_origin=None, python=sys.executable):
    origin = validate_trusted_origin(trusted_origin)
    audit = require_completed(benchmark_directory)
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("Use a new acceptance/demo directory; existing work is never overwritten")
    for port in (8010, 8008, 8005):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError("Port " + str(port) + " is occupied; no service was stopped or replaced")
    directory.mkdir(parents=True)
    atomic_write(directory / "prelaunch-audit.json", audit)
    state = {"schema": "editguard-release-launch-v1", "status": "starting", "assigned_http_generations": 16,
             "http_population": {"direct_paraphrase_two_hop_restore": 12, "historical_comparison_atomic_rollback": 4},
             "benchmark_retried": False, "training_steps": 0, "session_tokens_recorded": False,
             "notice": "Operational development supplement; not extra independent held-out facts",
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    receipt_path = directory / "launch.json"
    atomic_write(receipt_path, state)
    env = dict(os.environ, EDITGUARD_REASONKE_TOKEN=secrets.token_urlsafe(32),
               EDITGUARD_REASONKE_URL="http://127.0.0.1:8010", HF_HUB_DOWNLOAD_TIMEOUT="180",
               TOKENIZERS_PARALLELISM="false")
    worker = measurement_site = demo_site = None
    try:
        with (directory / "worker.log").open("a", encoding="utf-8") as log:
            worker = subprocess.Popen([python, "-u", "-m", "editguard_reasonke", "--serve", "--port", "8010",
                "--manifest", "configs/reasonke-model.json"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(status="loading_model", worker_pid=worker.pid); atomic_write(receipt_path, state)
        wait_ready(worker, "http://127.0.0.1:8010/status", token=env["EDITGUARD_REASONKE_TOKEN"], seconds=1800)
        with (directory / "http-site.log").open("a", encoding="utf-8") as log:
            measurement_site = subprocess.Popen([python, "-u", "-m", "editguard_chat.server", "--backend", "reasonke",
                "--context-policy", "retrieved", "--port", "8008", "--storage", str(directory / "http-workspace")],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        wait_ready(measurement_site, "http://127.0.0.1:8008/api/status", seconds=30)
        state["status"] = "twelve_http_requests"; atomic_write(receipt_path, state)
        with (directory / "http-acceptance.log").open("a", encoding="utf-8") as log:
            completed = subprocess.run([python, "-u", "-m", "scripts.verify_operations_http", "--base-url",
                "http://127.0.0.1:8008", "--output", str(directory / "http-acceptance.json")], cwd=ROOT,
                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        stop_owned(measurement_site); measurement_site = None
        result_path = directory / "http-acceptance.json"
        if result_path.exists():
            actual = json.loads(result_path.read_text())
            state["http_outcome_counts"] = actual["outcome_counts"]
            state["http_cleanup_errors"] = actual["cleanup_errors"]
        state["http_command_exit"] = completed.returncode
        # Wrong factual answers remain outcomes; a stopped/unclean run cannot
        # silently be promoted to an operationally complete release.
        if completed.returncode:
            raise RuntimeError("HTTP workload stopped or cleanup failed; all outcomes and logs preserved")
        # Historical comparison needs an entirely separate empty knowledge
        # history, so the earlier measurement workspace is never reset.
        with (directory / "history-site.log").open("a", encoding="utf-8") as log:
            measurement_site = subprocess.Popen([python, "-u", "-m", "editguard_chat.server", "--backend", "reasonke",
                "--context-policy", "retrieved", "--port", "8008", "--storage", str(directory / "history-workspace")],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        wait_ready(measurement_site, "http://127.0.0.1:8008/api/status", seconds=30)
        state["status"] = "four_history_generations"; atomic_write(receipt_path, state)
        with (directory / "history-acceptance.log").open("a", encoding="utf-8") as log:
            completed = subprocess.run([python, "-u", "-m", "scripts.verify_operations_history", "--base-url",
                "http://127.0.0.1:8008", "--output", str(directory / "history-acceptance.json")], cwd=ROOT,
                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        stop_owned(measurement_site); measurement_site = None
        if (directory / "history-acceptance.json").exists():
            history = json.loads((directory / "history-acceptance.json").read_text())
            state["history_outcome_counts"] = history["outcome_counts"]
            state["history_consistency_checks"] = history["consistency_checks"]
        state["history_command_exit"] = completed.returncode
        if completed.returncode:
            raise RuntimeError("History/rollback workload stopped or consistency failed; evidence preserved")
        command = [python, "-u", "-m", "editguard_chat.server", "--backend", "reasonke", "--context-policy", "retrieved",
                   "--port", "8005", "--storage", str(directory / "demo-workspace")]
        if origin: command += ["--trusted-origin", origin]
        with (directory / "demo-site.log").open("a", encoding="utf-8") as log:
            demo_site = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        wait_ready(demo_site, "http://127.0.0.1:8005/api/status", seconds=30)
        state.update(status="demo_ready", demo_pid=demo_site.pid, website_port=8005,
                     demo_memory="Empty; fictional handbook requires explicit approval in Knowledge")
        atomic_write(receipt_path, state)
        return state
    except Exception as exc:
        state.update(status="stopped", error={"type": type(exc).__name__, "message": str(exc)})
        atomic_write(receipt_path, state)
        stop_owned(demo_site); stop_owned(worker)
        raise
    finally:
        stop_owned(measurement_site)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--benchmark-directory", required=True, type=Path)
    parser.add_argument("--trusted-origin")
    args = parser.parse_args()
    print(json.dumps(launch(args.directory, args.benchmark_directory, trusted_origin=args.trusted_origin), indent=2))


if __name__ == "__main__":
    main()
