"""After frozen depth inference, run 14 real India website generations once.

Inputs are two completed final ledgers and their frozen protocol, including its
four bound profile/development receipts. This helper audits them without a model
call before loading the pinned ReasonKE checkpoint. It then runs one finite
operational acceptance workload and leaves a separate empty demo workspace.
It never trains, retries benchmark questions or overwrites existing work.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys

from editguard_chat.server import validate_trusted_origin
from editguard_depth.metrics import audit_and_aggregate
from scripts.finish_operations_release import wait_ready, stop_owned
from scripts.verify_reasonke_live import atomic_write

ROOT = Path(__file__).resolve().parents[1]
PORTS = (8010, 8008, 8005)


def require_completed(directory):
    directory = Path(directory).resolve()
    paths = [directory / f"final-{model}.json" for model in ("reasonke", "instruction")]
    for path in paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (receipt.get("status") != "completed" or len(receipt.get("records", [])) != 272
                or any(row.get("outcome") == "unexecuted" for row in receipt["records"])):
            raise ValueError("Final depth inference must finish before another model starts: " + str(path))
    result = audit_and_aggregate(paths, protocol=directory / "protocol.json")
    if (result.get("phase") != "final" or result.get("execution_status") != "completed"
            or result.get("assigned_records") != 544 or result.get("independent_lineages") != 32
            or {name: data["overall"]["assigned"] for name, data in result.get("conditions", {}).items()}
            != {"R-RAG": 256, "G-RAG": 256, "R-ORACLE": 16, "G-ORACLE": 16}):
        raise ValueError("Frozen depth population or completed audit differs")
    return result


def ensure_idle():
    # Linux Colab: a completed last receipt can precede pipeline process exit.
    # Do not load a second checkpoint while any evaluator/worker is still alive.
    proc = Path("/proc")
    if proc.is_dir():
        modules = {"scripts.run_depth_pipeline", "editguard_depth.run",
                   "editguard_benchmark.run", "editguard_reasonke"}
        for path in proc.iterdir():
            if not path.name.isdecimal() or int(path.name) == os.getpid():
                continue
            try:
                args = (path / "cmdline").read_bytes().decode(errors="replace").split("\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if any(arg in modules for arg in args):
                raise ValueError("An evaluator/model process is still active; wait for its exit: PID " + path.name)
    for port in PORTS:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError(f"Port {port} is occupied; no existing service was stopped or replaced")


def launch(directory, benchmark_directory, *, trusted_origin=None, python=sys.executable):
    origin = validate_trusted_origin(trusted_origin)
    audit = require_completed(benchmark_directory)
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("Use a new acceptance/demo directory; existing work is never overwritten")
    ensure_idle()
    directory.mkdir(parents=True)
    atomic_write(directory / "prelaunch-audit.json", audit)
    state = {"schema": "editguard-depth-release-launch-v1", "status": "starting",
        "assigned_http_generations": 14, "context_policy": "iterative",
        "benchmark_retried": False, "primary_scoring_changed": False, "training_steps": 0,
        "session_tokens_recorded": False,
        "notice": "Separate India demonstration acceptance, not additional held-out benchmark evidence",
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
                "--manifest", "configs/reasonke-model.json"], cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT)
        state.update(status="loading_model", worker_pid=worker.pid)
        atomic_write(receipt_path, state)
        wait_ready(worker, "http://127.0.0.1:8010/status", token=env["EDITGUARD_REASONKE_TOKEN"], seconds=1800)
        with (directory / "http-site.log").open("a", encoding="utf-8") as log:
            measurement_site = subprocess.Popen([python, "-u", "-m", "editguard_chat.server", "--backend", "reasonke",
                "--context-policy", "iterative", "--port", "8008", "--storage", str(directory / "http-workspace")],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        wait_ready(measurement_site, "http://127.0.0.1:8008/api/status", seconds=30)
        state["status"] = "fourteen_http_generations"
        atomic_write(receipt_path, state)
        with (directory / "india-acceptance.log").open("a", encoding="utf-8") as log:
            completed = subprocess.run([python, "-u", "-m", "scripts.verify_india_http", "--base-url",
                "http://127.0.0.1:8008", "--output", str(directory / "india-acceptance.json")],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        stop_owned(measurement_site)
        measurement_site = None
        actual = json.loads((directory / "india-acceptance.json").read_text(encoding="utf-8"))
        state.update(http_command_exit=completed.returncode,
            http_outcome_counts=actual["outcome_counts"],
            http_verified_neural_response_count=actual["verified_neural_response_count"],
            http_consistency_checks=actual["consistency_checks"],
            http_initial_handbook_restored=actual["initial_handbook_restored"])
        atomic_write(receipt_path, state)
        # Wrong/abstained model answers remain visible outcomes. Execution,
        # provenance and consistency failures cannot become a successful release.
        if (completed.returncode or actual["status"] != "completed"
                or actual["verified_neural_response_count"] != 14
                or not actual["all_consistency_checks_passed"] or not actual["initial_handbook_restored"]
                or actual["outcome_counts"]["execution_error"] or actual["outcome_counts"]["unexecuted"]):
            raise RuntimeError("India acceptance stopped or failed operational checks; all outcomes retained without retry")
        command = [python, "-u", "-m", "editguard_chat.server", "--backend", "reasonke", "--context-policy", "iterative",
            "--port", "8005", "--storage", str(directory / "demo-workspace")]
        if origin:
            command += ["--trusted-origin", origin]
        with (directory / "demo-site.log").open("a", encoding="utf-8") as log:
            demo_site = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        wait_ready(demo_site, "http://127.0.0.1:8005/api/status", seconds=30)
        state.update(status="demo_ready", demo_pid=demo_site.pid, website_port=8005,
            demo_memory="Empty; approve the nine fictional India facts explicitly in Knowledge")
        atomic_write(receipt_path, state)
        return state
    except Exception as exc:
        state.update(status="stopped", error={"type": type(exc).__name__, "message": str(exc)})
        atomic_write(receipt_path, state)
        stop_owned(demo_site)
        stop_owned(worker)
        raise
    finally:
        stop_owned(measurement_site)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--benchmark-directory", required=True, type=Path)
    parser.add_argument("--trusted-origin")
    args = parser.parse_args()
    print(json.dumps(launch(args.directory, args.benchmark_directory,
                            trusted_origin=args.trusted_origin), indent=2))


if __name__ == "__main__":
    main()
