"""One fixed TRAIN-row integration profile of the final five-method pipeline.

Default/--plan uses only the standard library. --run executes exactly one
development episode in a fresh subprocess with a hard 600-second worker cap.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ke_repro.evaluate import CATEGORIES, EDIT_METHODS, METHODS, evaluate_edit, validate_rag
from ke_repro.smoke import Resources, digest, preflight, write_json

WORKER_CAP_SECONDS = 600


def plan():
    return {"mode": "plan", "models_loaded": False, "network_requested": False,
            "evaluation_split_opened": False, "source_split": "train", "source_index": 0,
            "methods": list(METHODS), "categories": list(CATEGORIES), "generated_queries": 25,
            "mend_updates": 1, "ft_optimizer_steps": 15, "meta_training_requested": False,
            "maximum_worker_seconds": WORKER_CAP_SECONDS,
            "runtime_estimate_seconds": None,
            "estimate_note": "Prior separate T4 development smoke runs took about 120s (MEND) and 132s (FT), each including model loading. A shared load near 100s plus a 20-40s episode is an uncertain planning guess, not a measured pipeline estimate or promise; use this profile's timings.",
            "stopping_condition": "One fixed train-row0 episode; 25 queries, one MEND edit and one 15-step FT edit; exact rollback required",
            "expected_artifacts": ["profile.json", "worker.log", "preflight.json"],
            "success_rule": "All 25 calls executed and exact rollback; abstentions and wrong answers are measured outcomes"}


def source_receipts(args):
    paths = sorted([*ROOT.glob("ke_repro/*.py"), ROOT / "scripts/profile_vlkeb_pipeline.py",
                    ROOT / "scripts/prepare_vlkeb_assets.py", ROOT / "scripts/prepare_vlkeb_eval.py",
                    ROOT / "scripts/prepare_vlkeb_rag.py", ROOT / "requirements-vlkeb.txt"])
    sources = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in paths}
    sources.update({"asset_manifest": digest(args.manifest),
                    "train_fixture": digest(args.assets / "train-fixture.json"),
                    "train_annotations": digest(args.assets / "upstream-train.json"),
                    "rag_calibration": digest(args.rag_calibration),
                    "rag_model_manifest": digest(args.rag_root / "minilm-manifest.json")})
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     text=True, timeout=20).strip()
    return {"source_file_sha256": sources, "actual_code_commit": commit}


def fixed_fixture(args):
    fixture = json.loads((args.assets / "train-fixture.json").read_text(encoding="utf-8"))
    train = json.loads((args.assets / "upstream-train.json").read_text(encoding="utf-8"))
    if (fixture.get("upstream_split") != "train" or fixture.get("purpose") != "development_smoke_only"
            or not train or fixture.get("record") != train[0]):
        raise ValueError("Pipeline profile accepts only the verified existing TRAIN-row0 fixture")
    return fixture


def execution_success(row):
    return (row.get("exact_restore") is True
            and set(row.get("methods", {})) == set(METHODS)
            and all(len(row["methods"][method]) == len(CATEGORIES)
                    and sorted(answer.get("category") for answer in row["methods"][method]) == sorted(CATEGORIES)
                    and all(answer.get("status") in ("completed", "abstention") for answer in row["methods"][method])
                    for method in METHODS)
            and set(row.get("method_edits", {})) == set(EDIT_METHODS)
            and all(not operation.get("error") and operation.get("exact_restore") is True
                    for operation in row.get("method_edits", {}).values()))


def worker(args):
    backend, resources, torch_module = None, None, None
    report = {**plan(), "mode": "execution", "execution_status": "failed",
              "provenance": "own_measured_development_only", "meta_training_executed": False}
    code = 1
    try:
        receipt_path = args.output / "preflight.json"
        if digest(receipt_path) != args.expected_preflight_sha256:
            raise ValueError("Verified pipeline preflight receipt changed")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["sources"] != source_receipts(args):
            raise ValueError("Code or source files changed after pipeline preflight")
        fixture = fixed_fixture(args)
        report.update(receipt["sources"], preflight=receipt["base_verification"])
        import torch
        from ke_repro.bridge import build_backend
        from ke_repro.rag import MiniLMEncoder, VisualCorrectionRetriever
        torch_module = torch
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        with Resources() as resources:
            started = time.perf_counter()
            backend = build_backend(args.upstream, args.assets, device=args.device)
            retriever = VisualCorrectionRetriever(backend, MiniLMEncoder(args.rag_root / "minilm"),
                                                   receipt["rag_threshold"])
            torch.cuda.synchronize()
            report.update(models_loaded=True, load_seconds=time.perf_counter() - started)
            started = time.perf_counter()
            row = evaluate_edit(backend, {"edit_id": "development-train-0000", "record": fixture["record"]},
                                args.assets, "development_not_final_test", torch.cuda.synchronize,
                                retriever=retriever)
            torch.cuda.synchronize()
            report.update(unit_seconds=time.perf_counter() - started, unit_row=row,
                          bounded_per_edit_fine_tuning_attempted=row.get("fine_tuning_attempted", False))
        report["resources"] = resources.summary(torch)
        success = execution_success(row)
        report.update(execution_status="completed" if success else "failed", execution_gate_passed=success,
                      compatibility_adaptations=backend.compatibility)
        code = 0 if success else 1
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if resources is not None and hasattr(resources, "elapsed") and torch_module is not None:
            report["resources"] = resources.summary(torch_module)
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as error:
                report.update(cleanup_error=str(error), execution_status="failed", execution_gate_passed=False)
                code = 1
        write_json(args.output / "profile.json", report)
    return code


def run(args):
    if args.output.exists():
        raise FileExistsError("Use a new profile output directory; previous evidence is preserved")
    args.output.mkdir(parents=True)
    report = {**plan(), "mode": "execution", "execution_status": "preflight_failed",
              "execution_gate_passed": False, "meta_training_executed": False}
    try:
        verification = preflight(args)
        calibration = validate_rag(args, verification)
        fixed_fixture(args)
        report.update(source_receipts(args), preflight=verification)
        receipt = {"base_verification": verification, "sources": source_receipts(args),
                   "rag_threshold": calibration["selection"]["threshold"]}
        receipt_path = args.output / "preflight.json"
        write_json(receipt_path, receipt)
        report["execution_status"] = "started"
        write_json(args.output / "profile.json", report)
        command = [sys.executable, "-m", "scripts.profile_vlkeb_pipeline", "--_worker"]
        for name in ("upstream", "assets", "manifest", "rag_root", "rag_calibration", "device", "output"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        command.extend(["--expected-preflight-sha256", digest(receipt_path)])
        timed_out = False
        with (args.output / "worker.log").open("w", encoding="utf-8") as log:
            try:
                code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=WORKER_CAP_SECONDS,
                                      env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                           "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}).returncode
            except subprocess.TimeoutExpired:
                timed_out, code = True, None
        report = json.loads((args.output / "profile.json").read_text(encoding="utf-8"))
        passed = code == 0 and report.get("execution_status") == "completed" and execution_success(report.get("unit_row", {}))
        report.update(worker_return_code=code, execution_gate_passed=passed)
        if not passed:
            report["execution_status"] = "timed_out" if timed_out else "failed"
        write_json(args.output / "profile.json", report)
        return 0 if passed else 1
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      execution_gate_passed=False)
        if report["execution_status"] != "preflight_failed":
            report["execution_status"] = "failed"
        write_json(args.output / "profile.json", report)
        (args.output / "worker.log").touch(exist_ok=True)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-preflight-sha256", help=argparse.SUPPRESS)
    for name, default in (("upstream", "tmp/vlkeb-upstream"), ("assets", "artifacts/vlkeb/assets"),
                          ("manifest", "configs/vlkeb-assets.json"), ("rag-root", "artifacts/vlkeb/rag"),
                          ("rag-calibration", "artifacts/vlkeb/rag/calibration.json"),
                          ("output", "artifacts/vlkeb/pipeline-profile")):
        parser.add_argument("--" + name, type=Path, default=ROOT / default)
    parser.add_argument("--device", choices=("cuda", "cuda:0"), default="cuda")
    args = parser.parse_args(argv)
    if not (args.run or args._worker):
        print(json.dumps(plan(), indent=2))
        return 0
    if args._worker and not args.expected_preflight_sha256:
        parser.error("Private worker requires its controller's preflight checksum")
    for name in ("upstream", "assets", "manifest", "rag_root", "rag_calibration", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.fixture = None
    try:
        code = worker(args) if args._worker else run(args)
        print(json.dumps({"profile": str(args.output / "profile.json"), "return_code": code}, indent=2))
        return code
    except Exception as error:
        print(json.dumps({"execution_status": "failed", "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
