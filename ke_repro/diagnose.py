"""Finite, no-fitting diagnosis of an existing VLKEB development patch.

Compares exactly two predeclared prompt formats on the same four TRAIN-fixture
cases. Separately records teacher-forced scoring under release/context labels.
No new edit, meta-training, example selection or benchmark evaluation is run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from .smoke import Resources, digest, generate_cases, load_cases, preflight, write_json


FORMATS = ("raw", "vqa")
ALIGNMENTS = ("release", "context")
TOKEN_BUDGET = 24


def formatted_cases(cases, prompt_format):
    if prompt_format not in FORMATS:
        raise ValueError("Only the fixed raw and official VQA formats are supported")
    return [{**case, "question": case["question"] if prompt_format == "raw"
             else f"Question: {case['question']} Answer:"} for case in cases]


def _teacher_forced(backend, direct):
    return {alignment: backend.score_target(direct["image"], direct["question"],
                                            direct["expected"], target_alignment=alignment)
            for alignment in ALIGNMENTS}


def diagnostic_operation(backend, cases, patch):
    """One fresh base, one existing patch load, two formats, then exact rollback."""
    if len(cases) != 4 or cases[0]["id"] != "direct":
        raise ValueError("Diagnosis requires the fixed four cases beginning with direct")
    inputs = {name: formatted_cases(cases, name) for name in FORMATS}
    base_hashes = backend.weight_hashes()
    tokenization = {name: backend.tokenization_report(rows[0]["question"], rows[0]["expected"])
                    for name, rows in inputs.items()}
    baseline = {name: generate_cases(backend, rows, TOKEN_BUDGET) for name, rows in inputs.items()}
    baseline_scores = {name: _teacher_forced(backend, rows[0]) for name, rows in inputs.items()}
    loaded = False
    try:
        load_metadata = backend.load_patch(patch)
        loaded = True
        loaded_hashes = backend.weight_hashes()
        patched = {name: generate_cases(backend, rows, TOKEN_BUDGET) for name, rows in inputs.items()}
        patched_scores = {name: _teacher_forced(backend, rows[0]) for name, rows in inputs.items()}
    finally:
        if loaded:
            rollback_metadata = backend.rollback()
    restored_hashes = backend.weight_hashes()
    return {"edit_count": 0, "training_executed": False,
            "new_parameter_update_fitted": False, "existing_patch_loaded": True,
            "prompt_formats": list(FORMATS), "cases_per_format": len(cases),
            "maximum_new_tokens": TOKEN_BUDGET,
            "base_hashes": base_hashes, "loaded_hashes": loaded_hashes,
            "restored_hashes": restored_hashes,
            "tokenization": tokenization, "baseline": baseline, "loaded_patch": patched,
            "teacher_forced": {"baseline": baseline_scores, "loaded_patch": patched_scores},
            "load_metadata": load_metadata, "rollback_metadata": rollback_metadata,
            "checks": {"existing_patch_changes_some_editable_weights": loaded_hashes != base_hashes,
                       "all_editable_matrices_restored_exactly": restored_hashes == base_hashes},
            "accuracy_estimate": None,
            "interpretation": "Fixed training-seen diagnostic, not held-out evaluation. Generated answers, teacher-forced target-token scores, BPE alignment and mechanical restoration are separate evidence. No new patch was fitted or successful example selected."}


def plan(args):
    return {"question": "Do the fixed official VQA prompt and context-token alignment explain the first patch's observed failures, and can a fresh process load and reverse that patch?",
            "purpose": "existing_patch_diagnosis_without_fitting", "dataset_split": "train",
            "edit_count": 0, "training_executed": False,
            "prompt_formats": {"raw": "{question}", "vqa": "Question: {question} Answer:"},
            "prompt_source": "https://github.com/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb",
            "generated_answer_count": 16, "teacher_forced_score_count": 8,
            "tokenization_report_count": 2, "maximum_new_tokens": TOKEN_BUDGET,
            "maximum_worker_seconds": args.timeout_seconds,
            "finite_stopping_condition": "One fresh backend, two fixed formats on four fixed cases before/after one existing patch load, eight teacher-forced scores, exact-weight rollback; no fitting.",
            "runtime_estimate_seconds": None,
            "estimate_note": "The timeout is a cap. This exact diagnostic has not yet been measured.",
            "expected_artifacts": ["run_plan.json", "preflight.json", "diagnostic.json", "diagnostic.log"]}


def worker(args):
    report = {"execution_status": "failed", "training_executed": False, "edit_count": 0}
    backend, resources, torch_module = None, None, None
    return_code = 1
    try:
        import torch
        from .bridge import build_backend

        torch_module = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; diagnosis does not silently switch to CPU")
        if digest(args.patch) != args.expected_patch_sha256:
            raise ValueError("The existing patch changed after preflight")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        _, cases = load_cases(args.assets, args.assets / "train-fixture.json")
        with Resources() as resources:
            backend = build_backend(args.upstream, args.assets, device=args.device)
            report.update(diagnostic_operation(backend, cases, args.patch))
            report["compatibility_adaptations"] = backend.compatibility
        report["resources"] = resources.summary(torch)
        report["execution_status"] = "completed"
        return_code = 0
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if resources is not None and hasattr(resources, "elapsed") and torch_module is not None:
            report["resources"] = resources.summary(torch_module)
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as error:
                report.update(cleanup_error=str(error), execution_status="failed")
                return_code = 1
        write_json(args.output / "diagnostic.json", report)
    return return_code


def controller(args):
    if args.output.exists():
        raise ValueError("Output already exists; use a fresh directory to preserve earlier evidence")
    if not args.patch.is_file():
        raise FileNotFoundError(f"The existing patch is missing: {args.patch}")
    args.output.mkdir(parents=True)
    write_json(args.output / "run_plan.json", plan(args))
    try:
        verification = preflight(args)
        verification["existing_patch_sha256"] = digest(args.patch)
        verification["existing_patch_bytes"] = args.patch.stat().st_size
        write_json(args.output / "preflight.json", verification)
    except Exception as error:
        write_json(args.output / "diagnostic.json", {"execution_status": "preflight_failed",
                   "error_type": type(error).__name__, "error": str(error),
                   "models_loaded": False, "training_executed": False, "edit_count": 0})
        raise
    command = [sys.executable, "-m", "ke_repro.diagnose", "--_worker",
               "--upstream", str(args.upstream), "--assets", str(args.assets),
               "--manifest", str(args.manifest), "--output", str(args.output),
               "--patch", str(args.patch), "--device", args.device,
               "--expected-patch-sha256", verification["existing_patch_sha256"]]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}
    timed_out = False
    with (args.output / "diagnostic.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.timeout_seconds, env=environment)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out, return_code = True, None
    path = args.output / "diagnostic.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "execution_status": "timed_out" if timed_out else "process_failed",
        "error": "Worker ended without a diagnostic report; inspect its log and runtime memory."}
    if return_code != 0:
        report["worker_reported_status"] = report.get("execution_status")
        report["execution_status"] = "timed_out" if timed_out else "process_failed"
    report["worker_return_code"] = return_code
    report["preflight"] = verification
    write_json(path, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if return_code == 0 and report.get("execution_status") == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("tmp/vlkeb-upstream"))
    parser.add_argument("--assets", type=Path, default=Path("artifacts/vlkeb/assets"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/vlkeb-assets.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cuda:0"), default="cuda")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-patch-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 1800:
        parser.error("The worker needs a finite timeout from 1 to 1800 seconds")
    if args._worker and not args.expected_patch_sha256:
        parser.error("A private worker requires its preflight patch hash")
    for name in ("upstream", "assets", "manifest", "output", "patch"):
        setattr(args, name, getattr(args, name).resolve())
    args.fixture = None
    try:
        return worker(args) if args._worker else controller(args)
    except Exception as error:
        print(json.dumps({"execution_status": "failed", "error_type": type(error).__name__,
                          "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
