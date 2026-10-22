"""One development edit, four generated answers, persistence and exact restoration.

The controller runs editing and reload in distinct processes. It deliberately
does not open an evaluation split, select examples by observed success, run
meta-training, or turn a one-example diagnostic into an accuracy estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import unicodedata

from . import UPSTREAM_COMMIT


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalize_answer(value: str) -> str:
    # Whole-answer comparison: a rejected target appearing inside prose is not a hit.
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in value).split())


def exact_short_answer(answer: str, expected: str) -> bool:
    return bool(normalize_answer(expected)) and normalize_answer(answer) == normalize_answer(expected)


def inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Fixture image must be a nonempty relative path")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Fixture image escapes the asset directory")
    return target


def load_cases(asset_root: Path, fixture_path: Path) -> tuple[dict, list[dict]]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if fixture.get("upstream_split") != "train" or fixture.get("purpose") != "development_smoke_only":
        raise ValueError("This smoke command accepts only a declared TRAIN development fixture")
    record = fixture["record"]
    for key in ("image", "image_rephrase", "m_loc", "src", "rephrase", "m_loc_q", "alt", "m_loc_a"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"Missing fixture field: {key}")
    specifications = [
        ("direct", "image", "src", "alt"),
        ("paraphrase", "image", "rephrase", "alt"),
        ("alternate_image", "image_rephrase", "src", "alt"),
        ("neighbor", "m_loc", "m_loc_q", "m_loc_a"),
    ]
    cases = []
    for name, image_key, question_key, expected_key in specifications:
        image = inside(asset_root / "images", record[image_key])
        if not image.is_file():
            raise FileNotFoundError(image)
        cases.append({"id": name, "image": str(image), "question": record[question_key],
                      "expected": record[expected_key]})
    return fixture, cases


def verify_upstream(upstream: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True, timeout=20
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"Wrong upstream revision: {revision}")
    clean = subprocess.run(["git", "-C", str(upstream), "diff", "--quiet", "HEAD", "--"], timeout=20)
    if clean.returncode != 0:
        raise ValueError("Pinned upstream tracked sources have changed")
    return revision


def make_plan(args: argparse.Namespace) -> dict:
    return {
        "question": "Can the published BLIP-2 MEND checkpoint execute one edit, generate short answers, persist it and exactly restore its six editable matrices on this runtime?",
        "purpose": "development_feasibility_and_resource_profile",
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset_split": "train",
        "training_executed": False,
        "meta_training_requested": False,
        "edit_count": 1,
        "questions_per_state": 4,
        "maximum_generated_tokens_per_question": args.max_new_tokens,
        "maximum_seconds_per_worker": args.timeout_seconds,
        "worker_count": 2,
        "expected_artifacts": ["run_plan.json", "edit.json", "reload.json", "smoke.json", "patch", "edit.log", "reload.log"],
        "finite_stopping_condition": "One edit followed by four diagnostic queries, same-process restoration, then one fresh-process patch load and restoration; no further fitting or example selection.",
        "checkpoint_resume": "Assets are separately reusable. This one-edit diagnostic restarts into a new output directory; it does not resume partial model state.",
        "runtime_estimate_seconds": None,
        "gpu_memory_estimate_bytes": None,
        "estimate_reason": "No run of this exact configuration has yet supplied a profile. Timeout is a cap, not a runtime estimate. Neighboring paper memory figures are not our measurements.",
        "scoring": "Conservative normalized WHOLE generated answer against the author target; preservation and neighbor target matching reported separately. No accuracy claim from this single training example.",
        "label_limitations": "Author-provided counterfactual benchmark labels; training-seen example, no independent human review. Not an approved real-world factual correction.",
    }


def preflight(args: argparse.Namespace) -> dict:
    # This module is stdlib-only; preflight can be run before installing any ML package.
    from scripts.prepare_vlkeb_assets import load_manifest, verify_assets

    manifest = load_manifest(args.manifest)
    verification = verify_assets(manifest, args.assets)
    if not verification.get("verified") or not verification.get("fixture_prepared"):
        raise ValueError("Assets or TRAIN fixture are not verified: " + json.dumps(verification))
    revision = verify_upstream(args.upstream)
    fixture_path = args.fixture or args.assets / "train-fixture.json"
    if digest(fixture_path) != digest(args.assets / "train-fixture.json"):
        raise ValueError("This fixed smoke accepts only the independently verified training fixture")
    fixture, cases = load_cases(args.assets, fixture_path)
    return {"assets_verified": True, "upstream_revision": revision,
            "asset_manifest_sha256": digest(args.manifest), "fixture_sha256": digest(fixture_path),
            "fixture_source_revision": fixture.get("source_revision"),
            "case_count": len(cases), "image_sha256": {case["id"]: digest(Path(case["image"])) for case in cases}}


class Resources:
    """Sample host RSS during construction as well as editing (50 ms interval)."""

    def __init__(self):
        self.peak_rss = 0
        self._stop = threading.Event()

    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self._stop.wait(0.05)

    def __exit__(self, *_):
        self._stop.set()
        self.thread.join()
        self.elapsed = time.perf_counter() - self.started

    def summary(self, torch) -> dict:
        gpu = {}
        if torch.cuda.is_available():
            gpu = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "device": torch.cuda.get_device_name()}
        return {"elapsed_seconds": self.elapsed, "sampled_peak_process_rss_bytes": self.peak_rss,
                "host_sampling_interval_seconds": 0.05, "gpu": gpu,
                "limits": "RSS sampling can miss short spikes; CUDA allocator figures do not include every driver allocation. These are diagnostic-run resources, not a serving latency benchmark."}


def generate_cases(backend, cases: list[dict], max_new_tokens: int) -> list[dict]:
    results = []
    for case in cases:
        start = time.perf_counter()
        answer = backend.generate(case["image"], case["question"], max_new_tokens=max_new_tokens)
        if not isinstance(answer, str):
            raise TypeError("Backend must return generated text")
        results.append({**case, "answer": answer, "seconds": time.perf_counter() - start,
                        "strict_target_match": exact_short_answer(answer, case["expected"])})
    return results


def answer_map(rows: list[dict]) -> dict:
    return {row["id"]: row["answer"] for row in rows}


def edit_phase(backend, cases, output: Path, max_new_tokens: int) -> dict:
    base_hashes = backend.weight_hashes()
    baseline = generate_cases(backend, cases, max_new_tokens)
    direct = cases[0]
    metadata = backend.edit(direct["image"], direct["question"], direct["expected"])
    edited_hashes = backend.weight_hashes()
    edited = generate_cases(backend, cases, max_new_tokens)
    exported = backend.save_patch(output / "patch")
    rollback_metadata = backend.rollback()
    restored_hashes = backend.weight_hashes()
    restored = generate_cases(backend, cases, max_new_tokens)
    return {
        "base_hashes": base_hashes, "edited_hashes": edited_hashes,
        "restored_hashes": restored_hashes, "baseline": baseline, "edited": edited,
        "restored": restored, "edit_metadata": metadata, "exported_patch": exported,
        "rollback_metadata": rollback_metadata,
        "checks": {
            "some_editable_matrix_changed": base_hashes != edited_hashes,
            "all_editable_matrices_restored_exactly": base_hashes == restored_hashes,
            "baseline_generated_answers_restored": answer_map(baseline) == answer_map(restored),
        },
    }


def reload_phase(backend, cases, output: Path, max_new_tokens: int) -> dict:
    original = json.loads((output / "edit.json").read_text(encoding="utf-8"))
    fresh_hashes = backend.weight_hashes()
    backend.load_patch(output / "patch")
    loaded_hashes = backend.weight_hashes()
    loaded = generate_cases(backend, cases, max_new_tokens)
    backend.rollback()
    restored_hashes = backend.weight_hashes()
    restored = generate_cases(backend, cases, max_new_tokens)
    return {
        "fresh_base_hashes": fresh_hashes, "loaded_hashes": loaded_hashes,
        "restored_hashes": restored_hashes, "loaded": loaded, "restored": restored,
        "checks": {
            "fresh_process_base_matches": fresh_hashes == original["base_hashes"],
            "loaded_editable_matrices_match": loaded_hashes == original["edited_hashes"],
            "loaded_generated_answers_match": answer_map(loaded) == answer_map(original["edited"]),
            "fresh_process_editable_matrices_restored_exactly": restored_hashes == fresh_hashes,
            "fresh_process_baseline_answers_restored": answer_map(restored) == answer_map(original["baseline"]),
        },
    }


def worker(args: argparse.Namespace) -> int:
    report_path = args.output / (args.phase + ".json")
    report = {"phase": args.phase, "execution_status": "failed"}
    backend = None
    resources = None
    torch_module = None
    try:
        import torch
        torch_module = torch
        from .bridge import build_backend
        torch.manual_seed(42)
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; this command does not silently switch to CPU")
            torch.cuda.manual_seed_all(42)
            torch.cuda.reset_peak_memory_stats()
        _, cases = load_cases(args.assets, args.fixture or args.assets / "train-fixture.json")
        with Resources() as resources:
            backend = build_backend(args.upstream, args.assets, device=args.device)
            operation = edit_phase if args.phase == "edit" else reload_phase
            report.update(operation(backend, cases, args.output, args.max_new_tokens))
            report["compatibility_adaptations"] = getattr(backend, "compatibility", {})
        report["resources"] = resources.summary(torch)
        report["execution_status"] = "completed"
        return_code = 0
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        if resources is not None and hasattr(resources, "elapsed") and torch_module is not None:
            report["resources"] = resources.summary(torch_module)
        return_code = 1
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as error:
                report["cleanup_error"] = str(error)
                report["execution_status"] = "failed"
                return_code = 1
        write_json(report_path, report)
    return return_code


def combine(edit: dict, reload: dict | None) -> dict:
    checks = {**edit.get("checks", {}), **((reload or {}).get("checks", {}))}
    completed = edit.get("execution_status") == "completed" and (reload or {}).get("execution_status") == "completed"
    return {
        "execution_status": "completed" if completed else "failed",
        "technical_gate_passed": completed and len(checks) == 8 and all(checks.values()),
        "technical_checks": checks,
        "direct_edit_target_matched": next((row["strict_target_match"] for row in edit.get("edited", []) if row["id"] == "direct"), False),
        "generated_answer_diagnostics": edit.get("edited", []),
        "neighbor_answer_preserved": answer_map(edit.get("baseline", [])).get("neighbor") == answer_map(edit.get("edited", [])).get("neighbor") if edit.get("edited") else False,
        "accuracy_estimate": None,
        "interpretation": "One training-seen counterfactual development edit. Technical success, strict target matching and neighbor preservation are different outcomes. No benchmark accuracy, model-selection or held-out generalization claim.",
    }


def controller(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ValueError("Output already exists; choose a new run directory to preserve previous evidence")
    args.output.mkdir(parents=True)
    write_json(args.output / "run_plan.json", make_plan(args))
    try:
        verified = preflight(args)
        write_json(args.output / "preflight.json", verified)
    except Exception as error:
        write_json(args.output / "smoke.json", {"execution_status": "preflight_failed", "technical_gate_passed": False,
                   "error": str(error), "models_loaded": False, "training_executed": False})
        raise
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}
    phases = {}
    for phase in ("edit", "reload"):
        command = [sys.executable, "-m", "ke_repro", "_worker", "--phase", phase,
                   "--upstream", str(args.upstream), "--assets", str(args.assets),
                   "--manifest", str(args.manifest), "--output", str(args.output),
                   "--device", args.device, "--max-new-tokens", str(args.max_new_tokens)]
        if args.fixture:
            command.extend(["--fixture", str(args.fixture)])
        print(json.dumps({"event": "worker_started", "phase": phase}), flush=True)
        timed_out = False
        with (args.output / (phase + ".log")).open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.timeout_seconds, env=environment)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                timed_out, return_code = True, None
        path = args.output / (phase + ".json")
        phases[phase] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "execution_status": "timed_out" if timed_out else "process_failed",
            "return_code": return_code, "error": "Worker ended without a phase report; inspect its log and runtime memory."}
        if return_code != 0:
            phases[phase]["worker_reported_status"] = phases[phase].get("execution_status")
            phases[phase]["execution_status"] = "timed_out" if timed_out else "process_failed"
            phases[phase]["return_code"] = return_code
            break
    summary = combine(phases["edit"], phases.get("reload"))
    summary["phase_execution"] = {key: {name: value for name, value in result.items()
                                       if name in ("execution_status", "error", "error_type", "return_code", "resources")}
                                  for key, result in phases.items()}
    summary["preflight"] = verified
    write_json(args.output / "smoke.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["technical_gate_passed"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "smoke", "_worker"))
    parser.add_argument("--upstream", type=Path, default=Path("tmp/vlkeb-upstream"))
    parser.add_argument("--assets", type=Path, default=Path("artifacts/vlkeb/assets"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/vlkeb-assets.json"))
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/vlkeb/smoke-1"))
    parser.add_argument("--device", choices=("cuda", "cuda:0", "cpu"), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--phase", choices=("edit", "reload"))
    args = parser.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 32 or not 1 <= args.timeout_seconds <= 3600:
        parser.error("Use 1–32 tokens and a finite worker timeout of 1–3600 seconds")
    if args.command == "_worker" and not args.phase:
        parser.error("A worker requires --phase")
    for key in ("upstream", "assets", "manifest", "output", "fixture"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.command == "plan":
        print(json.dumps(make_plan(args), indent=2))
        return 0
    try:
        return worker(args) if args.command == "_worker" else controller(args)
    except Exception as error:
        print(json.dumps({"execution_status": "failed", "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1
