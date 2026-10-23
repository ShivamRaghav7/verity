"""Sealed finite20-edit/five-case evaluation, including bounded per-edit FT.

Default is a stdlib-only plan. --seal validates and records code/configuration/
selection before inference; --run requires that unchanged receipt. Each durable
edit row is written only after restoration has been attempted and checked.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from .smoke import Resources, digest, exact_short_answer, preflight, write_json

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("base", "prompt_only", "mend_context", "ft_last_layer")
EDIT_METHODS = ("mend_context", "ft_last_layer")
CATEGORIES = ("direct", "paraphrase", "alternate_image", "text_locality", "visual_locality")
TOKEN_BUDGET = 24
SEED = 42
EDIT_COUNT = 20


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(args):
    files = sorted([*ROOT.glob("ke_repro/*.py"), ROOT / "scripts/prepare_vlkeb_eval.py",
                    ROOT / "scripts/prepare_vlkeb_assets.py", ROOT / "docs/VLKEB_EVALUATION_PROTOCOL.md"])
    return {"code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in files},
            "manifest_sha256": digest(args.manifest), "selection_sha256": digest(args.selection),
            "configuration": {"methods": list(METHODS), "categories": list(CATEGORIES),
                "edit_count": EDIT_COUNT, "token_budget": TOKEN_BUDGET, "seed": SEED,
                "question_format": "Question: {query} Answer:", "target_alignment": "context",
                "prompt_template": "Question: {edit_src} Answer: {alt}\nQuestion: {query} Answer:",
                "decoding": "greedy", "timeout_seconds_per_attempt": args.timeout_seconds,
                "ft_optimizer": "AdamW", "ft_steps_per_edit": 15, "ft_learning_rate": .0002,
                "ft_answer_termination_supervision": "newline; desired scored answer unchanged",
                "mend_answer_termination_supervision": "none; context-aligned target suffix only",
                "ft_parameters": "OPT layer31 fc1.weight and fc2.weight only",
                "ft_precision_adaptation": "Base release precision/eval; FP32 AdamW master weights for the two matrices; not faithful full-FP32 upstream FT",
                "device": args.device, "bootstrap_resamples": 10000,
                "scorer": "smoke.exact_short_answer; normalized whole generated continuation",
                "resume": "Exact sealed code/configuration only; preserve completed rows; interrupted incomplete edit reruns from fresh base and is disclosed"}}


def validate_selection(args):
    from scripts.prepare_vlkeb_assets import extract_fixture_images, load_manifest, verify_asset
    from scripts.prepare_vlkeb_eval import EVAL_ASSET, TRAIN_SHA256, select_edits

    verification = preflight(args)
    manifest = load_manifest(args.manifest)
    train_asset = next(a for a in manifest["assets"] if a["path"] == "upstream-train.json")
    if train_asset.get("sha256") != TRAIN_SHA256:
        raise ValueError("Evaluation requires the pinned training split")
    verify_asset(args.assets / "upstream-eval.json", EVAL_ASSET)
    expected = select_edits(json.loads((args.assets / "upstream-train.json").read_text(encoding="utf-8")),
                            json.loads((args.assets / "upstream-eval.json").read_text(encoding="utf-8")))
    saved = json.loads(args.selection.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Frozen selection differs from deterministic source derivation: {key}")
    if saved.get("source_train_sha256") != TRAIN_SHA256:
        raise ValueError("Selection has wrong training-source checksum")
    required = sorted({e["record"][k] for e in expected["edits"] for k in ("image", "image_rephrase", "m_loc")})
    derived = extract_fixture_images(args.assets, manifest, {"required_images": required},
                                     verify_only=True, archive_verified=True)
    if saved.get("image_integrity") != derived:
        raise ValueError("Selection images differ from verified source archive")
    verification.update(selection_sha256=digest(args.selection), eval_annotation_sha256=EVAL_ASSET["sha256"],
                        selected_lineage_ids=[e["edit_id"] for e in expected["edits"]])
    return saved, verification


def seal(args):
    if args.output.exists():
        raise FileExistsError("Seal needs a new output directory; previous evidence is preserved")
    _, verification = validate_selection(args)
    args.output.mkdir(parents=True)
    receipt = {"schema_version": 1, "purpose": "evaluation_preregistration_before_inference",
               "provenance": "not_executed", "models_loaded": False,
               "fingerprint": fingerprint(args), "preflight": verification}
    write_json(args.output / "seal.json", receipt)
    return {"status": "sealed_not_evaluated", "seal_sha256": digest(args.output / "seal.json"),
            "output": str(args.output), "models_loaded": False}


def checked_seal(args):
    path = args.output / "seal.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("purpose") != "evaluation_preregistration_before_inference" or receipt.get("fingerprint") != fingerprint(args):
        raise ValueError("Sealed code, configuration or selection changed; do not tune against viewed outputs")
    return receipt, digest(path)


def cases_for(edit, assets):
    from scripts.prepare_vlkeb_assets import relative_path
    row = edit["record"]
    specifications = (("direct", "image", "src", "alt"),
                      ("paraphrase", "image", "rephrase", "alt"),
                      ("alternate_image", "image_rephrase", "src", "alt"),
                      ("text_locality", None, "loc", "loc_ans"),
                      ("visual_locality", "m_loc", "m_loc_q", "m_loc_a"))
    return [{"category": name, "image": str(relative_path(assets / "images", row[image])) if image else None,
             "query": row[query], "expected": row[target]} for name, image, query, target in specifications]


def failure_case(case, status, error):
    return {**case, "answer": "", "status": status, "error": error,
            "strict_target_match": False, "seconds": None}


def generate_one(backend, case, prompt, synchronize):
    started = None
    try:
        synchronize()
        started = time.perf_counter()
        answer = backend.generate(case["image"], prompt, max_new_tokens=TOKEN_BUDGET)
        synchronize()
        elapsed = time.perf_counter() - started
        if not isinstance(answer, str):
            raise TypeError("Backend must return generated text")
        return {**case, "answer": answer, "status": "completed" if answer.strip() else "abstention",
                "seconds": elapsed, "strict_target_match": exact_short_answer(answer, case["expected"])}
    except Exception as error:
        result = failure_case(case, "generation_error", f"{type(error).__name__}: {error}")
        result["seconds"] = time.perf_counter() - started if started is not None else None
        return result


def evaluate_edit(backend, edit, assets, seal_sha, synchronize=lambda: None):
    cases = cases_for(edit, assets)
    before = backend.weight_hashes()
    result = {"schema_version": 1, "edit_id": edit["edit_id"], "seal_sha256": seal_sha,
              "provenance": "own_measured", "methods": {}, "base_hashes": before}
    for method in ("base", "prompt_only"):
        answers = []
        for case in cases:
            prompt = (f"Question: {case['query']} Answer:" if method == "base" else
                      f"Question: {edit['record']['src']} Answer: {edit['record']['alt']}\nQuestion: {case['query']} Answer:")
            answers.append(generate_one(backend, case, prompt, synchronize))
        result["methods"][method] = answers
    result["method_edits"] = {}
    result["exact_restore"] = True
    for method in EDIT_METHODS:
        operation = {"attempted": True, "base_hashes": backend.weight_hashes()}
        if operation["base_hashes"] != before:
            operation.update(exact_restore=False, restore_error="Method did not start from original base")
            result["method_edits"][method] = operation
            result["exact_restore"] = False
            result["methods"][method] = [failure_case(c, "not_executed", operation["restore_error"]) for c in cases]
            break
        start = time.perf_counter()
        try:
            synchronize()
            arguments = (cases[0]["image"], f"Question: {edit['record']['src']} Answer:", edit["record"]["alt"])
            operation["metadata"] = (backend.edit(*arguments, target_alignment="context") if method == "mend_context"
                                     else backend.edit_finetune(*arguments))
            synchronize()
            operation["seconds"] = time.perf_counter() - start
            operation["edited_hashes"] = backend.weight_hashes()
            result["methods"][method] = [generate_one(backend, c, f"Question: {c['query']} Answer:", synchronize) for c in cases]
        except Exception as error:
            operation["seconds"] = time.perf_counter() - start
            operation["error"] = f"{type(error).__name__}: {error}"
            result["methods"][method] = [failure_case(c, "edit_error", operation["error"]) for c in cases]
        finally:
            try:
                if backend.patches.active is not None:
                    operation["rollback_metadata"] = backend.rollback()
                operation["restored_hashes"] = backend.weight_hashes()
                operation["exact_restore"] = operation["restored_hashes"] == before
                if not operation["exact_restore"]:
                    operation["restore_error"] = "Editable matrix hashes differ after restoration"
            except Exception as error:
                operation.update(exact_restore=False, restore_error=f"{type(error).__name__}: {error}")
        result["method_edits"][method] = operation
        if not operation["exact_restore"]:
            result["exact_restore"] = False
            result["restore_error"] = operation["restore_error"]
            break
    result["restored_hashes"] = backend.weight_hashes()
    for method in EDIT_METHODS:
        if method not in result["methods"]:
            result["methods"][method] = [failure_case(c, "not_executed", "Earlier restoration failure halted further edits") for c in cases]
    # Preserve the original MEND metadata aliases for existing consumers.
    mend = result["method_edits"].get("mend_context", {})
    result["edit_metadata"] = mend.get("metadata", {})
    result["edit_seconds"] = mend.get("seconds")
    result["fine_tuning_attempted"] = result["method_edits"].get("ft_last_layer", {}).get("attempted", False)
    return result


def placeholder(edit, assets, reason):
    cases = cases_for(edit, assets)
    return {"edit_id": edit["edit_id"], "provenance": "not_executed",
            "methods": {method: [failure_case(c, "not_executed", reason) for c in cases] for method in METHODS}}


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate(rows):
    if len(rows) != EDIT_COUNT:
        raise ValueError("Aggregation requires all20 units, including unexecuted failures")
    counts, latency, retention = {}, {}, {}
    for method in METHODS:
        counts[method], latency[method], retention[method] = {}, {}, {}
        for category in CATEGORIES:
            answers = [next(a for a in row["methods"][method] if a["category"] == category) for row in rows]
            times = [a["seconds"] for a in answers if a.get("seconds") is not None]
            counts[method][category] = {"correct": sum(a["strict_target_match"] for a in answers),
                "denominator": EDIT_COUNT, "accuracy": sum(a["strict_target_match"] for a in answers) / EDIT_COUNT,
                "abstentions": sum(a["status"] == "abstention" for a in answers),
                "errors_or_unexecuted": sum(a["status"] not in ("completed", "abstention") for a in answers)}
            latency[method][category] = {"sample_count": len(times), "p50_seconds": percentile(times, .5),
                                        "p95_seconds": percentile(times, .95)}
            if category in ("text_locality", "visual_locality"):
                baseline = [next(a for a in row["methods"]["base"] if a["category"] == category) for row in rows]
                valid = [(b, a) for b, a in zip(baseline, answers) if b["strict_target_match"]]
                preserved = sum(a["status"] == "completed" and a["answer"] == b["answer"] for b, a in valid)
                factually_retained = sum(a["strict_target_match"] for _, a in valid)
                retention[method][category] = {"baseline_correct_denominator": len(valid),
                    "baseline_correct_factual_retained": factually_retained,
                    "baseline_correct_factual_retention_rate": factually_retained / len(valid) if valid else None,
                    "baseline_correct_answer_preserved": preserved,
                    "baseline_correct_preservation_rate": preserved / len(valid) if valid else None,
                    "answer_preservation_comparison": "verbatim generated continuation; factual retention is separately normalized whole-answer correctness",
                    "unconditional_denominator": EDIT_COUNT,
                    "unconditional_answer_preserved": sum(a["status"] == "completed" and b["status"] == "completed" and a["answer"] == b["answer"] for b, a in zip(baseline, answers))}
    paired = []
    rng = random.Random(SEED)
    samples = [[rng.randrange(EDIT_COUNT) for _ in range(EDIT_COUNT)] for _ in range(10000)]
    for first, second in (("mend_context", "base"), ("mend_context", "prompt_only"), ("prompt_only", "base"),
                          ("ft_last_layer", "base"), ("ft_last_layer", "prompt_only"), ("mend_context", "ft_last_layer")):
        for category in CATEGORIES:
            delta = [int(next(a for a in r["methods"][first] if a["category"] == category)["strict_target_match"])
                     - int(next(a for a in r["methods"][second] if a["category"] == category)["strict_target_match"]) for r in rows]
            distribution = [sum(delta[i] for i in sample) / EDIT_COUNT for sample in samples]
            paired.append({"first": first, "second": second, "category": category,
                "difference": sum(delta) / EDIT_COUNT, "paired_edit_bootstrap_95_percent_interval": [percentile(distribution, .025), percentile(distribution, .975)],
                "resamples": 10000, "unit_count": EDIT_COUNT, "seed": SEED})
    edit_latency = {}
    for method in EDIT_METHODS:
        edit_times = [r.get("method_edits", {}).get(method, {}).get("seconds") for r in rows]
        edit_times = [t for t in edit_times if t is not None]
        edit_latency[method] = {"sample_count": len(edit_times), "p50_seconds": percentile(edit_times, .5),
                                "p95_seconds": percentile(edit_times, .95)}
    return {"provenance": "own_measured_with_unexecuted_failures_in_denominators",
            "scores": counts, "locality_preservation": retention, "query_latency": latency,
            "paired_differences": paired, "edit_latency": edit_latency,
            "latency_protocol": "CUDA synchronize before/after each call; zero excluded warmups; includes errors with recorded durations; loading/editing separate from query latency; p95 linear interpolation; no cross-category pooled latency"}


def read_rows(args, selection, seal_sha):
    rows = {}
    expected_edits = {e["edit_id"]: e for e in selection["edits"]}
    directory = args.output / "edits"
    for path in directory.glob("*.json") if directory.exists() else ():
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("seal_sha256") != seal_sha or row.get("edit_id") != path.stem or path.stem not in expected_edits:
            raise ValueError("Saved edit row does not belong to this frozen experiment")
        if set(row.get("methods", {})) != set(METHODS) or any(sorted(a["category"] for a in row["methods"][m]) != sorted(CATEGORIES) for m in METHODS):
            raise ValueError("Saved edit row has incomplete methods/categories")
        expected_cases = {c["category"]: c for c in cases_for(expected_edits[path.stem], args.assets)}
        for method in METHODS:
            for answer in row["methods"][method]:
                if any(answer.get(k) != v for k, v in expected_cases[answer["category"]].items()):
                    raise ValueError("Saved case differs from frozen query/image/ground truth")
                if not isinstance(answer.get("answer"), str) or answer.get("status") not in ("completed", "abstention", "generation_error", "edit_error", "not_executed"):
                    raise ValueError("Saved answer has invalid text/status")
                matched = answer["status"] == "completed" and exact_short_answer(answer["answer"], answer["expected"])
                if answer.get("strict_target_match") != matched:
                    raise ValueError("Saved score differs from fixed whole-answer scorer")
                duration = answer.get("seconds")
                if duration is not None and (not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0):
                    raise ValueError("Saved latency is not finite and nonnegative")
        if row.get("exact_restore") and row.get("base_hashes") != row.get("restored_hashes"):
            raise ValueError("Saved restoration claim disagrees with tensor hashes")
        rows[path.stem] = row
    return rows


def worker(args):
    backend, resources, torch_module = None, None, None
    report = {"execution_status": "failed", "meta_training_executed": False, "bounded_per_edit_fine_tuning_requested": True}
    code = 1
    try:
        import torch
        from .bridge import build_backend
        torch_module = torch
        receipt, seal_sha = checked_seal(args)
        if seal_sha != args.expected_seal_sha256:
            raise ValueError("Seal changed after controller check")
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        rows = read_rows(args, selection, seal_sha)
        if any(not r.get("exact_restore") for r in rows.values()):
            raise ValueError("A restoration failure closed this run; resuming would risk contamination")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        with Resources() as resources:
            start = time.perf_counter()
            backend = build_backend(args.upstream, args.assets, device=args.device)
            report["load_seconds"] = time.perf_counter() - start
            for edit in selection["edits"]:
                if edit["edit_id"] in rows:
                    continue
                row = evaluate_edit(backend, edit, args.assets, seal_sha, torch.cuda.synchronize)
                row_path = args.output / "edits" / (edit["edit_id"] + ".json")
                if row_path.exists():
                    raise FileExistsError("Refusing to replace an immutable completed edit row")
                write_json(row_path, row)
                rows[edit["edit_id"]] = row
                print(json.dumps({"event": "edit_finished", "completed_units": len(rows), "exact_restore": row["exact_restore"]}), flush=True)
                if not row["exact_restore"]:
                    raise RuntimeError("Exact restoration failed; halted subsequent edits")
        report.update(execution_status="completed", completed_edit_units=len(rows), resources=resources.summary(torch),
                      compatibility_adaptations=backend.compatibility)
        code = 0
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
                code = 1
        write_json(args.output / "worker.json", report)
    return code


def run(args):
    _, seal_sha = checked_seal(args)
    selection, _ = validate_selection(args)
    previous_attempts = list(args.output.glob("attempt-*.json"))
    if previous_attempts and not args.resume:
        raise ValueError("Previous attempt exists; use --resume with exactly the same seal/code")
    previous = read_rows(args, selection, seal_sha)
    if any(not r.get("exact_restore") for r in previous.values()):
        raise ValueError("Cannot resume after an exact-restoration failure")
    attempt = len(previous_attempts) + 1
    attempt_path = args.output / f"attempt-{attempt:03}.json"
    write_json(attempt_path, {"status": "started", "seal_sha256": seal_sha, "prior_completed_units": len(previous),
                              "incomplete_edit_policy": "A worker interrupted before durable rollback-checked row reruns that edit from fresh base; no completed row is replaced"})
    command = [sys.executable, "-m", "ke_repro.evaluate", "--_worker", "--output", str(args.output),
               "--upstream", str(args.upstream), "--assets", str(args.assets), "--manifest", str(args.manifest),
               "--selection", str(args.selection), "--device", args.device, "--timeout-seconds", str(args.timeout_seconds),
               "--expected-seal-sha256", seal_sha]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}
    timed_out = False
    with (args.output / f"attempt-{attempt:03}.log").open("w", encoding="utf-8") as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_seconds, env=environment).returncode
        except subprocess.TimeoutExpired:
            timed_out, code = True, None
    worker_path = args.output / "worker.json"
    worker_report = json.loads(worker_path.read_text(encoding="utf-8")) if worker_path.exists() else {}
    rows = read_rows(args, selection, seal_sha)
    complete = code == 0 and worker_report.get("execution_status") == "completed" and len(rows) == EDIT_COUNT and all(r.get("exact_restore") for r in rows.values())
    reason = "Controller timeout" if timed_out else "Worker did not complete this edit"
    units = [rows.get(e["edit_id"], placeholder(e, args.assets, reason)) for e in selection["edits"]]
    summary = {"execution_status": "completed" if complete else "timed_out" if timed_out else "failed",
               "worker_return_code": code, "worker_report": worker_report, "seal_sha256": seal_sha,
               "meta_training_executed": False, "bounded_per_edit_fine_tuning_requested": True,
               "fine_tuning_attempted_units": sum(r.get("fine_tuning_attempted", False) for r in rows.values()),
               "completed_edit_units": len(rows), "unexecuted_edit_units": EDIT_COUNT - len(rows),
               "all20_units_in_denominators": True, "aggregate": aggregate(units),
               "limitations": "Small nonrandom benchmark slice, automated label audit, pretrained published editor with context-alignment adaptation; prompt baseline lacks second reference image; no RAG comparison or arbitrary-edit guarantee"}
    write_json(args.output / "evaluation.json", summary)
    write_json(attempt_path, {"status": summary["execution_status"], "seal_sha256": seal_sha,
                              "worker_return_code": code, "completed_units": len(rows), "unexecuted_units": EDIT_COUNT-len(rows),
                              "prior_completed_units": len(previous), "worker_report": worker_report,
                              "incomplete_edit_policy": "Only interrupted incomplete units can rerun from fresh base; completed rows are preserved"})
    print(json.dumps({key: summary[key] for key in ("execution_status", "completed_edit_units", "unexecuted_edit_units", "seal_sha256")}, indent=2))
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--seal", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--upstream", type=Path, default=Path("tmp/vlkeb-upstream"))
    parser.add_argument("--assets", type=Path, default=Path("artifacts/vlkeb/assets"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/vlkeb-assets.json"))
    parser.add_argument("--selection", type=Path, default=Path("artifacts/vlkeb/assets/eval20-selection.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/vlkeb/eval20"))
    parser.add_argument("--device", choices=("cuda", "cuda:0"), default="cuda")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--expected-seal-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 1800:
        parser.error("Use a finite attempt cap from1 to1800 seconds")
    if args._worker and not args.expected_seal_sha256:
        parser.error("Private worker needs its expected seal checksum")
    for name in ("upstream", "assets", "manifest", "selection", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.fixture = None
    if not (args.seal or args.run or args._worker):
        print(json.dumps({"mode": "plan", "models_loaded": False, "network_requested": False,
            "edit_count": EDIT_COUNT, "cases_per_edit": 5, "methods": list(METHODS), "generated_question_instances": 400,
            "maximum_mend_updates": EDIT_COUNT, "maximum_fine_tuned_edits": EDIT_COUNT, "maximum_ft_optimizer_steps": EDIT_COUNT * 15,
            "meta_training_requested": False,
            "maximum_attempt_seconds": args.timeout_seconds, "runtime_estimate_seconds": None,
            "resource_estimate": "Requires measured fixed development profile before release; timeout is a cap",
            "required_inputs": ["verified model/assets", "pinned train/eval annotations", "prepared eval20 selection", "unchanged pre-inference seal"],
            "expected_artifacts": ["seal.json", "edits/<edit_id>.json", "worker.json", "attempt-*.log", "attempt-*.json", "evaluation.json"],
            "stopping_condition": "Exactly20 edit units, five cases per each of four methods; FT15steps per edit; halt on restoration failure; no sample expansion",
            "checkpoint_resume": "--resume preserves completed immutable edit rows under the exact same seal; incomplete edit restarts from fresh base; no persistent optimizer or meta-training state"}, indent=2))
        return 0
    try:
        if args.seal:
            print(json.dumps(seal(args), indent=2))
            return 0
        return worker(args) if args._worker else run(args)
    except Exception as error:
        print(json.dumps({"execution_status": "failed", "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
