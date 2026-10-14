"""Four optimizer steps on reviewed development data; no eligible trained model.

Measures the actual trainer's full validation and checkpoint I/O. Both the saved
auxiliary artifact and this run are profiling evidence, not an accuracy result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from editguard.formatting import INPUT_FORMAT_VERSION, conditional_input, correction_text
from editguard.scope_inputs import SCOPE_INPUT_POLICIES, scope_policy_identity
from editguard.training import LOSS_POLICIES, canonical_hash, file_hash, json_file_bytes, loss_objective, run_training
from scripts.prepare_data import load_reviewed_rows
from scripts.profile_models import hardware
from scripts.review_dataset import checked_rows, read_json

PROFILE_SOURCE_FILES = ("editguard/training.py", "scripts/train_auxiliary.py",
                        "scripts/preflight_training.py", "scripts/prepare_data.py",
                        "editguard/formatting.py", "editguard/scope_inputs.py")


def current_source_hashes():
    return {name: file_hash(ROOT / name) for name in PROFILE_SOURCE_FILES}


def validate_output_paths(dataset, reviews, output, report):
    dataset, reviews, output, report = (Path(path).resolve() for path in (dataset, reviews, output, report))
    if output.exists() or report.exists():
        raise ValueError("Use fresh profiling output paths; evidence is never overwritten.")
    for target in (output, report):
        if target == dataset or dataset in target.parents or target in dataset.parents:
            raise ValueError("Profile outputs must remain separate from data inputs.")
        if target == reviews or target in reviews.parents or reviews in target.parents:
            raise ValueError("Profile outputs must remain separate from review inputs.")
    if output == report or output in report.parents or report in output.parents:
        raise ValueError("Profile artifact and report need separate output paths.")


def reviewed_inputs(task, dataset, reviews, review_policy):
    if task not in ("scope", "answer") or review_policy not in ("human", "automated_development"):
        raise ValueError("Unsupported profile task or review policy.")
    manifest = read_json(dataset / "manifest.json")
    if manifest.get("input_format_version") != INPUT_FORMAT_VERSION:
        raise ValueError("Dataset differs from the current input format.")
    registry = read_json(reviews)
    if (registry.get("dataset_id") != manifest["dataset_id"]
            or registry.get("origin_manifest_sha256") != file_hash(dataset / "manifest.json")):
        raise ValueError("Review registry must bind the exact dataset and manifest.")
    populations = {}
    for split in ("train", "validation"):
        original = checked_rows(dataset, task, split)
        rows = load_reviewed_rows(dataset / f"{task}_{split}.jsonl", reviews, review_policy=review_policy)
        if len(original) > 2000:
            raise ValueError("Development profile is capped at 2000 rows per split.")
        for row in rows:
            if row.get("input_format_version") != INPUT_FORMAT_VERSION:
                raise ValueError("Row differs from the current input format.")
            if row.get("correction_text") != correction_text(row["correction"]):
                raise ValueError("Correction descriptor differs from runtime formatting.")
            if task == "answer" and row["input_text"] != conditional_input(
                    row["question"], row["correction"], row.get("supporting_context", []), row["request_context"]["as_of"]):
                raise ValueError("Conditional input differs from runtime formatting.")
        populations[split] = rows
    def identities(rows):
        families, entities, lineages = set(), set(), set()
        for row in rows:
            families.add(row["family_id"])
            if row.get("question_family_id"):
                families.add(row["question_family_id"])
            entities.update((row["correction"]["entity_id"], row["expected_fact"]["entity_id"]))
            lineages.add(row["lineage_id"])
            for support in row.get("supporting_context", []):
                entities.add(support["entity_id"])
                if support.get("family_id"):
                    families.add(support["family_id"])
        return families, entities, lineages
    if any(left & right for left, right in zip(identities(populations["train"]), identities(populations["validation"]))):
        raise ValueError("Profile partitions share an entity, family or lineage.")
    return manifest, populations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("scope", "answer"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--review-policy", choices=("human", "automated_development"), default="human")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--loss-policy", choices=LOSS_POLICIES, default="uniform")
    parser.add_argument("--scope-input-policy", choices=SCOPE_INPUT_POLICIES, default="raw_v1")
    args = parser.parse_args()
    try:
        output = args.output_dir.resolve()
        validate_output_paths(args.dataset, args.reviews, output, args.report)
        manifest, rows = reviewed_inputs(args.task, args.dataset, args.reviews, args.review_policy)
        objective = loss_objective(args.task, args.loss_policy)
        scope_identity = scope_policy_identity(args.scope_input_policy)
        if args.task != "scope" and args.scope_input_policy != "raw_v1":
            raise ValueError("A scope input adaptation cannot change answer inputs.")
        models = read_json(ROOT / "configs/models.json")
        revision = models["model_revisions"][args.task]
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Profile model revision must pin a 40-character commit.")
        card = {"schema_version": 1, "task": args.task, "purpose": "pilot_profile", "review_policy": args.review_policy,
                "question": "What are actual optimizer, full-validation and checkpoint costs at the fixed pilot shape?",
                "dataset_id": manifest["dataset_id"], "manifest_sha256": file_hash(args.dataset / "manifest.json"),
                "reviews_sha256": file_hash(args.reviews), "model_id": models[f"{args.task}_candidate"],
                "model_revision": models["model_revisions"][args.task], "output_dir": str(output),
                "device": args.device, "seed": 42, "batch_size": 2, "input_length": 192,
                "target_length": 48 if args.task == "answer" else None, "precision": "float32", "scheduler": "linear",
                "max_steps": 4, "checkpoint_every_steps": 2, "validation_every_steps": 2, "keep_last_checkpoints": 2,
                "learning_rate": 2e-5, "weight_decay": .01, "max_grad_norm": 1.0, "cpu_threads": 2}
        if args.loss_policy != "uniform":
            card["loss_policy"] = args.loss_policy
        if args.scope_input_policy != "raw_v1":
            card.update(scope_identity)
        inputs = {"manifest": card["manifest_sha256"], "reviews": card["reviews_sha256"],
                  "train": file_hash(args.dataset / f"{args.task}_train.jsonl"),
                  "validation": file_hash(args.dataset / f"{args.task}_validation.jsonl")}
        sources = current_source_hashes()
        script_hash = file_hash(Path(__file__))
        model_config_hash = file_hash(ROOT / "configs/models.json")
        measurement = hardware()
        started = perf_counter()
        trained = run_training(card, train_rows=rows["train"], validation_rows=rows["validation"], purpose="smoke_test",
                               run_card_sha256=hashlib.sha256(json_file_bytes(card)).hexdigest())
        wall_seconds = perf_counter() - started
        if inputs != {"manifest": file_hash(args.dataset / "manifest.json"), "reviews": file_hash(args.reviews),
                      "train": file_hash(args.dataset / f"{args.task}_train.jsonl"),
                      "validation": file_hash(args.dataset / f"{args.task}_validation.jsonl")}:
            raise ValueError("Review/profile inputs changed during the bounded run.")
        if sources != current_source_hashes() or script_hash != file_hash(Path(__file__)) or model_config_hash != file_hash(ROOT / "configs/models.json"):
            raise ValueError("Trainer/configuration sources changed during the bounded profile.")
        log = [json.loads(line) for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()]
        training = trained["training"]
        profile = {"component": args.task, "backbone": card["model_id"], "model_revision": card["model_revision"],
                   "device": args.device, "batch_size": 2, "input_length": 192, "target_length": card["target_length"],
                   "precision": "float32", "measured_optimizer_steps": 3,
                   "warmup_seconds": log[0]["optimizer_seconds"],
                   "median_step_seconds": statistics.median(row["optimizer_seconds"] for row in log[1:]),
                   "step_seconds": [row["optimizer_seconds"] for row in log[1:]],
                   "full_validation_measurements": training["validation_measurements_this_process"],
                   "checkpoint_measurements": training["checkpoint_measurements_this_process"],
                   "train_rows": len(rows["train"]), "validation_rows": len(rows["validation"]),
                   "whole_process_work_seconds": wall_seconds,
                   "load_tokenize_export_overhead_seconds": max(0, wall_seconds - training["elapsed_training_seconds"]),
                   "checkpoint_state_bytes": max((directory / "state.pt").stat().st_size for directory in (output / "checkpoints").glob("step-*")),
                   "export_bytes": sum(path.stat().st_size for name in ("model", "tokenizer") for path in (output / name).rglob("*") if path.is_file()),
                   "peak_allocated_gpu_gib": max(row["peak_allocated_gpu_bytes"] or 0 for row in log) / 2**30 if args.device == "cuda" else None,
                   "peak_reserved_gpu_gib": max(row["peak_reserved_gpu_bytes"] or 0 for row in log) / 2**30 if args.device == "cuda" else None,
                   "limitations": "Four updates; three steady samples and two full validations/checkpoint writes. Extrapolation is uncertain; no generated-answer accuracy, final test or full-run measurement."}
        if args.loss_policy != "uniform":
            profile.update(loss_policy=args.loss_policy, loss_objective=objective,
                tokenization=trained["tokenization"])
        if args.scope_input_policy != "raw_v1":
            profile.update(scope_identity)
            profile["tokenization"] = trained["tokenization"]
        result = {"schema_version": 1, "purpose": "actual_trainer_development_profile", "status": "profile_complete",
                  "hardware": measurement, "profile": profile, "dataset_id": manifest["dataset_id"],
                  "manifest_sha256": inputs["manifest"], "reviews_sha256": inputs["reviews"],
                  "review_policy": args.review_policy, "human_reviewed_data": args.review_policy == "human",
                  "profile_artifact_purpose": trained["purpose"], "runtime_eligible": False, "test_rows_read": False,
                  "profile_run_identity": canonical_hash(trained), "profile_script_sha256": script_hash,
                  "trainer_source_sha256": sources, "model_config_sha256": model_config_hash,
                  "environment": trained["environment"], "environment_sha256": canonical_hash(trained["environment"]),
                  "data_files_sha256": {name: inputs[name] for name in ("train", "validation")},
                  "input_format_version": INPUT_FORMAT_VERSION,
                  "recorded_at": trained["updated_at"]}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("xb") as destination:
            destination.write(json_file_bytes(result))
        print(json.dumps({"report": str(args.report), "profile": profile, "runtime_eligible": False}, indent=2))
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
