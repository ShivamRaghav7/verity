"""Prepare two finite pilot run cards from reviewed inputs and actual profiles."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from editguard.formatting import INPUT_FORMAT_VERSION
from editguard.training import canonical_hash, json_file_bytes
from scripts.preflight_training import RESUME_FIELDS, file_hash, validate_run_card
from scripts.prepare_data import load_reviewed_rows
from scripts.profile_pilot_training import current_source_hashes, reviewed_inputs
from scripts.review_dataset import checked_rows, read_json


def make_card(task, dataset, reviews, profile_path, output, review_policy):
    if task not in ("scope", "answer") or review_policy not in ("human", "automated_development"):
        raise ValueError("Unsupported planning task or review policy.")
    dataset, reviews, profile_path, output = (Path(path).resolve() for path in (dataset, reviews, profile_path, output))
    if (output == dataset or dataset in output.parents or output in dataset.parents
            or output == reviews or reviews in output.parents or output in reviews.parents):
        raise ValueError("Trained artifact output must remain separate from dataset/review inputs.")
    manifest, populations = reviewed_inputs(task, dataset, reviews, review_policy)
    record = read_json(profile_path)
    profile = record.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("A complete profile object is required for run planning.")
    if (record.get("purpose") != "actual_trainer_development_profile" or record.get("runtime_eligible") is not False
            or record.get("status") != "profile_complete" or record.get("profile_error")
            or record.get("profile_artifact_purpose") != "smoke_test"
            or record.get("review_policy") != review_policy or record.get("test_rows_read") is not False
            or record.get("human_reviewed_data") is not (review_policy == "human")
            or record.get("dataset_id") != manifest["dataset_id"]
            or record.get("manifest_sha256") != file_hash(dataset / "manifest.json")
            or record.get("reviews_sha256") != file_hash(reviews)):
        raise ValueError("Run planning requires an actual profile bound to the exact reviewed pilot.")
    if (record.get("input_format_version") != INPUT_FORMAT_VERSION
            or record.get("trainer_source_sha256") != current_source_hashes()
            or record.get("profile_script_sha256") != file_hash(ROOT / "scripts/profile_pilot_training.py")
            or record.get("model_config_sha256") != file_hash(ROOT / "configs/models.json")):
        raise ValueError("Profile input-format, trainer or configuration source identity differs.")
    environment = record.get("environment")
    if not isinstance(environment, dict) or record.get("environment_sha256") != canonical_hash(environment):
        raise ValueError("Profile lacks a valid measured environment identity.")
    models = read_json(ROOT / "configs/models.json")
    if (profile.get("component") != task or profile.get("backbone") != models[f"{task}_candidate"]
            or profile.get("model_revision") != models["model_revisions"][task]
            or not re.fullmatch(r"[0-9a-f]{40}", profile.get("model_revision", ""))
            or type(profile.get("batch_size")) is not int or profile.get("batch_size") != 2
            or type(profile.get("input_length")) is not int or profile.get("input_length") != 192
            or profile.get("target_length") != (48 if task == "answer" else None)
            or profile.get("precision") != "float32" or profile.get("device") not in ("cpu", "cuda")
            or environment.get("device") != profile["device"]):
        raise ValueError("Profile task, model, device or fixed trainer shape differs.")
    for split in ("train", "validation"):
        if (record.get("data_files_sha256", {}).get(split) != file_hash(dataset / f"{task}_{split}.jsonl")
                or type(profile.get(f"{split}_rows")) is not int
                or len(populations[split]) != profile.get(f"{split}_rows")):
            raise ValueError("Reviewed populations differ from the measured profile.")
    def positive(value, name, allow_zero=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or not allow_zero and value == 0):
            raise ValueError(f"Profile {name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")
        return value
    steps_observed = profile.get("step_seconds")
    if (profile.get("measured_optimizer_steps") != 3 or not isinstance(steps_observed, list) or len(steps_observed) != 3):
        raise ValueError("Planning requires exactly three steady optimizer measurements.")
    for sample in steps_observed:
        positive(sample, "step_seconds")
    positive(profile.get("median_step_seconds"), "median_step_seconds")
    if sorted(steps_observed)[1] != profile["median_step_seconds"]:
        raise ValueError("Profile median differs from its observed optimizer samples.")
    batches_per_pass = math.ceil(profile["train_rows"] / profile["batch_size"])
    passes = 3 if task == "scope" else 5
    steps = batches_per_pass * passes
    validations = profile.get("full_validation_measurements", [])
    checkpoints = profile.get("checkpoint_measurements", [])
    if (not isinstance(validations, list) or not isinstance(checkpoints, list)
            or not all(isinstance(item, dict) for item in validations + checkpoints)
            or [item.get("global_step") for item in validations] != [2, 4]
            or [item.get("global_step") for item in checkpoints] != [2, 4]
            or any(item.get("rows") != profile["validation_rows"] for item in validations)):
        raise ValueError("Profile needs full validation/checkpoint measurements at steps2 and4.")
    for sample in validations + checkpoints:
        positive(sample.get("seconds"), "validation/checkpoint seconds")
    optimizer = steps * max(steps_observed)
    validation = passes * max(item["seconds"] for item in validations)
    checkpoint = passes * max(item["seconds"] for item in checkpoints)
    overhead = positive(profile.get("load_tokenize_export_overhead_seconds"), "load/tokenize/export overhead", allow_zero=True)
    raw = optimizer + validation + checkpoint + overhead
    estimate = math.ceil(2 * raw)
    gpu = None
    if profile["device"] == "cuda":
        gpu = math.ceil(max(positive(profile.get("peak_allocated_gpu_gib"), "allocated GPU memory"),
                            positive(profile.get("peak_reserved_gpu_gib"), "reserved GPU memory")) * 1.25 * 4) / 4
        if gpu > positive(record.get("hardware", {}).get("gpu_memory_gib"), "device capacity"):
            raise ValueError("Buffered GPU memory estimate exceeds the measured device capacity.")
    checkpoint_bytes = positive(profile.get("checkpoint_state_bytes"), "checkpoint bytes")
    export_bytes = positive(profile.get("export_bytes"), "export bytes")
    if type(checkpoint_bytes) is not int or type(export_bytes) is not int:
        raise ValueError("Profile checkpoint/export byte counts must be measured integers.")
    # Last two + a distinct older best + one in-progress checkpoint; exports
    # coexist with retained states. This is a conditional planning budget.
    disk_budget = math.ceil((4 * checkpoint_bytes + export_bytes) / 2**30 * 1.25)
    card = {"schema_version": 1, "task": task, "purpose": "initial_development_pilot_training",
            "question": ("Can a compact cross-encoder learn applicability for held-out pilot entities and reduce false applications relative to the embedding baseline?"
                         if task == "scope" else "Can a compact seq2seq model use corrections for held-out pilot entities while preserving subject, relation, value and comparison polarity under oracle routing?"),
            "dataset_id": manifest["dataset_id"], "review_policy": review_policy,
            "data_dir": str(dataset.resolve()), "train_path": str((dataset / f"{task}_train.jsonl").resolve()),
            "validation_path": str((dataset / f"{task}_validation.jsonl").resolve()), "reviews_path": str(reviews.resolve()),
            "profile_path": str(profile_path.resolve()), "output_dir": str(output.resolve()),
            "manifest_sha256": file_hash(dataset / "manifest.json"), "reviews_sha256": file_hash(reviews),
            "profile_sha256": file_hash(profile_path), "model_id": profile["backbone"], "model_revision": profile["model_revision"],
            "profile_environment_sha256": record["environment_sha256"],
            "profile_source_sha256": record["trainer_source_sha256"],
            "device": profile["device"], "seed": 42, "batch_size": profile["batch_size"],
            "input_length": profile["input_length"], "target_length": profile["target_length"], "precision": "float32",
            "scheduler": "linear", "max_steps": steps, "passes_over_accepted_train_rows": passes,
            "checkpoint_every_steps": batches_per_pass, "validation_every_steps": batches_per_pass,
            "keep_last_checkpoints": 2, "learning_rate": 2e-5, "weight_decay": .01, "max_grad_norm": 1.0, "cpu_threads": 2,
            "estimated_total_seconds": estimate, "estimated_peak_gpu_gib": gpu,
            "estimate_breakdown_seconds": {"optimizer_upper_observed_step_extrapolation": optimizer,
                 "full_validation": validation, "checkpoint_writes_and_pruning": checkpoint,
                 "load_tokenize_export_overhead": overhead, "unbuffered_sum": raw, "planning_multiplier": 2},
            "estimate_assumptions": "Conditional planning estimate from three steady optimizer samples and two full validation/checkpoint writes at this exact pilot shape; 2x time buffer and 25% reserved-memory margin rounded up to0.25GiB. Same GPU/software and data assumed; thermal throttling, OneDrive/disk variation and competing processes remain uncertain. Not a runtime or memory guarantee.",
            "checkpoint_plan": {"resume_policy": "exact_state_only", "state_fields": sorted(RESUME_FIELDS),
                                "selection": "lowest validation loss; earlier step wins equal loss",
                                "disk_budget_gib": disk_budget, "resume_instruction": "Reuse this unchanged run card with --resume <completed checkpoint directory>."},
            "expected_artifacts": ["model", "tokenizer", "manifest.json", "training_log.jsonl", "run_card.json", "checkpoints"],
            "resume_checkpoint": None,
            "limitations": ["Automated development review is disclosed and cannot establish independent human label quality." if review_policy == "automated_development" else "Human review is procedural, not an identity proof.",
                            "Only six entity families and twelve shared question forms; no final-test or unrestricted-language claim.",
                            "No early accuracy guarantee. Stop at the fixed cap; evaluate validation and document failures before any targeted revision."]}
    return card


def publish_cards(cards, output):
    """Validate both complete cards before atomically publishing their directory."""
    output = Path(output).resolve()
    if set(cards) != {"scope", "answer"} or output.exists():
        raise ValueError("Publish both tasks together into a fresh run-card directory.")
    for card in cards.values():
        dataset, reviews = Path(card["data_dir"]).resolve(), Path(card["reviews_path"]).resolve()
        if (output == dataset or output in dataset.parents or dataset in output.parents
                or output == reviews or output in reviews.parents or reviews in output.parents):
            raise ValueError("Run-card publication must remain separate from input artifacts.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.pending-{uuid4().hex}")
    staging.mkdir()
    try:
        for task, card in cards.items():
            with (staging / f"{task}.json").open("xb") as destination:
                destination.write(json_file_bytes(card))
                destination.flush()
                os.fsync(destination.fileno())
        results = {task: validate_run_card(staging / f"{task}.json") for task in cards}
        if output.exists():
            raise ValueError("Run-card output appeared during preflight; refusing overwrite.")
        os.rename(staging, output)
    finally:
        if staging.exists():
            resolved = staging.resolve()
            if resolved.parent != output.parent.resolve() or resolved.name != staging.name:
                raise ValueError("Staging cleanup target escaped the intended output parent.")
            shutil.rmtree(resolved)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--review-policy", choices=("human", "automated_development"), default="human")
    parser.add_argument("--scope-profile", type=Path, required=True)
    parser.add_argument("--answer-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Use a fresh run-card output directory.")
        cards = {task: make_card(task, args.dataset, args.reviews, profile, ROOT / f"artifacts/{task}/pilot-v2-initial", args.review_policy)
                 for task, profile in (("scope", args.scope_profile), ("answer", args.answer_profile))}
        results = publish_cards(cards, args.output)
        print(json.dumps({"preflight": results, "training_execution_requested": False}, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
