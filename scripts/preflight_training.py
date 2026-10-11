"""Validate a finite training run card without loading models or running training.

This preflight is not a training implementation. A successful result means the
declared inputs/profile/plan are internally consistent; it releases no GPU run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from scripts.prepare_data import load_reviewed_rows
except ModuleNotFoundError:
    from prepare_data import load_reviewed_rows


RESUME_FIELDS = {
    "model", "optimizer", "scheduler", "rng", "global_step",
    "data_hashes", "configuration_hash",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(card: dict, key: str) -> Path:
    value = card.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be an explicit absolute path.")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{key} must be an explicit absolute path.")
    return path.resolve()


def _integer(card: dict, key: str, maximum: int | None = None) -> int:
    value = card.get(key)
    if type(value) is not int or value <= 0 or (maximum and value > maximum):
        raise ValueError(f"{key} must be a positive integer" +
                         (f" no greater than {maximum}." if maximum else "."))
    return value


def _positive_float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and positive.")
    return float(value)


def _assert_hash(path: Path, expected: Any, field: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{field} must contain an exact SHA-256 digest.")
    if not path.is_file() or file_hash(path) != expected:
        raise ValueError(f"{field} does not match {path.name}.")


def validate_run_card(card_path: Path) -> dict:
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card.get("schema_version") != 1 or card.get("task") not in ("scope", "answer"):
        raise ValueError("Expected run card schema_version=1 and task=scope or answer.")
    task = card["task"]
    if not isinstance(card.get("question"), str) or not card["question"].strip():
        raise ValueError("Declare the concrete question this run resolves.")
    data_dir = _path(card, "data_dir")
    train_path = _path(card, "train_path")
    validation_path = _path(card, "validation_path")
    reviews_path = _path(card, "reviews_path")
    profile_path = _path(card, "profile_path")
    output_dir = _path(card, "output_dir")
    if train_path != data_dir / f"{task}_train.jsonl" or validation_path != data_dir / f"{task}_validation.jsonl":
        raise ValueError("Only this task's explicit train and validation files are allowed; no test input.")
    if output_dir == data_dir or output_dir in data_dir.parents:
        raise ValueError("Output directory must not contain or overwrite the input dataset.")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output_dir must refer to a directory.")
    manifest_path = data_dir / "manifest.json"
    _assert_hash(manifest_path, card.get("manifest_sha256"), "manifest_sha256")
    _assert_hash(reviews_path, card.get("reviews_sha256"), "reviews_sha256")
    _assert_hash(profile_path, card.get("profile_sha256"), "profile_sha256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if card.get("dataset_id") != manifest.get("dataset_id"):
        raise ValueError("Run-card dataset identity differs from the manifest.")
    for path in (train_path, validation_path):
        _assert_hash(path, manifest.get("files", {}).get(path.name), path.name)
    train_rows = load_reviewed_rows(train_path, reviews_path)
    validation_rows = load_reviewed_rows(validation_path, reviews_path)
    split_families, split_entities, split_lineages = {}, {}, {}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        families, entities, lineages = set(), set(), set()
        ids = set()
        for row in rows:
            if row.get("split") != split or row["id"] in ids:
                raise ValueError("Row split mismatch or duplicate ID in reviewed inputs.")
            ids.add(row["id"])
            families.add(row["family_id"])
            if row.get("question_family_id"):
                families.add(row["question_family_id"])
            lineages.add(row["lineage_id"])
            entities.update((row["correction"]["entity_id"], row["expected_fact"]["entity_id"]))
            if task == "scope" and row.get("label") not in (0, 1):
                raise ValueError("Scope labels must be binary.")
            if task == "answer" and not all(isinstance(row.get(key), str) and row[key].strip()
                                             for key in ("input_text", "target_text")):
                raise ValueError("Answer rows require nonempty input_text and target_text.")
        split_families[split], split_entities[split], split_lineages[split] = families, entities, lineages
    if any(populations["train"] & populations["validation"]
           for populations in (split_families, split_entities, split_lineages)):
        raise ValueError("Reviewed train/validation entity, family or lineage leakage.")

    profile_record = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = profile_record.get("profile")
    if profile_record.get("profile_error") or not isinstance(profile, dict):
        raise ValueError("A completed real-checkpoint optimizer profile is required.")
    revision = card.get("model_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model_revision must pin a resolved 40-character commit SHA.")
    if not isinstance(card.get("model_id"), str) or not card["model_id"].strip():
        raise ValueError("model_id is required.")
    batch_size = _integer(card, "batch_size")
    input_length = _integer(card, "input_length")
    target_length = _integer(card, "target_length") if task == "answer" else None
    if task == "scope" and card.get("target_length") is not None:
        raise ValueError("Scope target_length must be null.")
    matches = {"component": task, "backbone": card["model_id"], "model_revision": revision,
               "batch_size": batch_size, "input_length": input_length,
               "target_length": target_length, "precision": card.get("precision")}
    for field, expected in matches.items():
        if profile.get(field) != expected:
            raise ValueError(f"Run-card {field} differs from the measured profile.")
    if card.get("precision") != "float32":
        raise ValueError("Initial preflight supports the current float32 profiling path only.")
    _integer(profile, "measured_optimizer_steps")
    step_seconds = _positive_float(profile.get("median_step_seconds"), "profile median_step_seconds")
    max_steps = _integer(card, "max_steps", maximum=100_000)
    checkpoint_every = _integer(card, "checkpoint_every_steps")
    if checkpoint_every > max_steps:
        raise ValueError("A checkpoint must occur within the finite stopping cap.")
    _integer(card, "keep_last_checkpoints", maximum=10)
    _positive_float(card.get("learning_rate"), "learning_rate")
    if type(card.get("seed")) is not int:
        raise ValueError("An integer seed is required.")
    estimate = _positive_float(card.get("estimated_total_seconds"), "estimated_total_seconds")
    optimizer_estimate = max_steps * step_seconds
    if estimate < optimizer_estimate:
        raise ValueError("Total runtime estimate is below measured optimizer-step extrapolation.")
    if not isinstance(card.get("estimate_assumptions"), str) or not card["estimate_assumptions"].strip():
        raise ValueError("Declare runtime/memory estimate assumptions and excluded work.")
    if profile.get("device") == "cuda":
        measured_peak = _positive_float(profile.get("peak_allocated_gpu_gib"), "profile peak_allocated_gpu_gib")
        peak_estimate = _positive_float(card.get("estimated_peak_gpu_gib"), "estimated_peak_gpu_gib")
        if peak_estimate < measured_peak:
            raise ValueError("Estimated peak GPU memory is below the measured profile.")
        available = profile_record.get("hardware", {}).get("gpu_memory_gib")
        if isinstance(available, (int, float)) and peak_estimate > available:
            raise ValueError("Estimated peak memory exceeds the measured device capacity.")
    elif profile.get("device") != "cpu":
        raise ValueError("Profile must identify cpu or cuda execution.")

    checkpoint_plan = card.get("checkpoint_plan", {})
    if checkpoint_plan.get("resume_policy") != "exact_state_only":
        raise ValueError("Checkpoint resume_policy must be exact_state_only.")
    if not RESUME_FIELDS.issubset(set(checkpoint_plan.get("state_fields", []))):
        raise ValueError("Checkpoint plan omits model/optimizer/scheduler/RNG/step/data/config state.")
    artifacts = card.get("expected_artifacts")
    if not isinstance(artifacts, list) or not {"model", "training_log.jsonl", "run_card.json", "checkpoints"}.issubset(set(artifacts)):
        raise ValueError("Expected artifacts must include model, logs, run card and checkpoints.")
    resume_path = card.get("resume_checkpoint")
    if resume_path is not None:
        resume = _path(card, "resume_checkpoint")
        if output_dir not in resume.parents or not resume.is_dir():
            raise ValueError("Resume checkpoint must be an existing directory under output_dir.")
        metadata_path = resume / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("Resume metadata is required; exact-state restoration is not yet implemented.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for field in ("task", "model_id", "model_revision", "manifest_sha256", "reviews_sha256"):
            if metadata.get(field) != card[field]:
                raise ValueError(f"Resume metadata differs for {field}.")
        if type(metadata.get("global_step")) is not int or not 0 <= metadata["global_step"] < max_steps:
            raise ValueError("Resume global_step must precede the declared stopping cap.")

    return {"preflight_valid": True, "training_implementation_ready": False,
            "disclaimer": "No training ran. A verified training loop and complete run card are still required before releasing a substantial command.",
            "task": task, "dataset_id": card["dataset_id"], "model_id": card["model_id"],
            "model_revision": revision, "reviewed_train_rows": len(train_rows),
            "reviewed_validation_rows": len(validation_rows),
            "optimizer_step_extrapolation_seconds": optimizer_estimate,
            "declared_total_estimate_seconds": estimate,
            "profile_limitations": profile.get("limitations", "Not supplied by profile."),
            "output_dir": str(output_dir), "finite_stopping_condition": f"global_step reaches {max_steps}",
            "expected_artifacts": artifacts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_card", type=Path)
    args = parser.parse_args()
    try:
        result = validate_run_card(args.run_card)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
