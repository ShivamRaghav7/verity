"""Audit pilot train/validation token lengths without training or review changes.

Only cached pinned official tokenizers are loaded. No model weights, GPU
operations, optimizer, test rows, or human-review registry are accessed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.review_dataset import checked_rows, read_json
from editguard.formatting import INPUT_FORMAT_VERSION

TASKS = ("scope", "answer")
SPLITS = ("train", "validation")
INPUT_LIMITS = (192, 256)
TARGET_LIMIT = 48


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(lengths):
    if not lengths:
        raise ValueError("The audit requires a nonempty row population")
    ordered = sorted(lengths)

    def nearest_rank(fraction):
        return ordered[math.ceil(len(ordered) * fraction) - 1]

    return {"count": len(ordered), "min": ordered[0], "median": statistics.median(ordered),
            "p90": nearest_rank(.90), "p95": nearest_rank(.95), "p99": nearest_rank(.99),
            "max": ordered[-1], "mean": statistics.mean(ordered),
            "quantile_definition": "p90/p95/p99 use nearest rank ceil(p*n); median uses the arithmetic mean of central values for even n",
            "histogram_exact_lengths": {str(length): count for length, count in sorted(Counter(ordered).items())}}


def truncation_observations(lengths, limits):
    return {str(limit): {"rows_that_would_truncate": sum(length > limit for length in lengths),
                        "denominator": len(lengths),
                        "fraction": sum(length > limit for length in lengths) / len(lengths),
                        "excess_tokens_total": sum(max(0, length - limit) for length in lengths)}
            for limit in limits}


def _string(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row {row.get('id')} has an empty or invalid {key}")
    return value


def audit_dataset(data_dir, *, model_config_path, cache_dir, tasks=TASKS, splits=SPLITS):
    """Read only selected task train/validation files and measure exact inputs."""
    tasks, splits = tuple(tasks), tuple(splits)
    if not tasks or any(task not in TASKS for task in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError("Only unique scope/answer tasks are supported")
    if not splits or any(split not in SPLITS for split in splits) or len(set(splits)) != len(splits):
        raise ValueError("Only unique train/validation splits are supported; test is never read")
    data_dir = Path(data_dir).resolve()
    model_config_path, cache_dir = Path(model_config_path).resolve(), Path(cache_dir).resolve()
    manifest_path = data_dir / "manifest.json"
    manifest_hash = file_hash(manifest_path)
    manifest = read_json(manifest_path)
    if (manifest.get("purpose") != "development_pilot_not_final_evaluation"
            or manifest.get("test_files_included") is not False):
        raise ValueError("This audit requires a development-pilot manifest without test files")
    if not isinstance(manifest.get("dataset_id"), str) or not manifest["dataset_id"]:
        raise ValueError("Pilot manifest lacks a dataset identifier")
    dataset_input_version = manifest.get("input_format_version", "editguard-input-v1")
    if dataset_input_version not in ("editguard-input-v1", "editguard-input-v2"):
        raise ValueError("Pilot manifest uses an unknown input format")
    model_configuration = read_json(model_config_path)
    config_hash = file_hash(model_config_path)
    rows_by_task = {}
    data_hashes = {}
    for task in tasks:
        rows_by_task[task] = {}
        for split in splits:
            rows = checked_rows(data_dir, task, split)
            if not rows or manifest.get("counts", {}).get(task, {}).get(split) != len(rows):
                raise ValueError(f"{task}/{split} row count differs from the manifest or is empty")
            for row in rows:
                if not isinstance(row.get("id"), str) or not row["id"]:
                    raise ValueError("Every audited row needs a nonempty string identifier")
                if row.get("input_format_version", "editguard-input-v1") != dataset_input_version:
                    raise ValueError("Row input format differs from the pilot manifest")
                if task == "scope":
                    _string(row, "question")
                    _string(row, "correction_text")
                else:
                    _string(row, "input_text")
                    _string(row, "target_text")
            rows_by_task[task][split] = rows
            name = f"{task}_{split}.jsonl"
            data_hashes[name] = file_hash(data_dir / name)

    # Transformers may import Torch internally; this code never imports model
    # loader classes or touches a Torch device. Explicit cache/offline arguments
    # prevent downloads or a tokenizer revision drifting to a new branch tip.
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file

    report = {"schema_version": 1, "purpose": "developer_input_shape_audit",
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset_id": manifest["dataset_id"],
        "dataset_directory": str(data_dir), "manifest_sha256": manifest_hash,
        "data_files_sha256": data_hashes, "model_config_sha256": config_hash,
        "trainer_source_sha256": {name: file_hash(ROOT / name) for name in (
            "editguard/training.py", "scripts/train_auxiliary.py")},
        "audit_source_sha256": file_hash(Path(__file__)),
        "read_splits": list(splits), "test_read": False, "models_loaded": False,
        "dataset_input_format_version": dataset_input_version,
        "current_trainer_input_format_version": INPUT_FORMAT_VERSION,
        "matches_current_trainer_input_format": dataset_input_version == INPUT_FORMAT_VERSION,
        "gpu_used": False, "backpropagation": False, "reviews_read_or_changed": False,
        "labels_changed": False, "library_versions": {
            name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")},
        "input_limits_checked": list(INPUT_LIMITS), "answer_target_limit_checked": TARGET_LIMIT,
        "limitations": [
            "Generated pilot rows remain pending independent human review; lengths do not validate labels or answer correctness.",
            "Only current pilot train/validation inputs were counted; no test input, challenge row or review registry was read.",
            "Lengths include tokenizer special tokens with truncation=False and no padding; training pads to its declared input/target lengths.",
            "Observed finite candidate shapes require a new matching optimizer/resource profile before a substantial training command.",
            "Token counts provide no measured model throughput, training runtime, model memory, calibration or accuracy result.",
            "Unseen deployment wording may exceed these observed pilot lengths."
        ], "tasks": {}}
    for task in tasks:
        model_id = model_configuration.get(f"{task}_candidate")
        revision = model_configuration.get("model_revisions", {}).get(task)
        if not isinstance(model_id, str) or not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError(f"{task} tokenizer needs a model ID and a pinned 40-character revision")
        # Keep the lexical snapshot path: resolving a valid cache symlink would
        # point at blobs/ and erase the revision component on Unix caches.
        tokenizer_config = Path(cached_file(model_id, "tokenizer_config.json", revision=revision,
                                            cache_dir=str(cache_dir), local_files_only=True)).absolute()
        if revision not in tokenizer_config.parts:
            raise ValueError("Resolved tokenizer cache snapshot differs from the pinned revision")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                cache_dir=str(cache_dir), local_files_only=True)
        snapshot = tokenizer_config.parent
        tokenizer_files = {str(path.relative_to(snapshot)).replace("\\", "/"): file_hash(path)
            for path in sorted(snapshot.rglob("*")) if path.is_file() and path.name in (
                "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "vocab.txt",
                "spiece.model", "added_tokens.json")}
        task_report = {"model_id": model_id, "revision": revision,
            "tokenizer_class": type(tokenizer).__name__, "local_files_only": True,
            "tokenizer_files_sha256": tokenizer_files,
            "input_format_version": dataset_input_version,
            "input_format": "question_then_correction_pair" if task == "scope" else "dataset_input_text_v1",
            "exact_count_expression": "tokenizer(row['question'], row['correction_text'], truncation=False)['input_ids']" if task == "scope" else "tokenizer(row['input_text'], truncation=False)['input_ids']",
            "target_count_expression": "tokenizer(row['target_text'], truncation=False)['input_ids']" if task == "answer" else None,
            "splits": {}}
        all_inputs, all_targets = [], []
        for split, rows in rows_by_task[task].items():
            inputs, targets, observations = [], [], []
            for row in rows:
                tokens = tokenizer(row["question"], row["correction_text"], truncation=False)["input_ids"] if task == "scope" else tokenizer(row["input_text"], truncation=False)["input_ids"]
                input_length = len(tokens)
                target_length = len(tokenizer(row["target_text"], truncation=False)["input_ids"]) if task == "answer" else None
                inputs.append(input_length)
                if target_length is not None:
                    targets.append(target_length)
                observations.append({"row_id": row["id"], "input_tokens": input_length, "target_tokens": target_length})
            all_inputs.extend(inputs)
            all_targets.extend(targets)
            task_report["splits"][split] = {"rows": len(rows),
                "input_distribution": distribution(inputs), "input_truncation": truncation_observations(inputs, INPUT_LIMITS),
                "target_distribution": distribution(targets) if targets else None,
                "target_truncation": truncation_observations(targets, (TARGET_LIMIT,)) if targets else None,
                "row_observations": observations}
        proposed_input = next((limit for limit in INPUT_LIMITS if max(all_inputs) <= limit), None)
        proposed_target = TARGET_LIMIT if task == "answer" and max(all_targets) <= TARGET_LIMIT else None
        task_report["observed_candidate_shape"] = {"basis": "Smallest checked finite input shape containing every selected pilot input without truncation",
            "batch_size": 2, "input_tokens": proposed_input, "target_tokens": proposed_target,
            "all_inputs_fit": proposed_input is not None,
            "all_targets_fit": task == "scope" or proposed_target is not None,
            "requires_matching_optimizer_resource_profile": True,
            "trained_model_throughput_estimate": None}
        report["tasks"][task] = task_report

    unchanged = (file_hash(manifest_path) == manifest_hash and file_hash(model_config_path) == config_hash
        and all(file_hash(data_dir / name) == digest for name, digest in data_hashes.items()))
    if not unchanged:
        raise ValueError("Audit inputs changed during measurement; refusing the result")
    report["inputs_unchanged"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/generated/pilot-v2")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/models.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runtime/hf-cache/hub")
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--split", action="append", choices=SPLITS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new report path instead of overwriting")
    try:
        report = audit_dataset(args.data_dir, model_config_path=args.model_config,
            cache_dir=args.cache_dir, tasks=args.task or TASKS, splits=args.split or SPLITS)
        content = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also refuses a file created while the audit ran.
        with args.output.open("xb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps({"report": str(args.output.resolve()), "dataset_id": report["dataset_id"],
        "inputs_unchanged": report["inputs_unchanged"], "tasks": {
            task: {"row_counts": {split: result["rows"] for split, result in item["splits"].items()},
                   "observed_candidate_shape": item["observed_candidate_shape"]}
            for task, item in report["tasks"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
