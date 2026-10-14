"""Read-only paired scope diagnostic on the fixed, reviewed pilot-v3 rows.

Compare revision1/raw inputs and revision2/entity-neutral inputs. Exactly 720
forward passes: two trained exports times 240 train and 120 validation pairs.
No training, calibration selection, test/challenge access or data rewriting.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.comparison import compatible
from editguard.neural import ScopeCheckpoint, file_sha256, validate_artifact
from editguard.scope_inputs import scope_pair, validate_scope_policy
from editguard.training import scope_input_overlap, scope_training_pairs, state_digest
from scripts.diagnose_scope_revision import metrics, read_inputs
from scripts.prepare_data import content_hash, load_reviewed_rows
from scripts.review_dataset import read_json


def overlap_groups(partitions, policy="entity_neutral_v1"):
    """Label *both* models' validation records against the same neutral overlap.

    Shared inputs are selected without consulting labels or predictions. The
    shared helper additionally refuses contradictory labels after abstraction.
    """
    overlap = scope_input_overlap(partitions["train"], partitions["validation"], policy)
    train_pairs = set(scope_training_pairs(partitions["train"], policy))
    validation_pairs = scope_training_pairs(partitions["validation"], policy)
    labels = {row["id"]: "shared_with_train" if pair in train_pairs else "not_shared_with_train"
              for row, pair in zip(partitions["validation"], validation_pairs)}
    if len(labels) != len(partitions["validation"]):
        raise ValueError("Validation record IDs must be unique")
    return labels, overlap


def summarize(records):
    if not records:
        raise ValueError("Scope diagnostic requires a nonempty partition")
    groups = {}
    for field in ("question_style", "negative_type", "relation", "family_id", "neutral_overlap_group"):
        buckets = defaultdict(list)
        for record in records:
            buckets[str(record[field])].append(record)
        groups[field] = {name: metrics(rows) for name, rows in sorted(buckets.items())}
    return {"raw": metrics(records), "by": groups}


def checked_models(artifacts, dataset, reviews, partitions):
    manifests = {name: validate_artifact(path, "scope") for name, path in artifacts.items()}
    expected_policies = {"revision1_raw": "raw_v1", "revision2_neutral": "entity_neutral_v1"}
    manifest_hash, reviews_hash = file_sha256(dataset / "manifest.json"), file_sha256(reviews)
    dataset_id = read_json(dataset / "manifest.json")["dataset_id"]
    for name, manifest in manifests.items():
        if (manifest.get("dataset_id") != dataset_id or manifest.get("manifest_sha256") != manifest_hash
                or manifest.get("reviews_sha256") != reviews_hash
                or manifest.get("review_policy") != "automated_development"
                or manifest.get("human_reviewed_data") is not False
                or manifest.get("input_format") != "question_then_correction_pair"
                or manifest.get("completed_steps") != 360
                or manifest.get("selection", {}).get("metric") != "validation_loss"
                or validate_scope_policy(manifest) != expected_policies[name]):
            raise ValueError("Scope exports must bind the same reviewed pilot-v3 rows and expected input policies")
        for split, rows in partitions.items():
            if manifest.get("data_hashes", {}).get(split + "_rows") != content_hash(rows):
                raise ValueError("Reviewed scope rows differ from selected training export")
    neutral_overlap = scope_input_overlap(partitions["train"], partitions["validation"], "entity_neutral_v1")
    if manifests["revision2_neutral"].get("scope_input_overlap") != neutral_overlap:
        raise ValueError("Scope export's declared neutral input overlap differs from current transformation")
    for field in ("model_id", "model_revision", "pretrained_state_sha256", "input_length", "positive_class_index"):
        if manifests["revision1_raw"].get(field, 1 if field == "positive_class_index" else None) != manifests["revision2_neutral"].get(field, 1 if field == "positive_class_index" else None):
            raise ValueError("Paired scope comparison changes backbone, starting weights or input cap")
    return manifests


def source_snapshot(paths):
    return {path.resolve().relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/generated/pilot-v3")
    parser.add_argument("--origin", type=Path, default=ROOT / "data/generated/pilot-v2")
    parser.add_argument("--reviews", type=Path, default=ROOT / "reports/pilot-v3-reviews.automated.json")
    parser.add_argument("--raw-model", type=Path, default=ROOT / "artifacts/scope/pilot-v3-revision1")
    parser.add_argument("--neutral-model", type=Path, default=ROOT / "artifacts/scope/pilot-v3-neutral2")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory; existing evidence is immutable")
    for source in (args.dataset, args.origin, args.raw_model, args.neutral_model, args.reviews):
        if args.output.resolve() == source.resolve() or source.resolve() in args.output.resolve().parents:
            parser.error("Output must be separate from input data, reviews and model directories")
    # Existing reviewed integrity checks read only the two declared pilot splits.
    _, derivations, checks = read_inputs(args.dataset, args.origin, args.reviews)
    partitions = {split: load_reviewed_rows(args.dataset / f"scope_{split}.jsonl", args.reviews,
        require_all=True, review_policy="automated_development") for split in ("train", "validation")}
    if {split: len(rows) for split, rows in partitions.items()} != {"train": 240, "validation": 120}:
        raise ValueError("Exactly 240 reviewed training and 120 reviewed validation rows required")
    artifacts = {"revision1_raw": args.raw_model, "revision2_neutral": args.neutral_model}
    manifests = checked_models(artifacts, args.dataset, args.reviews, partitions)
    groups, neutral_overlap = overlap_groups(partitions)
    policy_overlaps = {name: scope_input_overlap(partitions["train"], partitions["validation"],
        validate_scope_policy(manifest)) for name, manifest in manifests.items()}
    source_paths = [args.dataset / "manifest.json", args.origin / "manifest.json", args.reviews,
        args.dataset / "catalogue.json", args.dataset / "derivation.json", Path(__file__),
        ROOT / "scripts/diagnose_scope_revision.py", ROOT / "scripts/prepare_data.py",
        ROOT / "scripts/review_dataset.py", ROOT / "editguard/training.py", ROOT / "editguard/neural.py",
        ROOT / "editguard/formatting.py", ROOT / "editguard/scope_inputs.py", ROOT / "editguard/comparison.py",
        *[path / name for path in (args.dataset, args.origin) for name in ("scope_train.jsonl", "scope_validation.jsonl")],
        *[path / name for path in artifacts.values() for name in ("manifest.json", "training_log.jsonl", "run_card.json")],
        *[path / name for key, path in artifacts.items() for name in manifests[key]["files_sha256"]]]
    before = source_snapshot(source_paths)
    os.environ.setdefault("HF_HOME", str(ROOT / "runtime/hf-cache"))
    import torch
    torch.set_num_threads(2)
    records, results, token_audits, state_checks = [], {}, {}, {}
    for name, path in artifacts.items():
        scope = ScopeCheckpoint(path, args.device)
        model_before = state_digest(scope.model.state_dict())
        if model_before != manifests[name]["selected_model_state_sha256"]:
            raise ValueError("Loaded scope parameters differ from selected exported state")
        results[name], token_audits[name] = {}, {}
        for split, rows in partitions.items():
            current, lengths = [], []
            for row in rows:
                transformed = scope_pair(row["question"], row["correction_text"], scope.scope_input_policy)
                length = len(scope.tokenizer(*transformed, truncation=False)["input_ids"])
                if length > scope.input_length:
                    raise ValueError("Diagnostic stops rather than truncating a scope pair")
                lengths.append(length)
                current.append({"model": name, "split": split, "id": row["id"], "reviewed_row_sha256": content_hash(row),
                    "source_v2_row_id": derivations[row["id"]]["source_row_id"], "label": row["label"],
                    "question_style": derivations[row["id"]]["question_style"],
                    "negative_type": row["negative_type"] or "applicable_positive",
                    "relation": row["expected_fact"]["relation"], "family_id": row["family_id"],
                    "neutral_overlap_group": groups[row["id"]] if split == "validation" else "train_diagnostic",
                    "scope_input_policy": scope.scope_input_policy, "transformed_pair_sha256": content_hash(transformed),
                    "compatible": compatible(row["correction"], row["request_context"]),
                    "raw_probability": scope.score(row["question"], row["correction_text"]),
                    "input_tokens": length, "human_reviewed_data": False})
            records.extend(current)
            results[name][split] = summarize(current)
            token_audits[name][split] = {"rows": len(rows), "maximum_input_tokens": max(lengths),
                "input_length": scope.input_length, "truncated_rows": 0}
            print(json.dumps({"model": name, "split": split, **results[name][split]["raw"]}), flush=True)
        model_after = state_digest(scope.model.state_dict())
        if model_after != model_before or scope.model.training or any(p.requires_grad for p in scope.model.parameters()):
            raise ValueError("Scope parameters changed or inference model was not frozen/evaluation mode")
        state_checks[name] = {"before": model_before, "after": model_after, "all_parameters_frozen": True,
            "evaluation_mode": True, "selected_export_match": True}
        del scope
        if args.device == "cuda":
            torch.cuda.empty_cache()
    if source_snapshot(source_paths) != before:
        raise ValueError("Original sources, model weights or raw reviewed inputs changed during diagnosis")
    for path in artifacts.values():
        validate_artifact(path, "scope")
    summary = {"schema_version": 1, "artifact_type": "editguard_scope_neutral_diagnosis", "status": "completed",
        "purpose": "validation_development_diagnostic", "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": read_json(args.dataset / "manifest.json")["dataset_id"], "checks": checks,
        "models": results, "model_states": state_checks, "token_audits": token_audits,
        "neutral_input_overlap": neutral_overlap, "input_overlap_by_model": policy_overlaps,
        "recorded_selections": {name: manifest["selection"] for name, manifest in manifests.items()},
        "exported_validation_nll_minus_recorded_selection": {name: results[name]["validation"]["raw"]["binary_negative_log_likelihood"] - manifest["selection"]["value"] for name, manifest in manifests.items()},
        "sources_sha256": before, "source_files_unchanged": True, "pair_forward_passes": len(records),
        "finite_cap": 720, "optimizer_updates": 0, "calibration_fits": 0, "device": args.device,
        "test_or_challenge_rows_read": False, "review_policy": "automated_development", "human_reviewed_data": False,
        "limitations": ["Entity-neutral input is our deterministic literal normalization adaptation; learned scores do not verify entity, plan, version, dates or provenance.",
            "Both models receive the same raw reviewed rows; only their declared scope input policy differs.",
            "Eight of 120 neutral validation inputs are identical to training inputs after abstraction. Their results are separated from the other 112; neither group proves arbitrary-language generalization.",
            "Only four train and two validation entity families, template-generated data and automated development review; no human-review claim.",
            "Raw threshold 0.5 metrics include no compatibility gates or calibrated uncertainty policy and are not full retrieved routing or generated-answer accuracy.",
            "Validation selected training checkpoints and will select calibration; these are development results, not a frozen final test.",
            "Train forwards are diagnostics rather than generalization evidence; no latency or resource-performance comparison is claimed."]}
    args.output.mkdir(parents=True)
    probabilities = args.output / "probabilities.jsonl"
    probabilities.write_text("".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records), encoding="utf-8")
    summary["probabilities_sha256"] = file_sha256(probabilities)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pair_forward_passes": len(records), "test_rows_read": False}), flush=True)


if __name__ == "__main__":
    main()
