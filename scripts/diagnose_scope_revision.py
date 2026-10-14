"""Read-only, bounded comparison of two scope models on the SAME pilot-v3 pairs.

No model fitting, threshold selection or test files. At most 720 forward passes:
two exported models times 240 train and 120 validation question/correction pairs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.comparison import compatible
from editguard.formatting import INPUT_FORMAT_VERSION, correction_text
from editguard.neural import calibrated_probability, file_sha256, validate_artifact
from scripts.calibrate_scope import operating_metrics
from scripts.prepare_data import content_hash, load_reviewed_rows
from scripts.review_dataset import checked_rows, integrity_flags, read_json


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def auroc(labels, probabilities):
    """Exact pairwise concordance, awarding half credit to probability ties."""
    positive = [p for y, p in zip(labels, probabilities) if y == 1]
    negative = [p for y, p in zip(labels, probabilities) if y == 0]
    return rate(sum((p > n) + .5 * (p == n) for p in positive for n in negative),
                len(positive) * len(negative))


def distribution(values):
    if not values:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None, "median": None}
    ordered = sorted(values)
    size = len(values)
    return {"count": size, "minimum": ordered[0], "maximum": ordered[-1],
            "mean": sum(values) / size,
            "median": (ordered[(size - 1) // 2] + ordered[size // 2]) / 2}


def metrics(records):
    if not records:
        raise ValueError("A diagnostic subset must contain at least one pair")
    labels = [r["label"] for r in records]
    probabilities = [r["raw_probability"] for r in records]
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("Labels must be explicit integer 0/1")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Every pair needs a finite probability in [0,1]")
    clipped = [min(max(p, 1e-15), 1 - 1e-15) for p in probabilities]
    confusion = Counter((y, int(p >= .5)) for y, p in zip(labels, probabilities))
    tp, fp, tn, fn = (confusion[key] for key in ((1, 1), (0, 1), (0, 0), (1, 0)))
    return {"rows": len(labels), "positive": sum(labels), "negative": len(labels) - sum(labels),
            "binary_negative_log_likelihood": -sum(y * math.log(p) + (1 - y) * math.log1p(-p)
                                                     for y, p in zip(labels, clipped)) / len(labels),
            "brier_score": sum((p - y) ** 2 for y, p in zip(labels, probabilities)) / len(labels),
            "auroc": auroc(labels, probabilities), "threshold_0_5": {
                "true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn,
                "accuracy": rate(tp + tn, len(labels)), "precision": rate(tp, tp + fp),
                "recall": rate(tp, tp + fn), "false_application_rate": rate(fp, fp + tn)},
            "positive_probability": distribution([p for y, p in zip(labels, probabilities) if y]),
            "negative_probability": distribution([p for y, p in zip(labels, probabilities) if not y])}


def apply_policy(records, calibration, *, gated=False):
    # This diagnostic operates on supplied pairs. The application additionally
    # resolves context, ranks retrieved candidates and considers ambiguity.
    labels = [r["label"] for r in records]
    scores = [calibrated_probability(r["raw_probability"], calibration["temperature"])
              for r in records]
    if not gated:
        return operating_metrics(labels, scores, calibration["threshold"], calibration["ambiguity_margin"])
    selected = []
    rejected = []
    for r, probability in zip(records, scores):
        (selected if r["compatible"] else rejected).append((r["label"], probability))
    result = operating_metrics([y for y, _ in selected], [p for _, p in selected],
                               calibration["threshold"], calibration["ambiguity_margin"])
    counts = result["counts"].copy()
    counts["total"] += len(rejected)
    counts["positive"] += sum(y for y, _ in rejected)
    counts["negative"] += sum(not y for y, _ in rejected)
    counts["missed_positive"] += sum(y for y, _ in rejected)
    counts["true_rejection"] += sum(not y for y, _ in rejected)
    clarification = counts["clarification_positive"] + counts["clarification_negative"]
    correct = counts["true_application"] + counts["true_rejection"]
    return {"counts": counts, "deterministic_rejected_pairs": len(rejected),
            "coverage": rate(counts["total"] - clarification, counts["total"]),
            "accuracy_including_clarification_as_unsuccessful": rate(correct, counts["total"]),
            "precision": rate(counts["true_application"], counts["true_application"] + counts["false_application"]),
            "recall_including_clarification_as_missed": rate(counts["true_application"], counts["positive"]),
            "false_application_rate": rate(counts["false_application"], counts["negative"]),
            "disclaimer": "Product/plan/version/date compatibility is deterministic. Relation is not gated. This is pair-level, not end-to-end routing."}


def summarize(records, policies):
    groups = {}
    for field in ("question_style", "negative_type", "relation", "family_id"):
        by_value = defaultdict(list)
        for record in records:
            by_value[str(record[field])].append(record)
        groups[field] = {key: metrics(value) for key, value in sorted(by_value.items())}
    return {"raw": metrics(records), "by": groups,
            "policies": {name: {"temperature": value["temperature"], "threshold": value["threshold"],
                "ambiguity_margin": value["ambiguity_margin"], "raw_pair": apply_policy(records, value),
                "compatibility_gated_pair": apply_policy(records, value, gated=True)}
                for name, value in policies.items()}}


def validate_partitions(partitions):
    """Reject cross-partition identity leakage; same-fact variants stay together."""
    overlaps = {}
    for field in ("family_id", "lineage_id", "edit_id"):
        overlap = {r[field] for r in partitions["train"]} & {r[field] for r in partitions["validation"]}
        overlaps[field] = sorted(overlap)
    for field in ("entity_id", "fact_id"):
        overlap = {r["expected_fact"][field] for r in partitions["train"]} & {
            r["expected_fact"][field] for r in partitions["validation"]}
        overlaps["queried_" + field] = sorted(overlap)
    overlap_inputs = {(r["question"], r["correction_text"]) for r in partitions["train"]} & {
        (r["question"], r["correction_text"]) for r in partitions["validation"]}
    overlaps["identical_question_correction_pairs"] = len(overlap_inputs)
    if any(overlaps.values()):
        raise ValueError("Pilot train/validation partition leakage")
    return overlaps


def read_inputs(dataset, origin, reviews):
    manifest = read_json(dataset / "manifest.json")
    old_manifest = read_json(origin / "manifest.json")
    if (manifest.get("pilot_version") != 3 or old_manifest.get("pilot_version") != 2
        or manifest.get("purpose") != "development_pilot_not_final_evaluation"
        or manifest.get("test_files_included") is not False
        or manifest.get("origin_hashes", {}).get("manifest.json") != file_sha256(origin / "manifest.json")):
        raise ValueError("Only the origin-bound pilot-v3 development partition is supported")
    catalogue = {fact["fact_id"]: fact for fact in read_json(dataset / "catalogue.json")}
    if file_sha256(dataset / "catalogue.json") != manifest["files"]["catalogue.json"]:
        raise ValueError("Catalogue checksum mismatch")
    derivations = {r["id"]: r for r in read_json(dataset / "derivation.json")}
    if file_sha256(dataset / "derivation.json") != manifest["files"]["derivation.json"]:
        raise ValueError("Derivation checksum mismatch")
    partitions, originals = {}, {}
    same_counts = {}
    for split, size in (("train", 240), ("validation", 120)):
        rows = checked_rows(dataset, "scope", split)
        accepted = load_reviewed_rows(dataset / f"scope_{split}.jsonl", reviews,
                                     require_all=True, review_policy="automated_development")
        if len(rows) != size or {r["id"] for r in rows} != {r["id"] for r in accepted}:
            raise ValueError("Diagnostic requires all reviewed rows from the fixed 360-pair pilot")
        original_rows = checked_rows(origin, "scope", split)
        originals[split] = {r["id"]: r for r in original_rows}
        changed = Counter()
        for row in rows:
            if integrity_flags(row, catalogue) or row["correction_text"] != correction_text(row["correction"]):
                raise ValueError("Structured scope label, ground truth or input-format integrity failed")
            if row.get("input_format_version") != INPUT_FORMAT_VERSION:
                raise ValueError("Input format mismatch")
            prior = originals[split][derivations[row["id"]]["source_row_id"]]
            if row["label"] != prior["label"] or row["correction_text"] != prior["correction_text"]:
                raise ValueError("Scope revision changed label or correction")
            identical = row["question"] == prior["question"]
            changed[("positive" if row["label"] else "negative") + ("_unchanged_question" if identical else "_changed_question")] += 1
            if row["label"] and not identical:
                raise ValueError("Positive question changed; paired comparison is ineligible")
        partitions[split] = rows
        same_counts[split] = dict(changed)
    return partitions, derivations, {"partition_overlaps": validate_partitions(partitions),
            "origin_comparison_counts": same_counts, "scope_pair_count": 360,
            "review_policy": "automated_development", "human_reviewed_data": False,
            "input_order": "question_then_correction", "positive_class_index": 1,
            "test_rows_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/generated/pilot-v3")
    parser.add_argument("--origin", type=Path, default=ROOT / "data/generated/pilot-v2")
    parser.add_argument("--reviews", type=Path, default=ROOT / "reports/pilot-v3-reviews.automated.json")
    parser.add_argument("--initial-model", type=Path, default=ROOT / "artifacts/scope/pilot-v2-initial")
    parser.add_argument("--revised-model", type=Path, default=ROOT / "artifacts/scope/pilot-v3-revision1")
    parser.add_argument("--initial-calibration", type=Path, default=ROOT / "reports/scope-pilot-v2/calibration.json")
    parser.add_argument("--revised-calibration", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory; existing evidence is immutable")
    for source in (args.dataset, args.origin, args.initial_model, args.revised_model):
        if args.output.resolve() == source.resolve() or source.resolve() in args.output.resolve().parents:
            parser.error("Output must be separate from dataset and model directories")
    partitions, derivations, checks = read_inputs(args.dataset, args.origin, args.reviews)
    policies = {"initial_v2_fixed": read_json(args.initial_calibration)}
    if args.revised_calibration:
        policies["revision1_validation_selected"] = read_json(args.revised_calibration)
    artifacts = {"initial_v2": args.initial_model, "revision1_v3": args.revised_model}
    manifests = {name: validate_artifact(path, "scope") for name, path in artifacts.items()}
    for name, path in artifacts.items():
        expected_dataset = args.origin if name == "initial_v2" else args.dataset
        if manifests[name]["manifest_sha256"] != file_sha256(expected_dataset / "manifest.json"):
            raise ValueError("Scope model dataset binding is invalid")
        if manifests[name].get("input_format") != "question_then_correction_pair":
            raise ValueError("Scope model input order differs from runtime")
    for policy_name, model_name in (("initial_v2_fixed", "initial_v2"), ("revision1_validation_selected", "revision1_v3")):
        if policy_name in policies and policies[policy_name]["scope_manifest_sha256"] != file_sha256(artifacts[model_name] / "manifest.json"):
            raise ValueError("Operating-policy/model binding mismatch")
    source_paths = [args.dataset / "manifest.json", args.origin / "manifest.json", args.reviews,
                    args.dataset / "catalogue.json", args.dataset / "derivation.json", args.initial_calibration,
                    ROOT / "editguard/training.py", ROOT / "editguard/neural.py", ROOT / "editguard/formatting.py",
                    *[path / "manifest.json" for path in artifacts.values()],
                    *[path / "training_log.jsonl" for path in artifacts.values()],
                    *[args.dataset / f"scope_{split}.jsonl" for split in partitions],
                    *[args.origin / f"scope_{split}.jsonl" for split in partitions]]
    if args.revised_calibration:
        source_paths.append(args.revised_calibration)
    source_hashes = {str(path.resolve().relative_to(ROOT)): file_sha256(path) for path in source_paths}
    from editguard.neural import ScopeCheckpoint
    import torch
    torch.set_num_threads(2)
    records, results, token_audits = [], {}, {}
    for name, path in artifacts.items():
        scope = ScopeCheckpoint(path, args.device)
        results[name], token_audits[name] = {}, {}
        for split, rows in partitions.items():
            lengths = []
            current = []
            for row in rows:
                length = len(scope.tokenizer(row["question"], row["correction_text"], truncation=False)["input_ids"])
                if length > scope.input_length:
                    raise ValueError("Diagnostic pair is truncated")
                lengths.append(length)
                record = {"model": name, "split": split, "id": row["id"], "row_sha256": content_hash(row),
                    "source_v2_row_id": derivations[row["id"]]["source_row_id"],
                    "label": row["label"], "question_style": derivations[row["id"]]["question_style"],
                    "negative_type": row["negative_type"] or "applicable_positive",
                    "relation": row["expected_fact"]["relation"], "family_id": row["family_id"],
                    "compatible": compatible(row["correction"], row["request_context"]),
                    "raw_probability": scope.score(row["question"], row["correction_text"]),
                    "input_tokens": length, "human_reviewed_data": False}
                current.append(record)
            records.extend(current)
            results[name][split] = summarize(current, policies)
            token_audits[name][split] = {"rows": len(rows), "maximum_input_tokens": max(lengths),
                "input_length": scope.input_length, "truncated_rows": 0}
            print(json.dumps({"model": name, "split": split, **results[name][split]["raw"]}), flush=True)
        del scope
        if args.device == "cuda":
            torch.cuda.empty_cache()
    current_hashes = {str(path.resolve().relative_to(ROOT)): file_sha256(path) for path in source_paths}
    if current_hashes != source_hashes:
        raise ValueError("Diagnostic inputs changed during execution")
    for name, path in artifacts.items():
        validate_artifact(path, "scope")
    logs = {}
    for name, path in artifacts.items():
        entries = [json.loads(line) for line in (path / "training_log.jsonl").read_text(encoding="utf-8").splitlines() if line]
        selected = manifests[name]["selection"]
        logs[name] = {"completed_steps": manifests[name]["completed_steps"], "selected_step": selected["global_step"],
            "selected_validation_loss": selected["value"],
            "validation_measurements": [{"step": r["global_step"], "validation_loss": r["validation_loss"]}
                for r in entries if r.get("validation_loss") is not None],
            "last_120_stochastic_training_loss_mean": sum(r["loss"] for r in entries[-120:]) / len(entries[-120:]),
            "training_loss_mean_disclaimer": "Training mode includes dropout and changing parameters; compare exported-model train/validation NLL instead.",
            "exported_validation_nll_minus_recorded_selection": results[name]["validation"]["raw"]["binary_negative_log_likelihood"] - selected["value"],
            "selection_validation_dataset": "v2" if name == "initial_v2" else "v3"}
    summary = {"schema_version": 1, "artifact_type": "editguard_scope_revision_diagnosis", "status": "completed",
        "purpose": "validation_development_diagnostic", "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": read_json(args.dataset / "manifest.json")["dataset_id"],
        "checks": checks, "token_audits": token_audits, "models": results, "training_logs": logs,
        "policies": {name: {key: value[key] for key in ("temperature", "threshold", "ambiguity_margin", "scope_manifest_sha256")}
                     for name, value in policies.items()},
        "sources_sha256": source_hashes, "script_sha256": file_sha256(Path(__file__)),
        "pair_forward_passes": len(records), "finite_cap": 720, "device": args.device,
        "limitations": ["Both models are evaluated on the same v3 pairs; v2 training used different negative wording.",
            "The initial model's recorded validation loss used v2; do not directly compare it with v3 selection loss.",
            "All positive questions and corrections are unchanged across v2/v3; the complete negative sets are not identical.",
            "Train metrics are an exported-model diagnostic, not generalization evidence.",
            "Only four train and two validation families; template-based automated development data, no human-review claim.",
            "The revised operating point, if supplied, was selected on these validation pairs; its metrics are optimistic.",
            "Compatibility gating is deterministic; pair diagnostics do not establish full retrieved routing performance.",
            "No new parameter search, optimizer updates or test/challenge access. No latency comparison is claimed."]}
    args.output.mkdir(parents=True)
    probability_path = args.output / "probabilities.jsonl"
    probability_path.write_text("".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records), encoding="utf-8")
    summary["probabilities_sha256"] = file_sha256(probability_path)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pair_forward_passes": len(records), "test_rows_read": False}), flush=True)


if __name__ == "__main__":
    main()
