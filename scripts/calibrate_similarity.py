"""Finite reviewed validation selection for the cosine-threshold editor baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from editguard.comparison import compatible
from editguard.formatting import INPUT_FORMAT_VERSION, correction_text
from editguard.neural import file_sha256
from scripts.prepare_data import REVIEW_POLICIES, canonical_bytes, content_hash, load_reviewed_rows
from scripts.review_dataset import checked_rows, read_json

VERSION = "editguard-similarity-validation-v1"
ROUTING_POLICY = "top_compatible_cosine_gte_threshold_v1"
POOLING = "masked_mean_l2"
EMBEDDING_MAX_TOKENS = 256
THRESHOLDS = tuple(index / 100 for index in range(-100, 101)) + (1.01,)
COSTS = {"false_application": 4.0, "missed_positive": 1.0, "clarification": .25, "correct": 0.0}


def _validated(labels, scores, eligibility=None):
    labels, scores = list(labels), list(scores)
    eligible = [True] * len(labels) if eligibility is None else list(eligibility)
    if not labels or len(labels) != len(scores) or len(labels) != len(eligible):
        raise ValueError("Selection requires nonempty equally sized labels, cosine scores and compatibility flags.")
    if any(type(label) is not int or label not in (0, 1) for label in labels) or set(labels) != {0, 1}:
        raise ValueError("Validation requires explicit integer positive and negative labels.")
    if any(type(flag) is not bool for flag in eligible):
        raise ValueError("Compatibility flags must be explicit booleans.")
    # Float32 normalized dot products can overshoot the mathematical endpoints
    # slightly. Keep the measured value unchanged instead of relabeling it.
    if any(isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
           or not -1.000001 <= score <= 1.000001 for score in scores):
        raise ValueError("Every pair requires a finite cosine score in [-1,1], with at most 1e-6 rounding overshoot.")
    return labels, [float(score) for score in scores], eligible


def _rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def operating_metrics(labels, scores, threshold, eligibility=None):
    """Binary apply/reject decisions, with no learned confidence/abstention band."""
    labels, scores, eligible = _validated(labels, scores, eligibility)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not -1 <= threshold <= 1.01:
        raise ValueError("Cosine threshold must be finite between -1 and the 1.01 never-apply sentinel.")
    tp = fp = tn = fn = 0
    for label, score, gate in zip(labels, scores, eligible):
        apply = gate and score >= threshold
        if apply and label:
            tp += 1
        elif apply:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    total, positive = len(labels), sum(labels)
    cost = COSTS["false_application"] * fp + COSTS["missed_positive"] * fn
    return {"threshold": float(threshold),
            "counts": {"total": total, "positive": positive, "negative": total - positive,
                       "true_application": tp, "false_application": fp, "true_rejection": tn,
                       "missed_positive": fn, "clarification": 0,
                       "incompatible_pairs": sum(not flag for flag in eligible),
                       "applicable_pairs_rejected_by_gate": sum(label and not gate for label, gate in zip(labels, eligible))},
            "total_cost": cost, "mean_cost": _rate(cost, total),
            "precision": _rate(tp, tp + fp), "recall": _rate(tp, positive),
            "false_application_rate": _rate(fp, total - positive), "pair_accuracy": _rate(tp + tn, total),
            "application_rate": _rate(tp + fp, total), "routing_decision_coverage": _rate(total, total),
            "clarification_rate": _rate(0, total), "generated_answer_coverage": None}


def fit_threshold(labels, scores, eligibility=None):
    """Choose among 202 finite thresholds using compatibility-gated row cost."""
    labels, scores, eligible = _validated(labels, scores, eligibility)
    candidates = [operating_metrics(labels, scores, threshold, eligible) for threshold in THRESHOLDS]
    selected = min(candidates, key=lambda item: (item["total_cost"], item["counts"]["false_application"],
                                                -item["routing_decision_coverage"]["numerator"], item["threshold"]))
    return {"threshold": selected["threshold"], "compatibility_gated_metrics": selected,
            "raw_pair_metrics": operating_metrics(labels, scores, selected["threshold"]),
            "candidates": candidates, "declared_costs": dict(COSTS), "rows": len(labels)}


def pair_eligible(row):
    """Use only inference selectors/correction metadata, never scoring truth."""
    context, correction = row["request_context"], row["correction"]
    for key in ("entity_id", "plan", "version"):
        if not isinstance(context.get(key), str) or not context[key].strip():
            raise ValueError("Request scope fields must be explicit nonempty strings.")
    for value in (context["as_of"], correction["effective_from"], correction.get("effective_to")):
        if value is not None and (not isinstance(value, str) or date.fromisoformat(value).isoformat() != value):
            raise ValueError("Scope dates must be canonical ISO dates.")
    if correction.get("effective_to") and correction["effective_to"] <= correction["effective_from"]:
        raise ValueError("Correction interval must have an exclusive end after its start.")
    approved = correction.get("status", correction.get("approval_status")) in ("approved", "approved_in_simulated_scenario")
    return approved and compatible(correction, context)


def relevant_config(config):
    if config.get("input_format_version") != INPUT_FORMAT_VERSION:
        raise ValueError("Inference input format differs from the runtime formatter.")
    model_id, revision = config.get("embedding_model_id"), config.get("embedding_revision")
    if not isinstance(model_id, str) or not model_id.strip() or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Embedding model identity must include a pinned 40-character revision.")
    if config.get("local_files_only") is not True or not isinstance(config.get("cache_dir"), str) or not config["cache_dir"].strip():
        raise ValueError("Similarity selection requires explicit cached-only model loading.")
    for name in ("retrieval_k", "prompt_edit_k"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError("Retrieval limits must be positive integers.")
    if config.get("device") not in ("cpu", "cuda"):
        raise ValueError("Inference device must be cpu or cuda.")
    return {"embedding_model_id": model_id, "embedding_revision": revision,
            "embedding_max_tokens": EMBEDDING_MAX_TOKENS, "pooling": POOLING,
            "input_format_version": INPUT_FORMAT_VERSION, "routing_policy": ROUTING_POLICY,
            "retrieval_k": config["retrieval_k"], "prompt_edit_k": config["prompt_edit_k"]}


def reviewed_inputs(config_path, dataset, reviews, *, max_rows, review_policy="human"):
    if type(max_rows) is not int or not 2 <= max_rows <= 10000:
        raise ValueError("max_rows must be an integer between 2 and 10000.")
    config = read_json(config_path)
    routing = relevant_config(config)
    manifest = read_json(dataset / "manifest.json")
    original = checked_rows(dataset, "scope", "validation")
    registry = read_json(reviews)  # Reject duplicate registry keys before the shared review gate.
    accepted = load_reviewed_rows(dataset / "scope_validation.jsonl", reviews, require_all=True, review_policy=review_policy)
    accepted_ids = {row["id"] for row in accepted}
    rows = [row for row in original if row["id"] in accepted_ids]
    if len(rows) > max_rows:
        raise ValueError("Accepted validation partition exceeds the declared row cap; no silent truncation.")
    for row in rows:
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError("Validation labels must be explicit integer zero or one.")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Validation questions must be nonempty strings.")
        if row.get("correction_text") != correction_text(row["correction"]):
            raise ValueError("Correction descriptors differ from the runtime input formatter.")
        pair_eligible(row)  # Validate exact scope and dates before model import.
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Accepted validation must contain both classes.")
    bindings = {**routing, "routing_config_sha256": content_hash(routing),
                "inference_config_sha256": file_sha256(config_path), "dataset_id": manifest["dataset_id"],
                "dataset_manifest_sha256": file_sha256(dataset / "manifest.json"),
                "validation_file_sha256": file_sha256(dataset / "scope_validation.jsonl"),
                "review_registry_sha256": file_sha256(reviews), "review_policy": review_policy,
                "human_reviewed_data": review_policy == "human", "review_audit_sha256": registry.get("audit_sha256"),
                "accepted_row_sha256": {row["id"]: content_hash(row) for row in rows},
                "source_validation_rows": len(original), "accepted_validation_rows": len(rows),
                "reviewed_rejected_rows": len(original) - len(rows), "max_rows": max_rows,
                "test_rows_read": False, "population": "single_focal_question_correction_pairs",
                "policy_source_sha256": file_sha256(ROOT / "editguard/comparison.py"),
                "embedding_source_sha256": file_sha256(ROOT / "editguard/neural.py"),
                "formatter_source_sha256": file_sha256(ROOT / "editguard/formatting.py")}
    return config, rows, bindings


def score_pairs(retriever, rows):
    """No labels or expected facts enter the embedding inference call."""
    records = []
    for row in rows:
        hits = retriever.rank(row["question"], [row["correction"]], correction_text, k=1)
        if len(hits) != 1:
            raise ValueError("Embedding baseline omitted a fixed validation pair.")
        question_tokens = len(retriever.tokenizer(row["question"])["input_ids"])
        correction_tokens = len(retriever.tokenizer(correction_text(row["correction"]))["input_ids"])
        records.append({"id": row["id"], "family_id": row["family_id"], "split": "validation", "label": row["label"],
                        "cosine_score": hits[0]["score"], "compatible": pair_eligible(row), "row_sha256": content_hash(row),
                        "question_tokens": question_tokens, "correction_tokens": correction_tokens,
                        "embedding_input_truncated": max(question_tokens, correction_tokens) > EMBEDDING_MAX_TOKENS})
    return records


def embedding_fingerprint(retriever):
    digest = hashlib.sha256()
    for name, tensor in sorted(retriever.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().view(retriever.torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def safe_output(output, dataset, reviews, config_path):
    target = output.resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory; calibration evidence is not overwritten.")
    source = dataset.resolve()
    if target == source or target in source.parents or source in target.parents:
        raise ValueError("Output must remain separate from the source dataset.")
    for path in (reviews.resolve(), config_path.resolve()):
        if target == path or target in path.parents:
            raise ValueError("Output must not contain or replace source review/configuration files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/inference.json")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--review-policy", choices=REVIEW_POLICIES, default="human",
                        help="Human review by default; automated_development requires an explicitly typed machine registry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Defaults to the declared inference device")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        safe_output(args.output, args.dataset, args.reviews, args.config)
        config, rows, bindings = reviewed_inputs(args.config, args.dataset, args.reviews, max_rows=args.max_rows, review_policy=args.review_policy)
        if args.preflight_only:
            print(json.dumps({"preflight_valid": True, "model_execution_requested": False, **bindings}, indent=2))
            return
        from editguard.neural import EmbeddingRetriever
        device = args.device or config["device"]
        retriever = EmbeddingRetriever(config["embedding_model_id"], config["embedding_revision"], device=device,
                                       local_files_only=True, cache_dir=config["cache_dir"])
        if any(parameter.requires_grad for parameter in retriever.model.parameters()):
            raise ValueError("Embedding baseline must be frozen.")
        before = embedding_fingerprint(retriever)
        started = perf_counter()
        records = score_pairs(retriever, rows)
        for record in records:
            record.update(review_policy=args.review_policy, human_reviewed_data=args.review_policy == "human")
        fit = fit_threshold([row["label"] for row in records], [row["cosine_score"] for row in records],
                            [row["compatible"] for row in records])
        after = embedding_fingerprint(retriever)
        if before != after:
            raise ValueError("Frozen embedding tensor state changed during selection.")
        current = {"inference_config_sha256": file_sha256(args.config),
                   "dataset_manifest_sha256": file_sha256(args.dataset / "manifest.json"),
                   "validation_file_sha256": file_sha256(args.dataset / "scope_validation.jsonl"),
                   "review_registry_sha256": file_sha256(args.reviews)}
        if any(bindings[key] != value for key, value in current.items()):
            raise ValueError("Selection inputs changed during inference; no frozen artifact published.")
        artifact = {"schema_version": 1, "artifact_type": "editguard_similarity_threshold", "calibration_version": VERSION,
                    "purpose": "similarity_validation_calibration", "split": "validation", "status": "frozen", **bindings,
                    "created_at": datetime.now(timezone.utc).isoformat(), "threshold": fit["threshold"],
                    "score_type": "cosine_similarity_not_probability", "uncertainty_band": None,
                    "declared_costs": fit["declared_costs"], "raw_pair_metrics": fit["raw_pair_metrics"],
                    "compatibility_gated_metrics": fit["compatibility_gated_metrics"],
                    "selection": {"objective": "minimum_compatibility_gated_per_row_cost", "threshold_grid": list(THRESHOLDS),
                                  "tie_break": "fewer_false_applications_then_higher_routing_coverage_then_lower_threshold"},
                    "embedding_state_sha256_before": before, "embedding_state_sha256_after": after,
                    "embedding_weights_unchanged": True, "embedding_device": device,
                    "tokenizer_sha256": hashlib.sha256(canonical_bytes({"vocabulary": retriever.tokenizer.get_vocab(),
                          "special_tokens": retriever.tokenizer.special_tokens_map, "padding_side": retriever.tokenizer.padding_side,
                          "truncation_side": retriever.tokenizer.truncation_side})).hexdigest(),
                    "embedding_truncated_pairs": sum(row["embedding_input_truncated"] for row in records),
                    "inference_and_selection_seconds": perf_counter() - started, "script_sha256": file_sha256(Path(__file__)),
                    "disclaimer": "Validation operating-point selection on single focal pairs. Cosine is not a probability; no ECE, full-memory retrieval recall or generated-answer accuracy claim. No test rows were read."}
        args.output.mkdir(parents=True)
        score_path, grid_path = args.output / "validation_cosine_scores.jsonl", args.output / "selection_grid.json"
        score_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
        grid_path.write_text(json.dumps({**fit, "review_policy": args.review_policy,
            "human_reviewed_data": args.review_policy == "human", "review_registry_sha256": bindings["review_registry_sha256"]}, indent=2) + "\n", encoding="utf-8")
        artifact.update(pair_scores_sha256=file_sha256(score_path), selection_grid_sha256=file_sha256(grid_path))
        temporary = args.output / "calibration.json.tmp"
        temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, args.output / "calibration.json")
        print(json.dumps({"output": str(args.output), "rows": len(rows), "threshold": artifact["threshold"],
                          "purpose": artifact["purpose"], "test_rows_read": False,
                          "review_policy": args.review_policy, "human_reviewed_data": args.review_policy == "human"}, indent=2))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
