"""Bounded frozen-model diagnosis of answer polarity on the reviewed development pilot.

At most 192 teacher-forced examples: 96 training + 48 validation + the already
recorded 48 oracle contexts. No generation, parameter updates or test access.
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

from editguard.neural import file_sha256, validate_artifact
from scripts.prepare_data import content_hash, load_reviewed_rows


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def grouped_summary(rows):
    if not rows:
        return {"rows": 0}
    total_tokens = sum(row["target_tokens"] for row in rows)
    first_sum = sum(row["first_token_nll"] for row in rows)
    total_sum = sum(row["total_nll"] for row in rows)
    return {"rows": len(rows), "target_tokens": total_tokens,
        "token_weighted_mean_nll": total_sum / total_tokens,
        "mean_first_token_nll": first_sum / len(rows),
        "other_token_mean_nll": (total_sum - first_sum) / (total_tokens - len(rows)),
        "first_token_fraction_of_tokens": len(rows) / total_tokens,
        "first_token_fraction_of_total_nll": first_sum / total_sum if total_sum else None,
        "first_token_top1_correct": rate(sum(row["first_token_top1_correct"] for row in rows), len(rows)),
        "mean_probability_yes_at_first_token": sum(row["first_probability_yes"] for row in rows) / len(rows),
        "mean_probability_no_at_first_token": sum(row["first_probability_no"] for row in rows) / len(rows),
        "first_top1_visible_counts": dict(Counter(row["first_top1_text"] for row in rows)),
        "recorded_generated_first_token_correct": rate(
            sum(row["recorded_generated_first_token_correct"] is True for row in rows),
            sum(row["recorded_generated_first_token_correct"] is not None for row in rows))}


def summaries(rows):
    return {"all": grouped_summary(rows), "subsets": {
        key: grouped_summary([row for row in rows if row["subset"] == key])
        for key in sorted({row["subset"] for row in rows})}, "relations": {
        relation: grouped_summary([row for row in rows if row["relation"] == relation])
        for relation in sorted({row["relation"] for row in rows})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/generated/pilot-v2")
    parser.add_argument("--reviews", type=Path, default=ROOT / "reports/pilot-v2-reviews.automated.json")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/answer/pilot-v2-initial")
    parser.add_argument("--recorded-evaluation", type=Path, default=ROOT / "reports/pilot-v2-validation-initial")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pilot-v2-answer-polarity")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory; recorded evidence is immutable")
    if {path.name for path in args.dataset.iterdir()} != {
            "manifest.json", "scope_train.jsonl", "scope_validation.jsonl", "answer_train.jsonl", "answer_validation.jsonl",
            "catalogue.json", "derivation.json", "selection.json", "split_assignments.json"}:
        parser.error("Only the exact development pilot directory without test/challenge files is accepted")
    for source in (args.dataset, args.model, args.recorded_evaluation):
        if args.output.resolve() == source.resolve() or source.resolve() in args.output.resolve().parents:
            parser.error("Output must be separate from source data, model and recorded evaluation")
    manifest = validate_artifact(args.model, "answer")
    dataset = read_json(args.dataset / "manifest.json")
    registry = read_json(args.reviews)
    evaluation = read_json(args.recorded_evaluation / "manifest.json")
    if (dataset.get("purpose") != "development_pilot_not_final_evaluation" or dataset.get("pilot_version") != 2
            or manifest["dataset_id"] != dataset["dataset_id"]
            or manifest["manifest_sha256"] != file_sha256(args.dataset / "manifest.json")
            or manifest["reviews_sha256"] != file_sha256(args.reviews)
            or registry["review_policy"] != "automated_development" or registry["human_reviewed_data"] is not False
            or evaluation["purpose"] != "validation_development" or evaluation["status"] != "completed"
            or evaluation["test_rows_read"] is not False or evaluation["human_reviewed_data"] is not False):
        raise ValueError("Development dataset, completed model or machine-review identity differs")
    partitions = {}
    for split, count in (("train", 96), ("validation", 48)):
        path = args.dataset / f"answer_{split}.jsonl"
        if file_sha256(path) != dataset["files"][path.name]:
            raise ValueError("Development answer data differs from manifest")
        rows = load_reviewed_rows(path, args.reviews, require_all=True, review_policy="automated_development")
        if len(rows) != count or content_hash(rows) != manifest["data_hashes"][split + "_rows"]:
            raise ValueError("Exact reviewed training/validation rows differ from trained artifact")
        partitions[split] = rows
    if {row["family_id"] for row in partitions["train"]} & {row["family_id"] for row in partitions["validation"]}:
        raise ValueError("Development entity-family splits overlap")
    oracles = read_rows(args.recorded_evaluation / "oracle_predictions.jsonl")
    if len(oracles) != 48:
        raise ValueError("Exactly 48 recorded oracle outputs are required")
    matched = {(row["edit_id"], row["question"]): row for row in partitions["validation"]}
    if len(matched) != 48:
        raise ValueError("Validation questions are not uniquely bound to edits")
    oracle_partition, context_comparisons = [], []
    for record in oracles:
        row = matched[(record["selected_edit_id"], record["question"])]
        if (record["status"] != "answered" or record["request_context"] != row["request_context"]
                or record["expected_polarity"] != row["expected_polarity"]
                or record["generation_trace"]["artifact_manifest_sha256"] != file_sha256(args.model / "manifest.json")
                or any(record["expected_fact"][key] != value for key, value in row["expected_fact"].items())):
            raise ValueError("Recorded oracle result is not bound to the matching reviewed answer target")
        oracle_partition.append({**row, "input_text": record["input_text"], "recorded_answer": record["answer"],
                                 "recorded_oracle_id": record["id"]})
        context_comparisons.append({"id": row["id"], "recorded_oracle_id": record["id"],
            "input_text_equal": row["input_text"] == record["input_text"],
            "dataset_support_fact_ids": [fact["fact_id"] for fact in row["supporting_context"]],
            "oracle_support_fact_ids": [fact["fact_id"] for fact in record["supporting_context"]]})
    partitions["oracle_context"] = oracle_partition
    source_paths = [args.dataset / "manifest.json", args.reviews,
                    args.dataset / "answer_train.jsonl", args.dataset / "answer_validation.jsonl",
                    args.recorded_evaluation / "manifest.json", args.recorded_evaluation / "oracle_predictions.jsonl",
                    args.model / "manifest.json", args.model / "training_log.jsonl", args.model / "run_card.json",
                    ROOT / "editguard/training.py", ROOT / "editguard/neural.py", ROOT / "editguard/formatting.py",
                    ROOT / "scripts/prepare_data.py", Path(__file__),
                    *[args.model / name for name in manifest["files_sha256"]]]
    source_hashes = {str(path.resolve().relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in source_paths}
    os.environ.setdefault("HF_HOME", str(ROOT / "runtime/hf-cache"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model / manifest["tokenizer_path"], local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model / manifest["model_path"], local_files_only=True,
                                               dtype=torch.float32, attn_implementation="eager").to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def fingerprint():
        digest = __import__("hashlib").sha256()
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    model_before = fingerprint()
    if model_before != manifest["selected_model_state_sha256"]:
        raise ValueError("Loaded model-state digest differs from selected training export")
    polarity_ids = {word: tokenizer(word, add_special_tokens=False)["input_ids"] for word in ("Yes", "No")}
    if any(len(value) != 1 for value in polarity_ids.values()):
        raise ValueError("Polarity diagnosis requires Yes/No to be single-token labels")
    polarity_ids = {key: value[0] for key, value in polarity_ids.items()}
    records, token_audits, results = [], {}, {}
    with torch.inference_mode():
        for partition, rows in partitions.items():
            inputs, targets = [row["input_text"] for row in rows], [row["target_text"] for row in rows]
            input_lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in inputs]
            target_lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in targets]
            if max(input_lengths) > manifest["input_length"] or max(target_lengths) > manifest["target_length"]:
                raise ValueError("A conditional input/target would truncate; diagnostic stops")
            token_audits[partition] = {"rows": len(rows), "max_input_tokens": max(input_lengths),
                "max_target_tokens": max(target_lengths), "input_cap": manifest["input_length"],
                "target_cap": manifest["target_length"], "input_truncated_rows": 0, "target_truncated_rows": 0}
            current = []
            for start in range(0, len(rows), 2):
                encoded = tokenizer(inputs[start:start + 2], padding="max_length", max_length=manifest["input_length"],
                                    truncation=False, return_tensors="pt").to(args.device)
                labels = tokenizer(targets[start:start + 2], padding="max_length", max_length=manifest["target_length"],
                                   truncation=False, return_tensors="pt")["input_ids"].to(args.device)
                labels[labels == tokenizer.pad_token_id] = -100
                prediction = model(**encoded, labels=labels)
                logits = prediction.logits.float()
                losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=-100)
                probabilities = logits[:, 0, :].softmax(dim=-1)
                argmax = logits.argmax(dim=-1)
                for local, row in enumerate(rows[start:start + 2]):
                    index = start + local
                    valid = labels[local] != -100
                    target_ids = labels[local][valid].cpu().tolist()
                    nlls = losses[local][valid].cpu().tolist()
                    top1_ids = argmax[local][valid].cpu().tolist()
                    generated_ids = tokenizer(row["recorded_answer"], add_special_tokens=False)["input_ids"] if row.get("recorded_answer") else []
                    polarity = row["expected_polarity"]
                    subset = f"comparison_{polarity}" if row["question_type"] == "comparison" else row["question_type"]
                    record = {"partition": partition, "id": row["id"], "edit_id": row["edit_id"],
                        "family_id": row["family_id"], "relation": row["expected_fact"]["relation"],
                        "question_type": row["question_type"], "subset": subset, "expected_polarity": polarity,
                        "input_tokens": input_lengths[index], "target_tokens": len(target_ids), "target_token_ids": target_ids,
                        "token_nll": nlls, "teacher_forced_top1_ids": top1_ids,
                        "first_token_nll": nlls[0], "total_nll": sum(nlls), "mean_token_nll": sum(nlls) / len(nlls),
                        "other_token_mean_nll": sum(nlls[1:]) / (len(nlls) - 1),
                        "first_target_token_id": target_ids[0], "first_target_text": tokenizer.decode([target_ids[0]]),
                        "first_top1_id": top1_ids[0], "first_top1_text": tokenizer.decode([top1_ids[0]]),
                        "first_token_top1_correct": top1_ids[0] == target_ids[0],
                        "first_probability_yes": float(probabilities[local, polarity_ids["Yes"]].cpu()),
                        "first_probability_no": float(probabilities[local, polarity_ids["No"]].cpu()),
                        "recorded_generated_first_token_id_reencoded": generated_ids[0] if generated_ids else None,
                        "recorded_generated_first_text_reencoded": tokenizer.decode([generated_ids[0]]) if generated_ids else None,
                        "recorded_generated_first_token_correct": generated_ids[0] == target_ids[0] if generated_ids else None,
                        "recorded_generated_answer": row.get("recorded_answer"),
                        "recorded_oracle_id": row.get("recorded_oracle_id"), "human_reviewed_data": False}
                    current.append(record)
            records.extend(current)
            results[partition] = summaries(current)
            print(json.dumps({"partition": partition, **results[partition]}, sort_keys=True), flush=True)
    model_after = fingerprint()
    if model_after != model_before or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Model parameters changed or were not frozen")
    if source_hashes != {str(path.resolve().relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in source_paths}:
        raise ValueError("Original evidence, data, code or model bytes changed during diagnosis")
    validate_artifact(args.model, "answer")
    paired = defaultdict(dict)
    for record in records:
        if record["partition"] in ("validation", "oracle_context"):
            paired[record["id"]][record["partition"]] = record
    context_effects = []
    for context in context_comparisons:
        original, oracle = paired[context["id"]]["validation"], paired[context["id"]]["oracle_context"]
        context_effects.append({**context, "subset": original["subset"],
            "dataset_first_token_top1_correct": original["first_token_top1_correct"],
            "oracle_first_token_top1_correct": oracle["first_token_top1_correct"],
            "first_token_nll_difference_oracle_minus_dataset": oracle["first_token_nll"] - original["first_token_nll"],
            "mean_token_nll_difference_oracle_minus_dataset": oracle["mean_token_nll"] - original["mean_token_nll"]})
    summary = {"schema_version": 1, "artifact_type": "editguard_answer_polarity_diagnosis",
        "purpose": "validation_development_diagnostic", "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset_id": dataset["dataset_id"],
        "review_policy": "automated_development", "human_reviewed_data": False,
        "teacher_forced_examples": len(records), "finite_cap": 192, "generation_calls": 0, "optimizer_updates": 0,
        "test_or_challenge_rows_read": False, "model_state_sha256_before": model_before,
        "model_state_sha256_after": model_after, "source_files_unchanged": True, "source_sha256": source_hashes,
        "yes_no_token_ids": polarity_ids, "device": args.device, "token_audits": token_audits,
        "partitions": results, "context_inputs_identical": sum(row["input_text_equal"] for row in context_comparisons),
        "context_inputs_changed": sum(not row["input_text_equal"] for row in context_comparisons),
        "recorded_selected_validation_loss": manifest["selection"]["value"],
        "reproduced_validation_loss_difference": results["validation"]["all"]["token_weighted_mean_nll"] - manifest["selection"]["value"],
        "limitations": ["Post-hoc development diagnosis, not a pre-registered final test or independently human-reviewed evaluation.",
            "Teacher forcing provides preceding gold target tokens; it does not measure free-running full-answer accuracy.",
            "First-token decisions have only decoder start context and are directly comparable to greedy first choices, subject to numerical differences.",
            "The copied fact continuation follows gold prefixes in this analysis; its low loss cannot excuse a wrong Yes/No decision.",
            "Recorded generated token IDs are re-encoded visible answer strings; original generation token arrays were not retained.",
            "Context changes are observational paired forwards, not proof of a single causal mechanism or unseen arithmetic reliability.",
            "No generation, updates, new data, timing comparison or final-test access."]}
    args.output.mkdir(parents=True)
    records_path = args.output / "teacher_forced_records.jsonl"
    records_path.write_text("".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records), encoding="utf-8")
    context_path = args.output / "context_comparison.jsonl"
    context_path.write_text("".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in context_effects), encoding="utf-8")
    summary["outputs_sha256"] = {path.name: file_sha256(path) for path in (records_path, context_path)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "teacher_forced_examples": len(records),
                      "generation_calls": 0, "test_rows_read": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
