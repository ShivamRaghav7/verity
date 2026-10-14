"""Compare two frozen answer models on the same reviewed development rows.

At most 288 teacher-forced examples: two models times 96 train + 48 validation.
No generated answers, optimizer updates, threshold selection or test access.
Both unweighted and fixed 8x comparison-first-token objectives are reported for
both models so unlike training losses are never compared as the same metric.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
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

PILOT_FILES = {"manifest.json", "scope_train.jsonl", "scope_validation.jsonl",
               "answer_train.jsonl", "answer_validation.jsonl", "catalogue.json",
               "derivation.json", "selection.json", "split_assignments.json"}
WEIGHTED_POLICY = "comparison_first_token_8x"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def first_weight(row):
    """Independent, metadata-checked reproduction of the fixed objective rule."""
    if row.get("reviewed") is not True or row.get("review_status") != "accepted":
        raise ValueError("Only reviewed accepted answer rows may be diagnosed")
    kind, polarity = row.get("question_type"), row.get("expected_polarity")
    if kind == "comparison":
        if polarity not in ("yes", "no"):
            raise ValueError("Comparison row needs an explicit yes/no polarity")
        if not any(row["target_text"].startswith(polarity.title() + punctuation)
                   for punctuation in (". ", ", ")):
            raise ValueError("Comparison target does not endorse its declared polarity")
        return 8
    if kind not in ("direct", "paraphrase") or polarity is not None:
        raise ValueError("Non-comparison row needs a supported form and no polarity")
    return 1


def grouped_summary(rows):
    if not rows:
        return {"rows": 0}
    token_count = sum(r["target_tokens"] for r in rows)
    weight_sum = sum(r["loss_weight_sum"] for r in rows)
    total = sum(r["total_nll"] for r in rows)
    weighted_total = sum(r["weighted_total_nll"] for r in rows)
    first_total = sum(r["first_token_nll"] for r in rows)
    return {"rows": len(rows), "target_tokens": token_count,
        "unweighted_nll_sum": total, "unweighted_token_nll": total / token_count,
        "comparison_first_token_8x_nll_sum": weighted_total,
        "comparison_first_token_8x_weight_sum": weight_sum,
        "comparison_first_token_8x_token_nll": weighted_total / weight_sum,
        "mean_first_token_nll": first_total / len(rows),
        "other_token_mean_nll": (total - first_total) / (token_count - len(rows))
                                if token_count != len(rows) else None,
        "first_token_top1_correct": rate(sum(r["first_token_top1_correct"] for r in rows), len(rows)),
        "mean_probability_yes_at_first_token": sum(r["first_probability_yes"] for r in rows) / len(rows),
        "mean_probability_no_at_first_token": sum(r["first_probability_no"] for r in rows) / len(rows),
        "first_top1_visible_counts": dict(Counter(r["first_top1_text"] for r in rows))}


def summaries(rows):
    return {"all": grouped_summary(rows), "subsets": {
        key: grouped_summary([r for r in rows if r["subset"] == key])
        for key in sorted({r["subset"] for r in rows})}, "relations": {
        key: grouped_summary([r for r in rows if r["relation"] == key])
        for key in sorted({r["relation"] for r in rows})}}


def selection_reproduction(manifest, validation_summary):
    """Choose the actual selection objective, never conflate unlike losses."""
    selection = manifest["selection"]
    policy = manifest.get("loss_objective", {}).get("policy", "uniform")
    if policy == "uniform" and selection["metric"] == "validation_loss":
        key = "unweighted_token_nll"
    elif policy == WEIGHTED_POLICY and selection["metric"] == "validation_weighted_token_loss":
        key = "comparison_first_token_8x_token_nll"
    else:
        raise ValueError("Selection metric and recorded loss policy do not agree")
    reproduced = validation_summary[key]
    if not math.isfinite(reproduced) or abs(reproduced - selection["value"]) > 1e-5:
        raise ValueError("Frozen validation objective differs from selected checkpoint loss")
    return {"metric": selection["metric"], "policy": policy,
        "recorded_value": selection["value"], "reproduced_value": reproduced,
        "difference": reproduced - selection["value"], "absolute_tolerance": 1e-5}


def paired_changes(initial, revision):
    old, new = ({r["id"]: r for r in rows} for rows in (initial, revision))
    if len(old) != len(initial) or len(new) != len(revision) or old.keys() != new.keys():
        raise ValueError("Paired diagnostics need unique identical row identifiers")
    for key in old:
        if any(old[key][field] != new[key][field] for field in
               ("partition", "edit_id", "subset", "target_token_ids", "first_token_weight")):
            raise ValueError("Paired conditional targets or objective weights differ")
    improved = sum(not old[key]["first_token_top1_correct"] and new[key]["first_token_top1_correct"] for key in old)
    regressed = sum(old[key]["first_token_top1_correct"] and not new[key]["first_token_top1_correct"] for key in old)
    return {"rows": len(old), "first_token_top1_improved": improved,
        "first_token_top1_regressed": regressed, "first_token_top1_unchanged": len(old) - improved - regressed,
        "mean_first_token_nll_difference_revision_minus_initial":
            sum(new[key]["first_token_nll"] - old[key]["first_token_nll"] for key in old) / len(old) if old else None,
        "free_running_answer_accuracy": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/generated/pilot-v2")
    parser.add_argument("--reviews", type=Path, default=ROOT / "reports/pilot-v2-reviews.automated.json")
    parser.add_argument("--initial", type=Path, default=ROOT / "artifacts/answer/pilot-v2-initial")
    parser.add_argument("--revision", type=Path, default=ROOT / "artifacts/answer/pilot-v2-polarity1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pilot-v2-answer-revision-diagnosis")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory; recorded evidence is immutable")
    if {p.name for p in args.dataset.iterdir()} != PILOT_FILES:
        parser.error("Only the exact development pilot without test/challenge files is accepted")
    for source in (args.dataset, args.initial, args.revision):
        if args.output.resolve() == source.resolve() or source.resolve() in args.output.resolve().parents:
            parser.error("Output must be separate from input data and models")
    manifests = {"initial": validate_artifact(args.initial, "answer"),
                 "revision": validate_artifact(args.revision, "answer")}
    dataset, registry = read_json(args.dataset / "manifest.json"), read_json(args.reviews)
    if (dataset.get("purpose") != "development_pilot_not_final_evaluation" or dataset.get("pilot_version") != 2
            or registry.get("review_policy") != "automated_development" or registry.get("human_reviewed_data") is not False):
        raise ValueError("Only the disclosed machine-reviewed pilot-v2 development data is accepted")
    for manifest in manifests.values():
        if (manifest["dataset_id"] != dataset["dataset_id"]
                or manifest["manifest_sha256"] != file_sha256(args.dataset / "manifest.json")
                or manifest["reviews_sha256"] != file_sha256(args.reviews)
                or manifest["review_policy"] != "automated_development" or manifest["human_reviewed_data"] is not False
                or manifest["completed_steps"] != 240 or manifest["max_steps"] != 240):
            raise ValueError("Completed model dataset, review or fixed step budget differs")
    for field in ("model_id", "model_revision", "tokenizer_hash", "pretrained_state_sha256",
                  "input_length", "target_length", "data_hashes", "input_format_version"):
        if manifests["initial"][field] != manifests["revision"][field]:
            raise ValueError("Matched model/data conditions differ: " + field)
    if (manifests["initial"].get("loss_objective", {}).get("policy", "uniform") != "uniform"
            or manifests["revision"].get("loss_objective", {}).get("policy") != WEIGHTED_POLICY):
        raise ValueError("Expected the initial uniform and fixed 8x revision objectives")
    partitions = {}
    for split, count in (("train", 96), ("validation", 48)):
        path = args.dataset / f"answer_{split}.jsonl"
        if file_sha256(path) != dataset["files"][path.name]:
            raise ValueError("Development answer file differs from dataset manifest")
        rows = load_reviewed_rows(path, args.reviews, require_all=True, review_policy="automated_development")
        if len(rows) != count or content_hash(rows) != manifests["initial"]["data_hashes"][split + "_rows"]:
            raise ValueError("Reviewed rows differ from matched trained artifacts")
        if len({r["id"] for r in rows}) != count:
            raise ValueError("Each reviewed row must have a unique identifier")
        for row in rows:
            first_weight(row)
        partitions[split] = rows
    for field in ("family_id", "lineage_id", "edit_id"):
        if {r[field] for r in partitions["train"]} & {r[field] for r in partitions["validation"]}:
            raise ValueError("Development train/validation identities overlap: " + field)
    source_paths = [args.dataset / name for name in sorted(PILOT_FILES)] + [args.reviews, Path(__file__),
        ROOT / "editguard/neural.py", ROOT / "editguard/formatting.py", ROOT / "scripts/prepare_data.py"]
    for model_root, manifest in ((args.initial, manifests["initial"]), (args.revision, manifests["revision"])):
        source_paths += [model_root / name for name in ("manifest.json", "training_log.jsonl", "run_card.json")]
        source_paths += [model_root / name for name in manifest["files_sha256"]]
    source_hashes = {str(p.resolve().relative_to(ROOT)).replace("\\", "/"): file_sha256(p) for p in source_paths}
    os.environ.setdefault("HF_HOME", str(ROOT / "runtime/hf-cache"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Configured CUDA device is unavailable")
    records, model_results = [], {}
    for name, model_root in (("initial", args.initial), ("revision", args.revision)):
        manifest = manifests[name]
        tokenizer = AutoTokenizer.from_pretrained(model_root / manifest["tokenizer_path"], local_files_only=True)
        if tokenizer.padding_side != "right":
            raise ValueError("Objective diagnosis requires right-padded targets")
        polarity_ids = {word: tokenizer(word, add_special_tokens=False)["input_ids"] for word in ("Yes", "No")}
        if any(len(ids) != 1 for ids in polarity_ids.values()):
            raise ValueError("Yes/No labels must be independent single-token labels")
        polarity_ids = {word: ids[0] for word, ids in polarity_ids.items()}
        model = AutoModelForSeq2SeqLM.from_pretrained(model_root / manifest["model_path"], local_files_only=True,
            dtype=torch.float32, attn_implementation="eager").to(args.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        def fingerprint():
            digest = hashlib.sha256()
            for key, tensor in sorted(model.state_dict().items()):
                digest.update(key.encode()); digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        before = fingerprint()
        if before != manifest["selected_model_state_sha256"]:
            raise ValueError("Loaded tensor state differs from selected checkpoint export")
        result, token_audits = {}, {}
        with torch.inference_mode():
            for partition, rows in partitions.items():
                inputs, targets = [r["input_text"] for r in rows], [r["target_text"] for r in rows]
                input_lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in inputs]
                target_lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in targets]
                if max(input_lengths) > manifest["input_length"] or max(target_lengths) > manifest["target_length"]:
                    raise ValueError("Conditional input or target would truncate; diagnosis stops")
                token_audits[partition] = {"rows": len(rows), "max_input_tokens": max(input_lengths),
                    "max_target_tokens": max(target_lengths), "input_truncated_rows": 0, "target_truncated_rows": 0}
                current = []
                for start in range(0, len(rows), 2):
                    encoded = tokenizer(inputs[start:start + 2], padding="max_length", max_length=manifest["input_length"],
                        truncation=False, return_tensors="pt").to(args.device)
                    labels = tokenizer(targets[start:start + 2], padding="max_length", max_length=manifest["target_length"],
                        truncation=False, return_tensors="pt")["input_ids"].to(args.device)
                    labels[labels == tokenizer.pad_token_id] = -100
                    logits = model(**encoded, labels=labels).logits.float()
                    losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=-100)
                    probabilities = logits[:, 0, :].softmax(dim=-1)
                    argmax = logits.argmax(dim=-1)
                    for local, row in enumerate(rows[start:start + 2]):
                        valid = labels[local] != -100
                        target_ids = labels[local][valid].cpu().tolist()
                        nlls, top1_ids = losses[local][valid].cpu().tolist(), argmax[local][valid].cpu().tolist()
                        weight = first_weight(row)
                        if weight == 8 and target_ids[0] != polarity_ids[row["expected_polarity"].title()]:
                            raise ValueError("Reviewed polarity does not match target token zero")
                        weighted_nll = sum(nlls) + (weight - 1) * nlls[0]
                        record = {"model": name, "partition": partition, "id": row["id"], "edit_id": row["edit_id"],
                            "family_id": row["family_id"], "relation": row["expected_fact"]["relation"],
                            "question_type": row["question_type"], "expected_polarity": row["expected_polarity"],
                            "subset": f"comparison_{row['expected_polarity']}" if weight == 8 else row["question_type"],
                            "input_tokens": input_lengths[start + local], "target_tokens": len(target_ids),
                            "target_token_ids": target_ids, "token_nll": nlls, "teacher_forced_top1_ids": top1_ids,
                            "first_token_weight": weight, "loss_weight_sum": len(target_ids) + weight - 1,
                            "first_token_nll": nlls[0], "total_nll": sum(nlls), "weighted_total_nll": weighted_nll,
                            "first_top1_text": tokenizer.decode([top1_ids[0]]),
                            "first_token_top1_correct": top1_ids[0] == target_ids[0],
                            "first_probability_yes": float(probabilities[local, polarity_ids["Yes"]].cpu()),
                            "first_probability_no": float(probabilities[local, polarity_ids["No"]].cpu()),
                            "human_reviewed_data": False}
                        current.append(record)
                records.extend(current)
                result[partition] = summaries(current)
        after = fingerprint()
        if before != after or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Frozen parameters changed or a parameter remained trainable")
        model_results[name] = {"model_manifest_sha256": file_sha256(model_root / "manifest.json"),
            "model_state_sha256_before": before, "model_state_sha256_after": after,
            "token_audits": token_audits, "yes_no_token_ids": polarity_ids, "partitions": result,
            "selection_reproduction": selection_reproduction(manifest, result["validation"]["all"])}
        print(json.dumps({"model": name, "validation": result["validation"],
                          "selection_reproduction": model_results[name]["selection_reproduction"]}, sort_keys=True), flush=True)
        validate_artifact(model_root, "answer")
        del model, tokenizer
        if args.device == "cuda":
            torch.cuda.empty_cache()
    if source_hashes != {str(p.resolve().relative_to(ROOT)).replace("\\", "/"): file_sha256(p) for p in source_paths}:
        raise ValueError("Source, dataset, review registry or model files changed during diagnosis")
    if len(records) != 288:
        raise ValueError("Teacher-forced finite cap or exact row count differs")
    pairs = {partition: paired_changes(
        [r for r in records if r["partition"] == partition and r["model"] == "initial"],
        [r for r in records if r["partition"] == partition and r["model"] == "revision"])
        for partition in partitions}
    summary = {"schema_version": 1, "artifact_type": "editguard_answer_revision_diagnosis",
        "purpose": "validation_development_diagnostic", "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset_id": dataset["dataset_id"],
        "review_policy": "automated_development", "human_reviewed_data": False,
        "teacher_forced_examples": len(records), "finite_cap": 288, "generation_calls": 0,
        "optimizer_updates": 0, "test_or_challenge_rows_read": False,
        "source_files_unchanged": True, "source_sha256": source_hashes, "device": args.device,
        "models": model_results, "paired_first_token_changes": pairs,
        "limitations": ["Development diagnosis with machine-reviewed controlled facts; no human-review or final-test claim.",
            "Teacher forcing provides preceding gold tokens and does not measure free-running complete-answer accuracy.",
            "First token has decoder start context only; later tokens benefit from gold prefixes.",
            "Each model has both losses reported on identical targets; the two recorded selection losses use different objectives.",
            "No generation, fitting, optimizer updates, test/challenge access or unseen arithmetic reliability claim."]}
    args.output.mkdir(parents=True)
    records_path = args.output / "teacher_forced_records.jsonl"
    records_path.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    summary["outputs_sha256"] = {records_path.name: file_sha256(records_path)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "teacher_forced_examples": len(records), "generation_calls": 0,
                      "test_or_challenge_rows_read": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
