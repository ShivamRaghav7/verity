"""Audit the two completed targeted revisions using saved bytes and stdlib only.

This does not load Torch, train, generate answers, or open frozen test rows.
Restricted checkpoint descriptors are reused from the previous independent audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_scope_revision import (FIELDS, IDENTITY, canonical_hash,
    checkpoint_descriptors, file_hash, read_json, read_rows, state_digest, tensor_bytes)
from editguard.scope_inputs import scope_pair

BASE_IDENTITY = (*IDENTITY, "loss_objective")
SCOPE_IDENTITY = ("scope_input_policy", "scope_input_policy_version", "scope_input_source_sha256")
ARCHIVE_NAMES = ("manifest.json", "run_card.json", "training_log.jsonl", "latest_checkpoint.json")
DATA_NAMES = {"manifest.json", "scope_train.jsonl", "scope_validation.jsonl",
              "answer_train.jsonl", "answer_validation.jsonl", "catalogue.json",
              "derivation.json", "selection.json", "split_assignments.json"}


def expected_objective(task):
    if task == "scope":
        return {"schema_version": 1, "policy": "uniform", "comparison_first_token_weight": 1,
                "normalization": "original_model_loss",
                "rule": "Use model.loss unchanged; scope validation weights rows, answer validation weights valid target tokens."}
    if task == "answer":
        return {"schema_version": 1, "policy": "comparison_first_token_8x", "comparison_first_token_weight": 8,
                "normalization": "sum_weighted_token_nll_over_sum_weights",
                "rule": "first target token of reviewed comparison yes/no rows has weight 8; other valid tokens 1; padding 0"}
    raise ValueError("Only the declared scope and answer targeted runs may be audited")


def safe_relative(root, name):
    path = root / name
    if path.resolve() == root.resolve() or root.resolve() not in path.resolve().parents:
        raise ValueError("Audit source path escaped its declared root")
    return path


def tokenizer_record(artifact, task):
    """Reproduce tokenizer.get_vocab/special_tokens_map from their saved JSON."""
    tokenizer = read_json(artifact / "tokenizer/tokenizer.json")
    config = read_json(artifact / "tokenizer/tokenizer_config.json")
    if task == "scope":
        vocabulary = dict(tokenizer["model"]["vocab"])
        special_keys = ("unk_token", "sep_token", "pad_token", "cls_token", "mask_token")
    elif task == "answer":
        vocabulary = {item[0]: index for index, item in enumerate(tokenizer["model"]["vocab"])}
        special_keys = ("eos_token", "unk_token", "pad_token")
    else:
        raise ValueError("Unknown task for tokenizer identity")
    vocabulary.update({item["content"]: item["id"] for item in tokenizer["added_tokens"]})
    # Transformers 5's saved T5 extra_special_tokens are in the vocabulary but
    # are not members of special_tokens_map; their full file is hashed too.
    return {"vocabulary": vocabulary, "special_tokens": {key: config[key] for key in special_keys},
            "padding_side": config.get("padding_side", "right"),
            "truncation_side": config.get("truncation_side", "right")}


def selected_export_comparison(handle, safe, safe_base, state, archive, prefix, task, check):
    """Check every checkpoint tensor, including T5 embeddings omitted by export."""
    keys = set(safe) - {"__metadata__"}
    omitted = set(state) - keys
    aliases = {}
    if task == "answer":
        aliases = {"encoder.embed_tokens.weight": "shared.weight",
                   "decoder.embed_tokens.weight": "shared.weight"}
        check("selected_export_only_expected_shared_keys_omitted", omitted == set(aliases))
    else:
        check("selected_export_no_keys_omitted", not omitted)
    check("selected_export_no_extra_keys", not (keys - set(state)))
    for name, tensor in state.items():
        exported_name = aliases.get(name, name)
        descriptor = safe[exported_name]
        lo, hi = descriptor["data_offsets"]
        handle.seek(safe_base + lo)
        check("selected_tensor:" + name,
              descriptor["dtype"] == "F32" and descriptor["shape"] == list(tensor.shape)
              and handle.read(hi - lo) == tensor_bytes(archive, prefix, tensor))
    return {"checkpoint_tensor_count": len(state), "export_tensor_count": len(keys),
            "shared_embedding_aliases_verified": aliases,
            "every_checkpoint_tensor_compared": True}


def archive_exact(artifact, destination, root):
    """Precheck every destination before copying any immutable metadata record."""
    pairs = [(artifact / name, destination / name) for name in ARCHIVE_NAMES]
    for source, target in pairs:
        if target.exists() and file_hash(source) != file_hash(target):
            raise ValueError("Refusing to overwrite a differing archived training record")
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for source, target in pairs:
        if not target.exists():
            shutil.copyfile(source, target)
        result[str(target.relative_to(root)).replace("\\", "/")] = file_hash(target)
    return result


def audit_run(root, artifact, task, archive_output):
    sources, checks = {}, {}

    def track(path):
        name = str(path.relative_to(root)).replace("\\", "/")
        sources[name] = file_hash(path)
        return sources[name]

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError(f"{task} audit failed: {name}")

    cap, expected_rows, passes, revision = ((360, {"train": 240, "validation": 120}, 3, 2)
        if task == "scope" else (240, {"train": 96, "validation": 48}, 5, 1))
    manifest = read_json(artifact / "manifest.json")
    card = read_json(artifact / "run_card.json")
    logs = read_rows(artifact / "training_log.jsonl")
    latest = read_json(artifact / "latest_checkpoint.json")
    for name in ARCHIVE_NAMES:
        track(artifact / name)
    check("completed_finite_cap", manifest["status"] == "completed" and manifest["purpose"] == "trained"
          and manifest["task"] == task and manifest["completed_steps"] == manifest["max_steps"] == card["max_steps"] == cap)
    check("revision_budget", card["revision_number"] == revision and card["maximum_targeted_revisions"] == 2)
    prior_key = "prior_scope" if task == "scope" else "initial_answer"
    prior_path = Path(card[prior_key + "_artifact"]) / "manifest.json"
    prior = read_json(prior_path)
    check("prior_artifact_binding", track(prior_path) == card[prior_key + "_manifest_sha256"]
          and prior["task"] == task and prior["status"] == "completed" and prior["purpose"] == "trained"
          and prior["model_id"] == manifest["model_id"] and prior["model_revision"] == manifest["model_revision"]
          and prior["pretrained_state_sha256"] == manifest["pretrained_state_sha256"]
          and Path(card["output_dir"]).resolve() != prior_path.parent.resolve())
    check("run_card_bytes", track(artifact / "run_card.json") == manifest["run_card_sha256"])
    configured_card = root / f"configs/run_cards/targeted-revisions/{task}.json"
    check("configured_card_matches_archived", track(configured_card) == manifest["run_card_sha256"])
    check("configuration_hash", canonical_hash({key: value for key, value in card.items()
          if key != "resume_checkpoint"}) == manifest["configuration_hash"])
    check("environment_hash", canonical_hash(manifest["environment"]) == manifest["environment_hash"])
    check("profile_environment", card["profile_environment_sha256"] == manifest["environment_hash"])
    profile_path = Path(card["profile_path"])
    check("profile_bytes", track(profile_path) == card["profile_sha256"])
    profile = read_json(profile_path)
    check("profile_dataset_environment", profile["dataset_id"] == manifest["dataset_id"]
          and profile["environment_sha256"] == manifest["environment_hash"])
    snapshot = root / "reports/training-source-v2"
    historical = read_json(snapshot / "manifest.json")
    track(snapshot / "manifest.json")
    for group in (manifest["trainer_source_hashes"], card["profile_source_sha256"], profile["trainer_source_sha256"]):
        for name, digest in group.items():
            reference = safe_relative(snapshot, name)
            check("historical_source_binding:" + name, historical["files_sha256"].get(name) == digest)
            check("source:" + name, track(reference) == digest)
            # Also bind currently imported formatter/input policy semantics.
            if name in ("editguard/formatting.py", "editguard/scope_inputs.py"):
                check("current_input_source:" + name, track(safe_relative(root, name)) == digest)
    check("profile_trainer_sources", profile["trainer_source_sha256"] == card["profile_source_sha256"]
          == manifest["trainer_source_hashes"])
    objective = expected_objective(task)
    check("loss_objective", manifest["loss_objective"] == manifest["training"]["loss_objective"] == objective
          and card.get("loss_policy", "uniform") == objective["policy"])
    if task == "scope":
        check("scope_input_policy", manifest["scope_input_policy"] == card["scope_input_policy"] == "entity_neutral_v1"
              and manifest["scope_input_policy_version"] == card["scope_input_policy_version"] == "editguard-scope-input-v1"
              and manifest["scope_input_source_sha256"] == card["scope_input_source_sha256"]
              == manifest["trainer_source_hashes"]["editguard/scope_inputs.py"])
    else:
        check("answer_does_not_have_scope_adaptation", not any(key in manifest for key in SCOPE_IDENTITY))
        check("weighted_card_objective", card["loss_objective"] == objective)

    data_dir = Path(card["data_dir"])
    check("development_only_inventory", {item.name for item in data_dir.iterdir()} == DATA_NAMES)
    dataset = read_json(data_dir / "manifest.json")
    registry_path = Path(card["reviews_path"])
    registry = read_json(registry_path)
    check("dataset_manifest", track(data_dir / "manifest.json") == manifest["manifest_sha256"])
    check("registry_bytes", track(registry_path) == manifest["reviews_sha256"])
    check("automated_not_human", registry["review_policy"] == manifest["review_policy"] == "automated_development"
          and registry["human_reviewed_data"] is False and manifest["human_reviewed_data"] is False)
    check("dataset_review_identity", registry["dataset_id"] == dataset["dataset_id"] == manifest["dataset_id"]
          and registry["origin_manifest_sha256"] == manifest["manifest_sha256"])
    check("registry_audit_binding", track(safe_relative(root, registry["audit_path"])) == registry["audit_sha256"])
    accepted, pairs_by_split = {}, {}
    for split in ("train", "validation"):
        path = data_dir / f"{task}_{split}.jsonl"
        rows = read_rows(path)
        check(split + ":row_count", len(rows) == expected_rows[split])
        check(split + ":raw_hash", track(path) == dataset["files"][path.name])
        result = []
        for row in rows:
            decision = registry["rows"][row["id"]]
            check("review:" + row["id"], decision["decision"] == "accept"
                  and decision["reviewer_type"] == "automated_development"
                  and decision["row_sha256"] == canonical_hash(row)
                  and datetime.fromisoformat(decision["reviewed_at"]).utcoffset() is not None)
            result.append({**row, "reviewed": True, "review_status": "accepted",
                           "review_policy": "automated_development", "human_reviewed": False})
        accepted[split] = result
        check(split + ":accepted_hash", canonical_hash(result) == manifest["data_hashes"][split + "_rows"])
        tokenization = manifest["tokenization"][split]
        check(split + ":no_truncation", tokenization["rows"] == len(result)
              and tokenization["input_truncated_rows"] == tokenization["target_truncated_rows"] == 0
              and tokenization["max_input_tokens"] <= card["input_length"]
              and (task == "scope" or tokenization["max_target_tokens"] <= card["target_length"]))
        if task == "scope":
            pairs = [scope_pair(row["question"], row["correction_text"], "entity_neutral_v1") for row in result]
            pairs_by_split[split] = dict(zip(pairs, [row["label"] for row in result]))
            check(split + ":transformed_pairs", canonical_hash(pairs) == tokenization["scope_pair_text_sha256"]
                  and len(set(pairs)) == len(pairs) == tokenization["unique_scope_pairs"])
        else:
            polarities = Counter(row["expected_polarity"] for row in result if row["question_type"] == "comparison")
            check(split + ":symmetric_polarity_counts", dict(polarities) == tokenization["weighted_polarity_rows"]
                  == {"yes": len(rows) // 4, "no": len(rows) // 4})
            check(split + ":weighted_target_denominator", tokenization["loss_weight_sum"] == (1976 if split == "train" else 976))
    if task == "scope":
        shared = set(pairs_by_split["train"]) & set(pairs_by_split["validation"])
        overlap = manifest["scope_input_overlap"]["cross_split"]
        check("transformed_overlap_disclosed", len(shared) == overlap["identical_input_pairs"] == overlap["same_label_pairs"] == 8
              and all(pairs_by_split["train"][pair] == pairs_by_split["validation"][pair] for pair in shared)
              and overlap["contradictory_label_pairs"] == 0)
    check("entity_split_disjoint", not ({r["correction"]["family_id"] for r in accepted["train"]}
          & {r["correction"]["family_id"] for r in accepted["validation"]}))
    check("log_exact_steps", [row["global_step"] for row in logs] == list(range(1, cap + 1)))
    check("finite_loss_gradients", all(math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"])
          and row["loss"] >= 0 and row["gradient_norm"] >= 0 for row in logs))
    check("elapsed_monotonic", all(a["elapsed_training_seconds"] < b["elapsed_training_seconds"] for a, b in zip(logs, logs[1:])))
    check("linear_learning_rate", all(math.isclose(row["learning_rate_after_step"], card["learning_rate"]
          * (1 - row["global_step"] / cap), abs_tol=1e-18) for row in logs))
    check("log_objective_policy", all(row["loss_policy"] == objective["policy"]
          and row["comparison_first_token_weight"] == objective["comparison_first_token_weight"]
          and row["loss_weight_sum"] > 0 for row in logs))
    if task == "scope":
        check("log_scope_policy", all(row["scope_input_policy"] == "entity_neutral_v1" for row in logs))
    interval = card["validation_every_steps"]
    observations = [{"global_step": row["global_step"], "loss": row["validation_loss"]}
                    for row in logs if row["validation_loss"] is not None]
    check("validation_schedule", [item["global_step"] for item in observations] == list(range(interval, cap + 1, interval)))
    check("finite_validation", all(math.isfinite(item["loss"]) and item["loss"] >= 0 for item in observations))
    best = min(observations, key=lambda item: (item["loss"], item["global_step"]))
    metric = "validation_loss" if task == "scope" else "validation_weighted_token_loss"
    check("selected_best_validation", manifest["selection"] == manifest["training"]["best_validation"]
          == {"global_step": best["global_step"], "value": best["loss"], "metric": metric,
              "checkpoint": f"step-{best['global_step']:08d}"})
    orders, rng = {}, random.Random(card["seed"])
    steps_per_epoch = expected_rows["train"] // card["batch_size"]
    check("declared_passes", steps_per_epoch * passes == cap and card["passes_over_accepted_train_rows"] == passes)
    for epoch in range(passes):
        order = list(range(len(accepted["train"])))
        rng.shuffle(order)
        orders[epoch] = order
        rows = logs[epoch * steps_per_epoch:(epoch + 1) * steps_per_epoch]
        batch_size = card["batch_size"]
        check(f"sampler_exact_epoch:{epoch}", all(row["sampler_epoch"] == epoch
              and row["sampler_cursor"] == (i + 1) * batch_size
              and row["batch_indices"] == order[i * batch_size:i * batch_size + batch_size] for i, row in enumerate(rows)))
    check("latest_pointer", latest["global_step"] == cap and Path(latest["path"]).resolve()
          == (artifact / f"checkpoints/step-{cap:08d}").resolve())
    for name, expected in manifest["files_sha256"].items():
        check("export:" + name, track(safe_relative(artifact, name)) == expected)
    check("tokenizer_identity", canonical_hash(tokenizer_record(artifact, task)) == manifest["tokenizer_hash"])
    identity_fields = (*BASE_IDENTITY, *SCOPE_IDENTITY) if task == "scope" else BASE_IDENTITY
    with (artifact / "model/model.safetensors").open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        safe = json.loads(handle.read(header_size))
        safe_base = 8 + header_size
        if task == "scope":
            check("model_classifier_shape", safe["classifier.weight"]["shape"] == [2, 768])
        else:
            check("model_t5_embedding_shape", safe["shared.weight"]["shape"] == [32128, 512])
        checkpoint_results, export_verification = [], None
        directories = sorted((artifact / "checkpoints").iterdir())
        check("declared_retention_only", len(directories) == card["keep_last_checkpoints"] == 2)
        for directory in directories:
            metadata = read_json(directory / "metadata.json")
            track(directory / "metadata.json")
            state_path = directory / "state.pt"
            check("checkpoint_bytes:" + directory.name, track(state_path) == metadata["state_sha256"])
            check("checkpoint_identity:" + directory.name, all(metadata[key] == manifest[key] for key in identity_fields))
            with zipfile.ZipFile(state_path) as archive:
                state, prefix = checkpoint_descriptors(archive)
                step = state["global_step"]
                check("payload_fields:" + directory.name, set(state) == FIELDS and metadata["state_fields"] == sorted(FIELDS))
                check("payload_identity:" + directory.name, state["identity"] == {key: manifest[key] for key in identity_fields}
                      and step == metadata["global_step"] == int(directory.name.split("-")[1]))
                check("scheduler_step:" + directory.name, state["scheduler"]["last_epoch"] == step)
                check("checkpoint_sampler:" + directory.name, state["sampler"]["size"] == expected_rows["train"]
                      and state["sampler"]["epoch"] == (step - 1) // steps_per_epoch
                      and state["sampler"]["cursor"] == expected_rows["train"]
                      and state["sampler"]["order"] == orders[(step - 1) // steps_per_epoch])
                parameter_count = len(state["model"]) - (2 if task == "answer" else 0)
                check("optimizer_updated:" + directory.name, len(state["optimizer"]["state"]) == parameter_count
                      and all(struct.unpack("<f", tensor_bytes(archive, prefix, item["step"]))[0] == step
                              for item in state["optimizer"]["state"].values()))
                check("rng_saved:" + directory.name, set(state["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
                      and len(state["rng"]["torch_cuda"]) == 1)
                digest = state_digest(archive, prefix, state["model"])
                selected = step == manifest["selection"]["global_step"]
                if selected:
                    export_verification = selected_export_comparison(handle, safe, safe_base,
                        state["model"], archive, prefix, task, check)
                    check("selected_digest", digest == manifest["selected_model_state_sha256"])
                if step == cap:
                    check("final_digest", digest == manifest["training"]["final_state_sha256"])
                checkpoint_results.append({"step": step, "state_bytes": state_path.stat().st_size,
                    "state_sha256": metadata["state_sha256"], "model_state_sha256": digest,
                    "optimizer_updated_parameters": len(state["optimizer"]["state"]), "model_tensor_count": len(state["model"]),
                    "selected_export_checked": selected, "metadata_sha256": file_hash(directory / "metadata.json")})
        check("selected_export_checked", export_verification is not None)
    check("weights_changed", manifest["selected_model_state_sha256"] != manifest["pretrained_state_sha256"])
    training = manifest["training"]
    optimizer_sum = sum(row["optimizer_seconds"] for row in logs)
    validation_sum = sum(item["seconds"] for item in training["validation_measurements_this_process"])
    checkpoint_sum = sum(item["seconds"] for item in training["checkpoint_measurements_this_process"])
    measured = {"training_loop_seconds": training["elapsed_training_seconds"], "full_process_wall_seconds": None,
        "startup_tokenization_export_seconds": None, "optimizer_phase_sum_seconds": optimizer_sum,
        "optimizer_phase_median_seconds": statistics.median(row["optimizer_seconds"] for row in logs),
        "validation_sum_seconds": validation_sum, "checkpoint_and_pruning_sum_seconds": checkpoint_sum,
        "remaining_loop_overhead_seconds": training["elapsed_training_seconds"] - optimizer_sum - validation_sum - checkpoint_sum,
        "peak_allocated_gpu_gib": max(row["peak_allocated_gpu_bytes"] for row in logs) / 2**30,
        "peak_reserved_gpu_gib": max(row["peak_reserved_gpu_bytes"] for row in logs) / 2**30,
        "max_sampled_rss_gib": max(training["resources"]["process_rss_bytes"], *(row["process_rss_bytes"] for row in logs)) / 2**30}
    check("positive_timing_remainder", measured["remaining_loop_overhead_seconds"] >= 0)
    check("sources_unchanged", all(file_hash(root / name) == digest for name, digest in sources.items()))
    check("no_ml_import", "torch" not in sys.modules and "transformers" not in sys.modules)
    archives = archive_exact(artifact, archive_output, root)
    return {"status": "passed", "task": task, "passing_checks": len(checks), "checks": checks,
        "source_hashes": sources, "archives": archives, "selection": manifest["selection"],
        "model_manifest_sha256": file_hash(artifact / "manifest.json"),
        "loss_objective": objective, "scope_input_policy": manifest.get("scope_input_policy"),
        "scope_input_overlap": manifest.get("scope_input_overlap"),
        "validation_loss_observations": observations,
        "training_mean_loss_by_pass": [statistics.mean(row["loss"] for row in logs[i:i + steps_per_epoch])
                                      for i in range(0, cap, steps_per_epoch)],
        "measurements": measured, "retained_checkpoints": checkpoint_results,
        "selected_export_verification": export_verification}


def audit(root, scope, answer, output, archive_output):
    verification_sources = {name: file_hash(root / name) for name in (
        "scripts/audit_targeted_runs.py", "scripts/audit_scope_revision.py")}
    runs = {task: audit_run(root, artifact, task, archive_output / task)
            for task, artifact in (("scope", scope), ("answer", answer))}
    if any(file_hash(root / name) != digest for name, digest in verification_sources.items()):
        raise ValueError("Audit verification sources changed during verification")
    report = {"schema_version": 1, "purpose": "independent_completed_targeted_training_artifact_audit",
        "audited_at": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "passing_checks": sum(run["passing_checks"] for run in runs.values()), "runs": runs,
        "verification_source_sha256": verification_sources,
        "model_loaded": False, "training_or_gpu_executed": False,
        "test_or_challenge_rows_read": False, "human_reviewed_data": False,
        "verification_method": "Standard-library checksums, independently reproduced reviewed-row hashes, restricted checkpoint descriptor unpickling, complete selected tensor-byte comparison including T5 shared embeddings, exact tensor-state digest, optimizer/sampler/scheduler checks.",
        "limitations": ["Loss is not classifier accuracy or corrected-answer accuracy.",
            "The answer objective is weighted; its loss must not be compared numerically with the initial unweighted loss.",
            "The scope representation differs from revision1; 8 same-label abstracted pairs overlap train/validation and are disclosed, despite raw entity/lineage separation.",
            "Saved state checks do not independently verify label quality, calibration or predictive generalization.",
            "Actual durations cover the training loop; complete startup/export wall times were not recorded.",
            "Weights/checkpoints remain local ignored files; tracked metadata does not ship deployable weights."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=ROOT / "artifacts/scope/pilot-v3-neutral2")
    parser.add_argument("--answer", type=Path, default=ROOT / "artifacts/answer/pilot-v2-polarity1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/targeted-training-audit.json")
    parser.add_argument("--archive-output", type=Path, default=ROOT / "reports/targeted-training")
    args = parser.parse_args()
    report = audit(ROOT, args.scope.resolve(), args.answer.resolve(), args.output.resolve(), args.archive_output.resolve())
    print(json.dumps({"status": report["status"], "passing_checks": report["passing_checks"],
        "runs": {task: {key: run[key] for key in ("selection", "measurements", "passing_checks")}
                 for task, run in report["runs"].items()}}, indent=2))


if __name__ == "__main__":
    main()
