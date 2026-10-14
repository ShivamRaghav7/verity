"""Audit the completed scope pilot revision without loading ML packages or test data.

Checkpoint pickle globals are restricted to opaque tensor/storage descriptors.
Tensor values are compared as raw bytes; no Torch objects or GPU are constructed.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import random
import shutil
import statistics
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"model", "optimizer", "scheduler", "rng", "global_step", "sampler", "identity", "best_validation"}
IDENTITY = ("task", "purpose", "model_id", "model_revision", "dataset_id", "manifest_sha256",
            "reviews_sha256", "run_card_sha256", "configuration_hash", "environment_hash", "data_hashes",
            "pretrained_state_sha256", "tokenizer_hash", "trainer_source_hashes")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class StorageKind:
    name: str
    dtype: str
    width: int


@dataclass(frozen=True)
class StorageRef:
    kind: StorageKind
    key: str
    location: str
    count: int


@dataclass(frozen=True)
class TensorRef:
    storage: StorageRef
    offset: int
    shape: tuple
    stride: tuple


def rebuild_tensor(storage, offset, size, stride, requires_grad, backward_hooks):
    if not isinstance(storage, StorageRef):
        raise ValueError("Unexpected tensor storage")
    return TensorRef(storage, offset, tuple(size), tuple(stride))


class DescriptorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch", "FloatStorage"): StorageKind("FloatStorage", "torch.float32", 4),
            ("torch", "ByteStorage"): StorageKind("ByteStorage", "torch.uint8", 1),
            ("torch._utils", "_rebuild_tensor_v2"): rebuild_tensor,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unsupported checkpoint global: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, value):
        if len(value) != 5 or value[0] != "storage" or not isinstance(value[1], StorageKind):
            raise ValueError("Unsupported checkpoint storage reference")
        return StorageRef(value[1], str(value[2]), value[3], value[4])


def checkpoint_descriptors(archive):
    pickles = [name for name in archive.namelist() if name.endswith("/data.pkl")]
    if len(pickles) != 1:
        raise ValueError("Expected exactly one checkpoint pickle")
    prefix = pickles[0][:-len("data.pkl")]
    if archive.read(prefix + "byteorder") != b"little":
        raise ValueError("Only little-endian recorded tensor storage is supported")
    return DescriptorUnpickler(io.BytesIO(archive.read(pickles[0]))).load(), prefix


def tensor_bytes(archive, prefix, tensor):
    if not isinstance(tensor, TensorRef):
        raise ValueError("Expected opaque tensor descriptor")
    count = 1
    for size, stride in reversed(list(zip(tensor.shape, tensor.stride))):
        if size != 1 and stride != count:
            raise ValueError("Expected contiguous checkpoint tensor")
        count *= size
    storage = archive.read(prefix + "data/" + tensor.storage.key)
    if len(storage) != tensor.storage.count * tensor.storage.kind.width:
        raise ValueError("Storage length differs from descriptor")
    start = tensor.offset * tensor.storage.kind.width
    value = storage[start:start + count * tensor.storage.kind.width]
    if len(value) != count * tensor.storage.kind.width:
        raise ValueError("Tensor view extends beyond storage")
    return value


def state_digest(archive, prefix, state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.storage.kind.dtype.encode("ascii"))
        digest.update(str(tensor.shape).encode("ascii"))
        digest.update(tensor_bytes(archive, prefix, tensor))
    return digest.hexdigest()


def audit(root, artifact, output, archive_output):
    source_hashes, checks = {}, {}

    def track(path):
        source_hashes[str(path.relative_to(root)).replace("\\", "/")] = file_hash(path)
        return source_hashes[str(path.relative_to(root)).replace("\\", "/")]

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError(f"Audit failed: {name}")

    manifest = read_json(artifact / "manifest.json")
    card = read_json(artifact / "run_card.json")
    logs = read_rows(artifact / "training_log.jsonl")
    latest = read_json(artifact / "latest_checkpoint.json")
    for name in ("manifest.json", "run_card.json", "training_log.jsonl", "latest_checkpoint.json"):
        track(artifact / name)
    check("completed_finite_cap", manifest["status"] == "completed" and manifest["purpose"] == "trained"
          and manifest["task"] == "scope" and manifest["completed_steps"] == card["max_steps"] == 360)
    check("run_card_bytes", track(artifact / "run_card.json") == manifest["run_card_sha256"])
    check("configured_card_matches_archived", track(root / "configs/run_cards/pilot-v3/scope.json") == manifest["run_card_sha256"])
    config = {key: value for key, value in card.items() if key != "resume_checkpoint"}
    check("configuration_hash", canonical_hash(config) == manifest["configuration_hash"])
    check("environment_hash", canonical_hash(manifest["environment"]) == manifest["environment_hash"])
    check("profile_environment", card["profile_environment_sha256"] == manifest["environment_hash"])
    profile_path = Path(card["profile_path"])
    check("profile_bytes", track(profile_path) == card["profile_sha256"])
    profile = read_json(profile_path)
    check("profile_dataset_environment", profile["dataset_id"] == manifest["dataset_id"]
          and profile["environment_sha256"] == manifest["environment_hash"])
    for group in (manifest["trainer_source_hashes"], card["profile_source_sha256"]):
        for name, digest in group.items():
            current = root / name
            if file_hash(current) == digest:
                reference = current
            else:
                # Completed experiments bind the source used at run time. A
                # later opt-in trainer objective must not invalidate history.
                snapshot = root / "reports/training-source-v1"
                historical = read_json(snapshot / "manifest.json")
                check("historical_source_binding:" + name,
                      historical["files_sha256"].get(name) == digest)
                track(snapshot / "manifest.json")
                reference = snapshot / name
                if snapshot.resolve() not in reference.resolve().parents:
                    raise ValueError("Historical source path escaped its snapshot.")
            check("source:" + name, track(reference) == digest)

    data_dir = Path(card["data_dir"])
    names = {item.name for item in data_dir.iterdir()}
    check("development_only_inventory", names == {"manifest.json", "scope_train.jsonl", "scope_validation.jsonl",
          "answer_train.jsonl", "answer_validation.jsonl", "catalogue.json", "derivation.json", "selection.json", "split_assignments.json"})
    dataset = read_json(data_dir / "manifest.json")
    registry_path = Path(card["reviews_path"])
    registry = read_json(registry_path)
    check("dataset_manifest", track(data_dir / "manifest.json") == manifest["manifest_sha256"])
    check("registry_bytes", track(registry_path) == manifest["reviews_sha256"])
    check("automated_not_human", registry["review_policy"] == manifest["review_policy"] == "automated_development"
          and registry["human_reviewed_data"] is False and manifest["human_reviewed_data"] is False)
    check("dataset_review_identity", registry["dataset_id"] == dataset["dataset_id"] == manifest["dataset_id"]
          and registry["origin_manifest_sha256"] == manifest["manifest_sha256"])
    check("registry_audit_binding", track(root / registry["audit_path"]) == registry["audit_sha256"])
    accepted = {}
    for split in ("train", "validation"):
        path = data_dir / f"scope_{split}.jsonl"
        rows = read_rows(path)
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
              and tokenization["max_input_tokens"] <= card["input_length"])
    check("entity_split_disjoint", not ({r["correction"]["family_id"] for r in accepted["train"]}
          & {r["correction"]["family_id"] for r in accepted["validation"]}))
    check("log_exact_steps", [row["global_step"] for row in logs] == list(range(1, 361)))
    check("finite_loss_gradients", all(math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"])
          and row["loss"] >= 0 and row["gradient_norm"] >= 0 for row in logs))
    check("elapsed_monotonic", all(a["elapsed_training_seconds"] < b["elapsed_training_seconds"] for a, b in zip(logs, logs[1:])))
    check("linear_learning_rate", all(math.isclose(row["learning_rate_after_step"], card["learning_rate"]
          * (1 - row["global_step"] / 360), abs_tol=1e-18) for row in logs))
    observations = [{"global_step": row["global_step"], "loss": row["validation_loss"]}
                    for row in logs if row["validation_loss"] is not None]
    check("validation_schedule", [item["global_step"] for item in observations] == [120, 240, 360])
    best = min(observations, key=lambda item: (item["loss"], item["global_step"]))
    check("selected_best_validation", manifest["selection"] == {"global_step": best["global_step"],
          "value": best["loss"], "metric": "validation_loss", "checkpoint": f"step-{best['global_step']:08d}"})
    orders, rng = {}, random.Random(card["seed"])
    for epoch in range(3):
        order = list(range(len(accepted["train"])))
        rng.shuffle(order)
        orders[epoch] = order
        rows = logs[epoch * 120:(epoch + 1) * 120]
        check(f"sampler_exact_epoch:{epoch}", all(row["sampler_epoch"] == epoch and row["sampler_cursor"] == (i + 1) * 2
              and row["batch_indices"] == order[i * 2:i * 2 + 2] for i, row in enumerate(rows)))
    check("latest_pointer", latest["global_step"] == 360 and Path(latest["path"]).resolve() == (artifact / "checkpoints/step-00000360").resolve())
    for name, expected in manifest["files_sha256"].items():
        check("export:" + name, track(artifact / name) == expected)
    tokenizer = read_json(artifact / "tokenizer/tokenizer.json")
    token_config = read_json(artifact / "tokenizer/tokenizer_config.json")
    tokenizer_record = {"vocabulary": tokenizer["model"]["vocab"],
                        "special_tokens": {key: token_config[key] for key in ("unk_token", "sep_token", "pad_token", "cls_token", "mask_token")},
                        "padding_side": "right", "truncation_side": "right"}
    check("tokenizer_identity", canonical_hash(tokenizer_record) == manifest["tokenizer_hash"])
    with (artifact / "model/model.safetensors").open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        safe = json.loads(handle.read(header_size))
        safe_base = 8 + header_size
        check("model_classifier_shape", safe["classifier.weight"]["shape"] == [2, 768])
        checkpoint_results = []
        for directory in sorted((artifact / "checkpoints").iterdir()):
            metadata = read_json(directory / "metadata.json")
            track(directory / "metadata.json")
            state_path = directory / "state.pt"
            check("checkpoint_bytes:" + directory.name, track(state_path) == metadata["state_sha256"])
            check("checkpoint_identity:" + directory.name, all(metadata[key] == manifest[key] for key in IDENTITY))
            with zipfile.ZipFile(state_path) as archive:
                state, prefix = checkpoint_descriptors(archive)
                step = state["global_step"]
                check("payload_fields:" + directory.name, set(state) == FIELDS and metadata["state_fields"] == sorted(FIELDS))
                check("payload_identity:" + directory.name, state["identity"] == {key: manifest[key] for key in IDENTITY}
                      and step == metadata["global_step"] == int(directory.name.split("-")[1]))
                check("scheduler_step:" + directory.name, state["scheduler"]["last_epoch"] == step)
                check("checkpoint_sampler:" + directory.name, state["sampler"]["size"] == 240
                      and state["sampler"]["epoch"] == (step - 1) // 120 and state["sampler"]["cursor"] == 240
                      and state["sampler"]["order"] == orders[(step - 1) // 120])
                check("optimizer_updated:" + directory.name, len(state["optimizer"]["state"]) == len(state["model"])
                      and all(struct.unpack("<f", tensor_bytes(archive, prefix, item["step"]))[0] == step
                              for item in state["optimizer"]["state"].values()))
                check("rng_saved:" + directory.name, set(state["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
                      and len(state["rng"]["torch_cuda"]) == 1)
                digest = state_digest(archive, prefix, state["model"])
                selected = step == manifest["selection"]["global_step"]
                if selected:
                    check("selected_export_keys", set(state["model"]) == set(safe) - {"__metadata__"})
                    for name, tensor in state["model"].items():
                        descriptor = safe[name]
                        lo, hi = descriptor["data_offsets"]
                        handle.seek(safe_base + lo)
                        check("selected_tensor:" + name, descriptor["dtype"] == "F32" and descriptor["shape"] == list(tensor.shape)
                              and handle.read(hi - lo) == tensor_bytes(archive, prefix, tensor))
                    check("selected_digest", digest == manifest["selected_model_state_sha256"])
                if step == 360:
                    check("final_digest", digest == manifest["training"]["final_state_sha256"])
                checkpoint_results.append({"step": step, "state_bytes": state_path.stat().st_size,
                    "state_sha256": metadata["state_sha256"], "model_state_sha256": digest,
                    "optimizer_updated_parameters": len(state["optimizer"]["state"]), "model_tensor_count": len(state["model"]),
                    "selected_export_checked": selected, "metadata_sha256": file_hash(directory / "metadata.json")})
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
    # Confirm sources read by this audit have not changed while it ran.
    check("sources_unchanged", all(file_hash(root / name) == digest for name, digest in source_hashes.items()))
    check("no_ml_import", "torch" not in sys.modules and "transformers" not in sys.modules)
    archives = {}
    archive_output.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "run_card.json", "training_log.jsonl", "latest_checkpoint.json"):
        source, destination = artifact / name, archive_output / name
        if destination.exists() and file_hash(destination) != file_hash(source):
            raise ValueError("Refusing to overwrite a differing archived training record")
        shutil.copyfile(source, destination)
        archives[str(destination.relative_to(root)).replace("\\", "/")] = file_hash(destination)
    report = {"schema_version": 1, "purpose": "independent_completed_scope_revision_artifact_audit",
        "audited_at": datetime.now(timezone.utc).isoformat(), "dataset_id": manifest["dataset_id"],
        "model_manifest_sha256": source_hashes[str((artifact / "manifest.json").relative_to(root)).replace("\\", "/")],
        "status": "passed", "checks": checks, "passing_checks": len(checks), "source_hashes": source_hashes,
        "archives": archives, "selection": manifest["selection"], "validation_loss_observations": observations,
        "training_mean_loss_by_pass": [statistics.mean(row["loss"] for row in logs[i:i + 120]) for i in (0, 120, 240)],
        "measurements": measured, "retained_checkpoints": checkpoint_results, "model_loaded": False,
        "training_or_gpu_executed": False, "test_or_challenge_rows_read": False, "human_reviewed_data": False,
        "limitations": ["Loss is cross-entropy, not classifier accuracy or corrected-answer accuracy.",
            "Validation population differs from initial pilot negatives; loss values are not a paired comparison.",
            "This checks saved bytes and training state, not independent label quality or predictive generalization.",
            "Actual recorded duration covers the training loop; complete startup/export wall time was not recorded.",
            "Weights/checkpoints remain local ignored files; tracked metadata does not ship deployable weights."],
        "verification_method": "Standard-library checksums, independently reproduced reviewed-row hashes, restricted checkpoint descriptor unpickling, complete selected tensor-byte comparison, exact tensor-state digest, optimizer/sampler/scheduler checks."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts/scope/pilot-v3-revision1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pilot-v3-training-audit.json")
    parser.add_argument("--archive-output", type=Path, default=ROOT / "reports/pilot-v3-training/scope")
    args = parser.parse_args()
    report = audit(ROOT, args.artifact.resolve(), args.output.resolve(), args.archive_output.resolve())
    print(json.dumps({key: report[key] for key in ("status", "passing_checks", "selection", "measurements")}, indent=2))


if __name__ == "__main__":
    main()
