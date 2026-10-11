"""Finite auxiliary-model training with exact, hash-checked checkpoint state.

ML packages are intentionally imported inside functions. Importing this module
does not make Torch or Transformers dependencies of the structured application.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from time import perf_counter
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_FIELDS = {
    "model", "optimizer", "scheduler", "rng", "global_step", "sampler",
    "identity", "best_validation",
}


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_file_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    atomic_bytes(path, json_file_bytes(value))


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class DeterministicSampler:
    """Independent shuffle RNG plus an explicit epoch order and batch cursor."""

    def __init__(self, size, seed):
        if type(size) is not int or size < 1:
            raise ValueError("Sampler needs at least one row")
        self.size = size
        self.rng = random.Random(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = list(range(size))
        self.rng.shuffle(self.order)

    def next_indices(self, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.cursor == self.size:
            self.epoch += 1
            self.cursor = 0
            self.order = list(range(self.size))
            self.rng.shuffle(self.order)
        indices = self.order[self.cursor:min(self.size, self.cursor + batch_size)]
        self.cursor += len(indices)
        return indices

    def state_dict(self):
        return {"size": self.size, "epoch": self.epoch, "cursor": self.cursor,
                "order": list(self.order), "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        if state.get("size") != self.size or sorted(state.get("order", [])) != list(range(self.size)):
            raise ValueError("Sampler size or permutation differs from current rows")
        if type(state.get("cursor")) is not int or not 0 <= state["cursor"] <= self.size:
            raise ValueError("Invalid checkpoint sampler cursor")
        if type(state.get("epoch")) is not int or state["epoch"] < 0:
            raise ValueError("Invalid checkpoint sampler epoch")
        self.epoch, self.cursor, self.order = state["epoch"], state["cursor"], list(state["order"])
        self.rng.setstate(state["rng"])


def capture_rng():
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {
        "generator": numpy_state[0], "state": numpy_state[1].tolist(),
        "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values["generator"], np.asarray(values["state"], dtype=np.uint32),
                         values["position"], values["has_gauss"], values["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count differs from checkpoint")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def configure_determinism(seed, cpu_threads=2):
    # Must precede CUDA initialization in the CLI process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment_record(device):
    import torch
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "numpy", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    driver_identity = None
    if device == "cuda":
        try:
            measured = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=10)
            driver_identity = sorted(line.strip() for line in measured.stdout.splitlines() if line.strip())
            if not driver_identity:
                raise ValueError("nvidia-smi returned no GPU identity")
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("CUDA exact-state resume requires measured NVIDIA driver and UUID identity") from exc
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "device": device, "packages": packages, "torch_cuda": torch.version.cuda,
            "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                            if device == "cuda" else [],
            "nvidia_driver_uuid_identity": driver_identity,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES") if device == "cuda" else None,
            "cpu_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "attention_implementation": "eager", "precision": "float32", "tf32": False}


def state_digest(state):
    """Hash actual tensor values, rather than the nondeterministic archive bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def validate_identity(expected, actual):
    if expected != actual:
        differing = sorted(key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key))
        raise ValueError("Exact resume identity differs: " + ", ".join(differing))


def save_checkpoint(directory, *, model, optimizer, scheduler, sampler, global_step, identity, best_validation):
    import torch
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise ValueError(f"Checkpoint already exists; refusing to overwrite {directory}")
    temporary = directory.with_name(directory.name + ".tmp-" + uuid4().hex)
    temporary.mkdir()
    try:
        state = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "rng": capture_rng(), "sampler": sampler.state_dict(), "global_step": global_step,
                 "identity": identity, "best_validation": best_validation}
        state_path = temporary / "state.pt"
        with state_path.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        metadata = {"schema_version": 1, **identity, "global_step": global_step,
                    "state_sha256": file_hash(state_path), "state_fields": sorted(CHECKPOINT_FIELDS),
                    "best_validation": best_validation, "created_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(temporary / "metadata.json", metadata)
        os.replace(temporary, directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return directory


def load_checkpoint(directory, *, model, optimizer, scheduler, sampler, identity, max_steps):
    import torch
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    actual_identity = {key: metadata.get(key) for key in identity}
    validate_identity(identity, actual_identity)
    state_path = directory / "state.pt"
    if metadata.get("state_sha256") != file_hash(state_path):
        raise ValueError("Checkpoint state checksum differs; refusing corrupt state")
    # The state contains tensors and primitive RNG representations only.
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not CHECKPOINT_FIELDS.issubset(state):
        raise ValueError("Checkpoint is missing exact-state fields")
    validate_identity(identity, state["identity"])
    step = state["global_step"]
    if type(step) is not int or not 0 <= step < max_steps or metadata.get("global_step") != step:
        raise ValueError("Resume step must precede the unchanged finite stopping cap")
    sampler.load_state_dict(state["sampler"])
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    return {"global_step": step, "best_validation": state["best_validation"]}


def _batch(encoded, indices, device):
    return {key: value[indices].to(device) for key, value in encoded.items()}


def validation_loss(model, encoded, batch_size, device, task):
    import torch
    model.eval()
    total, denominator = 0.0, 0
    with torch.inference_mode():
        size = len(encoded["labels"])
        for start in range(0, size, batch_size):
            indices = list(range(start, min(start + batch_size, size)))
            batch = _batch(encoded, indices, device)
            loss = model(**batch).loss
            weight = int((batch["labels"] != -100).sum().item()) if task == "answer" else len(indices)
            total += float(loss.item()) * weight
            denominator += weight
    model.train()
    if not denominator:
        raise ValueError("Validation has no scored examples or target tokens")
    value = total / denominator
    if not math.isfinite(value):
        raise ValueError("Validation loss is not finite")
    return value


def _resources(device):
    import torch
    record = {"peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
              "process_rss_bytes": None}
    try:
        import psutil
        record["process_rss_bytes"] = psutil.Process().memory_info().rss
    except ImportError:
        pass
    return record


def _trim_log(log_path, step):
    if not log_path.exists():
        return
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    committed = [line for line in lines if line["global_step"] <= step]
    if len(committed) != len(lines):
        backup = log_path.with_name("training_log.uncommitted-" + uuid4().hex + ".jsonl")
        shutil.copy2(log_path, backup)
        temporary = log_path.with_name(log_path.name + ".tmp-" + uuid4().hex)
        temporary.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in committed), encoding="utf-8")
        os.replace(temporary, log_path)


def _prune_checkpoints(root, keep_last, best):
    directories = sorted((path for path in root.glob("step-*") if path.is_dir() and ".tmp-" not in path.name),
                         key=lambda path: int(path.name.split("-")[1]))
    retained = set(directories[-keep_last:])
    if best:
        retained.add(root / best["checkpoint"])
    for directory in directories:
        if directory not in retained:
            resolved = directory.resolve()
            if root.resolve() not in resolved.parents:
                raise ValueError("Checkpoint cleanup escaped the run's checkpoint directory")
            shutil.rmtree(resolved)


def train_loop(model, encoded_train, encoded_validation, *, config, identity, output_dir,
               resume_checkpoint=None, stop_after_steps=None):
    """Shared real optimizer loop; also exercised by a tiny CPU regression model."""
    import torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = config["device"]
    max_steps = config["max_steps"]
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")
    if stop_after_steps is not None and (type(stop_after_steps) is not int or not 1 <= stop_after_steps <= max_steps):
        raise ValueError("stop_after_steps must be an absolute step within max_steps")
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config.get("weight_decay", 0.01), betas=(0.9, 0.999), eps=1e-8)
    # A fixed linear schedule depends on the original cap, including on resume.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1.0 - step / max_steps))
    sampler = DeterministicSampler(len(encoded_train["labels"]), config["seed"])
    step, best = 0, None
    log_path = output_dir / "training_log.jsonl"
    if resume_checkpoint:
        restored = load_checkpoint(resume_checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
                                   sampler=sampler, identity=identity, max_steps=max_steps)
        step, best = restored["global_step"], restored["best_validation"]
        _trim_log(log_path, step)
    elif log_path.exists() or (output_dir / "checkpoints").exists():
        raise ValueError("Run output contains training state; supply an exact resume checkpoint or a new output directory")
    if stop_after_steps is not None and stop_after_steps <= step:
        raise ValueError("stop_after_steps must exceed the resumed global step")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    checkpoint_root = output_dir / "checkpoints"
    last_checkpoint = None
    while step < max_steps:
        indices = sampler.next_indices(config["batch_size"])
        batch = _batch(encoded_train, indices, device)
        if device == "cuda":
            torch.cuda.synchronize()
        step_started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not math.isfinite(float(loss.detach().item())):
            raise ValueError("Training loss is not finite; retain the previous checkpoint for diagnosis")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
        if not math.isfinite(float(gradient_norm.item())):
            raise ValueError("Gradient norm is not finite; no update committed")
        optimizer.step()
        scheduler.step()
        step += 1
        if device == "cuda":
            torch.cuda.synchronize()
        duration = perf_counter() - step_started
        val = None
        should_validate = step % config["validation_every_steps"] == 0 or step == max_steps
        if should_validate:
            val = validation_loss(model, encoded_validation, config["batch_size"], device, config["task"])
            if best is None or val < best["value"]:
                best = {"metric": "validation_loss", "value": val, "global_step": step,
                        "checkpoint": f"step-{step:08d}"}
        record = {"global_step": step, "loss": float(loss.detach().item()), "validation_loss": val,
                  "learning_rate_after_step": scheduler.get_last_lr()[0], "gradient_norm": float(gradient_norm.item()),
                  "optimizer_seconds": duration, "elapsed_training_seconds": perf_counter() - started,
                  "batch_indices": indices, "sampler_epoch": sampler.epoch, "sampler_cursor": sampler.cursor,
                  **_resources(device)}
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            log.flush()
        stopping = stop_after_steps is not None and step >= stop_after_steps
        if should_validate or step % config["checkpoint_every_steps"] == 0 or step == max_steps or stopping:
            last_checkpoint = save_checkpoint(checkpoint_root / f"step-{step:08d}", model=model, optimizer=optimizer,
                scheduler=scheduler, sampler=sampler, global_step=step, identity=identity, best_validation=best)
            atomic_json(output_dir / "latest_checkpoint.json", {"path": str(last_checkpoint.resolve()), "global_step": step})
            _prune_checkpoints(checkpoint_root, config["keep_last_checkpoints"], best)
        if stopping:
            break
    return {"global_step": step, "best_validation": best,
            "status": "completed" if step == max_steps else "interrupted",
            "last_checkpoint": str(last_checkpoint.resolve()) if last_checkpoint else None,
            "elapsed_training_seconds": perf_counter() - started, "final_state_sha256": state_digest(model.state_dict()),
            "resources": _resources(device)}


def tokenize_rows(rows, *, task, tokenizer, input_length, target_length=None):
    import torch
    if task == "scope":
        first, second = [row["question"] for row in rows], [row["correction_text"] for row in rows]
        lengths = [len(tokenizer(question, correction, truncation=False)["input_ids"])
                   for question, correction in zip(first, second)]
        encoded = tokenizer(first, second, padding="max_length", truncation=True,
                            max_length=input_length, return_tensors="pt")
        encoded["labels"] = torch.tensor([row["label"] for row in rows], dtype=torch.long)
        target_lengths = []
    else:
        first = [row["input_text"] for row in rows]
        lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in first]
        encoded = tokenizer(first, padding="max_length", truncation=True, max_length=input_length, return_tensors="pt")
        targets = [row["target_text"] for row in rows]
        target_lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in targets]
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=target_length, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        encoded["labels"] = labels
    statistics = {"rows": len(rows), "max_input_tokens": max(lengths),
                  "input_truncated_rows": sum(length > input_length for length in lengths),
                  "max_target_tokens": max(target_lengths) if target_lengths else None,
                  "target_truncated_rows": sum(length > target_length for length in target_lengths)}
    if statistics["input_truncated_rows"] or statistics["target_truncated_rows"]:
        raise ValueError("Training inputs/targets would be truncated; revise and reprofile the run lengths: " + json.dumps(statistics))
    return dict(encoded), statistics


def _file_manifest(directory):
    return {str(path.relative_to(directory)).replace("\\", "/"): file_hash(path)
            for sub in ("model", "tokenizer") for path in sorted((directory / sub).rglob("*")) if path.is_file()}


def synthetic_rows(task):
    """Explicit developer fixture, never a reviewed dataset or performance claim."""
    correction = "Correction for Nimbus, Basic, version v2: device_limit changed from 5 to 3 devices, effective from 2026-10-01. Source: synthetic developer fixture."
    questions = ["How many devices does Nimbus Basic v2 support?", "What is Nimbus Basic v2's monthly price?",
                 "Can Nimbus Basic v2 connect four devices?", "How many devices does Nimbus Pro v2 support?"]
    if task == "scope":
        train = [{"id": f"synthetic-scope-{i}", "question": question, "correction_text": correction, "label": label}
                 for i, (question, label) in enumerate(zip(questions, [1, 0, 1, 0]))]
        validation = [{"id": "synthetic-validation-positive", "question": "What is the device limit on Nimbus Basic v2?", "correction_text": correction, "label": 1},
                      {"id": "synthetic-validation-negative", "question": "How many devices on Nimbus Basic v1?", "correction_text": correction, "label": 0}]
    else:
        train = [{"id": f"synthetic-answer-{i}",
                  "input_text": f"answer correction: {correction} Question: {question} As of: 2026-10-11. Unchanged context: Nimbus Basic v2 costs 9 USD/month.",
                  "target_text": answer} for i, (question, answer) in enumerate([
                      (questions[0], "Nimbus Basic v2 supports up to 3 devices."),
                      (questions[2], "No. Nimbus Basic v2 supports up to 3 devices.")])]
        validation = [{"id": "synthetic-answer-validation", "input_text": train[0]["input_text"].replace(questions[0], "What is Nimbus Basic v2's device limit?"),
                       "target_text": "Nimbus Basic v2 supports up to 3 devices."}]
    return train, validation


def run_training(card, *, train_rows, validation_rows, purpose, run_card_sha256,
                 resume_checkpoint=None, stop_after_steps=None, run_card_source=None):
    """Load pinned cached pretrained weights, optimize, and publish the best model."""
    if type(card.get("max_steps")) is not int or not 1 <= card["max_steps"] <= 100_000:
        raise ValueError("max_steps must be an integer between 1 and 100000")
    if purpose == "smoke_test" and card["max_steps"] > 4:
        raise ValueError("Developer smoke is capped at four optimizer steps")
    if stop_after_steps is not None and (type(stop_after_steps) is not int or not 1 <= stop_after_steps <= card["max_steps"]):
        raise ValueError("stop_after_steps must be an absolute step within max_steps")
    if purpose == "trained" and (not card.get("manifest_sha256") or not card.get("reviews_sha256")
            or any(not row.get("reviewed") or row.get("review_status") != "accepted" for row in [*train_rows, *validation_rows])):
        raise ValueError("Production training requires preflight-approved exact reviewed rows and hashes")
    configure_determinism(card["seed"], card.get("cpu_threads", 2))
    import torch
    from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
    if purpose not in ("trained", "smoke_test"):
        raise ValueError("Unsupported training artifact purpose")
    task = card["task"]
    device = card["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("The profiled CUDA device is unavailable")
    if card.get("precision") != "float32":
        raise ValueError("Only the profiled float32 path is implemented")
    if card.get("scheduler", "linear") != "linear":
        raise ValueError("Only the declared linear learning-rate schedule is implemented")
    if card["batch_size"] != 2:
        raise ValueError("Initial trainer requires the profiled batch_size=2")
    kwargs = {"revision": card["model_revision"], "local_files_only": True, "dtype": torch.float32,
              "attn_implementation": "eager"}
    loader = AutoModelForSequenceClassification if task == "scope" else AutoModelForSeq2SeqLM
    if task == "scope":
        kwargs["num_labels"] = 2
    tokenizer = AutoTokenizer.from_pretrained(card["model_id"], revision=card["model_revision"], local_files_only=True)
    model = loader.from_pretrained(card["model_id"], **kwargs)
    if getattr(model.config, "_commit_hash", None) != card["model_revision"]:
        raise ValueError("Loaded pretrained checkpoint revision differs from the run card")
    pretrained_hash = state_digest(model.state_dict())
    train, train_statistics = tokenize_rows(train_rows, task=task, tokenizer=tokenizer,
                                           input_length=card["input_length"], target_length=card.get("target_length"))
    validation, validation_statistics = tokenize_rows(validation_rows, task=task, tokenizer=tokenizer,
                                           input_length=card["input_length"], target_length=card.get("target_length"))
    environment = environment_record(device)
    config = {**card, "validation_every_steps": card.get("validation_every_steps", card["checkpoint_every_steps"])}
    for name in ("checkpoint_every_steps", "validation_every_steps", "keep_last_checkpoints", "cpu_threads"):
        value = config.get(name, 2 if name == "cpu_threads" else None)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, default in (("learning_rate", None), ("weight_decay", .01), ("max_grad_norm", 1.0)):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
            raise ValueError(f"{name} must be finite and " + ("nonnegative" if name == "weight_decay" else "positive"))
    config_identity = {key: value for key, value in config.items() if key not in ("resume_checkpoint",)}
    identity = {"task": task, "purpose": purpose, "model_id": card["model_id"], "model_revision": card["model_revision"],
                "dataset_id": card["dataset_id"], "manifest_sha256": card.get("manifest_sha256"),
                "reviews_sha256": card.get("reviews_sha256"), "run_card_sha256": run_card_sha256,
                "configuration_hash": canonical_hash(config_identity), "environment_hash": canonical_hash(environment),
                "data_hashes": {"train_rows": canonical_hash(train_rows), "validation_rows": canonical_hash(validation_rows)},
                "pretrained_state_sha256": pretrained_hash,
                "tokenizer_hash": canonical_hash({"vocabulary": tokenizer.get_vocab(), "special_tokens": tokenizer.special_tokens_map,
                    "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side}),
                "trainer_source_hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path)
                    for path in (Path(__file__), ROOT / "scripts/train_auxiliary.py", ROOT / "scripts/preflight_training.py", ROOT / "scripts/prepare_data.py")}}
    output = Path(card["output_dir"])
    if resume_checkpoint:
        resume_checkpoint = Path(resume_checkpoint).resolve()
        if output.resolve() not in resume_checkpoint.parents or not resume_checkpoint.is_dir():
            raise ValueError("Resume checkpoint must be an existing directory inside output_dir")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Fresh training needs an empty output directory; refusing to overwrite artifacts")
    output.mkdir(parents=True, exist_ok=True)
    base_manifest = {"schema_version": 1, "artifact_type": "editguard_auxiliary", **identity,
                     "status": "running", "model_path": "model", "tokenizer_path": "tokenizer",
                     "environment": environment, "input_format_version": "editguard-input-v1",
                     "input_format": "question_then_correction_pair" if task == "scope" else "dataset_input_text_v1",
                     "positive_class_index": 1 if task == "scope" else None,
                     "num_labels": 2 if task == "scope" else None,
                     "tokenization": {"train": train_statistics, "validation": validation_statistics},
                     "input_length": card["input_length"], "target_length": card.get("target_length"),
                     "max_steps": card["max_steps"], "reviewed_inputs": purpose == "trained",
                     "disclaimer": "Synthetic developer fixture; no ML performance claim and runtime must reject this artifact." if purpose == "smoke_test" else "Validation-loss model selection; scope calibration and frozen test evaluation remain separate."}
    # Resume identity must be validated before modifying any existing run artifact.
    if resume_checkpoint:
        metadata = json.loads((resume_checkpoint / "metadata.json").read_text(encoding="utf-8"))
        validate_identity(identity, {key: metadata.get(key) for key in identity})
    else:
        raw_card = Path(run_card_source).read_bytes() if run_card_source else json_file_bytes(card)
        if hashlib.sha256(raw_card).hexdigest() != run_card_sha256:
            raise ValueError("Run-card bytes differ from their declared hash")
        atomic_bytes(output / "run_card.json", raw_card)
        atomic_json(output / "manifest.json", base_manifest)
    try:
        result = train_loop(model, train, validation, config=config, identity=identity, output_dir=output,
                            resume_checkpoint=resume_checkpoint, stop_after_steps=stop_after_steps)
    except BaseException as exc:
        atomic_json(output / "manifest.json", {**base_manifest,
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)}, "files_sha256": {},
            "recovery": "Resume the last complete exact-state checkpoint; uncommitted work is discarded."})
        raise
    manifest = {**base_manifest, "status": result["status"], "completed_steps": result["global_step"],
                "selection": result["best_validation"], "training": result,
                "updated_at": datetime.now(timezone.utc).isoformat(), "files_sha256": {}}
    if result["status"] == "completed":
        selection = result["best_validation"]
        if not selection:
            raise ValueError("Completed training has no validation-selected checkpoint")
        selected = output / "checkpoints" / selection["checkpoint"]
        metadata = json.loads((selected / "metadata.json").read_text(encoding="utf-8"))
        if metadata["state_sha256"] != file_hash(selected / "state.pt"):
            raise ValueError("Selected checkpoint checksum differs")
        state = torch.load(selected / "state.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        for name, artifact in (("model", model), ("tokenizer", tokenizer)):
            temporary = output / (name + ".tmp-" + uuid4().hex)
            artifact.save_pretrained(temporary)
            destination = output / name
            if destination.exists():
                raise ValueError("Completed model output already exists; use a new run")
            os.replace(temporary, destination)
        manifest["selected_model_state_sha256"] = state_digest(model.state_dict())
        manifest["files_sha256"] = _file_manifest(output)
    atomic_json(output / "manifest.json", manifest)
    return manifest
