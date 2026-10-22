"""Finite three-edit study: profile, develop, seal, then one held-out evaluation.

No final images are encoded during profiling, fitting or validation selection.
Feature-cache reuse requires exact source/code/runtime fingerprints and a full
artifact checksum. Completed reports and models are never overwritten. An
interrupted 25-step projection fit restarts in a new artifact directory; final
inference may resume its sealed feature cache, without refitting or reselection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import statistics
import time

from .data import CLASSES, file_hash
from .model import MODEL_ID, MODEL_REVISION, image_tensor, load_clip
from .study_data import audit_study_manifest
from .study_metrics import grouped_accuracy_interval, select_global_candidate, summarize_study
from .study_protocol import (
    EDIT_WORDS, font_records, iter_photo_tasks, load_study_config, render_study_variant,
    render_text_card, resolve_fonts, text_task_rows,
)

CODE_NAMES = ("study", "study_metrics", "study_protocol", "study_data", "data", "model", "editor", "metrics")
METHODS = ("base", "object_only", "protected", "finetune", "oracle_mask")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _versions():
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "Pillow")}


def _resources(device):
    import torch

    resident = peak = None
    try:
        import ctypes
        if __import__("os").name == "nt":
            from ctypes import wintypes

            class MemoryCounters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

            counters = MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                resident, peak = counters.WorkingSetSize, counters.PeakWorkingSetSize
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if __import__("sys").platform == "darwin" else 1024)
    except (ImportError, OSError, AttributeError):
        pass
    return {"cpu_resident_bytes": resident, "process_peak_resident_bytes": peak,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None}


def _inputs(args):
    config = load_study_config(args.config)
    manifest = _read(args.dataset / "manifest.json")
    audit_study_manifest(manifest, args.dataset, config)
    if manifest["input_hashes"]["config_sha256"] != file_hash(args.config):
        raise ValueError("Study config changed after data preparation")
    if manifest["fonts"] != font_records():
        raise ValueError("Study font assets changed after data preparation")
    return config, manifest


def _fingerprint(args):
    return {"config_sha256": file_hash(args.config),
            "manifest_sha256": file_hash(args.dataset / "manifest.json"),
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "code": {name: file_hash(Path(__file__).with_name(name + ".py")) for name in CODE_NAMES},
            "fonts": font_records(), "runtime_versions": _versions(), "device": args.device}


def _runtime(args):
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; device will not silently change")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _sync(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def _render(rows, dataset, fonts):
    from PIL import Image

    images = []
    for row in rows:
        options = {"font_path": fonts[row["font_id"]], "position": row["position"]}
        if row["task_kind"] == "text":
            image = render_text_card(row["word"], row["template_id"], **options)
        else:
            with Image.open(dataset / row["path"]) as source:
                image, _ = render_study_variant(source, row["variant"], word=row["word"] or "camera", **options)
        images.append(image)
    return images


def _text_features(model, tokenizer, config, kind):
    import torch
    from torch.nn.functional import normalize

    prompts = [config[kind + "_prompts"].format(label=label) for label in CLASSES.values()]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        return normalize(model.text_projection(model.text_model(**encoded).pooler_output), dim=-1).cpu()


def _save_tensor_new(path, payload):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
    _write_new(path.with_suffix(path.suffix + ".checksum.json"), {"sha256": file_hash(path)})


def _load_tensor_checked(path):
    import torch

    expected = _read(path.with_suffix(path.suffix + ".checksum.json"))
    if file_hash(path) != expected["sha256"]:
        raise ValueError("Cached artifact checksum mismatch: " + str(path))
    return torch.load(path, weights_only=True, map_location="cpu")


def _features(args, config, manifest, split, *, model=None, tokenizer=None):
    import torch

    path = args.artifacts / (split + "-features.pt")
    fingerprint = _fingerprint(args)
    expected_rows = iter_photo_tasks(manifest, split, config) + text_task_rows(split, config)
    if path.exists():
        cached = _load_tensor_checked(path)
        if cached["fingerprint"] != fingerprint or cached["rows"] != expected_rows:
            raise ValueError("Feature cache inputs changed; preserve old cache and choose new study directory")
        if cached["features"].shape != (len(expected_rows), cached["base_weight"].shape[1]):
            raise ValueError("Feature cache shape mismatch")
        return cached
    if model is None or tokenizer is None:
        model, tokenizer, _ = load_clip(device=args.device)
    fonts = resolve_fonts()
    features, durations = [], []
    started = time.perf_counter()
    for offset in range(0, len(expected_rows), config["batch_size"]):
        batch = expected_rows[offset:offset + config["batch_size"]]
        pixels = image_tensor(_render(batch, args.dataset, fonts)).to(args.device)
        _sync(args.device)
        before = time.perf_counter()
        with torch.inference_mode():
            features.append(model.vision_model(pixel_values=pixels).pooler_output.cpu())
        _sync(args.device)
        durations.append(time.perf_counter() - before)
    features = torch.cat(features)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Encoder features contain non-finite values")
    payload = {"fingerprint": fingerprint, "features": features, "rows": expected_rows,
               "base_weight": model.visual_projection.weight.detach().cpu().clone(),
               "object_text": _text_features(model, tokenizer, config, "object"),
               "text_text": _text_features(model, tokenizer, config, "text"),
               "measurements": {"encoder_and_render_seconds": time.perf_counter() - started,
                    "encoder_batch_seconds": durations, "encoder_batches": len(durations),
                    "questions": len(expected_rows), **_resources(args.device)}}
    _save_tensor_new(path, payload)
    return payload


def _training_tensors(cache, config, word, weight):
    rows, features = cache["rows"], cache["features"]
    index = {row["task_id"]: position for position, row in enumerate(rows)}
    support = [i for i, row in enumerate(rows) if row["task_kind"] == "object"
               and row["role"] == "support" and row["class_id"] in config["support_classes"][word]
               and row["variant"] == "distractor" and row["word"] == word]
    blank = [index[rows[i]["task_id"].replace("|distractor|", "|blank|")] for i in support]
    retained = [i for i, row in enumerate(rows) if row["task_kind"] == "object"
                and row["role"] == "retention" and row["variant"] == "clean"]
    text = [i for i, row in enumerate(rows) if row["task_kind"] == "text"]
    if (len(support), len(retained), len(text)) != (4, 64, 32):
        raise ValueError("Training tensors differ from the declared four-pair/64-photo/32-card protocol")
    return features[support], features[blank] @ weight.T, features[retained], features[text], [rows[i]["image_id"] for i in support]


def weighted_retention(object_features, text_features, weight):
    """Rows scaled so editor covariance is C_object + weight * C_text exactly."""
    import torch

    if weight < 0:
        raise ValueError("text retention weight must be nonnegative")
    if weight == 0:
        return object_features
    count = len(object_features) + len(text_features)
    return torch.cat([object_features * math.sqrt(count / len(object_features)),
                      text_features * math.sqrt(weight * count / len(text_features))])


def _finetune(base, keys, targets, objects, cards, *, lr, steps, retention_weight):
    import torch

    weight = torch.nn.Parameter(base.clone())
    optimizer = torch.optim.AdamW([weight], lr=lr, weight_decay=0)
    object_targets, card_targets = objects @ base.T, cards @ base.T
    log, durations = [], []
    started = time.perf_counter()
    for step in range(steps):
        before = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        fitting = (keys @ weight.T - targets).square().mean()
        object_loss = (objects @ weight.T - object_targets).square().mean()
        text_loss = (cards @ weight.T - card_targets).square().mean()
        loss = fitting + retention_weight * (object_loss + text_loss)
        loss.backward()
        optimizer.step()
        durations.append(time.perf_counter() - before)
        log.append({"step": step + 1, "support_mse": float(fitting.detach()),
                    "object_retention_mse": float(object_loss.detach()),
                    "text_retention_mse": float(text_loss.detach()), "loss": float(loss.detach())})
    return weight.detach(), {"steps": steps, "seconds": time.perf_counter() - started,
                             "step_seconds": durations, "training_log": log}


def _predict(features, weight, text):
    from torch.nn.functional import normalize
    return (normalize(features @ weight.T, dim=-1) @ text.T).argmax(dim=-1).tolist()


def _observations(cache, weights):
    objects = [i for i, row in enumerate(cache["rows"]) if row["task_kind"] == "object"]
    cards = [i for i, row in enumerate(cache["rows"]) if row["task_kind"] == "text"]
    predictions = {}
    for method, weight in weights.items():
        predictions[method] = {}
        for indices, key in ((objects, "object_text"), (cards, "text_text")):
            values = _predict(cache["features"][indices], weight, cache[key])
            predictions[method].update({index: list(CLASSES)[value] for index, value in zip(indices, values)})
    rows = [{**row, **{method: values[index] for method, values in predictions.items()}}
            for index, row in enumerate(cache["rows"])]
    by_task = {row["task_id"]: row for row in rows}
    for row in rows:
        # This privileged baseline knows exactly where the artificial label is.
        blank = (by_task[row["task_id"].replace("|distractor|", "|blank|")]
                 if row["task_kind"] == "object" and row["variant"] == "distractor" else row)
        row["oracle_mask"] = blank["base"]
    return [rows[i] for i in objects], [rows[i] for i in cards]


def profile(args):
    config, manifest = _inputs(args)
    output = args.reports / "profile.json"
    if output.exists():
        raise FileExistsError("Profile exists; preserve measured evidence")
    _runtime(args)
    import torch
    from .editor import fit_projection_patch

    model, tokenizer, load_seconds = load_clip(device=args.device)
    tasks = iter_photo_tasks(manifest, "train", config)
    fonts, durations = resolve_fonts(), []
    pooled_batches = []
    for batch_index in range(5):
        batch = tasks[batch_index * config["batch_size"]:(batch_index + 1) * config["batch_size"]]
        pixels = image_tensor(_render(batch, args.dataset, fonts)).to(args.device)
        _sync(args.device)
        before = time.perf_counter()
        with torch.inference_mode():
            pooled = model.vision_model(pixel_values=pixels).pooler_output.cpu()
        _sync(args.device)
        if batch_index >= 2:
            durations.append(time.perf_counter() - before)
        pooled_batches.append(pooled)
    features = torch.cat(pooled_batches)
    base = model.visual_projection.weight.detach().cpu().clone()
    keys, targets = features[:4], features[4:8] @ base.T
    proxy_objects, proxy_cards = features[8:24].repeat(4, 1), features[24:40].repeat(2, 1)
    before = time.perf_counter()
    fit_projection_patch(base, keys, targets, weighted_retention(proxy_objects, proxy_cards, 1))
    fit_seconds = time.perf_counter() - before
    _, fine = _finetune(base, keys, targets, proxy_objects, proxy_cards,
                         lr=config["projection_finetune_learning_rates"][0], steps=4, retention_weight=1)
    counts = {split: len(iter_photo_tasks(manifest, split, config)) + len(text_task_rows(split, config))
              for split in ("train", "validation", "final")}
    median_batch = statistics.median(durations)
    result = {"status": "measured_preflight_profile", "created_at": _now(), "fingerprint": _fingerprint(args),
              "question": "Are this fixed encoder, three-edit grid and 25-step matched baselines practical on the measured device?",
              "model": {"id": MODEL_ID, "revision": MODEL_REVISION}, "device": args.device,
              "model_load_seconds": load_seconds, "batch_size": config["batch_size"],
              "warmup_batches": 2, "timed_encoder_batches": 3, "encoder_batch_seconds": durations,
              "editor_fit_seconds": fit_seconds, "projection_finetune_four_step_profile": fine,
              "resources": _resources(args.device), "questions_by_split": counts,
              "planning_estimates_seconds": {
                  "development_encoder": sum(math.ceil(counts[s] / config["batch_size"]) for s in ("train", "validation")) * median_batch,
                  "final_encoder": math.ceil(counts["final"] / config["batch_size"]) * median_batch,
                  "nine_independent_plus_eighteen_sequential_editor_fits": fit_seconds * 27,
                  "six_finetunes_25_steps": statistics.median(fine["step_seconds"]) * 25 * 6},
              "finite_stop": "Development: 9 independent edits, 6 x 25 FT updates, 18 sequential edits; final: 648 questions, no fitting",
              "resume": "Complete feature caches reuse exact fingerprints/checksums. Interrupted fitting restarts in a new output directory; no exact optimizer-resume claim.",
              "limitations": ["Three steady encoder batches and one fit/four optimizer updates; estimates are uncertain",
                               "Fit/optimizer profile uses four keys, 64 repeated object features and 32 repeated proxy card features for matching tensor sizes; these are resource measurements, not semantic training examples",
                               "Encoder timing excludes rendering, disk writes and report generation",
                               "No validation/final accuracy was measured in this profile"]}
    estimates = result["planning_estimates_seconds"]
    estimates["development_total_including_one_model_load"] = (
        load_seconds + estimates["development_encoder"]
        + estimates["nine_independent_plus_eighteen_sequential_editor_fits"]
        + estimates["six_finetunes_25_steps"])
    estimates["final_total_including_one_model_load"] = load_seconds + estimates["final_encoder"]
    _write_new(output, result)
    return result


def develop(args):
    development_started = time.perf_counter()
    config, manifest = _inputs(args)
    output = args.reports / "development.json"
    if output.exists() or (args.artifacts / "models").exists():
        raise FileExistsError("Development artifacts exist; no silent refit or overwrite")
    measured = _read(args.reports / "profile.json")
    if measured["fingerprint"] != _fingerprint(args):
        raise ValueError("Profile is stale; inputs/runtime differ from the measured preflight")
    _runtime(args)
    from .editor import fit_projection_patch, projection_sha256, save_projection_patch

    model, tokenizer, load_seconds = load_clip(device=args.device)
    train = _features(args, config, manifest, "train", model=model, tokenizer=tokenizer)
    validation = _features(args, config, manifest, "validation", model=model, tokenizer=tokenizer)
    base = train["base_weight"]
    artifacts, candidate_results, observations, measurements = {}, {}, {}, {}
    protected_ids = [row["candidate_id"] for row in config["editing_candidates"] if row["text_retention_weight"] > 0]
    finetune_ids = ["finetune-" + str(index) for index in range(len(config["projection_finetune_learning_rates"]))]
    for word in EDIT_WORDS:
        keys, targets, objects, cards, support_ids = _training_tensors(train, config, word, base)
        weights = {"base": base}
        artifacts[word] = {}
        for candidate in config["editing_candidates"]:
            candidate_id = candidate["candidate_id"]
            retained = weighted_retention(objects, cards, candidate["text_retention_weight"])
            before = time.perf_counter()
            patch = fit_projection_patch(base, keys, targets, retained,
                covariance_ridge=candidate["covariance_ridge"], fit_ridge=candidate["fit_ridge"],
                metadata={"word": word, "candidate": candidate, "support_image_ids": support_ids,
                          "fingerprint": _fingerprint(args), "purpose": "development_candidate"})
            measurements[word + "/" + candidate_id] = {"fit_seconds": time.perf_counter() - before}
            path = args.artifacts / "models" / word / (candidate_id + ".pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            save_projection_patch(patch, path)
            artifacts[word][candidate_id] = {"path": str(path.resolve()), "kind": "projection_patch", "sha256": file_hash(path)}
            weights[candidate_id] = patch.edited_weight()
        for index, lr in enumerate(config["projection_finetune_learning_rates"]):
            candidate_id = "finetune-" + str(index)
            weight, log = _finetune(base, keys, targets, objects, cards, lr=lr,
                steps=config["projection_finetune_steps"], retention_weight=config["projection_finetune_retention_weight"])
            path = args.artifacts / "models" / word / (candidate_id + ".pt")
            _save_tensor_new(path, {"weight": weight, "learning_rate": lr, "log": log,
                                    "base_sha256": projection_sha256(base), "fingerprint": _fingerprint(args)})
            artifacts[word][candidate_id] = {"path": str(path.resolve()), "kind": "full_weight", "sha256": file_hash(path)}
            weights[candidate_id], measurements[word + "/" + candidate_id] = weight, log
        obj, txt = _observations(validation, weights)
        observations[word] = {"object": obj, "text": txt}
        candidate_results[word] = {method: summarize_study(obj, txt, method, word) for method in weights}
        candidate_results[word]["oracle_mask"] = summarize_study(obj, txt, "oracle_mask", word)
    protected_selection = select_global_candidate(protected_ids, {
        candidate: [candidate_results[word][candidate] for word in EDIT_WORDS] for candidate in protected_ids})
    fine_selection = select_global_candidate(finetune_ids, {
        candidate: [candidate_results[word][candidate] for word in EDIT_WORDS] for candidate in finetune_ids})
    selected = {word: {"object_only": artifacts[word]["object-only"],
                       "protected": artifacts[word][protected_selection["selected"]],
                       "finetune": artifacts[word][fine_selection["selected"]]} for word in EDIT_WORDS}
    selected_config = next(row for row in config["editing_candidates"] if row["candidate_id"] == protected_selection["selected"])
    sequential = []
    for order in itertools.permutations(EDIT_WORDS):
        current, stages, patches = base.clone(), [], []
        for stage, word in enumerate(order, 1):
            keys, targets, objects, cards, support_ids = _training_tensors(train, config, word, current)
            fit_started = time.perf_counter()
            patch = fit_projection_patch(current, keys, targets,
                weighted_retention(objects, cards, selected_config["text_retention_weight"]),
                covariance_ridge=selected_config["covariance_ridge"], fit_ridge=selected_config["fit_ridge"],
                metadata={"word": word, "order": list(order), "stage": stage,
                          "support_image_ids": support_ids, "fingerprint": _fingerprint(args),
                          "target_policy": "matched blank outputs of current parent projection"})
            fit_seconds = time.perf_counter() - fit_started
            path = args.artifacts / "sequential" / ("-".join(order)) / f"stage-{stage}-{word}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            save_projection_patch(patch, path)
            current = patch.edited_weight()
            patches.append(patch)
            stages.append({"word": word, "stage": stage, "path": str(path.resolve()),
                           "sha256": file_hash(path), "before_sha256": patch.before_sha256,
                           "after_sha256": patch.after_sha256, "fit_seconds": fit_seconds})
        for patch in reversed(patches):
            if projection_sha256(current) != patch.after_sha256:
                raise RuntimeError("Sequential rollback state mismatch")
            current = patch.before_weight.clone()
        if projection_sha256(current) != projection_sha256(base):
            raise RuntimeError("Sequential rollback did not restore exact original projection")
        sequential.append({"order": list(order), "stages": stages, "exact_rollback": True})
    result = {"status": "completed_validation_selection", "created_at": _now(),
              "fingerprint": _fingerprint(args), "question": config["question"],
              "selected": selected, "all_artifacts": artifacts, "sequential": sequential,
              "selection": {"protected": protected_selection, "finetune": fine_selection},
              "validation_results": candidate_results, "validation_observations": observations,
              "measurements": {"load_seconds": load_seconds, "train_features": train["measurements"],
                  "validation_features": validation["measurements"], "fitting": measurements,
                  "total_development_seconds": time.perf_counter() - development_started,
                  "sequential_fit_seconds": sum(stage["fit_seconds"] for order in sequential for stage in order["stages"]),
                  "resources": _resources(args.device)},
              "final_images_encoded": False, "final_labels_used_for_selection": False,
              "finite_stop": "9 independent edits, 6 x 25 FT updates, and all 18 sequential edits completed",
              "limitations": config["limitations"]}
    _write_new(output, result)
    return result


def _model_files():
    from huggingface_hub import try_to_load_from_cache

    paths = {}
    for name in ("config.json", "model.safetensors", "pytorch_model.bin", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "special_tokens_map.json", "tokenizer.json"):
        path = try_to_load_from_cache(MODEL_ID, name, revision=MODEL_REVISION)
        if isinstance(path, str) and Path(path).is_file():
            paths["model/" + name] = Path(path)
    if (not {"model/config.json", "model/vocab.json", "model/merges.txt", "model/tokenizer_config.json"}.issubset(paths)
            or not any(key in paths for key in ("model/model.safetensors", "model/pytorch_model.bin"))):
        raise FileNotFoundError("Pinned local model weights/config are unavailable for sealing")
    return paths


def _bindings(args, development):
    paths = {"config": args.config, "manifest": args.dataset / "manifest.json",
             "profile": args.reports / "profile.json", "development": args.reports / "development.json",
             "train_features": args.artifacts / "train-features.pt",
             "validation_features": args.artifacts / "validation-features.pt"}
    for name in CODE_NAMES:
        paths["code/" + name] = Path(__file__).with_name(name + ".py")
    for name, path in resolve_fonts().items():
        paths["font/" + name] = path
    for word, candidates in development["all_artifacts"].items():
        for candidate, record in candidates.items():
            paths["candidate/" + word + "/" + candidate] = Path(record["path"])
    for order in development["sequential"]:
        for stage in order["stages"]:
            paths["sequential/" + "-".join(order["order"]) + "/" + str(stage["stage"])] = Path(stage["path"])
    paths.update(_model_files())
    return {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in paths.items()}


def load_and_validate_seal(path):
    """Model-free validation rejects changed or incomplete frozen evidence."""
    sealed = _read(path)
    required = {"config", "manifest", "profile", "development", "train_features", "validation_features"}
    required.update("code/" + name for name in CODE_NAMES)
    required.update("font/" + name for name in ("train_sans", "validation_serif", "final_sans_bold"))
    required.update(("model/config.json", "model/vocab.json", "model/merges.txt", "model/tokenizer_config.json"))
    if (sealed.get("format_version") != 2 or sealed.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}
            or not required.issubset(sealed.get("bindings", {}))):
        raise ValueError("Unknown or incomplete study seal")
    if sealed.get("methods") != list(METHODS) or sealed.get("edit_words") != list(EDIT_WORDS):
        raise ValueError("Sealed methods or edit words changed")
    bindings = sealed["bindings"]
    if not any(name in bindings for name in ("model/model.safetensors", "model/pytorch_model.bin")):
        raise ValueError("Study seal is missing the actual pinned model weights")
    for word in EDIT_WORDS:
        for candidate in ("object-only", "protected-text-1", "protected-text-4", "finetune-0", "finetune-1"):
            if "candidate/" + word + "/" + candidate not in bindings:
                raise ValueError("Study seal is missing a declared independent candidate")
    if (len(sealed.get("sequential", [])) != 6
            or {tuple(row.get("order", [])) for row in sealed["sequential"]} != set(itertools.permutations(EDIT_WORDS))):
        raise ValueError("Study seal must contain all six declared edit orders")
    for order in sealed["sequential"]:
        if [stage.get("word") for stage in order.get("stages", [])] != order["order"]:
            raise ValueError("Sequential stage words differ from the declared order")
        for stage in order["stages"]:
            if "sequential/" + "-".join(order["order"]) + "/" + str(stage["stage"]) not in bindings:
                raise ValueError("Study seal is missing a sequential-stage artifact")
    if sealed.get("runtime_versions") != _versions():
        raise ValueError("Runtime versions changed after sealing")
    for name, binding in sealed["bindings"].items():
        if file_hash(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("Sealed input changed: " + name)
    development = _read(Path(bindings["development"]["path"]))
    if sealed.get("selected") != development.get("selected") or sealed["sequential"] != development.get("sequential"):
        raise ValueError("Seal selections differ from the hash-bound development evidence")
    return sealed


def seal(args):
    config, manifest = _inputs(args)
    output = args.reports / "final-seal.json"
    if output.exists() or (args.reports / "final.json").exists() or (args.artifacts / "final-features.pt").exists():
        raise FileExistsError("Final evidence exists; resealing is forbidden")
    development = _read(args.reports / "development.json")
    if development["fingerprint"] != _fingerprint(args):
        raise ValueError("Development evidence is stale")
    result = {"format_version": 2, "created_at": _now(), "status": "sealed_before_final_inference",
              "model": {"id": MODEL_ID, "revision": MODEL_REVISION}, "methods": list(METHODS),
              "edit_words": list(EDIT_WORDS), "runtime_versions": _versions(), "device": args.device,
              "fingerprint": _fingerprint(args), "bindings": _bindings(args, development),
              "selected": development["selected"], "sequential": development["sequential"],
              "final_photos": sum(row["split"] == "final" for row in manifest["images"]),
              "final_object_questions": len(iter_photo_tasks(manifest, "final", config)),
              "final_text_questions": len(text_task_rows("final", config)),
              "outputs": {"report": str((args.reports / "final.json").resolve()),
                          "features": str((args.artifacts / "final-features.pt").resolve())},
              "stopping_condition": "One sealed 648-question inference dataset; score independent and all 18 already-fitted sequential states; no optimization",
              "review_policy": "Automated source-label, exact-content, rendering and schema audit; no independent human label review"}
    _write_new(output, result)
    load_and_validate_seal(output)
    return result


def _load_weight(record, base):
    from .editor import load_projection_patch, projection_sha256

    path = Path(record["path"])
    if file_hash(path) != record["sha256"]:
        raise ValueError("Selected model artifact changed")
    if record.get("kind", "projection_patch") == "projection_patch":
        patch = load_projection_patch(path)
        if patch.before_sha256 != projection_sha256(base):
            raise ValueError("Selected patch has the wrong parent projection")
        return patch.edited_weight()
    payload = _load_tensor_checked(path)
    if payload["base_sha256"] != projection_sha256(base):
        raise ValueError("Selected fine-tuning model has the wrong original projection")
    return payload["weight"]


def finalize(args):
    final_started = time.perf_counter()
    output = args.reports / "final.json"
    if output.exists():
        raise FileExistsError("Frozen final report exists; no repeat evaluation or overwrite")
    sealed = load_and_validate_seal(args.reports / "final-seal.json")
    config, manifest = _inputs(args)
    if sealed["fingerprint"] != _fingerprint(args) or sealed["device"] != args.device:
        raise ValueError("Final arguments differ from the sealed protocol")
    if sealed["outputs"] != {"report": str(output.resolve()), "features": str((args.artifacts / "final-features.pt").resolve())}:
        raise ValueError("Final output paths differ from seal")
    _runtime(args)
    from .editor import load_projection_patch, projection_sha256

    final = _features(args, config, manifest, "final")
    base = final["base_weight"]
    train = _load_tensor_checked(args.artifacts / "train-features.pt")
    if projection_sha256(base) != projection_sha256(train["base_weight"]):
        raise ValueError("Final encoder checkpoint projection differs from trained parent")
    results, observations, intervals = {}, {}, {}
    for word in EDIT_WORDS:
        weights = {"base": base, **{method: _load_weight(record, base) for method, record in sealed["selected"][word].items()}}
        objects, cards = _observations(final, weights)
        results[word] = {method: summarize_study(objects, cards, method, word) for method in METHODS}
        observations[word] = {"object": objects, "text": cards}
        targeted = [row for row in objects if row["variant"] == "distractor" and row["word"] == word]
        intervals[word] = {method: {
            "target_distractor": grouped_accuracy_interval(targeted, method, seed=config["bootstrap_seed"], samples=config["bootstrap_samples"]),
            "legitimate_text_macro": grouped_accuracy_interval(cards, method, group_key="class_id", seed=config["bootstrap_seed"], samples=config["bootstrap_samples"])}
            for method in METHODS if method != "base"}
    sequential = []
    for order in sealed["sequential"]:
        stages, current, patches = [], base.clone(), []
        for stage in order["stages"]:
            path = Path(stage["path"])
            patch = load_projection_patch(path)
            if file_hash(path) != stage["sha256"] or projection_sha256(current) != patch.before_sha256:
                raise ValueError("Sequential stage artifact/parent differs from seal")
            current = patch.edited_weight()
            patches.append(patch)
            objects, cards = _observations(final, {"base": base, "sequential": current})
            stages.append({"stage": stage["stage"], "applied_word": stage["word"],
                           "results": {word: summarize_study(objects, cards, "sequential", word) for word in EDIT_WORDS}})
        for patch in reversed(patches):
            if projection_sha256(current) != patch.after_sha256:
                raise RuntimeError("Sequential final rollback parent mismatch")
            current = patch.before_weight.clone()
        if projection_sha256(current) != projection_sha256(base):
            raise RuntimeError("Sequential final rollback failed exact restoration")
        sequential.append({"order": order["order"], "stages": stages, "exact_rollback": True})
    result = {"status": "completed_frozen_selectivity_evaluation", "created_at": _now(),
              "seal_sha256": file_hash(args.reports / "final-seal.json"),
              "question": config["question"], "model": sealed["model"], "methods": list(METHODS),
              "training_performed": False, "final_based_selection_performed": False,
              "dataset": {"id": manifest["dataset_id"], "final_photos": sealed["final_photos"],
                          "object_questions": len(next(iter(observations.values()))["object"]),
                          "text_questions": len(next(iter(observations.values()))["text"]),
                          "human_reviewed": False, "denominator_policy": "Every attempted question counted; no abstention exclusions"},
              "selection": _read(args.reports / "development.json")["selection"],
              "results": results, "paired_uncertainty_vs_base": intervals,
              "sequential": sequential, "observations": observations,
              "measurements": {**final["measurements"], "resources_after_scoring": _resources(args.device),
                               "total_final_seconds": time.perf_counter() - final_started},
              "features_sha256": file_hash(args.artifacts / "final-features.pt"),
              "limitations": config["limitations"] + [
                  "Oracle masking is privileged and task-aware: mask the known synthetic object-overlay region, leave legitimate text cards unchanged; this is not a real overlay detector or learned editor",
                  "Intervals group photo augmentations together; text macro intervals have only eight word groups",
                  "Projection editing is not a faithful reproduction of a named multimodal editing algorithm"]}
    _write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("profile", "develop", "seal", "finalize"))
    parser.add_argument("--dataset", type=Path, default=Path("data/generated/visionpatch/study-v2"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/visionpatch/study-v2"))
    parser.add_argument("--reports", type=Path, default=Path("reports/visionpatch/study-v2"))
    parser.add_argument("--config", type=Path, default=Path("configs/visionpatch-study-v2.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = globals()[args.phase](args)
    print(json.dumps({"status": result["status"], "reports": str(args.reports),
                      "selection": result.get("selection"), "results": result.get("results"),
                      "planning_estimates_seconds": result.get("planning_estimates_seconds")}, indent=2))


if __name__ == "__main__":
    main()
