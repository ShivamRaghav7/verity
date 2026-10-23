"""An isolated, explicitly adapted VLKEB MEND/BLIP-2 development bridge.

Original algorithm/model code is loaded from the pinned upstream checkout and
is not vendored here. The released learned gradient transforms and loss are
reused. We calculate their factors without upstream's functional-model mutation
tail, and apply the executor's ``V.T @ U * learned_lr * lr_scale`` update once.
Generation and transactional patch persistence are our application adapters.
Tiny-model tests validate adapters; they do not establish checkpoint accuracy.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import importlib
import importlib.machinery
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import torch

from .generation import generate_blip2_answer


UPSTREAM_COMMIT = "10951b7b3788928f578b73f07eac9e1eaa0316f3"
PATCH_FORMAT = "editguard-vlkeb-mend-patch-v1"


def tensor_hash(tensor: torch.Tensor) -> str:
    """Hash dtype, shape and exact bytes, including half-precision bit patterns."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ParameterPatch:
    """One transactional patch, saved as before/after tensors for exact reversal.

    Restoring saved originals is intentional: subtracting a float16 update does
    not reliably recover original bytes. This first smoke run allows one active
    patch; it makes no claims about concurrent or lifelong editing.
    """

    def __init__(self, model, parameter_names, *, identity=None):
        self.model = model
        self.names = tuple(parameter_names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Patch parameters must be nonempty and unique")
        parameters = dict(model.named_parameters())
        if set(self.names) - parameters.keys():
            raise ValueError("An editable parameter is missing from the model")
        self.identity = dict(identity or {})
        self.active = None

    def weight_hashes(self):
        parameters = dict(self.model.named_parameters())
        return {name: tensor_hash(parameters[name]) for name in self.names}

    def _snapshot(self):
        parameters = dict(self.model.named_parameters())
        return {name: parameters[name].detach().cpu().clone() for name in self.names}

    def _copy(self, values):
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name in self.names:
                parameters[name].copy_(values[name].to(parameters[name].device))

    def apply(self, factors, learned_rates, *, lr_scale=1.0, metadata=None):
        if self.active is not None:
            raise RuntimeError("Roll back the active patch before applying another edit")
        if tuple(factors) != self.names or len(learned_rates) != len(self.names):
            raise ValueError("Factor/rate order must match the declared editable parameters")
        if not torch.isfinite(torch.as_tensor(lr_scale)).all():
            raise ValueError("lr_scale must be finite")
        before = self._snapshot()
        parameters = dict(self.model.named_parameters())
        norms = {}
        try:
            with torch.no_grad():
                for index, name in enumerate(self.names):
                    parameter = parameters[name]
                    u, v = factors[name]
                    if u.ndim != 2 or v.ndim != 2 or u.shape[0] != v.shape[0]:
                        raise ValueError(f"Invalid factor dimensions for {name}")
                    if (v.shape[1], u.shape[1]) != tuple(parameter.shape):
                        raise ValueError(f"Factor shape does not match {name}")
                    # Same matrix orientation/sign/rate as the upstream executor.
                    delta = (v.to(parameter.device).T @ u.to(parameter.device))
                    delta = delta * learned_rates[index].to(parameter.device) * lr_scale
                    if not bool(torch.isfinite(delta).all()):
                        raise RuntimeError(f"Non-finite edit update for {name}")
                    norms[name] = float(delta.float().norm().item())
                    parameter.add_(delta.to(parameter.device))
                    if not bool(torch.isfinite(parameter).all()):
                        raise RuntimeError(f"Edit overflowed parameter {name}")
                    del delta
            after = self._snapshot()
            self.active = {"format": PATCH_FORMAT, "identity": self.identity,
                           "names": list(self.names), "before": before,
                           "after": after, "metadata": dict(metadata or {})}
        except BaseException:
            self._copy(before)
            raise
        return {"update_norms": norms, "weight_hashes": self.weight_hashes(),
                "changed_parameters": [name for name in self.names
                                       if tensor_hash(before[name]) != tensor_hash(after[name])]}

    def save(self, path):
        if self.active is None:
            raise RuntimeError("There is no active patch to save")
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite a patch: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        if temporary.exists():
            raise FileExistsError(f"An incomplete patch file already exists: {temporary}")
        try:
            torch.save(self.active, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {"path": str(path), "sha256": _file_hash(path),
                "bytes": path.stat().st_size}

    def load(self, path):
        if self.active is not None:
            raise RuntimeError("Roll back the active patch before loading another patch")
        # Our own patch format contains only tensors and primitive metadata.
        patch = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(patch, dict) or patch.get("format") != PATCH_FORMAT:
            raise ValueError("Unsupported patch format")
        if patch.get("identity") != self.identity or patch.get("names") != list(self.names):
            raise ValueError("Patch belongs to a different model/editor configuration")
        parameters = dict(self.model.named_parameters())
        for section in ("before", "after"):
            values = patch.get(section)
            if not isinstance(values, dict) or set(values) != set(self.names):
                raise ValueError("Patch has missing or unexpected parameters")
            for name in self.names:
                value, parameter = values[name], parameters[name]
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                    raise ValueError(f"Patch shape does not match {name}")
                if value.dtype != parameter.dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Patch dtype/values do not match {name}")
        hashes = self.weight_hashes()
        if any(hashes[name] != tensor_hash(patch["before"][name]) for name in self.names):
            raise ValueError("Current base weights do not match the patch's originals")
        try:
            self._copy(patch["after"])
            loaded_hashes = self.weight_hashes()
            if any(loaded_hashes[name] != tensor_hash(patch["after"][name])
                   for name in self.names):
                raise RuntimeError("Reloaded patch failed its exact-weight check")
        except BaseException:
            self._copy(patch["before"])
            raise
        self.active = patch
        return {"weight_hashes": self.weight_hashes(), "metadata": patch.get("metadata", {})}

    def rollback(self):
        if self.active is None:
            raise RuntimeError("There is no active patch to roll back")
        before = self.active["before"]
        self._copy(before)
        hashes = self.weight_hashes()
        exact = all(hashes[name] == tensor_hash(before[name]) for name in self.names)
        if not exact:
            raise RuntimeError("Rollback failed its exact-weight check")
        self.active = None
        return {"exact_weight_restore": True, "weight_hashes": hashes}


def extract_mend_factors(editor, batch):
    """Reuse MEND's loss and released learned transforms, without mutation tail.

    Corresponds to upstream MEND.py ``edit`` through ``transformed_factors``.
    Evaluation mode freezes its running normalization statistics. No optimizer
    runs, auxiliary weights are frozen, and the base model is not changed here.
    """
    editor.eval()
    editor.model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
            outputs = editor.model(batch)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
            loss = editor.edit_loss_fn(editor.config, logits, batch["labels"])["nll"]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("The MEND inner loss is not finite")
            loss.backward()
        parameters = dict(editor.model.named_parameters())
        factors = {}
        with torch.no_grad():
            for name in editor.config.inner_params:
                parameter = parameters[name]
                if not hasattr(parameter, "__x__") or not hasattr(parameter, "__delta__"):
                    raise RuntimeError(f"MEND did not capture activations/gradients for {name}")
                if editor.config.shared:
                    shape = editor.get_shape(parameter)
                    index = editor.shape_dict[shape].index(name)
                    transform = editor.mend[str(tuple(shape))]
                    u, v = transform(parameter.__x__, parameter.__delta__, index)
                else:
                    transform = editor.mend[name.replace(".", "#")]
                    u, v = transform(parameter.__x__, parameter.__delta__)
                factors[name] = (u.detach().cpu(), v.detach().cpu())
        return factors, {"inner_nll": float(loss.detach().item()),
                         "factor_rows": {name: u.shape[0] for name, (u, _) in factors.items()}}
    finally:
        editor.model.zero_grad(set_to_none=True)
        clear_hook_tensors(editor.model)


def clear_hook_tensors(model):
    for parameter in model.parameters():
        for attribute in ("__x__", "__delta__"):
            if hasattr(parameter, attribute):
                delattr(parameter, attribute)


def _namespace_package(name: str, path: Path):
    if name in sys.modules:
        return
    package = ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    package.__spec__.submodule_search_locations = [str(path)]
    sys.modules[name] = package


def _upstream_modules(root: Path):
    """Import selected originals without executing EasyEdit's eager __init__s."""
    prefix = "_editguard_vlkeb_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    base = root / "easyeditor"
    for suffix in ("", "trainer", "trainer.algs", "trainer.algs.higher_utils",
                   "trainer.blip2_models", "trainer.blip2_models.common",
                   "dataset", "dataset.processor"):
        _namespace_package(prefix + ("." + suffix if suffix else ""),
                           base.joinpath(*suffix.split(".")) if suffix else base)
    return (importlib.import_module(prefix + ".trainer.blip2_models.blip2_opt"),
            importlib.import_module(prefix + ".trainer.algs.MEND"),
            importlib.import_module(prefix + ".dataset.processor.blip_processors"))


@contextmanager
def _low_memory_opt_loader(opt_class, report):
    """Change loading allocation only; keep original class, weights and dtype."""
    original = opt_class.from_pretrained
    local_descriptor = opt_class.__dict__.get("from_pretrained")

    def load(*args, **kwargs):
        kwargs["low_cpu_mem_usage"] = True
        kwargs["output_loading_info"] = True
        model, info = original(*args, **kwargs)
        report.update(info)
        missing = set(info.get("missing_keys", ()))
        if missing == {"lm_head.weight"} and model.config.tie_word_embeddings:
            if model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr():
                missing.clear()
        if missing or info.get("mismatched_keys") or info.get("error_msgs"):
            raise RuntimeError(f"OPT checkpoint did not completely populate its model: {info}")
        return model

    opt_class.from_pretrained = staticmethod(load)
    try:
        yield
    finally:
        if local_descriptor is None:
            delattr(opt_class, "from_pretrained")
        else:
            opt_class.from_pretrained = local_descriptor


def stage_vision_to_device(encoder, normalization, device):
    """Move loaded vision weights before constructing the language model.

    This allocation-only adaptation addresses observed host-RAM exhaustion in a
    fresh Colab reload worker. No dtype conversion, quantization or architecture
    change is requested; the final complete model would move to the same device.
    """
    return encoder.to(device), normalization.to(device)


def target_label_spec(tokenizer, question, target, *, target_alignment="release"):
    """Keep release labels or explicitly align labels to the actual input suffix.

    Context alignment is a declared experimental adaptation. A prefix that
    retokenizes at the boundary is rejected rather than guessing which question
    tokens should receive the answer loss.
    """
    if target_alignment not in ("release", "context"):
        raise ValueError("target_alignment must be 'release' or 'context'")
    standalone = tokenizer.encode(target, add_special_tokens=False)
    prefix = tokenizer.encode(question)
    joined = tokenizer.encode(question + " " + target)
    prefix_matches = joined[:len(prefix)] == prefix
    context = joined[len(prefix):] if prefix_matches else None
    if target_alignment == "context" and not prefix_matches:
        raise ValueError("Question prefix retokenizes in the joined input; context alignment is ambiguous")
    selected = standalone if target_alignment == "release" else context
    if not selected or len(selected) > 64:
        raise ValueError("The development slice supports targets of 1 to 64 selected tokens")
    prompt_length = (len(tokenizer.encode(question, add_special_tokens=False))
                     if target_alignment == "release" else len(prefix))
    return {"target_ids": selected, "prompts_len": prompt_length,
            "metadata": {"target_alignment": target_alignment,
                         "standalone_target_token_count": len(standalone),
                         "context_target_token_count": len(context) if context is not None else None,
                         "selected_target_token_count": len(selected),
                         "question_prefix_matches_joined_input": prefix_matches,
                         "prompt_length_includes_default_special_tokens": target_alignment == "context",
                         "context_alignment_is_experimental_adaptation": target_alignment == "context"}}


class VLKEBBackend:
    def __init__(self, model, editor, processor, *, identity, compatibility):
        self.model, self.editor, self.processor = model, editor, processor
        self.compatibility = compatibility
        self.patches = ParameterPatch(model, editor.config.inner_params, identity=identity)

    def _image(self, image):
        if isinstance(image, (str, Path)):
            from PIL import Image

            with Image.open(image) as opened:
                image = opened.convert("RGB")
        if not isinstance(image, torch.Tensor):
            image = self.processor(image.convert("RGB"))
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if tuple(image.shape) != (1, 3, 364, 364):
            raise ValueError("The released BLIP-2 configuration expects a 1x3x364x364 image")
        return image.to(self.model.device)

    def generate(self, image, question, max_new_tokens=32):
        try:
            return generate_blip2_answer(self.model, None if image is None else self._image(image), question,
                                        max_new_tokens=max_new_tokens)
        finally:
            clear_hook_tensors(self.model)

    def edit(self, image, question, target, *, target_alignment="release"):
        if self.patches.active is not None:
            raise RuntimeError("Roll back the active patch before applying another edit")
        batch = self._edit_batch(image, question, target, target_alignment=target_alignment)
        factors, information = extract_mend_factors(self.editor, batch)
        result = self.patches.apply(factors, self.editor.edit_lrs.detach().cpu(),
                                    lr_scale=float(self.editor.config.lr_scale),
                                    metadata={"question": question, "target": target,
                                              **batch["alignment_metadata"],
                                              "algorithm": "VLKEB released MEND; factor adapter"})
        result.update(information)
        result.update(batch["alignment_metadata"])
        return result

    def _edit_batch(self, image, question, target, *, target_alignment="release"):
        """Release collation by default; explicit context-label variant when asked."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty question is required")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("A nonempty approved target is required")
        tokenizer = self.model.opt_tokenizer
        question_tokens = tokenizer.encode(question, add_special_tokens=False)
        if len(question_tokens) > 256:
            raise ValueError("The development slice supports questions of at most 256 tokens")
        spec = target_label_spec(tokenizer, question, target, target_alignment=target_alignment)
        # Default labels remain exactly the standalone tokens in the released
        # CaptionDataset.collate_fn. The optional variant uses actual suffix IDs.
        labels = torch.tensor([spec["target_ids"]], dtype=torch.long, device=self.model.device)
        batch = {"image": self._image(image), "text_input": [question + " " + target],
                 "labels": labels, "prompts_len": [spec["prompts_len"]],
                 "alignment_metadata": spec["metadata"]}
        return batch

    def score_target(self, image, question, target, *, target_alignment="release"):
        """Measure upstream teacher-forced target tokens, never free generation.

        The full target is deliberately part of the input here. This diagnostic
        can separate an editing/tokenization failure from a decoding failure,
        but cannot establish that the model independently generated an answer.
        """
        batch = self._edit_batch(image, question, target, target_alignment=target_alignment)
        self.editor.eval()
        try:
            with torch.no_grad():
                output = self.model(batch)
                logits = output if isinstance(output, torch.Tensor) else output.logits
                score = self.editor.edit_loss_fn(self.editor.config, logits, batch["labels"])
            pred_ids = score["pred_ids"].detach().cpu().tolist()[0]
            target_ids = score["targ_ids"].detach().cpu().tolist()[0]
            return {"conditioning": "teacher_forced_full_target_in_input",
                    "is_generated_answer": False,
                    **batch["alignment_metadata"],
                    "nll": float(score["nll"].item()),
                    "token_accuracy": float(score["acc"].item()),
                    "pred_ids": pred_ids, "target_ids": target_ids,
                    "pred_tokens": self.model.opt_tokenizer.convert_ids_to_tokens(pred_ids),
                    "target_tokens": self.model.opt_tokenizer.convert_ids_to_tokens(target_ids)}
        finally:
            clear_hook_tensors(self.model)

    def tokenization_report(self, question, target):
        """Expose the standalone-label/context BPE boundary without changing it."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty question is required")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("A nonempty target is required")
        tokenizer = self.model.opt_tokenizer
        question_ids = tokenizer.encode(question)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        leading_space_ids = tokenizer.encode(" " + target, add_special_tokens=False)
        joined_ids = tokenizer.encode(question + " " + target)
        common = 0
        for question_id, joined_id in zip(question_ids, joined_ids):
            if question_id != joined_id:
                break
            common += 1
        suffix = joined_ids[-len(target_ids):] if target_ids else []
        return {"question_ids_with_default_special_tokens": question_ids,
                "question_ids_without_special_tokens": tokenizer.encode(question, add_special_tokens=False),
                "standalone_target_ids": target_ids,
                "leading_space_target_ids": leading_space_ids,
                "joined_input_ids": joined_ids,
                "release_suffix_ids": suffix,
                "joined_target_boundary_ids": joined_ids[common:],
                "question_prefix_retokenized": common < len(question_ids),
                "standalone_labels_match_input_suffix": target_ids == suffix,
                "standalone_target_tokens": tokenizer.convert_ids_to_tokens(target_ids),
                "leading_space_target_tokens": tokenizer.convert_ids_to_tokens(leading_space_ids),
                "release_suffix_tokens": tokenizer.convert_ids_to_tokens(suffix),
                "interpretation": "The release collator labels a standalone target while the input joins question, space and target. Different IDs expose a BPE boundary mismatch; this report does not change training or scoring."}

    def weight_hashes(self):
        return self.patches.weight_hashes()

    def save_patch(self, path):
        return self.patches.save(path)

    def load_patch(self, path):
        return self.patches.load(path)

    def rollback(self):
        return self.patches.rollback()

    def close(self):
        clear_hook_tensors(self.model)
        for handle in getattr(self.model, "handles", ()):
            handle.remove()
        self.patches.active = None
        self.patches.model = None
        self.model = self.editor = self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_backend(upstream_root, asset_root, device="cuda"):
    """Build the release architecture on CUDA. No download or training is run.

    Inputs must already have been fetched and integrity-checked by the asset
    preparation command. This constructor loads the original trusted research
    checkpoint with pickle, as upstream does; never supply an untrusted file.
    The full checkpoint/backend has not been validated by tiny adapter tests.
    """
    import yaml

    root, assets = Path(upstream_root).resolve(), Path(asset_root).resolve()
    requested_device = torch.device(device)
    if requested_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The full released MEND/BLIP-2 smoke needs a CUDA runtime; use the Colab notebook")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected pinned VLKEB commit {UPSTREAM_COMMIT}, found {commit}")
    modified = subprocess.run(["git", "-C", str(root), "diff", "--name-only", "HEAD", "--",
                               "easyeditor", "hparams"], check=True,
                              capture_output=True, text=True).stdout.strip()
    if modified:
        raise ValueError(f"The pinned upstream implementation has local changes: {modified}")
    required = {"opt": assets / "opt-2.7b", "bert": assets / "bert-base-uncased",
                "eva": assets / "eva_vit_g.pth", "blip": assets / "blip2_pretrained_opt2.7b.pth",
                "mend": assets / "mend.pt"}
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Prepare required assets first: " + ", ".join(missing))
    # Extract editor tensors before constructing the large base model; release
    # checkpoints can contain much larger optimizer states that we do not need.
    archive = torch.load(required["mend"], map_location="cpu", weights_only=False)
    if not isinstance(archive, dict) or not isinstance(archive.get("model"), dict):
        raise ValueError("The released MEND archive must contain a model state dictionary")
    editor_state = {key.replace("gtn.", "mend."): value
                    for key, value in archive["model"].items()}
    del archive
    gc.collect()

    blip_module, mend_module, processor_module = _upstream_modules(root)
    with (root / "hparams/MEND/blip2.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config.update(device=str(requested_device), model_parallel=False,
                  name=str(required["opt"]), tokenizer_name=str(required["opt"]),
                  qformer_name_or_path=str(required["bert"]),
                  state_dict_file=str(required["eva"]),
                  qformer_checkpoint=str(required["blip"]), archive=str(required["mend"]))
    config["edit_lr"] = float(config["edit_lr"])
    config = SimpleNamespace(**config)
    load_report = {}

    class AuditedBlip2OPT(blip_module.Blip2OPT):
        @classmethod
        def init_vision_encoder(cls, *args, **kwargs):
            encoder, normalization = super().init_vision_encoder(*args, **kwargs)
            return stage_vision_to_device(encoder, normalization, requested_device)

        def load_from_pretrained(self, url_or_filename):
            result = super().load_from_pretrained(url_or_filename)
            prefixes = ("Qformer.", "query_tokens", "ln_vision.", "opt_proj.")
            required_missing = [name for name in result.missing_keys if name.startswith(prefixes)]
            load_report["qformer_missing_keys"] = list(result.missing_keys)
            load_report["qformer_unexpected_keys"] = list(result.unexpected_keys)
            if required_missing:
                raise RuntimeError("BLIP checkpoint is missing required Q-Former/projection weights: "
                                   + ", ".join(required_missing))
            return result

    opt_report = {}
    with _low_memory_opt_loader(blip_module.OPTForCausalLM, opt_report):
        model = AuditedBlip2OPT(
            vit_model="eva_clip_g", img_size=364, use_grad_checkpoint=True,
            vit_precision="fp32", freeze_vit=True, opt_model=str(required["opt"]),
            state_dict_file=str(required["eva"]),
            qformer_name_or_path=str(required["bert"]),
            qformer_checkpoint=str(required["blip"]),
        )
    model.to(requested_device)
    editable = set(config.inner_params)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in editable)
    editor = mend_module.MEND(model, config, lambda: None)
    editor.load_state_dict(editor_state)
    del editor_state
    gc.collect()
    editor.edit_lrs.requires_grad_(False)
    for parameter in editor.mend.parameters():
        parameter.requires_grad_(False)
    editor.eval()
    processor = processor_module.BlipImageEvalProcessor(image_size=364, mean=None, std=None)
    identity = {"upstream_commit": UPSTREAM_COMMIT, "architecture": "BLIP2OPT/eva_clip_g/opt-2.7b",
                "asset_manifest_sha256": _file_hash(Path(__file__).resolve().parent.parent / "configs/vlkeb-assets.json"),
                "editor_checkpoint_sha256": _file_hash(required["mend"]),
                "config_sha256": _file_hash(root / "hparams/MEND/blip2.yaml")}
    compatibility = {"upstream_commit": UPSTREAM_COMMIT, "opt_load_report": opt_report,
                     "blip_load_report": load_report,
                     "adaptations": [
                         "Selective namespace imports bypass unrelated EasyEdit package initializers",
                         "Same custom OPT class/weights/dtypes; low_cpu_mem_usage changes loading allocation",
                         "Loaded FP32 EVA encoder and normalization move to CUDA before Q-Former/OPT construction after observed host-RAM exhaustion; final weights and computation are unchanged",
                         "Original MEND loss/hooks/learned transforms; skip functional mutation tail and apply factors once in FP32 on the selected device",
                         "Evaluation mode freezes dropout and MEND normalization; no auxiliary-model training",
                         "Default release labels match CaptionDataset rather than the differing executor; optional context-aligned suffix labels are explicitly marked experimental",
                         "Target-free bounded cached greedy continuation, rather than teacher-forced benchmark prediction",
                         "Single patch stores original/edited tensors for exact rollback and reload",
                     ],
                     "checkpoint_reproduction_verified": False}
    return VLKEBBackend(model, editor, processor, identity=identity, compatibility=compatibility)
