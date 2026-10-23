"""Fixed per-edit FT baseline; not MEND meta-training or full-FP32 reproduction."""

from __future__ import annotations

import torch


PARAMETER_NAMES = (
    "opt_model.model.decoder.layers.31.fc1.weight",
    "opt_model.model.decoder.layers.31.fc2.weight",
)
STEPS = 15
LEARNING_RATE = 2e-4


def finetune_last_layer(model, batch, loss_function):
    """Run exactly 15 AdamW updates using two FP32 master weight copies.

    VLKEB supplies the layer, step count and learning rate. Its FT implementation
    casts the whole model to FP32 and enables training mode. This resource
    adaptation retains released dtypes/evaluation mode and correct context labels.
    ``loss_function(logits, labels)`` must return the scalar target-token NLL.
    The caller must provide transactional patch capture around this mutation.
    """
    if batch.get("alignment_metadata", {}).get("target_alignment") != "context":
        raise ValueError("FT baseline requires explicit context-aligned target labels")
    parameters = dict(model.named_parameters())
    missing = set(PARAMETER_NAMES) - parameters.keys()
    if missing:
        raise ValueError("Missing exact FT parameters: " + ", ".join(sorted(missing)))
    selected = {name: parameters[name] for name in PARAMETER_NAMES}
    if any(not bool(torch.isfinite(value).all()) for value in selected.values()):
        raise ValueError("FT baseline requires finite initial parameters")
    flags = {name: parameter.requires_grad for name, parameter in parameters.items()}
    losses = []
    try:
        model.eval()
        for name, parameter in parameters.items():
            parameter.requires_grad_(name in selected)
        masters = {name: torch.nn.Parameter(parameter.detach().float().clone())
                   for name, parameter in selected.items()}
        # Match upstream FT.py's actual AdamW call: its default weight decay is
        # 0.01, even though the YAML declares a separate unused weight_decay=0.
        optimizer = torch.optim.AdamW(list(masters.values()), lr=LEARNING_RATE)
        for _ in range(STEPS):
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            logits = output if isinstance(output, torch.Tensor) else output.logits
            loss = loss_function(logits, batch["labels"])
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise RuntimeError("FT target NLL must be finite and scalar")
            losses.append(float(loss.detach().item()))
            loss.backward()
            for name, parameter in selected.items():
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("FT gradient is missing or non-finite: " + name)
                masters[name].grad = parameter.grad.detach().float().clone()
            optimizer.step()
            with torch.no_grad():
                for name, parameter in selected.items():
                    master = masters[name]
                    if not bool(torch.isfinite(master).all()):
                        raise RuntimeError("FT master weight is non-finite: " + name)
                    parameter.copy_(master.to(dtype=parameter.dtype))
                    if not bool(torch.isfinite(parameter).all()):
                        raise RuntimeError("FT model weight overflowed: " + name)
        with torch.no_grad():
            output = model(batch)
            logits = output if isinstance(output, torch.Tensor) else output.logits
            final_loss = loss_function(logits, batch["labels"])
            if final_loss.ndim != 0 or not bool(torch.isfinite(final_loss)):
                raise RuntimeError("FT final target NLL must be finite and scalar")
        return {"algorithm": "VLKEB-inspired bounded FT-LLM resource adaptation",
                "training_kind": "per_edit_target_fitting_not_meta_training",
                "completed_steps": STEPS, "loss_per_step": losses,
                "final_teacher_forced_nll": float(final_loss.item()),
                "is_generated_answer_accuracy": False,
                "parameter_names": list(PARAMETER_NAMES),
                "optimizer_recipe": {"name": "AdamW", "lr": LEARNING_RATE,
                                     "weight_decay": optimizer.defaults["weight_decay"],
                                     "betas": list(optimizer.defaults["betas"]),
                                     "eps": optimizer.defaults["eps"]},
                "master_dtype": "torch.float32",
                "model_parameter_dtypes": {name: str(value.dtype) for name, value in selected.items()},
                "model_mode": "eval", "target_alignment": "context",
                "answer_termination_supervision": batch["alignment_metadata"].get("answer_termination_supervision"),
                "desired_target": batch["alignment_metadata"].get("desired_target"),
                "training_target": batch["alignment_metadata"].get("training_target"),
                "full_fp32_upstream_reproduction": False,
                "adaptations": ["Only selected weights have FP32 master copies; released base dtypes retained",
                                "Evaluation mode retained instead of upstream training mode",
                                "Context-aligned labels replace released standalone-label convention",
                                "Backend supervises an appended newline to teach short-answer termination"],
                "kl_regularization": False}
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in parameters.items():
            parameter.requires_grad_(flags[name])
            for attribute in ("__x__", "__delta__"):
                if hasattr(parameter, attribute):
                    delattr(parameter, attribute)
