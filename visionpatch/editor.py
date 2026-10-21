"""A minimum-disturbance edit of a visual projection matrix.

This experimental baseline changes one linear projection, not a full vision
encoder. It is inspired by associative-memory editing, but is not a faithful
implementation of EditingClassifiers, ROME, or any multimodal LLM editor.
Support and retention features must come from disjoint, declared data pools.
The caller is responsible for selecting those pools and evaluating generality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


METHOD = "minimum_disturbance_projection_v1"
FORMAT_VERSION = 1
_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def _matrix(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided or value.ndim != 2 or 0 in value.shape:
        raise ValueError(f"{name} must be a non-empty dense two-dimensional matrix")
    if value.dtype not in _DTYPES:
        raise ValueError(f"{name} must use a supported real floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def projection_sha256(weight: Tensor) -> str:
    """Hash shape, dtype and exact parameter bytes, independent of device."""
    snapshot = _matrix("weight", weight).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(f"{snapshot.dtype}:{tuple(snapshot.shape)}:".encode("ascii"))
    # View as bytes first so that bfloat16 does not need NumPy dtype support.
    digest.update(snapshot.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _json_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict) or not all(isinstance(k, str) for k in metadata):
        raise ValueError("metadata must be a JSON object with string keys")
    try:
        result = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain finite JSON-serializable values") from exc
    return result


@dataclass(frozen=True)
class ProjectionPatch:
    """Frozen patch record with checked tensor snapshots and a reversible state.

    Tensor storage remains mutable in PyTorch. Every operation therefore checks
    the stored hashes instead of relying on dataclass immutability for integrity.
    ``before_weight`` is intentionally retained: subtracting a floating-point
    update would not guarantee bit-exact restoration.
    """

    before_weight: Tensor
    left_factor: Tensor
    right_factor: Tensor
    before_sha256: str
    after_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rank_bound(self) -> int:
        """An upper bound, not a numerical-rank measurement."""
        return min(self.left_factor.shape[1], *self.before_weight.shape)

    def edited_weight(self) -> Tensor:
        """Return an independent CPU tensor after checking artifact integrity."""
        return _checked_edited_weight(self)


def _checked_edited_weight(patch: ProjectionPatch) -> Tensor:
    if not isinstance(patch, ProjectionPatch):
        raise TypeError("patch must be a ProjectionPatch")
    before = _matrix("before_weight", patch.before_weight)
    left = _matrix("left_factor", patch.left_factor)
    right = _matrix("right_factor", patch.right_factor)
    if left.shape[0] != before.shape[0] or right.shape[1] != before.shape[1]:
        raise ValueError("patch factor dimensions do not match the projection")
    if left.shape[1] != right.shape[0]:
        raise ValueError("patch factor inner dimensions do not match")
    if projection_sha256(before) != patch.before_sha256:
        raise ValueError("patch before-weight snapshot failed its integrity check")
    candidate = (before.detach().cpu().double()
                 + left.detach().cpu().double() @ right.detach().cpu().double())
    candidate = candidate.to(dtype=before.dtype).contiguous()
    _matrix("edited_weight", candidate)
    if projection_sha256(candidate) != patch.after_sha256:
        raise ValueError("patch edited weight failed its integrity check")
    return candidate


def fit_projection_patch(
    weight: Tensor,
    support_keys: Tensor,
    desired_outputs: Tensor,
    retention_features: Tensor,
    *,
    covariance_ridge: float = 1e-3,
    fit_ridge: float = 1e-6,
    metadata: dict[str, Any] | None = None,
) -> ProjectionPatch:
    """Fit a low-rank correction using CPU float64 linear solves.

    Shapes: ``weight[out,in]``, ``support_keys[N,in]``,
    ``desired_outputs[N,out]``, ``retention_features[R,in]``. For row-major
    support K and retention R, C = R.T R / len(R) + ridge * scale * I.
    The update is E.T (K C^-1 K.T + fit_ridge I)^-1 K C^-1, where
    E = desired_outputs - K weight.T. No explicit matrix inverse is formed.

    The covariance penalizes changes along frequently observed retention
    directions. It does not guarantee unchanged predictions on other images.
    Only this supplied projection is edited; it must be bias-free or the caller
    must subtract a fixed bias from desired_outputs before invoking this fit.
    """
    _matrix("weight", weight)
    _matrix("support_keys", support_keys)
    _matrix("desired_outputs", desired_outputs)
    _matrix("retention_features", retention_features)
    if support_keys.shape[1] != weight.shape[1]:
        raise ValueError("support_keys input dimension must match weight")
    if desired_outputs.shape != (support_keys.shape[0], weight.shape[0]):
        raise ValueError("desired_outputs must have shape [support_count, output_dimension]")
    if retention_features.shape[1] != weight.shape[1]:
        raise ValueError("retention_features input dimension must match weight")
    for name, value in (("covariance_ridge", covariance_ridge), ("fit_ridge", fit_ridge)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite positive number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")

    before = weight.detach().cpu().contiguous().clone()
    w = before.double()
    keys = support_keys.detach().cpu().double()
    targets = desired_outputs.detach().cpu().double()
    retained = retention_features.detach().cpu().double()
    covariance = retained.T @ retained / len(retained)
    scale = float(covariance.diagonal().mean())
    if scale == 0:
        scale = 1.0
    covariance += covariance_ridge * scale * torch.eye(w.shape[1], dtype=torch.float64)
    _matrix("regularized covariance", covariance)
    whitened_keys = torch.linalg.solve(covariance, keys.T)
    gram = keys @ whitened_keys
    # Remove small arithmetic asymmetries before the symmetric solve.
    gram = (gram + gram.T) * 0.5
    gram += fit_ridge * torch.eye(len(keys), dtype=torch.float64)
    residual = targets.T - w @ keys.T
    left = torch.linalg.solve(gram, residual.T).T.contiguous()
    right = whitened_keys.T.contiguous()
    _matrix("left_factor", left)
    _matrix("right_factor", right)
    edited = (w + left @ right).to(dtype=before.dtype).contiguous()
    _matrix("edited_weight", edited)
    before_hash, after_hash = projection_sha256(before), projection_sha256(edited)
    if before_hash == after_hash:
        raise ValueError("edit produced no representable weight change")
    fit_metadata = _json_metadata(metadata)
    if "fit" in fit_metadata:
        raise ValueError("metadata key 'fit' is reserved for measured fit information")
    fit_metadata["fit"] = {
        "method": METHOD,
        "support_count": len(keys),
        "retention_count": len(retained),
        "covariance_ridge": float(covariance_ridge),
        "fit_ridge": float(fit_ridge),
        "covariance_scale": scale,
        "support_mse_before": float(residual.square().mean()),
        "support_mse_after": float((targets - keys @ edited.double().T).square().mean()),
    }
    _json_metadata(fit_metadata)
    return ProjectionPatch(before, left, right, before_hash, after_hash, fit_metadata)


def apply_projection_patch_(weight: Tensor, patch: ProjectionPatch) -> None:
    """Apply once to its exact source state; reject a stale or edited model."""
    edited = _checked_edited_weight(patch)
    if projection_sha256(weight) != patch.before_sha256:
        raise ValueError("projection does not match the patch source state; refusing apply")
    with torch.no_grad():
        weight.copy_(edited.to(device=weight.device))
    if projection_sha256(weight) != patch.after_sha256:
        raise RuntimeError("applied projection failed its exact-state verification")


def rollback_projection_patch_(weight: Tensor, patch: ProjectionPatch) -> None:
    """Restore the saved original bytes, only from the exact edited state."""
    _checked_edited_weight(patch)
    if projection_sha256(weight) != patch.after_sha256:
        raise ValueError("projection does not match the patch edited state; refusing rollback")
    with torch.no_grad():
        weight.copy_(patch.before_weight.to(device=weight.device))
    if projection_sha256(weight) != patch.before_sha256:
        raise RuntimeError("restored projection failed its exact-state verification")


def save_projection_patch(patch: ProjectionPatch, path: str | Path) -> None:
    """Save tensors and JSON metadata without serializing Python classes."""
    _checked_edited_weight(patch)
    payload = {
        "format_version": FORMAT_VERSION,
        "method": METHOD,
        "before_weight": patch.before_weight.detach().cpu().contiguous().clone(),
        "left_factor": patch.left_factor.detach().cpu().contiguous().clone(),
        "right_factor": patch.right_factor.detach().cpu().contiguous().clone(),
        "before_sha256": patch.before_sha256,
        "after_sha256": patch.after_sha256,
        "metadata_json": json.dumps(_json_metadata(patch.metadata), sort_keys=True, allow_nan=False),
    }
    torch.save(payload, Path(path))


def load_projection_patch(path: str | Path) -> ProjectionPatch:
    """Load through PyTorch's restricted weights-only loader and validate hashes."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected = {"format_version", "method", "before_weight", "left_factor", "right_factor",
                "before_sha256", "after_sha256", "metadata_json"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("invalid projection patch artifact fields")
    if (type(payload["format_version"]) is not int
            or payload["format_version"] != FORMAT_VERSION or payload["method"] != METHOD):
        raise ValueError("unsupported projection patch artifact format/method")
    try:
        decoded_metadata = json.loads(payload["metadata_json"])
        if not isinstance(decoded_metadata, dict):
            raise ValueError("projection patch metadata must be a JSON object")
        metadata = _json_metadata(decoded_metadata)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid projection patch metadata JSON") from exc
    patch = ProjectionPatch(
        payload["before_weight"], payload["left_factor"], payload["right_factor"],
        payload["before_sha256"], payload["after_sha256"], metadata,
    )
    _checked_edited_weight(patch)
    return patch
