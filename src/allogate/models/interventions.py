"""Differentiable State Gate replacement with calibrated scalar baselines."""

from __future__ import annotations

from torch import Tensor

from .gvp import ScalarVector


def apply_state_replacement(
    features: ScalarVector,
    gate: Tensor | None,
    baseline_scalar: Tensor | None,
) -> ScalarVector:
    """Interpolate stored state toward a frozen scalar baseline.

    At gate zero scalar channels equal the calibrated baseline and vector
    channels are zero. This baseline is a feature baseline, not the invariant
    no-intervention kinetic target used by later evaluation stages.
    """

    if gate is None:
        return features
    scalar, vector = features
    if gate.shape != scalar.shape[:-1]:
        raise ValueError("State Gate shape must match the feature item axis")
    if baseline_scalar is None:
        raise ValueError("a State intervention requires a frozen scalar baseline")
    if baseline_scalar.requires_grad:
        raise ValueError("a State baseline must be frozen and cannot require gradients")
    if baseline_scalar.shape != scalar.shape:
        raise ValueError("State baseline must match scalar feature identity and channels")
    if baseline_scalar.device != scalar.device or vector.device != scalar.device:
        raise ValueError("State baseline and features must share a device")
    gate_scalar = gate.to(device=scalar.device, dtype=scalar.dtype).unsqueeze(-1)
    baseline = baseline_scalar.to(dtype=scalar.dtype)
    replaced_scalar = baseline + gate_scalar * (scalar - baseline)
    replaced_vector = vector * gate_scalar.unsqueeze(-1)
    return replaced_scalar, replaced_vector
