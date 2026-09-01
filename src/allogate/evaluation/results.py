"""Read-only numerical records emitted by the Gate evaluation service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from allogate.config.hashing import stable_digest
from allogate.kinetics.reference import array_digest


def _readonly(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy(order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimension(s)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class NoInterventionReference:
    identity_digest: str
    gate_uids: tuple[str, ...]
    q0: np.ndarray

    def __post_init__(self) -> None:
        if len(set(self.gate_uids)) != len(self.gate_uids):
            raise ValueError("no-intervention Gate UIDs must be unique")
        object.__setattr__(self, "q0", _readonly(self.q0, ndim=1, name="q0"))

    @property
    def digest(self) -> str:
        return stable_digest(
            {
                "schema": "allogate.no_intervention.v1",
                "identity_digest": self.identity_digest,
                "gate_uids": list(self.gate_uids),
                "q0": array_digest(self.q0),
            }
        )


@dataclass(frozen=True, slots=True)
class HessianBlockResult:
    name: str
    gate_uids: tuple[str, ...]
    values: np.ndarray
    maximum_asymmetry: float

    def __post_init__(self) -> None:
        values = _readonly(self.values, ndim=3, name="Hessian values")
        expected = (len(values), len(self.gate_uids), len(self.gate_uids))
        if values.shape != expected:
            raise ValueError("Hessian values do not match their Gate block")
        if not np.isfinite(self.maximum_asymmetry) or self.maximum_asymmetry < 0.0:
            raise ValueError("maximum_asymmetry must be finite and non-negative")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class GateEvaluationResult:
    identity_digest: str
    reference_digest: str
    gate_uids: tuple[str, ...]
    gate_values: np.ndarray
    q: np.ndarray
    q0: np.ndarray
    delta_q: np.ndarray
    jacobian_uids: tuple[str, ...]
    jacobian: np.ndarray
    hessian_blocks: tuple[HessianBlockResult, ...]
    all_one: bool
    intervention_l1: float

    def __post_init__(self) -> None:
        gate_values = _readonly(self.gate_values, ndim=1, name="gate_values")
        q = _readonly(self.q, ndim=1, name="q")
        q0 = _readonly(self.q0, ndim=1, name="q0")
        delta = _readonly(self.delta_q, ndim=1, name="delta_q")
        jacobian = _readonly(self.jacobian, ndim=2, name="jacobian")
        if gate_values.shape != (len(self.gate_uids),):
            raise ValueError("gate_values do not match the canonical Gate layout")
        if q.shape != q0.shape or q.shape != delta.shape:
            raise ValueError("q, q0, and delta_q must share one output shape")
        if jacobian.shape != (len(q), len(self.jacobian_uids)):
            raise ValueError("Jacobian does not match outputs and Gate UIDs")
        if not np.allclose(delta, q - q0, rtol=0.0, atol=1.0e-12):
            raise ValueError("delta_q is inconsistent with q and q0")
        object.__setattr__(self, "gate_values", gate_values)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "q0", q0)
        object.__setattr__(self, "delta_q", delta)
        object.__setattr__(self, "jacobian", jacobian)
