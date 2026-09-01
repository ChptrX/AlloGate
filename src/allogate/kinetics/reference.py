"""Portable, immutable projection from learned CVs to a fixed kinetic target."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any

import numpy as np

from allogate.config.hashing import stable_digest


_UNIT = re.compile(r"^[A-Za-z][A-Za-z0-9._/-]*$")


def _frozen_float64(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy(order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimension(s)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def array_digest(array: np.ndarray) -> str:
    """Hash a numeric array with its canonical dtype and shape."""

    canonical = np.asarray(array, dtype="<f8", order="C")
    header = stable_digest(
        {"schema": "allogate.ndarray.v1", "dtype": "float64-le", "shape": list(canonical.shape)}
    ).encode("ascii")
    return sha256(header + canonical.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class KineticReference:
    """A fixed radial-basis projection from CV coordinates to values in [0, 1].

    The object stores scientific arrays only. Training paths, host information,
    checkpoints, and runtime device choices cannot enter its semantic identity.
    """

    centers: np.ndarray
    target_values: np.ndarray
    cv_mean: np.ndarray
    cv_whitener: np.ndarray
    bandwidth: float
    lag_time: float
    lag_unit: str = "frame"
    schema_version: str = "allogate.kinetic_reference.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "allogate.kinetic_reference.v1":
            raise ValueError(f"unsupported kinetic-reference schema: {self.schema_version}")
        centers = _frozen_float64(self.centers, ndim=2, name="centers")
        targets = _frozen_float64(self.target_values, ndim=1, name="target_values")
        mean = _frozen_float64(self.cv_mean, ndim=1, name="cv_mean")
        whitener = _frozen_float64(self.cv_whitener, ndim=2, name="cv_whitener")
        if not centers.shape[0] or not centers.shape[1]:
            raise ValueError("centers must be non-empty")
        if targets.shape != (centers.shape[0],):
            raise ValueError("target_values must contain one value per center")
        if np.any(targets < 0.0) or np.any(targets > 1.0):
            raise ValueError("target_values must lie in [0, 1]")
        dimensions = centers.shape[1]
        if mean.shape != (dimensions,) or whitener.shape != (dimensions, dimensions):
            raise ValueError("CV normalization arrays do not match the center dimension")
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0.0:
            raise ValueError("bandwidth must be finite and positive")
        if not np.isfinite(self.lag_time) or self.lag_time <= 0.0:
            raise ValueError("lag_time must be finite and positive")
        if _UNIT.fullmatch(self.lag_unit) is None:
            raise ValueError("lag_unit must be a portable unit token")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "target_values", targets)
        object.__setattr__(self, "cv_mean", mean)
        object.__setattr__(self, "cv_whitener", whitener)
        object.__setattr__(self, "bandwidth", float(self.bandwidth))
        object.__setattr__(self, "lag_time", float(self.lag_time))

    @property
    def cv_dimensions(self) -> int:
        return self.centers.shape[1]

    @property
    def digest(self) -> str:
        return stable_digest(
            {
                "schema_version": self.schema_version,
                "centers": array_digest(self.centers),
                "target_values": array_digest(self.target_values),
                "cv_mean": array_digest(self.cv_mean),
                "cv_whitener": array_digest(self.cv_whitener),
                "bandwidth": self.bandwidth,
                "lag_time": self.lag_time,
                "lag_unit": self.lag_unit,
            }
        )

    def evaluate_numpy(self, cv: Any) -> np.ndarray:
        values = _frozen_float64(cv, ndim=2, name="cv")
        if values.shape[1] != self.cv_dimensions:
            raise ValueError("CV input dimension does not match the kinetic reference")
        whitened = (values - self.cv_mean) @ self.cv_whitener.T
        displacement = whitened[:, None, :] - self.centers[None, :, :]
        logits = -0.5 * np.sum(displacement * displacement, axis=-1) / (self.bandwidth**2)
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=1, keepdims=True)
        return weights @ self.target_values

    def evaluate_torch(self, cv: Any) -> Any:
        """Evaluate the same fixed target with differentiable float64 arithmetic."""

        try:
            import torch
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required for differentiable kinetic projection") from error
        if not isinstance(cv, torch.Tensor) or cv.ndim != 2:
            raise ValueError("cv must be a two-dimensional PyTorch tensor")
        if cv.shape[1] != self.cv_dimensions:
            raise ValueError("CV input dimension does not match the kinetic reference")
        values = cv.to(dtype=torch.float64)
        mean = torch.tensor(self.cv_mean, dtype=torch.float64, device=cv.device)
        whitener = torch.tensor(self.cv_whitener, dtype=torch.float64, device=cv.device)
        centers = torch.tensor(self.centers, dtype=torch.float64, device=cv.device)
        targets = torch.tensor(self.target_values, dtype=torch.float64, device=cv.device)
        whitened = (values - mean) @ whitener.T
        displacement = whitened[:, None, :] - centers[None, :, :]
        logits = -0.5 * torch.sum(displacement * displacement, dim=-1) / (self.bandwidth**2)
        return torch.softmax(logits, dim=1) @ targets


def save_kinetic_reference(path: str | Path, reference: KineticReference) -> None:
    """Write a portable array bundle containing no source or host paths."""

    np.savez_compressed(
        Path(path),
        schema_version=np.asarray(reference.schema_version),
        centers=reference.centers,
        target_values=reference.target_values,
        cv_mean=reference.cv_mean,
        cv_whitener=reference.cv_whitener,
        bandwidth=np.asarray(reference.bandwidth, dtype=np.float64),
        lag_time=np.asarray(reference.lag_time, dtype=np.float64),
        lag_unit=np.asarray(reference.lag_unit),
        reference_digest=np.asarray(reference.digest),
    )


def load_kinetic_reference(path: str | Path) -> KineticReference:
    with np.load(Path(path), allow_pickle=False) as source:
        reference = KineticReference(
            centers=source["centers"],
            target_values=source["target_values"],
            cv_mean=source["cv_mean"],
            cv_whitener=source["cv_whitener"],
            bandwidth=float(source["bandwidth"].item()),
            lag_time=float(source["lag_time"].item()),
            lag_unit=str(source["lag_unit"].item()),
            schema_version=str(source["schema_version"].item()),
        )
        expected = str(source["reference_digest"].item())
    if reference.digest != expected:
        raise ValueError("kinetic-reference semantic digest mismatch")
    return reference
