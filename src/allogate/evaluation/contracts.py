"""Backend-independent request and identity contracts for Gate evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from allogate.config.hashing import stable_digest


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def require_digest(value: str, name: str) -> None:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class EvaluationIdentity:
    registry_digest: str
    layout_digest: str
    model_digest: str
    target_digest: str
    numerical_policy: str = "torch-float64-exact-autograd-v1"
    schema_version: str = "allogate.evaluation_identity.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "allogate.evaluation_identity.v1":
            raise ValueError(f"unsupported evaluation identity: {self.schema_version}")
        for name in ("registry_digest", "layout_digest", "model_digest", "target_digest"):
            require_digest(getattr(self, name), name)
        if _NAME.fullmatch(self.numerical_policy) is None:
            raise ValueError("numerical_policy must be a portable token")

    @property
    def digest(self) -> str:
        return stable_digest(
            {
                "schema_version": self.schema_version,
                "registry_digest": self.registry_digest,
                "layout_digest": self.layout_digest,
                "model_digest": self.model_digest,
                "target_digest": self.target_digest,
                "numerical_policy": self.numerical_policy,
            }
        )


@dataclass(frozen=True, slots=True)
class HessianBlockSpec:
    name: str
    gate_uids: tuple[str, ...]

    def __post_init__(self) -> None:
        if _NAME.fullmatch(self.name) is None:
            raise ValueError("Hessian block name must be a portable token")
        if not self.gate_uids or any(not uid for uid in self.gate_uids):
            raise ValueError("a Hessian block must contain at least one Gate UID")
        if len(set(self.gate_uids)) != len(self.gate_uids):
            raise ValueError("a Hessian block cannot repeat a Gate UID")


@dataclass(frozen=True, slots=True)
class GateEvaluationRequest:
    identity_digest: str
    overrides: tuple[tuple[str, float], ...] = ()
    jacobian_uids: tuple[str, ...] = ()
    hessian_blocks: tuple[HessianBlockSpec, ...] = ()

    def __post_init__(self) -> None:
        require_digest(self.identity_digest, "identity_digest")
        override_uids = [uid for uid, _ in self.overrides]
        if any(not uid for uid in override_uids) or len(set(override_uids)) != len(override_uids):
            raise ValueError("Gate overrides must use unique, non-empty identifiers")
        for uid, value in self.overrides:
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"Gate override outside [0, 1] for {uid}")
        if any(not uid for uid in self.jacobian_uids):
            raise ValueError("Jacobian Gate identifiers cannot be empty")
        if len(set(self.jacobian_uids)) != len(self.jacobian_uids):
            raise ValueError("Jacobian Gate identifiers must be unique")
        names = [block.name for block in self.hessian_blocks]
        if len(set(names)) != len(names):
            raise ValueError("Hessian block names must be unique")
