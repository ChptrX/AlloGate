"""Deterministic single-Gate dose schedules for finite interventions."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .contracts import GateEvaluationRequest


@dataclass(frozen=True, slots=True)
class DoseSchedule:
    gate_uids: tuple[str, ...]
    doses: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.gate_uids or len(set(self.gate_uids)) != len(self.gate_uids):
            raise ValueError("dose schedule Gate UIDs must be non-empty and unique")
        normalized = tuple(sorted({float(dose) for dose in self.doses}, reverse=True))
        if not normalized:
            raise ValueError("dose schedule must contain at least one dose")
        if any(not math.isfinite(dose) or not 0.0 <= dose <= 1.0 for dose in normalized):
            raise ValueError("dose values must be finite and lie in [0, 1]")
        object.__setattr__(self, "doses", normalized)

    def requests(self, identity_digest: str) -> tuple[GateEvaluationRequest, ...]:
        return tuple(
            GateEvaluationRequest(identity_digest=identity_digest, overrides=((uid, dose),))
            for uid in self.gate_uids
            for dose in self.doses
        )
