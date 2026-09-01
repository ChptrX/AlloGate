"""Bind canonical global Gate values to runtime model tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from allogate.models.graph import GateControls

from .identities import EntityLevel, GateFamily
from .registry import GateRegistry


_CONTROL_CONTRACT: dict[str, tuple[GateFamily, EntityLevel, EntityLevel | None]] = {
    "residue_state": (GateFamily.STATE, EntityLevel.RESIDUE, None),
    "local_state": (GateFamily.STATE, EntityLevel.LOCAL_ELEMENT, None),
    "unit_state": (GateFamily.STATE, EntityLevel.STRUCTURAL_UNIT, None),
    "residue_to_local": (GateFamily.RELAY, EntityLevel.RESIDUE, None),
    "local_to_unit": (GateFamily.RELAY, EntityLevel.LOCAL_ELEMENT, None),
    "residue_edges": (GateFamily.CONTACT, EntityLevel.CONTACT, EntityLevel.RESIDUE),
    "local_edges": (GateFamily.CONTACT, EntityLevel.CONTACT, EntityLevel.LOCAL_ELEMENT),
    "unit_contact_edges": (
        GateFamily.CONTACT,
        EntityLevel.CONTACT,
        EntityLevel.STRUCTURAL_UNIT,
    ),
}


@dataclass(frozen=True, slots=True)
class ControlBinding:
    control_name: str
    gate_uids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.control_name not in _CONTROL_CONTRACT:
            raise ValueError(f"unsupported formal model control: {self.control_name}")
        if not self.gate_uids:
            raise ValueError("a control binding must contain at least one Gate UID")


@dataclass(frozen=True, slots=True)
class ModelGateBinding:
    """Runtime alignment; repeated UIDs support multiple graphs in one batch."""

    registry_digest: str
    layout_digest: str
    bindings: tuple[ControlBinding, ...]

    @classmethod
    def build(
        cls,
        registry: GateRegistry,
        controls: Mapping[str, tuple[str, ...]],
    ) -> "ModelGateBinding":
        bindings = tuple(
            ControlBinding(name, tuple(uids)) for name, uids in sorted(controls.items())
        )
        if len({binding.control_name for binding in bindings}) != len(bindings):
            raise ValueError("duplicate model control binding")
        for binding in bindings:
            expected_family, expected_level, endpoint_level = _CONTROL_CONTRACT[binding.control_name]
            for uid in binding.gate_uids:
                gate = registry.resolve(uid)
                if gate.family is not expected_family or gate.level is not expected_level:
                    raise ValueError(
                        f"Gate {uid} is incompatible with control {binding.control_name}"
                    )
                if endpoint_level is not None and any(
                    target.level is not endpoint_level for target in gate.targets
                ):
                    raise ValueError(
                        f"Gate {uid} endpoints are incompatible with control {binding.control_name}"
                    )
        return cls(registry.digest, registry.layout.digest, bindings)

    def bind(self, registry: GateRegistry, values: Tensor) -> GateControls:
        if registry.digest != self.registry_digest or registry.layout.digest != self.layout_digest:
            raise ValueError("Gate Registry or tensor layout does not match this runtime binding")
        if values.ndim != 1 or values.numel() != len(registry.layout):
            raise ValueError("global Gate tensor has the wrong shape")
        if not bool(torch.isfinite(values).all()) or bool(torch.any((values < 0.0) | (values > 1.0))):
            raise ValueError("global Gate values must be finite and within [0, 1]")
        resolved: dict[str, Tensor] = {}
        for binding in self.bindings:
            indices = torch.tensor(
                [registry.layout.index(uid) for uid in binding.gate_uids],
                dtype=torch.long,
                device=values.device,
            )
            resolved[binding.control_name] = values.index_select(0, indices)
        return GateControls(**resolved)

    def all_one(self, registry: GateRegistry, reference: Tensor) -> GateControls:
        values = reference.new_ones(len(registry.layout))
        return self.bind(registry, values)
