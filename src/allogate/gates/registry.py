"""Immutable Gate Registry and deterministic global tensor layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from allogate.config.hashing import stable_digest

from .identities import EntityLevel, GateFamily, GateSpec


_FAMILY_ORDER = {family: index for index, family in enumerate(GateFamily)}
_LEVEL_ORDER = {level: index for index, level in enumerate(EntityLevel)}


def _gate_sort_key(gate: GateSpec) -> tuple[int, int, str]:
    return _FAMILY_ORDER[gate.family], _LEVEL_ORDER[gate.level], gate.uid


@dataclass(frozen=True, slots=True)
class LayoutEntry:
    uid: str
    global_index: int
    group: str
    group_index: int


@dataclass(frozen=True, slots=True)
class GateTensorLayout:
    entries: tuple[LayoutEntry, ...]

    def __post_init__(self) -> None:
        expected = list(range(len(self.entries)))
        actual = [entry.global_index for entry in self.entries]
        if actual != expected:
            raise ValueError("layout entries must have contiguous global indices")
        if len({entry.uid for entry in self.entries}) != len(self.entries):
            raise ValueError("layout entries contain duplicate Gate UIDs")
        group_counts: dict[str, int] = {}
        for entry in self.entries:
            expected_group_index = group_counts.get(entry.group, 0)
            if entry.group_index != expected_group_index:
                raise ValueError(f"non-contiguous index in layout group {entry.group}")
            group_counts[entry.group] = expected_group_index + 1

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def digest(self) -> str:
        return stable_digest(
            [
                {
                    "uid": entry.uid,
                    "global_index": entry.global_index,
                    "group": entry.group,
                    "group_index": entry.group_index,
                }
                for entry in self.entries
            ]
        )

    def index(self, uid: str) -> int:
        matches = [entry.global_index for entry in self.entries if entry.uid == uid]
        if len(matches) != 1:
            raise KeyError(uid)
        return matches[0]

    def entries_for_group(self, group: str) -> tuple[LayoutEntry, ...]:
        return tuple(entry for entry in self.entries if entry.group == group)

    def vector(self, overrides: Mapping[str, float] | None = None) -> tuple[float, ...]:
        values = [1.0] * len(self.entries)
        for uid, value in (overrides or {}).items():
            numeric = float(value)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"Gate value outside [0, 1] for {uid}")
            values[self.index(uid)] = numeric
        return tuple(values)


@dataclass(frozen=True, slots=True)
class GateRegistry:
    namespace: str
    gates: tuple[GateSpec, ...]
    schema_version: str = "allogate.gate_registry.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "allogate.gate_registry.v1":
            raise ValueError(f"unsupported Gate Registry schema: {self.schema_version}")
        ordered = tuple(sorted(self.gates, key=_gate_sort_key))
        if not ordered:
            raise ValueError("a Gate Registry must contain at least one Gate")
        if any(gate.namespace != self.namespace for gate in ordered):
            raise ValueError("every Gate must use the Registry namespace")
        if len({gate.uid for gate in ordered}) != len(ordered):
            raise ValueError("duplicate Gate UID")
        canonical_uids = {gate.uid for gate in ordered}
        aliases: set[str] = set()
        for gate in ordered:
            for alias in gate.aliases:
                if alias in aliases or alias in canonical_uids:
                    raise ValueError(f"ambiguous Gate alias: {alias}")
                aliases.add(alias)
        object.__setattr__(self, "gates", ordered)

    @classmethod
    def from_specs(cls, namespace: str, gates: Iterable[GateSpec]) -> "GateRegistry":
        return cls(namespace=namespace, gates=tuple(gates))

    @property
    def digest(self) -> str:
        return stable_digest(
            {
                "schema_version": self.schema_version,
                "namespace": self.namespace,
                "gates": [
                    {
                        "uid": gate.uid,
                        "family": gate.family.value,
                        "targets": [target.uid for target in gate.targets],
                        "provenance_nodes": list(gate.provenance_nodes),
                        "relation": gate.relation,
                        "default": gate.default,
                        "aliases": list(gate.aliases),
                    }
                    for gate in self.gates
                ],
            }
        )

    @property
    def layout(self) -> GateTensorLayout:
        group_counts: dict[str, int] = {}
        entries: list[LayoutEntry] = []
        for global_index, gate in enumerate(self.gates):
            group = f"{gate.family.value}.{gate.level.value}"
            group_index = group_counts.get(group, 0)
            entries.append(LayoutEntry(gate.uid, global_index, group, group_index))
            group_counts[group] = group_index + 1
        return GateTensorLayout(tuple(entries))

    def resolve(self, uid_or_alias: str) -> GateSpec:
        matches = [
            gate for gate in self.gates if gate.uid == uid_or_alias or uid_or_alias in gate.aliases
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown or ambiguous Gate identifier: {uid_or_alias!r}")
        return matches[0]

    def select(
        self,
        *,
        family: GateFamily | None = None,
        level: EntityLevel | None = None,
    ) -> tuple[GateSpec, ...]:
        return tuple(
            gate
            for gate in self.gates
            if (family is None or gate.family is family) and (level is None or gate.level is level)
        )
