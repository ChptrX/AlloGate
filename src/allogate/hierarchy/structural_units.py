"""Protein-independent global structural units and their interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_SPAN_PATTERN = re.compile(
    r"^(?P<chain>[^:\s]+):(?P<start>[1-9][0-9]*)(?:-(?P<stop>[1-9][0-9]*))?$"
)


@dataclass(frozen=True, order=True, slots=True)
class ResidueSpan:
    chain_id: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not self.chain_id or ":" in self.chain_id or any(
            char.isspace() for char in self.chain_id
        ):
            raise ValueError("chain_id must be a non-empty token without ':'")
        if self.start < 1 or self.stop < self.start:
            raise ValueError("residue span must satisfy 1 <= start <= stop")

    @classmethod
    def parse(cls, value: str) -> "ResidueSpan":
        match = _SPAN_PATTERN.fullmatch(value.strip())
        if match is None:
            raise ValueError(f"invalid residue span: {value!r}")
        start = int(match.group("start"))
        stop = int(match.group("stop") or start)
        return cls(chain_id=match.group("chain"), start=start, stop=stop)

    def overlaps(self, other: "ResidueSpan") -> bool:
        return (
            self.chain_id == other.chain_id
            and self.start <= other.stop
            and other.start <= self.stop
        )

    def __str__(self) -> str:
        suffix = str(self.start) if self.start == self.stop else f"{self.start}-{self.stop}"
        return f"{self.chain_id}:{suffix}"


@dataclass(frozen=True, slots=True)
class GlobalStructuralUnitSpec:
    unit_id: str
    residues: tuple[ResidueSpan, ...]

    def __post_init__(self) -> None:
        if not self.unit_id or any(char.isspace() for char in self.unit_id):
            raise ValueError("unit_id must be a non-empty token")
        if not self.residues:
            raise ValueError("a global structural unit must contain at least one residue span")
        ordered = sorted(self.residues)
        for left, right in zip(ordered, ordered[1:]):
            if left.overlaps(right):
                raise ValueError(f"overlapping spans inside {self.unit_id}: {left} and {right}")

    @classmethod
    def from_strings(
        cls, unit_id: str, residues: Iterable[str]
    ) -> "GlobalStructuralUnitSpec":
        return cls(
            unit_id=unit_id,
            residues=tuple(ResidueSpan.parse(item) for item in residues),
        )


@dataclass(frozen=True, order=True, slots=True)
class InterfaceSpec:
    """Stable identity for an unordered pair of global structural units."""

    interface_id: str
    global_unit_ids: tuple[str, str]

    def __post_init__(self) -> None:
        if not self.interface_id or any(char.isspace() for char in self.interface_id):
            raise ValueError("interface_id must be a non-empty token")
        if (
            len(self.global_unit_ids) != 2
            or self.global_unit_ids[0] == self.global_unit_ids[1]
        ):
            raise ValueError("an interface must connect two distinct global structural units")
        if any(
            not item or any(char.isspace() for char in item)
            for item in self.global_unit_ids
        ):
            raise ValueError("global structural unit identifiers must be non-empty tokens")
        object.__setattr__(self, "global_unit_ids", tuple(sorted(self.global_unit_ids)))


def validate_disjoint_global_units(
    units: Iterable[GlobalStructuralUnitSpec],
) -> tuple[GlobalStructuralUnitSpec, ...]:
    materialized = tuple(units)
    identifiers = [unit.unit_id for unit in materialized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("global structural unit identifiers must be unique")
    for index, left_unit in enumerate(materialized):
        for right_unit in materialized[index + 1 :]:
            for left_span in left_unit.residues:
                for right_span in right_unit.residues:
                    if left_span.overlaps(right_span):
                        raise ValueError(
                            f"global structural units {left_unit.unit_id} and "
                            f"{right_unit.unit_id} overlap at {left_span} / {right_span}"
                        )
    return materialized


def validate_interfaces(
    interfaces: Iterable[InterfaceSpec],
    global_units: Iterable[GlobalStructuralUnitSpec],
) -> tuple[InterfaceSpec, ...]:
    materialized = tuple(interfaces)
    known = {unit.unit_id for unit in global_units}
    if len({item.interface_id for item in materialized}) != len(materialized):
        raise ValueError("interface identifiers must be unique")
    pairs = [item.global_unit_ids for item in materialized]
    if len(set(pairs)) != len(pairs):
        raise ValueError("global structural unit pairs must define unique interfaces")
    for item in materialized:
        missing = set(item.global_unit_ids).difference(known)
        if missing:
            raise ValueError(
                f"interface {item.interface_id} references unknown units: {sorted(missing)}"
            )
    return materialized


# Source compatibility for the evaluation-first 0.1 alpha.
StructuralUnitSpec = GlobalStructuralUnitSpec
validate_disjoint_units = validate_disjoint_global_units
