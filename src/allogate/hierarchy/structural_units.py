"""Study-independent structural units addressed by chain and residue number."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_SPAN_PATTERN = re.compile(r"^(?P<chain>[^:\s]+):(?P<start>[1-9][0-9]*)(?:-(?P<stop>[1-9][0-9]*))?$")


@dataclass(frozen=True, order=True, slots=True)
class ResidueSpan:
    chain_id: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not self.chain_id or ":" in self.chain_id or any(char.isspace() for char in self.chain_id):
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
        return self.chain_id == other.chain_id and self.start <= other.stop and other.start <= self.stop

    def __str__(self) -> str:
        suffix = str(self.start) if self.start == self.stop else f"{self.start}-{self.stop}"
        return f"{self.chain_id}:{suffix}"


@dataclass(frozen=True, slots=True)
class StructuralUnitSpec:
    unit_id: str
    residues: tuple[ResidueSpan, ...]

    def __post_init__(self) -> None:
        if not self.unit_id or any(char.isspace() for char in self.unit_id):
            raise ValueError("unit_id must be a non-empty token")
        if not self.residues:
            raise ValueError("a structural unit must contain at least one residue span")
        ordered = sorted(self.residues)
        for left, right in zip(ordered, ordered[1:]):
            if left.overlaps(right):
                raise ValueError(f"overlapping spans inside {self.unit_id}: {left} and {right}")

    @classmethod
    def from_strings(cls, unit_id: str, residues: Iterable[str]) -> "StructuralUnitSpec":
        return cls(unit_id=unit_id, residues=tuple(ResidueSpan.parse(item) for item in residues))


def validate_disjoint_units(units: Iterable[StructuralUnitSpec]) -> tuple[StructuralUnitSpec, ...]:
    """Validate stable identities and non-overlap across all structural units."""

    materialized = tuple(units)
    identifiers = [unit.unit_id for unit in materialized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("structural unit identifiers must be unique")
    for index, left_unit in enumerate(materialized):
        for right_unit in materialized[index + 1 :]:
            for left_span in left_unit.residues:
                for right_span in right_unit.residues:
                    if left_span.overlaps(right_span):
                        raise ValueError(
                            f"structural units {left_unit.unit_id} and {right_unit.unit_id} overlap "
                            f"at {left_span} / {right_span}"
                        )
    return materialized

