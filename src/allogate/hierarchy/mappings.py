"""Deterministic mappings between adjacent levels of the public hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class IndexMapping:
    """Map every child index to exactly one parent index."""

    child_to_parent: NDArray[np.int64]
    parent_count: int

    def __post_init__(self) -> None:
        mapping = np.asarray(self.child_to_parent, dtype=np.int64)
        if mapping.ndim != 1 or mapping.size == 0:
            raise ValueError("child_to_parent must be a non-empty one-dimensional array")
        if self.parent_count < 1:
            raise ValueError("parent_count must be positive")
        if mapping.min() < 0 or mapping.max() >= self.parent_count:
            raise ValueError("child_to_parent contains an out-of-range parent index")
        counts = np.bincount(mapping, minlength=self.parent_count)
        if np.any(counts == 0):
            raise ValueError("every parent must contain at least one child")
        frozen = np.array(mapping, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "child_to_parent", frozen)

    @property
    def child_count(self) -> int:
        return int(self.child_to_parent.size)

    @property
    def children_per_parent(self) -> NDArray[np.int64]:
        return np.bincount(self.child_to_parent, minlength=self.parent_count).astype(np.int64)

    def compose(self, parent_mapping: "IndexMapping") -> "IndexMapping":
        if self.parent_count != parent_mapping.child_count:
            raise ValueError("adjacent mappings do not share the intermediate level")
        return IndexMapping(parent_mapping.child_to_parent[self.child_to_parent], parent_mapping.parent_count)


def _balanced_sizes(length: int, *, target: int, minimum: int, maximum: int) -> list[int]:
    if length < minimum:
        raise ValueError(f"a contiguous hierarchy run of length {length} is shorter than minimum {minimum}")
    least_segments = math.ceil(length / maximum)
    most_segments = length // minimum
    if least_segments > most_segments:
        raise ValueError(f"cannot partition run of length {length} within [{minimum}, {maximum}]")
    preferred = max(1, round(length / target))
    segment_count = min(max(preferred, least_segments), most_segments)
    quotient, remainder = divmod(length, segment_count)
    sizes = [quotient + (index < remainder) for index in range(segment_count)]
    if min(sizes) < minimum or max(sizes) > maximum:
        raise RuntimeError("internal local-element partitioning error")
    return sizes


def build_contiguous_local_elements(
    chain_index: ArrayLike,
    structural_unit_index: ArrayLike,
    *,
    target_size: int = 6,
    minimum_size: int = 4,
    maximum_size: int = 8,
) -> tuple[IndexMapping, IndexMapping]:
    """Partition ordered residues without crossing chain or structural-unit boundaries.

    Returns residue-to-local-element and local-element-to-structural-unit mappings.
    Structural-unit indices must be contiguous integers starting at zero.
    """

    chains = np.asarray(chain_index)
    units = np.asarray(structural_unit_index, dtype=np.int64)
    if chains.ndim != 1 or units.ndim != 1 or len(chains) != len(units) or len(units) == 0:
        raise ValueError("chain and structural-unit indices must be aligned non-empty vectors")
    if target_size < 1 or minimum_size < 1 or not minimum_size <= target_size <= maximum_size:
        raise ValueError("sizes must satisfy 1 <= minimum <= target <= maximum")
    unique_units = set(int(value) for value in np.unique(units))
    if units.min() != 0 or unique_units != set(range(int(units.max()) + 1)):
        raise ValueError("structural-unit indices must be contiguous integers starting at zero")

    residue_to_local = np.empty(len(units), dtype=np.int64)
    local_to_unit: list[int] = []
    local_index = 0
    run_start = 0
    for position in range(1, len(units) + 1):
        boundary = position == len(units)
        if not boundary:
            boundary = chains[position] != chains[run_start] or units[position] != units[run_start]
        if not boundary:
            continue
        cursor = run_start
        for size in _balanced_sizes(
            position - run_start,
            target=target_size,
            minimum=minimum_size,
            maximum=maximum_size,
        ):
            residue_to_local[cursor : cursor + size] = local_index
            local_to_unit.append(int(units[run_start]))
            cursor += size
            local_index += 1
        run_start = position

    residue_mapping = IndexMapping(residue_to_local, local_index)
    unit_count = int(units.max()) + 1
    local_mapping = IndexMapping(np.asarray(local_to_unit, dtype=np.int64), unit_count)
    return residue_mapping, local_mapping

