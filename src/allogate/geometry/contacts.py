"""Backend-independent construction of deterministic directed geometric edges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .residue_features import cosine_switch


@dataclass(frozen=True, slots=True)
class DirectedGeometry:
    source: NDArray[np.int64]
    target: NDArray[np.int64]
    distance: NDArray[np.floating]
    unit_direction: NDArray[np.floating]
    baseline_weight: NDArray[np.floating]

    def __post_init__(self) -> None:
        source = np.asarray(self.source, dtype=np.int64)
        target = np.asarray(self.target, dtype=np.int64)
        distance = np.asarray(self.distance)
        direction = np.asarray(self.unit_direction)
        weight = np.asarray(self.baseline_weight)
        edge_count = len(source)
        if source.shape != (edge_count,) or target.shape != (edge_count,):
            raise ValueError("source and target must be one-dimensional")
        if distance.shape != (edge_count,) or weight.shape != (edge_count,):
            raise ValueError("distance and baseline_weight must have one value per edge")
        if direction.shape != (edge_count, 3):
            raise ValueError("unit_direction must have shape (edges, 3)")
        if np.any(distance < 0.0) or np.any(weight < 0.0):
            raise ValueError("distances and baseline weights cannot be negative")


def _coordinates(value: ArrayLike) -> NDArray[np.floating]:
    coordinates = np.asarray(value)
    if not np.issubdtype(coordinates.dtype, np.floating):
        coordinates = coordinates.astype(np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (nodes, 3)")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    return coordinates


def directed_geometry_from_pairs(coordinates: ArrayLike, undirected_pairs: ArrayLike) -> DirectedGeometry:
    """Expand canonical undirected pairs into target-major directed edges."""

    xyz = _coordinates(coordinates)
    pairs = np.asarray(undirected_pairs, dtype=np.int64)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("undirected_pairs must have shape (edges, 2)")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= len(xyz)):
        raise ValueError("an edge index is outside the coordinate array")
    if np.any(pairs[:, 0] >= pairs[:, 1]):
        raise ValueError("undirected pairs must be canonical with source < target")
    if len({tuple(pair) for pair in pairs.tolist()}) != len(pairs):
        raise ValueError("undirected pairs must be unique")

    source = np.concatenate((pairs[:, 0], pairs[:, 1]))
    target = np.concatenate((pairs[:, 1], pairs[:, 0]))
    order = np.lexsort((source, target))
    source = source[order]
    target = target[order]
    displacement = xyz[target] - xyz[source]
    distance = np.linalg.norm(displacement, axis=-1)
    direction = np.divide(
        displacement,
        distance[:, None],
        out=np.zeros_like(displacement),
        where=distance[:, None] > 0.0,
    )
    return DirectedGeometry(
        source=source,
        target=target,
        distance=distance,
        unit_direction=direction,
        baseline_weight=np.ones_like(distance),
    )


def contact_edges(
    coordinates: ArrayLike,
    graph_index: ArrayLike,
    *,
    switch_on: float,
    switch_off: float,
    excluded_pairs: ArrayLike | None = None,
) -> DirectedGeometry:
    """Build within-graph radius contacts and smooth all-one baseline weights."""

    xyz = _coordinates(coordinates)
    graph = np.asarray(graph_index, dtype=np.int64)
    if graph.shape != (len(xyz),) or np.any(graph < 0):
        raise ValueError("graph_index must be one non-negative integer per node")
    excluded: set[tuple[int, int]] = set()
    if excluded_pairs is not None:
        raw_excluded = np.asarray(excluded_pairs, dtype=np.int64)
        if raw_excluded.size:
            if raw_excluded.ndim != 2 or raw_excluded.shape[1] != 2:
                raise ValueError("excluded_pairs must have shape (pairs, 2)")
            excluded = {tuple(sorted((int(left), int(right)))) for left, right in raw_excluded}

    pairs: list[tuple[int, int]] = []
    for left in range(len(xyz)):
        candidates = np.flatnonzero(graph[left + 1 :] == graph[left]) + left + 1
        if not len(candidates):
            continue
        distances = np.linalg.norm(xyz[candidates] - xyz[left], axis=-1)
        for right in candidates[distances < switch_off]:
            pair = (left, int(right))
            if pair not in excluded:
                pairs.append(pair)
    geometry = directed_geometry_from_pairs(xyz, np.asarray(pairs, dtype=np.int64))
    weights = cosine_switch(geometry.distance, start=switch_on, stop=switch_off)
    return DirectedGeometry(
        source=geometry.source,
        target=geometry.target,
        distance=geometry.distance,
        unit_direction=geometry.unit_direction,
        baseline_weight=weights,
    )

