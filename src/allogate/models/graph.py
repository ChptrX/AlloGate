"""Typed tensor contracts for backend-ready multiscale graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .gvp import ScalarVector


@dataclass(frozen=True, slots=True)
class DirectedEdges:
    """Directed edges with invariant/equivariant features and frozen baseline weights."""

    source: Tensor
    target: Tensor
    features: ScalarVector
    baseline_weight: Tensor
    node_count: int

    def __post_init__(self) -> None:
        scalar, vector = self.features
        edge_count = self.source.numel()
        if self.node_count < 1:
            raise ValueError("node_count must be positive")
        if self.source.dtype != torch.long or self.target.dtype != torch.long:
            raise TypeError("source and target indices must use torch.long")
        if self.source.shape != (edge_count,) or self.target.shape != (edge_count,):
            raise ValueError("source and target must be one-dimensional")
        if scalar.shape[0] != edge_count or vector.shape[0] != edge_count:
            raise ValueError("edge features must have one row per edge")
        if scalar.ndim != 2 or vector.ndim != 3 or vector.shape[-1] != 3:
            raise ValueError("edge features must have shapes (E, S) and (E, V, 3)")
        if self.baseline_weight.shape != (edge_count,):
            raise ValueError("baseline_weight must have one value per edge")
        tensors = (self.source, self.target, scalar, vector, self.baseline_weight)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all edge tensors must share a device")
        if edge_count:
            if bool(torch.any(self.source < 0)) or bool(torch.any(self.target < 0)):
                raise ValueError("edge indices cannot be negative")
            if int(torch.maximum(self.source.max(), self.target.max())) >= self.node_count:
                raise ValueError("an edge index is outside node_count")
        if bool(torch.any(self.baseline_weight < 0)):
            raise ValueError("baseline weights cannot be negative")


@dataclass(frozen=True, slots=True)
class ParentIndex:
    """Tensor mapping every child to one parent at the next hierarchy scale."""

    child_to_parent: Tensor
    parent_count: int

    def __post_init__(self) -> None:
        if self.child_to_parent.dtype != torch.long or self.child_to_parent.ndim != 1:
            raise TypeError("child_to_parent must be a one-dimensional torch.long tensor")
        if self.parent_count < 1 or not self.child_to_parent.numel():
            raise ValueError("ParentIndex must contain children and at least one parent")
        if bool(torch.any(self.child_to_parent < 0)) or int(self.child_to_parent.max()) >= self.parent_count:
            raise ValueError("child_to_parent contains an out-of-range index")
        counts = torch.bincount(self.child_to_parent, minlength=self.parent_count)
        if bool(torch.any(counts == 0)):
            raise ValueError("every parent must contain at least one child")

    @property
    def child_count(self) -> int:
        return int(self.child_to_parent.numel())


@dataclass(frozen=True, slots=True)
class MultiscaleGraph:
    residue_features: ScalarVector
    residue_edges: DirectedEdges
    residue_to_local: ParentIndex
    local_edges: DirectedEdges
    local_to_unit: ParentIndex
    unit_contact_edges: DirectedEdges
    unit_covalent_edges: DirectedEdges

    def __post_init__(self) -> None:
        residue_scalar, residue_vector = self.residue_features
        residue_count = residue_scalar.shape[0]
        if residue_vector.shape[0] != residue_count:
            raise ValueError("residue scalar and vector features must align")
        if self.residue_edges.node_count != residue_count:
            raise ValueError("residue edge node_count does not match residue features")
        if self.residue_to_local.child_count != residue_count:
            raise ValueError("residue_to_local does not cover every residue")
        if self.local_edges.node_count != self.residue_to_local.parent_count:
            raise ValueError("local edge node_count does not match local elements")
        if self.local_to_unit.child_count != self.residue_to_local.parent_count:
            raise ValueError("local_to_unit does not cover every local element")
        unit_count = self.local_to_unit.parent_count
        if self.unit_contact_edges.node_count != unit_count or self.unit_covalent_edges.node_count != unit_count:
            raise ValueError("structural-unit edge node_count does not match structural units")


@dataclass(frozen=True, slots=True)
class GateControls:
    residue_state: Tensor | None = None
    residue_edges: Tensor | None = None
    residue_to_local: Tensor | None = None
    local_state: Tensor | None = None
    local_edges: Tensor | None = None
    local_to_unit: Tensor | None = None
    unit_state: Tensor | None = None
    unit_contact_edges: Tensor | None = None
    unit_covalent_edges: Tensor | None = None


@dataclass(frozen=True, slots=True)
class FeatureBaselines:
    """Frozen scalar replacement values for State Gates at each scale."""

    residue_scalar: Tensor | None = None
    local_scalar: Tensor | None = None
    unit_scalar: Tensor | None = None

