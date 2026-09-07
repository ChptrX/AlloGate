"""Typed tensor contracts for protein-independent four-level graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .gvp import ScalarVector


def _pair_to(
    value: ScalarVector,
    device: torch.device | str,
    dtype: torch.dtype | None,
) -> ScalarVector:
    scalar, vector = value
    return (
        scalar.to(device=device, dtype=dtype or scalar.dtype),
        vector.to(device=device, dtype=dtype or vector.dtype),
    )


@dataclass(frozen=True, slots=True)
class DirectedEdges:
    """Directed homogeneous edges with frozen all-one baseline weights."""

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
        if not bool(torch.isfinite(self.baseline_weight).all()) or bool(
            torch.any(self.baseline_weight < 0)
        ):
            raise ValueError("baseline weights must be finite and non-negative")

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "DirectedEdges":
        return DirectedEdges(
            self.source.to(device=device),
            self.target.to(device=device),
            _pair_to(self.features, device, dtype),
            self.baseline_weight.to(
                device=device, dtype=dtype or self.baseline_weight.dtype
            ),
            self.node_count,
        )


@dataclass(frozen=True, slots=True)
class ParentIndex:
    """Map every child to exactly one parent at the next hierarchy scale."""

    child_to_parent: Tensor
    parent_count: int

    def __post_init__(self) -> None:
        if self.child_to_parent.dtype != torch.long or self.child_to_parent.ndim != 1:
            raise TypeError("child_to_parent must be a one-dimensional torch.long tensor")
        if self.parent_count < 1 or not self.child_to_parent.numel():
            raise ValueError("ParentIndex must contain children and at least one parent")
        if bool(torch.any(self.child_to_parent < 0)) or int(
            self.child_to_parent.max()
        ) >= self.parent_count:
            raise ValueError("child_to_parent contains an out-of-range index")
        counts = torch.bincount(self.child_to_parent, minlength=self.parent_count)
        if bool(torch.any(counts == 0)):
            raise ValueError("every parent must contain at least one child")

    @property
    def child_count(self) -> int:
        return int(self.child_to_parent.numel())

    def to(self, device: torch.device | str) -> "ParentIndex":
        return ParentIndex(self.child_to_parent.to(device=device), self.parent_count)


@dataclass(frozen=True, slots=True)
class InterfaceEdges:
    """Local-unit edges grouped into explicit, possibly inactive interfaces."""

    edges: DirectedEdges
    edge_to_interface: Tensor
    interface_count: int

    def __post_init__(self) -> None:
        edge_count = self.edges.source.numel()
        if self.interface_count < 1:
            raise ValueError("interface_count must be positive")
        if (
            self.edge_to_interface.dtype != torch.long
            or self.edge_to_interface.shape != (edge_count,)
        ):
            raise TypeError("edge_to_interface must be one torch.long value per edge")
        if self.edge_to_interface.device != self.edges.source.device:
            raise ValueError("interface grouping and edges must share a device")
        if edge_count and (
            bool(torch.any(self.edge_to_interface < 0))
            or int(self.edge_to_interface.max()) >= self.interface_count
        ):
            raise ValueError("edge_to_interface contains an out-of-range interface")

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "InterfaceEdges":
        return InterfaceEdges(
            self.edges.to(device, dtype=dtype),
            self.edge_to_interface.to(device=device),
            self.interface_count,
        )


@dataclass(frozen=True, slots=True)
class InterfaceIncidence:
    """Connect every unordered interface to its two global structural units."""

    interface_index: Tensor
    global_unit_index: Tensor
    baseline_weight: Tensor
    interface_count: int
    global_unit_count: int

    def __post_init__(self) -> None:
        count = self.interface_index.numel()
        if self.interface_count < 1 or self.global_unit_count < 1:
            raise ValueError("interface and global structural unit counts must be positive")
        if (
            self.interface_index.dtype != torch.long
            or self.global_unit_index.dtype != torch.long
            or self.interface_index.shape != (count,)
            or self.global_unit_index.shape != (count,)
        ):
            raise TypeError("interface incidence indices must be aligned torch.long vectors")
        if self.baseline_weight.shape != (count,):
            raise ValueError("interface incidence requires one baseline weight per row")
        tensors = (self.interface_index, self.global_unit_index, self.baseline_weight)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("interface incidence tensors must share a device")
        if count:
            if (
                bool(torch.any(self.interface_index < 0))
                or int(self.interface_index.max()) >= self.interface_count
                or bool(torch.any(self.global_unit_index < 0))
                or int(self.global_unit_index.max()) >= self.global_unit_count
            ):
                raise ValueError("interface incidence contains an out-of-range index")
        if not bool(torch.isfinite(self.baseline_weight).all()) or bool(
            torch.any(self.baseline_weight < 0)
        ):
            raise ValueError("interface incidence weights must be finite and non-negative")
        endpoint_counts = torch.bincount(
            self.interface_index, minlength=self.interface_count
        )
        if not bool(torch.all(endpoint_counts == 2)):
            raise ValueError("every interface must have exactly two endpoint rows")
        encoded = self.interface_index * self.global_unit_count + self.global_unit_index
        if encoded.unique().numel() != count:
            raise ValueError("an interface cannot repeat a global structural unit endpoint")

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "InterfaceIncidence":
        return InterfaceIncidence(
            self.interface_index.to(device=device),
            self.global_unit_index.to(device=device),
            self.baseline_weight.to(
                device=device, dtype=dtype or self.baseline_weight.dtype
            ),
            self.interface_count,
            self.global_unit_count,
        )


@dataclass(frozen=True, slots=True)
class MultiscaleGraph:
    residue_features: ScalarVector
    residue_edges: DirectedEdges
    residue_to_local_unit: ParentIndex
    local_unit_edges: DirectedEdges
    local_unit_to_global_unit: ParentIndex
    interface_edges: InterfaceEdges
    interface_to_global_unit: InterfaceIncidence
    global_unit_covalent_edges: DirectedEdges

    def __post_init__(self) -> None:
        residue_scalar, residue_vector = self.residue_features
        residue_count = residue_scalar.shape[0]
        if residue_scalar.ndim != 2 or residue_vector.ndim != 3:
            raise ValueError("residue features must have shapes (N, S) and (N, V, 3)")
        if residue_vector.shape[0] != residue_count or residue_vector.shape[-1] != 3:
            raise ValueError("residue scalar and vector features must align")
        if self.residue_edges.node_count != residue_count:
            raise ValueError("residue edge node_count does not match residue features")
        if self.residue_to_local_unit.child_count != residue_count:
            raise ValueError("residue_to_local_unit does not cover every residue")
        local_count = self.residue_to_local_unit.parent_count
        if self.local_unit_edges.node_count != local_count:
            raise ValueError("local-unit edge node_count does not match local units")
        if self.local_unit_to_global_unit.child_count != local_count:
            raise ValueError("local_unit_to_global_unit does not cover every local unit")
        if self.interface_edges.edges.node_count != local_count:
            raise ValueError("interface member edges must connect local structural units")
        interface_count = self.interface_edges.interface_count
        if self.interface_to_global_unit.interface_count != interface_count:
            raise ValueError("interface definitions and incidence counts differ")
        global_count = self.local_unit_to_global_unit.parent_count
        if self.interface_to_global_unit.global_unit_count != global_count:
            raise ValueError("interface incidence and global structural unit counts differ")
        if self.global_unit_covalent_edges.node_count != global_count:
            raise ValueError("global covalent edges do not match global structural units")
        reference_device = residue_scalar.device
        devices = {
            reference_device,
            residue_vector.device,
            self.residue_edges.source.device,
            self.residue_to_local_unit.child_to_parent.device,
            self.local_unit_edges.source.device,
            self.local_unit_to_global_unit.child_to_parent.device,
            self.interface_edges.edges.source.device,
            self.interface_to_global_unit.interface_index.device,
            self.global_unit_covalent_edges.source.device,
        }
        if len(devices) != 1:
            raise ValueError("all multiscale graph tensors must share a device")

    @property
    def residue_count(self) -> int:
        return int(self.residue_features[0].shape[0])

    @property
    def local_unit_count(self) -> int:
        return self.residue_to_local_unit.parent_count

    @property
    def interface_count(self) -> int:
        return self.interface_edges.interface_count

    @property
    def global_unit_count(self) -> int:
        return self.local_unit_to_global_unit.parent_count

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "MultiscaleGraph":
        return MultiscaleGraph(
            residue_features=_pair_to(self.residue_features, device, dtype),
            residue_edges=self.residue_edges.to(device, dtype=dtype),
            residue_to_local_unit=self.residue_to_local_unit.to(device),
            local_unit_edges=self.local_unit_edges.to(device, dtype=dtype),
            local_unit_to_global_unit=self.local_unit_to_global_unit.to(device),
            interface_edges=self.interface_edges.to(device, dtype=dtype),
            interface_to_global_unit=self.interface_to_global_unit.to(
                device, dtype=dtype
            ),
            global_unit_covalent_edges=self.global_unit_covalent_edges.to(
                device, dtype=dtype
            ),
        )


@dataclass(frozen=True, slots=True)
class GateControls:
    residue_state: Tensor | None = None
    residue_edges: Tensor | None = None
    residue_to_local_unit: Tensor | None = None
    local_unit_state: Tensor | None = None
    local_unit_edges: Tensor | None = None
    local_unit_to_global_unit: Tensor | None = None
    interface_edges: Tensor | None = None
    interface_state: Tensor | None = None
    interface_to_global_unit: Tensor | None = None
    global_unit_state: Tensor | None = None
    global_unit_covalent_edges: Tensor | None = None


@dataclass(frozen=True, slots=True)
class FeatureBaselines:
    """Frozen scalar replacement values for State Gates at every node level."""

    residue_scalar: Tensor | None = None
    local_unit_scalar: Tensor | None = None
    interface_scalar: Tensor | None = None
    global_unit_scalar: Tensor | None = None
