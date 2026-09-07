"""Backend-ready lagged graph batches for public slow-CV training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from allogate.models.graph import MultiscaleGraph
from allogate.models.graph import (
    DirectedEdges,
    InterfaceEdges,
    InterfaceIncidence,
    ParentIndex,
)


def _validate_graph_index(
    value: Tensor,
    *,
    item_count: int,
    graph_count: int,
    name: str,
) -> None:
    if value.dtype != torch.long or value.shape != (item_count,):
        raise TypeError(f"{name} must be one torch.long value per item")
    if value.device.type != "cpu" and item_count == 0:
        raise ValueError(f"{name} cannot be empty")
    if item_count == 0 or bool(torch.any(value < 0)) or int(value.max()) >= graph_count:
        raise ValueError(f"{name} contains an out-of-range graph index")
    counts = torch.bincount(value, minlength=graph_count)
    if bool(torch.any(counts == 0)):
        raise ValueError(f"{name} must represent every graph in the batch")


@dataclass(frozen=True, slots=True)
class ModelBatch:
    """One variable-size batch plus graph membership at every node level."""

    graph: MultiscaleGraph
    residue_graph_index: Tensor
    local_unit_graph_index: Tensor
    interface_graph_index: Tensor
    global_unit_graph_index: Tensor
    graph_count: int

    def __post_init__(self) -> None:
        if self.graph_count < 1:
            raise ValueError("graph_count must be positive")
        for value, count, name in (
            (
                self.residue_graph_index,
                self.graph.residue_count,
                "residue_graph_index",
            ),
            (
                self.local_unit_graph_index,
                self.graph.local_unit_count,
                "local_unit_graph_index",
            ),
            (
                self.interface_graph_index,
                self.graph.interface_count,
                "interface_graph_index",
            ),
            (
                self.global_unit_graph_index,
                self.graph.global_unit_count,
                "global_unit_graph_index",
            ),
        ):
            _validate_graph_index(
                value,
                item_count=count,
                graph_count=self.graph_count,
                name=name,
            )
            if value.device != self.graph.residue_features[0].device:
                raise ValueError("graph membership and graph tensors must share a device")

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "ModelBatch":
        return ModelBatch(
            graph=self.graph.to(device, dtype=dtype),
            residue_graph_index=self.residue_graph_index.to(device=device),
            local_unit_graph_index=self.local_unit_graph_index.to(device=device),
            interface_graph_index=self.interface_graph_index.to(device=device),
            global_unit_graph_index=self.global_unit_graph_index.to(device=device),
            graph_count=self.graph_count,
        )

    def model_arguments(self) -> dict[str, object]:
        return {
            "residue_graph_index": self.residue_graph_index,
            "local_unit_graph_index": self.local_unit_graph_index,
            "interface_graph_index": self.interface_graph_index,
            "global_unit_graph_index": self.global_unit_graph_index,
            "graph_count": self.graph_count,
        }


@dataclass(frozen=True, slots=True)
class LaggedBatch:
    """Paired configurations at time t and t+lag from the same trajectories."""

    time: ModelBatch
    lagged: ModelBatch
    trajectory_index: Tensor | None = None

    def __post_init__(self) -> None:
        if self.time.graph_count != self.lagged.graph_count:
            raise ValueError("lagged sides must contain the same number of graphs")
        if self.trajectory_index is not None:
            if (
                self.trajectory_index.dtype != torch.long
                or self.trajectory_index.shape != (self.time.graph_count,)
            ):
                raise TypeError(
                    "trajectory_index must be one torch.long value per lagged pair"
                )
            if bool(torch.any(self.trajectory_index < 0)):
                raise ValueError("trajectory_index cannot contain negative values")

    @property
    def pair_count(self) -> int:
        return self.time.graph_count

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "LaggedBatch":
        return LaggedBatch(
            self.time.to(device, dtype=dtype),
            self.lagged.to(device, dtype=dtype),
            None
            if self.trajectory_index is None
            else self.trajectory_index.to(device=device),
        )


def _collate_edges(
    values: tuple[DirectedEdges, ...],
    node_offsets: tuple[int, ...],
) -> DirectedEdges:
    if not values:
        raise ValueError("cannot collate an empty edge collection")
    scalar_channels = {value.features[0].shape[1] for value in values}
    vector_channels = {value.features[1].shape[1] for value in values}
    if len(scalar_channels) != 1 or len(vector_channels) != 1:
        raise ValueError("edge feature channel counts must agree across batches")
    return DirectedEdges(
        source=torch.cat(
            [value.source + offset for value, offset in zip(values, node_offsets)]
        ),
        target=torch.cat(
            [value.target + offset for value, offset in zip(values, node_offsets)]
        ),
        features=(
            torch.cat([value.features[0] for value in values]),
            torch.cat([value.features[1] for value in values]),
        ),
        baseline_weight=torch.cat([value.baseline_weight for value in values]),
        node_count=sum(value.node_count for value in values),
    )


def _offsets(counts: tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    total = 0
    for count in counts:
        result.append(total)
        total += count
    return tuple(result)


def collate_model_batches(values: Iterable[ModelBatch]) -> ModelBatch:
    """Concatenate variable-size model batches without assuming one protein size."""

    batches = tuple(values)
    if not batches:
        raise ValueError("cannot collate an empty model-batch collection")
    graph_offsets = _offsets(tuple(item.graph_count for item in batches))
    residue_offsets = _offsets(tuple(item.graph.residue_count for item in batches))
    local_offsets = _offsets(tuple(item.graph.local_unit_count for item in batches))
    interface_offsets = _offsets(
        tuple(item.graph.interface_count for item in batches)
    )
    global_offsets = _offsets(tuple(item.graph.global_unit_count for item in batches))
    graphs = tuple(item.graph for item in batches)
    graph = MultiscaleGraph(
        residue_features=(
            torch.cat([item.residue_features[0] for item in graphs]),
            torch.cat([item.residue_features[1] for item in graphs]),
        ),
        residue_edges=_collate_edges(
            tuple(item.residue_edges for item in graphs), residue_offsets
        ),
        residue_to_local_unit=ParentIndex(
            torch.cat(
                [
                    item.residue_to_local_unit.child_to_parent + offset
                    for item, offset in zip(graphs, local_offsets)
                ]
            ),
            sum(item.local_unit_count for item in graphs),
        ),
        local_unit_edges=_collate_edges(
            tuple(item.local_unit_edges for item in graphs), local_offsets
        ),
        local_unit_to_global_unit=ParentIndex(
            torch.cat(
                [
                    item.local_unit_to_global_unit.child_to_parent + offset
                    for item, offset in zip(graphs, global_offsets)
                ]
            ),
            sum(item.global_unit_count for item in graphs),
        ),
        interface_edges=InterfaceEdges(
            _collate_edges(
                tuple(item.interface_edges.edges for item in graphs),
                local_offsets,
            ),
            torch.cat(
                [
                    item.interface_edges.edge_to_interface + offset
                    for item, offset in zip(graphs, interface_offsets)
                ]
            ),
            sum(item.interface_count for item in graphs),
        ),
        interface_to_global_unit=InterfaceIncidence(
            interface_index=torch.cat(
                [
                    item.interface_to_global_unit.interface_index
                    + interface_offset
                    for item, interface_offset in zip(graphs, interface_offsets)
                ]
            ),
            global_unit_index=torch.cat(
                [
                    item.interface_to_global_unit.global_unit_index + global_offset
                    for item, global_offset in zip(graphs, global_offsets)
                ]
            ),
            baseline_weight=torch.cat(
                [item.interface_to_global_unit.baseline_weight for item in graphs]
            ),
            interface_count=sum(item.interface_count for item in graphs),
            global_unit_count=sum(item.global_unit_count for item in graphs),
        ),
        global_unit_covalent_edges=_collate_edges(
            tuple(item.global_unit_covalent_edges for item in graphs),
            global_offsets,
        ),
    )
    return ModelBatch(
        graph=graph,
        residue_graph_index=torch.cat(
            [
                item.residue_graph_index + offset
                for item, offset in zip(batches, graph_offsets)
            ]
        ),
        local_unit_graph_index=torch.cat(
            [
                item.local_unit_graph_index + offset
                for item, offset in zip(batches, graph_offsets)
            ]
        ),
        interface_graph_index=torch.cat(
            [
                item.interface_graph_index + offset
                for item, offset in zip(batches, graph_offsets)
            ]
        ),
        global_unit_graph_index=torch.cat(
            [
                item.global_unit_graph_index + offset
                for item, offset in zip(batches, graph_offsets)
            ]
        ),
        graph_count=sum(item.graph_count for item in batches),
    )


def collate_lagged_batches(values: Iterable[LaggedBatch]) -> LaggedBatch:
    """Collate lagged samples while preserving per-pair trajectory labels."""

    batches = tuple(values)
    if not batches:
        raise ValueError("cannot collate an empty lagged-batch collection")
    labels = tuple(item.trajectory_index for item in batches)
    if any(label is None for label in labels) and any(label is not None for label in labels):
        raise ValueError("trajectory labels must be present for every sample or none")
    trajectory_index = (
        None
        if labels[0] is None
        else torch.cat([label for label in labels if label is not None])
    )
    return LaggedBatch(
        collate_model_batches(item.time for item in batches),
        collate_model_batches(item.lagged for item in batches),
        trajectory_index,
    )
