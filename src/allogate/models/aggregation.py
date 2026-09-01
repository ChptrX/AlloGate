"""Double-backward-safe aggregation with immutable all-one denominators."""

from __future__ import annotations

import torch
from torch import Tensor

from .graph import ParentIndex
from .gvp import ScalarVector


def pair_add(left: ScalarVector, right: ScalarVector) -> ScalarVector:
    return left[0] + right[0], left[1] + right[1]


def pair_concat(*values: ScalarVector) -> ScalarVector:
    if not values:
        raise ValueError("at least one scalar/vector value is required")
    return torch.cat([value[0] for value in values], dim=-1), torch.cat(
        [value[1] for value in values], dim=-2
    )


def pair_index(value: ScalarVector, index: Tensor) -> ScalarVector:
    return value[0][index], value[1][index]


def _gate_or_ones(gate: Tensor | None, reference: Tensor) -> Tensor:
    if gate is None:
        return torch.ones_like(reference)
    if gate.shape != reference.shape:
        raise ValueError(f"gate shape {tuple(gate.shape)} does not match {tuple(reference.shape)}")
    return gate.to(dtype=reference.dtype, device=reference.device)


def _index_sum(values: Tensor, index: Tensor, output_count: int) -> Tensor:
    output = values.new_zeros((output_count, *values.shape[1:]))
    if index.numel():
        output.index_add_(0, index, values)
    return output


def fixed_edge_mean(
    messages: ScalarVector,
    target: Tensor,
    baseline_weight: Tensor,
    node_count: int,
    gate: Tensor | None = None,
) -> ScalarVector:
    """Aggregate controlled edges without recomputing the baseline denominator."""

    scalar, vector = messages
    if scalar.shape[0] != target.numel() or vector.shape[0] != target.numel():
        raise ValueError("messages and target indices must align")
    if baseline_weight.shape != target.shape:
        raise ValueError("baseline_weight and target must align")
    controlled_weight = baseline_weight * _gate_or_ones(gate, baseline_weight)
    scalar_sum = _index_sum(scalar * controlled_weight.unsqueeze(-1), target, node_count)
    vector_sum = _index_sum(vector * controlled_weight[:, None, None], target, node_count)
    denominator = _index_sum(baseline_weight, target, node_count).clamp_min(1.0)
    return scalar_sum / denominator[:, None], vector_sum / denominator[:, None, None]


def fixed_parent_mean(
    children: ScalarVector,
    mapping: ParentIndex,
    gate: Tensor | None = None,
) -> ScalarVector:
    """Pool children with the all-one child count as an immutable denominator."""

    scalar, vector = children
    if scalar.shape[0] != mapping.child_count or vector.shape[0] != mapping.child_count:
        raise ValueError("children do not match the hierarchy mapping")
    if mapping.child_to_parent.device != scalar.device or vector.device != scalar.device:
        raise ValueError("hierarchy indices and child features must share a device")
    baseline = scalar.new_ones(mapping.child_count)
    controlled = _gate_or_ones(gate, baseline)
    scalar_sum = _index_sum(scalar * controlled[:, None], mapping.child_to_parent, mapping.parent_count)
    vector_sum = _index_sum(vector * controlled[:, None, None], mapping.child_to_parent, mapping.parent_count)
    denominator = torch.bincount(mapping.child_to_parent, minlength=mapping.parent_count).to(scalar.dtype)
    return scalar_sum / denominator[:, None], vector_sum / denominator[:, None, None]


def fixed_graph_mean(
    items: ScalarVector,
    graph_index: Tensor,
    graph_count: int,
    gate: Tensor | None = None,
) -> ScalarVector:
    """Fixed-count pooling for a flat variable-size graph batch."""

    mapping = ParentIndex(graph_index, graph_count)
    return fixed_parent_mean(items, mapping, gate)
