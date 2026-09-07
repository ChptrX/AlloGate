"""Residue -> local structural unit -> interface -> global unit encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .aggregation import (
    fixed_edge_mean,
    fixed_interface_mean,
    fixed_parent_mean,
    pair_add,
    pair_concat,
)
from .graph import FeatureBaselines, GateControls, MultiscaleGraph
from .gvp import ChannelDimensions, GeometricVectorPerceptron, ScalarVector
from .interventions import apply_state_replacement
from .message_passing import RelationMessageLayer


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    residues: ScalarVector
    local_structural_units: ScalarVector
    interfaces: ScalarVector
    global_structural_units: ScalarVector
    global_interface_messages: ScalarVector
    global_covalent_messages: ScalarVector
    global_combined_messages: ScalarVector


class MultiscaleEncoder(nn.Module):
    """Encode four configurable levels while retaining interface provenance."""

    def __init__(
        self,
        residue_input_dimensions: ChannelDimensions,
        edge_dimensions: ChannelDimensions,
        hidden_dimensions: ChannelDimensions,
        *,
        layers_per_scale: int = 1,
    ) -> None:
        super().__init__()
        if layers_per_scale < 1:
            raise ValueError("layers_per_scale must be positive")
        self.residue_input_dimensions = residue_input_dimensions
        self.edge_dimensions = edge_dimensions
        self.hidden_dimensions = hidden_dimensions
        self.residue_projection = GeometricVectorPerceptron(
            residue_input_dimensions, hidden_dimensions
        )
        self.residue_layers = nn.ModuleList(
            RelationMessageLayer(hidden_dimensions, edge_dimensions)
            for _ in range(layers_per_scale)
        )
        self.local_unit_layers = nn.ModuleList(
            RelationMessageLayer(hidden_dimensions, edge_dimensions)
            for _ in range(layers_per_scale)
        )
        self.interface_message = RelationMessageLayer(
            hidden_dimensions, edge_dimensions
        )
        self.interface_layers = nn.ModuleList(
            GeometricVectorPerceptron(hidden_dimensions, hidden_dimensions)
            for _ in range(layers_per_scale)
        )
        doubled = ChannelDimensions(
            2 * hidden_dimensions.scalars, 2 * hidden_dimensions.vectors
        )
        self.global_unit_fusion = GeometricVectorPerceptron(
            doubled, hidden_dimensions
        )
        self.global_unit_layers = nn.ModuleList(
            RelationMessageLayer(hidden_dimensions, edge_dimensions)
            for _ in range(layers_per_scale)
        )

    @staticmethod
    def _zero_anchored(
        layer: GeometricVectorPerceptron, value: ScalarVector
    ) -> ScalarVector:
        zero = torch.zeros_like(value[0]), torch.zeros_like(value[1])
        transformed = layer(value)
        baseline = layer(zero)
        return transformed[0] - baseline[0], transformed[1] - baseline[1]

    def forward(
        self,
        graph: MultiscaleGraph,
        gates: GateControls | None = None,
        baselines: FeatureBaselines | None = None,
    ) -> EncoderOutput:
        controls = gates or GateControls()
        feature_baselines = baselines or FeatureBaselines()
        residues = self.residue_projection(graph.residue_features)
        for layer in self.residue_layers:
            residues, _ = layer(residues, graph.residue_edges, controls.residue_edges)
        residues = apply_state_replacement(
            residues, controls.residue_state, feature_baselines.residue_scalar
        )

        local_units = fixed_parent_mean(
            residues,
            graph.residue_to_local_unit,
            controls.residue_to_local_unit,
        )
        for layer in self.local_unit_layers:
            local_units, _ = layer(
                local_units, graph.local_unit_edges, controls.local_unit_edges
            )
        local_units = apply_state_replacement(
            local_units,
            controls.local_unit_state,
            feature_baselines.local_unit_scalar,
        )

        member_edges = graph.interface_edges
        member_messages = self.interface_message.edge_messages(
            local_units, member_edges.edges
        )
        interfaces = fixed_edge_mean(
            member_messages,
            member_edges.edge_to_interface,
            member_edges.edges.baseline_weight,
            member_edges.interface_count,
            controls.interface_edges,
        )
        active = torch.bincount(
            member_edges.edge_to_interface,
            weights=member_edges.edges.baseline_weight,
            minlength=member_edges.interface_count,
        ) > 0
        for layer in self.interface_layers:
            interfaces = self._zero_anchored(layer, interfaces)
            interfaces = (
                interfaces[0] * active[:, None],
                interfaces[1] * active[:, None, None],
            )
        interfaces = apply_state_replacement(
            interfaces,
            controls.interface_state,
            feature_baselines.interface_scalar,
        )
        interfaces = (
            interfaces[0] * active[:, None],
            interfaces[1] * active[:, None, None],
        )

        global_interface_messages = fixed_interface_mean(
            interfaces,
            graph.interface_to_global_unit,
            controls.interface_to_global_unit,
        )
        local_global = fixed_parent_mean(
            local_units,
            graph.local_unit_to_global_unit,
            controls.local_unit_to_global_unit,
        )
        global_units = self.global_unit_fusion(
            pair_concat(local_global, global_interface_messages)
        )

        covalent_messages: ScalarVector | None = None
        combined_messages: ScalarVector | None = None
        for layer in self.global_unit_layers:
            covalent_messages = layer.relation_aggregate(
                global_units,
                graph.global_unit_covalent_edges,
                controls.global_unit_covalent_edges,
            )
            combined_messages = pair_add(
                global_interface_messages, covalent_messages
            )
            global_units = layer.update_from_aggregate(
                global_units, combined_messages
            )
        if covalent_messages is None or combined_messages is None:
            raise RuntimeError("the encoder has no global structural unit layer")
        global_units = apply_state_replacement(
            global_units,
            controls.global_unit_state,
            feature_baselines.global_unit_scalar,
        )
        return EncoderOutput(
            residues=residues,
            local_structural_units=local_units,
            interfaces=interfaces,
            global_structural_units=global_units,
            global_interface_messages=global_interface_messages,
            global_covalent_messages=covalent_messages,
            global_combined_messages=combined_messages,
        )
