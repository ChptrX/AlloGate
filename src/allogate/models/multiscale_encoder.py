"""Study-independent residue → local element → structural unit encoder."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .aggregation import fixed_parent_mean, pair_add
from .graph import FeatureBaselines, GateControls, MultiscaleGraph
from .gvp import ChannelDimensions, GeometricVectorPerceptron, ScalarVector
from .interventions import apply_state_replacement
from .message_passing import RelationMessageLayer


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    residues: ScalarVector
    local_elements: ScalarVector
    structural_units: ScalarVector
    unit_contact_messages: ScalarVector
    unit_covalent_messages: ScalarVector
    unit_combined_messages: ScalarVector


class MultiscaleEncoder(nn.Module):
    """Encode three generic hierarchy levels while retaining relation provenance."""

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
        self.residue_projection = GeometricVectorPerceptron(residue_input_dimensions, hidden_dimensions)
        self.residue_layers = nn.ModuleList(
            RelationMessageLayer(hidden_dimensions, edge_dimensions) for _ in range(layers_per_scale)
        )
        self.local_layers = nn.ModuleList(
            RelationMessageLayer(hidden_dimensions, edge_dimensions) for _ in range(layers_per_scale)
        )
        self.unit_relation_layers = nn.ModuleList(
            RelationMessageLayer(hidden_dimensions, edge_dimensions) for _ in range(layers_per_scale)
        )

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

        local_elements = fixed_parent_mean(residues, graph.residue_to_local, controls.residue_to_local)
        for layer in self.local_layers:
            local_elements, _ = layer(local_elements, graph.local_edges, controls.local_edges)
        local_elements = apply_state_replacement(
            local_elements, controls.local_state, feature_baselines.local_scalar
        )

        structural_units = fixed_parent_mean(local_elements, graph.local_to_unit, controls.local_to_unit)
        contact_messages: ScalarVector | None = None
        covalent_messages: ScalarVector | None = None
        combined_messages: ScalarVector | None = None
        for layer in self.unit_relation_layers:
            contact_messages = layer.relation_aggregate(
                structural_units, graph.unit_contact_edges, controls.unit_contact_edges
            )
            covalent_messages = layer.relation_aggregate(
                structural_units, graph.unit_covalent_edges, controls.unit_covalent_edges
            )
            combined_messages = pair_add(contact_messages, covalent_messages)
            structural_units = layer.update_from_aggregate(structural_units, combined_messages)
        if contact_messages is None or covalent_messages is None or combined_messages is None:
            raise RuntimeError("the encoder has no structural-unit relation layer")
        structural_units = apply_state_replacement(
            structural_units, controls.unit_state, feature_baselines.unit_scalar
        )
        return EncoderOutput(
            residues=residues,
            local_elements=local_elements,
            structural_units=structural_units,
            unit_contact_messages=contact_messages,
            unit_covalent_messages=covalent_messages,
            unit_combined_messages=combined_messages,
        )
