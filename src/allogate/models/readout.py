"""Invariant Direct and zero-preserving Route readouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .aggregation import fixed_graph_mean
from .gvp import ChannelDimensions, GeometricVectorPerceptron, ScalarVector
from .multiscale_encoder import EncoderOutput


class _ScalarHead(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, output_channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class DirectReadout(nn.Module):
    """Read pooled information stored at all three hierarchy scales."""

    def __init__(
        self,
        feature_dimensions: ChannelDimensions,
        *,
        projection_channels: int,
        hidden_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        projection_dimensions = ChannelDimensions(projection_channels, 0)
        self.residue_projection = GeometricVectorPerceptron(feature_dimensions, projection_dimensions)
        self.local_projection = GeometricVectorPerceptron(feature_dimensions, projection_dimensions)
        self.unit_projection = GeometricVectorPerceptron(feature_dimensions, projection_dimensions)
        self.head = _ScalarHead(3 * projection_channels, hidden_channels, output_channels)

    def forward(
        self,
        encoded: EncoderOutput,
        *,
        residue_graph_index: Tensor,
        local_graph_index: Tensor,
        unit_graph_index: Tensor,
        graph_count: int,
        residue_relay: Tensor | None = None,
        local_relay: Tensor | None = None,
        unit_relay: Tensor | None = None,
    ) -> Tensor:
        residue_pool = fixed_graph_mean(encoded.residues, residue_graph_index, graph_count, residue_relay)
        local_pool = fixed_graph_mean(encoded.local_elements, local_graph_index, graph_count, local_relay)
        unit_pool = fixed_graph_mean(encoded.structural_units, unit_graph_index, graph_count, unit_relay)
        invariant = torch.cat(
            (
                self.residue_projection(residue_pool)[0],
                self.local_projection(local_pool)[0],
                self.unit_projection(unit_pool)[0],
            ),
            dim=-1,
        )
        return self.head(invariant)


class RelationRouteReadout(nn.Module):
    """Shared relation readout anchored to return exactly zero for zero messages."""

    def __init__(
        self,
        feature_dimensions: ChannelDimensions,
        *,
        projection_channels: int,
        hidden_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.feature_dimensions = feature_dimensions
        self.projection = GeometricVectorPerceptron(
            feature_dimensions, ChannelDimensions(projection_channels, 0)
        )
        self.head = _ScalarHead(projection_channels, hidden_channels, output_channels)

    def _zero_pair(self, reference: ScalarVector) -> ScalarVector:
        scalar, vector = reference
        return torch.zeros_like(scalar), torch.zeros_like(vector)

    def forward(self, relation_messages: ScalarVector, unit_graph_index: Tensor, graph_count: int) -> Tensor:
        pooled = fixed_graph_mean(relation_messages, unit_graph_index, graph_count)
        zero_pair = self._zero_pair(pooled)
        projected = self.projection(pooled)[0] - self.projection(zero_pair)[0]
        return self.head(projected) - self.head(torch.zeros_like(projected))


@dataclass(frozen=True, slots=True)
class ReadoutOutput:
    direct: Tensor
    route_contact: Tensor
    route_covalent: Tensor
    route_combined: Tensor


class MultiscaleReadout(nn.Module):
    """Expose one Direct CV and relation-specific Route CVs with a shared head."""

    def __init__(
        self,
        feature_dimensions: ChannelDimensions,
        *,
        projection_channels: int,
        hidden_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.direct = DirectReadout(
            feature_dimensions,
            projection_channels=projection_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
        )
        self.route = RelationRouteReadout(
            feature_dimensions,
            projection_channels=projection_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
        )

    def forward(
        self,
        encoded: EncoderOutput,
        *,
        residue_graph_index: Tensor,
        local_graph_index: Tensor,
        unit_graph_index: Tensor,
        graph_count: int,
        residue_relay: Tensor | None = None,
        local_relay: Tensor | None = None,
        unit_relay: Tensor | None = None,
    ) -> ReadoutOutput:
        direct = self.direct(
            encoded,
            residue_graph_index=residue_graph_index,
            local_graph_index=local_graph_index,
            unit_graph_index=unit_graph_index,
            graph_count=graph_count,
            residue_relay=residue_relay,
            local_relay=local_relay,
            unit_relay=unit_relay,
        )
        return ReadoutOutput(
            direct=direct,
            route_contact=self.route(encoded.unit_contact_messages, unit_graph_index, graph_count),
            route_covalent=self.route(encoded.unit_covalent_messages, unit_graph_index, graph_count),
            route_combined=self.route(encoded.unit_combined_messages, unit_graph_index, graph_count),
        )

