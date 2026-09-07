"""Invariant Direct and zero-preserving interface Route readouts."""

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
    """Read pooled states from all four public hierarchy levels."""

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
        self.residue_projection = GeometricVectorPerceptron(
            feature_dimensions, projection_dimensions
        )
        self.local_unit_projection = GeometricVectorPerceptron(
            feature_dimensions, projection_dimensions
        )
        self.interface_projection = GeometricVectorPerceptron(
            feature_dimensions, projection_dimensions
        )
        self.global_unit_projection = GeometricVectorPerceptron(
            feature_dimensions, projection_dimensions
        )
        self.head = _ScalarHead(
            4 * projection_channels, hidden_channels, output_channels
        )

    def forward(
        self,
        encoded: EncoderOutput,
        *,
        residue_graph_index: Tensor,
        local_unit_graph_index: Tensor,
        interface_graph_index: Tensor,
        global_unit_graph_index: Tensor,
        graph_count: int,
        residue_relay: Tensor | None = None,
        local_unit_relay: Tensor | None = None,
        interface_relay: Tensor | None = None,
        global_unit_relay: Tensor | None = None,
    ) -> Tensor:
        residue_pool = fixed_graph_mean(
            encoded.residues, residue_graph_index, graph_count, residue_relay
        )
        local_pool = fixed_graph_mean(
            encoded.local_structural_units,
            local_unit_graph_index,
            graph_count,
            local_unit_relay,
        )
        interface_pool = fixed_graph_mean(
            encoded.interfaces,
            interface_graph_index,
            graph_count,
            interface_relay,
        )
        global_pool = fixed_graph_mean(
            encoded.global_structural_units,
            global_unit_graph_index,
            graph_count,
            global_unit_relay,
        )
        invariant = torch.cat(
            (
                self.residue_projection(residue_pool)[0],
                self.local_unit_projection(local_pool)[0],
                self.interface_projection(interface_pool)[0],
                self.global_unit_projection(global_pool)[0],
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
        self.head = _ScalarHead(
            projection_channels, hidden_channels, output_channels
        )

    @staticmethod
    def _zero_pair(reference: ScalarVector) -> ScalarVector:
        scalar, vector = reference
        return torch.zeros_like(scalar), torch.zeros_like(vector)

    def forward(
        self,
        relation_messages: ScalarVector,
        global_unit_graph_index: Tensor,
        graph_count: int,
    ) -> Tensor:
        pooled = fixed_graph_mean(
            relation_messages, global_unit_graph_index, graph_count
        )
        zero_pair = self._zero_pair(pooled)
        projected = self.projection(pooled)[0] - self.projection(zero_pair)[0]
        return self.head(projected) - self.head(torch.zeros_like(projected))


@dataclass(frozen=True, slots=True)
class ReadoutOutput:
    direct: Tensor
    route_interface: Tensor
    route_covalent: Tensor
    route_combined: Tensor


class MultiscaleReadout(nn.Module):
    """Expose Direct and relation-specific Route CVs with one shared route head."""

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
        local_unit_graph_index: Tensor,
        interface_graph_index: Tensor,
        global_unit_graph_index: Tensor,
        graph_count: int,
        residue_relay: Tensor | None = None,
        local_unit_relay: Tensor | None = None,
        interface_relay: Tensor | None = None,
        global_unit_relay: Tensor | None = None,
    ) -> ReadoutOutput:
        direct = self.direct(
            encoded,
            residue_graph_index=residue_graph_index,
            local_unit_graph_index=local_unit_graph_index,
            interface_graph_index=interface_graph_index,
            global_unit_graph_index=global_unit_graph_index,
            graph_count=graph_count,
            residue_relay=residue_relay,
            local_unit_relay=local_unit_relay,
            interface_relay=interface_relay,
            global_unit_relay=global_unit_relay,
        )
        return ReadoutOutput(
            direct=direct,
            route_interface=self.route(
                encoded.global_interface_messages,
                global_unit_graph_index,
                graph_count,
            ),
            route_covalent=self.route(
                encoded.global_covalent_messages,
                global_unit_graph_index,
                graph_count,
            ),
            route_combined=self.route(
                encoded.global_combined_messages,
                global_unit_graph_index,
                graph_count,
            ),
        )
