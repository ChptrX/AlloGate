"""Composition root for the public trainable AlloGate model."""

from __future__ import annotations

from torch import Tensor, nn

import torch

from allogate.config.schemas import ExperimentConfig, MethodConfig

from .graph import FeatureBaselines, GateControls, MultiscaleGraph
from .gvp import ChannelDimensions
from .multiscale_encoder import MultiscaleEncoder
from .readout import MultiscaleReadout, ReadoutOutput


class AlloGateGeometryModel(nn.Module):
    """Combine the protein-independent encoder and auditable CV readouts."""

    def __init__(self, encoder: MultiscaleEncoder, readout: MultiscaleReadout) -> None:
        super().__init__()
        if (
            encoder.hidden_dimensions
            != readout.direct.residue_projection.input_dimensions
        ):
            raise ValueError("encoder and readout feature dimensions do not match")
        self.encoder = encoder
        self.readout = readout

    @classmethod
    def from_config(
        cls,
        config: MethodConfig,
        *,
        seed: int | None = None,
    ) -> "AlloGateGeometryModel":
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            encoder_config = config.encoder
            hidden = ChannelDimensions(
                encoder_config.hidden_scalar_channels,
                encoder_config.hidden_vector_channels,
            )
            encoder = MultiscaleEncoder(
                ChannelDimensions(
                    encoder_config.residue_input_scalar_channels,
                    encoder_config.residue_input_vector_channels,
                ),
                ChannelDimensions(
                    encoder_config.edge_scalar_channels,
                    encoder_config.edge_vector_channels,
                ),
                hidden,
                layers_per_scale=encoder_config.message_layers_per_scale,
            )
            readout = MultiscaleReadout(
                hidden,
                projection_channels=config.readout.projection_channels,
                hidden_channels=config.readout.hidden_channels,
                output_channels=config.collective_variable_dimension,
            )
            return cls(encoder, readout)

    @classmethod
    def from_experiment_config(
        cls, config: ExperimentConfig
    ) -> "AlloGateGeometryModel":
        return cls.from_config(config.method, seed=config.optimization.seed)

    def forward(
        self,
        graph: MultiscaleGraph,
        *,
        residue_graph_index: Tensor,
        local_unit_graph_index: Tensor,
        interface_graph_index: Tensor,
        global_unit_graph_index: Tensor,
        graph_count: int,
        gates: GateControls | None = None,
        baselines: FeatureBaselines | None = None,
        residue_relay: Tensor | None = None,
        local_unit_relay: Tensor | None = None,
        interface_relay: Tensor | None = None,
        global_unit_relay: Tensor | None = None,
    ) -> ReadoutOutput:
        encoded = self.encoder(graph, gates, baselines)
        return self.readout(
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
