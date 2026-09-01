"""Composition root for the second public whitelist milestone."""

from __future__ import annotations

from torch import Tensor, nn

from .graph import FeatureBaselines, GateControls, MultiscaleGraph
from .multiscale_encoder import MultiscaleEncoder
from .readout import MultiscaleReadout, ReadoutOutput


class AlloGateGeometryModel(nn.Module):
    """Combine the generic multiscale encoder and its auditable readouts."""

    def __init__(self, encoder: MultiscaleEncoder, readout: MultiscaleReadout) -> None:
        super().__init__()
        if encoder.hidden_dimensions != readout.direct.residue_projection.input_dimensions:
            raise ValueError("encoder and readout feature dimensions do not match")
        self.encoder = encoder
        self.readout = readout

    def forward(
        self,
        graph: MultiscaleGraph,
        *,
        residue_graph_index: Tensor,
        local_graph_index: Tensor,
        unit_graph_index: Tensor,
        graph_count: int,
        gates: GateControls | None = None,
        baselines: FeatureBaselines | None = None,
        residue_relay: Tensor | None = None,
        local_relay: Tensor | None = None,
        unit_relay: Tensor | None = None,
    ) -> ReadoutOutput:
        encoded = self.encoder(graph, gates, baselines)
        return self.readout(
            encoded,
            residue_graph_index=residue_graph_index,
            local_graph_index=local_graph_index,
            unit_graph_index=unit_graph_index,
            graph_count=graph_count,
            residue_relay=residue_relay,
            local_relay=local_relay,
            unit_relay=unit_relay,
        )
