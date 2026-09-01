"""Relation-aware equivariant message passing with intervention-safe aggregation."""

from __future__ import annotations

from torch import Tensor, nn

from .aggregation import fixed_edge_mean, pair_add, pair_concat, pair_index
from .graph import DirectedEdges
from .gvp import ChannelDimensions, GeometricVectorPerceptron, ScalarVector


class RelationMessageLayer(nn.Module):
    """Encode edges, aggregate a named relation, and update nodes separately."""

    def __init__(self, node_dimensions: ChannelDimensions, edge_dimensions: ChannelDimensions) -> None:
        super().__init__()
        message_input = ChannelDimensions(
            scalars=2 * node_dimensions.scalars + edge_dimensions.scalars,
            vectors=2 * node_dimensions.vectors + edge_dimensions.vectors,
        )
        self.node_dimensions = node_dimensions
        self.edge_dimensions = edge_dimensions
        self.message_encoder = GeometricVectorPerceptron(message_input, node_dimensions)
        self.node_update = GeometricVectorPerceptron(
            ChannelDimensions(2 * node_dimensions.scalars, 2 * node_dimensions.vectors),
            node_dimensions,
            scalar_activation=None,
        )

    def edge_messages(self, nodes: ScalarVector, edges: DirectedEdges) -> ScalarVector:
        source = pair_index(nodes, edges.source)
        target = pair_index(nodes, edges.target)
        return self.message_encoder(pair_concat(source, target, edges.features))

    def relation_aggregate(
        self,
        nodes: ScalarVector,
        edges: DirectedEdges,
        gate: Tensor | None = None,
    ) -> ScalarVector:
        messages = self.edge_messages(nodes, edges)
        return fixed_edge_mean(messages, edges.target, edges.baseline_weight, edges.node_count, gate)

    def update_from_aggregate(self, nodes: ScalarVector, aggregate: ScalarVector) -> ScalarVector:
        delta = self.node_update(pair_concat(nodes, aggregate))
        return pair_add(nodes, delta)

    def forward(
        self,
        nodes: ScalarVector,
        edges: DirectedEdges,
        gate: Tensor | None = None,
    ) -> tuple[ScalarVector, ScalarVector]:
        aggregate = self.relation_aggregate(nodes, edges, gate)
        return self.update_from_aggregate(nodes, aggregate), aggregate

