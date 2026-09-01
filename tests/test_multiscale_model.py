import pytest


torch = pytest.importorskip("torch")

from allogate.models.aggregation import fixed_edge_mean  # noqa: E402
from allogate.models.graph import (  # noqa: E402
    DirectedEdges,
    FeatureBaselines,
    GateControls,
    MultiscaleGraph,
    ParentIndex,
)
from allogate.models.gvp import ChannelDimensions  # noqa: E402
from allogate.models.model import AlloGateGeometryModel  # noqa: E402
from allogate.models.multiscale_encoder import MultiscaleEncoder  # noqa: E402
from allogate.models.readout import MultiscaleReadout  # noqa: E402


DTYPE = torch.double
INPUT_DIMS = ChannelDimensions(3, 2)
EDGE_DIMS = ChannelDimensions(2, 1)
HIDDEN_DIMS = ChannelDimensions(5, 3)


def edges(node_count: int, directed_pairs: list[tuple[int, int]], *, seed: int) -> DirectedEdges:
    generator = torch.Generator().manual_seed(seed)
    source = torch.tensor([pair[0] for pair in directed_pairs], dtype=torch.long)
    target = torch.tensor([pair[1] for pair in directed_pairs], dtype=torch.long)
    count = len(directed_pairs)
    scalar = torch.randn(count, EDGE_DIMS.scalars, generator=generator, dtype=DTYPE)
    vector = torch.randn(count, EDGE_DIMS.vectors, 3, generator=generator, dtype=DTYPE)
    return DirectedEdges(source, target, (scalar, vector), torch.ones(count, dtype=DTYPE), node_count)


def synthetic_graph() -> tuple[MultiscaleGraph, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(3)
    residue_count = 10
    residue_features = (
        torch.randn(residue_count, INPUT_DIMS.scalars, generator=generator, dtype=DTYPE),
        torch.randn(residue_count, INPUT_DIMS.vectors, 3, generator=generator, dtype=DTYPE),
    )
    residue_pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (4, 5), (5, 4), (7, 8), (8, 7)]
    local_pairs = [(0, 1), (1, 0), (2, 3), (3, 2), (3, 4), (4, 3)]
    unit_pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
    graph = MultiscaleGraph(
        residue_features=residue_features,
        residue_edges=edges(residue_count, residue_pairs, seed=4),
        residue_to_local=ParentIndex(torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), 5),
        local_edges=edges(5, local_pairs, seed=5),
        local_to_unit=ParentIndex(torch.tensor([0, 1, 2, 2, 3]), 4),
        unit_contact_edges=edges(4, unit_pairs, seed=6),
        unit_covalent_edges=edges(4, unit_pairs, seed=7),
    )
    graph_indices = (
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        torch.tensor([0, 0, 1, 1, 1]),
        torch.tensor([0, 0, 1, 1]),
    )
    return graph, graph_indices


def model() -> AlloGateGeometryModel:
    encoder = MultiscaleEncoder(INPUT_DIMS, EDGE_DIMS, HIDDEN_DIMS)
    readout = MultiscaleReadout(
        HIDDEN_DIMS,
        projection_channels=4,
        hidden_channels=6,
        output_channels=2,
    )
    return AlloGateGeometryModel(encoder, readout).double()


def rotate_graph(graph: MultiscaleGraph, rotation: torch.Tensor) -> MultiscaleGraph:
    def rotate_edges(value: DirectedEdges) -> DirectedEdges:
        scalar, vector = value.features
        return DirectedEdges(
            value.source,
            value.target,
            (scalar, vector @ rotation.T),
            value.baseline_weight,
            value.node_count,
        )

    residue_scalar, residue_vector = graph.residue_features
    return MultiscaleGraph(
        residue_features=(residue_scalar, residue_vector @ rotation.T),
        residue_edges=rotate_edges(graph.residue_edges),
        residue_to_local=graph.residue_to_local,
        local_edges=rotate_edges(graph.local_edges),
        local_to_unit=graph.local_to_unit,
        unit_contact_edges=rotate_edges(graph.unit_contact_edges),
        unit_covalent_edges=rotate_edges(graph.unit_covalent_edges),
    )


def call(model_instance: AlloGateGeometryModel, graph: MultiscaleGraph, indices, gates=None, baselines=None):
    residue_index, local_index, unit_index = indices
    return model_instance(
        graph,
        residue_graph_index=residue_index,
        local_graph_index=local_index,
        unit_graph_index=unit_index,
        graph_count=2,
        gates=gates,
        baselines=baselines,
    )


def test_fixed_denominator_does_not_renormalize_after_gate() -> None:
    messages = (torch.tensor([[2.0], [4.0]]), torch.zeros(2, 0, 3))
    target = torch.tensor([0, 0])
    baseline = torch.ones(2)
    half_closed = fixed_edge_mean(messages, target, baseline, 1, torch.tensor([1.0, 0.0]))
    all_closed = fixed_edge_mean(messages, target, baseline, 1, torch.zeros(2))
    torch.testing.assert_close(half_closed[0], torch.tensor([[1.0]]))
    torch.testing.assert_close(all_closed[0], torch.zeros(1, 1))


@pytest.mark.parametrize("total_weight", [0.2, 0.7])
def test_fixed_denominator_preserves_subunit_baseline_weight_sums(total_weight: float) -> None:
    messages = (torch.tensor([[2.0], [4.0]]), torch.zeros(2, 0, 3))
    target = torch.tensor([0, 0])
    baseline = torch.tensor([0.25, 0.75]) * total_weight
    all_open = fixed_edge_mean(messages, target, baseline, 2)
    half_closed = fixed_edge_mean(messages, target, baseline, 2, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(all_open[0], torch.tensor([[3.5], [0.0]]))
    torch.testing.assert_close(half_closed[0], torch.tensor([[0.5], [0.0]]))
    assert torch.isfinite(all_open[0]).all()


def test_fixed_denominator_rejects_invalid_baseline_weights() -> None:
    messages = (torch.tensor([[2.0]]), torch.zeros(1, 0, 3))
    target = torch.tensor([0])
    for baseline in (torch.tensor([-0.1]), torch.tensor([float("nan")])):
        with pytest.raises(ValueError, match="finite and non-negative"):
            fixed_edge_mean(messages, target, baseline, 1)


def test_model_supports_variable_graph_sizes_and_is_rotation_invariant() -> None:
    graph, indices = synthetic_graph()
    model_instance = model()
    output = call(model_instance, graph, indices)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=DTYPE))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    rotated = call(model_instance, rotate_graph(graph, rotation), indices)
    assert output.direct.shape == (2, 2)
    for field in ("direct", "route_contact", "route_covalent", "route_combined"):
        torch.testing.assert_close(getattr(rotated, field), getattr(output, field), atol=1.0e-10, rtol=1.0e-10)


def test_closed_contact_relation_has_exactly_zero_route() -> None:
    graph, indices = synthetic_graph()
    model_instance = model()
    gate = torch.zeros(len(graph.unit_contact_edges.source), dtype=DTYPE)
    output = call(model_instance, graph, indices, GateControls(unit_contact_edges=gate))
    torch.testing.assert_close(output.route_contact, torch.zeros_like(output.route_contact), atol=0.0, rtol=0.0)
    assert torch.count_nonzero(output.route_covalent) > 0


def test_contact_gate_supports_second_derivatives() -> None:
    graph, indices = synthetic_graph()
    model_instance = model()
    gate = torch.full(
        (len(graph.unit_contact_edges.source),),
        0.7,
        dtype=DTYPE,
        requires_grad=True,
    )
    output = call(model_instance, graph, indices, GateControls(unit_contact_edges=gate))
    first = torch.autograd.grad(output.route_contact.square().sum(), gate, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), gate)[0]
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_unit_state_replacement_does_not_change_relation_routes() -> None:
    graph, indices = synthetic_graph()
    model_instance = model()
    reference = call(model_instance, graph, indices)
    controls = GateControls(unit_state=torch.zeros(4, dtype=DTYPE))
    baselines = FeatureBaselines(unit_scalar=torch.zeros(4, HIDDEN_DIMS.scalars, dtype=DTYPE))
    intervened = call(model_instance, graph, indices, controls, baselines)
    for field in ("route_contact", "route_covalent", "route_combined"):
        torch.testing.assert_close(getattr(intervened, field), getattr(reference, field))
    assert not torch.allclose(intervened.direct, reference.direct)
