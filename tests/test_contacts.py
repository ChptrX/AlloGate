import numpy as np

from allogate.geometry.contacts import contact_edges, directed_geometry_from_pairs


def rotation_matrix() -> np.ndarray:
    angle = 0.41
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_contact_edges_never_cross_graphs() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.0, 0.0, 0.0]])
    graph_index = np.array([0, 0, 1, 1])
    edges = contact_edges(coordinates, graph_index, switch_on=1.0, switch_off=1.5)
    assert set(zip(edges.source.tolist(), edges.target.tolist())) == {(0, 1), (1, 0), (2, 3), (3, 2)}


def test_directed_geometry_is_rigid_motion_equivariant() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.2, 1.0, 0.7]])
    pairs = np.array([[0, 1], [0, 2]])
    reference = directed_geometry_from_pairs(coordinates, pairs)
    rotation = rotation_matrix()
    moved = directed_geometry_from_pairs(coordinates @ rotation.T + np.array([5.0, -2.0, 1.0]), pairs)
    np.testing.assert_allclose(moved.distance, reference.distance, atol=1.0e-12)
    np.testing.assert_allclose(moved.unit_direction, reference.unit_direction @ rotation.T, atol=1.0e-12)


def test_excluded_pair_is_not_reintroduced() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    edges = contact_edges(
        coordinates,
        np.zeros(3, dtype=np.int64),
        switch_on=1.0,
        switch_off=2.0,
        excluded_pairs=np.array([[0, 1]]),
    )
    assert (0, 1) not in set(zip(edges.source.tolist(), edges.target.tolist()))
    assert (1, 0) not in set(zip(edges.source.tolist(), edges.target.tolist()))

