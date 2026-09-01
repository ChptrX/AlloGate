import numpy as np
import pytest

from allogate.geometry.residue_features import (
    cosine_switch,
    gaussian_radial_basis,
    hybrid_beta_carbon,
    virtual_beta_carbon,
)


BACKBONE = np.array(
    [
        [[-1.2, 0.1, 0.2], [0.0, 0.0, 0.0], [1.1, 0.7, -0.1]],
        [[0.3, 1.0, -0.2], [1.0, 1.2, 0.4], [1.9, 0.8, 0.9]],
    ],
    dtype=np.float64,
)


def rotation_matrix() -> np.ndarray:
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    angle = 0.63
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def test_virtual_beta_carbon_is_rigid_motion_equivariant() -> None:
    rotation = rotation_matrix()
    translation = np.array([3.0, -1.0, 0.25])
    transformed = BACKBONE @ rotation.T + translation
    expected = virtual_beta_carbon(BACKBONE) @ rotation.T + translation
    np.testing.assert_allclose(virtual_beta_carbon(transformed), expected, atol=1.0e-12)


def test_hybrid_beta_carbon_selects_observed_coordinates() -> None:
    observed = np.full((2, 3), 9.0)
    result = hybrid_beta_carbon(BACKBONE, observed, np.array([True, False]))
    np.testing.assert_array_equal(result[0], observed[0])
    np.testing.assert_allclose(result[1], virtual_beta_carbon(BACKBONE)[1])


def test_distance_features_have_expected_boundaries() -> None:
    np.testing.assert_allclose(cosine_switch([7.0, 8.0, 9.0, 10.0, 11.0], start=8.0, stop=10.0), [1, 1, 0.5, 0, 0])
    basis = gaussian_radial_basis([0.0, 2.0], count=3, maximum=2.0)
    assert basis.shape == (2, 3)
    assert basis[0, 0] == pytest.approx(1.0)
    assert basis[1, -1] == pytest.approx(1.0)


def test_backbone_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="shape"):
        virtual_beta_carbon(np.zeros((3, 4)))

