import pytest


torch = pytest.importorskip("torch")

from allogate.models.gvp import ChannelDimensions, GeometricVectorPerceptron  # noqa: E402


def test_gvp_rotation_contract_and_gradients() -> None:
    torch.manual_seed(7)
    layer = GeometricVectorPerceptron(ChannelDimensions(5, 3), ChannelDimensions(4, 2)).double()
    scalars = torch.randn(8, 5, dtype=torch.double, requires_grad=True)
    vectors = torch.randn(8, 3, 3, dtype=torch.double, requires_grad=True)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.double))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1

    scalar_output, vector_output = layer((scalars, vectors))
    rotated_scalar_output, rotated_vector_output = layer((scalars, vectors @ rotation.T))

    torch.testing.assert_close(rotated_scalar_output, scalar_output, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(rotated_vector_output, vector_output @ rotation.T, atol=1.0e-10, rtol=1.0e-10)

    (scalar_output.square().mean() + vector_output.square().mean()).backward()
    assert scalars.grad is not None and torch.isfinite(scalars.grad).all()
    assert vectors.grad is not None and torch.isfinite(vectors.grad).all()


def test_gvp_supports_scalar_only_input() -> None:
    layer = GeometricVectorPerceptron(ChannelDimensions(3, 0), ChannelDimensions(2, 0))
    scalars = torch.randn(4, 3)
    vectors = torch.empty(4, 0, 3)
    output_scalars, output_vectors = layer((scalars, vectors))
    assert output_scalars.shape == (4, 2)
    assert output_vectors.shape == (4, 0, 3)

