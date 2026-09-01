from pathlib import Path

import numpy as np
import pytest

from allogate.kinetics import KineticReference, load_kinetic_reference, save_kinetic_reference


def reference() -> KineticReference:
    return KineticReference(
        centers=np.asarray([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        target_values=np.asarray([0.0, 1.0, 0.5]),
        cv_mean=np.asarray([0.25, -0.5]),
        cv_whitener=np.asarray([[2.0, 0.0], [0.0, 0.5]]),
        bandwidth=0.75,
        lag_time=5.0,
        lag_unit="ns",
    )


def test_reference_is_semantic_and_arrays_are_read_only() -> None:
    first = reference()
    second = reference()
    assert first.digest == second.digest
    assert not first.centers.flags.writeable
    with pytest.raises(ValueError):
        first.centers[0, 0] = 3.0


def test_numpy_and_torch_projection_agree_and_support_double_backward() -> None:
    torch = pytest.importorskip("torch")
    target = reference()
    cv_numpy = np.asarray([[0.1, -0.2], [0.7, 0.3]])
    cv = torch.tensor(cv_numpy, dtype=torch.float64, requires_grad=True)
    observed = target.evaluate_torch(cv)
    np.testing.assert_allclose(observed.detach().numpy(), target.evaluate_numpy(cv_numpy), atol=1e-12)
    first = torch.autograd.grad(observed.sum(), cv, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), cv)[0]
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_reference_round_trip_verifies_semantic_digest(tmp_path: Path) -> None:
    destination = tmp_path / "reference.npz"
    expected = reference()
    save_kinetic_reference(destination, expected)
    observed = load_kinetic_reference(destination)
    assert observed.digest == expected.digest
    np.testing.assert_array_equal(observed.target_values, expected.target_values)


def test_reference_rejects_invalid_scientific_arrays() -> None:
    with pytest.raises(ValueError, match="one value per center"):
        KineticReference(
            centers=np.zeros((2, 1)),
            target_values=np.zeros(1),
            cv_mean=np.zeros(1),
            cv_whitener=np.eye(1),
            bandwidth=1.0,
            lag_time=1.0,
        )
