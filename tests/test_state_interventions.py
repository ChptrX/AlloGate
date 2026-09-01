import pytest


torch = pytest.importorskip("torch")

from allogate.models.interventions import apply_state_replacement  # noqa: E402


def test_state_gate_interpolates_to_frozen_baseline() -> None:
    scalar = torch.tensor([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
    vector = torch.ones(2, 1, 3, requires_grad=True)
    baseline = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    gate = torch.tensor([0.0, 0.5], requires_grad=True)
    output_scalar, output_vector = apply_state_replacement((scalar, vector), gate, baseline)
    torch.testing.assert_close(output_scalar[0], baseline[0])
    torch.testing.assert_close(output_vector[0], torch.zeros_like(output_vector[0]))
    torch.testing.assert_close(output_scalar[1], torch.tensor([4.5, 5.5]))
    output_scalar.sum().backward()
    assert gate.grad is not None and torch.isfinite(gate.grad).all()


def test_state_intervention_requires_a_baseline() -> None:
    with pytest.raises(ValueError, match="requires"):
        apply_state_replacement((torch.ones(2, 1), torch.ones(2, 1, 3)), torch.ones(2), None)

