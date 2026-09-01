import numpy as np
import pytest

from allogate.config.hashing import stable_digest
from allogate.evaluation import (
    GateEvaluationRequest,
    GateEvaluationService,
    HessianBlockSpec,
)
from allogate.gates import EntityLevel, EntityRef, GateRegistry, GateSpec


def registry() -> GateRegistry:
    residue_a = EntityRef("example", EntityLevel.RESIDUE, "residue-0000")
    residue_b = EntityRef("example", EntityLevel.RESIDUE, "residue-0001")
    local = EntityRef("example", EntityLevel.LOCAL_ELEMENT, "local-0000")
    contact = EntityRef("example", EntityLevel.CONTACT, "contact-0000-0001")
    return GateRegistry.from_specs(
        "example",
        (
            GateSpec.state(residue_a, aliases=("a",)),
            GateSpec.relay(local, aliases=("b",)),
            GateSpec.contact(residue_a, residue_b, provenance_node=contact.uid, aliases=("c",)),
        ),
    )


class PolynomialResponse:
    def __init__(self) -> None:
        self.calls = 0
        self.shift = 0.0

    def __call__(self, gates):
        import torch

        self.calls += 1
        a, b, c = gates
        return torch.stack((a * a + 3.0 * a * b + torch.sin(c), b * c + 0.5 * c * c)) + self.shift


def service(response=None) -> GateEvaluationService:
    return GateEvaluationService(
        registry(),
        response or PolynomialResponse(),
        model_digest=stable_digest({"model": "polynomial-test-v1"}),
        target_digest=stable_digest({"target": "fixed-test-v1"}),
    )


def test_q0_is_captured_once_and_reused_for_finite_interventions() -> None:
    response = PolynomialResponse()
    evaluator = service(response)
    assert response.calls == 1
    q0 = evaluator.no_intervention.q0.copy()
    result = evaluator.evaluate(
        GateEvaluationRequest(identity_digest=evaluator.identity.digest, overrides=(("a", 0.0),))
    )
    assert response.calls == 2
    np.testing.assert_array_equal(result.q0, q0)
    assert result.reference_digest == evaluator.no_intervention.digest
    assert not result.all_one
    assert result.intervention_l1 == 1.0


def test_jacobian_and_exact_hessian_follow_canonical_uid_order() -> None:
    evaluator = service()
    result = evaluator.evaluate(
        GateEvaluationRequest(
            identity_digest=evaluator.identity.digest,
            jacobian_uids=("c", "a", "b"),
            hessian_blocks=(
                HessianBlockSpec("first-pair", ("b", "a")),
                HessianBlockSpec("second-pair", ("c", "b")),
            ),
        )
    )
    assert result.jacobian_uids == tuple(entry.uid for entry in evaluator.registry.layout.entries)
    np.testing.assert_allclose(
        result.jacobian,
        np.asarray([[5.0, 3.0, np.cos(1.0)], [0.0, 1.0, 2.0]]),
        atol=1e-12,
    )
    first, second = result.hessian_blocks
    np.testing.assert_allclose(first.values[0], [[2.0, 3.0], [3.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(second.values[1], [[0.0, 1.0], [1.0, 1.0]], atol=1e-12)
    assert first.maximum_asymmetry <= 1e-12
    assert second.maximum_asymmetry <= 1e-12


def test_jacobian_agrees_with_centered_finite_difference() -> None:
    evaluator = service()
    point = {"a": 0.7, "b": 0.6, "c": 0.8}
    analytic = evaluator.evaluate(
        GateEvaluationRequest(
            identity_digest=evaluator.identity.digest,
            overrides=tuple(point.items()),
            jacobian_uids=("a", "b", "c"),
        )
    )
    epsilon = 1.0e-5
    columns = []
    for uid in analytic.jacobian_uids:
        alias = evaluator.registry.resolve(uid).aliases[0]
        plus = dict(point)
        minus = dict(point)
        plus[alias] += epsilon
        minus[alias] -= epsilon
        q_plus = evaluator.evaluate(
            GateEvaluationRequest(
                identity_digest=evaluator.identity.digest, overrides=tuple(plus.items())
            )
        ).q
        q_minus = evaluator.evaluate(
            GateEvaluationRequest(
                identity_digest=evaluator.identity.digest, overrides=tuple(minus.items())
            )
        ).q
        columns.append((q_plus - q_minus) / (2.0 * epsilon))
    numerical = np.stack(columns, axis=1)
    np.testing.assert_allclose(analytic.jacobian, numerical, rtol=1e-8, atol=1e-8)


def test_identity_mismatch_and_alias_duplicates_fail_closed() -> None:
    evaluator = service()
    with pytest.raises(ValueError, match="identity"):
        evaluator.evaluate(GateEvaluationRequest(identity_digest="0" * 64))
    uid = evaluator.registry.resolve("a").uid
    with pytest.raises(ValueError, match="aliases"):
        evaluator.evaluate(
            GateEvaluationRequest(
                identity_digest=evaluator.identity.digest,
                overrides=(("a", 0.5), (uid, 0.5)),
            )
        )


def test_replay_detects_model_drift_without_replacing_q0() -> None:
    response = PolynomialResponse()
    evaluator = service(response)
    digest = evaluator.no_intervention.digest
    response.shift = 0.01
    with pytest.raises(RuntimeError, match="immutable q0"):
        evaluator.replay_no_intervention()
    assert evaluator.no_intervention.digest == digest
