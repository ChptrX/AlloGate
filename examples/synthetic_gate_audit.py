"""Run a complete AlloGate audit on deterministic synthetic CV coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from allogate.artifacts import ContentAddressedStore
from allogate.config.hashing import stable_digest
from allogate.evaluation import (
    DoseSchedule,
    GateEvaluationRequest,
    GateEvaluationService,
    HessianBlockSpec,
)
from allogate.gates import EntityLevel, EntityRef, GateRegistry, GateSpec
from allogate.kinetics import KineticReference


def _registry() -> GateRegistry:
    residue_a = EntityRef("synthetic", EntityLevel.RESIDUE, "chain-a-residue-0001")
    residue_b = EntityRef("synthetic", EntityLevel.RESIDUE, "chain-a-residue-0002")
    local = EntityRef("synthetic", EntityLevel.LOCAL_ELEMENT, "local-0001")
    contact = EntityRef("synthetic", EntityLevel.CONTACT, "contact-0001-0002")
    return GateRegistry.from_specs(
        "synthetic",
        (
            GateSpec.state(residue_a, aliases=("state",)),
            GateSpec.relay(local, aliases=("relay",)),
            GateSpec.contact(
                residue_a,
                residue_b,
                provenance_node=contact.uid,
                aliases=("contact",),
            ),
        ),
    )


def _kinetic_reference() -> KineticReference:
    return KineticReference(
        centers=np.asarray([[-1.0], [0.0], [1.0]]),
        target_values=np.asarray([0.0, 0.5, 1.0]),
        cv_mean=np.zeros(1),
        cv_whitener=np.eye(1),
        bandwidth=0.45,
        lag_time=1.0,
        lag_unit="frame",
    )


def run_example(artifact_root: str | Path | None = None) -> dict[str, Any]:
    registry = _registry()
    reference = _kinetic_reference()
    base_cv = torch.tensor([[-0.8], [0.0], [0.8]], dtype=torch.float64)
    gate_to_cv = torch.tensor(
        [[0.30, -0.10, 0.05], [-0.20, 0.25, 0.10], [0.05, -0.15, 0.35]],
        dtype=torch.float64,
    )

    def response(gates: torch.Tensor) -> torch.Tensor:
        displacement = gate_to_cv @ (gates - 1.0)
        return reference.evaluate_torch(base_cv + displacement[:, None])

    model_digest = stable_digest(
        {
            "schema": "allogate.synthetic_response.v1",
            "base_cv": base_cv.tolist(),
            "gate_to_cv": gate_to_cv.tolist(),
        }
    )
    evaluator = GateEvaluationService(
        registry,
        response,
        model_digest=model_digest,
        target_digest=reference.digest,
    )
    gate_uids = tuple(entry.uid for entry in registry.layout.entries)
    differential = evaluator.evaluate(
        GateEvaluationRequest(
            identity_digest=evaluator.identity.digest,
            jacobian_uids=gate_uids,
            hessian_blocks=(HessianBlockSpec("all-gates", gate_uids),),
        )
    )
    relay_uid = registry.resolve("relay").uid
    dose_results = [
        evaluator.evaluate(request)
        for request in DoseSchedule((relay_uid,), (1.0, 0.5, 0.0)).requests(
            evaluator.identity.digest
        )
    ]
    summary: dict[str, Any] = {
        "schema_version": "allogate.synthetic_example.v1",
        "evaluation_identity": evaluator.identity.digest,
        "reference_digest": evaluator.no_intervention.digest,
        "gate_uids": list(gate_uids),
        "q0": differential.q0.tolist(),
        "jacobian": differential.jacobian.tolist(),
        "hessian_maximum_asymmetry": differential.hessian_blocks[0].maximum_asymmetry,
        "relay_dose_response": [
            {"dose": float(result.gate_values[registry.layout.index(relay_uid)]), "q": result.q.tolist()}
            for result in dose_results
        ],
    }
    if artifact_root is not None:
        store = ContentAddressedStore(artifact_root)
        artifact = store.put_json(summary, logical_type="synthetic-gate-audit")
        store.bind("examples/synthetic-gate-audit", artifact)
        summary["artifact_digest"] = artifact.digest
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="optional directory for a content-addressed result",
    )
    arguments = parser.parse_args()
    print(json.dumps(run_example(arguments.artifact_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
