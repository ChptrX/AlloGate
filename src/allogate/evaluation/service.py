"""One identity-checked service for zero-, first-, and second-order Gate evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from allogate.gates.registry import GateRegistry

from .contracts import EvaluationIdentity, GateEvaluationRequest
from .results import GateEvaluationResult, HessianBlockResult, NoInterventionReference


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Gate evaluation requires the optional PyTorch dependency") from error
    return torch


class GateEvaluationService:
    """Differentiate a fixed response with respect to canonical Gate values.

    ``response`` accepts the full one-dimensional Gate tensor in Registry order
    and returns one scalar or a one-dimensional tensor of target values.
    """

    def __init__(
        self,
        registry: GateRegistry,
        response: Callable[[Any], Any],
        *,
        model_digest: str,
        target_digest: str,
        device: Any = None,
        baseline_tolerance: float = 1.0e-10,
    ) -> None:
        torch = _torch()
        if baseline_tolerance < 0.0 or not np.isfinite(baseline_tolerance):
            raise ValueError("baseline_tolerance must be finite and non-negative")
        self.registry = registry
        self.response = response
        self.device = torch.device("cpu" if device is None else device)
        self.baseline_tolerance = float(baseline_tolerance)
        self.identity = EvaluationIdentity(
            registry_digest=registry.digest,
            layout_digest=registry.layout.digest,
            model_digest=model_digest,
            target_digest=target_digest,
        )
        self._gate_uids = tuple(entry.uid for entry in registry.layout.entries)
        all_one = torch.ones(len(self._gate_uids), dtype=torch.float64, device=self.device)
        with torch.no_grad():
            q0 = self._call_response(all_one)
        self._reference = NoInterventionReference(
            identity_digest=self.identity.digest,
            gate_uids=self._gate_uids,
            q0=q0.detach().cpu().numpy(),
        )

    @property
    def no_intervention(self) -> NoInterventionReference:
        return self._reference

    def _call_response(self, gate_values: Any) -> Any:
        torch = _torch()
        output = self.response(gate_values)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Gate response must return a PyTorch tensor")
        flattened = output.reshape(1) if output.ndim == 0 else output.reshape(-1)
        if not torch.isfinite(flattened).all():
            raise FloatingPointError("Gate response contains non-finite values")
        return flattened.to(dtype=torch.float64)

    def replay_no_intervention(self) -> float:
        """Check that the current model still reproduces the captured q0."""

        torch = _torch()
        gates = torch.ones(len(self._gate_uids), dtype=torch.float64, device=self.device)
        with torch.no_grad():
            observed = self._call_response(gates).detach().cpu().numpy()
        if observed.shape != self._reference.q0.shape:
            raise RuntimeError("the Gate response output shape changed after q0 capture")
        difference = float(np.max(np.abs(observed - self._reference.q0), initial=0.0))
        if difference > self.baseline_tolerance:
            raise RuntimeError(f"all-one response changed immutable q0 by {difference:g}")
        return difference

    def _canonical_uids(self, identifiers: tuple[str, ...]) -> tuple[str, ...]:
        resolved = [self.registry.resolve(identifier).uid for identifier in identifiers]
        if len(set(resolved)) != len(resolved):
            raise ValueError("Gate identifier list contains aliases of the same Gate")
        return tuple(sorted(resolved, key=self.registry.layout.index))

    def evaluate(self, request: GateEvaluationRequest) -> GateEvaluationResult:
        torch = _torch()
        if request.identity_digest != self.identity.digest:
            raise ValueError("evaluation request identity does not match this service")

        full = torch.ones(len(self._gate_uids), dtype=torch.float64, device=self.device)
        resolved_overrides: dict[str, float] = {}
        for identifier, value in request.overrides:
            uid = self.registry.resolve(identifier).uid
            if uid in resolved_overrides:
                raise ValueError("Gate overrides contain aliases of the same Gate")
            resolved_overrides[uid] = float(value)
            full[self.registry.layout.index(uid)] = float(value)

        jacobian_uids = self._canonical_uids(request.jacobian_uids)
        canonical_blocks = tuple(
            (block.name, self._canonical_uids(block.gate_uids)) for block in request.hessian_blocks
        )
        derivative_uid_set = set(jacobian_uids)
        for _, gate_uids in canonical_blocks:
            derivative_uid_set.update(gate_uids)
        derivative_uids = tuple(sorted(derivative_uid_set, key=self.registry.layout.index))
        derivative_indices = [self.registry.layout.index(uid) for uid in derivative_uids]

        if derivative_uids:
            index_tensor = torch.as_tensor(derivative_indices, dtype=torch.int64, device=self.device)
            local = full.index_select(0, index_tensor).detach().clone().requires_grad_(True)
            gates = full.index_copy(0, index_tensor, local)
        else:
            local = None
            gates = full
        q = self._call_response(gates)
        q_numpy = q.detach().cpu().numpy()
        if q_numpy.shape != self._reference.q0.shape:
            raise RuntimeError("the Gate response output shape differs from q0")

        all_one = bool(torch.equal(full, torch.ones_like(full)))
        if all_one:
            difference = float(np.max(np.abs(q_numpy - self._reference.q0), initial=0.0))
            if difference > self.baseline_tolerance:
                raise RuntimeError(f"all-one evaluation changed immutable q0 by {difference:g}")

        derivative_position = {uid: position for position, uid in enumerate(derivative_uids)}
        need_hessian = bool(canonical_blocks)
        output_gradients: list[Any] = []
        if local is not None:
            for output_index in range(len(q)):
                if q[output_index].requires_grad:
                    gradient = torch.autograd.grad(
                        q[output_index],
                        local,
                        create_graph=need_hessian,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                else:
                    gradient = None
                output_gradients.append(torch.zeros_like(local) if gradient is None else gradient)

        if jacobian_uids:
            jacobian_positions = [derivative_position[uid] for uid in jacobian_uids]
            jacobian_tensor = torch.stack(output_gradients)[:, jacobian_positions]
            jacobian = jacobian_tensor.detach().cpu().numpy()
        else:
            jacobian = np.empty((len(q), 0), dtype=np.float64)

        hessian_results: list[HessianBlockResult] = []
        for name, block_uids in canonical_blocks:
            positions = [derivative_position[uid] for uid in block_uids]
            raw = torch.zeros(
                (len(q), len(positions), len(positions)), dtype=torch.float64, device=self.device
            )
            assert local is not None
            for output_index, gradient in enumerate(output_gradients):
                for row_index, position in enumerate(positions):
                    component = gradient[position]
                    if component.requires_grad:
                        row = torch.autograd.grad(
                            component,
                            local,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                    else:
                        row = None
                    if row is not None:
                        raw[output_index, row_index] = row[positions]
            asymmetry = torch.max(torch.abs(raw - raw.transpose(-1, -2))).item()
            symmetric = 0.5 * (raw + raw.transpose(-1, -2))
            hessian_results.append(
                HessianBlockResult(
                    name=name,
                    gate_uids=block_uids,
                    values=symmetric.detach().cpu().numpy(),
                    maximum_asymmetry=float(asymmetry),
                )
            )

        gate_values = full.detach().cpu().numpy()
        delta = q_numpy - self._reference.q0
        return GateEvaluationResult(
            identity_digest=self.identity.digest,
            reference_digest=self._reference.digest,
            gate_uids=self._gate_uids,
            gate_values=gate_values,
            q=q_numpy,
            q0=self._reference.q0,
            delta_q=delta,
            jacobian_uids=jacobian_uids,
            jacobian=jacobian,
            hessian_blocks=tuple(hessian_results),
            all_one=all_one,
            intervention_l1=float(np.sum(np.abs(1.0 - gate_values))),
        )
