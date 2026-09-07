"""Differentiable VAMP-2 objective for descriptor-free slow coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from allogate.config.schemas import KineticObjectiveConfig
from allogate.models.readout import ReadoutOutput


def covariance(left: Tensor, right: Tensor) -> Tensor:
    if left.shape != right.shape or left.ndim != 2 or len(left) < 2:
        raise ValueError("covariance inputs must be matching [N, K] tensors with N >= 2")
    return left.T @ right / (left.shape[0] - 1)


def _vamp2_score(time: Tensor, lagged: Tensor, regularization: float) -> Tensor:
    time = time.to(dtype=torch.float64)
    lagged = lagged.to(dtype=torch.float64)
    x = time - time.mean(dim=0)
    y = lagged - lagged.mean(dim=0)
    c00, c01, c11 = covariance(x, x), covariance(x, y), covariance(y, y)
    dimension = c00.shape[0]
    identity = torch.eye(dimension, dtype=c00.dtype, device=c00.device)
    left = 0.5 * (c00 + c00.T) + regularization * identity
    right = 0.5 * (c11 + c11.T) + regularization * identity
    solved_left = torch.cholesky_solve(c01, torch.linalg.cholesky(left))
    solved_right = torch.cholesky_solve(c01.T, torch.linalg.cholesky(right))
    return (solved_left * solved_right.T).sum()


def _canonical_alignment(
    direct_time: Tensor,
    direct_lagged: Tensor,
    route_time: Tensor,
    route_lagged: Tensor,
    regularization: float,
) -> Tensor:
    direct = torch.cat((direct_time, direct_lagged), dim=0).to(torch.float64)
    route = torch.cat((route_time, route_lagged), dim=0).to(torch.float64)
    direct = direct - direct.mean(dim=0)
    route = route - route.mean(dim=0)
    cdd = covariance(direct, direct)
    cdr = covariance(direct, route)
    crr = covariance(route, route)
    dimension = cdd.shape[0]
    identity = torch.eye(dimension, dtype=cdd.dtype, device=cdd.device)
    left = 0.5 * (cdd + cdd.T) + regularization * identity
    right = 0.5 * (crr + crr.T) + regularization * identity
    solved_left = torch.cholesky_solve(cdr, torch.linalg.cholesky(left))
    solved_right = torch.cholesky_solve(cdr.T, torch.linalg.cholesky(right))
    squared = (solved_left * solved_right.T).sum()
    return (1.0 - squared / dimension).clamp(0.0, 1.0)


def _standard_deviation_floor(values: tuple[Tensor, ...], floor: float) -> Tensor:
    penalties = []
    for value in values:
        variance = value.to(torch.float64).var(dim=0, correction=1).clamp_min(0.0)
        standard_deviation = torch.sqrt(variance + 1.0e-12)
        penalties.append(torch.relu(floor - standard_deviation).square())
    return torch.cat(penalties).mean()


def _decorrelation(values: tuple[Tensor, ...]) -> Tensor:
    penalties: list[Tensor] = []
    for value in values:
        value = value.to(torch.float64)
        if value.shape[1] < 2:
            penalties.append(value.new_zeros(()))
            continue
        centered = value - value.mean(dim=0)
        gram = centered.T @ centered
        norms = torch.sqrt(torch.diag(gram).clamp_min(1.0e-24))
        correlation = gram / (norms[:, None] * norms[None, :]).clamp_min(1.0e-12)
        mask = ~torch.eye(
            value.shape[1], dtype=torch.bool, device=value.device
        )
        penalties.append(correlation[mask].square().mean())
    return torch.stack(penalties).mean()


def trajectory_centered_pairs(
    time: Tensor,
    lagged: Tensor,
    trajectory_index: Tensor,
) -> tuple[Tensor, Tensor]:
    if time.shape != lagged.shape or time.ndim != 2:
        raise ValueError("trajectory centering requires matching [N, K] tensors")
    labels = trajectory_index.to(device=time.device, dtype=torch.long).reshape(-1)
    if labels.shape != (len(time),):
        raise ValueError("trajectory_index must contain one label per pair")
    pooled = torch.cat((time, lagged), dim=0)
    pooled_labels = torch.cat((labels, labels), dim=0)
    _, inverse = torch.unique(pooled_labels, sorted=True, return_inverse=True)
    count = int(inverse.max()) + 1
    totals = pooled.new_zeros((count, pooled.shape[1])).index_add_(0, inverse, pooled)
    counts = pooled.new_zeros(count).index_add_(
        0, inverse, pooled.new_ones(len(pooled))
    )
    means = totals / counts.clamp_min(1.0)[:, None]
    centered = pooled - means[inverse]
    return centered[: len(time)], centered[len(time) :]


@dataclass(frozen=True, slots=True)
class SlowCVLossResult:
    loss: Tensor
    interface_route_vamp2: Tensor
    direct_vamp2: Tensor
    alignment: Tensor
    standard_deviation_penalty: Tensor
    decorrelation_penalty: Tensor
    centered_interface_route_vamp2: Tensor
    centered_direct_vamp2: Tensor

    def metrics(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach())
            for name in (
                "loss",
                "interface_route_vamp2",
                "direct_vamp2",
                "alignment",
                "standard_deviation_penalty",
                "decorrelation_penalty",
                "centered_interface_route_vamp2",
                "centered_direct_vamp2",
            )
        }


def _loss_terms(
    direct_time: Tensor,
    direct_lagged: Tensor,
    route_time: Tensor,
    route_lagged: Tensor,
    config: KineticObjectiveConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    route_score = _vamp2_score(
        route_time, route_lagged, config.covariance_regularization
    )
    direct_score = _vamp2_score(
        direct_time, direct_lagged, config.covariance_regularization
    )
    alignment = _canonical_alignment(
        direct_time,
        direct_lagged,
        route_time,
        route_lagged,
        config.covariance_regularization,
    )
    standard_deviation = _standard_deviation_floor(
        (direct_time, direct_lagged, route_time, route_lagged),
        config.standard_deviation_floor,
    )
    decorrelation = _decorrelation(
        (direct_time, direct_lagged, route_time, route_lagged)
    )
    loss = (
        -config.interface_route_weight * route_score
        - config.direct_weight * direct_score
        + config.alignment_weight * alignment
        + config.standard_deviation_weight * standard_deviation
        + config.decorrelation_weight * decorrelation
    )
    return loss, route_score, direct_score, alignment, standard_deviation, decorrelation


def slow_cv_loss(
    time: ReadoutOutput,
    lagged: ReadoutOutput,
    config: KineticObjectiveConfig,
    trajectory_index: Tensor | None = None,
) -> SlowCVLossResult:
    pooled = _loss_terms(
        time.direct,
        lagged.direct,
        time.route_interface,
        lagged.route_interface,
        config,
    )
    centered_route = pooled[1].new_zeros(())
    centered_direct = pooled[2].new_zeros(())
    loss = pooled[0]
    if config.trajectory_centered_weight > 0.0:
        if trajectory_index is None:
            raise ValueError(
                "trajectory_index is required when trajectory_centered_weight is positive"
            )
        direct_time, direct_lagged = trajectory_centered_pairs(
            time.direct, lagged.direct, trajectory_index
        )
        route_time, route_lagged = trajectory_centered_pairs(
            time.route_interface, lagged.route_interface, trajectory_index
        )
        centered = _loss_terms(
            direct_time, direct_lagged, route_time, route_lagged, config
        )
        loss = loss + config.trajectory_centered_weight * centered[0]
        centered_route, centered_direct = centered[1], centered[2]
    return SlowCVLossResult(
        loss=loss,
        interface_route_vamp2=pooled[1],
        direct_vamp2=pooled[2],
        alignment=pooled[3],
        standard_deviation_penalty=pooled[4],
        decorrelation_penalty=pooled[5],
        centered_interface_route_vamp2=centered_route,
        centered_direct_vamp2=centered_direct,
    )
