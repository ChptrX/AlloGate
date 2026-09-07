"""Deterministic within-trajectory lag-pair identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .manifests import TrajectoryManifest


@dataclass(frozen=True, order=True, slots=True)
class LaggedPairRecord:
    trajectory_id: str
    time_index: int
    lagged_index: int
    group_id: str
    split: str

    def __post_init__(self) -> None:
        if self.time_index < 0 or self.lagged_index <= self.time_index:
            raise ValueError("a lagged pair must satisfy 0 <= time < lagged")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")


@dataclass(frozen=True, slots=True)
class LaggedPairManifest:
    lag_time_ps: float
    frame_stride: int
    pairs: tuple[LaggedPairRecord, ...]
    schema_version: str = "allogate.lagged_pairs.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "allogate.lagged_pairs.v1":
            raise ValueError(f"unsupported lagged-pair schema: {self.schema_version}")
        if self.lag_time_ps <= 0.0 or self.frame_stride < 1:
            raise ValueError("lag time and frame stride must be positive")
        identities = [
            (item.trajectory_id, item.time_index, item.lagged_index)
            for item in self.pairs
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("lagged-pair identities must be unique")


def build_lagged_pair_manifest(
    trajectories: TrajectoryManifest,
    frame_counts: Mapping[str, int],
    *,
    lag_time_ps: float,
    frame_stride: int = 1,
) -> LaggedPairManifest:
    """Build pairs without crossing trajectory or declared split boundaries."""

    if lag_time_ps <= 0.0 or frame_stride < 1:
        raise ValueError("lag_time_ps and frame_stride must be positive")
    expected = {record.trajectory_id for record in trajectories.records}
    supplied = set(frame_counts)
    if supplied != expected:
        raise ValueError(
            f"frame-count identities differ from the trajectory manifest: "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
        )
    pairs: list[LaggedPairRecord] = []
    for record in trajectories.records:
        frame_count = int(frame_counts[record.trajectory_id])
        if frame_count < 1:
            raise ValueError(f"frame count for {record.trajectory_id} must be positive")
        raw_lag = lag_time_ps / record.time_step_ps
        lag_frames = int(round(raw_lag))
        if lag_frames < 1 or abs(raw_lag - lag_frames) > 1.0e-9:
            raise ValueError(
                f"lag_time_ps is not an integer frame lag for {record.trajectory_id}"
            )
        for time_index in range(0, frame_count - lag_frames, frame_stride):
            pairs.append(
                LaggedPairRecord(
                    trajectory_id=record.trajectory_id,
                    time_index=time_index,
                    lagged_index=time_index + lag_frames,
                    group_id=record.group_id,
                    split=record.split,
                )
            )
    return LaggedPairManifest(
        lag_time_ps=float(lag_time_ps),
        frame_stride=int(frame_stride),
        pairs=tuple(pairs),
    )
