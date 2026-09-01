"""Trajectory manifests with portable, non-sensitive path rules."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable


def _portable_relative_path(raw: str, *, field_name: str) -> str:
    value = raw.strip().replace("\\", "/")
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(raw).is_absolute():
        raise ValueError(f"{field_name} must be relative, not {raw!r}")
    parts = PurePosixPath(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{field_name} must not contain empty, '.' or '..' path components")
    return PurePosixPath(*parts).as_posix()


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    trajectory_id: str
    topology: str
    trajectory: str
    time_step_ps: float
    group_id: str
    split: str

    def __post_init__(self) -> None:
        if not self.trajectory_id or any(char.isspace() for char in self.trajectory_id):
            raise ValueError("trajectory_id must be a non-empty token")
        if not self.group_id or any(char.isspace() for char in self.group_id):
            raise ValueError("group_id must be a non-empty token")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if self.time_step_ps <= 0.0:
            raise ValueError("time_step_ps must be positive")
        object.__setattr__(self, "topology", _portable_relative_path(self.topology, field_name="topology"))
        object.__setattr__(
            self,
            "trajectory",
            _portable_relative_path(self.trajectory, field_name="trajectory"),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryManifest:
    records: tuple[TrajectoryRecord, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("a trajectory manifest must contain at least one record")
        identifiers = [record.trajectory_id for record in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("trajectory_id values must be unique")

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, str]]) -> "TrajectoryManifest":
        required = {
            "trajectory_id",
            "topology",
            "trajectory",
            "time_step_ps",
            "group_id",
            "split",
        }
        records: list[TrajectoryRecord] = []
        for number, row in enumerate(rows, start=2):
            missing = required.difference(row)
            extra = set(row).difference(required)
            if missing or extra:
                raise ValueError(
                    f"manifest row {number} has missing fields {sorted(missing)} "
                    f"and extra fields {sorted(extra)}"
                )
            try:
                time_step_ps = float(row["time_step_ps"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"manifest row {number} has an invalid time_step_ps") from error
            records.append(
                TrajectoryRecord(
                    trajectory_id=row["trajectory_id"],
                    topology=row["topology"],
                    trajectory=row["trajectory"],
                    time_step_ps=time_step_ps,
                    group_id=row["group_id"],
                    split=row["split"],
                )
            )
        return cls(tuple(records))

    @classmethod
    def from_csv(cls, path: str | Path) -> "TrajectoryManifest":
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return cls.from_rows(csv.DictReader(handle))

