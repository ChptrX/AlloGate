from pathlib import Path

import pytest

from allogate.data.manifests import TrajectoryManifest, TrajectoryRecord


def record(**changes: object) -> TrajectoryRecord:
    values: dict[str, object] = {
        "trajectory_id": "traj_000",
        "topology": "inputs/topology.pdb",
        "trajectory": "inputs/traj_000.xtc",
        "time_step_ps": 20.0,
        "group_id": "group_00",
        "split": "train",
    }
    values.update(changes)
    return TrajectoryRecord(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "path",
    ["/" + "private/data.xtc", "C:" + "\\private\\data.xtc", ".." + "/data.xtc"],
)
def test_private_or_escaping_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        record(trajectory=path)


def test_manifest_reads_portable_csv(tmp_path: Path) -> None:
    path = tmp_path / "trajectories.csv"
    path.write_text(
        "trajectory_id,topology,trajectory,time_step_ps,group_id,split\n"
        "traj_000,inputs/topology.pdb,inputs/traj_000.xtc,20,group_00,train\n"
        "traj_001,inputs/topology.pdb,inputs/traj_001.xtc,20,group_01,test\n",
        encoding="utf-8",
    )
    manifest = TrajectoryManifest.from_csv(path)
    assert len(manifest.records) == 2
    assert manifest.records[1].split == "test"


def test_manifest_rejects_duplicate_identity() -> None:
    with pytest.raises(ValueError, match="unique"):
        TrajectoryManifest((record(), record(trajectory="inputs/other.xtc")))
