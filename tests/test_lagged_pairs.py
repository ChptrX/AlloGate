import pytest

from allogate.data import (
    TrajectoryManifest,
    TrajectoryRecord,
    build_lagged_pair_manifest,
)


def trajectories() -> TrajectoryManifest:
    return TrajectoryManifest(
        (
            TrajectoryRecord(
                "traj_0",
                "inputs/topology.pdb",
                "inputs/traj_0.xtc",
                2.0,
                "rep_0",
                "train",
            ),
            TrajectoryRecord(
                "traj_1",
                "inputs/topology.pdb",
                "inputs/traj_1.xtc",
                2.0,
                "rep_1",
                "validation",
            ),
        )
    )


def test_lagged_pairs_are_within_trajectory_and_keep_splits() -> None:
    manifest = build_lagged_pair_manifest(
        trajectories(),
        {"traj_0": 6, "traj_1": 5},
        lag_time_ps=4.0,
        frame_stride=2,
    )
    assert [
        (item.trajectory_id, item.time_index, item.lagged_index, item.split)
        for item in manifest.pairs
    ] == [
        ("traj_0", 0, 2, "train"),
        ("traj_0", 2, 4, "train"),
        ("traj_1", 0, 2, "validation"),
        ("traj_1", 2, 4, "validation"),
    ]


def test_lag_must_be_an_integer_number_of_frames() -> None:
    with pytest.raises(ValueError, match="integer frame lag"):
        build_lagged_pair_manifest(
            trajectories(),
            {"traj_0": 6, "traj_1": 5},
            lag_time_ps=3.0,
        )
