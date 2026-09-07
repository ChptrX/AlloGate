import json
from pathlib import Path
import runpy

import pytest


torch = pytest.importorskip("torch")

from allogate.config import (  # noqa: E402
    ExperimentConfig,
    experiment_config_from_mapping,
    load_experiment_config,
)
from allogate.models import AlloGateGeometryModel  # noqa: E402
from allogate.training import collate_model_batches  # noqa: E402


def test_public_yaml_exposes_adjustable_training_parameters() -> None:
    config = load_experiment_config(
        Path(__file__).parents[1] / "examples" / "configs" / "protein_training.yaml"
    )
    assert config.method.schema_version == "allogate.method.v2"
    assert config.method.encoder.message_layers_per_scale == 2
    assert config.optimization.batch_size == 128
    assert config.objective.trajectory_centered_weight == 0.25


def test_configuration_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown optimization"):
        experiment_config_from_mapping(
            {
                "optimization": {
                    "epochs": 3,
                    "batch_size": 2,
                    "minimum_epochs": 1,
                    "mystery": True,
                }
            }
        )


def test_seeded_model_factory_is_reproducible() -> None:
    config = ExperimentConfig()
    first = AlloGateGeometryModel.from_experiment_config(config)
    second = AlloGateGeometryModel.from_experiment_config(config)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], atol=0.0, rtol=0.0)


def test_variable_size_batch_collation_preserves_graph_membership() -> None:
    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "examples" / "synthetic_training.py")
    )
    first = namespace["make_model_batch"](torch.tensor([0.0]))
    second = namespace["make_model_batch"](torch.tensor([0.5, 1.0]))
    combined = collate_model_batches((first, second))
    assert combined.graph_count == 3
    assert combined.graph.residue_count == 12
    assert combined.graph.local_unit_count == 6
    assert combined.graph.interface_count == 3
    assert combined.graph.global_unit_count == 6
    assert combined.residue_graph_index.tolist() == [0] * 4 + [1] * 4 + [2] * 4


def test_synthetic_training_writes_auditable_artifacts(tmp_path: Path) -> None:
    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "examples" / "synthetic_training.py")
    )
    output = tmp_path / "training"
    summary = namespace["run_example"](output)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(output / "best-model.pt", weights_only=True)
    assert summary["model_digest"] == run["model_digest"]
    assert checkpoint["model_digest"] == run["model_digest"]
    assert checkpoint["experiment_digest"] == run["experiment_digest"]
    assert checkpoint["initial_model_digest"] == run["initial_model_digest"]
    assert checkpoint["run_digest"] == run["run_digest"]
    assert (output / "history.json").is_file()
    assert len(run["model_digest"]) == 64
