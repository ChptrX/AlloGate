"""Train a tiny AlloGate model on deterministic synthetic lagged graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from allogate.config import (
    EncoderConfig,
    ExperimentConfig,
    KineticObjectiveConfig,
    MethodConfig,
    OptimizationConfig,
    ReadoutConfig,
    load_experiment_config,
)
from allogate.models import (
    AlloGateGeometryModel,
    DirectedEdges,
    InterfaceEdges,
    InterfaceIncidence,
    MultiscaleGraph,
    ParentIndex,
)
from allogate.training import LaggedBatch, ModelBatch, SlowCVTrainer


def _edges(
    graph_count: int,
    *,
    nodes_per_graph: int,
    pairs: tuple[tuple[int, int], ...],
    phases: torch.Tensor,
) -> DirectedEdges:
    sources: list[int] = []
    targets: list[int] = []
    scalars: list[list[float]] = []
    vectors: list[list[list[float]]] = []
    for graph_index in range(graph_count):
        phase = float(phases[graph_index])
        offset = graph_index * nodes_per_graph
        for edge_index, (source, target) in enumerate(pairs):
            angle = phase + 0.2 * edge_index
            sources.append(offset + source)
            targets.append(offset + target)
            scalars.append(
                [1.0 + 0.1 * torch.sin(torch.tensor(angle)).item(), float(source != target)]
            )
            vectors.append(
                [[
                    torch.cos(torch.tensor(angle)).item(),
                    torch.sin(torch.tensor(angle)).item(),
                    0.0,
                ]]
            )
    count = len(sources)
    return DirectedEdges(
        source=torch.tensor(sources, dtype=torch.long),
        target=torch.tensor(targets, dtype=torch.long),
        features=(
            torch.tensor(scalars, dtype=torch.float32),
            torch.tensor(vectors, dtype=torch.float32),
        ),
        baseline_weight=torch.ones(count),
        node_count=graph_count * nodes_per_graph,
    )


def make_model_batch(phases: torch.Tensor) -> ModelBatch:
    graph_count = len(phases)
    residue_scalars: list[list[float]] = []
    residue_vectors: list[list[list[float]]] = []
    for phase in phases.tolist():
        for residue in range(4):
            angle = phase + 0.35 * residue
            residue_scalars.append(
                [
                    torch.sin(torch.tensor(angle)).item(),
                    torch.cos(torch.tensor(angle)).item(),
                ]
            )
            residue_vectors.append(
                [[
                    torch.cos(torch.tensor(angle)).item(),
                    torch.sin(torch.tensor(angle)).item(),
                    0.25,
                ]]
            )
    residue_pairs = ((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2))
    local_pairs = ((0, 1), (1, 0))
    global_pairs = ((0, 1), (1, 0))
    interface_member_edges = _edges(
        graph_count,
        nodes_per_graph=2,
        pairs=local_pairs,
        phases=phases,
    )
    graph = MultiscaleGraph(
        residue_features=(
            torch.tensor(residue_scalars, dtype=torch.float32),
            torch.tensor(residue_vectors, dtype=torch.float32),
        ),
        residue_edges=_edges(
            graph_count,
            nodes_per_graph=4,
            pairs=residue_pairs,
            phases=phases,
        ),
        residue_to_local_unit=ParentIndex(
            torch.tensor(
                [
                    2 * graph_index + local
                    for graph_index in range(graph_count)
                    for local in (0, 0, 1, 1)
                ],
                dtype=torch.long,
            ),
            2 * graph_count,
        ),
        local_unit_edges=_edges(
            graph_count,
            nodes_per_graph=2,
            pairs=local_pairs,
            phases=phases,
        ),
        local_unit_to_global_unit=ParentIndex(
            torch.arange(2 * graph_count, dtype=torch.long),
            2 * graph_count,
        ),
        interface_edges=InterfaceEdges(
            interface_member_edges,
            torch.repeat_interleave(torch.arange(graph_count), 2),
            graph_count,
        ),
        interface_to_global_unit=InterfaceIncidence(
            interface_index=torch.repeat_interleave(torch.arange(graph_count), 2),
            global_unit_index=torch.arange(2 * graph_count),
            baseline_weight=torch.ones(2 * graph_count),
            interface_count=graph_count,
            global_unit_count=2 * graph_count,
        ),
        global_unit_covalent_edges=_edges(
            graph_count,
            nodes_per_graph=2,
            pairs=global_pairs,
            phases=phases,
        ),
    )
    return ModelBatch(
        graph=graph,
        residue_graph_index=torch.repeat_interleave(torch.arange(graph_count), 4),
        local_unit_graph_index=torch.repeat_interleave(torch.arange(graph_count), 2),
        interface_graph_index=torch.arange(graph_count),
        global_unit_graph_index=torch.repeat_interleave(torch.arange(graph_count), 2),
        graph_count=graph_count,
    )


def default_config() -> ExperimentConfig:
    return ExperimentConfig(
        method=MethodConfig(
            encoder=EncoderConfig(
                residue_input_scalar_channels=2,
                residue_input_vector_channels=1,
                edge_scalar_channels=2,
                edge_vector_channels=1,
                hidden_scalar_channels=8,
                hidden_vector_channels=2,
                message_layers_per_scale=1,
            ),
            readout=ReadoutConfig(projection_channels=6, hidden_channels=8),
            collective_variable_dimension=2,
        ),
        objective=KineticObjectiveConfig(
            trajectory_centered_weight=0.1,
        ),
        optimization=OptimizationConfig(
            seed=17,
            epochs=4,
            batch_size=6,
            learning_rate=2.0e-3,
            minimum_epochs=2,
            early_stopping_patience=2,
            learning_rate_patience=1,
        ),
    )


def run_example(
    output_directory: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    config = (
        default_config()
        if config_path is None
        else load_experiment_config(config_path)
    )
    phases = torch.linspace(-1.0, 1.0, 6)
    batch = LaggedBatch(
        make_model_batch(phases),
        make_model_batch(phases + 0.25),
        trajectory_index=torch.tensor([0, 0, 1, 1, 2, 2]),
    )
    model = AlloGateGeometryModel.from_experiment_config(config)
    result = SlowCVTrainer(model, config).fit(
        [batch],
        [batch],
        output_directory=output_directory,
    )
    return {
        "schema_version": "allogate.synthetic_training.v1",
        "experiment_digest": result.experiment_digest,
        "run_digest": result.run_digest,
        "initial_model_digest": result.initial_model_digest,
        "model_digest": result.model_digest,
        "best_epoch": result.best_epoch,
        "completed_epochs": result.completed_epochs,
        "best_validation_loss": result.best_validation_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_example(arguments.output, arguments.config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
