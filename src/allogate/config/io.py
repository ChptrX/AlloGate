"""Strict JSON/YAML loading for path-free experiment configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schemas import (
    EncoderConfig,
    ExperimentConfig,
    HierarchyConfig,
    KineticObjectiveConfig,
    MethodConfig,
    OptimizationConfig,
    ReadoutConfig,
    RepresentationConfig,
)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _strict(mapping: dict[str, Any], allowed: set[str], name: str) -> dict[str, Any]:
    extra = set(mapping).difference(allowed)
    if extra:
        raise ValueError(f"unknown {name} fields: {sorted(extra)}")
    return mapping


def experiment_config_from_mapping(value: Mapping[str, Any]) -> ExperimentConfig:
    root = _strict(
        _mapping(value, "experiment config"),
        {"schema_version", "method", "objective", "optimization"},
        "experiment config",
    )
    method_raw = _strict(
        _mapping(root.get("method", {}), "method"),
        {
            "schema_version",
            "representation",
            "hierarchy",
            "encoder",
            "readout",
            "collective_variable_dimension",
        },
        "method",
    )
    representation = RepresentationConfig(
        **_strict(
            _mapping(method_raw.pop("representation", {}), "representation"),
            set(RepresentationConfig.__dataclass_fields__),
            "representation",
        )
    )
    hierarchy = HierarchyConfig(
        **_strict(
            _mapping(method_raw.pop("hierarchy", {}), "hierarchy"),
            set(HierarchyConfig.__dataclass_fields__),
            "hierarchy",
        )
    )
    encoder = EncoderConfig(
        **_strict(
            _mapping(method_raw.pop("encoder", {}), "encoder"),
            set(EncoderConfig.__dataclass_fields__),
            "encoder",
        )
    )
    readout = ReadoutConfig(
        **_strict(
            _mapping(method_raw.pop("readout", {}), "readout"),
            set(ReadoutConfig.__dataclass_fields__),
            "readout",
        )
    )
    method = MethodConfig(
        representation=representation,
        hierarchy=hierarchy,
        encoder=encoder,
        readout=readout,
        **method_raw,
    )
    objective = KineticObjectiveConfig(
        **_strict(
            _mapping(root.get("objective", {}), "objective"),
            set(KineticObjectiveConfig.__dataclass_fields__),
            "objective",
        )
    )
    optimization = OptimizationConfig(
        **_strict(
            _mapping(root.get("optimization", {}), "optimization"),
            set(OptimizationConfig.__dataclass_fields__),
            "optimization",
        )
    )
    return ExperimentConfig(
        schema_version=str(root.get("schema_version", "allogate.training_experiment.v1")),
        method=method,
        objective=objective,
        optimization=optimization,
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        raw = json.loads(text)
    elif source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError(
                "YAML configuration requires the 'training' optional dependency"
            ) from error
        raw = yaml.safe_load(text)
    else:
        raise ValueError("experiment configuration must use .json, .yaml, or .yml")
    return experiment_config_from_mapping(_mapping(raw, "experiment config"))
