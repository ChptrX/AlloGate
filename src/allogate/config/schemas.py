"""Immutable, path-free configuration for public AlloGate experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RepresentationConfig:
    contact_on_angstrom: float = 8.0
    contact_off_angstrom: float = 10.0
    radial_basis_count: int = 16
    radial_basis_max_angstrom: float = 20.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.contact_on_angstrom < self.contact_off_angstrom:
            raise ValueError("contact radii must satisfy 0 <= on < off")
        if self.radial_basis_count < 2:
            raise ValueError("radial_basis_count must be at least two")
        if self.radial_basis_max_angstrom <= 0.0:
            raise ValueError("radial_basis_max_angstrom must be positive")


@dataclass(frozen=True, slots=True)
class HierarchyConfig:
    """Rules for residue -> local unit -> interface -> global unit construction."""

    target_local_structural_unit_size: int = 6
    minimum_local_structural_unit_size: int = 4
    maximum_local_structural_unit_size: int = 8
    respect_chain_boundaries: bool = True
    respect_global_structural_unit_boundaries: bool = True

    def __post_init__(self) -> None:
        if self.minimum_local_structural_unit_size < 1:
            raise ValueError("minimum_local_structural_unit_size must be positive")
        if not (
            self.minimum_local_structural_unit_size
            <= self.target_local_structural_unit_size
            <= self.maximum_local_structural_unit_size
        ):
            raise ValueError("local structural unit sizes must satisfy minimum <= target <= maximum")


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    residue_input_scalar_channels: int = 3
    residue_input_vector_channels: int = 2
    edge_scalar_channels: int = 2
    edge_vector_channels: int = 1
    hidden_scalar_channels: int = 64
    hidden_vector_channels: int = 16
    message_layers_per_scale: int = 1

    def __post_init__(self) -> None:
        channels = (
            self.residue_input_scalar_channels,
            self.residue_input_vector_channels,
            self.edge_scalar_channels,
            self.edge_vector_channels,
            self.hidden_scalar_channels,
            self.hidden_vector_channels,
        )
        if any(value < 0 for value in channels):
            raise ValueError("encoder channel counts cannot be negative")
        if self.residue_input_scalar_channels + self.residue_input_vector_channels < 1:
            raise ValueError("residue inputs require at least one channel")
        if self.edge_scalar_channels + self.edge_vector_channels < 1:
            raise ValueError("edge inputs require at least one channel")
        if self.hidden_scalar_channels < 1 or self.hidden_vector_channels < 1:
            raise ValueError("hidden scalar and vector channel counts must be positive")
        if self.message_layers_per_scale < 1:
            raise ValueError("message_layers_per_scale must be positive")


@dataclass(frozen=True, slots=True)
class ReadoutConfig:
    projection_channels: int = 64
    hidden_channels: int = 64

    def __post_init__(self) -> None:
        if self.projection_channels < 1 or self.hidden_channels < 1:
            raise ValueError("readout channel counts must be positive")


@dataclass(frozen=True, slots=True)
class MethodConfig:
    schema_version: str = "allogate.method.v2"
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    collective_variable_dimension: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != "allogate.method.v2":
            raise ValueError(f"unsupported method schema: {self.schema_version}")
        if self.collective_variable_dimension < 1:
            raise ValueError("collective_variable_dimension must be positive")


@dataclass(frozen=True, slots=True)
class KineticObjectiveConfig:
    """Configurable slow-coordinate objective evaluated on lagged graph pairs."""

    lag_time: float = 1.0
    lag_unit: str = "frame"
    covariance_regularization: float = 1.0e-4
    interface_route_weight: float = 1.0
    direct_weight: float = 0.1
    alignment_weight: float = 0.05
    standard_deviation_floor: float = 0.5
    standard_deviation_weight: float = 0.03
    decorrelation_weight: float = 0.02
    trajectory_centered_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.lag_time <= 0.0:
            raise ValueError("lag_time must be positive")
        if not self.lag_unit or any(char.isspace() for char in self.lag_unit):
            raise ValueError("lag_unit must be a non-empty token")
        if self.covariance_regularization <= 0.0:
            raise ValueError("covariance_regularization must be positive")
        non_negative = (
            self.interface_route_weight,
            self.direct_weight,
            self.alignment_weight,
            self.standard_deviation_floor,
            self.standard_deviation_weight,
            self.decorrelation_weight,
            self.trajectory_centered_weight,
        )
        if any(value < 0.0 for value in non_negative):
            raise ValueError("objective weights and floors must be non-negative")
        if self.interface_route_weight + self.direct_weight <= 0.0:
            raise ValueError("at least one slow-coordinate score weight must be positive")


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Optimizer settings; all fields are user-adjustable and scientifically hashed."""

    seed: int = 2026
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 1.0
    minimum_epochs: int = 10
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1.0e-4
    learning_rate_patience: int = 5
    learning_rate_factor: float = 0.5
    minimum_learning_rate: float = 1.0e-6
    dtype: str = "float32"
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("epochs must be positive and batch_size must be at least two")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 1 <= self.minimum_epochs <= self.epochs:
            raise ValueError("minimum_epochs must lie within the configured epoch range")
        if self.early_stopping_patience < 1 or self.learning_rate_patience < 1:
            raise ValueError("patience values must be positive")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        if not 0.0 < self.learning_rate_factor < 1.0:
            raise ValueError("learning_rate_factor must lie in (0, 1)")
        if not 0.0 < self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("minimum_learning_rate must lie in (0, learning_rate]")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete path-free training specification for one public experiment."""

    schema_version: str = "allogate.training_experiment.v1"
    method: MethodConfig = field(default_factory=MethodConfig)
    objective: KineticObjectiveConfig = field(default_factory=KineticObjectiveConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    def __post_init__(self) -> None:
        if self.schema_version != "allogate.training_experiment.v1":
            raise ValueError(f"unsupported training experiment schema: {self.schema_version}")
