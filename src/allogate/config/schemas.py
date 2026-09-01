"""Small immutable schemas for method settings in the first public milestone."""

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
    target_local_element_size: int = 6
    minimum_local_element_size: int = 4
    maximum_local_element_size: int = 8
    respect_chain_boundaries: bool = True
    respect_structural_unit_boundaries: bool = True

    def __post_init__(self) -> None:
        if self.minimum_local_element_size < 1:
            raise ValueError("minimum_local_element_size must be positive")
        if not (
            self.minimum_local_element_size
            <= self.target_local_element_size
            <= self.maximum_local_element_size
        ):
            raise ValueError("local-element sizes must satisfy minimum <= target <= maximum")


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    scalar_channels: int = 64
    vector_channels: int = 16
    message_layers_per_scale: int = 1

    def __post_init__(self) -> None:
        if self.scalar_channels < 1 or self.vector_channels < 1:
            raise ValueError("encoder scalar and vector channel counts must be positive")
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
    schema_version: str = "allogate.method.v1"
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    collective_variable_dimension: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != "allogate.method.v1":
            raise ValueError(f"unsupported method schema: {self.schema_version}")
        if self.collective_variable_dimension < 1:
            raise ValueError("collective_variable_dimension must be positive")
