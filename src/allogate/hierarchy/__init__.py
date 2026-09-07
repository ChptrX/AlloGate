"""Generic local-unit, interface, and global-unit definitions."""

from .mappings import (
    IndexMapping,
    build_contiguous_local_elements,
    build_contiguous_local_structural_units,
)
from .structural_units import (
    GlobalStructuralUnitSpec,
    InterfaceSpec,
    ResidueSpan,
    StructuralUnitSpec,
    validate_disjoint_global_units,
    validate_disjoint_units,
    validate_interfaces,
)

__all__ = [
    "GlobalStructuralUnitSpec",
    "IndexMapping",
    "InterfaceSpec",
    "ResidueSpan",
    "StructuralUnitSpec",
    "build_contiguous_local_elements",
    "build_contiguous_local_structural_units",
    "validate_disjoint_global_units",
    "validate_disjoint_units",
    "validate_interfaces",
]
