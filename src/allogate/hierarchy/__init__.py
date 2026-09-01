"""Generic hierarchy definitions."""

from .mappings import IndexMapping, build_contiguous_local_elements
from .structural_units import ResidueSpan, StructuralUnitSpec, validate_disjoint_units

__all__ = [
    "IndexMapping",
    "ResidueSpan",
    "StructuralUnitSpec",
    "build_contiguous_local_elements",
    "validate_disjoint_units",
]

