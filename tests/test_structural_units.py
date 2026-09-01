import pytest

from allogate.hierarchy.structural_units import (
    ResidueSpan,
    StructuralUnitSpec,
    validate_disjoint_units,
)


def test_span_round_trip() -> None:
    assert str(ResidueSpan.parse("A:4-19")) == "A:4-19"
    assert str(ResidueSpan.parse("B:7")) == "B:7"


def test_disjoint_units_support_multiple_chains() -> None:
    units = (
        StructuralUnitSpec.from_strings("unit_00", ["A:1-20", "B:1-3"]),
        StructuralUnitSpec.from_strings("unit_01", ["A:21-40", "B:4-8"]),
    )
    assert validate_disjoint_units(units) == units


def test_overlap_is_rejected() -> None:
    units = (
        StructuralUnitSpec.from_strings("unit_00", ["A:1-20"]),
        StructuralUnitSpec.from_strings("unit_01", ["A:20-30"]),
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_units(units)

