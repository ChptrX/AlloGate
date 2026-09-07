import pytest

from allogate.hierarchy.structural_units import (
    GlobalStructuralUnitSpec,
    InterfaceSpec,
    ResidueSpan,
    validate_disjoint_global_units,
    validate_interfaces,
)


def test_span_round_trip() -> None:
    assert str(ResidueSpan.parse("A:4-19")) == "A:4-19"
    assert str(ResidueSpan.parse("B:7")) == "B:7"


def test_disjoint_units_support_multiple_chains() -> None:
    units = (
        GlobalStructuralUnitSpec.from_strings("unit_00", ["A:1-20", "B:1-3"]),
        GlobalStructuralUnitSpec.from_strings("unit_01", ["A:21-40", "B:4-8"]),
    )
    assert validate_disjoint_global_units(units) == units


def test_overlap_is_rejected() -> None:
    units = (
        GlobalStructuralUnitSpec.from_strings("unit_00", ["A:1-20"]),
        GlobalStructuralUnitSpec.from_strings("unit_01", ["A:20-30"]),
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_global_units(units)


def test_interfaces_are_canonical_and_reference_known_global_units() -> None:
    units = (
        GlobalStructuralUnitSpec.from_strings("unit_b", ["A:1-10"]),
        GlobalStructuralUnitSpec.from_strings("unit_a", ["A:11-20"]),
    )
    interface = InterfaceSpec("interface_a_b", ("unit_b", "unit_a"))
    assert interface.global_unit_ids == ("unit_a", "unit_b")
    assert validate_interfaces((interface,), units) == (interface,)
