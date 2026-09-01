import numpy as np
import pytest

from allogate.hierarchy.mappings import IndexMapping, build_contiguous_local_elements


def test_contiguous_elements_respect_unit_boundaries() -> None:
    chains = np.array(["A"] * 24)
    units = np.array([0] * 12 + [1] * 12)
    residue_to_local, local_to_unit = build_contiguous_local_elements(chains, units)
    np.testing.assert_array_equal(residue_to_local.children_per_parent, [6, 6, 6, 6])
    np.testing.assert_array_equal(local_to_unit.child_to_parent, [0, 0, 1, 1])
    composed = residue_to_local.compose(local_to_unit)
    np.testing.assert_array_equal(composed.child_to_parent, units)


def test_contiguous_elements_respect_chain_boundaries() -> None:
    chains = np.array(["A"] * 6 + ["B"] * 6)
    units = np.zeros(12, dtype=np.int64)
    residue_to_local, local_to_unit = build_contiguous_local_elements(chains, units)
    np.testing.assert_array_equal(residue_to_local.children_per_parent, [6, 6])
    np.testing.assert_array_equal(local_to_unit.child_to_parent, [0, 0])


def test_short_run_is_rejected_instead_of_crossing_a_boundary() -> None:
    with pytest.raises(ValueError, match="shorter than minimum"):
        build_contiguous_local_elements(["A"] * 3, [0] * 3)


def test_mapping_rejects_empty_parent() -> None:
    with pytest.raises(ValueError, match="every parent"):
        IndexMapping(np.array([0, 2]), parent_count=3)

