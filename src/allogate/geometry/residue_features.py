"""Generic residue-level geometric features.

The virtual-beta-carbon convention is compatible with the backbone feature
used by ProteinMPNN. Its constants are documented rather than hidden in model
code; see THIRD_PARTY_NOTICES.md and docs/provenance-ledger.csv.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


_CB_CROSS_COEFFICIENT = -0.58273431
_CB_N_DIRECTION_COEFFICIENT = 0.56802827
_CB_C_DIRECTION_COEFFICIENT = -0.54067466


def _backbone_array(coordinates: ArrayLike) -> NDArray[np.floating]:
    array = np.asarray(coordinates)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    if array.ndim < 2 or array.shape[-2:] != (3, 3):
        raise ValueError("backbone coordinates must end in [N, CA, C] x xyz with shape (..., 3, 3)")
    if not np.isfinite(array).all():
        raise ValueError("backbone coordinates must be finite")
    return array


def virtual_beta_carbon(backbone: ArrayLike) -> NDArray[np.floating]:
    """Construct virtual C-beta positions from N, C-alpha, and C coordinates.

    The output is translation equivariant and rotation/reflection equivariant
    for proper rotations when the cross-product orientation is preserved. The
    coefficient convention matches the geometry used by ProteinMPNN; the code
    here is a new array-oriented implementation with explicit validation.
    """

    coordinates = _backbone_array(backbone)
    nitrogen, alpha_carbon, carbonyl_carbon = np.moveaxis(coordinates, -2, 0)
    toward_nitrogen = alpha_carbon - nitrogen
    toward_carbonyl = carbonyl_carbon - alpha_carbon
    plane_normal = np.cross(toward_nitrogen, toward_carbonyl)
    offset = (
        _CB_CROSS_COEFFICIENT * plane_normal
        + _CB_N_DIRECTION_COEFFICIENT * toward_nitrogen
        + _CB_C_DIRECTION_COEFFICIENT * toward_carbonyl
    )
    return alpha_carbon + offset


def hybrid_beta_carbon(
    backbone: ArrayLike,
    observed_beta_carbon: ArrayLike,
    observed_mask: ArrayLike,
) -> NDArray[np.floating]:
    """Use observed C-beta coordinates where present and virtual ones otherwise."""

    virtual = virtual_beta_carbon(backbone)
    observed = np.asarray(observed_beta_carbon, dtype=virtual.dtype)
    mask = np.asarray(observed_mask, dtype=bool)
    if observed.shape != virtual.shape:
        raise ValueError("observed_beta_carbon must match the virtual coordinate shape")
    if mask.shape != virtual.shape[:-1]:
        raise ValueError("observed_mask must match the residue batch shape")
    if not np.isfinite(observed[mask]).all():
        raise ValueError("selected observed C-beta coordinates must be finite")
    return np.where(mask[..., None], observed, virtual)


def cosine_switch(distance: ArrayLike, *, start: float, stop: float) -> NDArray[np.floating]:
    """Smoothly switch from one to zero over the closed interval [start, stop]."""

    if not 0.0 <= start < stop:
        raise ValueError("switch radii must satisfy 0 <= start < stop")
    values = np.asarray(distance)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    phase = np.pi * (values - start) / (stop - start)
    transition = 0.5 * (1.0 + np.cos(phase))
    return np.where(values <= start, 1.0, np.where(values >= stop, 0.0, transition))


def gaussian_radial_basis(
    distance: ArrayLike,
    *,
    count: int,
    maximum: float,
) -> NDArray[np.floating]:
    """Expand distances on uniformly spaced Gaussian radial basis centers."""

    if count < 2:
        raise ValueError("count must be at least two")
    if maximum <= 0.0:
        raise ValueError("maximum must be positive")
    values = np.asarray(distance)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    centers = np.linspace(0.0, maximum, count, dtype=values.dtype)
    spacing = centers[1] - centers[0]
    scaled = (values[..., None] - centers) / spacing
    return np.exp(-(scaled * scaled))

