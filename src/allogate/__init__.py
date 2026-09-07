"""Public AlloGate method package."""

from .config.hashing import stable_digest
from .data import (
    LaggedPairManifest,
    LaggedPairRecord,
    TrajectoryManifest,
    TrajectoryRecord,
    build_lagged_pair_manifest,
)
from .gates import (
    EntityLevel,
    EntityRef,
    GateFamily,
    GateRegistry,
    GateSpec,
    ProvenanceDAG,
)
from .hierarchy.structural_units import (
    GlobalStructuralUnitSpec,
    InterfaceSpec,
    ResidueSpan,
)
from .kinetics import KineticReference

__all__ = [
    "EntityLevel",
    "EntityRef",
    "GateFamily",
    "GateRegistry",
    "GateSpec",
    "GlobalStructuralUnitSpec",
    "InterfaceSpec",
    "KineticReference",
    "LaggedPairManifest",
    "LaggedPairRecord",
    "ProvenanceDAG",
    "ResidueSpan",
    "TrajectoryManifest",
    "TrajectoryRecord",
    "build_lagged_pair_manifest",
    "stable_digest",
]

__version__ = "0.2.0a1"
