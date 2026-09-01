"""Public AlloGate method package."""

from .config.hashing import stable_digest
from .data.manifests import TrajectoryManifest, TrajectoryRecord
from .gates import EntityLevel, EntityRef, GateFamily, GateRegistry, GateSpec, ProvenanceDAG
from .hierarchy.structural_units import ResidueSpan, StructuralUnitSpec
from .kinetics import KineticReference

__all__ = [
    "EntityLevel",
    "EntityRef",
    "GateFamily",
    "GateRegistry",
    "GateSpec",
    "KineticReference",
    "ProvenanceDAG",
    "ResidueSpan",
    "StructuralUnitSpec",
    "TrajectoryManifest",
    "TrajectoryRecord",
    "stable_digest",
]

__version__ = "0.1.0a2"
