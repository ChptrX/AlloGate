# Changelog

All notable public changes to AlloGate are recorded here. The project follows
Semantic Versioning while it remains in alpha; minor alpha releases may change
experimental tensor contracts when the change is documented below.

## 0.2.0a1 - 2026-09-07

### Added

- A strict, path-free training experiment schema in JSON or YAML.
- A seeded model factory and configurable DeepTICA-style slow-CV trainer.
- Variable-size batching for proteins and conformations with different graph
  sizes.
- Split-preserving, within-trajectory lag-pair manifests.
- Immutable checkpoint, history, and run-manifest artifacts with SHA-256 model
  and experiment identities.
- A deterministic synthetic training workflow exercised by continuous
  integration.

### Changed

- The public hierarchy is now named and represented explicitly as residues,
  local structural units, interfaces, and global structural units.
- Interfaces are first-class stable entities rather than an implicit contact
  relation between global units.
- Direct readout pools all four levels; relation readouts expose interface,
  covalent, and combined routes.
- The public package is no longer tied to a fixed residue count, hierarchy size,
  trajectory layout, replica count, GPU count, or protein identity.

### Compatibility

- `StructuralUnitSpec`, `validate_disjoint_units`,
  `build_contiguous_local_elements`, `EntityLevel.LOCAL_ELEMENT`, and
  `EntityLevel.STRUCTURAL_UNIT` remain temporary source aliases for the 0.1
  alpha API. New code should use the explicit 0.2 terminology.

### Scope boundary

- Raw trajectory readers, State-baseline estimation, kinetic-reference fitting,
  distributed orchestration, and empirical cross-protein qualification remain
  separate workflows. The 0.2 line should stay a pre-release until those items
  are validated for public use.
