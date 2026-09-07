# Protein graph data contract

## Identity layer

Each trajectory is declared in a relative-path `TrajectoryManifest` with:

| Field | Contract |
|---|---|
| `trajectory_id` | unique stable token |
| `topology` | portable relative path |
| `trajectory` | portable relative path |
| `time_step_ps` | positive stored-frame spacing |
| `group_id` | replica/ensemble grouping key |
| `split` | train, validation, or test |

Absolute paths and parent-directory escapes are rejected. Resolve relative paths
in the private runtime adapter; never place machine-specific paths in the
scientific manifest committed to a public repository.

## Graph tensors

`MultiscaleGraph` contains:

| Tensor group | Meaning |
|---|---|
| `residue_features` | invariant scalar and equivariant vector residue channels |
| `residue_edges` | directed residue graph with fixed baseline weights |
| `residue_to_local_unit` | one parent for every residue |
| `local_unit_edges` | directed local structural unit graph |
| `local_unit_to_global_unit` | one global parent for every local unit |
| `interface_edges` | local-unit contact edges plus edge-to-interface identity |
| `interface_to_global_unit` | exactly two global endpoints per interface |
| `global_unit_covalent_edges` | declared global covalent/backbone relations |

Every edge feature pair has shapes `(E, S)` and `(E, V, 3)`. Every node feature
pair has shapes `(N, S)` and `(N, V, 3)`. Indices use `torch.long`, floating
tensors share a dtype/device, and all runtime indices are zero based.

Baseline edge/incidence weights define the all-one denominator. They are
geometry/reference data and must not be recomputed after a Gate is changed.

## Local structural units

`build_contiguous_local_structural_units` partitions ordered residues according
to configurable minimum, target, and maximum sizes. It refuses to cross chain
or global-unit boundaries. The public default target is six residues, but it is
not a frozen biological assumption.

## Interfaces

An `InterfaceSpec` names an unordered pair of `GlobalStructuralUnitSpec`
identities. Runtime contact edges between local units are assigned to that
stable interface identity. Both directed edge orientations normally share the
same interface group.

Interfaces remain in the tensor layout even when no contact is active in one
frame. Their state and downstream Interface Route contribution are then exactly
zero. This prevents frame-dependent identity changes.

## Global structural units

`GlobalStructuralUnitSpec` accepts one or more chain-aware residue spans.
Global units must be non-overlapping. They may represent domains, subdomains,
chains, motifs, or another preregistered structural partition appropriate to
the protein.

The source code never supplies a default partition. Publish the partition and
its rationale with each trained model.

## Batching

`ModelBatch` carries graph membership at residue, local-unit, interface, and
global-unit levels. `collate_model_batches` offsets and concatenates these
variable-size structures; it does not assume that two proteins or conformations
have equal counts.

For one protein trajectory, topology-level mappings and stable interface
identities will usually be reused while geometry-derived features and active
contact weights change by frame.

## Recommended adapter checks

A trajectory adapter should verify:

- atom/residue identity and chain order are stable across frames;
- units cover exactly the intended modeled residues;
- coordinate and distance units are declared and converted once;
- missing atoms and alternate locations follow an explicit policy;
- periodic-boundary handling is explicit;
- interface identities do not depend on the current batch order;
- lag pairs never cross trajectories;
- train-only statistics do not read validation/test data.
