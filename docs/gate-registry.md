# Gate Registry and provenance contract

## Semantic identities

An `EntityRef` identifies one scientific object by:

- a public namespace;
- an entity level;
- a stable semantic key.

Supported levels are residue, contact, local element, interface, structural unit, relation, and readout. Keys are public tokens rather than topology-array positions, tensor offsets, filenames, or paths.

## Gate families

### State Gate

A State Gate controls information stored at one entity. Closing the Gate replaces scalar channels with a frozen calibrated feature baseline and sets equivariant vector channels to zero:

```text
state(g) = baseline + g * (state - baseline)
vector(g) = g * vector
```

Feature baselines are distinct from the invariant no-intervention kinetic target used by later analysis stages.

### Relay Gate

A Relay Gate controls information transmitted from an entity into the next hierarchy operation. It multiplies the message or pooled contribution, while the denominator remains fixed at its all-one value.

### Contact Gate

A Contact Gate controls one undirected pair of distinct entities. Endpoint order is canonicalized, so `(left, right)` and `(right, left)` produce the same Gate UID. Runtime directed edges may bind the same canonical UID more than once.

## Canonical UID

A Gate UID is derived from:

- schema version;
- public namespace;
- Gate family;
- canonical entity target UID or contact endpoint UIDs;
- relation name;
- intervention semantics.

It does not include:

- tensor index or batch position;
- training fold or checkpoint;
- display name or alias;
- local paths, host information, or runtime device;
- intervention dose.

The UID therefore remains stable if a Registry is constructed in a different order or if display labels change.

## Registry and tensor layout

`GateRegistry` verifies unique UIDs and aliases and orders Gates by a fixed family/level/UID contract. The global `GateTensorLayout` is derived from that order and contains contiguous global and group-local indices.

The immutable no-intervention vector is all ones. A non-one value must be supplied explicitly as an intervention.

Runtime graph batches use `ModelGateBinding` to align semantic Gate UIDs with flattened model tensors. Repeated UIDs are expected when the same semantic entity occurs in multiple trajectories in one batch. Runtime alignment is separate from Registry identity and therefore cannot alter a Gate UID.

## Provenance DAG

The provenance graph stores entities as nodes and explicit relations such as:

```text
residue → local element → structural unit → readout
residue → contact → local element/interface → relation/readout
```

Every edge must reference known nodes. Duplicate nodes, duplicate edges, self-loops, and cycles are rejected. A Registry passes provenance qualification only when every Gate references existing provenance nodes.

Parent-child relationships represent lineage, not independent evidence. Downstream analysis must avoid counting a child effect and its aggregate parent as two independent observations.

## Frozen insertion points in the current encoder

- Residue State is applied after residue message passing and before residue relay.
- Local-element State is applied after local message passing and before structural-unit relay.
- Structural-unit relation messages and Route outputs are computed before structural-unit State replacement.
- Structural-unit State therefore affects Direct stored-state readout but does not retroactively change relation-specific Route messages.

Interface entities and Gates are supported by the Registry and provenance schemas. Their neural insertion point remains deferred until a public dynamic-interface encoder is introduced.

