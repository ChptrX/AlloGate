# Gate Registry and provenance contract

## Semantic identities

An `EntityRef` identifies one scientific object by a public namespace, an entity
level, and a stable semantic key. Supported levels are residue, contact, local
structural unit, interface, global structural unit, relation, and readout.

Keys are public semantic tokens rather than topology-array positions, batch
offsets, filenames, host paths, or display labels.

## Gate families

### State Gate

A State Gate controls information stored at one entity:

```text
scalar(g) = baseline + g * (state - baseline)
vector(g) = g * vector
```

Scalar baselines are frozen, identity-specific training statistics. Vector
baselines are zero. Feature baselines are distinct from the fixed kinetic
reference used after CV training.

### Relay Gate

A Relay Gate controls information transmitted into the next hierarchy
operation. It multiplies a child, interface, or readout contribution while the
denominator remains fixed at its all-one value.

### Contact Gate

A Contact Gate controls one unordered pair of distinct entities. Endpoint order
is canonicalized. Both directed runtime edges bind to the same semantic Gate
UID.

## Canonical UIDs

A Gate UID is derived from its schema, public namespace, family, canonical
targets, relation name, and intervention semantics. It does not include:

- tensor index or batch position;
- training split, fold, or checkpoint;
- display name or alias;
- local path, host, or device;
- intervention dose.

Registry construction order and display-label changes therefore cannot change a
Gate UID.

## Runtime control bindings

`ModelGateBinding` maps global Registry values to repeated runtime tensors. The
public encoder exposes these formal controls:

| Model operation | Gate family | Semantic level |
|---|---|---|
| residue/local/interface/global state | State | corresponding node level |
| residue -> local unit | Relay | residue |
| local unit -> global unit | Relay | local structural unit |
| interface -> global endpoints | Relay | interface |
| residue or local-unit geometric edge | Contact | endpoint level |
| interface-member edge | Contact | local structural unit endpoints |

The immutable no-intervention vector is all ones. Every non-one value must be
requested explicitly.

## Provenance DAG

The explicit acyclic provenance graph contains relationships such as:

```text
residue -> local structural unit -> global structural unit -> Direct readout
local-unit contact -> interface -> global structural unit -> Interface Route
global covalent relation -> Covalent Route
```

Duplicate nodes, duplicate edges, self-loops, unknown endpoints, and cycles are
rejected. A Registry passes provenance qualification only when every Gate
references an existing provenance node.

Parent-child relationships encode lineage, not independent observations.
Downstream statistics must not count a local effect and its aggregate global
effect as independent evidence.

## Frozen insertion points

- Residue State follows residue message passing.
- Local Structural Unit State follows local-unit message passing.
- Interface State follows contact-group aggregation and precedes interface
  relay to both global endpoints.
- Global relation messages and Route outputs are computed before Global
  Structural Unit State replacement.
- Relay closure never triggers post-intervention degree renormalization.
