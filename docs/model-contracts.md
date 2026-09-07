# Public multiscale model contracts

## Structural hierarchy

The public encoder uses four generic, variable-size levels:

```text
residue -> local structural unit -> interface -> global structural unit
```

Every residue belongs to one local structural unit, and every local structural
unit belongs to one global structural unit. A local unit cannot cross a chain or
global-unit boundary.

An interface is an unordered pair of global structural units. Directed contact
messages between local units are grouped into the corresponding interface.
Every interface has exactly two global endpoints, while an interface with no
active geometric contacts remains present as an exact-zero state. This stable
identity is important when comparing trajectories and binding Gates.

Global structural units are study-declared using chain-aware residue spans.
Their number, size, identity, and interface topology are data, not source-code
constants.

## Scalar and vector channels

Every level carries invariant scalar channels and three-dimensional equivariant
vector channels. GVP layers mix vector channels but never mix xyz axes. Vector
lengths are the only vector-to-scalar signal.

Consequently:

- scalar outputs are invariant to proper rotations;
- vector outputs transform with the input rotation;
- features built from relative geometry are translation independent;
- graph-level CVs are permutation invariant after fixed graph pooling.

## Interface construction

The interface path is explicit:

1. Encode messages on local-unit interface-member edges.
2. Aggregate those messages by stable interface identity with the all-one
   edge-weight denominator.
3. Apply zero-anchored interface transforms and the Interface State Gate.
4. Relay the interface state to both global endpoints through an explicit
   incidence tensor.
5. Combine the relayed interface message with the global covalent message.

Closing every contributing edge or interface relay therefore makes the
Interface Route exactly zero. Trainable biases cannot create a false interface
signal because interface transforms and Route heads are zero anchored.

## Fixed intervention denominators

For a target node or grouped interface, messages are aggregated as:

```text
sum_j baseline_weight[j] * gate[j] * message[j]
-------------------------------------------------
sum_j baseline_weight[j]  from the all-one graph
```

The denominator is never recomputed after applying a Gate. Closing one edge
removes its contribution without amplifying surviving neighbors. The same rule
applies to residue-to-local, local-to-global, interface-to-global, and
graph-level relays.

## Direct and Route readouts

The Direct readout pools stored states from residues, local structural units,
interfaces, and global structural units.

The shared Route head reads relation messages rather than reconstructing them
from stored global states:

- `route_interface` reads interface-to-global messages;
- `route_covalent` reads global covalent messages;
- `route_combined` reads their additive combination.

Global State replacement affects Direct but cannot retroactively change a Route
message computed before that insertion point.

## Trainable slow coordinates

Training consumes paired batches at time `t` and `t + lag`. The public objective
maximizes regularized non-reversible VAMP-2 scores for the Interface Route and
Direct CVs, with independently adjustable weights. Optional penalties control
canonical alignment, variance collapse, CV correlation, and within-trajectory
centering.

All covariance arithmetic is performed in float64 even when the geometric model
uses float32. Training and validation pairs must remain within one trajectory,
and train/validation/test membership is inherited from the trajectory manifest.

## Interpretation boundary

Gates measure model-internal dependence on stored or relayed structural
information. They are not molecular mutations and do not establish physical
causality without independent structural and experimental evidence.
