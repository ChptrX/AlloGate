# Public multiscale model contracts

## Hierarchy

The current encoder operates on three generic and configurable levels:

```text
residue-like node → local element → structural unit
```

Each child belongs to exactly one parent. Counts are inferred from mappings rather than constants. Every parent must contain at least one child. Contiguous local elements cannot cross a chain or structural-unit boundary.

## Scalar and vector channels

At each level a feature is represented by invariant scalar channels and three-dimensional equivariant vector channels. The GVP and message-passing layers may mix vector channels, but never xyz coordinates. Vector lengths are the only vector-to-scalar signal.

Consequently:

- scalar outputs are invariant to proper rotations;
- vector outputs transform with the same rotation as the input;
- feature construction from relative geometry is translation independent.

## Fixed intervention denominator

For target node `i`, relation messages are aggregated as

```text
sum_j baseline_weight[j] * gate[j] * message[j]
-------------------------------------------------
sum_j baseline_weight[j]  from the all-one graph
```

The denominator is never recomputed after applying a Gate. Closing one edge therefore removes its contribution without amplifying remaining neighbors. Closing every edge of a relation produces an exactly zero relation aggregate.

The same rule applies when children are relayed to a parent: the denominator is the original all-one child count.

## Relation provenance

At the structural-unit scale, contact and covalent relations are encoded by the same message layer but aggregated separately. The encoder exposes:

- contact messages;
- covalent messages;
- their exact additive combination.

This separation is retained through the readout instead of reconstructing relation identity after node pooling.

## Direct and Route readouts

The Direct readout pools stored node states from all hierarchy levels and combines their invariant projections.

The Route readout consumes relation messages, not pooled structural-unit node states. One shared head reads contact, covalent, and combined messages. Both the invariant projection and scalar head are anchored at zero, so an exactly zero relation message produces an exactly zero Route output despite trainable biases.

## Current interpretation boundary

Tensor-valued controls are now bound to stable public Gate identities through the Gate Registry. Canonical UIDs, State/Relay/Contact semantics, and provenance relationships are defined in `gate-registry.md`. Artifact identities and the evaluation service remain deferred.
