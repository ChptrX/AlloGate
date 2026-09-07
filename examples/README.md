# Synthetic public examples

The examples contain no molecular structure, trajectory, fitted checkpoint, or
study-specific residue identity. They validate software contracts rather than
making a biological claim.

## Configurable training

`synthetic_training.py` builds variable-size four-level graph tensors, pairs
them at a synthetic lag, initializes the model from a seeded experiment
configuration, trains Direct and Interface Route slow coordinates, and writes
an auditable checkpoint/history/manifest bundle.

```text
python examples/synthetic_training.py --output synthetic-training-run
```

Copy `configs/protein_training.yaml` as the starting point for a real
experiment. Its channel counts must match the tensors emitted by the trajectory
adapter.

## Gate evaluation

`synthetic_gate_audit.py` creates semantic State, Relay, and Contact Gate
identities; captures one immutable all-one target; computes the full Jacobian
and one named Hessian block; and evaluates a finite Relay-Gate dose schedule.

```text
python examples/synthetic_gate_audit.py
```

To create a content-addressed result and path-free logical reference:

```text
python examples/synthetic_gate_audit.py --artifact-root example-artifacts
```
