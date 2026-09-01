# Synthetic public example

`synthetic_gate_audit.py` demonstrates the complete public evaluation contract without molecular structures, trajectories, checkpoints, or fitted study parameters. It creates three semantic Gate families, a one-dimensional fixed kinetic reference, exact first and second derivatives, and a finite Relay-Gate dose response.

From an environment containing the `evaluation` extra, run:

```text
python examples/synthetic_gate_audit.py
```

To also create a content-addressed result and a path-free logical reference:

```text
python examples/synthetic_gate_audit.py --artifact-root example-artifacts
```

The numerical model in this example is intentionally synthetic. It validates software contracts and does not represent a molecular or biological result.
