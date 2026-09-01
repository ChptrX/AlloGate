# AlloGate

[![Tests](https://github.com/ChptrX/AlloGate/actions/workflows/test.yml/badge.svg)](https://github.com/ChptrX/AlloGate/actions/workflows/test.yml)
[![Public content audit](https://github.com/ChptrX/AlloGate/actions/workflows/privacy.yml/badge.svg)](https://github.com/ChptrX/AlloGate/actions/workflows/privacy.yml)
[![Package smoke test](https://github.com/ChptrX/AlloGate/actions/workflows/package.yml/badge.svg)](https://github.com/ChptrX/AlloGate/actions/workflows/package.yml)

> **Status: evaluation-first alpha.** Slow-CV training and kinetic-reference fitting are not yet included.

AlloGate is an open method for learning slow collective variables from molecular-dynamics trajectories and auditing how structural information affects a fixed kinetic target through differentiable gates.

This repository is being rebuilt from a blank public tree. It contains only generic, provenance-reviewed components. It does not contain study-specific structures, trajectories, checkpoints, residue identities, server configuration, or legacy source bundles.

## Current public milestone

Version `0.1.0a2` now contains four provenance-reviewed migration batches:

- scalar/vector geometric operations and a paper-specified geometric vector perceptron;
- virtual-beta-carbon and generic distance features;
- relative-path trajectory manifests;
- generic, non-overlapping structural-unit definitions and deterministic method hashing.
- variable-size graph and hierarchy tensor contracts;
- deterministic contact geometry and smooth baseline edge weights;
- fixed-denominator, relation-aware equivariant message passing;
- residue → local-element → structural-unit encoding;
- invariant Direct readout and zero-preserving Contact/Covalent Route readouts.
- canonical State, Relay, and Contact Gate definitions;
- deterministic global Gate tensor layouts and runtime bindings;
- scalar-baseline State replacement at frozen insertion points;
- an explicit acyclic provenance graph connecting structural entities and readouts.
- a path-free, immutable kinetic-reference format with NumPy/PyTorch evaluation;
- one identity-checked Gate evaluation service for finite responses, Jacobians, and named Hessian blocks;
- immutable all-one `q0` capture with explicit drift replay;
- deterministic single-Gate dose schedules and a content-addressed result store.

Training and kinetic-reference construction are not yet part of these migration batches. The public evaluation layer consumes an already-frozen model and target; it does not infer or refit either one.

## Installation for development

```text
python -m pip install -e ".[test]"
python -m pytest
```

The GNN component requires PyTorch and can be installed separately with `.[gnn]`.
The differentiable evaluation service can be installed without the GNN modules using `.[evaluation]`.

## Synthetic end-to-end example

The public example uses deterministic synthetic CV coordinates and no molecular or study data:

```text
python examples/synthetic_gate_audit.py
```

It constructs State, Relay, and Contact Gate identities; captures one immutable all-one target; computes the full Jacobian and one named Hessian block; and evaluates a finite Relay-Gate dose schedule. See `examples/README.md` for optional content-addressed output.

## Scientific and source provenance

The GVP implementation follows the mathematical construction introduced by Jing et al., *Learning from Protein Structure with Geometric Vector Perceptrons* (ICLR 2021). The source was newly organized for AlloGate, while known open implementations that informed the project's earlier development remain credited in `THIRD_PARTY_NOTICES.md`.

Virtual-beta-carbon reconstruction is a documented geometry convention compatible with ProteinMPNN-style backbone features. Its constants and provenance are explicit in the API documentation and third-party notice.

See `docs/provenance-ledger.csv` for the per-file migration decision.

The public message-passing contract is documented in `docs/model-contracts.md`.
Gate identity and provenance contracts are documented in `docs/gate-registry.md`.
Evaluation and fixed-target contracts are documented in `docs/evaluation-contract.md` and `docs/kinetic-reference.md`.
The release and paper-code checks are listed in `docs/release-checklist.md`.

## Scope boundary

AlloGate gates measure model-internal dependence on structural information. A closed gate is not equivalent to a physical mutation and does not by itself establish molecular causality.
