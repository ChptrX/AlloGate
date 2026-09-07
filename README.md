# AlloGate

[![Tests](https://github.com/ChptrX/AlloGate/actions/workflows/test.yml/badge.svg)](https://github.com/ChptrX/AlloGate/actions/workflows/test.yml)
[![Public content audit](https://github.com/ChptrX/AlloGate/actions/workflows/privacy.yml/badge.svg)](https://github.com/ChptrX/AlloGate/actions/workflows/privacy.yml)
[![Package smoke test](https://github.com/ChptrX/AlloGate/actions/workflows/package.yml/badge.svg)](https://github.com/ChptrX/AlloGate/actions/workflows/package.yml)

> **Status: trainable public alpha.** The protein-independent model, configurable
> slow-CV training objective, Gate evaluation layer, and synthetic reference
> workflows are implemented. Raw trajectory readers and kinetic-reference fitting
> remain separate adapters/workflows.

AlloGate learns descriptor-free slow collective variables from molecular-dynamics
graphs and audits how structural information influences a frozen kinetic target.
The public package contains no protein-specific residue ranges, trajectory paths,
checkpoints, or server assumptions.

## Protein-independent hierarchy

The model uses explicit, variable-size mappings:

```text
residues
   |
   v
local structural units ---- contact groups ----> interfaces
   |                                             |
   +---------------------+-----------------------+
                         v
               global structural units
                         |
                         v
              Direct and Route CVs
```

- **Local structural units** are short, configurable residue groups that never
  cross a chain or global-unit boundary.
- **Interfaces** aggregate contact messages between local units and connect
  exactly two global structural units. Inactive interfaces remain exact zeros.
- **Global structural units** are investigator-declared, non-overlapping residue
  groups such as domains, subdomains, chains, or functional regions.

Counts are inferred from each batch. The model has no fixed number of residues,
local units, interfaces, global units, replicas, trajectories, GPUs, or CVs.

## Installation

```text
python -m pip install -e ".[training,test]"
python -m pytest
```

PyTorch is optional for the configuration, manifest, Registry, and artifact
modules. Install `.[training]`, `.[gnn]`, or `.[evaluation]` for the corresponding
neural functionality.

## Configurable training

Every model, objective, and optimizer parameter is represented in a strict,
path-free experiment configuration. Start from
`examples/configs/protein_training.yaml` and change the hierarchy, channel
dimensions, lag, loss weights, optimizer, batch size, stopping policy, or dtype.
Unknown keys fail rather than being silently ignored.

The training API accepts lagged, variable-size graph batches, so trajectory
reading can be supplied by MDAnalysis, MDTraj, an existing preprocessing
pipeline, or another explicit adapter:

```python
from allogate.config import load_experiment_config
from allogate.models import AlloGateGeometryModel
from allogate.training import SlowCVTrainer

config = load_experiment_config("experiment.yaml")
model = AlloGateGeometryModel.from_experiment_config(config)
trainer = SlowCVTrainer(model, config, device="cuda")
result = trainer.fit(train_loader, validation_loader, output_directory="run")
```

`train_loader` and `validation_loader` yield `LaggedBatch` objects. Use
`collate_lagged_batches` as a PyTorch `DataLoader` collation function. See
`docs/training.md` and `docs/data-contract.md` for the complete contract and
leakage-safe split guidance.

Run the deterministic synthetic training workflow:

```text
python examples/synthetic_training.py --output synthetic-training-run
```

The run writes an immutable best checkpoint, epoch history, and a path-free run
manifest binding the experiment, initial model, and trained model digests.

## Gate evaluation

The public evaluation layer provides canonical State, Relay, and Contact Gate
identities; fixed-denominator interventions; exact Jacobians and named Hessian
blocks; deterministic dose schedules; and content-addressed results.

```text
python examples/synthetic_gate_audit.py
```

Closing a Gate removes model information without renormalizing surviving
neighbors. A Gate response measures dependence inside the trained model; it is
not equivalent to a physical mutation and does not by itself establish molecular
causality.

## Scientific basis and reproducibility

The geometric encoder follows the scalar/vector constraints of geometric vector
perceptrons. Descriptor-free GNN collective variables and the DeepTICA-style
slow-mode objective follow the public methodology described by Zhang et al.,
*J. Chem. Theory Comput.* 20, 10787-10797 (2024),
<https://doi.org/10.1021/acs.jctc.4c01197>.

For a shareable result, retain:

- the trajectory and lag-pair manifests;
- the experiment configuration and its digest;
- initial and trained model digests;
- the split policy and random seed;
- the Gate Registry, layout, kinetic-reference, and no-intervention digests;
- software version, environment lock file, and raw-data checksums.

Implementation lineage is recorded in `THIRD_PARTY_NOTICES.md` and
`docs/provenance-ledger.csv`. Release and paper-code checks are listed in
`docs/release-checklist.md`; user-visible API changes are recorded in
`CHANGELOG.md`.

## Public-data boundary

This repository intentionally excludes study structures, trajectories,
checkpoints, absolute host paths, scheduler configuration, legacy source
bundles, and unpublished residue identities. Only synthetic or explicitly
redistributable examples should be committed.
