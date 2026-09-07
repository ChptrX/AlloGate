# Configurable training workflow

## Scope

The public trainer learns slow collective variables from lagged geometric graph
pairs. It implements a DeepTICA-style variational objective without assuming a
specific protein, residue count, domain list, file layout, GPU count, or
trajectory naming convention.

Trajectory decoding is deliberately an adapter boundary. Convert frames with
MDAnalysis, MDTraj, or an existing simulation pipeline into the tensor contract
in `data-contract.md`; the scientific model and trainer then remain identical
across proteins and trajectory formats.

## Experiment configuration

`ExperimentConfig` has three path-free sections:

- `method`: geometry, hierarchy, channel counts, message layers, readout width,
  and CV dimension;
- `objective`: lag identity, covariance regularization, variational score
  weights, alignment, variance, decorrelation, and trajectory-centering terms;
- `optimization`: seed, epochs, batch size, learning rate, weight decay,
  clipping, scheduling, stopping, dtype, and deterministic-algorithm policy.

Start with `examples/configs/protein_training.yaml`. JSON and YAML are accepted.
All mappings are strict: a misspelled or unknown key fails at load time. Local
paths and device choices are not stored in the scientific configuration.

## Leakage-safe data preparation

1. Define each source trajectory once in `TrajectoryManifest`.
2. Assign complete trajectories, not frames, to train, validation, and test.
3. If correlated replicas belong to one simulation group, split by `group_id`
   rather than by trajectory.
4. Build lag pairs with `build_lagged_pair_manifest`. It rejects lags that are
   not an integer number of stored frames and never crosses a trajectory.
5. Fit normalization statistics and any State feature baseline on training data
   only.
6. Do not use test trajectories for early stopping, hyperparameter selection,
   interface definition, state definition, or lag selection.

The configured lag is part of the scientific digest. The pair manifest stores
the same lag in picoseconds; a release workflow should assert that the converted
values agree before training.

## Dataset and batching API

A dataset item may contain one or more lagged pairs as a `LaggedBatch`. Its two
`ModelBatch` values contain the graph at `t` and `t + lag`, and an optional
trajectory label vector supports the centered objective.

For ordinary PyTorch datasets:

```python
from torch.utils.data import DataLoader
from allogate.training import collate_lagged_batches

train_loader = DataLoader(
    train_dataset,
    batch_size=config.optimization.batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_lagged_batches,
)
```

Use a deterministic generator/worker initialization when shuffling. Every
kinetic batch must contain at least two pairs; larger covariance batches are
usually preferable. The final batch may be smaller than the configured batch
size, but it may never be larger.

## Model construction and fitting

```python
from allogate.config import load_experiment_config
from allogate.models import AlloGateGeometryModel
from allogate.training import SlowCVTrainer

config = load_experiment_config("experiment.yaml")
model = AlloGateGeometryModel.from_experiment_config(config)
trainer = SlowCVTrainer(model, config, device="cuda:0")
result = trainer.fit(
    train_loader,
    validation_loader,
    output_directory="runs/protein-a",
)
```

The seeded factory is important: the run identity binds the initial model
digest as well as the path-free experiment digest. If a pretrained model is
supplied instead, its exact initial state is still captured in the run.

The trainer refuses a non-empty output directory. A successful run writes:

| Artifact | Meaning |
|---|---|
| `best-model.pt` | best validation checkpoint plus complete scientific config |
| `history.json` | per-epoch train/validation metrics and learning rate |
| `run.json` | path-free run identity and initial/final model digests |

PyTorch checkpoints use PyTorch serialization. Load only checkpoints from a
trusted source and prefer `weights_only=True`.

## Objective

For Direct coordinates `d` and Interface Route coordinates `r`, the default loss
minimizes:

```text
- w_route * VAMP2(r_t, r_t+lag)
- w_direct * VAMP2(d_t, d_t+lag)
+ w_align * canonical_alignment_penalty(d, r)
+ w_std * variance_floor_penalty(d, r)
+ w_corr * decorrelation_penalty(d, r)
```

An optional trajectory-centered copy of the same objective can be added with an
independent weight. Covariance statistics and linear solves use float64.
Geometric forwards use the configured float32 or float64 dtype.

Report the complete loss definition and weights; a VAMP-2 score alone is not a
kinetic validation.

## Validation and reporting

Before interpreting a trained CV:

- repeat training across declared seeds or folds;
- inspect covariance conditioning and CV collapse;
- compare implied timescales over multiple lags;
- test held-out trajectory/replica stability;
- record orientation/sign conventions;
- construct and qualify the fixed kinetic reference separately;
- evaluate Gates only after freezing both model and reference.

Software tests demonstrate tensor, symmetry, differentiation, and artifact
contracts. They do not establish convergence or biological validity for a new
protein.
