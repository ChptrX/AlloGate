# First public migration decisions

## GVP decision

The legacy `gvp.py` file is quarantined and has not been copied into the public tree. The public module is a newly structured implementation of the scalar/vector constraints described by Jing et al. It uses explicit channel matrices and equations rather than carrying forward the legacy class hierarchy or message-passing code.

Because the provenance audit showed a meaningful relationship to the mLcolvar/Geometric-GNN-Dojo/GVP-PyTorch implementation family, AlloGate retains those projects in `THIRD_PARTY_NOTICES.md`. “Newly implemented” is not used to claim that the mathematical architecture was invented by AlloGate.

## Virtual-C-beta decision

The legacy geometry file is not copied. The public implementation exposes named constants, validates shapes and finite values, and documents compatibility with the ProteinMPNN coefficient convention. ProteinMPNN and its paper remain credited.

## First whitelist

Approved in this batch:

- GVP scalar/vector primitive;
- virtual/observed beta-carbon selection, cosine switching, and Gaussian radial bases;
- canonical scientific-configuration hashing;
- generic method schema without runtime paths;
- trajectory CSV manifest with relative-path enforcement;
- generic structural-unit ranges with overlap validation.

Explicitly deferred:

- legacy message-passing and multiscale model assembly;
- training and checkpoint loading;
- TICA, MSM, committor, and TPT workflows;
- Gate Registry, Gate Evaluation Engine, Hessian, and dose response;
- artifact storage and public export;
- every compatibility adapter, vendor package, server helper, production configuration, and study-specific example.

## Second whitelist

Approved in this batch:

- deterministic, within-graph contact construction;
- residue/local-element/structural-unit tensor contracts with variable graph sizes;
- configurable contiguous local-element construction that cannot cross chain or structural-unit boundaries;
- GVP relation messages with fixed all-one denominators;
- separately retained contact and covalent structural-unit messages;
- shared, zero-anchored Route readout and multiscale Direct readout;
- second-derivative-safe tensor aggregation based on indexed addition.

The legacy model, hierarchy, CSR helper, graph cache, and readout files were not copied. Public modules use new type contracts, names, validation, and composition boundaries. The GVP scientific and implementation lineage already recorded for the first batch also applies to the equivariant message-passing layer.

Still deferred after the second batch:

- data loaders for molecular trajectory formats;
- slow-CV objectives and training orchestration;
- kinetic reference, committor, and TPT;
- formal State/Relay/Contact Gate identities and provenance registry;
- finite-dose intervention, Hessian atlas, artifact store, and workflows.

## Third whitelist

Approved in this batch:

- stable semantic identities for residue, contact, local-element, interface, structural-unit, relation, and readout entities;
- State, Relay, and Contact Gate specifications whose UIDs do not depend on display names or runtime tensor indices;
- deterministic Registry ordering and global tensor layout;
- explicit provenance DAG validation, ancestry, descendants, and Registry cross-checking;
- runtime alignment of global Gate tensors to repeated entities in variable-size graph batches;
- State Gate replacement with frozen scalar feature baselines and zero vector baseline;
- frozen State insertion points at residue, local-element, and structural-unit scales.

The old Gate identity, Registry, provenance, and model-gating modules were not copied. The public implementation uses different schemas, UID format, class structure, dependency boundary, and runtime binding design. In particular, the Registry and provenance layer use only the Python standard library and remain importable without PyTorch.

Still deferred after the third batch:

- calibrated baseline estimation and storage;
- immutable no-intervention kinetic target identity;
- unified Gate evaluation service, Jacobian, and Hessian scheduling;
- dose-response plans and out-of-distribution checks;
- content-addressed artifacts, public export, and evidence bundles.

## Fourth whitelist

Approved in this combined batch:

- a portable fixed kinetic-reference object and matching NumPy/PyTorch projection;
- semantic identities binding Registry, tensor layout, frozen model, and fixed target;
- one all-one `q0` capture reused by every finite intervention;
- one evaluation service for order-zero response, exact Jacobian, and named exact Hessian blocks;
- deterministic single-Gate dose schedules restricted to interpolation values in `[0, 1]`;
- a small content-addressed artifact store with path-free logical references.

The old evaluation, automatic-differentiation, kinetic-reference, and artifact-store files were not copied. Public modules use new standard-library/dataclass contracts, canonical Registry UIDs, read-only NumPy result records, a single PyTorch reverse-mode implementation, and no legacy adapter or workflow numbering.

Still deferred after the fourth batch:

- fitting slow collective variables and kinetic references;
- trajectory-format readers and streaming training datasets;
- state discretization, transition-model estimation, boundary selection, and kinetic validation;
- calibrated State feature-baseline estimation;
- multi-Gate design, uncertainty analysis, public examples, and complete evidence-bundle export;
- every legacy compatibility adapter, production configuration, scheduler helper, and study-specific asset.
