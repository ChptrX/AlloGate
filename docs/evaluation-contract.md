# Gate evaluation contract

## One immutable identity

Every evaluation service binds four semantic digests before it captures a baseline:

- the Gate Registry;
- the derived global tensor layout;
- the frozen model or model ensemble;
- the fixed kinetic target.

The resulting `EvaluationIdentity` excludes paths, devices, batch sizes, display labels, and scheduler settings. Requests carrying another identity are rejected rather than silently reinterpreted.

## No-intervention reference

The service constructs the full Gate vector as exact float64 ones and evaluates it once to capture `q0`. Every later finite intervention reports `delta_q = q(g) - q0` against that same read-only reference. `replay_no_intervention()` is an explicit drift check; it never replaces `q0`.

The feature baselines used by State Gates and this target-space `q0` serve different purposes and are never interchangeable.

## Derivatives

Jacobian columns and Hessian rows are returned in canonical Registry order, even if a request uses aliases or a different ordering. Hessians are computed only for named requested blocks. The service records the maximum raw asymmetry and stores the symmetrized block.

All second derivatives are exact automatic derivatives of the supplied PyTorch response. The service does not silently switch to finite differences or a step-specific evaluator.

## Finite doses

`DoseSchedule` creates deterministic single-Gate requests with values in `[0, 1]`. A dose response measures dependence of the fixed model/target pair on an internal intervention; it is not a physical perturbation claim. Multi-Gate combinations should be declared explicitly rather than inferred from single-Gate results.
