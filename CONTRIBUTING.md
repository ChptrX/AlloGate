# Contributing to AlloGate

All public contributions must preserve the repository's clean-room and provenance boundary.

## Before opening a change

- Do not add private structures, trajectories, checkpoints, host paths, credentials, scheduler files, or legacy source bundles.
- Do not copy or lightly modify third-party implementations. Implement from a stated public specification and record known scientific or source lineage.
- Add or update a row in `docs/provenance-ledger.csv` for every new public implementation module.
- Keep Gate identities independent of runtime tensor positions, filenames, devices, folds, and display labels.
- Preserve exact all-one no-intervention semantics and fixed aggregation denominators.

## Local checks

```text
python -m pip install -e ".[test]"
python -m pytest
python tools/public_content_audit.py
python examples/synthetic_gate_audit.py
```

The internal upstream-similarity audit uses separately quarantined reference trees and is run by maintainers before release. Those trees must never be committed.

## Reporting provenance concerns

If a contribution may reproduce third-party code, identify the file and likely source in the pull request. Uncertain code remains quarantined until its license and lineage are resolved.
