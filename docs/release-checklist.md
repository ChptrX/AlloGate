# Public release checklist

## Required human sign-off

- Confirm the copyright owner and authority to publish the newly written code under Apache-2.0.
- Confirm that the author list and ORCID identifiers in `CITATION.cff` are complete and ordered correctly.
- Add the final public repository URL and, when available, the method-paper DOI.
- Confirm that every acknowledged implementation lineage is represented in `THIRD_PARTY_NOTICES.md`.

## Repository gate

- Start from the blank-history `main` branch; do not import legacy Git history.
- Confirm that `.audit/`, archives, wheels, binary trajectories, checkpoints, and build products are ignored.
- Run all unit tests and the synthetic end-to-end example.
- Run `tools/public_content_audit.py` over the public tree.
- Run the internal normalized-function and continuous-block similarity audit against the reviewed upstream snapshots.
- Build both wheel and source distribution from a clean checkout and install each into a fresh environment.
- Audit the wheel and source distribution without extraction; reject unsafe paths, unexpected payloads, and private content.
- Verify that both release artifacts include `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## GitHub gate

- Require the test, package-smoke, and public-content workflows on pull requests.
- Enable private security advisories and secret scanning where available.
- Protect `main` from force pushes and require review for release changes.
- Create the first release as a pre-release until the public training and kinetic-reference fitting workflows are complete.

## Paper-code gate

- Pin the release tag and archive DOI used by the manuscript.
- Record exact input-manifest, configuration, model, Registry, layout, kinetic-reference, and no-intervention digests for every reported result.
- Publish only redistributable synthetic or explicitly cleared example data.
- State that Gate interventions measure model dependence and do not alone establish physical causality.
