# Third-party notices and implementation provenance

AlloGate's newly written public source is intended for distribution under Apache-2.0. The following projects and papers are credited because they define, demonstrate, or occur in the known implementation lineage of the geometric features used by AlloGate.

No source tree from these projects is vendored in this repository.

## Geometric Vector Perceptron

Scientific definition:

- Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael J. L. Townshend, and Ron O. Dror. “Learning from Protein Structure with Geometric Vector Perceptrons.” ICLR 2021. <https://arxiv.org/abs/2009.01411>

Known open implementation lineage reviewed during the AlloGate provenance audit:

- GVP-PyTorch, Bowen Jing and contributors: <https://github.com/drorlab/gvp-pytorch> — MIT License.
- Geometric GNN Dojo, Chaitanya K. Joshi, Simon V. Mathis, and contributors: <https://github.com/chaitjo/geometric-gnn-dojo> — MIT License.
- mlcolvar, Luigi Bonati, Enrico Trizio, Andrea Rizzi, and contributors: <https://github.com/luigibonati/mlcolvar> — MIT License.
- Jintu Zhang's mlcolvar fork and graph extensions: <https://github.com/jintuzhang/mlcolvar> — MIT License.
- GNN-CV workflow repository: <https://github.com/jintuzhang/gnncv> — MIT License.

The public AlloGate implementation in `src/allogate/models/gvp.py` was newly organized from the published mathematical constraints and uses different source structure and identifiers. These credits are intentionally retained because earlier private AlloGate development may have been informed by the implementation family above.

## Virtual beta carbon

The coefficient convention used by `virtual_beta_carbon` is compatible with the virtual-C-beta construction in:

- Justas Dauparas et al. “Robust deep learning–based protein sequence design using ProteinMPNN.” *Science* 378, 49–56 (2022). DOI: <https://doi.org/10.1126/science.add2187>.
- ProteinMPNN source, Justas Dauparas and contributors: <https://github.com/dauparas/ProteinMPNN> — MIT License.

AlloGate contains a newly written NumPy implementation with validation and explicit named constants. This notice preserves the known formula/source provenance.

## Optional validation software

Future AlloGate releases may offer deeptime as an optional, separately installed reference backend. deeptime is not copied or bundled here:

- deeptime: <https://github.com/deeptime-ml/deeptime> — LGPL-3.0.

PyEMMA and MSMBuilder may be cited in method documentation as historical software references. They are not dependencies and no source from them is included:

- PyEMMA: <https://github.com/markovmodel/PyEMMA> — LGPL-3.0.
- MSMBuilder: <https://github.com/msmbuilder/msmbuilder> — LGPL-2.1.

