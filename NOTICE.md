# Source and scope

The experiment and plotting functions are extracted from this repository's
combustion assessment, communication-channel study and matched Gaussian study.
Scientific operation order, fixed configurations, seeds, intervention states
and plotted estimands are preserved. This branch removes unrelated historical
features and consolidates the figure-producing dependencies.

The shared `src/iwai_figures/plotting/orx_figstyle.py` is vendored from the installed
orx academic plotting skill. The local style adapter applies the manuscript's
LaTeX Computer Modern fonts and preserves the existing figure layouts.

The PDFs under `reference/figures/` were supplied in the two paper archives.
Their archive and file hashes are recorded in `reference/archives.json`; they
serve as verification targets. Source protocols are retained under `reference/`
for provenance. Their historical paths describe the source experiments and are
not runtime imports or required directories in this branch.

This collaborator edition documents the figures in the two supplied drafts.
It does not reproduce every quantitative statement or control mentioned only in
their prose, and it does not select the figures or claims for a final manuscript.
