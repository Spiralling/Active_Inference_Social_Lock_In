# IWAI 2026 camera-ready LaTeX project

## Manuscript

**Paradigm Shifts: Theory Revision and Belief Sharing in Epistemic Communities**

`main.tex` contains the main text, bibliography, and all six appendices. `figures/`
contains the nine supplied vector PDF assets used in eight numbered figures.
The two constant-world plots are included together in Figure 7.

The compiled paper has 25 pages: 11 main-text pages, 2 reference pages, and 12
appendix pages. It uses the standard `llncs` class without changing page geometry
or reducing the body font size or line spacing.

## Build

Use a TeX installation with the LNCS class and the standard packages named in
`main.tex`, including `latexmk`.

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The figures and bibliography are local. Building the manuscript does not require
Python, the simulation engine, an internet connection, or the omitted raw data.

## Optional checks

```sh
python -m pip install numpy scipy pymupdf
python checks/validate_manuscript.py
```

Run these checks after compiling. They verify algebraic identities, transcription
of the archived summary data, figure checksums, cross-references, and page counts.
They do **not** rerun the experiments or reconstruct the historical bootstrap
curves. The recorded checks passed 48/48; see `checks/validation_results.json`.

## Scientific scope

This version uses the completed four-variable combustion assessment from
`IWAI_camera_ready_handoff_2026-09-19.zip` (scientific source revision `a869664`).
It excludes the subsequently proposed belief-separation experiments and their
finite model-score-memory configuration. Belief utility remains absent.

`provenance/` contains unchanged source and summary snapshots from the supplied
handoff. These support checking the manuscript's numerical specifications and
reported summaries; they are not a standalone copy of the simulation engine.
To rerun the complete experiment suite, use the original handoff and its recorded
software environment. Reconstructing the archived figures also requires the
companion raw-data archive, `IWAI_camera_ready_data_2026-09-19.zip`, which was not
supplied in this conversation. No new simulation was performed for this revision.

## Required author actions before upload

Replace the affiliation and email placeholders in `main.tex` with the authors'
confirmed details, then recompile. Match the submission metadata to the revised
title and abstract. Add the completed and signed Springer copyright form to the
submission ZIP. No author signature has been supplied or created.

IWAI's official instructions, checked 19 September 2026, allow 12 main-text pages
excluding references and 12 appendix pages including additional references. They
request the final PDF and a source ZIP containing the signed copyright form and
list 20 September 2026 as the camera-ready deadline.

Source: https://iwaiworkshop.github.io/ (Submissions / Camera-ready / Important dates).

See `REVISION_NOTES.md` for the substantive changes and `main_changes.diff` for the
comparison with the preceding figures-updated manuscript. No font files are
included in this project; the PDF figures retain their embedded fonts.
