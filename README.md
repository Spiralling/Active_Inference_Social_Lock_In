# IWAI figure experiments: collaborator edition

This branch isolates the code behind **16 unique figure assets** in
`IWAI2026_camera_ready_source.zip` and `IWAI2026_figure_layout_source.zip`.
It is a guide to those two drafts, not the final paper's reproduction release.
The drafts contain different experiments. In particular, the earlier and later
camp-separation figures must not be treated as interchangeable plots.

Start with **[SUMMARY.md](SUMMARY.md)** for the branch contents, model explanation,
and a setup/results/interpretation section for every numbered figure. The exact
paper-to-file mapping is in [reference/figure_inventory.json](reference/figure_inventory.json).

## Setup

Use Python 3.12 and the pinned scientific packages:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Rendering also requires a TeX installation with `pdflatex`, `latex`, `dvipng`,
TikZ, `geometry`, `amsmath`, T1 Computer Modern/CM-Super fonts and `type1cm`.
The figures use real LaTeX text. The original verification environment is recorded
in `checks/`; other numerical backends can differ in final floating-point bits.
There is no runtime dependency on orx, the earlier repository tree, or an online service.

## Redraw the supplied figures

```bash
python reproduce.py figures
python reproduce.py compare
```

These commands read the compact, unsmoothed per-seed data in `data/`, write PDF,
SVG and PNG files under `build/figures/`, and compare the PDFs against the exact
archive targets in `reference/figures/`. Comparison checks the entire page at
216 dpi, text and dimensions; PDF byte equality is reported separately because
container metadata may differ. The reference PDFs are never drawing inputs.

## Run the experiments and redraw from fresh results

```bash
python reproduce.py simulate
python reproduce.py reduce
python reproduce.py figures --data build/recomputed_data --output build/recomputed_figures
python reproduce.py compare --figures build/recomputed_figures --report build/recomputed_comparison.json
```

The first command runs the fixed seed lists and all conditions required by the
displayed figures. It writes full numerical histories under `runs/full/` and
refuses to overwrite an existing family output. Use `--output runs/another_run`
for another run, and pass the same root to `reduce --raw runs/another_run`.
Allow several minutes of CPU work and several GB of disk space. Actual measured
durations for this extraction are in the validation records; they are not a
hardware-independent runtime promise.

To run one complete family, use `simulate --family camera`, `communication`, or
`gaussian`. Reducing all figures requires all three families. No seeds are
selected after observing outcomes, and no new hyperparameters are fitted.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests include real archived trajectory fixtures and independent mathematical
checks. `checks/` records the extraction's full-run and figure comparisons.
`scripts/compare_archived_runs.py` additionally compares complete raw arrays
when the old run directories are available; they are not required to redraw or
rerun the experiments from this branch.

`runs/` and `build/` are ignored. Tracked data contain only the per-seed inputs
needed to draw the supplied figures; large raw histories, unrelated notebooks,
prototype experiments, manuscript prose and previous handoff packages are absent.
