# IWAI communication experiment handoff

Start with [RESULTS.md](RESULTS.md) and [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md). Browse [figures/figure_pack.pdf](figures/figure_pack.pdf) for the captioned figure collection. No manuscript source was edited by this task.

This is the completed implementation of `reference/IWAI_Experiment_Implementation_Brief.md`: four changed-signal network arms, seven constant-world camp conditions, and a fixed state-1,500 intervention with two paired continuations. All prescribed seeds and nonseparating outcomes are retained. The model uses the original revised combustion candidate family, with no belief utility, reliability gates or model-score decay.

## Contents

- `figures/`: publication PDFs, SVGs, PNG previews, exact captions, copyable LaTeX blocks, generating scripts, vendored orx style and font records.
- `analysis/`: all 1,100 seed/window rows, exact event/censoring records, network factorial and branch paired effects, forecast comparisons, complete summaries and portable plot inputs.
- `protocol.json`: frozen conditions, seeds, timing, endpoints, windows and display choices.
- `METHODS.md`: general model first, Gaussian details later, and pipeline pseudocode linked to equations.
- `run.py`: actual-engine production wrapper, with independent channel switches and exact checkpoints. The original engine is included at `../four_variable_experiments/engine`, and the original world compiler at `../combustion_experiments/world.py`.
- `VALIDATION.md`, `INDEPENDENT_*.json`, `ANALYSIS_CHECK.json`: independent mathematical, raw-data, archive and statistical checks, with their scripts and logs.
- `reference/`: supplied brief/recommendations and camera-ready source provenance. The recommendations' external pilot trajectories were unavailable and are not treated as validated production runs.
- `RUNNER_SCHEMA.md`: array axes, aliases, deterministic covariance storage and random-stream reconstruction.
- `EXECUTION_PROVENANCE.json`, `SOURCE_PROVENANCE.json`, `requirements.lock.txt`: source and environment identities.

The compact handoff's top-level `data_provenance/` contains every raw chunk/stream hash and run record. `raw_examples/` contains exact representative chunks and their precision histories. The optional complete-data ZIP adds all 1,670 raw chunks, 167 precision chunks and the three original network regression arrays. Extract both ZIPs into the same parent directory to combine them; their common package root then contains `raw_data/` and `legacy_network/`.

## Regenerate the figures

Run these commands from the extracted **package root**, which contains `paper/`:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r paper/communication_experiments/requirements.lock.txt
.venv/bin/python paper/communication_experiments/figures/plot_figures.py
.venv/bin/python paper/communication_experiments/figures/validate_figures.py
```

The recorded environment is Python 3.12.9, JAX/jaxlib 0.9.0, NumPy 2.3.4, SciPy 1.16.3 and Matplotlib 3.10.7 on macOS arm64 CPU. Figures require a working LaTeX installation with Computer Modern/CM-Super fonts; the original used TeX Live 2025. The renderer invokes `pdflatex` to build the captioned pack. On this Mac it recognizes `/Library/TeX/texbin`; elsewhere it also uses the normal executable search path. No simulation is needed to regenerate figures from `analysis/plot_data.npz`.

The figure renderer and validator were tested on a fresh extraction of the final handoff. See top-level `PACKAGE_VALIDATION.json` beside the ZIP for the tested archive identity and checks; it is outside the ZIP to avoid a self-referential archive hash.

## Tests and analysis from complete raw data

```bash
.venv/bin/python paper/communication_experiments/independent_tests.py
.venv/bin/python paper/communication_experiments/test_analysis.py
.venv/bin/python paper/communication_experiments/analyze.py --raw raw_data
.venv/bin/python paper/communication_experiments/independent_analysis_check.py raw_data
.venv/bin/python paper/communication_experiments/audit.py raw_data --archive legacy_network --output paper/communication_experiments/INDEPENDENT_RAW_AUDIT.json
.venv/bin/python paper/communication_experiments/compare_independent_analysis.py
```

The last command compares the independent raw-audit reductions with the publication analysis. `audit.py` can omit `--archive` if the old network arrays are not present; it then explicitly marks archive regression as not requested. The final delivered audit did include those arrays and found exact equality for all three arms. Neither the compact examples nor `data_provenance/` are a substitute for `raw_data/` in these full-data commands.

These entry points are real scripts included in the package. The simulation and analysis do not fetch external data. The independent stationary calculation is separately runnable:

```bash
.venv/bin/python paper/communication_experiments/stationary_predictions.py
```

## Repeat production through orx

The frozen experiment is `a86619b3-3074-4305-a939-7df683956319` in project `49b2c5d7-2a4b-4c60-8487-589091edec33`; it was launched with `orx exp run ... --backend local`. Its immutable producer commit is `139225df25e641d387f4ea5e020e54203fe32c6b`. The original project's fixed command calls `paper/four_variable_experiments/run.py`, whose committed dispatch selects this new suite.

For a new imported orx project made from this handoff, use the actual producer entry point `python paper/communication_experiments/run.py` as its fixed run command, commit the source/protocol, and launch with `orx exp run <experiment-id> --backend local`. Project/experiment IDs are local records; the existing IDs are provenance, not portable credentials or universal identifiers. Set up the pinned environment before recording that command. The default output is `paper/communication_experiments/runs/confirmation-2026-09-20/`; the producer refuses to overwrite an existing run directory.

Preserve the frozen producer when making a scientific change. Put any further setting changes on a new experiment branch and label them exploratory unless independently confirmed. All optional studies listed as deferred in the brief remain unrun.

## Scientific cautions for the writing agent

Use the narrow claims in `RESULTS.md`; in particular, separating camps have private model weights and restricted conditional exchange, the variance control removes the effect, and no-communication camps also separate. The fixed-time intervention leaves conditional beliefs and local scores identical across branches. Candidate choice, predictive accuracy, log-odds differences and entropy are separate readouts.

All figures use real LaTeX fonts but are numerical Matplotlib plots, not TikZ. Existing graphical-abstract/TikZ assets belong to the prior manuscript handoff and need no change for this suite. No public repository or DOI was created, and no bibliography claims were added. Historical engine comments should be interpreted using `LEGACY_ENGINE_NOTICE.md`.
