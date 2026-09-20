# Matched Gaussian comparison and revised figure pack

This extension adds plain and socially gated single-Gaussian baselines to the completed communication experiment. It preserves the original candidate runs, underlying engine and combustion references. Read `RESULTS.md` for the measured outcomes and `METHODS.md` for equations, reference constructions, update timing, readouts and pseudocode. The results report is generated after the production run and independent audit.

The revised `figures/figure_pack.pdf` contains the original seven communication figures plus the new main comparison and sensitivity figure. The original figure assets remain in `../communication_experiments/figures/`. New figures use the same orx style and actual LaTeX Computer Modern fonts, with vector PDF/SVG assets and PNG previews. Captions and LaTeX integration blocks are included.

## Exact design

Four new arms cross plain/gated sharing with two Gaussian reference constructions: observable information pooling and moment matching of the equal candidate mixture. Each uses 16 agents, the same gas/mass observation groups, a constant signal, retention .3, base cross-group influence .05, observation noise standard deviation .5, 5,000 updates and seeds 10000–10049. All Gaussian agents have the same reference and initial state within an arm. The gate uses inherited parameter .01 and reweights the weighted graph before learning.

Existing `C10`, `C11` and `C10_gasvar_matched` candidate trajectories are reused with their provenance and exactly paired observation streams. They are not new independent confirmations. Fifty paired streams are the replication unit; four new arms do not constitute 200 independent replications.

`protocol.json` fixes the scientific settings and analysis before these new production outcomes. `SOURCE_PROVENANCE.json` identifies the parent sources. `RUNNER_SCHEMA.md` documents the lossless full-state checkpoints and trajectory arrays. `STATIONARY_PREDICTIONS.json` and `INDEPENDENT_DESIGN_AUDIT.md` contain an independent mathematical forecast made before the new trajectories were examined.

## Reproduce

Run commands from the handoff root containing `paper/`. Use the pinned environment in `requirements.txt`, Python 3.12.9, and a working LaTeX installation including Computer Modern/T1 fonts. The recorded execution uses macOS arm64 CPU with JAX float64. Last-bit agreement on other numerical backends is not guaranteed.

```bash
python paper/gaussian_comparison/independent_tests.py
python paper/gaussian_comparison/analyze.py --raw gaussian_raw_data --candidate-raw raw_data
python paper/gaussian_comparison/figures/plot_figures.py
python paper/gaussian_comparison/figures/validate_figures.py
```

Figure regeneration uses the included portable analysis inputs; raw arrays are needed only to repeat raw-data reduction. The optional Gaussian raw-data archive extracts `gaussian_raw_data/`. The earlier complete-data archive supplies `raw_data/` for the candidate runs. Keep those raw sources distinct. The scripts accept explicit raw paths, so differently named extraction roots are also usable.

The producer entry point is `python paper/gaussian_comparison/run.py`. Its output directory must not already exist. In the existing orx project the inherited fixed command still calls `paper/four_variable_experiments/run.py`; the child branch's committed dispatch selects this new suite. Production is launched using `orx exp run <experiment-id> --backend local`, from a committed source snapshot. For a new imported project, record the producer entry point as its fixed command. Local experiment IDs are provenance records, not portable credentials.

The source branch is `orx/matched-gaussian-camp-comparison`, a child of the completed communication experiment. Scientific source changes after a completed run belong on a new child; reporting and packaging changes are tracked separately. No manuscript or upstream repository is edited by this handoff.

## Interpretation

These are comparisons of specified mechanisms and priors, not a pure candidate-count ablation. A Gaussian cannot retain a general two-component predictive distribution. The primary distance uses observable mixture means; the MAP distance is secondary. Continual selective observations and retained references can sustain numerical disagreement even without a gate or alternative candidates. Candidate probabilities separately determine whether agents favor opposite theories. The gas-variance control, model-weight-sharing control and full predictive losses must remain visible when discussing separation.
