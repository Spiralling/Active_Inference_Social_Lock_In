# Four-variable publication experiments

**[Open the captioned figure review](figures/figure_review.pdf)**. See [results and paper recommendations](RESULTS.md) and [coverage of the accepted paper's figures](COVERAGE.md).

The proposed interpretable model has now been implemented and assessed in **29 conditions**. The cycle and network conditions use 50 noise seeds each; disagreement and adaptation conditions use 20. A single fixed protocol was executed through local orx, without changing parameters after inspecting its outcomes. The run completed in 198 seconds, excluding setup and figure generation.

The compact model preserves the cycle and network phenomena. The mixed-world disagreement experiment yields a transient split followed by incumbent consensus, rather than persistent opposing paradigms. In the candidate-consistent adaptation control, full communication speeds revision and improves predictive accuracy relative to isolated agents. These are new results from a new world, not exact reproductions of the old ten-variable experiments.

## Figures

| File | Content |
| --- | --- |
| [Figure 1](figures/fig01_model.pdf) | Actual implemented four-variable candidate networks, with explicit P and O |
| [Figure 3](figures/fig03_walls.pdf) | Mean disagreement together with both camps' theory probabilities |
| [Figure 4](figures/fig04_cycle.pdf) | Transition and posterior uncertainty, with an unchanged-world control |
| [Figure 9](figures/fig09_topology.pdf) | Original graph-row composition, newly simulated trajectories |
| [Complete figure review](figures/figure_review.pdf) | Main candidates and supplementary memory, communication, attention, prediction and long-horizon checks |

All plots use actual LaTeX Computer Modern/CM-Super fonts and orx academic plotting guidance at 4.8-inch LNCS width. PDF and SVG are vector outputs; PNGs are inspection previews. Ribbons are pointwise 95% percentile-bootstrap intervals for the across-seed mean, with 2,000 whole-seed resamples. Agents within a seed are not counted as independent replicates. Captions identify distinct windows or summaries used in supplementary panels.

The historical utility, graph-editing/proposal and retrospective-reliability figures are outside this candidate-inference suite. The [coverage audit](COVERAGE.md) explains why they must not be presented as rerun results. No existing manuscript source or old simulation output was overwritten.

## Implemented model

Each candidate has one explanatory commitment and the same three observable commitments: combustion C, gas uptake G and solid mass change M. Values are dimensionless and signed.

| Candidate | Directed coefficients | Joint reference mean (z,C,G,M) |
| --- | --- | --- |
| Phlogiston | P→C: +.8; P→G: −.8; P→M: −.8 | (0,1,−1,−1) |
| Oxygen | O→C: +.8; O→G: +.8; G→M: +.9 | (0,1,+1,+1) |

Both explanatory coordinates have residual variance .25; the other conditional residual variances are .5. CPD intercepts are `(I-B)mu`, not the joint means themselves. The shared observation operator selects C,G,M. There is no aggregate node, utility term or direct P/O observation. P and O have different meanings; parameter fusion stays within corresponding candidate frames, while cross-candidate readouts use only C,G,M. K=1 baselines use observable marginal priors and a computationally inert independent dummy coordinate, avoiding averages of P and O.

P and O are centred continuous coordinates within their respective explanations. Their reference mean zero denotes the chosen origin, not a probability that phlogiston or oxygen is absent. The categorical probability `q(oxygen)` applies to the whole supplied oxygen explanation.

All state updates call the unmodified accepted `rival._step`: communicate model log scores, fuse within each candidate, score observations before incorporating them, update Gaussian information, and apply anchored forgetting. The world is a prescribed persistent signal with sensor noise. It is not a sequence of fresh latent samples from a true candidate DAG. Initial prior discrimination is predominantly due to different mean predictions, so these runs demonstrate selection among supplied explanations, not isolated recovery of wiring.

## Provenance

- Scientific source commit: **`463b003`**, branch `orx/four-variable-fixed-assessment`.
- Publication assets and analysis branch: **`publication/four-variable-figures`**.
- orx project: `49b2c5d7-2a4b-4c60-8487-589091edec33`.
- orx experiment: `78bf7816-b882-4d39-8fed-e4c74aa250b7`.
- Completed run: `8fcbd7ad-02ab-43c9-8b1b-a45fcc0cb11f`.
- [Protocol](protocol.json), [orx receipt](orx_run.json), [complete log](orx_run.log), [run summary](runs/assessment-2026-09-19/summary.json).

The run executed an isolated Git snapshot. Its frozen outputs are copied into [runs/assessment-2026-09-19](runs/assessment-2026-09-19/), containing all per-condition arrays, final states, graph adjacencies, seed lists, source copies, environment details and hashes. Full arrays occupy approximately 1.27 GB and are excluded from Git; the current workspace retains them. Re-executing the source regenerates them. The vendored engine is an unmodified copy of the recovered publication source at `5e8574af7c345f40221f06e76062c468e9a8d850`; the extra modules preserve its import graph, but utility and prototype routines are not executed by this assessment.

## Reproduce and inspect

Run commands from the repository root. The existing `.venv-reproduction` uses Python 3.12.9 and the versions in [requirements.lock.txt](requirements.lock.txt). Figure rendering also needs `latex`, `dvipng` and `pdflatex` (TeX Live 2025 was used).

Rerun the frozen source through the configured local orx experiment:

```bash
orx exp run 78bf7816-b882-4d39-8fed-e4c74aa250b7 --backend local
orx runs 49b2c5d7-2a4b-4c60-8487-589091edec33
```

Each orx run extracts a fresh snapshot, so its fixed output path does not overwrite earlier runs. The configured command uses the existing environment's absolute Python path. On another machine, import this repository into orx and set the equivalent environment path before creating the baseline. The Python entrypoint used by that fixed command is:

```bash
.venv-reproduction/bin/python paper/four_variable_experiments/run.py
```

It accepts `--protocol` and `--output` for explicit standalone reproduction, refuses an existing output directory, and prints every condition's final summary. In the recorded assessment the protocol is committed and the command is fixed; variants should use a new immutable source/configuration rather than alter the completed run.

Regenerate figures, inspect numerical provenance, or reproduce the seed-level analysis from saved outputs:

```bash
.venv-reproduction/bin/python paper/four_variable_experiments/figures/plot_figures.py --run-dir paper/four_variable_experiments/runs/assessment-2026-09-19
.venv-reproduction/bin/python paper/four_variable_experiments/verify.py
.venv-reproduction/bin/python paper/four_variable_experiments/analyze.py
.venv-reproduction/bin/python -m unittest discover -s tests -p test_four_variable_experiments.py -v
```

Seven construction/scoring tests passed. Five independent inference controls checked label permutation, duplicated candidates, observationally equivalent hidden coordinates, C-only evidence and source-free pooling. An independent batch check matched held-out predictive scores against SciPy. The [final validator](VALIDATION.json) checked 29 conditions, 43 files, 302 arrays and 46 engine source files. It also confirmed identical pre-change states in paired intervention controls and the exact source-off contraction recurrence. No validation asserts that an intended scientific finding must occur.

## Readout definitions

- `q` and `L`: posterior candidate probabilities and unnormalised accumulated log scores, after each update. Initial probabilities are uniform before the first plotted point.
- `mu_map`, `mu_mixture`: shared observable means, with slot zero explicitly set to zero; that slot is not an estimate combining P and O.
- `mse_obs`: squared error of the mixture mean against the known noiseless C,G,M signal, averaged over those three coordinates.
- `heldout_nll`: negative log density of four fresh full-observation draws per agent per step under its Gaussian-mixture predictive distribution. Evaluation noise is separate from training observations and paired across matched conditions. Lower is better; this includes predictive uncertainty rather than only the mean.
- Crossings: first post-update index where group-mean q(oxygen) exceeds .9. Non-crossings remain −1 and are explicitly counted; they are not silently replaced by the horizon.
- Tail summary: final quarter of the relevant trajectory. The source-off and long continuation each use their own horizons. Supplementary figure endpoint summaries may use a different explicitly captioned window.

The full-information adaptation reference observes all three coordinates after the change and stops sharing then. It receives more direct information than each selective observer, but does not pool the community's measurements. It is not a centralized oracle or an equal-observation-budget competitor.
