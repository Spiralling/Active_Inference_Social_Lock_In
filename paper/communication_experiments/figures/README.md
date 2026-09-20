# Publication figure sources and outputs

Run `python paper/communication_experiments/figures/plot_figures.py` from the
repository root using the supplied pinned environment and a working LaTeX
installation. No simulation is launched. The input is the portable
`analysis/plot_data.npz` and its companion statistical summaries. Run
`validate_figures.py` afterwards. The figures use the vendored **orx** academic
figure style, adapted to 4.8-inch LNCS width, and actual LaTeX T1 Computer Modern
text and math. They are Matplotlib vector plots, not TikZ or generated images.

The PDF and SVG files are publication assets. PNG files are inspection
previews. `figure_pack.pdf` collects every plot with its precise caption;
`integration_blocks.tex` provides figure environments for manuscript editing.
The camera-ready manuscript itself has not been edited.

Main candidates, in reading order:

1. `fig_network_channels`: all four communication arms, group probabilities,
   and late full-signal MSE/held-out mixture NLL.
2. `fig_camp_channels`: all four camp arms and every individual gas-observer
   probability at the two endpoints. Fixed horizontal jitter only separates
   plotted points; it never changes a probability or a statistical calculation.
3. `fig_reference_diagnostics`: the four prespecified conditional-only
   settings, gas scores against the frozen forecasts, and endpoint preferences.
4. `fig_weight_sharing_intervention`: the full gas-group continuation and the
   first 100 updates, with the latter display window specified before outcomes.

Supporting figures:

- `fig_all_camp_conditions`: all seven conditions across the full horizon,
  plus statewise opposition fractions for the two relevant communication arms.
- `fig_intervention_prediction`: all-seed paired probability/MSE/NLL changes
  and the full mass-group continuation, including its lack of detectable change.
- `fig_individual_and_uncertainty`: all mass-observer endpoint probabilities,
  every individual in the fixed illustrative seed 10000, and separate within-
  agent entropy and between-agent disagreement.

The primary inferential unit is the independent seed, never an individual
agent. Curves and summary points use 2,000 whole-seed bootstrap resamples with
analysis seed 20260920. Pairing is retained across arms. Intervals are pointwise
95% percentile intervals of means; they are not min–max envelopes, standard
deviations or simultaneous bands. Individual strips show all 400 dependent
values in each arm and have no inferential uncertainty band. Entropy curves
use a fixed ten-update display grid; probability curves retain every update.
All tabular calculations use every raw round in the exact declared window.

`analysis/` retains all five factorial contrasts, all paired intervention
contrasts, every seed/window record, all observed and censored events, and both
primary group-average and secondary individual opposition measures. Figures
do not substitute an endpoint for a window-occupancy criterion. The raw data
companion is required to recompute the analysis; it is not required to render
these already reduced publication plots.
