# Four-variable figures

The [captioned review PDF](figure_review.pdf) is the convenient entry point. Each
figure also has a vector PDF for LaTeX, a vector SVG, and a PNG inspection preview.
These figures belong to the new four-variable assessment; they are not exact
reproductions of the historical ten-variable simulations.

The main-paper candidates retain the accepted paper's experiment roles:

| Figure | File | Purpose |
|---|---|---|
| 1 | [fig01_model.pdf](fig01_model.pdf) | Implemented candidate prior networks, with explicit P/O explanatory variables and C/G/M observables. |
| 3 | [fig03_walls.pdf](fig03_walls.pdf) | Disagreement under selective evidence: distance alongside the two camps' candidate probabilities. |
| 4 | [fig04_cycle.pdf](fig04_cycle.pdf) | World-change cycle, unchanged-world control, and mean individual entropy. |
| 9 | [fig09_topology.pdf](fig09_topology.pdf) | Original graph-row composition and restricted-agent trajectories on the four contact graphs. |

The supplementary diagnostic figures are for choosing what the paper can claim;
they need not all be included in the manuscript.

| Figure | File | Purpose |
|---|---|---|
| S1 | [figS1_matched_memory.pdf](figS1_matched_memory.pdf) | Disagreement comparison at the same parameter-memory coefficient. |
| S2 | [figS2_social_controls.pdf](figS2_social_controls.pdf) | Communication routes and informed-node placement. |
| S3 | [figS3_memory.pdf](figS3_memory.pdf) | No-forgetting and single-Gaussian cycle controls, including observable prediction error. |
| S4 | [figS4_direct_evidence.pdf](figS4_direct_evidence.pdf) | Diagnostic attention weights 0, 0.01, and 1. |
| S5 | [figS5_adaptation.pdf](figS5_adaptation.pdf) | Selective networked and isolated agents against independent full-information references. |
| S6 | [figS6_wall_controls.pdf](figS6_wall_controls.pdf) | Every wall-control endpoint, with probability and distance kept distinct. |
| S7 | [figS7_persistence.pdf](figS7_persistence.pdf) | Long continuation and the evidence-off log-odds recurrence. |

## Rebuild

From the repository root, after the assessment finishes:

```bash
.venv-reproduction/bin/python paper/four_variable_experiments/figures/plot_figures.py --run-dir paper/four_variable_experiments/runs/assessment-2026-09-19
```

`plot_figures.py` reads only saved arrays; it does not run the model, select seeds,
or tune parameters. `model.tex` defines the TikZ diagram. Local vendored modules
`publication_style.py`, `orx_figstyle.py`, and `publication_topology.py` supply
the orx plotting style and original Figure 9 composition. `font_reference.json`
records the accepted manuscript's font specification and source hash without
requiring a copy of that manuscript to render. A working LaTeX installation is required.

Every plot is built at 4.8 inches (the LNCS text width) and uses actual LaTeX
Computer Modern / CM-Super, as requested. PDFs contain vector curves and text.
The orx default sans-serif preference is deliberately overridden by the user's
request for the manuscript typeface. PNGs are previews only.

## Statistical definitions and provenance

Within each simulation seed, agent-level quantities are reduced to the indicated
group mean. The heavy curve is the arithmetic mean of those seed-level curves.
Every ribbon is a **pointwise 95% percentile-bootstrap interval for the mean**,
using 2,000 resamples of complete seed trajectories and fixed bootstrap seed
20260919. This preserves dependence within each run. It is not a minimum-to-maximum
envelope, an agent-level interval, a simultaneous band, or mean plus/minus one
standard deviation. No smoothing or clipping is performed; probability and
entropy intervals remain in their mathematical domains by construction.

Figure 9's thin lines display all restricted agents in seed 0; they illustrate
within-run differences and do not define its uncertainty ribbon. Entropy is
computed for each agent before averaging, not from the average posterior.
Wall distances are Euclidean distances between camp means on C/G/M, not distances
between graph structures. MAP and mixture readouts are distinct. The trust gate
in the wall comparison gates social influence, not sensory reliability.

`figure_manifest.json` records all consumed condition files, input/output hashes,
plotting-source hashes, exact bootstrap settings, summary statistics and embedded
font families. `captions.md` contains the captions as plain text. The captioned
review is deliberately a review document; the numerical exports are sized for
direct inclusion in the manuscript at `width=\linewidth`.
