# Guidance for incorporating this suite into the camera-ready paper

This handoff supplements the supplied latest camera-ready source. It does not edit or replace that manuscript. The candidate family, general inference framework, utility-free formulation and existing cycle/topology results remain available. The new experiments isolate the two communication channels and qualify the camp-separation claim.

Read [RESULTS.md](RESULTS.md) before selecting figures. Use [METHODS.md](METHODS.md) for the general mathematical description, Gaussian specialization and equation-linked pseudocode. Exact condition definitions, time conventions and statistical rules are frozen in [protocol.json](protocol.json).

## Figure candidates

All numerical figures use the orx publication style adapted to a 4.8-inch LNCS text width, actual LaTeX Computer Modern text/math, and vector PDF/SVG exports. PNGs are viewing previews. They are generated with Matplotlib; they are not TikZ diagrams. Generating code, font settings and all plot inputs are included. `figures/figure_pack.pdf` collects the figures and captions; `figures/integration_blocks.tex` supplies copyable LaTeX figure blocks.

| PDF in `figures/` | What it adds | Suggested placement |
|---|---|---|
| `fig_network_channels.pdf` | Four communication arms, separate observer groups, and prediction alongside candidate preference | Strongest main-text candidate; complements the existing topology experiment |
| `fig_camp_channels.pdf` | Opposing preferences with private model weights; individual probability distributions at both horizons | Main-text candidate if camp separation remains a central claim |
| `fig_reference_diagnostics.pdf` | Measured score signs, independently derived stationary forecasts, and the variance/retention/influence controls | Keep adjacent to the camp result; do not bury the variance qualification |
| `fig_weight_sharing_intervention.pdf` | Fixed-time support transmission removes opposition while conditional learning is unchanged | Main text if space permits; otherwise a direct mechanism check in the appendix |
| `fig_all_camp_conditions.pdf` | All seven prescribed camp conditions over the full horizon | Supplement |
| `fig_intervention_prediction.pdf` | Paired changes in predictive losses after enabling model sharing | Supplement; retain the predictive cost of separation in the main-text discussion |
| `fig_individual_and_uncertainty.pdf` | Individual probabilities and entropy/disagreement diagnostics | Supplement |

These are alternatives and supporting views, not a recommendation to add seven figures to a 12-page paper. The most economical addition is the network comparison plus a compact camp result with its reference-variance qualification. Preserve the existing topology figure if it serves the paper's network argument; the new complete-graph factorial comparison does not test topology and should not silently replace that claim. The accepted cycle and graphical abstract need no change merely because this suite was added.

Use the exact supplied captions as the starting point. All reported interval bands are **pointwise 95% whole-seed bootstrap intervals for a mean**, with 2,000 resamples and 50 paired seeds; they are not min–max envelopes, standard deviations or confidence about individual agents. Individual distributions are descriptive, with agents clustered within seeds. Means, endpoints and reporting windows must remain distinct.

## Changes needed in the methods

1. Define conditional beliefs and model support separately in the general model, before specializing to Gaussians. Conditional exchange uses a logarithmic pool within matching candidate frames. Support exchange uses damped pooling of old log weights. Both local learning rules stay active when their respective communication channel is disabled.
2. State the actual order: snapshot; support sharing; conditional sharing; draw observation and score before depositing it; update every candidate; apply conditional retention. There is no model-score forgetting, utility term or reliability gate in this suite.
3. Preserve candidate-specific hidden coordinates $P$ and $O$. Only $(C,G,M)$ can be mixed or compared across candidates. The world is a prescribed signal with private noise, not a causal-DAG sampling experiment.
4. Describe the network intervention as **post-change** channel switching from a shared prehistory. The first changed observation is round 80 and produces state 81. Retain the weak weights $(1,.01,.01)$ rather than calling restricted agents unobservant.
5. Describe the new camp experiment explicitly: common signal $(1,1,1)$; gas-only and mass-only groups; shared candidate references and uniform initial model probabilities; $\omega=.3$, $\sigma_o=.5$, $\beta=.05$. It is different from the older negative-gas camp setting and the previous crossed-mean/score-decay suite.
6. Define $W_\beta$ numerically as in the brief. Its cross-group total is $\beta$, not an entry-wise edge weight. With support exchange enabled, the immediate cross-group log-weight contribution is $\alpha_m\beta=.001$.
7. State the matched-variance control in the main text: release gas residual variance 0.5 → 0.5448, matching gas marginal variance 0.8648. It changes the full reference distribution while preserving means and directed coefficients.
8. Separate primary group-average opposition from the secondary all-eight-individual threshold. State exact windows and censoring rules. Explain that the intervention is applied to all seeds at fixed state 1,500, regardless of their outcomes.

## Claim wording supported by the results

Suggested results prose, to adapt to the manuscript's voice:

> Sharing conditional beliefs and sharing theoretical support had different effects. In the changed-signal network, support exchange alone led the restricted group to favor oxygen in all 50 runs. Adding conditional exchange accelerated adoption and improved that group's full-observable predictions. Conditional exchange alone did not transmit the same theoretical preference, despite providing a modest late-window predictive benefit.

> With private model weights, gas and mass observers retained opposing preferences while exchanging conditional information across groups. This occurred in all 50 runs under the specified reporting criterion through 5,000 updates. It depended on restricted cross-group influence and the candidate references: stronger conditional exchange, higher retention or matching the gas reference variance removed opposition. The result therefore demonstrates conditional coexistence of preferences, not permanent disagreement under unrestricted communication.

> Enabling model-weight exchange at a fixed time removed opposition without changing the branches' conditional states or local predictive scores. Gas observers then favored oxygen despite their still-negative local score difference, and their full-observable prediction improved. This intervention identifies transmission of model support as the direct cause of the change in this experiment.

The two positive camp statements should travel with their controls. Do not describe these as communities with different truths, spontaneous camps, endogenous trust adaptation or discovery of oxygen. The groups, measurement masks, reference family and candidate menu were supplied. The original gas-only preference is sensitive to reference uncertainty, and separation is predictively costly here.

## Claims to remove or qualify

- **“Rival representations sustain disagreement despite unrestricted communication.”** The separating condition has no model-weight exchange and restricted conditional exchange. Both-channel conditions agree on oxygen over the tested late windows.
- **“Communication creates the division.”** The no-communication camp arm also separates. The supported statement is that conditional communication can coexist with the division under the tested conditions.
- **“A fixed log-odds gap is a fixed point of opposite beliefs.”** A drifting common component can take both groups to the same candidate. Use the exact recurrence and distinguish these quantities.
- **“The effect follows from graph topology.”** Matching one reference marginal variance reverses the gas score. The experiment does not isolate causal topology.
- **“Both channels are required for adoption.”** Model-weight-only exchange also produces adoption in the network; both channels accelerate it and improve restricted observers' prediction.
- **“More communication helps everyone” or “the channels act synergistically.”** Fully observing agents show modest prediction costs, and the factorial interaction depends on metric and window.
- **“50/50 establishes permanent separation.”** This is a finite-horizon result in 50 independent paired streams; report the criterion and Wilson interval.
- **“Opposition shows a benefit of two representations over one.”** All arms retain two candidates; candidate count was not tested.

Read `LEGACY_ENGINE_NOTICE.md` before quoting archived source comments. Historical claims in those comments are not the conclusions of this suite.

## Reproducibility and availability paragraph

The package contains the frozen protocol, original engine, production wrapper, all per-seed/aggregate tables, plotting inputs, figure scripts, reference sources, independent audits and exact stream manifests. The optional full-data archive supplies every raw chunk and deterministic precision history. Numerical verification includes exact recovery of all three existing network arms, exact paired-channel conditional/observation invariants, full state-1,500 restart equivalence, and independent equation and statistical checks.

Use the producer commit and run identifiers in `EXECUTION_PROVENANCE.json`. A publication repository URL or DOI should refer to an actual public deposit; none was created in this task. Do not fabricate one. The source archive's original manuscript remains unchanged, and no new references were added by these experiments.
