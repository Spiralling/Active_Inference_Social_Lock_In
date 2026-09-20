# Incorporating the matched comparison

Use the revised nine-page `figures/figure_pack.pdf`. Pages 1–7 reproduce the previous communication figure pack. Page 8 adds `fig_matched_camp_models.pdf`, the Gaussian/gated-Gaussian/candidate comparison under the new setup. Page 9 adds `fig_matched_camp_diagnostics.pdf`. Captions and LaTeX integration blocks accompany the assets.

The main comparison belongs beside the discussion of camp disagreement. It can replace the older comparison with unmatched retention if that figure remains in the manuscript. The changed-signal network, cycle, graphical abstract and topology figures address other questions; this extension does not replace those experiments.

## State what is being compared

All new arms share the world, observation groups, retention, noise and base communication matrix. Plain and gated Gaussian arms start with the same reference for both groups. Two reference constructions assess sensitivity: logarithmic pooling of observable marginal Gaussians, and moment matching of their equal mixture. The candidate condition retains the original two references, so this compares specified update/representation mechanisms rather than isolating candidate count.

The social gate reweights conditional sharing by observable mean disagreement. It does not gate observations, revise retained evidence or introduce the removed belief-utility term. The base weighting is preserved before gating. Realized cross-group influence is measured rather than assumed equal to the base .05.

Make candidate communication explicit: the separating candidate condition shares conditional beliefs but keeps model weights private. The shared-weight condition is shown as a control. The reference-variance diagnostic remains visible in predictive outcomes and the preceding diagnostic figure.

## Keep the readouts distinct

The primary camp distance uses posterior predictive means on C,G,M. For candidates this is the probability-weighted mixture mean; the selected-candidate (MAP) distance is supplementary. A readout change can alter a plotted curve without changing the inference trajectories. Hidden P/O coordinates never enter either distance.

Candidate probabilities identify opposing theoretical preferences. Single-Gaussian arms do not have an oxygen probability: a normalized K=1 probability of one supplies no such information. Full-observable MSE and held-out NLL assess accuracy, separately for gas and mass observers. A larger distance is not automatically beneficial.

Instantaneous Euclidean distance is positive under ordinary sampling noise. The supplementary analysis averages signed camp differences over time and reports those coordinates, alongside distinct distance summaries. The norm of the across-seed mean vector is descriptive only; it has no percentile interval that could spuriously exclude zero at a zero vector. An independent stationary calculation checks the plain arms' mechanism.

## Claims this comparison does not establish

- Contraction of conditional pooling alone does not establish consensus in a system receiving unequal new observations and repeatedly restoring references. The zero-preservation property of Gaussian pooling is a separate mathematical statement.
- Numerical disagreement between Gaussian predictions is distinct from maintaining explicit alternative explanations. Avoid saying that multiple candidates are necessary for every kind of persistent disagreement.
- Private model weights restrict communication. A private-weight camp result does not establish opposing beliefs under unrestricted support exchange.
- Matching retention does not match the initial predictive density of a Gaussian to a two-component mixture. Moment matching preserves only its first two moments.
- The comparison neither discovers causal graphs nor tests historical phlogiston chemistry. It uses supplied reference models and an imposed noisy signal.

Use the measured outcomes in `RESULTS.md` and its per-seed tables. No ordering of curves is a criterion for accepting a run. The frozen protocol, independent checks, source identities and raw-data manifests accompany the figure pack.
