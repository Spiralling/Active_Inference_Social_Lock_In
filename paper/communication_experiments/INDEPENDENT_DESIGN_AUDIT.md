# Independent design and source audit

This audit was prepared before inspecting any new confirmatory trajectories. It checks the requested specification rather than requiring a positive outcome.

The camera-ready source ZIP's `provenance/world.py`, `run.py` and `protocol.json` exactly match `paper/combustion_experiments/`. They do not match the older similarly named `paper/four_variable_experiments/` experiment wrappers. The accepted engine nevertheless lives under `paper/four_variable_experiments/engine/`; all 46 Python file hashes match the supplied archive's `accepted_engine_hashes.json`.

The lean source archive SHA256 is `ce89bacb39a8cfceeb5a39074c66182de67e6a75889aec46493a919481b15e75`. The original world source SHA256 is `dd227ae6e0e20e653b979a329fdce5eb389afb2fc491365b10f506ce6889dee1`. Its first latent coordinate is candidate-specific. Only `(C,G,M)` support cross-candidate mixture predictions and distances.

The external `Camp_Report_Independent_Check.zip` and the pilot support archive were not available locally. The recommendations file without `(1)` was available; its pilot claims are not independently verified. `stationary_predictions.py` independently reconstructs all four forecasts from the specified full 16-agent, four-state affine conditional dynamics and a SciPy discrete Lyapunov solution. A convergent covariance series separately checks that solution. The largest discrepancy from the brief's rounded predictions is 4.71e-10 nats per round; the largest stationarity residual is 1.43e-14. Matching the gas marginal at beta=0 yields zero gas-score difference numerically. These are calculations of expected scores, not adaptive simulation results or threshold predictions.

## Risks checked explicitly

1. Conditional and model sharing are independent switches. Turning a channel off must not freeze local learning. `W_beta` contains cross-group entries beta/8; binarizing it would destroy the experiment.
2. With gates off, paired `00/01`, `10/11`, and intervention branches must have identical conditional states and local scores. Differences in mixture losses can come solely from different candidate weights.
3. Prequential scores use post-sharing, pre-deposit beliefs. Computing scores after learning from the same datum would answer a different question.
4. Attention transforms observations and the working observation map by square roots. Actual transformed sample noise is sigma² diag(attention); working noise in the score is sigma² I. The covariance of an information-vector innovation is Hᵀ diag(attention²) H / sigma², not Hᵀ diag(attention) H / sigma² for fractional masks.
5. The network must branch once from state80 with the archived phase-two key reset to seed+1000. A camp branch at state1500 changes round1500 onward, never round1499.
6. Scores must be recorded directly and checked against the actual matrix recurrence, not reconstructed with the old complete-pool `wall_details` formula.
7. Camp opposition is a within-seed group-mean criterion. The simultaneous all-eight strong-individual event is a separate secondary outcome. Aggregating seeds or replacing missing event times with the horizon would distort the required endpoints.
8. MSE targets the noiseless signal. NLL is the logarithm of a candidate mixture evaluated on four new full observations, not a mixture-weighted component loss.

## Interpretation constraints

At the primary setting the mean gas source is negative but the overall source is positive: about 0.289788365 nats per round. Model sharing at alpha=.02 and beta=.05 multiplies the group difference by .998 before adding new scores. A bounded group gap cannot by itself imply opposing long-run choices while the community average drifts. Private model weights make the source signs decisive under the stationary conditions described in the proposal.

The matched-gas diagnostic reverses the gas source's sign analytically. A positive primary outcome is therefore a reference-uncertainty-dependent effect, not an isolated consequence of causal wiring. The group assignments, observation masks and communication weights remain prescribed. No feedback from theory support to attention, trust or conditional learning is present.

Historical engine comments contain stronger claims about fixed opposing probabilities. They are retained as source provenance and must not override the corrected recurrence analysis in `METHODS.md`.
