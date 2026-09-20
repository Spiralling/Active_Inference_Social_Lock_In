# Communication experiments: results and interpretation

All experiments required by the implementation brief were completed, without changing the prescribed world, candidate means, model-score memory or gates. The results support a distinction between exchanging beliefs **within** a candidate and exchanging **support for** candidates. They also show why a separation result needs the reference-variance control beside it.

The new network comparison is the strongest addition for the paper: sharing model weights alone transmits support for oxygen to restricted observers, and adding conditional exchange improves their prediction and accelerates adoption. The camp experiment demonstrates a narrower result: opposing choices persist over the tested horizon when model weights remain private and cross-group conditional influence is restricted. It does not demonstrate opposing fixed beliefs under unrestricted exchange of both kinds of information.

## What was implemented and run

The actual archived adaptive engine was called at every update. A new wrapper exposes its two existing switches independently; in particular, it implements the previously missing model-weight-only network arm. It also records direct pre-learning scores, observable conditional means, post-update predictions, exact random states and deterministic precision histories. The engine's 46 Python files are unchanged.

- **Changed-signal network:** four communication arms, 24 agents, 50 paired seeds, 600 updates, a common 80-update prehistory and the original phase-two random-key reset. Four agents retain full observations; the other 20 have weights $(1,.01,.01)$ after the change. The three existing arms reproduce every archived array exactly.
- **Constant-world camps:** seven conditions, 16 agents, 50 fresh paired seeds, 5,000 updates. Eight observe gas and eight observe mass. Everyone experiences the same signal $(C,G,M)=(1,1,1)$. Both candidates and their original means are retained.
- **Fixed intervention:** all 50 primary conditional-only runs branch at exactly state 1,500. Model-weight exchange is enabled in one continuation; the private-weight continuation is reused only after an exact full-restart check.

There are 13 logical conditions and 650 condition/seed histories, including reused data. The independent replications are **50 network streams and 50 camp streams**, paired across their respective conditions. The network comparison extends the archived seeds; it is not independent reconfirmation of its original three arms. The camp setting was selected analytically in the supplied proposal, which also reported a pilot. This is confirmation after that exploration, not a retrospective claim of preregistration. The external pilot trajectories were unavailable and are not presented as verified.

The frozen producer is commit `139225df25e641d387f4ea5e020e54203fe32c6b`, orx run `ccaba0e8-39ca-4993-a219-64ce4cd09de5`. The full suite completed in about 145 seconds, with no failed conditions or replacement seeds. See [the protocol](protocol.json), [methods and equation-linked pseudocode](METHODS.md), and [execution provenance](EXECUTION_PROVENANCE.json).

## 1. Communication channels have different effects

These are averages over completed states **451–600**, first within seed and then across 50 seeds. MSE is full-observable mixture-mean error relative to the noiseless signal. NLL is the mean negative log mixture density of four fresh full-observation replicates per agent and update. Lower prediction losses are better.

| Post-change communication | Restricted observers: $q(O)$ | Restricted MSE | Restricted NLL | Fully observing: $q(O)$ | Fully observing MSE | Fully observing NLL |
|---|---:|---:|---:|---:|---:|---:|
| Neither (`N00`) | 0.579145 | 0.325758 | 4.665943 | 0.999996 | 0.036829 | 4.309365 |
| Conditional only (`N10`) | 0.102894 | 0.287230 | 4.615481 | 1.000000 | 0.041395 | 4.330399 |
| Model weights only (`N01`) | 0.923908 | 0.030871 | 4.422448 | 0.970148 | 0.037549 | 4.310418 |
| Both (`N11`) | 1.000000 | 0.004791 | 4.307251 | 1.000000 | 0.041395 | 4.330399 |

The restricted group crosses mean $q(O)>.9$ in **50/50** seeds with either model-weight-only or both-channel exchange, and **0/50** seeds with neither or conditional-only exchange. Median first crossing is completed state **496** for model weights only and **249** for both (round indices 495 and 248). The corresponding 20-update sustained crossings become detectable at median states **515** and **268**. Noncrossings remain right-censored at 600; they are not events at 600 or proof that a crossing could never occur.

The direct comparison of both channels against model weights only gives restricted observers a late-window MSE reduction of **0.026080** (paired 95% bootstrap interval 0.024878–0.027258) and NLL reduction of **0.115197** (0.113362–0.116951). Conditional exchange therefore adds predictive value in that comparison, even though model-weight exchange alone already produces adoption.

There are two qualifications worth keeping in the text:

1. **Candidate preference and prediction are distinct.** Conditional sharing alone lowers restricted agents' oxygen support substantially relative to neither channel, while modestly improving their late-window prediction. At the endpoint, however, its MSE is worse than neither channel. The window and endpoint answer different questions; both are retained in the tables. Do not substitute the favorable one for the other.
2. **Benefits are not uniform across agents or metric scales.** Fully observing agents have slightly worse late MSE and NLL with conditional sharing. For restricted observers, the factorial interaction is positive on oxygen probability, negative on NLL, and has an interval spanning zero on late MSE: $+0.012448$ [−0.001410, +0.026665]. These are scale-specific effects, not a blanket “synergy” result.

All five paired factorial comparisons, both groups, and both window and endpoint estimates are in [network_factorial_contrasts.csv](analysis/network_factorial_contrasts.csv). Adoption and censoring are in [event_summary.csv](analysis/event_summary.csv) and [individual event records](analysis/events.csv).

## 2. Opposing camp preferences survive conditional exchange, with specific conditions

The primary condition uses total cross-group conditional weight $\beta=.05$, conditional retention $\omega=.3$, and **private model weights**. Each agent receives conditional information from the other group, but does not receive its accumulated support for theories. When the model-weight channel is enabled, $\alpha_m=.02$ and its immediate total cross-group contribution is only $\alpha_m\beta=.001$.

Primary opposition at a state means the gas group's mean $q(O)<.1$ and the mass group's mean $q(O)>.9$. A seed passes if this holds for at least 90% of the reporting window. The following probabilities are window averages, not endpoints.

| Condition | Gas $q(O)$, states 1,126–1,500 | Gas $q(O)$, states 3,751–5,000 | Primary passes at 1,500 | Primary passes at 5,000 |
|---|---:|---:|---:|---:|
| Neither channel (`C00`) | 0.001868 | $1.85\times10^{-9}$ | 50/50 | 50/50 |
| Conditional only (`C10`) | 0.023302 | $6.21\times10^{-6}$ | 50/50 | 50/50 |
| Model weights only (`C01`) | 1.000000 | 1.000000 | 0/50 | 0/50 |
| Both (`C11`) | 1.000000 | 1.000000 | 0/50 | 0/50 |
| Conditional only, $\beta=.5$ | 0.999993 | 1.000000 | 0/50 | 0/50 |
| Conditional only, $\omega=.9$ | 0.744802 | 0.938449 | 0/50 | 0/50 |
| Conditional only, matched gas variance | 0.786661 | 0.982782 | 0/50 | 0/50 |

The mass group favors oxygen strongly throughout these late windows (its smallest displayed condition/window mean is approximately 0.999979). The primary opposition occupancy is 1 for `C00` and `C10`, and 0 for all other conditions, in both windows. A 50/50 pass fraction has Wilson 95% interval **[0.9287, 1]**; 0/50 has **[0, 0.0713]**. Observing all successes does not establish certainty outside these conditions or permanent fixation.

The stricter secondary event requires all eight individuals in each group to pass their respective confidence threshold. `C10` passes that occupancy criterion in **47/50** seeds at the 1,500 horizon and **50/50** at 5,000. Group-average opposition must not be described as unanimous strong individual opposition at the earlier horizon.

`C00` also separates, with an even stronger gas preference for release. Therefore conditional exchange is **compatible with** opposing commitments here; it does not create the separation. The two enabling features tested by the controls are restricted conditional influence and continued reference influence. Higher retention weakens the reference anchor; stronger cross-group conditional exchange changes the predictions used to score later observations. Both remove the primary opposition outcome in this suite.

## 3. The reference-variance control is decisive

The original gas reference means are identical in the two candidates, but the gas marginal variances are 0.82 for release and 0.8648 for oxygen. The diagnostic changes only release's gas residual variance from 0.5 to 0.5448, matching its marginal gas variance to 0.8648. Means and directed coefficients are unchanged, and the full precision and information vector are recompiled.

| Conditional-only condition | Frozen gas-score prediction | Measured late gas score | 95% bootstrap interval |
|---|---:|---:|---:|
| Primary | −0.003022079 | −0.003018703 | [−0.003042254, −0.002993742] |
| $\beta=.5$ | +0.013773431 | +0.013823549 | [+0.013677775, +0.013972398] |
| $\omega=.9$ | +0.001130619 | +0.001135666 | [+0.001124682, +0.001146624] |
| Gas variance matched | +0.001078491 | +0.001086645 | [+0.001067943, +0.001105433] |

Scores are oxygen minus release, in nats per agent per round; late means use states 3,751–5,000. All corresponding gas and mass forecasts lie inside their pointwise bootstrap intervals at both reporting horizons. This is agreement with stationary expected scores, not a deterministic prediction of individual adoption times. The intervals are correlated across paired conditions and are not simultaneous confidence bounds.

Matching gas variance reverses the gas score sign and removes opposing preferences. The original separation is therefore sensitive to the reference uncertainty specified in the model. It cannot be presented as an isolated effect of causal wiring, a generic consequence of representing alternatives, or evidence of irreducible paradigm incommensurability. The control is also not a pure test of topology: changing a residual variance changes the full reference distribution, even while coefficients and means are fixed.

See [forecast comparisons](analysis/stationary_forecast_comparison.csv) and the [independent stationary calculation](STATIONARY_PREDICTIONS.json). No scores were forced to agree under positive cross-group influence.

## 4. Transmitting model weights changes choices without changing local learning

All 50 primary `C10` runs satisfy the group-average opposition criterion at the fixed branch state 1,500. With private weights retained, all 50 continue to satisfy the late-window criterion. Enabling model-weight exchange removes it in all 50.

For the enabled branch, the first 20-state nonopposing interval starts at **state 1,503 in every seed** and is confirmed at **1,522**. This is loss of the opposition threshold, not yet strong oxygen adoption. Both groups subsequently exceed mean $q(O)>.9$; median first qualifying state is **1,508**, and a 20-state interval becomes detectable at median state **1,527**. The unchanged branch has no such events by state 5,000.

Conditional states, local candidate scores, training observations and held-out samples are **identical** between the two branches. The gas group's local mean score still favors release. The reversal is consequently attributable to transmitting accumulated model support, which changes the candidate mixture and its predictions without changing the conditional learner.

The predictive difference is substantial for gas observers. Their late full-observable MSE falls from **1.278857** to **0.019563**, a paired change of **−1.259294** [−1.259576, −1.259026]. Their NLL falls from **4.832537** to **2.841172**, a change of **−1.991365** [−1.992763, −1.989873]. The mass group's late metrics coincide to the recorded numerical precision. This shows a predictive cost of preserving the original division in the specified world; diversity is not treated as an objective to maximize.

Mechanistically, $z_{t+1}=[(1-\alpha_m)I+\alpha_mW]z_t+s_t$. With equal groups and $\beta=.05$, sharing contracts their mean log-odds difference by 0.998 before adding new scores, while the common component accumulates a positive mean score. A persistent difference in log odds is compatible with eventual agreement in candidate choice. No autonomous bistability or feedback from confidence to observation or trust has been demonstrated.

## What belongs in the camera-ready paper

The network comparison adds a missing control to an existing accepted experiment and is the most useful new evidence. The camp suite is worth presenting as a tightly qualified communication result, with the gas-variance control visible beside it. The fixed-time intervention provides a direct mechanism check and can fit in the supplement if the page budget is tight.

Recommended claims are:

- Within-candidate conditional exchange and between-candidate support exchange have different consequences for choice and prediction.
- In this supplied model family, restricted conditional exchange can coexist with opposing candidate preferences when model weights remain private.
- That separation depends on the specified reference uncertainty and retention. Transmitting theoretical support removes it and improves gas observers' full-observable predictions in the tested world.

Do not claim that model-weight sharing is universally necessary, that multiple representations are necessary for all belief disagreement, that the result escapes unrestricted consensus, or that selecting oxygen identifies the true causal graph. Both candidates are supplied in advance and the world is a prescribed signal, not repeated samples from a known causal DAG. The conditions use assigned observation groups and fixed trust, with no endogenous emergence of camps.

The required work is complete; no optional sensitivity grids, negative-gas world, changed candidate means, score decay, adaptive trust, extra candidates or new single-Gaussian comparison were run. No manuscript was edited. [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) maps the figure candidates and caveats to the paper, while [VALIDATION.md](VALIDATION.md) summarizes the independent checks. All nonseparating outcomes and exact per-seed records remain in the handoff.
