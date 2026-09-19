# Four-variable assessment: results and implications for the paper

The fixed assessment completed all **29 conditions**. The four-variable example produces a clear imposed paradigm transition, informative minority-to-population trajectories, and a useful comparison of social learning channels. It does **not** preserve the original positive claim that the principal disagreement experiment maintains opposing paradigms: that split is temporary, and both camps eventually choose the incumbent despite a remaining distance between their commitment means.

This is a new, smaller experimental world using the accepted inference engine. It is not an exact reproduction of the historical six- or ten-variable figures, and it does not use the separate independent-case covariance-learning suite. No parameters were changed in response to the results reported here.

The numerical sources are the [frozen protocol](runs/assessment-2026-09-19/protocol.json), [summary](runs/assessment-2026-09-19/summary.json), saved per-condition trajectories, and [paired analysis](ANALYSIS.json). The [figure review](figures/figure_review.pdf), [figure manifest](figures/figure_manifest.json), and [coverage audit](COVERAGE.md) connect these results to the manuscript.

## What was simulated

Each candidate uses four coordinates: a candidate-specific latent explanatory commitment and three common observable commitments, combustion C, gas uptake G, and mass change M. The incumbent has `P -> C,G,M`, with coefficients `+.8,−.8,−.8`. The rival has `O -> C,G` with coefficients `+.8,+.8`, and `G -> M` with coefficient `+.9`. Their reference means are `(0,1,−1,−1)` and `(0,1,+1,+1)`; residual variances are `(.25,.5,.5,.5)`. Intercepts enforce those means. These are phlogiston-inspired belief dependencies, not a calibrated historical or chemical model.

Observations directly measure C, G and M with independent Gaussian sensor noise. The hidden explanatory coordinate is never observed. The world supplies a persistent prescribed signal, with an imposed change where specified; it does not draw a fresh latent case from either candidate DAG each round. Both candidates are supplied from the beginning, initially with equal model probabilities. There is no belief utility, theory generation, pruning, active experiment policy, or retrospective reliability reweighting.

The accepted inference step pools model scores, fuses Gaussian parameters within each candidate frame, scores the incoming observation before depositing it, and applies anchored forgetting. Baseline model-score pooling is `alpha_m=.02`, with `rho_L=0`. Parameter fusion and model-score pooling are distinct communication routes. The single-Gaussian controls use correctly marginalized priors over C/G/M, with an independent unused coordinate only for array alignment; they do not average P and O as if those were the same physical quantity.

| Experiment | Agents | Stored steps | Independent seeds | Sensor SD | Memory parameter | Evidence/world |
|---|---:|---:|---:|---:|---:|---|
| Cycle and three controls | 16 | 240 | 50 | 1 | .9; no-forgetting control 1 | All channels; reference mean changes at index 80 |
| Four graphs and five network controls | 24 | 600 | 50 | 1 | .9 | Four fully attentive sources; other agents' G/M weight .01 after index 80 |
| Eleven disagreement conditions | 16 | 1500 | 20 | .5 | 1, .95, .3 or .9, as below | Two camps of eight; G-only versus M-only evidence in mixed world `(C,G,M)=(1,−1,+1)` |
| Disagreement continuation | 16 | 3500 additional | Same 20 | .5 | .3 | Continue the original rival condition to 5000 total steps |
| Source-off continuation | 16 | 400 additional | Same 20 | .5 | 1 | All observation weights zero; no anchor replenishment |
| Three adaptation conditions | 16 | 1500 | 20 | 1 | .9 | Common history, then oxygen-consistent signal and different access/sharing conditions |

Here omega is retained information before replenishment from the reference prior; omega=1 disables that anchored forgetting. Trust weights on the saved graph adjacencies are `W=(A+I)/row_sum(A+I)`, not Metropolis weights. The selective social gate uses C/G/M only, with `nu_s=.01`; it gates social influence, not sensory reliability. Network attention and sharing controls branch from identical pre-change states within the same base graph. Adaptation controls also share their pre-change state. The hub/rim comparison uses the same wheel and role-specific evidence, but has placement-specific pre-change histories.

## How to read the numbers

`qO` is the posterior weight on the entire oxygen candidate. A crossing is the **first strict crossing of group-mean qO above .9**, using zero-based stored post-update indices. It is not a sustained-crossing rule. The change occurs at index 80, so a crossing at 116 is 36 indices after the change. Non-crossings remain censored rather than being replaced by the last step. Network group means and errors below refer to the 20 restricted agents; cycle and adaptation means use all agents.

MSE measures the model-mixture mean's squared error against the noiseless C/G/M signal. Held-out NLL measures the full candidate-mixture predictive distribution against four fresh noisy C/G/M observations per agent and step. These held-out observations are never assimilated. Evaluation follows the current update and forgetting; this is a current-world held-out score, not a pre-update score of the training datum. NLL is in nats per three-dimensional observation and includes predictive uncertainty. Lower values are better. Absolute NLLs should not be compared across sensor noise settings.

“Tail” averages the final quarter of each saved condition and then averages across agents and independent seeds. Figure ribbons are pointwise 95% percentile bootstrap intervals from 2000 whole-seed resamples, seed 20260919. Agents are averaged within each seed first. They are not min–max envelopes, standard deviations, or simultaneous confidence bands. Paired effect intervals use the same bootstrap count and seed on seed-level differences. Entropy is computed per agent before averaging. Probabilities and entropy require no clipping or smoothing.

## The cycle works, with a bounded claim about adaptation

| Condition | Group crossings | Median first crossing | Final mean qO | Tail MSE | Tail held-out NLL |
|---|---:|---:|---:|---:|---:|
| Imposed change, omega=.9 | 50/50 | 116 | .99994 | .00860 | 4.27731 |
| No change | 0/50 | — | .000047 | .00911 | 4.27833 |
| Change, no forgetting | 0/50 | — | .87743 | .39125 | 4.84841 |
| Single Gaussian | Not applicable | — | Not applicable | .02281 | 4.29734 |

The ordinary cycle moves from mean qO approximately .00655 before the change to nearly one, with a mean-individual entropy peak of approximately .563 nats. All 800 individual trajectories cross .9 during the 240-step run. The no-change population does not cross; two individual trajectories have very early temporary crossings, so the null result should be stated at the population level rather than as “no agent ever chooses oxygen.”

Without forgetting, the population is still moving toward the rival at the end of this horizon; 26/800 individuals cross but no seed's population mean does. This shows slower adaptation under accumulated information for these settings. It does not establish that forgetting is necessary for any eventual transition.

The single Gaussian also revises its observable prediction. Its absence of a categorical model posterior is an architectural fact, not evidence that it cannot adapt. The supplied-candidate condition has lower tail MSE by .01421, paired 95% interval `[−.01522,−.01331]`, and lower held-out NLL by .02002, interval `[−.02159,−.01851]`. The appropriate claim is better predictive performance and an explicit model-selection trajectory in this configured comparison, not a universal necessity theorem for multiple representations.

**Paper use:** [Figure 4](figures/fig04_cycle.pdf) is a strong compact replacement for the old cycle illustration. Retain the no-change control and describe the imposed signal change, supplied alternatives, and precise entropy statistic in the main text.

## Minority transmission depends on both graph and communication route

| Network condition | Median restricted-group crossing | Censored groups / 50 | Censored individuals / 1000 | Final restricted mean qO |
|---|---:|---:|---:|---:|
| Complete | 180 | 0 | 0 | ≈1 |
| Wheel, sources include hub | 186 | 0 | 0 | ≈1 |
| Community | 329.5 | 0 | 0 | .999999 |
| Cycle | 392 | 0 | 0 | .999671 |
| Wheel, four rim sources | 314 | 0 | 0 | ≈1 |
| Complete, model-score sharing off | — | 50 | 898 | .460355 |
| Complete, all social sharing off | 347.5 | 0 | 0 | .999969 |
| Complete, restricted G/M attention zero | 185 | 0 | 0 | ≈1 |
| Complete, everyone fully attentive | 115 | 0 | 0 | ≈1 |

The original graph-row composition remains informative. Complete and hub-seeded wheel graphs propagate the transition earlier than the particular community and cycle graphs. Moving the sources off the wheel hub substantially delays the transition. This is evidence of placement and topology sensitivity on these fixed graphs, not a universal graph ordering: graph density, path lengths, row-normalized influence, and source placement all vary. There is one saved realization of each graph, not an ensemble of random graph draws.

The strongest additional comparison separates the sharing routes. Turning off model-score sharing while retaining Gaussian parameter fusion leaves restricted mean qO near .46 at step 600; none of the 50 restricted groups crosses .9, although 102 individual trajectories do. Turning off **both** sharing routes permits eventual selection through the .01 direct evidence trickle in all 50 runs. Thus `alpha_m=0` must not be called isolation. Parameter pooling without model-score sharing is detrimental here; full communication is beneficial relative to complete isolation as well as to that partial intervention.

Relative to model-score sharing off, full sharing improves tail restricted-agent MSE by .31633, paired interval `[−.32505,−.30813]`, and held-out NLL by .33432, interval `[−.34306,−.32594]`. Relative to all sharing off, it improves MSE by .00937, interval `[−.00970,−.00904]`, and NLL by .11188, interval `[−.11254,−.11121]`. In the latter comparison both groups cross in every paired seed; the mean paired crossing difference is −167.02 steps, interval `[−169.12,−164.84]`. No finite crossing difference is assigned to the censored model-score-sharing-off groups.

The zero-direct condition confirms that fully informed agents can support the restricted group's transition without a direct G/M channel. It does not mean those agents receive no data: C remains observed, and after a diagnostic history its candidate-conditioned predictions need not coincide. The full-attention endpoint is much faster. The .01 default must therefore be described as a small direct evidence stream, not zero information.

**Paper use:** [Figure 9](figures/fig09_topology.pdf) preserves the useful original design. State graph weights, source placement, direct attention and crossing definitions. The [sharing controls](figures/figS2_social_controls.pdf) deserve at least a main-text sentence because they sharpen the mechanism beyond “social influence causes lock-in.”

## The principal disagreement experiment gives a temporary split, not lasting opposing paradigms

The mixed signal G=−1, M=+1 does not match either candidate's complete reference mean. Camp A observes G only; camp B observes M only. The following are **new-world** conditions. “Original” identifies the inherited memory settings, not historical reproduction. Gaussian conditions use one observable-marginal Gaussian; rivals use the two supplied candidates.

`D` below is the final Euclidean distance between the camps' mean MAP predictions on C/G/M, averaged over seeds. It is not distance between graph structures. The qO columns are final camp-average probabilities, then averaged across seeds.

| Condition | Omega | Reference anchor | Final D | Final qO, camp A / B | Tail held-out NLL |
|---|---:|---|---:|---|---:|
| Plain, original memory | 1 | Camp | .00025 | — | 2.55191 |
| Gated, original memory | .95 | Camp | .07994 | — | 2.59996 |
| Rivals, original memory | .3 | Candidate | .72746 | `2.7e−181 / 2.1e−123` | 4.32950 |
| Plain, matched memory | .9 | Camp | .06676 | — | 2.66878 |
| Gated, matched memory | .9 | Camp | 1.94570 | — | 4.44485 |
| Rivals, matched memory | .9 | Candidate | .04309 | `2.4e−43 / 4.2e−41` | 2.59860 |
| Plain, shared anchor | .9 | Shared | .05017 | — | 2.66770 |
| Gated, shared anchor | .9 | Shared | .60567 | — | 2.96602 |
| Rivals, no forgetting | 1 | Candidate, not replenished | .00025 | `.09929 / .09908` | 2.53134 |
| Rivals, model-score sharing off | .3 | Candidate | 2.36723 | `0 / 1` | 5.25057 |
| Rivals, all sharing off | .3 | Candidate | 2.80562 | `0 / 1` | 5.65768 |

The first three distance curves superficially resemble the intended baseline/gate/rivals hierarchy. The model probabilities change the interpretation. In the original-memory rival condition, all 20 seeds temporarily have camp-A mean qO below .1 while camp-B mean qO exceeds .9. The median last index of that split is 262. None retains it at step 1500: both camps overwhelmingly choose the incumbent. A plateau in commitment distance therefore cannot stand in for persistence of distinct paradigm choices.

The asymmetry is visible in the predictive evidence sources. Over the final quarter, the rival-minus-incumbent source averages approximately −1.588 nats/step in camp A and +1.114 in camp B; the equal-camp mean is −.237. A continuing negative common drift can move both camps toward the incumbent while their log-odds difference remains large. The initial-prior analytic check already shows unequal opposing evidence, but the late source values must be measured after updating, rather than inferred from initial priors.

Matching memory reverses the apparent gate-versus-rivals distance ordering: the gated Gaussian has much larger final D than represented rivals. Sharing the Gaussian anchor reduces, but does not eliminate, the gated distance. This means the original three-arm contrast cannot isolate representation alone. Different memory and anchors materially affect what is preserved. Large distance is also not automatically desirable: the largest disagreements here accompany substantially worse held-out prediction of the full mixed signal.

When model-score sharing is disabled, the two camps do retain opposite candidate choices over the observed horizon. That is a valid finite-run result under selective evidence, but it cannot rescue the claim that the ordinary communicated rival posterior preserves the split. Disabling both communication routes increases the distance further and worsens prediction again.

Two continuations resolve otherwise ambiguous traces:

- **Continued evidence to 5000 total steps:** the original-memory rival condition keeps a tail mean distance of .73537 and a mean log-odds gap near −135.13, but both camps choose the incumbent. qO eventually underflows to numerical zero; saved raw log scores remain finite. The longer run supports the same interpretation rather than a lasting two-paradigm split.
- **Remove all evidence sources and anchor replenishment:** a 400-step continuation from the original rival endpoint sets all sensory weights to zero and omega=1. The camp log-odds gap follows the expected `(1−.02)` contraction per step to within `1.54e−12`; it ends near −.04177, and observable mean distance vanishes. This checks unforced pooling. Because both camps already selected the incumbent at the continuation's start, it is not a demonstration that pooling destroys an existing pair of opposing paradigm choices. It also disables replenishment as well as incoming observations.

**Paper use:** [Figure 3](figures/fig03_walls.pdf) can honestly show a transient split and the danger of using mean distance alone. It cannot replace the original positive “persistent rival paradigms” figure under the old caption. The [matched-memory](figures/figS1_matched_memory.pdf), [complete wall controls](figures/figS6_wall_controls.pdf), and [continuations](figures/figS7_persistence.pdf) support a narrower statement about driven disagreement. A camera-ready paper should either include this negative result and narrow the claim, or retain the separately documented original-world experiment with explicit provenance and limitations. It should not silently substitute the new curve for the old scientific result.

## The candidate-consistent adaptation check is a useful positive comparison

These runs start with a common full-attention prehistory. At index 80 the signal becomes the rival reference mean, including G=M=+1. Afterward, the two ordinary conditions retain complementary G-only and M-only camps. The social condition communicates both model scores and within-candidate parameters; the isolated condition communicates neither. The reference agents independently observe all C/G/M channels and do not communicate. They have more direct information, so they are not an equal-access comparator or a pooled community oracle.

| Condition | Group crossings | Median first crossing | Tail MSE | Tail held-out NLL |
|---|---:|---:|---:|---:|
| Selective evidence, full social sharing | 20/20 | 130.5 | .00796 | 4.32940 |
| Same selective evidence, isolated | 20/20 | 251.5 | .02135 | 4.37046 |
| Full-information independent reference | 20/20 | 113 | .03529 | 4.31215 |

All three conditions reach final qO effectively one; every individual crosses within the run. Full sharing improves adaptation relative to isolation with the same selective evidence: the mean paired crossing difference is −119.85 steps, interval `[−123.95,−115.55]`; tail MSE improves by .01339, interval `[−.01387,−.01293]`; NLL improves by .04106, interval `[−.04215,−.03995]`.

The full-information reference selects the rival earlier and achieves the lowest held-out NLL, while the socially sharing condition has lower mean-squared error. There is no contradiction: MSE evaluates only the predictive mean, whereas NLL evaluates its uncertainty as well. The conditions also differ in access and social averaging. Social sharing has a .01725 NLL disadvantage against the reference, interval `[.01633,.01824]`, despite a .02734 MSE advantage. The paper should show both quantities rather than choose the one that gives the desired ranking.

**Paper use:** [The adaptation figure](figures/figS5_adaptation.pdf) is a strong supplementary result and a possible main-text replacement for an unsupported persistent-wall claim. It demonstrates a benefit of communication under distributed evidence without presenting disagreement itself as an epistemic good.

## What the experiments establish, and what remains open

The supplied priors genuinely differ in their observable dependence patterns. However, approximately **97% of their initial predictive Gaussian KL separation at sensor SD 1 comes from their different means**; the covariance contribution is small. The independent analytic decomposition and label/observational-equivalence controls are recorded in [INDEPENDENT_VALIDATION.json](INDEPENDENT_VALIDATION.json) and explained in [COVERAGE.md](COVERAGE.md). Selection in these runs demonstrates discrimination between supplied explanations with different means and dependencies. It does not isolate wiring as the cause of discrimination, establish causal-DAG recovery, or prove a universal need for multiple structures.

Direct C/G/M measurements deposit diagonal Fisher information. They do not introduce new dependence edges. The fixed menu and prescribed attention also do not test endogenous theory discovery or active choice of experiments. Parameter sharing can affect predictive scores within this architecture; it is important to state precisely which communication route is removed in each control.

The present results support a compact paper centered on **selection among competing supplied explanations, the role of information retention, and transmission through a scientific community**. They support graph- and placement-dependent delays and a distinction between parameter sharing and model-score sharing. They do not currently support a generic permanent-schism claim or a claim that the four-node world establishes structural recovery. Finite non-crossings must remain finite-horizon observations.

For a stronger structural claim, the next separate assessment should match candidate means and examine model recovery over specified coupling/noise settings; that would answer a different question from the present mean-change illustration. For general network claims, additional graph realizations and placements are needed. For a persistent-schism claim, balanced long-run evidence sources and absolute camp log odds must be established analytically and tested, rather than selecting a visually flat mean-distance trace. None of these stronger claims should be implied before those tests exist.

The [model diagram](figures/fig01_model.pdf), cycle and topology plots can form a coherent main figure set. The current diagram replaces unnecessary background nodes with explicit candidate-specific latent meaning. Figure 2 need not be retained as a second abstract cartoon; a concrete uncertainty or social-control panel would be more informative. Original utility figures and the graph-editing prototype should be retired. Retrospective reliability is a separate algorithm and has not been rerun here; none of these results revives its variance claim. The exact original-figure mapping is in the coverage audit.

## Reproducibility and verification

The completed ORX local run is **`8fcbd7ad-02ab-43c9-8b1b-a45fcc0cb11f`**, source commit **`463b003`**, branch `orx/four-variable-fixed-assessment`; see [orx_run.json](orx_run.json). The earlier provisional registration listed in that receipt was not run. The fixed assessment used 50 seeds for cycle/network and 20 for disagreement/adaptation, with all 29 conditions retained, including the unsuccessful intended wall replacement.

Execution used Python 3.12.9, JAX/JAXlib .9.0 with float64 enabled, NumPy 2.3.4, SciPy 1.16.3, Equinox .13.6, NetworkX 3.5 and Matplotlib 3.10.7 on macOS arm64. The run directory preserves the protocol, world arrays, graph adjacencies, source copies, environment, final Gaussian states, raw log scores, model probabilities, observable predictions and evaluation trajectories. Array and file hashes are recorded in its manifests. The validation receipts cover 29 conditions, 43 files, 302 arrays and 46 source files; no run is listed as failed.

Independent checks cover candidate construction and observed predictive densities, correct observable marginalization, K=1 recurrence, label permutation, duplicate candidates, latent-only observational equivalence, attention endpoints, and source-free model-score contraction. Rendering consumes saved arrays rather than rerunning or tuning experiments. Figures use LaTeX Computer Modern/CM-Super, vector PDF/SVG outputs, explicit statistical captions and a manifest of input/output hashes. These receipts document what was executed and plotted; they do not remove the scientific scope limits above.
