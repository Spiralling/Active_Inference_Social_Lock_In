# IWAI 2026: Communication and camp-separation experiments

## Implementation brief for the coding agent

**Status:** Proposed experiments, not completed results.  
**Basis:** The revised four-variable manuscript, the alternative experiment report [R1], and the independent mathematical check [R2].  
**Objective:** Establish how sharing beliefs within candidate theories and sharing support between theories affect theoretical commitment and prediction. Preserve the revised scientific setup; do not select an experiment solely because it produces separation.  
**Scope:** Implement, validate, run, and report the experiments below. Do not overwrite archived experiments or change the manuscript until the results have been reviewed.

This brief adopts the communication experiments recommended in the latest assessment. Relative to [R1], it makes the gas-variance-matching control essential, narrows the initial experiment set, and specifies a fixed-time intervention after 1,500 updates. The exact reporting windows, sustained-event convention, seed allocation, and output organization below are proposed implementation decisions, not claims about what the earlier report or archived experiments already did.

## 1. Execution priorities

Run these stages in order. Deliver the first stage's results even if time does not permit completion of the others.

| Priority | Experiment | Required conditions | Purpose |
|---|---|---|---|
| 0 | Implementation validation | Reference compilation, independent communication switches, paired observations, update order, and archived-result regression checks | Establish that experimental differences arise from the intended interventions. |
| 1 | Existing changed-signal network experiment | Neither channel, conditional sharing only, model-weight sharing only, both channels | Complete the missing model-weight-only comparison without changing the existing world. |
| 2 | New constant-world camp experiment | The same four communication conditions, plus three diagnostic controls | Test whether restricted conditional exchange and private model weights sustain different candidate preferences, and identify their dependence on memory and reference uncertainty. |
| 3 | Fixed-time model-weight-sharing intervention | Branch every primary conditional-only camp run at state 1,500 into continued private weights versus model-weight sharing | Test whether transmitting theoretical support removes a division that develops under conditional-only exchange. |

**Defer** the calibrated negative-gas world, changed candidate means, model-score forgetting, belief-dependent trust, different recurring model priors for the camps, and additional candidate theories. Also defer a large single-Gaussian comparison unless a separate claim about the benefit of representing multiple candidates is required. All primary conditions here retain two candidates; they isolate communication channels, not candidate count.

The primary camp condition has cross-group exchange of within-theory beliefs but **no model-weight exchange**. Do not describe a successful result as coexistence under unrestricted exchange of both kinds of belief. [R1, Sections 1, 3, and 7]

## 2. Source files and provenance

Use the latest four-variable implementation, not the older six-commitment camp experiment or ten-commitment cycle.

Relevant supplied materials:

- `IWAI2026_camera_ready_source.zip`: `camera_ready/main.tex` and the `camera_ready/provenance/` directory, including `world.py`, `run.py`, `protocol.json`, `world.npz`, `graphs.npz`, and archived summaries. [R3]
- `IWAI_camera_ready_handoff_2026-09-19.zip`: the earlier full handoff. Its engine is under `IWAI_camera_ready_handoff_2026-09-19/paper/four_variable_experiments/engine/`, including `src/structural/rival.py`. [R4]
- `IWAI_camp_separation_recommendations(1).md`: scientific rationale and proposed communication experiments. [R1]
- `Camp_Report_Independent_Check.zip`: `camp_report_independent_check/check_report.py`, `results.json`, `provenance.json`, and `README.md`. These verify stationary expectations, not adaptive trajectories. [R2]

The lean camera-ready archive does not include the underlying adaptive engine, but the larger handoff does. Verify the actual import path rather than substituting a different installed `src` package. Restore the recorded dependencies where possible and record all environment differences.

[R1] reports a separate 20-realization pilot and refers to `IWAI_camp_separation_support.zip`. That support archive was not supplied with the report in this conversation. Its pilot trajectories have not been independently verified here. Do not treat the reported pilot outcomes as production-engine results, or reproduce its random samples by guessing from the master seed alone.

Create a new versioned experiment directory. Record source commit identifiers, checksums of reference arrays and imported engine files, the frozen protocol, dependencies, numerical precision, and random-number conventions. Preserve all original files and outputs. The independent check is an auxiliary validation, not a substitute inference engine.

## 3. Preserve the scientific model and indexing

### 3.1 Candidate definitions

The candidate states are

\[
x_P=(P,C,G,M)^\top,\qquad x_O=(O,C,G,M)^\top.
\]

The shared observables are combustion extent \(C\), net gas uptake \(G\), and solid mass change \(M\). The first coordinate has a different interpretation in each candidate. Pool conditional beliefs only within matching candidates. Compute across-candidate predictions and distances only on \((C,G,M)\). [R3, `world.py`]

Use the existing directed coefficients and residual variances:

| Candidate | Directed dependencies and coefficients |
|---|---|
| Release/phlogiston, code candidate `0` | \(C\to P:0.8\), \(C\to G:0.8\), \(P\to M:-0.8\) |
| Oxygen, code candidate `1` | \(C\to O:0.8\), \(O\to G:0.8\), \(O\to M:0.8\) |

The reference residual variances, in storage order `(z, C, G, M)`, are `(0.25, 0.5, 0.5, 0.5)` for both candidates. The reference joint means are `(0, 1, 1, -1)` for release and `(0, 1, 1, 1)` for oxygen. The symbol `z` is a storage label, not a common physical quantity.

References are identical across agents within each candidate. Initial candidate probabilities are `(0.5, 0.5)` for every agent. Keep `rho_L = 0.0`; model scores accumulate without forgetting. Disable the optional social and model-weight reliability gates. Keep all candidates learning even when their probabilities become very small.

The only reference modification in the required suite is the explicitly labelled variance diagnostic in Section 6.3.

### 3.2 Coordinate and time conventions

| Object | Convention |
|---|---|
| Agent index | Zero-based `i = 0, ..., N-1`. |
| Candidate index | `0 = release/phlogiston`, `1 = oxygen`. |
| Full-state coordinate index | `0 = candidate-specific hidden quantity`, `1 = C`, `2 = G`, `3 = M`. |
| Observation-row index | `0 = C`, `1 = G`, `2 = M`. |
| State entering round `t` | \((\Pi_{ik,t},h_{ik,t},L_{ik,t})\). |
| Round index | `t = 0, ..., T-1`. |
| Completed updates after round `t` | `n = t + 1`. |
| Output array entry `q[t, i, k]` | Probability in state `t+1`; it already includes observation `t`. |
| Output array entry `ell[t, i, k]` | Score assigned to observation `t` before learning from it. |

The first changed-world observation in the network experiment is round `t=80`. Its common prehistory ends at state `80`, after rounds `0,...,79`. Activating model sharing at state `1500` means changing the rule beginning with round `t=1500`, not changing output entry `1499`.

### 3.3 Update order and notation

`Pi` is the precision matrix, the inverse covariance; `h = Pi @ mu` is the information vector. A superscript `0` denotes a fixed reference, a bar denotes the current round after sharing, and `+` denotes the state after local learning but before conditional memory.

Keep the executed order:

1. Read one entering-state snapshot for all agents.
2. Pool old model log weights when that channel is enabled.
3. Pool Gaussian precision matrices and information vectors within each candidate when conditional sharing is enabled.
4. Draw the current raw observation and compute every candidate's predictive score from the post-sharing, pre-learning state.
5. Add the observation's Gaussian information to every candidate and its score to the corresponding model log weight.
6. Apply conditional memory to both Gaussian information quantities, then record state `t+1`.

With all gates and model-score decay off, the defining updates are

\[
\bar L_{ik,t}=(1-\alpha_m)L_{ik,t}
+\alpha_m\sum_j W^m_{ij,t}L_{jk,t},
\]

\[
\bar\Pi_{ik,t}=\sum_jW^x_{ij,t}\Pi_{jk,t},\qquad
\bar h_{ik,t}=\sum_jW^x_{ij,t}h_{jk,t},
\]

\[
L_{ik,t+1}=\bar L_{ik,t}+\ell_{ik,t},
\]

\[
\Pi_{ik,t+1}=(1-\omega)\Pi^0_k+\omega(\bar\Pi_{ik,t}+J_{i,t}),
\qquad
h_{ik,t+1}=(1-\omega)h^0_k+\omega(\bar h_{ik,t}+j_{i,t}).
\]

Here \(\alpha_m\) controls model-weight sharing, \(\omega\) is conditional retention, and \((J_{i,t},j_{i,t})\) are the likelihood's information increments. A disabled conditional channel is equivalent to \(W^x=I_N\), not to freezing the Gaussian beliefs. A disabled model-weight channel is \(\alpha_m=0\), not to freezing model probabilities.

Preserve the existing attention transformation and likelihood. Raw observations are generated on every channel; attention weights determine their influence. In particular, weight `0.01` is not zero evidence, and weighting a channel is not the same as changing its sampled noise. [R3, manuscript and `run.py`; R4, `rival.py`]

## 4. Define the communication interventions independently

Let `a` indicate conditional sharing and `b` indicate model-weight sharing. Use suffix `ab` in condition identifiers.

| Suffix | Conditional sharing | Model-weight sharing | Intended engine configuration |
|---|---|---|---|
| `00` | Off | Off | `fuse_nets=False, alpha_m=0.0` |
| `10` | On | Off | `fuse_nets=True, alpha_m=0.0` |
| `01` | Off | On | `fuse_nets=False, alpha_m=0.02` |
| `11` | On | On | `fuse_nets=True, alpha_m=0.02` |

Pass the experiment's prescribed matrix `W` to the enabled channels. For the current engine, both enabled channels use that matrix with the gates disabled. The effective conditional matrix is identity when `fuse_nets=False`; the effective model-weight operator is identity when `alpha_m=0`.

The existing network wrapper's `all_off` route disables both channels. It cannot implement `01`. Expose the two switches independently, inspect the resulting configuration, and verify behavior with unit tests. This does not require changing the scientific update equations. [R1, Section 5.4; R3, `run.py`; R4, `rival.py`]

For the camp experiment, use the weighted matrix

\[
W_\beta=\begin{pmatrix}
(1-\beta)J_8&\beta J_8\\
\beta J_8&(1-\beta)J_8
\end{pmatrix},\qquad J_8=\mathbf1_8\mathbf1_8^\top/8.
\]

Each receiver assigns total influence \(\beta\) to the opposite group and \(1-\beta\) to its own, including itself. The primary value is `beta=0.05`. Each cross-group entry is `0.05/8`, **not** `0.05`. At `beta=0.5`, the matrix is the uniform complete pool `ones((16,16))/16`. At `beta=0`, exchange continues within groups; this is not complete isolation.

Use `W_beta` directly. Do not replace its positive entries with an unweighted adjacency and normalize again, which would erase the intended group weights. Test nonnegativity, row sums, column sums, symmetry, and group influence totals. Distinguish the total cross-group weight `beta=0.05` from the effective model-log-weight contribution `alpha_m * beta = 0.001` when model sharing is on. The prescribed observation groups and communication weights are not emergent communities. [R1, Section 3.1]

## 5. Priority 1: Complete the changed-signal network comparison

### 5.1 Conditions and common prehistory

Run `N00`, `N10`, `N01`, and `N11` on the existing complete graph.

| Setting | Value |
|---|---|
| Population and candidates | `N=24`, `K=2` |
| Base communication matrix | `W = ones((24,24))/24` |
| Conditional retention | `omega=0.9` |
| Observation noise standard deviation | `sigma_o=1.0` |
| Model sharing when enabled | `alpha_m=0.02` |
| Total horizon | `T=600` rounds, including the shared prehistory |
| Prehistory | Rounds `0..79`, both channels on, every agent with weights `(1,1,1)` |
| World before the change | Observable signal `(1,1,-1)` |
| World from round `80` | Observable signal `(1,1,1)` |
| Fully observing agents after the change | Code agents `0,1,2,3`, weights `(1,1,1)` |
| Restricted agents after the change | Code agents `4..23`, weights `(1,0.01,0.01)` |

For each seed, generate the prehistory once. Clone its full state at `80` into all four arms, then apply the channel intervention and changed observation regime. Do not run separate arm-specific histories before round `80`. The conditions are “neither/one/both channels **after the change**,” not lifelong isolation versus communication. [R1, Section 5.2; R3, protocol and network wrapper]

### 5.2 Pairing and seeds

The archived network seed list is `0..49`. Its phase-two observation stream starts from `seed+1000`. Preserve the exact legacy random-key layout when extending that archive. Recompute all four arms if the old trajectories or paired samples cannot be recovered exactly. Do not combine an unpaired new arm with old summaries and call the result a paired comparison.

The existing three conditions are regression targets, not numbers to impose on new results. The missing `N01` arm has no supplied numerical result. A same-seed extension of the archived comparison is not a new independent confirmation of its existing three arms. Record that distinction.

### 5.3 Required outputs

Report the 20 restricted agents and four fully observing agents separately. For each arm provide candidate-probability trajectories, final and last-quarter probabilities, adoption events, individual preference distributions, and full-observable predictive MSE and mixture NLL as defined in Section 9.

For each metric `Y`, calculate paired effects within seed:

- Model-weight sharing without conditional sharing: `Y01 - Y00`.
- Model-weight sharing with conditional sharing: `Y11 - Y10`.
- Conditional sharing without model-weight sharing: `Y10 - Y00`.
- Conditional sharing with model-weight sharing: `Y11 - Y01`.
- Factorial interaction: `(Y11 - Y10) - (Y01 - Y00)`.

The interaction depends on the metric's scale. Interpret probability, squared error, and log loss separately; do not label every nonzero interaction “synergy.” This experiment is informative whether model-weight-only sharing succeeds, fails, or trades earlier adoption against prediction quality.

## 6. Priority 2: Constant-world camp experiment

### 6.1 Common settings

| Setting | Value |
|---|---|
| Population and candidates | `N=16`, `K=2` |
| Gas group A | Code agents `0..7`, weights `(0,1,0)` |
| Mass group B | Code agents `8..15`, weights `(0,0,1)` |
| Common observable signal | `(1,1,1)` in every round |
| Full storage signal | `(0,1,1,1)`; the first entry is unobserved padding |
| Initial state | Candidate references and equal model probabilities for every agent |
| Primary conditional matrix | `W_0.05` |
| Primary retention | `omega=0.3` |
| Observation noise standard deviation | `sigma_o=0.5` |
| Model-score decay and gates | Off throughout |
| Run length | `T=5000`; analyze the first `1500` separately |
| Replications | `50` fresh paired seed streams, frozen before confirmation |

There is no prehistory intervention or change point in this experiment. Generate all observations from the same world for both groups. Observation masks differ; underlying truths do not. [R1, Sections 2 and 3]

Use new confirmation seeds rather than the earlier pilot realizations. A proposed reservation is `10000..10049`, subject to checking the development-seed registry before freezing the protocol. If those streams have already been inspected, choose and record an unused set once, before reviewing outcomes. Seeds are independent across replications and paired across conditions. Record the generator and stream derivation; an integer seed alone is insufficient provenance.

### 6.2 Seven required conditions

| ID | Communication | Conditional matrix when on | Retention | References |
|---|---|---|---:|---|
| `C00` | Neither channel | Not used | 0.3 | Original |
| `C10` | Conditional only | `W_0.05` | 0.3 | Original |
| `C01` | Model weights only | Not used; model weights use `W_0.05` | 0.3 | Original |
| `C11` | Both channels | `W_0.05`; model weights also use `W_0.05` | 0.3 | Original |
| `C10_beta50` | Conditional only | `W_0.5` | 0.3 | Original |
| `C10_omega90` | Conditional only | `W_0.05` | 0.9 | Original |
| `C10_gasvar_matched` | Conditional only | `W_0.05` | 0.3 | Gas-variance diagnostic below |

These are the initial requirements, rather than the full sensitivity grid in [R1]. Preserve common observation and held-out arrays across all seven conditions. The retention and variance controls start from their specified references at round zero; they are not changes applied after camps have formed.

### 6.3 Essential gas-variance diagnostic

In `C10_gasvar_matched` only, increase the **release candidate's gas residual variance** from `0.5` to `0.5448`. Keep its reference mean, all directed coefficients, and all other residual variances unchanged. Leave the oxygen reference unchanged. Recompile the full Gaussian reference, including both precision and information vector.

For the directed model `x = b + B @ x + epsilon`, use

\[
b=(I-B)\mu^0,\qquad
\Sigma^0=(I-B)^{-1}R(I-B)^{-\top},\qquad
\Pi^0=(\Sigma^0)^{-1},\qquad h^0=\Pi^0\mu^0,
\]

where `R` is the diagonal residual covariance. Keep the original compilation conventions and prefer linear solves over explicit inverses in numerical code.

The intervention matches the reference marginal gas variances at `0.8648`; the original release value is `0.82`. It does not make the full candidate distributions identical. In particular, cross-group sharing can still transmit effects of different relationships among variables. Do not force gas scores to agree when `beta>0`. [R1, Sections 3.3 and 6.3]

This is an essential control because the independent calculation predicts that it reverses the gas group's small negative score at the primary setting. It tests a specified reference-dependent mechanism; it is not an isolated comparison of graph topology. [R2]

### 6.4 Analytical predictions to compare with the simulations

The following values are stationary expectations from [R2], including variation in predictions caused by earlier noisy observations. Scores are `oxygen minus release`, in nats per agent per round.

| Conditional-only condition | Expected gas score | Expected mass score |
|---|---:|---:|
| `C10` | -0.003022079 | +0.582598809 |
| `C10_beta50` | +0.013773431 | +0.931656831 |
| `C10_omega90` | +0.001130619 | +0.014425528 |
| `C10_gasvar_matched` | +0.001078491 | +0.582569828 |

Compare empirical late score means with these predictions and quantify uncertainty across seeds. Do not require a noisy finite trajectory to equal a stationary expectation to numerical precision. A discrepancy should prompt checks of transients, Monte Carlo uncertainty, sampling versus working noise, matrix weights, and score timing before any scientific setting is changed.

These are not confirmed probabilities, crossing times, or success rates. In particular, the small positive gas scores in the controls do not guarantee a threshold crossing by the finite horizon. The report's analytical choice of `beta=0.05` and its previous pilot must be disclosed; this is confirmation after exploration, not a retrospectively claimed preregistration.

## 7. Priority 3: Activate model-weight sharing at a fixed time

For every `C10` seed, save the complete state after exactly `1500` updates. Branch from that state into:

| ID | Rounds `0..1499` | Rounds `1500..4999` |
|---|---|---|
| `S_keep_private` | `C10` | Continue `C10`: conditional sharing only |
| `S_enable_weights` | `C10` | Keep `W^x=W_0.05`; enable model sharing with `W^m=W_0.05`, `alpha_m=0.02` |

The unchanged branch is the corresponding segment of the original `C10` run; reuse it after verifying exact restart equivalence. Do not reset conditional beliefs, accumulated log weights, reference anchors, or observation time. Preserve identical future observations and held-out samples across the two branches. Save or explicitly reconstruct the correct random-key state at the checkpoint.

**Branch every seed**, regardless of whether it has separated at update `1500`. The intervention time is fixed in advance, not triggered by observing a desirable outcome. Report the number of seeds already separated at the branch point, but keep the all-seed paired analysis primary. Any subset analysis is secondary and must disclose its selection rule.

Report change in gas-group support, the loss or continuation of opposing commitments, and predictive costs or benefits. A useful additional diagnostic is time to 20 consecutive nonopposing updates after the intervention. Treat that dwell length as the proposed reporting convention in Section 9, not a source-defined physical threshold.

With gates disabled and fixed attention, both branches must have identical conditional states and local predictive scores. Only their model weights, mixture predictions, and other model-weight-dependent readouts can differ. Failure of that invariant is an implementation issue.

## 8. Mandatory implementation checks

### 8.1 Reference and arithmetic checks

Use 64-bit floating-point arithmetic, consistent with the archived protocol. Verify positive-definite reference and working precisions, finite predictive log densities, and probabilities summing to one. Use stable log-sum-exp operations for probabilities and mixture density evaluation. Do not compute log odds by taking logarithms of rounded or clipped probabilities; use `L[...,1] - L[...,0]`.

Check candidate means, covariance entries, observation maps, and the modified gas variance against compiled arrays. Never average or subtract the hidden `P` and `O` coordinates across candidates. A three-observable marginal implementation is acceptable for independent verification, but production results should use the established engine or document and numerically validate any replacement.

### 8.2 Paired-channel invariants

For identical entering states, observation arrays, attention, and references, with gates disabled:

- `00` and `01` have identical conditional parameters and per-candidate predictive scores at every round.
- `10` and `11` have identical conditional parameters and per-candidate predictive scores at every round.
- `S_keep_private` and `S_enable_weights` have the same property after branching.

Model log weights should be the only differing state component within each pair. Mixture means and mixture NLL need not agree because they use those weights. Score history recording must not change random-key consumption or state updates.

### 8.3 Verify the exact log-odds recurrence

Define

\[
z_{i,t}=L_{iO,t}-L_{iP,t},\qquad
s_{i,t}=\ell_{iO,t}-\ell_{iP,t}.
\]

Record `ell` directly, and independently verify

\[
s_t=z_{t+1}-\big[(1-\alpha_m)I_N+\alpha_m W^m_t\big]z_t
\]

when `rho_L=0`. Use the entering state at a restart, not a new zero-log-odds initialization.

**Do not reuse the existing `wall_details` source reconstruction unchanged.** It assumes uniform complete pooling and subtracts a population mean. For `W_beta`, use the actual matrix product above. Otherwise the inferred local evidence will be wrong. Likewise, under private model weights, `s_t = z_{t+1} - z_t`; no population average is subtracted. [R3, `run.py`, `wall_details`]

### 8.4 Observation and restart checks

Generate or reproducibly reconstruct observation noise independently of condition-specific control flow. Audit hashes of identical samples across paired conditions. Evaluation noise must be separate from learning noise and must never enter the state update. Preserve the archived key layout for the network regression comparison.

Check that a run stopped and resumed at state `1500` is identical to its uninterrupted counterpart, including model weights, observations, and evaluation draws. Store exact checkpoint keys or a deterministic time-indexed stream specification. For fractional attention, verify the actual information-increment covariance rather than copying the binary-mask stationary-noise formula.

Useful deterministic tests include `W_beta` endpoints, identity sharing, candidate-label permutation with aligned outputs, and zero attention giving equal candidate score constants and no Gaussian information increment. For the matched-gas reference with `beta=0`, gas-only candidate scores should coincide numerically; this special-case test does not imply coincidence at positive cross-group influence.

If engine instrumentation is needed to export `ell` and checkpoints, keep the update arithmetic and random calls unchanged and test equivalence to the uninstrumented engine. State-changing optimizations require separate validation.

## 9. Outcomes and statistical analysis

### 9.1 Exact reporting windows

Use completed-update state `n` in summaries and store its correspondence to the round index.

| Summary | Completed-update states | Slice into post-update arrays |
|---|---|---|
| Network final quarter, `T=600` | `451..600` | `[450:600]` |
| Camp first-horizon final quarter | `1126..1500` | `[1125:1500]` |
| Camp long-horizon final quarter | `3751..5000` | `[3750:5000]` |

Report endpoints separately from window averages. Use the same final-quarter windows for probabilities, distance, MSE, and NLL. For branch comparisons, additionally show the full trajectory from state `1500`; do not silently change the tail definition to a quarter of the continuation length.

### 9.2 Camp separation: primary outcome

At completed state `n`, define the group means

\[
\bar q_{A,n}(O)=\frac18\sum_{i\in A}q_{iO,n},\qquad
\bar q_{B,n}(O)=\frac18\sum_{i\in B}q_{iO,n},
\]

and the opposing-group indicator

\[
I_n=\mathbf1\{\bar q_{A,n}(O)<0.1\ \text{and}\ \bar q_{B,n}(O)>0.9\}.
\]

Calculate `I_n` within each seed before combining seeds. For each reporting window provide the fraction of updates with `I_n=1`, whether that fraction is at least `0.9`, the longest continuous opposing interval, and subsequent loss and return of separation. Report the count and fraction of seeds meeting the 90% occupancy convention. Do not turn this into an assertion that the software must obtain a particular success count. [R1, Section 6.1]

Also report the proportions of individual gas observers with `q(O)<0.1` and mass observers with `q(O)>0.9`, and the full distribution of individual probabilities. As a secondary summary, record the occupancy of simultaneous 90%-of-agents strong opposition. With eight agents in each group, that individual threshold requires all eight, unlike the primary group-mean criterion. Keep their names distinct.

### 9.3 Adoption and loss-of-separation events

Retain the archived first-crossing statistic: first post-change round with group-mean `q(O)>0.9`, and the corresponding individual crossings. A noncrossing is right-censored at the horizon, not an event at `T` and not evidence of permanent failure.

As an additional predeclared statistic, define sustained adoption as 20 consecutive recorded states above `0.9`. Record both the first state of that qualifying interval and the state at which its 20-update duration becomes observable. For branch diagnostics, define sustained loss as 20 consecutive states with `I_n=0` after the branch; distinguish loss of opposition from adoption of oxygen by both groups. Search for qualifying intervals only within the available observation horizon.

The 20-update requirement is a proposed descriptive convention. Do not substitute it for the original first-crossing statistic in archived comparisons. Report event counts, censored counts, and medians among observed events with that conditioning stated. Do not compute an ordinary average after replacing censored times by the horizon.

### 9.4 Mechanism diagnostics

Retain per-agent scores, raw log weights, and their difference. Report signed group-average log odds, their difference, and the community-average log odds. The log odds of a mean probability are not the mean of individual log odds.

For uncertainty and between-agent disagreement, calculate

\[
H(q)=-\sum_kq_k\log q_k,\qquad
B_n=H\!\left(\frac1N\sum_iq_{i,n}\right)-\frac1N\sum_iH(q_{i,n}).
\]

Use natural logarithms, with `0*log(0)=0`. Compute within each seed before averaging seeds. Confident disagreement can produce large `B_n` with small mean individual entropy; large individual entropy instead indicates uncertainty within agents.

Keep signed group differences in observable means and both selected-candidate and mixture-mean distances as secondary outcomes. Never use a positive norm alone as evidence of different preferred candidates. Record sufficient checkpoints to compare means before sharing, after sharing, after learning, and after memory in an audit run; distinguish these stages in any reported distance.

### 9.5 Prediction

Preserve the wrapper's post-update independent-replicate evaluation. After recording state `t+1`, predict a new observation from the **same prescribed signal `u_t`** using all three observable channels. This is distinct from the pre-update, attention-weighted score used for learning. [R3, `run.py`, `batch`]

For each agent, form the candidate-weighted observable mean

\[
\hat u_{i,t+1}=\sum_kq_{ik,t+1}\,H\mu_{ik,t+1}.
\]

Measure `MSE` as the average of its three squared errors relative to noiseless `u_t`. Do not score against the already noisy training datum.

Use four independent full-observation evaluation draws per agent and round, paired across arms. For each draw, evaluate the Gaussian mixture with weights `q[t+1]`, component means `H @ mu[t+1]`, and component covariances

\[
H\Pi_{ik,t+1}^{-1}H^\top+\sigma_o^2 I_3.
\]

`NLL` is the negative log of the **mixture density**, averaged over the four draws. It is not the weighted average of component negative log densities. Average agent-level losses within each group; do not first pool all agents into a different community predictor. Preserve full-observable losses even for agents who attended only one channel. Lower MSE and NLL are better.

Attended-channel evaluation can be an additional diagnostic, labelled separately. Do not equate decisive theory choice or persistent disagreement with predictive improvement.

### 9.6 Uncertainty, pairing, and freezing the analysis

The independent replication unit is the seed, not an agent or a time step. First calculate each seed's outcome or paired contrast, then aggregate across seeds. Use 2,000 whole-seed bootstrap resamples for mean outcomes and paired contrasts, with a fixed recorded analysis seed. A proposed analysis seed is `20260920`; this is a reproducibility setting, not evidence about the experiment.

Resample the same seed indices jointly across paired arms. Trajectory intervals obtained this way are pointwise 95% intervals, not simultaneous coverage of the entire curve. For seed-level pass fractions, report numerator and denominator and an appropriate binomial interval; a Wilson 95% interval is the proposed default. Do not infer certainty from all seeds meeting a criterion.

Freeze the primary outcomes, windows, conditions, seed list, and figure-selection rule before confirmation. Retain every completed seed, including nonseparating ones. Save failures and investigate numerical errors without silently replacing their seeds. Any scientific change after viewing outcomes creates a new exploratory version, followed by a separately identified confirmation if needed.

## 10. Result interpretation and optional follow-up

Do not prewrite a positive conclusion. Use the following distinctions when reporting results:

| Observation | Supported interpretation |
|---|---|
| `C10` separates but `C11` does not | Under these conditions, exchanging conditional beliefs differs from also exchanging theoretical support. |
| `C10_beta50` removes the gas group's negative scores | Stronger cross-group conditional influence changes how later local observations are evaluated. |
| The variance diagnostic removes the gas preference | The effect depends on the specified reference uncertainty; it is not purely a topological effect. |
| Higher conditional retention removes separation | Continued reference influence is important at the tested settings. |
| Model-weight-only sharing changes adoption but not conditional beliefs | The direct transmission of theoretical support explains that difference. |
| Both-channel sharing improves prediction beyond model-only sharing | Conditional exchange adds predictive value in that comparison. |
| Separation fails production confirmation | Report the failure and its diagnostics. Do not add score decay, different means, or weaker model sharing to the same frozen experiment until it separates. |

None of these four-arm comparisons establishes that two candidates are universally more accurate than one Gaussian, identifies a true causal graph, or demonstrates a historical fact about chemistry. A successful camp result has prescribed observation groups, restricted conditional influence, and private model weights. Explain these conditions where making the claim, not only in a distant qualification.

After delivering the required results, optional follow-up may include the full communication comparison at `beta=0.5`, the fixed grid `beta in {0,0.01,0.05,0.10,0.20,0.50}`, observation-mask swaps, access to both assays after the branch, or `omega=1`. Treat `beta=0` as within-group pooling rather than full isolation. Do not apply the stationary finite-retention calculation to `omega=1`.

Additional single-channel arms in the existing 16-agent adaptation experiment are secondary to the required 24-agent comparison. Further worlds, score-memory rules, reliability gates, and representation-count comparisons need separate scientific specifications and author approval.

## 11. Deliverables and completion criteria

Create a self-contained results handoff containing:

1. **Frozen protocol:** all conditions, matrices, references, seeds, timing, evaluation settings, and source identities, plus a dated record of any amendments.
2. **Code and environment:** runnable production-engine wrappers, analysis scripts, figure scripts, dependency record, and regression/invariant tests. Do not claim a hypothetical command exists; provide the actual tested entry points.
3. **Data:** per-seed observations or exact reproducible sample streams with hashes; per-round candidate scores, log weights and probabilities; observable conditional and mixture means; predictive losses; group metrics; initial and branch checkpoints including random state. Store covariance information sufficiently to reproduce evaluation. Deterministic precisions may be stored once per condition rather than duplicated across seeds, provided that equality is tested.
4. **Validation report:** reference compilation, switch behavior, recurrence residuals, restart equivalence, random-stream pairing, archived-regression discrepancies, and numerical failures. Include maximum errors and the precision/backend used.
5. **Summaries:** machine-readable per-seed and aggregate tables, paired effects, intervals, event and censoring records, and all nonseparating outcomes. Keep analytical predictions and simulated measurements in separately named fields.
6. **Figures:** vector PDF assets with candidate probabilities by group, individual preference distributions, measured score differences for the diagnostic controls, and predictive performance for the communication comparison. Use separate aligned panels for different quantities, label interval definitions and time conventions, and select illustrative seed traces by a rule fixed before viewing outcomes.
7. **Interpretive report:** what was run, what was not run, which hypotheses were supported, what the controls showed, remaining numerical or provenance limitations, and the narrow claims justified for the manuscript.

A suggested directory structure is:

```text
communication_experiments/
  README.md
  protocol.json
  provenance.json
  environment/
  source/
  tests/
  observations_or_stream_manifest/
  checkpoints/
  runs/
  analysis/
  figures/
  VALIDATION.md
  RESULTS.md
```

The task is complete when the required experiments have valid, reproducible outputs and an accurate account of their results, **not** when a target separation plot has been obtained. No adaptive simulations were run in preparing this brief.

## References to the supplied materials

**[R1]** `IWAI_camp_separation_recommendations(1).md`, *Camp separation in the revised phlogiston setup*. Principal basis: Sections 2–6 for model preservation, weighted conditional sharing, the four-arm communication design, diagnostics, and controls; Section 7 for deferred model-memory extensions. Its reported pilot is preliminary and was not verified from its unavailable support archive here.

**[R2]** `Camp_Report_Independent_Check.zip`, especially `camp_report_independent_check/README.md`, `check_report.py`, `results.json`, and `provenance.json`. Independent full-agent stationary-score checks; additional gas-variance diagnostic calculation; source-path verification. No adaptive trajectories or confirmatory pilot results are contained in this check.

**[R3]** `IWAI2026_camera_ready_source.zip`: `camera_ready/main.tex` and `camera_ready/provenance/{world.py,run.py,protocol.json,world.npz,graphs.npz,summary.json}`. Basis for candidate definitions, observation and evaluation conventions, archived conditions, prehistory branching, and the existing source-reconstruction helper.

**[R4]** `IWAI_camera_ready_handoff_2026-09-19.zip`: `IWAI_camera_ready_handoff_2026-09-19/paper/four_variable_experiments/engine/src/structural/rival.py` and its package dependencies. Basis for actual `RivalConfig` switches, initialization, predictive scoring, update order, and random-key consumption. Inspect the imported version and record its checksum before production runs.
