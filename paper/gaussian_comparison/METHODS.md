# Matched Gaussian camp comparison

This extension compares conditional-sharing mechanisms in the completed constant-world camp experiment. It retains the two supplied combustion candidates, observable signal, observation groups, sensor noise and base communication weights. It adds single-Gaussian baselines rather than changing the candidate model. It is a paired follow-up selected after seeing the earlier results, not an independent confirmation or a preregistered study.

## Common information and initial conditions

There are 16 agents. Agents 0–7 attend only gas uptake, and agents 8–15 attend only mass change. All receive private noisy samples of the same fixed observable signal $u^*=(C,G,M)=(1,1,1)$, with sensor standard deviation $\sigma_o=.5$. Sampling is independent of attention; attention specifies the channels used for learning. Each arm uses conditional retention $\omega=.3$, 5,000 updates, and the same 50 streams initialized by seeds 10000–10049. Reporting windows are completed states 1126–1500 and 3751–5000.

The base communication matrix is

$$W_\beta=\begin{pmatrix}(1-\beta)J_8&\beta J_8\\\beta J_8&(1-\beta)J_8\end{pmatrix},\qquad J_8=\mathbf1\mathbf1^\top/8,\quad\beta=.05.\tag{1}$$

The total weight assigned to the other group is .05; each cross-group entry is .00625. Self influence is included. The observation groups and matrix are supplied, not learned communities.

All agents in a single-Gaussian condition start at the same reference and retain that same reference. The comparison therefore introduces no camp-specific initial belief or recurring anchor. In the candidate conditions, every agent has the same two references and starts with equal candidate probabilities, as in the parent study.

## Two specified Gaussian reference constructions

First marginalize the original release and oxygen references to the *common* observable coordinates. Denote their means and covariances by $\mu_k^u,\Sigma_k^u$, for $k\in\{P,O\}$. The candidate-specific hidden coordinates $P$ and $O$ are never averaged or compared.

The primary reference uses an equal logarithmic pool of these observable marginals:

$$P_0=\tfrac12\big[(\Sigma_P^u)^{-1}+(\Sigma_O^u)^{-1}\big],\qquad h_0=\tfrac12\big[(\Sigma_P^u)^{-1}\mu_P^u+(\Sigma_O^u)^{-1}\mu_O^u\big].\tag{2}$$

Its mean is $P_0^{-1}h_0$, which generally differs from the arithmetic average of the two means. Correlations and unequal precision matter.

The sensitivity reference matches the mean and covariance of the equal candidate mixture:

$$\mu_0=\tfrac12(\mu_P^u+\mu_O^u),\qquad \Sigma_0=\tfrac12\sum_{k\in\{P,O\}}\big[\Sigma_k^u+(\mu_k^u-\mu_0)(\mu_k^u-\mu_0)^\top\big],\quad P_0=\Sigma_0^{-1},\ h_0=P_0\mu_0.\tag{3}$$

The between-mean term is essential. This reference matches two moments, not the full mixture density. Both constructions are compared under plain and gated sharing. They define four new conditions; neither construction is chosen on the basis of the resulting separation or prediction accuracy.

An independent, unobserved, mean-zero coordinate of variance .25 is prepended only to match the engine's four-dimensional storage format. It contributes neither to the gate nor to readouts. The underlying directed candidate coefficients remain unchanged.

## Social gating and the common update

Plain sharing uses $V_t=W_\beta$. Gated sharing computes the inherited mean-disagreement kernel from entering-state beliefs on $C,G,M$:

$$z_{ij,t}^2=\sum_{v\in\{C,G,M\}}\frac{(\mu_{jv,t}-\mu_{iv,t})^2}{1/(P_{i,t})_{vv}+1/(P_{j,t})_{vv}},\qquad\gamma_{ij,t}=\frac{\nu+1}{\nu+z_{ij,t}^2},\quad\nu=.01,\tag{4}$$

$$V_{ij,t}=\frac{W_{\beta,ij}\gamma_{ij,t}}{\sum_l W_{\beta,il}\gamma_{il,t}}.\tag{5}$$

The reciprocal diagonal precision in Eq. (4) is an exact Gaussian conditional variance, not a marginal variance. The approximation is to use these diagonal conditional variances in place of a full covariance distance. The value $\nu=.01$ comes from the earlier Gaussian comparison; it is not tuned here. The gate modifies conditional social influence, not observation reliability, utility or model support. Realized total cross-group influence is recorded per agent and update; it can differ from the base .05 after gating.

The wrapper supplies Eq. (5) to the unchanged engine with its internal gate disabled. This ensures that the existing gate kernel reweights the intended base matrix rather than replacing it with a normalized binary adjacency. The denominator is strictly positive because the self weight is positive. Exact row normalization therefore needs no additive epsilon. Equal entering means recover the base matrix, up to arithmetic rounding; the same holds as the kernel becomes constant.

With observation $o_{i,t}$, fixed attention $a_i$, and the original observation map $H$, the engine computes $J_i=H^\top\operatorname{diag}(a_i)H/\sigma_o^2$ and $j_{i,t}=H^\top\operatorname{diag}(a_i)o_{i,t}/\sigma_o^2$. Its update is

$$\bar P_{i,t}=\sum_j V_{ij,t}P_{j,t},\quad\bar h_{i,t}=\sum_j V_{ij,t}h_{j,t},\qquad P_{i,t+1}=(1-\omega)P_0+\omega(\bar P_{i,t}+J_i),\quad h_{i,t+1}=(1-\omega)h_0+\omega(\bar h_{i,t}+j_{i,t}).\tag{6}$$

Candidate scores are evaluated after sharing but before the new observation is deposited. For the new single-Gaussian conditions, the sole candidate probability is identically one; it is not a measure of confidence that oxygen is correct. The two-candidate comparisons retain their separately defined model-weight update. `C10` keeps those weights private; `C11` shares them with $\alpha_m=.02$. `C10_gasvar_matched` retains the parent variance control.

## Common readouts and uncertainty

Use each agent's posterior predictive mean on the observable coordinates, $\hat u_{i,t}$. For a single Gaussian this is its marginal mean; for candidates it is the probability-weighted mixture mean. The primary distance is

$$D_t=\left\|\frac18\sum_{i\in A}\hat u_{i,t}-\frac18\sum_{i\in B}\hat u_{i,t}\right\|_2.\tag{7}$$

The selected-candidate (MAP) distance is a secondary diagnostic. It can change abruptly when the maximizing candidate changes and is not interchangeable with Eq. (7). Neither distance alone establishes opposing theory choices. Those choices require the separate candidate probabilities.

The mean instantaneous distance includes private observation noise. To examine systematic displacement, also average the *signed vector difference* over the reporting window within each seed before taking its norm. A norm after averaging seeds is a third, distinct quantity. Every uncertainty interval resamples whole paired seeds, using 2,000 bootstrap draws with analysis seed 20260920; agents and timepoints are not independent replications.

Prediction is assessed by full-observable MSE against $u^*$ and negative log predictive density of four fresh full observations per agent per update. Held-out draws are paired with the parent candidate runs and never affect inference. Each Gaussian density includes sensor covariance; candidate NLL uses the complete predictive mixture. Lower losses are better. Per-group metrics avoid hiding different consequences for gas and mass observers.

For plain sharing, deterministic precision and expected information have the stationary equations

$$P_i^*=(1-\omega)P_0+\omega\sum_jW_{ij}P_j^*+\omega J_i,\qquad\mathbb E h_i^*=(1-\omega)h_0+\omega\sum_jW_{ij}\mathbb E h_j^*+\omega J_i u^*.\tag{8}$$

These equations apply on the three observable coordinates. Since $\omega<1$, the corresponding linear system is invertible. Unequal $J_i$ and a reference that does not match the signal can yield unequal expected means despite communication. A contraction of the pooling operator does not remove these repeated information inputs. The independent auditor derives numerical predictions before seeing the new simulation results. Eq. (8) is not asserted for the belief-dependent gate.

## Pipeline pseudocode

```text
Compile both observable candidate marginals.
For each reference construction, compile Eq. (2) or Eq. (3).
For each plain/gated condition and each paired seed:
    Initialize every agent at the same Gaussian reference.
    For each round t = 0,...,4999:
        Snapshot all entering beliefs and the RNG key.
        Set V = W from Eq. (1), or calculate Eqs. (4)–(5).
        Record realized influence arriving from the other group.
        Pool conditional information with V; draw the paired observation.
        Score the observation under post-sharing, pre-learning beliefs.
        Call the original engine to deposit evidence and retain references (Eq. 6).
        Record observable predictions and compute Eq. (7).
        Evaluate independent paired held-out observations; do not train on them.
        Save trajectories and exact initial/final states at chunk boundaries.
Verify stream pairing against the archived candidate runs.
Reduce paired seed/window metrics and compare Eq. (8) with the plain runs.
Render the main comparison and sensitivity figure with the recorded intervals.
```

This is a comparison of specified update mechanisms and reference constructions. Matched retention does not make a single Gaussian and a two-component mixture identical initial predictive distributions. Persistent numerical disagreement is not proof of irreducible paradigm disagreement, and separation is not itself a measure of predictive success.
