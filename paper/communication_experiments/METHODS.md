# Communication experiments: model, execution and interpretation

This suite implements the supplied experiment brief without changing the two candidate theories. Its interventions concern communication, conditional retention and one explicitly specified reference variance. Model-score forgetting, utility, reliability gates, active sensing and candidate generation are absent.

## General model

Agent $i$ maintains a conditional density $r_{ik,t}(x_k)$ within each supplied candidate $k$, and log support $L_{ik,t}$ for those candidates. Different candidates can use different latent coordinates. A common observation space allows their predictions to be compared without identifying their latent variables with one another. All agents retain and update both candidates, even when one has negligible probability.

Two communication channels act on an entering-state snapshot. Conditional sharing takes a logarithmic pool within the same candidate:

$$\bar r_{ik,t}(x_k)\propto\prod_j r_{jk,t}(x_k)^{W^x_{ij}}.\tag{1}$$

Model sharing acts on old log support:

$$\bar L_{ik,t}=(1-\alpha_m)L_{ik,t}+\alpha_m\sum_jW^m_{ij}L_{jk,t}.\tag{2}$$

Each agent observes a private noisy measurement of the prescribed common signal. Attention determines its working likelihood $\Lambda_{ik,t}$. Before depositing that measurement, the agent evaluates

$$\ell_{ik,t}=\log\int\Lambda_{ik,t}(y_{i,t}\mid x_k)\bar r_{ik,t}(x_k)\,dx_k.\tag{3}$$

Conditional learning and retention then give

$$r^+_{ik,t}\propto\Lambda_{ik,t}\bar r_{ik,t},\qquad
r_{ik,t+1}\propto(r^0_k)^{1-\omega}(r^+_{ik,t})^\omega.\tag{4}$$

Log support accumulates without decay:

$$L_{ik,t+1}=\bar L_{ik,t}+\ell_{ik,t},\qquad
q_{ik,t+1}=\operatorname{softmax}_k(L_{i,t+1}).\tag{5}$$

These are socially pooled model weights, not a claim that each agent computes a centralized Bayesian posterior over all community observations. Disabling either communication channel leaves local learning active. Equations (1)–(5) do not make attention or trust depend on $q$ in this suite.

## Gaussian specialization and attention

Production uses the unchanged accepted `src.structural.rival._step` in float64. For each candidate, $r=\mathcal N(\Pi^{-1}h,\Pi^{-1})$. Equation (1) pools $\Pi$ and $h$ linearly. With $D_i=\operatorname{diag}(\sqrt{a_i})$, the working likelihood is

$$D_i y_{i,t}\mid x_k\sim\mathcal N(D_iHx_k,\sigma_o^2I).\tag{6}$$

The raw observation is sampled as $y_{i,t}=H\phi_t+\epsilon_{i,t}$ with $\epsilon\sim\mathcal N(0,\sigma_o^2 I)$. The actual transformed noise covariance is therefore $\sigma_o^2\operatorname{diag}(a_i)$; it is not the working covariance in (6). A weight of 0.01 is weak evidence, not zero evidence or a change to the sampled raw noise. Unattended coordinates add identical score constants to both candidates.

$$J_i=H^\top\operatorname{diag}(a_i)H/\sigma_o^2,\qquad
j_{i,t}=H^\top\operatorname{diag}(a_i)y_{i,t}/\sigma_o^2.\tag{7}$$

$$\Pi_{ik,t+1}=(1-\omega)\Pi_k^0+\omega(\bar\Pi_{ik,t}+J_i),\qquad
h_{ik,t+1}=(1-\omega)h_k^0+\omega(\bar h_{ik,t}+j_{i,t}).\tag{8}$$

The prediction in (3) uses the post-sharing, pre-deposit state. Evaluation uses the post-update state and four new full-observation replicates of the current noiseless signal. Reported MSE compares the mixture mean with that noiseless signal. Reported NLL is the negative logarithm of the Gaussian mixture density, averaged over evaluation draws and then agents; it is not the mixture-weighted average of component NLLs.

## Candidate references

The storage coordinates are $(z,C,G,M)$; $z$ means release $P$ in candidate 0 and incorporated oxygen $O$ in candidate 1. Only $(C,G,M)$ are mixed, compared or evaluated across candidates. The directed dependencies are

- Release: $C\to P$ (0.8), $C\to G$ (0.8), $P\to M$ (−0.8).
- Oxygen: $C\to O$ (0.8), $O\to G$ (0.8), $O\to M$ (0.8).

Reference means are $(0,1,1,-1)$ and $(0,1,1,1)$ and residual variances are $(0.25,0.5,0.5,0.5)$. They are shared by all agents. The sole reference intervention changes release's gas residual variance to 0.5448, matching marginal gas variance to oxygen's 0.8648. It preserves the means and directed coefficients but changes the full precision and information vector. It does not match all covariances or isolate topology.

## Experimental design

| Suite | Signal and observation groups | Communication interventions | Replications |
|---|---|---|---|
| Network | 24 agents; common fully observed history through state 80; mass signal changes from −1 to +1 at round 80; four agents retain all channels, 20 have weights $(1,.01,.01)$ | Neither, conditional only, model weights only, both; complete uniform matrix; $\omega=.9$, $\sigma_o=1$ | Same 50 seeds as archive, 600 updates total |
| Camps | 16 agents; constant observable signal $(1,1,1)$; eight gas-only, eight mass-only | Same four channel combinations; group cross-weight $\beta=.05$; $\omega=.3$, $\sigma_o=.5$ | 50 fresh paired seeds, 5,000 updates |
| Camp diagnostics | Same signal, groups and paired samples | Conditional-only with $\beta=.5$, with $\omega=.9$, or matched gas variance; separate interventions | Same 50 seeds |
| Fixed intervention | Every conditional-only primary camp state at update 1,500 | Continue private model weights or enable model sharing at $\alpha_m=.02$; conditional sharing unchanged | All 50 seeds, continuation to update 5,000 |

For camps, each receiver assigns total conditional weight $\beta$ to the other group. Each cross-group entry is $\beta/8$. When model sharing is enabled its immediate cross-group contribution is $\alpha_m\beta=.001$. At $\beta=0$, agents still pool within groups. The groups and observation masks are assigned, not emergent.

The network experiment recomputes all four arms from each seed's single common prehistory. At the change it resets random keys to `seed+1000`, exactly as the archived wrapper did. This is a paired extension of an existing experiment, not a new independent confirmation of its original three arms. Camps use `PRNGKey(10000)` through `PRNGKey(10049)`, with exact keys carried through the state-1,500 branch. Evaluation uses `fold_in(entering_key,314159)` and never changes a training key.

## Pipeline pseudocode

```text
Compile the original references; compile the specified variance diagnostic separately.
Freeze conditions, seed lists, endpoints, windows and figure rules.
Validate reference compilation, independent switches and random-stream pairing.

For each network seed:
    Initialize identical candidate references and L = 0.
    Run 80 fully observed rounds with both channels enabled.
    Save the full state; reset phase-two random key to PRNGKey(seed + 1000).
    For each of the four communication arms:
        Clone that same state and key.
        Run rounds 80 through 599 under the changed signal and observation masks.

For each camp seed and each of seven specified conditions:
    Initialize that condition's references, L = 0 and PRNGKey(seed).
    For rounds t = 0,...,4999:
        Read a common entering-state snapshot.
        Share old log support if enabled.                         (2)
        Share conditional information within each candidate.     (1)
        Draw the same paired raw observation; score before learning. (3),(6)
        Deposit weighted information into every candidate.       (7)
        Apply conditional retention and record state t+1.        (8),(5)
        Evaluate an independent full-observation replicate mixture.
        Save histories and full state/key at prescribed boundaries.
    For primary C10 only:
        Resume from exact state 1500 with model sharing enabled.
        Retain the unchanged C10 continuation as the private arm.
        Verify exact private restart and conditional/score equality between arms.

Check direct score histories against the exact log-odds recurrence.
Compare the three old network arms with archived arrays.
Calculate every endpoint and paired contrast within seed first.
Aggregate over seeds with the frozen bootstrap; retain failures and censored events.
Render all preselected figures and report both supporting and limiting outcomes.
```

## Readouts and statistical conventions

Primary camp opposition means **group-average** $q_A(O)<.1$ and $q_B(O)>.9$ at a given state. Each seed passes a reporting window if this holds on at least 90% of its updates. The secondary all-individual event requires at least 90% of each eight-person group to pass its corresponding individual threshold, which requires all eight. These are different endpoints.

Windows are completed states 451–600 for networks, 1,126–1,500 for the first camp horizon, and 3,751–5,000 for the long horizon. Endpoints are reported separately. First crossings and 20-update sustained events retain both interval start and detection state; absent events are right-censored, never replaced by the horizon in an ordinary event-time mean. Seeds are the replication unit. Intervals for means and paired contrasts use 2,000 whole-seed bootstrap resamples with seed 20260920; trajectory intervals are pointwise, not simultaneous. Pass fractions use Wilson intervals.

Individual entropy $\langle H(q_i)\rangle$ is distinguished from between-agent disagreement $H(\langle q_i\rangle)-\langle H(q_i)\rangle$. Log odds are obtained from raw $L_O-L_P$, not from clipped probabilities or logits of group means. Predictive MSE and NLL are evaluated separately from preference and confidence.

## Mechanistic checks and limits

Let $z=L_O-L_P$ and $s=\ell_O-\ell_P$. The exact ungated recurrence is

$$z_{t+1}=[(1-\alpha_m)I+\alpha_mW]z_t+s_t.\tag{9}$$

For the equal weighted groups, model sharing contracts their group-average difference by $1-2\alpha_m\beta$ before the new scores arrive. The population average accumulates the average score. A bounded difference therefore does not imply permanently opposing choices when the population average drifts. With private model weights, each agent instead accumulates its own score stream. This calculation is a mechanism diagnostic, not a threshold-success guarantee.

The forecasts in `protocol.json` are stationary expectations including observation-induced variation in the conditional means. They are not simulation outcomes. `stationary_predictions.py` independently derives them where provided in this handoff. The primary setting was chosen analytically in the supplied proposal and a prior pilot was reported; this suite is confirmation after exploration. Missing external pilot trajectories are not represented as independently verified.

Any separation demonstrated here depends on prescribed observation groups, restricted conditional exchange, candidate reference influence and the specified communication channels. The suite does not establish permanent fixation under all noise realizations, unrestricted two-channel coexistence, discovery of a new candidate, causal identification, superiority over a single Gaussian, or a historical claim about chemistry.
