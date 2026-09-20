# Camp separation in the revised phlogiston setup

## Recommendation and experimental plan

**Basis:** the four-variable camera-ready manuscript and source archive supplied for review, not the older ten-variable/six-variable experiments. **Status:** a proposed extension, supported by new equation-level calculations and an independent pilot. The pilot is not a rerun of the authors' missing adaptive engine, and its results should not be inserted into the paper without implementation checks and confirmatory runs.

## 1. Recommendation

Keep the revised phlogiston and oxygen candidates, their reference means, and the common observable world unchanged. First test **selective observation with predominantly within-group conditional sharing and private model weights**. Then switch model-weight sharing on, using identical observations, to test whether it removes the separation.

This is the smallest change I recommend for demonstrating actual opposing theory preferences while retaining cross-group communication. It uses operations already defined in the revised paper: distinct matrices for conditional and model-weight communication, and the existing switch that disables model-weight sharing. It does not require reversing the gas signal, inventing a new candidate, assigning opposite initial convictions, or adding a preference for a particular theory.

There is already a concrete basis for this recommendation. With eight gas observers, eight mass observers, the existing constant signal, conditional retention `omega = 0.3`, and 5% total cross-group conditional influence, an independent implementation of the displayed equations produced opposing group preferences in all 20 realizations at both 1,500 and 5,000 completed updates. Adding model-weight sharing at the paper's `alpha_m = 0.02` removed the separation in that pilot. Section 4 gives the results and limitations.

**The resulting claim is different from the old paper's claim.** It would demonstrate that local theory choice can remain divided despite exchange of within-theory beliefs, and that transmitting theory support can remove that division. It would not demonstrate that model-weight sharing itself preserves opposing schools. This distinction should be explicit, rather than trying to recover the old conclusion by changing the experiment until it appears.

For the camera-ready revision, I recommend two additions:

1. Complete the existing communication ablation with **model-weight-only sharing**, especially in the changed-signal network experiment where the conditional-only result is already interesting.
2. Add the controlled two-group communication experiment below, with genuine camp separation measured through candidate probabilities, not only distance between quantity estimates.

An optional finite-memory extension for model weights is specified in Section 7. That extension addresses coexistence with both channels active, but introduces a new learning assumption and is not necessary for the minimal result.

## 2. What stays unchanged, and why the current experiment does not separate camps

### 2.1 Preserve the revised scientific setup

Use the existing candidate states

\[
x_P=(P,C,G,M)^\top,\qquad x_O=(O,C,G,M)^\top,
\]

with the supplied directed coefficients and residual variances. The shared observables remain combustion extent \(C\), net gas uptake \(G\), and solid mass change \(M\). Preserve the observable reference means

\[
\mu_P^{u,0}=(1,1,-1)^\top,\qquad
\mu_O^{u,0}=(1,1,1)^\top.
\]

Use the constant signal

\[
u_t=(1,1,1)^\top
\]

and the same independent observation noise. Group A observes gas, \(w_A=(0,1,0)\), and group B observes mass, \(w_B=(0,0,1)\). Start every agent with both candidates at their references and with \(q(P)=q(O)=1/2\). These are the revised constant-world definitions, not a newly balanced world. [S1, §§3.1, 4.4, Appendices B.2 and D.1; S2, `world.py`]

In particular, **do not give the gas signal opposite signs in the two candidates**. The revised setup deliberately assigns the same positive gas reference mean to both. Introducing opposing gas means would undo an important aspect of that revision. [S2, `wall_truth()` and reference specification]

### 2.2 Keep separate the two things that communication can change

Write the candidate log odds and new predictive evidence as

\[
z_{i,t}=L_{iO,t}-L_{iP,t},\qquad
s_{i,t}=\ell_{iO,t}-\ell_{iP,t}.
\]

The revised update is

\[
z_{t+1}=P_m z_t+s_t,
\qquad P_m=(1-\alpha_m)I+\alpha_m W^m.
\]

Model-weight sharing directly mixes old log odds. Conditional sharing changes the predictive distributions used to compute the next \(s_{i,t}\). Consequently, setting \(\alpha_m=0\) does not make theory choice immune to communication: shared conditional beliefs can still change which candidate predicts an agent's observations better. [S1, §2.2 and Appendix E]

For a fixed connected model-weight network, its stationary weights \(v\) give

\[
c_{t+1}=c_t+v^\top s_t,\qquad c_t=v^\top z_t.
\]

The disagreement component may stabilize while this shared component continues to grow. In the present full-sharing constant-world experiment, both groups favour oxygen; a bounded log-odds gap is not a pair of opposing preferred theories. [S1, Appendix E]

Under stationary, integrable score processes with a positive network-averaged long-run drift and stable disagreement dynamics, the cumulative shared evidence will dominate bounded-order disagreement. Reducing a fixed positive model-sharing rate then delays agreement rather than establishing permanent coexistence. This is the relevant inference from the recurrence; it is not an unconditional theorem for arbitrary time-varying learning systems.

Two tempting modifications do not solve the problem by themselves:

- Different initial model probabilities change the starting state, not a continuing positive common drift.
- Exactly balancing *expected* opposing evidence removes the mean drift but generally leaves stochastic accumulation in the consensus direction. A zero-drift noisy accumulator is not a restoring mechanism or a stable fixed probability distribution.

## 3. Minimal new experiment: connected conditional sharing, private theory choice

### 3.1 Make cross-group influence explicit

Let each group contain \(n=8\) agents and define \(J_n=\mathbf 1_n\mathbf 1_n^\top/n\). Use

\[
W_\beta=
\begin{pmatrix}
(1-\beta)J_n&\beta J_n\\
\beta J_n&(1-\beta)J_n
\end{pmatrix},\qquad 0\leq\beta\leq\tfrac12.
\]

Each receiver gives total influence \(1-\beta\) to its own observation group and \(\beta\) to the other. Self-influence is included. At \(\beta=1/2\), this is the paper's uniform complete-graph pool. For every \(0<\beta\leq1/2\), all cross-group conditional weights are strictly positive: this is not a disconnected conditional network.

The matrix is a proposed **weighted** communication rule. Do not describe it as the paper's original unweighted adjacency normalization. Its symmetry and double stochasticity keep the groups equally influential, avoiding a centrality confound.

Use the following primary condition:

| Setting | Proposed value |
|---|---|
| Candidates, reference means, reference covariances | Unchanged from the revised constant-world setup |
| Agents and observation groups | 16 agents; 8 gas-only, 8 mass-only |
| Common signal | \((1,1,1)^\top\), with no change point |
| Observation noise standard deviation | \(0.5\) |
| Conditional retention | \(\omega=0.3\), as in the existing constant-world comparison |
| Conditional sharing | \(W^x=W_{0.05}\) |
| Model-weight sharing | Disabled: \(\alpha_m=0\) |
| Initial candidate probabilities | \(1/2,1/2\) for every agent |
| Primary horizon | 1,500 completed updates; retain the paper's plotting convention separately |
| Persistence continuation | 5,000 completed updates |
| Confirmatory replications | 50 new, independently seeded runs after freezing the design |

The proposed cross-group weight was selected through the analytical exploration described below, before the pilot. It is not a preregistered parameter. Freeze it and the sensitivity grid before using new seeds for confirmation.

The group assignment represents different observation access and stronger contact within observational specialties. It is prescribed, not an emergent network community. What can emerge from neutral initial probabilities is the **difference in preferred theory**.

### 3.2 Why separation is possible without changing the world

With model-weight sharing disabled,

\[
z_{i,T}=z_{i,0}+\sum_{t=0}^{T-1}s_{i,t}.
\]

With fixed observation weights and \(\omega<1\), the conditional Gaussian dynamics form a stable driven linear system after the deterministic precision transient. Their limiting score process has finite moments and is stationary and ergodic. Thus, after accounting for the decaying initialization transient,

\[
\frac{z_{i,T}}{T}\longrightarrow\bar s_i.
\]

If gas observers have \(\bar s_A<0\) and mass observers have \(\bar s_B>0\), their probabilities approach opposite candidates. This is a condition on the *learned predictive scores*, not an assumption that gas and mass evidence have been assigned equal and opposite strengths.

I calculated those stationary expected scores from the revised Gaussian equations for the proposed communication matrix. The calculation includes the variability of the communicated predictive means; it does not simply score the average prediction. Appendix A specifies it.

| Total cross-group conditional influence \(\beta\) | Gas-group \(\mathbb E[s]\) | Mass-group \(\mathbb E[s]\) |
|---:|---:|---:|
| 0, with complete sharing inside each group | −0.003808 | +0.542359 |
| 0.01 | −0.003717 | +0.550473 |
| **0.05** | **−0.003022** | **+0.582599** |
| 0.10 | −0.001663 | +0.622196 |
| 0.20 | +0.001836 | +0.700296 |
| 0.50, the uniform complete pool | +0.013773 | +0.931657 |

Values are nats per agent per round at \(\omega=0.3\), \(\sigma=0.5\), and eight agents per group. They are **new analytical predictions**, not numbers reported in the supplied manuscript. The \(\beta=0\) row is not the fully isolated condition: conditional sharing still occurs inside each group.

These calculations predict a useful transition. With private model weights, weak cross-group conditional exchange leaves opposite score signs; sufficiently strong exchange makes both groups favour oxygen. The calculated gas-score sign change is near \(\beta=0.15\), not a universal threshold.

This is not just a finite-distance effect. Opposite nonzero asymptotic score signs imply opposite asymptotic candidate probabilities for this private-weight configuration. Nevertheless, validate the implementation and report actual finite-horizon probabilities rather than presenting an unverified asymptotic argument as an empirical result.

### 3.3 What supplies the gas group's preference?

Both candidates predict the same gas reference mean, but their gas reference variances differ:

\[
\operatorname{Var}_P(G)=0.82,\qquad
\operatorname{Var}_O(G)=0.8648.
\]

These are the manuscript's observable marginal variances. [S1, Appendix B.2]

In a world of repeated measurements of a fixed gas value, the different predictive dispersion can give release a small expected log-score advantage when mass information has limited conditional influence. The mass group's oxygen advantage is much larger. Increasing conditional exchange allows mass observations to change the competing gas predictions and can remove the gas group's release advantage.

Therefore describe the result as **separation sustained by selective observation, reference-dependent prediction, and restricted cross-group conditional exchange**. It is not evidence that gas measurements historically supported phlogiston, that the two theories are equally well supported, or that disagreement is generated purely by graph structure.

The retention setting matters substantially. At \(\omega=0.9\) and \(\beta=0.05\), the same calculation gives positive scores in both groups, approximately \(+0.001131\) and \(+0.014426\). Keep this negative result as a planned sensitivity check. A claim that the effect is independent of memory would be false.

## 4. Independent pilot already performed

I implemented the revised Gaussian communication, pre-observation scoring, learning, and reference-memory equations in a separate NumPy calculation. It uses each candidate's three-observable marginal, which is equivalent for these updates because the hidden conditional factor is shared and remains fixed. The previous audit established that equivalence. [S3, §2]

The pilot used \(\beta=0.05\), \(\omega=0.3\), \(\sigma=0.5\), 16 agents, and 20 independent realizations from a generator initialized with master seed `24681357`. It is not the authors' seed protocol. All four communication conditions received identical observation samples. All started with neutral candidate probabilities and the original references.

In the following table, “model sharing on” means \(W^m=W_{0.05}\), \(\alpha_m=0.02\). Conditional sharing on means \(W^x=W_{0.05}\); off means \(W^x=I\).

| Communication | Mean gas-group \(q(O)\), update 1,500 | Mean mass-group \(q(O)\) | Realizations with opposing groups at update 1,500 | At update 5,000 |
|---|---:|---:|---:|---:|
| Neither channel | 0.000656 | Approximately 1 | 20/20 | 20/20 |
| Conditional only | **0.0130** | **Approximately 1** | **20/20** | **20/20** |
| Model weights only | Approximately 1 | Approximately 1 | 0/20 | 0/20 |
| Both channels | Approximately 1 | Approximately 1 | 0/20 | 0/20 |

“Opposing” uses the manuscript's simultaneous criterion: gas-group mean \(q(O)<0.1\), mass-group mean \(q(O)>0.9\). In the conditional-only arm, gas-group mean probability was approximately \(6.13\times10^{-7}\) at update 5,000. All realizations satisfied the opposing-group criterion throughout the final quarter of the 5,000-update pilot.

The conditional-sharing pilot's final-quarter mean score increments were approximately \(-0.003028\) and \(+0.581809\), consistent with the stationary predictions \(-0.003022\) and \(+0.582599\).

These results are preliminary implementation evidence, not a replacement for the paper's experiments. The pilot does not reproduce the absent engine, its random-number streams, its full diagnostics, or its predictive-loss evaluation. It does not establish superiority in prediction. Its code, output, and stationary-score calculation are supplied in the accompanying support archive.

An important additional control is \(\beta=0.5\) with conditional sharing only. The analytical calculation predicts eventual oxygen preference in both groups there, matching the qualitative outcome already reported for the paper's constant-world conditional-only condition. This tests whether cross-group conditional influence, rather than just disabling model-weight sharing, explains the new separation. [S1, Appendix D.1]

## 5. Complete the model-weight-only ablation

### 5.1 Four conditions, not three

For any fixed world, attention schedule, reference construction, and communication matrix \(W\), run:

| Condition | \(W^x\) | \(\alpha_m\) and \(W^m\) | Message content |
|---|---|---|---|
| Neither | \(I\) | \(0\) | No social messages |
| Conditional only | \(W\) | \(0\) | Within-candidate distributions |
| **Model weights only** | **\(I\)** | **\(0.02,\ W\)** | **Relative support for candidates** |
| Both | \(W\) | \(0.02,\ W\) | Both kinds of belief |

All conditions retain two candidates and update **every candidate** from the agent's own observation. Do not stop learning a low-probability candidate. Model-weight-only means that conditional states are not fused; it does not mean that the conditional distributions are frozen or that the agent receives raw observations.

### 5.2 Priority: the existing changed-signal network experiment

Complete the four conditions on the existing complete graph with 24 agents, \(\omega=0.9\), \(\sigma=1\), and \(T=600\). Keep the common fully observed, fully communicating 80-round prehistory. At the first changed observation, branch all four arms from exactly the same state. Four agents retain full observation; twenty receive \(w=(1,0.01,0.01)\). [S1, §4.3 and Appendix D.2; S2, `run.py`]

The existing restricted-group final oxygen probabilities are approximately 0.7723 with neither channel, 0.1458 with conditional sharing only, and 1 with both. Corresponding tail predictive NLLs are approximately 4.6659, 4.6155, and 4.3073. The missing model-weight-only cell distinguishes whether transmitting support is sufficient for adoption and whether conditional exchange adds predictive value after support is transmitted. [S1, §4.3 and Appendix D.2]

Do not insert a predicted numerical answer into the manuscript. The constant-world pilot is a different experiment and cannot fill this network table.

For each arm report restricted and informed groups separately: candidate probability, time to sustained adoption, per-agent preference distributions, all-observable MSE, and all-observable mixture NLL. Preserve the four independent held-out observations per agent per round and the last-quarter evaluation window used by the manuscript.

### 5.3 Also use the four conditions in the new camp experiment

At \(W=W_{0.05}\), run the complete factorial design from neutral initial probabilities. Repeat it at \(W=W_{0.5}\) as the well-mixed control. This links the new separation experiment directly to the original communication question.

If resources permit, also add both missing single-channel arms to the existing 16-agent, changed-signal adaptation experiment, branching after its shared prehistory. That improves comparability but is secondary to completing the published network ablation.

### 5.4 Pairing and implementation checks

The source wrapper constructs `RivalConfig` with `alpha_m` and `fuse_nets`. The intended model-weight-only configuration is `alpha_m = 0.02, fuse_nets = False`. The existing network helper's `all_off` path also sets `alpha_m = 0`, so it cannot be reused unchanged for this new cell. Expose the two switches independently. Verify their actual behavior in `src.structural.rival` before using them. That engine is not included in the supplied ZIP. [S2, `run.py`]

The protocol also contains `rho_L = 0.0`. Do not infer its meaning from its name or assume it implements the finite-memory proposal below. The revised mathematical specification accumulates model scores without forgetting. Inspect and test the engine's definition first. [S1, Eq. (6); S2, `protocol.json`]

Useful invariants follow from the published model. With fixed observations and attention, conditional learning does not depend on current model probabilities. Therefore:

- Neither-channel and model-weight-only arms must have the same conditional parameters and score increments at each round.
- Conditional-only and both-channel arms must also have the same conditional parameters and score increments at each round.

Only their model log weights should differ within each pair. An unexplained difference in conditional states indicates different random samples, a changed attention schedule, a hidden dependence on model support, or an implementation error.

Generate observation noise and held-out evaluation noise independently of arm-specific control flow. In the changed-signal experiments, reuse the same prehistory state and post-change observation arrays within each seed.

For a chosen outcome \(Y\), let \(Y_{ab}\) denote conditional sharing \(a\in\{0,1\}\) and model sharing \(b\in\{0,1\}\). Report paired effects such as

\[
Y_{01}-Y_{00},\qquad Y_{11}-Y_{10},\qquad
(Y_{11}-Y_{10})-(Y_{01}-Y_{00}).
\]

The last contrast is a scale-specific factorial interaction, not a universal measure of synergy. Interpret probability, MSE, and NLL separately. Do not use an uncensored arithmetic mean of threshold times when some runs never cross.

## 6. Define a convincing separation result

### 6.1 Primary outcome

Retain the existing probability-based definition:

\[
I_t=\mathbf 1\{\bar q_{A,t}(O)<0.1\ \text{and}\ \bar q_{B,t}(O)>0.9\}.
\]

Compute it within each realization before aggregating across realizations. Report the fraction of tail rounds with \(I_t=1\), the fraction of realizations satisfying the criterion for at least 90% of tail rounds, the longest continuous opposing interval, and any subsequent loss of separation. Predeclare the 90% persistence threshold as a reporting convention, not a theoretically unique boundary.

Also report the proportions of individual agents strongly preferring each candidate. A group mean can hide minorities or broad uncertainty. Plot at least several per-realization trajectories rather than relying only on a ribbon around a population average.

### 6.2 Mechanism diagnostics

Record individual log odds, their signed group means, the group gap, the shared component, and the pre-update score increments. Estimate tail score means with uncertainty over independent realizations; do not treat rounds from one trajectory as independent replicates.

Use the entropy decomposition already defined in the paper:

\[
\mathcal H(\bar q)-\frac1N\sum_i\mathcal H(q_i)
=\frac1N\sum_iD_{\rm KL}(q_i\|\bar q).
\]

Confident opposing groups have low mean individual entropy but high between-agent disagreement. High entropy inside every agent is uncertainty, not polarization. [S1, Appendix C.1]

Continue to record signed observable differences and the existing selected-candidate distance, but treat them as secondary outcomes. Record these before communication, after communication, after learning, and after memory. Do not mix the candidate-specific hidden coordinates \(P\) and \(O\). A nonzero norm of a noisy difference is not evidence of a nonzero expected difference. [S1, shared-observable readouts; S3, §§1–2]

### 6.3 Prediction and observation controls

Use common full-observable held-out samples to measure predictive costs of separation. Supplement these with held-out performance on each group's attended channel, where useful, without confusing either evaluation with the score that changes candidate weights.

The most informative planned controls are:

| Control | Question addressed |
|---|---|
| Increase \(\beta\) to 0.5 | Does stronger conditional exchange remove opposing score signs? |
| Switch model-weight sharing on after camps have formed | Can transmission of theory support dissolve existing separation? |
| Give both groups full observation after camps form | Does direct access to mass evidence remove the gas group's preference? |
| Swap the groups' observation masks | Does theory preference track information access rather than an agent label? |
| Repeat with \(\omega=0.9\) and with no conditional forgetting | How dependent is separation on continuing reference influence? |
| Match the gas reference variance as a separate diagnostic | How much does the gas group's preference depend on marginal dispersion? |
| Remove sampling noise while retaining positive working likelihood variance | Does the relevant result reflect expected-score differences rather than ongoing estimation noise? |

For the variance diagnostic, the manuscript's Appendix F already gives a useful intervention: increase the release gas residual variance from 0.5 to 0.5448. Use this as a **separately labelled control**, preserving the main experiment's means; do not silently replace the main candidate family. With no cross-group conditional exchange, the gas marginal filters then coincide and give identical gas-only scores. With cross-group exchange, other dependencies can still matter, so measure the result rather than assuming it. [S1, Appendix F; direct consequence of equal gas marginals]

For the noise control, distinguish sampling noise from the likelihood's assumed noise. Setting the working Gaussian variance to zero would make this update singular. A deterministic observation sequence with working \(\sigma=0.5\) is a valid diagnostic, but is a changed data-generating condition and must be labelled accordingly.

## 7. Optional extension: both communication channels remain active

### 7.1 Why a further assumption is needed

The primary result has genuine cross-group conditional exchange but no model-weight exchange. That distinction cannot be hidden. With fixed positive model-weight mixing and positive global score drift, the unmodified accumulator does not provide robust stationary opposing preferences.

One transparent extension is **finite memory of model evidence with the same neutral reference for all agents**. This is conceptually related to adaptive social-learning work that introduces forgetting or adaptation to reduce inertia, but the exact update and coexistence analysis below are proposed here, not taken as a theorem from that literature. [R1]

### 7.2 Define the extension exactly

Introduce a new retention parameter \(0<\lambda<1\), separate from conditional retention \(\omega\). After the existing model-weight communication and before adding the new score, use

\[
\widetilde L_{ik,t}=(1-\lambda)\log q_k^0+\lambda\bar L_{ik,t},
\qquad
L_{ik,t+1}=\widetilde L_{ik,t}+\ell_{ik,t},
\]

with \(q_P^0=q_O^0=1/2\) for everyone. Additive constants common to both candidates do not matter. The log-odds recurrence is

\[
z_{t+1}=\lambda P_mz_t+s_t.
\]

This discounts **old** log support, not the new observation. Discounting after scoring would instead multiply the new score by \(\lambda\), giving different coefficients. Pick one convention and test it. Do not call this ordinary cumulative Bayesian model selection.

For a stationary score process with finite moments, the stable filter has mean

\[
\mathbb E[z_*]=(I-\lambda P_m)^{-1}\mathbb E[s].
\]

With two symmetric groups, total cross-group model-sharing fraction \(r=\alpha_m\beta_m\), and group score means \(s_A,s_B\),

\[
\mathbb E[c_*]=\frac{(s_A+s_B)/2}{1-\lambda},
\qquad
\mathbb E[g_*]=\frac{s_A-s_B}{1-\lambda+2\lambda r}.
\]

These imply

\[
\mathbb E[z_{A,*}]=\mathbb E[c_*]+\tfrac12\mathbb E[g_*],
\qquad
\mathbb E[z_{B,*}]=\mathbb E[c_*]-\tfrac12\mathbb E[g_*].
\]

For \(s_A=-a<0\), \(s_B=b>a\), opposite stationary mean signs require

\[
\lambda r(b-a)<(1-\lambda)a.
\]

For strongly opposing probabilities, the mean-log-odds targets \(z_A<-\log9\), \(z_B>\log9\) are useful screens, but are not sufficient empirical criteria: logistic transformation and expectation do not commute, and observation noise causes fluctuations.

### 7.3 What the actual score asymmetry implies

Using the proposed \(\beta_x=0.05\) conditional network, \(\omega=0.3\), and the calculated scores \(s_A\approx-0.003022\), \(s_B\approx0.582599\):

- With \(\lambda=0.999\) and \(r=10^{-6}\), the stationary mean log odds are approximately \(-2.44\) and \(582.01\).
- With the same \(\lambda\) but \(r=0.001\), corresponding to \(\alpha_m=0.02\), \(\beta_m=0.05\), both means are strongly positive.

The first pair is an **analytical illustration**, not a simulated coexistence result or a claim that the expected gas probability equals the logistic of −2.44. With \(\beta_m=0.05\), reaching \(r=10^{-6}\) requires \(\alpha_m=2\times10^{-5}\), one thousandth of the paper's rate. The model-weight channel is then extremely weak relative to local evidence accumulation.

This is why I would not make this extension the primary camera-ready result. Finite model-evidence memory alone does not solve the substantive asymmetry of the present gas and mass scores at ordinary sharing strength. A both-channel coexistence claim should report the required effective cross-group influence, not merely state that the graph is connected.

With noisy observations, describe successful finite-memory results as stationary separation with a measured probability or occupancy. Do not promise that no agent will ever cross a preference threshold again.

### 7.4 Alternatives I would defer

Camp-specific recurring model priors could maintain opposing commitments, but would build different loyalties into the dynamics. Belief-dependent trust that effectively closes cross-group edges could also preserve camps, but the explanation would be selective rejection or effective disconnection. Endogenous experimental choice is interesting, but changes the question and requires a policy model.

None is inherently illegitimate. Each needs an explicit scientific interpretation and its own controls. For this revision, the two-channel comparison offers a cleaner, smaller extension. Established Gaussian social-learning work also makes clear that consensus conclusions depend on the particular update and assumptions; the new result should not be advertised as a general refutation of social-learning convergence. [R2]

## 8. Suggested execution order and manuscript changes

First restore the authors' adaptive engine and verify the current experiments. Add explicit tests of the two independent communication switches and of the log-odds recurrence. Reproduce the independent pilot's qualitative behavior on the same prescribed observation arrays before using the engine's own seed protocol.

Then run the missing model-weight-only arm of the 24-agent network experiment. Run all four arms of the new constant-world experiment at \(\beta=0.05\) and \(0.5\). Freeze a sensitivity grid such as \(\beta\in\{0,0.01,0.05,0.10,0.20,0.50\}\) and \(\omega\in\{0.3,0.9,1\}\); the zero-contact and no-forgetting conditions are controls, not settings covered by every stationary calculation above. Use new confirmatory seeds and retain nonseparating outcomes.

A compact presentation would replace the current persistence interpretation with a figure showing: group candidate probabilities under conditional-only and both-channel sharing; gas and mass long-run score means across \(\beta\); and predictive losses from the four-condition ablation. Keep the earlier full-sharing noise-distance finding as a control or appendix result, not as evidence of opposing schools.

**Proposed wording, conditional on reproduction and confirmation:**

> Under selective observation, agents can retain opposing candidate preferences despite exchanging within-candidate beliefs across groups. This occurs when conditional exchange is predominantly within observation groups and model weights remain private. Stronger cross-group conditional exchange changes the local predictive-score ordering, while transmitting model weights removes the separation in the tested conditions. The effect depends on reference beliefs and conditional retention; it does not establish coexistence under arbitrary connected communication.

For the model-weight-only addition, state the measured outcome rather than prewriting a positive conclusion. The relevant question is whether sharing support, sharing conditional predictions, or their combination explains adoption and predictive performance.

Release the actual engine, configuration files, raw trajectories or sufficient checkpointed arrays, figure scripts, exact seeds, and environment. Include score histories, not only final probabilities: the sign and size of the score sources are central to the interpretation.

## Appendix A. Stationary-score calculation behind the recommendation

This appendix specifies the independent calculation so that the proposed parameter choice is auditable.

Work on the three observable marginals in order \(C,G,M\). Let \(\Pi_k^0=(\Sigma_k^{u,0})^{-1}\), \(h_k^0=\Pi_k^0\mu_k^{u,0}\), and

\[
D_A=\sigma^{-2}\operatorname{diag}(0,1,0),\qquad
D_B=\sigma^{-2}\operatorname{diag}(0,0,1).
\]

Use \(D_c\) for the observation precision increment to avoid confusion with the averaging matrix \(J_n\). At the group level, the conditional communication matrix is

\[
B_\beta=\begin{pmatrix}1-\beta&\beta\\\beta&1-\beta\end{pmatrix}.
\]

All agents in a group have identical precisions, since precisions depend on the observation masks rather than measured values. Stack group indices and solve, componentwise in matrix or vector entries,

\[
\Pi_{c,k}=(1-\omega)\Pi_k^0
+\omega\sum_d(B_\beta)_{cd}\Pi_{d,k}+\omega D_c,
\]

\[
m^h_{c,k}=(1-\omega)h_k^0
+\omega\sum_d(B_\beta)_{cd}m^h_{d,k}+\omega D_cu.
\]

Thus the two-group solve uses \(I_2-\omega B_\beta\). This is invertible for \(\omega<1\).

Let \(h^{\rm grp}\) denote the six-dimensional vector of group-average information vectors for one candidate. Its stationary noise covariance \(C_h\) solves

\[
C_h=FC_hF^\top+Q,
\quad F=\omega(B_\beta\otimes I_3),
\quad Q=\frac{\omega^2}{n}\operatorname{blockdiag}(D_A,D_B).
\]

This expression for \(Q\) uses the binary masks and actual sampling variance \(\sigma^2\). For fractional weights it must be replaced with the covariance of the corresponding information increments; do not copy it unchanged.

Before scoring,

\[
\bar\Pi_{c,k}=\sum_d(B_\beta)_{cd}\Pi_{d,k},\qquad
\bar m^h_{c,k}=\sum_d(B_\beta)_{cd}m^h_{d,k},
\]

and

\[
\bar C_h=(B_\beta\otimes I_3)C_h(B_\beta\otimes I_3)^\top.
\]

The mean of the communicated predictive mean is

\[
\bar\mu_{c,k}=\bar\Pi_{c,k}^{-1}\bar m^h_{c,k},
\]

and its sampling covariance is

\[
C^\mu_{c,k}=\bar\Pi_{c,k}^{-1}(\bar C_h)_{cc}\bar\Pi_{c,k}^{-\top}.
\]

For the active coordinate \(r_c\), gas or mass,

\[
V_{c,k}=\sigma^2+(\bar\Pi_{c,k}^{-1})_{r_cr_c},
\]

\[
\mathbb E[\ell_{c,k}]
=-\frac12\left[
\log(2\pi V_{c,k})+
\frac{\sigma^2+(u_{r_c}-\bar\mu_{c,k,r_c})^2+(C^\mu_{c,k})_{r_cr_c}}{V_{c,k}}
\right].
\]

Inactive observation rows contribute equal constants to the two candidates, so they cancel in \(\mathbb E[s_c]=\mathbb E[\ell_{c,O}]-\mathbb E[\ell_{c,P}]\). The variance of the predictive mean is essential: omitting it would evaluate a deterministic average predictor instead of the actual stochastic filter.

For the completely isolated condition, compute each agent's scalar-filter covariance separately; it corresponds to the no-cross-group calculation with \(n=1\), not the \(n=8\) within-group pooling row in Section 3.

## Appendix B. Source and result provenance

**[S1] Revised manuscript.** `camera_ready/main.tex` and `camera_ready/main.pdf` in `IWAI2026_camera_ready_source(1).zip`, titled *Paradigm Shifts: Theory Revision and Belief Sharing in Epistemic Communities*. Relevant locations: §§2.2–2.3, §§3.1–3.2, §§4.3–4.4, Appendices B.2, C, D.1–D.2, E, and F. Numerical results attributed to the paper are archived results, not independently rerun here.

**[S2] Supplied experiment specification.** The same archive's `provenance/world.py`, `provenance/run.py`, `provenance/protocol.json`, and reference arrays. These specify the revised candidate family, observation masks, wrapper options, and branch design. The underlying adaptive engine is absent from the supplied package.

**[S3] Previous technical audit.** `TECHNICAL_AUDIT.md` supplied with the earlier review: stationary noise explanation for the residual distance; observable-marginal equivalence; scope of the validation and missing engine.

**[P1] New calculations and independent pilot.** `IWAI_camp_separation_support.zip` accompanying this note contains `stationary_scores.py`, `pilot.py`, `pilot_result.json`, `stationary_results.json`, and instructions. These are new work for this recommendation. The pilot supports a proposed extension; it is not a reproduction of a previously published result. No changed-signal network prediction has been represented as a completed experiment.

**[R1] Related research, checked externally.** Virginia Bordignon, Vincenzo Matta, and Ali H. Sayed, *Adaptive Social Learning*, arXiv:2004.02494, version 2 (2021). Used only to situate evidence-memory/adaptation as an established research direction. The exact proposed recurrence, threshold calculation, and coexistence claims in Section 7 require their own justification.

**[R2] Related research, checked externally.** James Z. Hare, Cesar Uribe, Lance Kaplan, and Ali Jadbabaie, *Non-Bayesian Social Learning with Gaussian Uncertain Models*, arXiv:1910.11251 (2019). Relevant background on communicating beliefs under model uncertainty; its convergence results are not automatically results for the revised paper's two-channel, finite-retention update.

## Bottom line

The most defensible addition is not a new world engineered to favour each group equally. It is a controlled demonstration, in the existing revised world, that **selective observation plus limited conditional exchange can sustain opposing private theory choices, whereas sharing theory support or sufficiently increasing conditional exchange can remove that separation**. The independent calculation and pilot make this a concrete, testable recommendation. The completed model-weight-only ablation would show which channel is responsible for which outcome.
