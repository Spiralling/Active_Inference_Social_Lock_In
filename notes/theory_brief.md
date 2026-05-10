# Theory Brief — Variational Paradigm Dynamics

**Working draft for David Hyland** · 2026-05-08
**Status:** notation locked enough to start coding; equations open for revision.

## What we're claiming

We extend Hyland & Albarracin (2025), *On the Variational Costs of Changing Our Minds*, from single-agent belief revision to a coupled multi-agent setting on a small-world network. The variational cost of mind-change generates **Kuhnian paradigm-shift dynamics**: communities lag behind a slowly-drifting environment, with hysteretic phase transitions in their adaptation. The headline empirical claim is a (λ, environment-drift-rate) phase diagram with hysteresis, dissociating cleanly from Bayesian rule-induction baselines (Oldenburg & Zhi-Xuan 2024).

## Single-agent recap (your 2025 paper)

In standard active inference, an agent updates beliefs by minimising variational free energy

$$
F[q] = \mathbb{E}_q[\ln q(s) - \ln p(o, s)]
= \underbrace{\mathrm{KL}[q(s) \,\|\, p(s)]}_{\text{complexity}} - \underbrace{\mathbb{E}_q[\ln p(o \mid s)]}_{\text{accuracy}}.
$$

Your move was to re-foreground the complexity term as a **motivational cost of mind-change** rather than a regulariser: the agent experiences resistance to revising its belief proportional to KL-divergence between the new posterior and its current prior. We adopt that reading and lift it to the population level.

## Multi-agent setting

### Population

$N=30$ agents on a Watts-Strogatz small-world graph $\mathcal{G}=(V, E)$ with rewiring probability $\beta=0.1$, mean degree $k=6$.

### Per-agent state

Each agent $i \in V$ carries:

| symbol | meaning |
|---|---|
| $C_i \in \mathbb{R}^{|\mathcal{O}|}$ | log-prior preferences over observation outcomes |
| $\{\gamma_{ij}\}_{j \in N(i)}$ | edge precisions (trust on incoming edges) |
| $d_i \in [0, 1]$ | delegation factor — weight on social vs. own observation; per-agent constant, drawn from $\text{Beta}(\alpha, \beta)$ with mean $\sim 0.7$ |
| $\lambda_i \geq 0$ | individual cost of mind-change (the parameter we sweep) |

### Environment

Two contexts $c \in \{1, 2\}$ with appropriate-action hidden state $s^*[c] \in \{0, 1\}$. Three evolution regimes for $s^*$:

- **stationary noise** — $s^*$ fixed with sub-percent random flips (baseline)
- **slow drift** — $s^*$ does a Bernoulli random walk with small step probability (for hysteresis)
- **discrete shift** — $s^*$ flips deterministically at a single time $t^*$ (for lag-under-shock)

### Update equations

At each timestep $t$, each agent $i$ aggregates two observation streams — direct environmental and social:

$$
q_i^{(t+1)}(o) \;\propto\; (1-d_i)\,\mathrm{p}_{\text{env}}(o \mid \mathrm{obs}_i^{(t)}; \gamma_{\text{env}})
\;+\; d_i \sum_{j \in N(i)} w_{ij}^{(t)}\, p_j^{(t)}(o)
$$

with social weights $w_{ij}^{(t)} = \gamma_{ij}^{(t)} / \sum_{k \in N(i)} \gamma_{ik}^{(t)}$.

The C-update is gated by your variational cost of mind-change:

$$
C_i^{(t+1)} \;=\; \arg\min_{C} \Bigl[\,
\mathrm{KL}\bigl[q_i^{(t+1)} \,\|\, p(o \mid C)\bigr]
\;+\; \lambda_i \cdot \mathrm{KL}\bigl[p(o \mid C) \,\|\, p(o \mid C_i^{(t)})\bigr]
\,\Bigr]
$$

— accuracy of new C against the just-computed posterior, *penalised* by the cost of moving away from the previous C. Setting $\lambda_i = 0$ recovers the Bayes-optimal update; $\lambda_i \to \infty$ freezes $C_i$. The asymmetry of KL is what generates hysteresis at the population level.

### Edge precision (trust) update

Trust is **evidence-coupled**: $\gamma_{ij}$ rises when $j$'s past predictions matched the environmental ground truth, falls otherwise. Multiplicative form:

$$
\gamma_{ij}^{(t+1)} \;=\; \gamma_{ij}^{(t)} \cdot \exp\bigl[-\eta_\gamma \cdot e_j^{(t)}\bigr]
$$

with $e_j^{(t)} = -\ln p_j^{(t)}(o^{*}_{t})$ the negative-log-likelihood that $j$ assigned to the *true* environmental observation $o^*_t$ at step $t$. Bad predictors lose precision exponentially. Trust update rate $\eta_\gamma$ is fixed (default ~0.1, small relative to the C-update rate).

## Predicted phenomena

| | low $\lambda$ | high $\lambda$ |
|---|---|---|
| **stationary** | C tracks environmental noise tightly | C drifts slowly around true $s^*$ |
| **slow drift** | C tracks $s^*$ smoothly | hysteretic: forward/reverse adaptation curves diverge; loop area $\propto \lambda$ |
| **discrete shift** | rapid recovery (a few timesteps) | persistent lag; community sticks to old paradigm well past $t^*$ |

The slow-drift regime is the headline — it's where the variational cost of mind-change generates the cleanest signature of the paradigm-shift dynamics.

## Dissociation from rule-induction (the formal argument)

Oldenburg & Zhi-Xuan (2024) treat norms as discrete obligative/prohibitive rules induced from compliance and sanction observations via Bayes' rule. Their inference is symmetric in time: the same evidence accrual that locks in a rule will, when reversed, unlock it on a comparable timescale. They have no asymmetric cost.

**Proposition (informal).** *In the limit $\lambda \to 0$, our model recovers symmetric environment-tracking on slow drift; for $\lambda > 0$, the forward and reverse adaptation curves diverge.* The hysteresis loop area is bounded below by a quantity monotone in $\lambda$.

This is testable: run both models on identical environment sequences; symmetric tracking vs. hysteretic loops is a single-figure result.

## Experimental plan

| ID | environment regime | sweep | observable | timeline |
|---|---|---|---|---|
| E1 | stationary | n/a (baseline) | dynamics stable | May 14 |
| E2 | discrete shift | $\lambda$ | adaptation lag | May 15-17 |
| E3 | slow drift | $\lambda$ (forward + reverse) | hysteresis loops | May 18-21 |
| E4 | slow drift | model (ours vs. rule-induction) | symmetric vs. hysteretic | May 24-27 |

## Open questions for you

1. **The trust-update form.** Multiplicative exponential decay is parsimonious but ad-hoc. Is there a free-energy-principled form you'd prefer — e.g., $\gamma_{ij}$ as itself a posterior over $j$'s reliability, updated Bayesianly? Trade-off: principled vs. extra parameters/state.

2. **The KL-asymmetry handle.** The $\lambda \cdot \mathrm{KL}[p_{C_{\text{new}}} \,\|\, p_{C_{\text{old}}}]$ term is what generates hysteresis. Should we instead consider the *symmetric* form (mean of forward and reverse KL, i.e. Jensen-Shannon)? My intuition says no — the asymmetry is exactly what we want — but worth flagging.

3. **Your contribution slot.** Most natural place for you to take ownership is §4 of the paper (the multi-agent model spec). Either own the derivation outright or audit Jonas's. Either works.

4. **The neighbour-coupled cost question.** An earlier formulation had $\lambda_i$ itself depending on local consensus around $C_i$ (Granovetter-cascade-flavoured). I dropped that in favour of evidence-coupled $\gamma_{ij}$, which is cleaner and avoids the trivial echo-chamber failure mode. Worth your sanity-check: do you see anything we lose by not coupling $\lambda$ to the neighbourhood?

## What this brief does and doesn't do

- **Does**: lock notation enough to start coding `src/environment.py` and `src/network.py`; give you a concrete object to react to.
- **Doesn't**: derive the bifurcation order from information-geometric curvature (we explicitly dropped that for IWAI scope; conjecture-confirmed-empirically is the framing).

## Citations to anchor

- Hyland & Albarracin 2025, "On the Variational Costs of Changing Our Minds", arXiv:2509.17957
- Albarracin, Demekas, Ramstead, Heins 2022, "Epistemic Communities Under Active Inference", *Entropy* 24(4):476 — the multi-agent AIF setup we extend
- Constant, Albarracin, Friston 2025, "Normative Active Inference", arXiv:2511.19334 — single-agent normative precursor
- Oldenburg & Zhi-Xuan 2024, "Bayesian Rule Induction for Normative Multi-Agent Systems", AAMAS — dissociation target
- Watts & Strogatz 1998, "Collective dynamics of small-world networks", *Nature* 393:440-442

— Jonas, May 8 2026
