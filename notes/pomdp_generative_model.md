# Categorical-POMDP generative model — Phase-0 design lock

*Created 2026-05-20. This is the gate for Phase 1. It pins the four decisions the
plan requires before any multi-agent scaffold is written. Companion to
`dynamical_systems_diagnosis.tex`, notebook 15 (go/no-go PASS) and notebook 14
(monostability of the old substrate).*

> **Status (updated 2026-05-20): plan APPROVED; reframed per the David-Hyland
> conversation** — fixed-state epistemic POMDP, theory = per-paradigm likelihood
> (option B), **belief-utility is the core mechanism**, resource dropped (§5 →
> `notes/archive/`). Immediate next step is the **Phase 0.5 environment-coupled
> fold gate** (extend `social_fold.py`), before the `gen_model/agent_pop/step`
> scaffold. `paper/` untouched until Phase 4.

---

## 0. What Phase 0 established

- **Notebook 15 (go/no-go, PASS).** A *categorical* paradigm belief with a
  *softmax-precision* social channel produces a clean supercritical pitchfork at
  $K_c=2$ (mean field + finite-$N$ agree). The bifurcation parameter is
  **trust-as-message-precision** $\kappa=\log\frac{q}{1-q}$, $K=\text{reports}\cdot\kappa/\lambda$.
  This is the fold the linear substrate provably could not host.
- **Notebook 14 (Jacobian scan).** The existing continuous substrate has spectral
  radius $\rho\equiv$ forgetting-rate $<1$ across the whole resource box, and its
  one nonlinearity (λ·U) blows up rather than folding. Tier-B parameter repair is
  ruled out. The rebuild is justified.
- **pymdp 1.0 API (0a, read-only).** Confirmed facts that shape the design below.

---

## 1. The pymdp 1.0 JAX API (what we build on)

`inferactively-pymdp 1.0.1` is installed; it is the JAX/equinox rewrite
(`pymdp/agent.py`, `control.py`, `inference.py`, `learning.py`,
`distribution.py`), with a numpy fallback in `pymdp/legacy/`.

| Feature | Where | Why it matters for us |
|---|---|---|
| **`batch_size=N` is vmapped** | `agent.py:813` (`vmap(infer_states)`), `:891` | N agents = one `Agent` with a leading batch dim; each carries its own A/B/C/D. This is our population container. |
| **`categorical_obs=True`** | `agent.py:763`, `process_obs` `:622` | An observation may be a *soft probability vector*, not a one-hot. A neighbour's reported paradigm-belief feeds in directly. |
| **`mask` arg of `infer_states`** | `agent.py:788` | Per-modality down-weighting toward a uniform likelihood — the natural hook for **trust-as-precision** on the social modality. |
| **`infer_policies` → `(q_pi, neg_efe)`** | `agent.py:853` | Returns the policy posterior and EFE/`G` directly — two of our three order parameters, free. |
| **`gamma` (policy precision), `alpha` (action precision)** | `agent.py:363` | Per-batch scalars; available as additional precision levers. |

**Load-bearing limitation:** pymdp does the *within-agent* inference loop and
nothing else. **All inter-agent routing is ours** (see §3). The `legacy` numpy
backend is a *different architecture* (Python agent-loop, not vmap) and API —
treat as last resort, not a perf knob.

---

## 2. The generative model (A / B / C / D) — David reframing (2026-05-20)

Fixed-state epistemic POMDP. Two observation modalities, one control factor.
Start with **2 paradigms** ($K=2$); generalises to $K>2$.

### Hidden state and the s-vs-θ factorization (PIN AT PHASE-1 START)
Confirmed: **theory = a per-paradigm likelihood $A_\theta(o\mid s,a)$ (option B)** —
agents holding different paradigms read the same $(s,a)$ into *different* predicted
observations, so **theory-ladenness is structural**. The factorization that needs
to be nailed before coding (cheap to get wrong, expensive to unwind):

- **$s$ — state of nature.** FIXED; the real data-generating world. **$B_s$ =
  identity** (actions are *measurements*, not control — David's core point).
- **$\theta$ — the paradigm/theory the agent holds.** This is the *switchable*
  factor: what shifts, what couples socially, and what folds. The agent infers
  $q(\theta)$ (e.g. `P(circular)=0.3`); population $q(\theta)$ occupancy is order
  parameter #1. Conditioned on $\theta$, the likelihood is $A_\theta(o\mid s,a)$.

**Does $\theta$ have its own transition?** For a *fixed truth* faithful to David,
$B_\theta=$ identity too, and paradigm "shift" = movement of the belief $q(\theta)$,
not of a state. Consequence: with strict $B=$ identity the posterior is an
*absorbing ratchet* — capture is an absorbing consensus (Zollman / Planck "one
funeral at a time"), and bistability = **two absorbing basins** selected by initial
conditions + belief-utility tilt + social coupling. The NB15 leak $\lambda$
(`posterior_rho`) becomes an *optional* forgetting knob (needed only for a
stationary pitchfork diagram / sustained switching), **not** load-bearing for
capture. *Open: keep $B$=strict-identity (absorbing capture) vs. a tiny $B_\theta$
leak (recurrent dynamics). Phase 0.5 informs this.*

### `A_θ` (likelihood) — modality 1: own experiment outcome (per-paradigm)
**Reframing of `src/world.py`, option B.** The regression family
$o\sim\mathcal N(h_0(x)+\theta h_1(x),\sigma^2)$ is *discretized* into
$A_\theta[o\mid s,x]$ — **one likelihood per paradigm θ**, binned over $n_o$
outcome categories. Theory-ladenness = the $A_\theta$ differ, so the *same*
$(s,x)$ gives a differently-interpreted observation under each paradigm.
**Theory discriminability $d$** (per-row KL between paradigms' predicted columns,
controlled by $x$ and $n_o$) is the key knob: low $d$ = underdetermined /
theory-laden data (weak environment pull); high $d$ = sharply disambiguating data
(strong pull that **flattens the fold** — this is the Phase 0.5 gate). Keeps the
relativistic-correction *story* ($h_0=x/2$, $h_1=x^3$) as structure; relabels the
information-geometric content honestly.

### `A_social` — modality 2: neighbours' reports (the social channel)
Soft `categorical_obs` = **trust-weighted aggregate of neighbours' emitted
$q_j(\theta)$** (built in §3). Likelihood is the reliability matrix
$A_{\text{social}}=\big(\begin{smallmatrix} q & 1-q\\ 1-q & q\end{smallmatrix}\big)$,
$q=\sigma(\kappa)$, $\kappa$ = message precision — the NB15 fold object. (Read-out
behind a swappable seam; default softmax-precision, cross-inhibition / stigmergic
field deferred — see §3, §4.)

### `C` + **belief-utility `U(θ)`** — THE CORE MECHANISM (in EFE, not the update)
This is the centrepiece, not a mild penalty. Belief-utility is a preference over
*which paradigm is true*, **separate from** the probability $q(\theta)$
(`P(circular)=0.3` but `U(circular)=0.9`). It enters **policy evaluation**: the
agent prefers experiments expected to leave it believing the high-utility
paradigm — roughly add $\sum_\theta U(\theta)\,q(\theta\mid\pi)$ to the EFE/policy
score. This is **motivated reasoning, formalized**, and the **Hyland λ·U done
right**: it shapes *which experiment you choose*, never the Bayesian update — so it
avoids the precision-additive blow-up that killed λ·U in NB14. **Pin: belief-utility
lives in the policy score, never in the state-precision update.** Heterogeneous
$U(\theta)$ is the **symmetry-breaking field** that turns NB15's symmetric
polarization into asymmetric **capture**. (`C` over the outcome modality stays flat
→ epistemic foraging baseline; [[feedback_hyland_aif_default]].)

### `D` (prior) and heterogeneity
$D$ = initial $q(\theta)$ prior per agent (the entrenched-paradigm lever,
[[project_notebook_09_paradigm_competition]]). Test whether categorical $D$-skew
*persists* above the fold threshold (it should — distinct basins), unlike the
$\mu_0/\tau_0$ wash-out in the linear substrate ([[project_v2_hysteresis_findings]]).

### Control factor
One control factor selecting the **experiment $x$**. `infer_policies` gives
$q(\pi)$ and EFE (now including the belief-utility term) — order parameter #2. (We
do **not** add an attention-allocation control in Phase 1; the memetic
attention-hyperprior is a deferred Phase-3 layer, kept seam-open in §3 — distinct
from Albarracin's per-agent Dirichlet.)

---

## 3. Inter-agent coupling — the real engineering (lives in OUR code)

pymdp gives zero help here. Each `step` (in `src/pomdp/step.py`,
operating on the batched `Agent` from `agent_pop.py`):

1. **Emit.** Each agent $j$ emits a paradigm signal from its current posterior:
   $r_j = q_j(s^{(0)})$ (a soft 2-vector) — sampled or passed soft. *Decision:*
   start with **soft emission** ($r_j=q_j$) to match notebook 15's probabilistic-
   report version that cleanly isolated the softmax-precision mechanism (hard MAP
   votes add a confounding threshold nonlinearity).
2. **Route through trust.** Build agent $i$'s social observation as the
   **trust-weighted mixture** of in-neighbour emissions:
   $$o^{\text{soc}}_i = \sum_{j\in\mathcal N(i)} T_{ij}\, r_j,\qquad \textstyle\sum_j T_{ij}=1,$$
   where $T_{ij}$ is the **trust weight** (row-stochastic). This $o^{\text{soc}}_i$
   is a soft categorical vector fed via `categorical_obs=True`.
3. **Precision via reliability + mask.** The *strength* of the social update is
   set by the reliability $q=\sigma(\kappa_i)$ inside `A_social`, optionally
   modulated per-agent through the `mask` arg (down-weighting the social modality
   for low-precision agents). **$\kappa$ = trust-as-message-precision is the
   bifurcation knob.**
4. **Infer / act / transition.** `infer_states([o_world_i, o_soc_i], prior)` →
   `infer_policies` (record `q_pi`, `neg_efe`) → `sample_action` → environment
   returns next $o_world$ → `update_empirical_prior`.

The graph $\mathcal N(i)$ and adjacency come from **`src/network.py`
`build_adjacency` (reused unchanged)**.

---

## 4. Trust = message precision (retained, but no longer the differentiator)

"Trust" is **how much weight agent $i$ puts on a *received* signal from $j$** — the
reliability/precision pair $(T_{ij},\kappa)$ in §3. Post-David, the
precision-vs-attention fight is **relaxed**: the novelty is now **belief-utility +
theory-as-likelihood** (§2), so trust need not be defended as "not Albarracin's
attention." We keep the message-precision reading because it is the parameter that
drove the NB15 fold and is the social-HGF reading — but it is a supporting actor.

**Learning rule (Phase 2):** $T_{ij}$ / $\kappa$ from the history of how well $j$'s
past reports predicted $i$'s own outcomes — a reliability estimate (Gamma-conjugate
idea re-expressed on the categorical channel). For Phase 1, $T$ is fixed from the
graph and $\kappa$ is a swept scalar. The §3 emit→route seam is kept open so a later
**stigmergic field / attention-hyperprior** (Phase 3, [[project_mediating_field_synthesis]])
can modulate routing without rearchitecting.

---

## 5. ~~Resource coupling~~ — DROPPED 2026-05-20 (superseded by belief utility)

**Resource dynamics are dropped** from the rebuild as of the 2026-05-20 David Hyland
conversation. The full resource-coupling design (two-channel inflow, the three closed
loops, and the `resource→κ` plan that was pinned here) is preserved in
`notes/archive/resource_coupling_setup.md`.

**Belief utility (Hyland λ·U) supersedes it** as the symmetry-breaking / paradigm-capture
mechanism. Agents carry a preference over *which theory is true* (e.g. `P(circular)=0.3`
but `U(circular)=0.9`), separate from their posterior probability. This enters the
**EFE / policy evaluation** (which experiment to run) — **NOT** the state-precision update
(the precision-additive form is exactly what blew up in notebook 14; belief utility lives
in `C`/`gamma`, see §2). Heterogeneous belief-utilities tilt the NB15 symmetric pitchfork
into genuine *capture* — the role `resource→κ` was going to play, now grounded in
motivated reasoning rather than money. Full belief-utility spec to be pinned in §2
(`C`/preferences), pending confirmation of the "theory = action↔state likelihood" reading.

---

## 6. Order parameters (wired from Phase 1)

`src/pomdp/observables.py` (extends `src/observables.py` conventions):
1. **Paradigm occupancy** — fraction with $\arg\max q_i(s^{(0)})=B$ (and the soft
   mean $\bar q(B)$). The notebook-15 $m$.
2. **EFE & policy posterior** — population stats of `neg_efe` and `q_pi` from
   `infer_policies`; the AIF-native readout David asked for.
3. **Polarization / bimodality** — bimodality coefficient or cluster separation of
   $\{q_i(B)\}$; distinguishes consensus from a split population.

(Hysteresis-area was dropped by the user; not a primary observable.)

---

## 7. Phase-1 module layout (for after review — NOT yet built)

```
src/pomdp/
  social_fold.py      # DONE (0b)
  jacobian_probe.py   # DONE (0c)
  gen_model.py        # build A/B/C/D (+pA for trust learning) from a PomdpConfig
  agent_pop.py        # batched pymdp Agent over the network; init from build_adjacency
  step.py             # the §3 inter-agent cycle: emit -> route -> infer -> act
  attention.py        # NO — renamed: trust.py (message-precision update, §4) in Phase 2
  observables.py      # the three order parameters, §6
```
`PomdpConfig` follows the `src/config.py` frozen-dataclass pattern.

---

## 8. Open questions to resolve at review

1. Outcome discretization granularity $n_o$ for `A_world` — coarse (2–3 bins) vs
   fine. Affects how faithfully the relativistic-correction story survives.
2. Soft vs sampled emission (§3.1): soft is cleaner; sampled is more "realistic"
   and reintroduces finite-$N$ noise. Default soft; revisit.
3. Trust learning rule exact form (§4) — defer to Phase 2 start, but sanity-check
   it reduces to the Gamma-conjugate reliability estimate.
4. ~~Whether factor 1 (experimental regime) earns its place.~~ **RESOLVED:** hidden
   states are FIXED ($B$=identity), so there is no regime-transition factor. Drop it
   for Phase 1 — reintroduce only if exogenous paradigm *shocks* are wanted as an
   experiment. (Keep this question alive for the David conversation on
   exogenous-vs-agent-modeled paradigm dynamics.)
5. **NEW (load-bearing):** the exact $s$-vs-$\theta$ factorization and whether
   $B_\theta$ is strict identity (absorbing capture) or has a small leak (recurrent
   dynamics) — see §2. Phase 0.5 informs it.
6. **NEW:** exact form of the belief-utility EFE term ($\sum_\theta U(\theta)q(\theta\mid\pi)$
   added to the policy score) and its regression test against flat-$U$ → pymdp `G`.

---

**Hard stop.** Phases 1–4 (scaffold, phase diagram, resource/trust re-graft +
Azoulay falsifier, paper rewrite) require human sign-off per the approved plan.

---

## 9. Simplified theory-laden model (2026-05-27)

**Status: BUILT AND TESTED.** Per David Hyland's suggestion (2026-05-26), a
simplified model drops the full POMDP action loop for initial experiments. Theory-
ladenness is implemented as a principled hierarchical context model.

### What changed

The full POMDP scaffold (§3 emit→route→act→infer) hit the **martingale wall**:
belief-utility in the policy cannot flip the inference basin. David suggested:
"even just a latent state + observations + message passing + motivated reasoning
should be enough." The team agreed on **Option 1 (theory-ladenness)** from
`where_we_are_2026_05_21.tex` §6.

### The context model

A latent context variable `c ∈ {0, 1}` is added to the joint state `(θ, c)`:

| Context | Interpretation | A_world columns |
|---------|---------------|----------------|
| c=0 ("normal science") | Data does not discriminate paradigms | Paradigm-averaged (identical across θ) |
| c=1 ("crisis/anomaly") | Data fully discriminates paradigms | Original `A_world[:, θ, x]` |

The agent infers the **joint** posterior `q(θ, c)` via standard Bayes on a
`K×C`-dimensional state. The joint posterior is carried forward through a proper
transition model `B = B_θ ⊗ B_c`, where `B_θ = identity` and:

```
B_c = [[1 - eps_crisis,   eps_resolve  ],
       [    eps_crisis,  1 - eps_resolve]]
```

- `eps_crisis`: P(c'=crisis | c=normal) — spontaneous anomaly detection rate
- `eps_resolve`: P(c'=normal | c=crisis) — crisis resolution rate (paradigm inertia)

After inference, the marginal `q(θ) = Σ_c q(θ, c)` is emitted for social coupling.
Lock-in emerges from the dynamics: committed agents see uninformative data (c=0),
reinforcing c=0 via B_c, further insulating beliefs from discriminating evidence.

### Why this is principled AIF

- The context prior is **not reset** each step — the joint posterior carries forward
  through B_c, so the agent accumulates evidence about whether it is in normal
  science or crisis.
- `eps_crisis` and `eps_resolve` are transition model parameters (part of B), not
  hand-tuned observation weights.
- The existing `agent_pop.infer_state` is reused unchanged — it works on any state
  dimension S; setting S = K*C with the joint matrices requires no modifications.
- Social observations are the marginal q(θ); `A_social_joint` maps the K-dimensional
  social obs to the K*C-dimensional state (context-independent).

### Module: `src/pomdp/simple_step.py`

```
SimpleConfig       — wraps PomdpConfig + context transition params
SimpleState        — carries q_joint (N, K*C) between steps
build_joint_A_world, build_joint_A_social, build_B_c  — called once per run
transition_joint   — applies B_c within each θ block (efficient, no K*C × K*C matrix)
simple_step        — transition → observe → emit → route → infer
run_simple         — thin Python loop, returns trajectories + context diagnostics
```

### Test results (18 tests in `tests/test_simple_step.py`)

- Joint model algebra: normalized columns, c=0 paradigm-independent, c=1 matches original
- Transition: preserves normalization, converges to stationary, identity at trivial B
- Context learns from evidence: agents shift from c=0 toward c=1 under anomalous data
- Capture/escape pair: high eps_resolve → capture; high eps_crisis + low eps_resolve → escape
- Environment drift wired via `theta_schedule`

### Experiment configs (E1–E3)

```
experiments/configs/E1_stationary.yaml     — sweep eps_resolve × q_reliability
experiments/configs/E2_discrete_shift.yaml — paradigm shift at t=150, sweep eps_resolve
experiments/configs/E3_slow_drift.yaml     — ramp schedule, sweep eps_resolve × social_mask
```

Runner: `python -m experiments.run_experiment <config.yaml> [--dry-run]`
