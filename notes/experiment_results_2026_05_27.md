# Experiment results — 2026-05-27

Simplified theory-laden model (`src/pomdp/simple_step.py`) with hierarchical
context variable. All runs use K=2 paradigms, Watts-Strogatz graph (N agents,
mean degree 4), fixed experiment at x=1.0 (obs_x_index=4).

---

## E1: Stationary environment, no social coupling

**Purpose:** Isolate the individual-level theory-ladenness effect without social
reinforcement. Agents start with uniform beliefs; truth is paradigm 1 throughout.

**Config:** N=40, T=200 steps, social_mask=0.0, q_reliability=0.55, D_c_normal=0.9.
Sweep: eps_resolve × eps_crisis, 5 seeds each.

**Result:** Without social coupling, all agents eventually converge to truth. The
speed of convergence degrades smoothly with eps_resolve (paradigm inertia) and
improves with eps_crisis (anomaly detection rate):

| eps_resolve | ec=0.05 | ec=0.10 | ec=0.20 |
|-------------|---------|---------|---------|
| 0.01        | 1.00    | 1.00    | 1.00    |
| 0.10        | 1.00    | 1.00    | 1.00    |
| 0.30        | 0.99    | 1.00    | 1.00    |
| 0.50        | 0.96    | 1.00    | 1.00    |
| 0.70        | 0.92    | 0.99    | 1.00    |
| 0.90        | 0.88    | 0.97    | 1.00    |

**Interpretation:** Theory-ladenness alone slows but never prevents convergence.
An isolated agent always eventually learns the truth — the context transition B_c
biases toward normal science, but anomalous evidence accumulates and eventually
overwhelms. This is the baseline: paradigm lock-in requires social coupling.

---

## E2: Discrete paradigm shift — the phase diagram

**Purpose:** Map the (q_reliability, eps_resolve) phase boundary between paradigm
adaptation and lock-in after a discrete shift in truth.

**Setup:** Truth = paradigm 0 for t<100, flips to paradigm 1 at t=100. Agents
start uniform, commit to paradigm 0 during the first 100 steps, then must adapt.
N=40, T=300 steps, eps_crisis=0.20, D_c_normal=0.9, social_mask=1.0.
Sweep: q_reliability × eps_resolve, 10 seeds each.

**Result — phase diagram** (mean final belief in TRUE paradigm, averaged over seeds):

```
              er=0.01   er=0.05   er=0.1    er=0.2    er=0.3    er=0.5    er=0.7
qr=0.500      0.969     0.968     0.967     0.964     0.959     0.948     0.933
qr=0.505      0.943     0.927     0.910     0.878     0.829     0.692     0.534
qr=0.510      0.869     0.779     0.695     0.490     0.274     0.060     0.014
qr=0.515      0.640     0.429     0.259     0.072     0.014     0.000     0.000
qr=0.520      0.318     0.117     0.041     0.004     0.000     0.000     0.000
qr=0.525      0.090     0.016     0.003     0.000     0.000     0.000     0.000
qr=0.530      0.013     0.003     0.000     0.000     0.000     0.000     0.000
qr=0.540      0.000     0.000     0.000     0.000     0.000     0.000     0.000
qr=0.550      0.000     0.000     0.000     0.000     0.000     0.000     0.000
qr=0.600      0.000     0.000     0.000     0.000     0.000     0.000     0.000
```

**Key findings:**

1. **Sharp phase boundary.** The transition from adaptation (~1.0) to lock-in (0.0)
   occurs over a narrow band of q_reliability (roughly 0.505–0.525). This is the
   signature of a collective phase transition, not a smooth individual-level effect.

2. **Diagonal boundary.** Both axes contribute: stronger social coupling (higher qr)
   AND more paradigm inertia (higher eps_resolve) push toward lock-in. You can
   compensate for strong social coupling with low inertia, and vice versa.

3. **Social coupling dominates.** Above qr ≈ 0.53, NO amount of individual openness
   (even eps_resolve=0.01) can break the lock-in. The social feedback loop creates
   an absorbing state that individual-level parameters cannot escape.

4. **Contrast with E1.** Without social coupling (E1), even eps_resolve=0.9 reaches
   mean_qB=0.88. With social coupling at qr=0.53, eps_resolve=0.01 gives 0.01.
   Social coupling transforms gradual resistance into absolute lock-in.

**Interpretation:** This IS the Kuhnian paradigm shift story. The phase boundary
represents the critical social coupling threshold above which a scientific community
becomes locked into an outdated paradigm regardless of evidence. Below the threshold,
individuals can enter "crisis" and update; above it, the collective reinforcement
overwhelms any individual's anomaly detection.

---

## E3: Slow drift (running, results pending)

**Purpose:** Same phase diagram but under a slow ramp (truth drifts from paradigm 0
to paradigm 1 over 200 steps). Tests whether the lock-in survives continuous drift
or whether gradual evidence accumulation can break the social barrier.

**Config:** N=40, T=400 steps, ramp from t=50 to t=250, eps_crisis=0.20,
D_c_normal=0.9. Sweep: q_reliability × eps_resolve, 10 seeds each.

*Results to be added when run completes.*

---

## Parameter glossary

| Parameter | Symbol | Role |
|-----------|--------|------|
| q_reliability | κ = log(q/(1-q)) | Social coupling strength (softmax-precision on the social channel) |
| eps_resolve | P(c'=normal \| c=crisis) | Paradigm inertia: how quickly crises are resolved back to normal science |
| eps_crisis | P(c'=crisis \| c=normal) | Anomaly sensitivity: how readily agents detect paradigm-inconsistent evidence |
| D_c_normal | P(c=normal) at t=0 | Initial theory-ladenness (how committed agents are to normal science) |
| social_mask | ∈ [0,1] | Attenuation on the social channel (0=isolated, 1=full coupling) |
| obs_x_index | index into x_grid | Which experiment generates observations (controls discriminability) |

---

## For the Wednesday call

The E2 phase diagram is the headline figure. It shows:
- Paradigm lock-in is a **collective** phenomenon — it requires social coupling above
  a critical threshold, not just individual stubbornness.
- The boundary is **sharp** (phase-transition-like), not gradual.
- Theory-ladenness (eps_resolve) modulates WHERE the boundary falls but doesn't
  change its character.
- This maps directly onto the abstract's claim: "beliefs on which much else depends
  become correspondingly harder to revise" — here, the "much else" is the social
  network's mutual reinforcement.
