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

**Result — phase diagram** (mean final belief in TRUE paradigm, averaged over seeds):

```
              er=0.01   er=0.05   er=0.1    er=0.2    er=0.3    er=0.5    er=0.7
qr=0.500      0.970     0.941     0.928     0.928     0.927     0.919     0.908
qr=0.505      0.927     0.851     0.787     0.714     0.624     0.406     0.226
qr=0.510      0.801     0.540     0.367     0.169     0.056     0.005     0.001
qr=0.515      0.488     0.161     0.048     0.006     0.001     0.000     0.000
qr=0.520      0.201     0.015     0.003     0.000     0.000     0.000     0.000
qr=0.525      0.040     0.001     0.000     0.000     0.000     0.000     0.000
qr=0.530      0.002     0.000     0.000     0.000     0.000     0.000     0.000
qr=0.540      0.000     0.000     0.000     0.000     0.000     0.000     0.000
qr=0.550      0.000     0.000     0.000     0.000     0.000     0.000     0.000
qr=0.600      0.000     0.000     0.000     0.000     0.000     0.000     0.000
```

**Key findings:**

1. **Same phase boundary as E2.** The critical transition sits at qr ≈ 0.505–0.525,
   confirming the boundary is robust to whether the environment shifts discretely or
   drifts slowly.

2. **Slow drift does not help escape lock-in.** Gradual evidence accumulation does
   not erode the social barrier. If the community is above the critical coupling
   threshold, they remain locked in even as the environment drifts away from their
   paradigm over 200 steps.

3. **Richer gradient at low qr.** Compared to E2, the slow drift shows more gradation
   in the transition zone (e.g., qr=0.505 ranges from 0.93 to 0.23 across eps_resolve).
   This is because agents commit more gradually during the ramp, so there is more
   variation in how deeply entrenched they become.

4. **Hysteresis implied.** The population commits to paradigm 0 during t=0–50 (before
   the ramp starts), then the environment drifts toward paradigm 1. The asymmetry
   between initial commitment (fast, under consistent evidence) and later adaptation
   (slow, against social pressure) IS the hysteresis. The phase diagram quantifies
   which parameter combinations produce it.

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

## E2 Definitive: all three fixes (eps_theta + mavericks + trust learning)

**Config:** eps_theta=0.05, 20% mavericks (social_mask=0.2), trust_learning=true
(rho=0.90), eps_crisis=0.20, D_c_normal=0.9, N=40, T=300 steps, shift at t=100.
Sweep: q_reliability × eps_resolve, 10 seeds each.

**Result — phase diagram** (mean final belief in TRUE paradigm, averaged over seeds):

```
          er=0.01   er=0.05   er=0.1    er=0.2    er=0.3    er=0.5    er=0.7
qr=0.55    0.808     0.808     0.805     0.794     0.781     0.757     0.734
qr=0.60    0.868     0.871     0.871     0.867     0.862     0.852     0.843
qr=0.65    0.904     0.907     0.907     0.906     0.904     0.641     0.461
qr=0.70    0.927     0.929     0.930     0.878     0.650     0.324     0.202
qr=0.75    0.944     0.945     0.883     0.637     0.343     0.174     0.094
qr=0.80    0.915     0.813     0.556     0.285     0.186     0.071     0.063
qr=0.85    0.733     0.466     0.286     0.155     0.055     0.047     0.043
qr=0.90    0.304     0.170     0.083     0.037     0.034     0.031     0.030
```

**Key findings:**

1. **Diagonal boundary at realistic social coupling.** The transition from
   adaptation (~0.9) to lock-in (~0.1) runs diagonally from (qr=0.65, er=0.5)
   to (qr=0.90, er=0.01). Both axes contribute: stronger social coupling AND
   more paradigm inertia push toward lock-in.

2. **Graded transition, not binary.** Unlike the baseline phase diagram (sharp
   0→1 at qr≈0.51), the three-fix model shows a smooth gradient across the
   boundary. This is more realistic: real paradigm shifts involve gradual
   erosion, not a binary switch.

3. **The boundary is in the realistic range (qr ≈ 0.70–0.85).** Scientific
   communities typically have moderate-to-high trust in peers. The model now
   predicts paradigm shifts CAN happen at realistic coupling — they require
   mavericks + trust erosion + accumulated anomalous evidence.

4. **eps_resolve matters.** At qr=0.70, the model transitions from 0.93
   (er=0.01, crises persist) to 0.20 (er=0.70, high inertia). Theory-ladenness
   (paradigm inertia) is now a meaningful parameter, not just noise — it
   determines WHERE on the boundary a community sits.

5. **Residual lock-in floor at ~0.03.** Even at (qr=0.90, er=0.70), mean_qB
   doesn't reach 0.0 — some mavericks partially shift. This is the "peripheral
   scientists" who see the anomalies but can't drag the community.

---

## For the Wednesday call

The definitive E2 phase diagram is the headline figure. It shows:
- Paradigm lock-in is a **collective** phenomenon with a diagonal boundary
  in (social coupling, paradigm inertia) space.
- The boundary is at **realistic** social coupling (qr ≈ 0.70–0.85), not the
  unrealistic qr ≈ 0.51 of the baseline model.
- The transition is **graded**, matching the empirical signature of paradigm
  change: gradual erosion, peripheral scientists shifting first, eventual
  tipping cascade.
- Three principled AIF mechanisms combine: paradigm leak (B_theta),
  heterogeneous social sensitivity (mavericks), and evidence-coupled trust
  (Gamma-conjugate learning).
- This maps onto the abstract: "beliefs on which much else depends become
  correspondingly harder to revise" — the "much else" is now formally
  decomposed into social coupling strength (q_reliability) and epistemic
  inertia (eps_resolve).
