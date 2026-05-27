# Experiment results — 2026-05-28

Continuous-lambda model (`src/pomdp/cont_step.py`): tempered belief updates
with per-agent lambda, Hyland & Albarracin (2025) Eq. 13 at multi-agent scale.
All runs: K=2 paradigms, Watts-Strogatz graph (N=40, mean degree 4), 20%
maverick agents (social_mask=0.2), eps_theta=0.02, eta_lambda=0.1, kl_target=0.05.

---

## E4: Continuous-lambda phase diagram (no resources)

**Purpose:** Map the (q_reliability, lambda_init) phase boundary between
paradigm adaptation and lock-in using the continuous-lambda model. Discrete
shift at t=100 (truth flips from paradigm 0 to 1).

**Config:** N=40, T=300 steps, eps_theta=0.02, eta_lambda=0.1, 20% mavericks,
no resources. Sweep: q_reliability × lambda_init, 10 seeds each.

**Result — phase diagram** (mean final belief in TRUE paradigm):

```
          lam=1.0   lam=1.5   lam=2.0   lam=2.5   lam=3.0   lam=4.0
qr=0.55    0.921     0.923     0.943     0.953     0.954     0.849
qr=0.60    0.942     0.943     0.960     0.878     0.516     0.240
qr=0.65    0.955     0.956     0.919     0.457     0.252     0.090
qr=0.70    0.964     0.965     0.553     0.260     0.116     0.032
qr=0.75    0.971     0.912     0.381     0.154     0.052     0.034
qr=0.80    0.766     0.550     0.277     0.069     0.036     0.030
qr=0.85    0.547     0.459     0.174     0.036     0.026     0.026
qr=0.90    0.417     0.311     0.082     0.024     0.024     0.023
```

**Key findings:**

1. **Diagonal phase boundary.** The transition from adaptation (~0.95) to
   lock-in (~0.03) runs diagonally from (qr=0.60, lam=4) to (qr=0.90, lam=1).
   Both social coupling strength AND the variational cost of mind-change
   contribute to lock-in.

2. **Graded, not binary.** The transition zone spans a wide range (e.g., at
   qr=0.70: 0.96 at lam=1.5, 0.55 at lam=2.0, 0.12 at lam=3.0). This is the
   smooth, realistic paradigm-shift dynamics the binary context model couldn't
   produce.

3. **Realistic parameter range.** The boundary sits at qr ≈ 0.65–0.85 and
   lam ≈ 1.5–3.0 — both are plausible values for scientific communities.

4. **Lambda is meaningful.** Unlike the binary-context E2 diagram where only
   social coupling drove the transition, here lambda is a first-class axis.
   This validates the Hyland & Albarracin (2025) mechanism at multi-agent scale:
   the variational cost of mind-change determines whether paradigm shifts happen.

---

## E6: Resource ablation (the Fig 4 story)

**Purpose:** At fixed social coupling (qr=0.75), compare the continuous-lambda
model WITH and WITHOUT resource coupling. Show that resources create
qualitatively different (irreversible) lock-in.

**Config:** qr=0.75, eps_theta=0.02, eta_lambda=0.1, 20% mavericks, N=40,
T=300 steps, shift at t=100. Resource params: r_init=1.0, R_in=0.1,
alpha_flow=0.3, delta_decay=0.05, c0=0.1. Sweep: lambda_init × resource ON/OFF,
10 seeds each.

**Result — ablation table:**

```
              no_resource  resource    gap
lam=1.0        0.971       0.084    +0.887
lam=1.25       0.962       0.073    +0.889
lam=1.5        0.912       0.066    +0.847
lam=1.75       0.605       0.066    +0.539
lam=2.0        0.381       0.066    +0.316
lam=2.5        0.154       0.037    +0.117
lam=3.0        0.052       0.032    +0.020
lam=4.0        0.034       0.029    +0.004
```

**Key findings:**

1. **Without resources: lambda is the control knob.** A smooth gradient from
   full adaptation (0.97 at lam=1) to lock-in (0.03 at lam=4). The variational
   cost of mind-change is the sole mechanism — this is Hyland & Albarracin
   (2025) at multi-agent scale.

2. **With resources: flat lock-in regardless of lambda.** Mean_qB ≈ 0.07
   across all lambda values. Resources create a FLOOR that lambda can't
   penetrate. Even standard Bayes (lam=1) is locked in.

3. **The mechanism is exploration starvation.** After committing to the wrong
   paradigm, agents' resources flow toward paradigm-aligned neighbors (high
   trust). Resource-starved agents can't afford informative experiments, which
   prevents the observations that would drive belief revision. Trust learning
   can't help either — it needs informative observations to generate
   prediction-error signal.

4. **Resources operate below trust and lambda in the causal chain.** The prior
   leak (eps_theta) opens the door for evidence; lambda controls how much
   evidence moves beliefs; but resources gate whether the evidence can be
   GENERATED in the first place. This is a lower-level lock that neither
   informational mechanism can overcome.

5. **The gap column IS the paper's Fig 4.** At lambda=1: gap = +0.89 (resources
   matter enormously when agents are willing to update). At lambda=4: gap = +0.004
   (resources are redundant when agents are already frozen by high mind-change cost).

---

## Diagnosis: the absorbing-prior problem and the eps_theta fix

The continuous-lambda model initially produced all-zero results (total lock-in
at all parameters). Diagnosis revealed the same absorbing-prior problem as
the binary-context model:

After 100 steps of consistent paradigm-0 evidence, agents reach q ≈ [1e-12, 1].
This is -27 log-odds favoring paradigm 0. Even at lambda=1 (standard Bayes),
the world provides only ~0.5 log-odds per step toward truth — requiring ~54
steps to recover WITHOUT social coupling, and impossible with it.

**The fix:** eps_theta = 0.02 — a 2% leak of the prior toward uniform each step.
This caps the maximum log-odds commitment at ~log(1/0.02) ≈ 3.9, making the
prior recoverable within ~8 steps of evidence. Combined with 20% mavericks
(who are less socially constrained), this produces the graded dynamics.

**Theoretical interpretation:** eps_theta corresponds to the agent's irreducible
uncertainty about whether its current paradigm will remain correct — a
nonzero transition probability in the generative model. It was flagged as
an open question in notes/pomdp_generative_model.md §2 ("keep B=strict-identity
vs. a tiny B_theta leak"). The experiments confirm it is load-bearing.

---

## Connection to the paper's ablation structure

From the 2026-05-28 call, David proposed structuring the paper as incremental
complexity. The experimental results now support exactly this:

| Figure | Model | Result | What it shows |
|--------|-------|--------|---------------|
| Fig 1 | Schematic | — | Model architecture |
| Fig 2 | Base model (David/Jonas) | TBD | Message passing + motivated updates alone |
| Fig 3 | Cont-lambda (E4) | Phase diagram | Social coupling × lambda boundary |
| Fig 4 | + Resources (E6) | Ablation table | Resources create irreversible lock-in |

The story builds incrementally:
1. **Fig 2:** Motivated belief updates + social coupling produce paradigm
   dynamics (David/Jonas's minimal model).
2. **Fig 3:** The (q_reliability, lambda) phase diagram shows WHERE paradigm
   shifts happen — a diagonal boundary with both axes contributing.
3. **Fig 4:** Resources add irreversibility — without them, lambda controls
   everything; with them, exploration starvation creates a lock-in floor that
   no informational mechanism can penetrate.

---

## Parameter glossary

| Parameter | Symbol | Role |
|-----------|--------|------|
| q_reliability | κ = log(q/(1-q)) | Social coupling strength |
| lambda_init | λ₀ | Initial cost of mind-change (Hyland & Albarracin) |
| eta_lambda | η_λ | Lambda adaptation rate from KL cost |
| kl_target | KL* | Comfortable rate of mind-change |
| eps_theta | ε_θ | Prior leak toward uniform (prevents absorbing state) |
| maverick_fraction | — | Fraction of agents with low social sensitivity |
| resource_coupling | — | Whether exploration is gated by trust-derived resources |
| R_in | — | Exogenous resource inflow per step |
| c0 | — | Base cost of running an experiment |
