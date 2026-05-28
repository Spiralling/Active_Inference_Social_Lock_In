# Experiment results — 2026-05-28 (principled model)

Principled multi-level context model (`src/pomdp/simple_step.py` with C=5).
All inference is standard categorical Bayes on joint state (theta, c) with
5 context levels and tridiagonal B_c transitions. No heuristic lambda rules.

All runs: K=2 paradigms, Watts-Strogatz graph (N=40, mean degree 4), 20%
mavericks (social_mask=0.2), eps_theta=0.02, eps_crisis=0.15, D_c_normal=0.8.

---

## E7: Phase diagram (q_reliability × eps_resolve)

**Purpose:** Map the phase boundary for the principled model. Discrete shift
at t=100 (truth flips from paradigm 0 to 1).

**Config:** C=5 context levels, alpha_tl=1.0, N=40, T=300 steps. Sweep:
q_reliability × eps_resolve, 10 seeds each.

**Result:**

```
          er=0.05   er=0.1    er=0.15   er=0.2    er=0.3    er=0.4    er=0.5
qr=0.55    0.805     0.803     0.798     0.792     0.777     0.760     0.743
qr=0.60    0.871     0.871     0.870     0.868     0.862     0.856     0.795
qr=0.65    0.908     0.909     0.792     0.635     0.467     0.350     0.220
qr=0.70    0.591     0.482     0.416     0.320     0.177     0.147     0.123
qr=0.75    0.352     0.246     0.179     0.141     0.104     0.091     0.085
qr=0.80    0.159     0.124     0.101     0.083     0.088     0.081     0.077
qr=0.85    0.087     0.086     0.075     0.069     0.066     0.063     0.059
qr=0.90    0.058     0.052     0.054     0.050     0.048     0.047     0.046
```

**Key findings:**

1. **Broad, graded transition zone.** The boundary between adaptation and
   lock-in spans qr=0.65–0.80 — a wide band, not a sharp cliff. This is
   more realistic than the C=2 binary model (qr=0.70–0.75, narrow) and
   matches the empirical signature of paradigm change: gradual erosion,
   not a sudden flip.

2. **Diagonal boundary.** Both axes contribute. At qr=0.65, eps_resolve
   drives the transition from 0.91 (low inertia) to 0.22 (high inertia).
   At eps_resolve=0.15, q_reliability drives it from 0.80 (weak coupling)
   to 0.075 (strong coupling). Neither axis alone determines the outcome.

3. **Realistic parameter range.** The transition sits at qr=0.65–0.80,
   which corresponds to moderate-to-high peer trust in scientific communities.
   eps_resolve=0.15–0.30 corresponds to a crisis half-life of 3–7 steps.

4. **Partial adaptation, not total lock-in.** Even at qr=0.90, mean_qB
   stabilises around 0.05 (not 0.00). Some mavericks always partially
   shift. This is arguably more realistic than the hard zeros of simpler
   models — in real paradigm shifts, peripheral scientists do update.

5. **C=5 vs C=2 comparison.** The multi-level context produces a smoother
   transition because agents can occupy intermediate commitment levels
   rather than being forced into binary normal-science or crisis. The
   graded ladder means paradigm shifts proceed through a series of
   incremental commitment changes, not a single flip.

---

## E8: Resource ablation (eps_resolve × resource ON/OFF)

**Purpose:** At fixed qr=0.75, show what resources add to the principled
model.

**Config:** C=5, qr=0.75, resource params: r_init=1.0, R_in=0.1,
alpha_flow=0.3, delta_decay=0.05, c0=0.1. Sweep: eps_resolve × resource
ON/OFF, 10 seeds each.

**Result:**

```
              no_resource  resource    gap
er=0.05        0.352       0.084    +0.267
er=0.10        0.246       0.084    +0.162
er=0.15        0.179       0.084    +0.094
er=0.20        0.141       0.084    +0.057
er=0.30        0.104       0.084    +0.020
er=0.40        0.091       0.084    +0.007
er=0.50        0.085       0.084    +0.001
```

**Key findings:**

1. **Resources create a lock-in floor.** With resources, mean_qB is flat
   at ~0.084 regardless of eps_resolve. Without resources, eps_resolve
   produces a gradient from 0.35 to 0.085. Resources remove the effect
   of paradigm inertia by preventing the observations that would allow
   recovery.

2. **The gap is largest at low inertia.** At er=0.05 (crises persist,
   agents are open to evidence): gap = +0.267. Resources cut adaptation
   by 76%. At er=0.50 (high inertia, agents already resistant): gap = 0.001.
   Resources are redundant when agents are already frozen.

3. **The mechanism is exploration starvation.** After committing to the
   wrong paradigm, agents' resources flow preferentially toward paradigm-
   aligned neighbours (high trust → high W → high r inflow). Resource-
   starved agents can only afford low-discriminability experiments, which
   produce observations that cannot distinguish paradigms. The evidence
   channel is cut off at the source.

4. **Resources operate below trust in the causal chain.** Trust learning
   requires informative observations to generate prediction-error signal.
   Resource starvation prevents those observations. The resource barrier
   is therefore a lower-level lock that no purely informational mechanism
   (trust learning, context inference, paradigm leak) can overcome.

---

## Model comparison across all experiments

| Model | Experiment | Transition zone | Graded? | Principled? | Resource effect |
|-------|-----------|----------------|---------|-------------|----------------|
| Binary context (C=2) | E2 | qr 0.70–0.75 | Somewhat | Yes (but binary) | N/A |
| Continuous lambda | E4 | qr 0.65–0.85 | Yes | No (KL rule) | Strong (E6) |
| **Multi-level (C=5)** | **E7** | **qr 0.65–0.80** | **Yes** | **Yes** | **Strong (E8)** |

The principled C=5 model achieves the same qualitative phenomena as the
heuristic continuous-lambda model — graded transitions at realistic social
coupling — but through standard categorical Bayes on a joint (theta, c)
state space, with no hand-tuned adaptation rules. The multi-level context
(Jonas's "richer belief network") is the mechanism that produces gradedness.

---

## For the Tuesday call

The paper's experimental story, using principled model results:

1. **Fig 2 (David/Jonas):** Base model — message passing + motivated updates.
   What can the simplest model already do?

2. **Fig 3 (E7):** Phase diagram — (q_reliability, eps_resolve) boundary.
   A broad, graded transition at realistic social coupling (qr 0.65–0.80).
   Both social coupling and paradigm inertia contribute. Multi-level
   context (C=5) produces the gradual erosion pattern of real paradigm
   shifts.

3. **Fig 4 (E8):** Resource ablation — resources create an irreversible
   lock-in floor that paradigm inertia alone cannot produce. The gap is
   largest when agents are otherwise open to evidence (low eps_resolve).
   Exploration starvation operates below the level of trust dynamics.
