# Model refinements — 2026-05-27

The E2/E3 phase diagrams reveal a clean collective phase transition between
paradigm adaptation and lock-in, but the critical social coupling threshold
(q_reliability ≈ 0.51–0.525) is unrealistically low. In real scientific
communities, trust in peers is much higher (0.7–0.9), yet paradigm shifts
DO happen. Three refinements are needed to push the boundary to a realistic
range. This note documents the diagnosis, the literature grounding, and the
implementation plan for each.

---

## Diagnosis: why the threshold is too low

The social channel overwhelms the world channel because three features
compound:

1. **c=0 zeroes the disruption.** Under normal science, paradigm likelihood
   columns are averaged to be identical — observations carry *zero* paradigm
   information. Even a small social signal dominates zero.

2. **Trust is static.** The trust matrix T is fixed from the network
   adjacency. Agents trust wrong neighbors forever, regardless of prediction
   accuracy. The social feedback loop never weakens.

3. **No heterogeneity.** All agents share the same eps_crisis, eps_resolve,
   and social sensitivity. There are no mavericks, no weak links in the
   consensus, no transient diversity.

Each fix addresses one of these. Combined, they should shift the critical
threshold to q_reliability ≈ 0.7–0.8 and produce a gradual tipping cascade
rather than a binary switch.

---

## Fix 1: Partial theory-ladenness (alpha blend)

**Problem addressed:** c=0 is completely uninformative.

**Mechanism:** Replace the binary c=0 averaging with a parameterized blend:

    A_c0[o, k, a] = (1 - alpha) * A_world[o, k, a] + alpha * avg[o, a]

- alpha = 1.0 (current): all paradigm columns identical under c=0
- alpha = 0.5: ~25% discriminability retained ((1-alpha)^2 * d_full)
- alpha = 0.0: c=0 is no different from c=1 (no theory-ladenness)

**Effect on phase boundary:** The environment-coupled fold (social_fold.py)
has B_field = d + h_U. Currently d=0 under c=0. With partial blending,
d_c0 = (1-alpha)^2 * d_full > 0, which competes against the social gain K.
The fold unrolls into a cusp at lower K values, pushing the critical
q_reliability upward.

**Implementation:** Add `alpha_tl: float = 1.0` to SimpleConfig. Modify
`build_joint_A_world()` to accept alpha and blend per-paradigm columns with
the average. ~10 lines changed.

**Literature:**
- Hanson (2025), "Computationally Reframing the Theory-Ladenness of
  Observation" — theory-ladenness is graded, not binary; observations are
  partially filtered by paradigm commitments.
  https://philsci-archive.pitt.edu/27598/1/Hanson_MakoHajime.pdf

---

## Fix 2: Agent heterogeneity

**Problem addressed:** No transient diversity.

**Mechanism:** Draw eps_crisis and eps_resolve per agent from a distribution
(e.g., Beta). Some agents are "mavericks" (high eps_crisis, low eps_resolve)
who enter crisis readily and stay there; others are "establishment" (low
eps_crisis, high eps_resolve) with strong paradigm inertia.

**Effect on phase boundary:** Heterogeneity smears the sharp transition into
a gradual tipping cascade. Mavericks shift first (they detect anomalies and
persist in crisis), then drag connected neighbors via the social channel.
The network topology matters: planted-SBM with maverick/establishment blocks
creates modular escape routes.

**Implementation:** Make `build_B_c()` accept (N,) arrays instead of
scalars, returning (N, 2, 2). Modify `transition_joint()` to vmap over
per-agent B_c matrices. Add distribution parameters to SimpleConfig.
~50 lines.

**Literature:**
- Zollman (2010), "The Epistemic Benefit of Transient Diversity" — less
  connectivity and diversity of practice are epistemically beneficial;
  minimally connected networks outperform fully connected ones.
  https://philpapers.org/rec/ZOLTEB-2
- Bizyaeva, Franci & Leonard (2023), "Nonlinear Opinion Dynamics with
  Tunable Sensitivity" — agent-level sensitivity parameter controls the
  bifurcation structure; heterogeneous sensitivity produces cascades.
  https://naomi.princeton.edu/wp-content/uploads/sites/744/2023/03/Nonlinear_Opinion_Dynamics_With_Tunable_Sensitivity.pdf

---

## Fix 3: Dynamic trust learning (highest priority)

**Problem addressed:** Static social coupling — wrong consensus never
weakens.

**Mechanism:** Trust erodes when neighbors' paradigm predictions fail.
After each step, compute a surprisal matrix: how well did agent j's emitted
q_j(theta) predict agent i's observed outcome? Agents whose paradigm beliefs
are inconsistent with the data lose trust from their neighbors.

Update rule (Gamma-conjugate, from src/trust.py):

    alpha_ij <- rho * alpha_ij + 1
    beta_ij  <- rho * beta_ij + eps_ij
    gamma_ij <- alpha_ij / beta_ij

where eps_ij = -log sum_theta q_j(theta) * A_world[o_i, theta, x] is the
cross-entropy of j's belief evaluated against i's observation.

T_ij is then derived from gamma_ij (row-normalized).

**Effect on phase boundary:** When the environment shifts (E2/E3), agents
locked into the wrong paradigm emit beliefs that poorly predict the new
observations. Their neighbors observe this, accumulate surprisal, and erode
trust. As trust in wrong-paradigm agents decays, the social reinforcement
loop weakens, eventually allowing evidence to break through. The critical
q_reliability shifts upward because lock-in now requires sustained
prediction accuracy, not just mutual agreement.

**Implementation:** Add (N, N) alpha and beta arrays to SimpleState. After
inference, compute categorical surprisal and update trust via
gamma_conjugate_step. Pass updated T into the next step. ~100 lines. The
Gamma-conjugate machinery already exists in `src/trust.py` for the
continuous substrate.

**Literature:**
- Albarracin, Demekas, Ramstead, Heins (2022), "Epistemic Communities
  under Active Inference" — confirmation bias and echo chambers emerge from
  agents sampling to justify their view; evidence-coupled trust is the
  natural remedy.
  https://www.mdpi.com/1099-4300/24/4/476
- Heins, Da Costa, Friston et al. (2024), "Belief sharing: a blessing or
  a curse" — naively sharing posteriors creates echo chambers; sharing
  likelihoods is safer. Trust learning achieves a similar effect by
  downweighting unreliable posterior contributions.
  https://arxiv.org/abs/2407.02465
- Hyland & Albarracin (2025), "On the Variational Costs of Changing Our
  Minds" — single-agent framework; multi-agent extension flagged as future
  work. Trust learning adds the social dimension.
  https://arxiv.org/abs/2509.17957

---

## Connection to the paper's theoretical framework

The Hyland & Albarracin (2025) key equation:

    q*(s) ∝ p(s) · exp[ λ^{-1} (c_s + alpha · log p(o|s)) ]

shows that affective utilities (c_s) multiplicatively interact with
likelihoods, creating asymmetric belief revision. Our hierarchical context
model maps onto this:

- **c=0 (normal science)** → effective λ → ∞ (beliefs frozen to evidence)
- **c=1 (crisis)** → effective λ → 0 (responsive to evidence)
- **B_c transition** → the dynamics of moving between these regimes

The three fixes enrich this mapping:
- Partial alpha → λ is graded, not binary (c=0 still carries some signal)
- Heterogeneity → per-agent λ drawn from a distribution
- Trust learning → the social component of c_s is evidence-coupled

---

## Implementation order

1. **Alpha blend** (immediate, ~10 min) — probe whether it shifts the
   boundary, minimal code change, validates the diagnosis.
2. **Trust learning** (next, ~2 hours) — highest leverage fix, directly
   addresses the social coupling dominance, principled AIF mechanism.
3. **Heterogeneity** (after trust learning) — orthogonal to trust, adds
   the tipping-cascade dynamics.

Each fix can be tested independently against the E2 phase diagram to
measure the boundary shift.

---

## Empirical results: cumulative boundary shift

Tested on the E2 discrete-shift scenario (truth flips at t=100, N=40,
eps_crisis=0.20, eps_resolve=0.30, 300 steps). 20% maverick agents
(social_mask=0.2) for heterogeneity condition. Trust rho=0.90.

```
                                     qr=0.55  qr=0.60  qr=0.70  qr=0.80  qr=0.90
baseline (absorbing B_theta)    0.000    0.000    0.000    0.000    0.000
+ eps_theta=0.02                0.917    0.060    0.018    0.008    0.003
+ eps_theta=0.05                0.783    0.883    0.054    0.022    0.008
+ heterogeneous social mask     0.740    0.822    0.549    0.103    0.046
+ trust learning                0.741    0.823    0.901    0.122    0.046
```

**Key findings:**

1. **eps_theta is the prerequisite.** Without paradigm leak, the absorbing
   prior prevents ANY adaptation regardless of other fixes. With eps_theta=0.05,
   the boundary moves from qr≈0.51 to qr≈0.55-0.60.

2. **Heterogeneity extends the range.** Mavericks (low social sensitivity)
   shift first, pulling the boundary to qr≈0.70. This is the Kuhnian
   "peripheral scientists" story — paradigm change starts at the margins.

3. **Trust learning amplifies mavericks.** At qr=0.70, heterogeneity alone
   gives 0.55; adding trust learning pushes it to 0.90. Trust in locked-in
   agents erodes as mavericks' predictions improve, creating a tipping cascade.

4. **The combined boundary (qr≈0.70-0.80) is realistic.** Real scientific
   communities have trust levels in this range. The model now predicts that
   paradigm shifts CAN happen under realistic social coupling — they just
   require the combination of paradigm leak, peripheral mavericks, and
   evidence-coupled trust.

---

## Updated fix: eps_theta (paradigm leak via B_theta)

**Not originally planned but discovered as the prerequisite.** The absorbing
B_theta = identity creates an inescapable prior trap after extended commitment.
Adding a small leak (eps_theta=0.02-0.05) toward uniform is the principled fix:
it corresponds to the agent's uncertainty that its current paradigm will remain
correct — a nonzero transition probability in the state model.

This is exactly the "tiny B_theta leak" flagged as an open question in
notes/pomdp_generative_model.md §2 ("Phase 0.5 informs it"). The experiments
confirm: the leak is load-bearing for paradigm shift dynamics.

---

## For the Wednesday call

Present the E2 phase diagram as the baseline, then show how each fix
shifts the boundary:

1. "Here's the phase transition — it's too sharp and the threshold is too
   low."
2. "Partial alpha retains some signal under normal science — does it help?"
3. "Trust learning makes the social channel self-correcting — this is the
   big one."
4. "What parameter regime should we target for the paper figures?"
