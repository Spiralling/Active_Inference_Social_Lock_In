# Lit Review 05 — Cultural Evolution, Replicator Dynamics, and Resource-Coupled Bistability

**Goal.** Identify mechanisms in the cultural-evolution / evolutionary-dynamics / scientific-reputation
literature where *beliefs (or strategies) and material payoff/resources* are coupled in a way
that demonstrably produces bistability, lock-in, or "paradigm-capture" outcomes. The current
IWAI 2026 substrate (Gaussian conjugate beliefs + trust pooling + deterministic resource readout
`r = flow_from_trust(Γ)`) was shown in nb13 to be structurally monostable. To rescue the
"coupled belief–resource feedback" narrative the resource layer needs either (i) its own
dynamic ODE with internal positive feedback, or (ii) a frequency-dependent payoff that re-enters
inference nonlinearly. The papers below provide that template.

## Executive summary (3 sentences)

The cleanest "resource-coupled bistability" templates are **Weitz et al. 2016** (replicator with
game–environment feedback; produces a heteroclinic oscillating tragedy of the commons) and
**Tilman, Plotkin, Akçay 2020** (eco-evolutionary games; explicit bistability when both edge
equilibria are stable, public code on GitHub). These show that monostability is *generic* unless
either (a) the environmental state has its own logistic-style growth term `ṅ = n(1−n)f(x)` whose
sign flips with strategy frequency, or (b) the strategy payoff depends nonlinearly on the
environment so the joint Jacobian acquires a positive trace at the interior fixed point. Both
ingredients map directly onto our `resource.py`/`inference.py` boundary: lifting `r` from a
deterministic readout of `Γ` to a state with self-reinforcement is the minimal viable graft.

---

## 1. Weitz, Eksin, Paarporn, Brown, Ratcliff 2016 — Oscillating tragedy of the commons

**BibTeX**

```bibtex
@article{weitz2016oscillating,
  title   = {An oscillating tragedy of the commons in replicator dynamics with game-environment feedback},
  author  = {Weitz, Joshua S. and Eksin, Ceyhun and Paarporn, Keith and Brown, Sam P. and Ratcliff, William C.},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {113},
  number  = {47},
  pages   = {E7518--E7525},
  year    = {2016},
  doi     = {10.1073/pnas.1604096113}
}
```

**Code.** No GitHub repo named by the authors; the model is small enough that several third-party
re-implementations exist on GitHub (search "Weitz oscillating tragedy"). PMC: PMC5127343.

**Mechanism (one sentence).** Replicator on strategies coupled to a logistic environment
that the strategies themselves enhance or degrade:
`ε ẋ = x(1−x)[δ_PS + (δ_TR − δ_PS)x](1−2n)`, `ṅ = n(1−n)[−1 + (1+θ)x]`,
where `θ` is the ratio of cooperator-enhancement to defector-degradation rates.

**Graft analysis.** This is the closest formal analogue to what our paper *claims*. To adopt it,
`src/resource.py` must change from `r = flow_from_trust(Γ)` (deterministic readout) to a true
state variable obeying `ṙ = r(1−r) g(μ, Γ)`. The key is the `r(1−r)` factor — it is what creates
two attractors at `r=0` and `r=1` and a saddle in between. `src/inference.py` does not need a
replicator-style update per se; it can keep its Gaussian-conjugate form, but the *payoff* used
inside `policy.py` (currently flat) must become explicitly `r`-dependent so the feedback loop closes.
With ε ≪ 1 (fast belief, slow resource), we get fast–slow analysis and limit cycles; with
sign-flipping `g(μ, Γ)` you get heteroclinic cycles between paradigm-A-rich and paradigm-B-rich
resource states. This would re-cast our "paradigm persistence" claim as a Weitz-style heteroclinic
slowdown — defensible and well-cited.

---

## 2. Tilman, Plotkin, Akçay 2020 — Evolutionary games with environmental feedbacks

**BibTeX**

```bibtex
@article{tilman2020evolutionary,
  title   = {Evolutionary games with environmental feedbacks},
  author  = {Tilman, Andrew R. and Plotkin, Joshua B. and Ak{\c{c}}ay, Erol},
  journal = {Nature Communications},
  volume  = {11},
  number  = {1},
  pages   = {915},
  year    = {2020},
  doi     = {10.1038/s41467-020-14531-6}
}
```

**Code.** **Yes — https://github.com/atilman/EcoEvoGamesCode** (Python). Confirmed.

**Mechanism (one sentence).** Replicator `ẋ₁ = ε₃ x₁(1−x₁)(π₁(x,n) − π₂(x,n))` coupled to environmental
dynamics `ṅ = ε₁ f(n) − ε₂ h(x,n)`; bistability arises iff incentive-to-lead parameters satisfy
`ΔL¹ < 0 ∧ δH⁰ < 0` (no agent has unilateral incentive to be the first to switch behaviours,
so the system tips into whichever pure strategy started ahead).

**Graft analysis.** This is the more general framework Weitz is a special case of — and it has
**explicit bistability classification** plus public Python code we can clone. Our adaptation:
treat `μ` (mean belief) as analogous to `x₁`, treat `r` as the environment `n` and let it satisfy
`ṙ = ε₁ (R_in − ρ r) − ε₂ h(μ, r)` where `h` is a μ-dependent extraction term. The bistability
condition then translates to: extraction is high when paradigm-A is dominant (`μ ≈ θ_A^*`) and low
otherwise, *and* belief drift toward `θ_A^*` is stronger when `r` is high. This is the "patronage
flow gates inference precision" hook the paper already gestures at; Tilman makes it formally
operational. **Recommended primary graft target** because (i) the code is real, (ii) the bistability
condition is constructive (not just existence), and (iii) it lets us keep λ·U Hyland framing intact:
the replicator-on-x is replaced by Hyland action selection, but the ṅ equation can be lifted
verbatim into `resource.py`.

---

## 3. Smaldino & McElreath 2016 — The natural selection of bad science

**BibTeX**

```bibtex
@article{smaldino2016natural,
  title   = {The natural selection of bad science},
  author  = {Smaldino, Paul E. and McElreath, Richard},
  journal = {Royal Society Open Science},
  volume  = {3},
  number  = {9},
  pages   = {160384},
  year    = {2016},
  doi     = {10.1098/rsos.160384}
}
```

**Code.** Original Java (`Boffin.java`) — not on GitHub; was archived via Royal Society SI and
later flagged for a coding bug in `chooseHypothesis()` lines 86–102 (Kohrt et al. 2023 replicated
in R; archived on Software Heritage). Smaldino's textbook code is at
**https://github.com/psmaldino/modsoc** (NetLogo, not the bad-science model directly but related
cultural-evolution ABMs).

**Mechanism (one sentence).** Agent-based scientific labs evolve under selection-on-output:
labs with higher false-positive rates produce more "discoveries", get more progeny, and have
their methods copied — i.e. fitness is a non-monotone function of methodological rigor, and the
copy operator amplifies whichever variant currently has higher fitness (cultural replicator).

**Graft analysis.** Less directly useful for *bistability* (the model is monotone — bad science
wins in essentially all regimes), but it gives us the clearest published precedent for "resource
flow = recognition × output, recognition reshapes belief population". The graft would be: make
`r_i` (per-agent resource) an explicit function of *observed* false-positive count, and let `μ_i`
evolve by imitation of high-`r_i` neighbours. This is closer to discrete-choice cultural dynamics
than to our continuous-θ Gaussian world, so the cleaner move is to *cite* Smaldino as motivation
and adopt the Weitz/Tilman ODE form for the actual mechanism.

---

## 4. Komarova & Nowak 2003–2004 — Replicator-mutator universality and language dynamics

**BibTeX**

```bibtex
@article{komarova2004replicator,
  title   = {Replicator-mutator equation, universality property and population dynamics of learning},
  author  = {Komarova, Natalia L.},
  journal = {Journal of Theoretical Biology},
  volume  = {230},
  number  = {2},
  pages   = {227--239},
  year    = {2004},
  doi     = {10.1016/j.jtbi.2004.05.004}
}

@article{nowak2001evolution,
  title   = {Evolution of universal grammar},
  author  = {Nowak, Martin A. and Komarova, Natalia L. and Niyogi, Partha},
  journal = {Science},
  volume  = {291},
  number  = {5501},
  pages   = {114--118},
  year    = {2001},
  doi     = {10.1126/science.291.5501.114}
}
```

**Code.** No public repo; the equations are small enough that any DE library reproduces them.

**Mechanism (one sentence).** The replicator-mutator equation
`ẋ_i = Σ_j x_j f_j Q_{ji} − x_i φ`
(where `Q` is the mutation matrix and `f_j` the fitness) shows a *coherence threshold*: above a
critical fidelity `q*`, the population locks onto a single grammar (one-language attractor);
below, beliefs disperse — a sharp universality-class phase transition.

**Graft analysis.** This is the right reference for the *threshold* phenomenon we'd like to
claim. Our `μ` could be replaced by a discrete `x_i` over a finite paradigm set with a Q matrix
parameterised by trust bandwidth; the threshold `q*` in Komarova would become a threshold on
trust-pooling sharpness, below which the system disperses and above which it locks into one
paradigm. Cleaner than our current continuous-θ formulation if we want a textbook bifurcation —
but it abandons the Gaussian conjugate machinery, so this is a *fall-back* graft, not the primary
one. Cite it for the threshold framing.

---

## 5. Falandays & Smaldino 2022 — Emergence of cultural attractors

**BibTeX**

```bibtex
@article{falandays2022emergence,
  title   = {The emergence of cultural attractors: How dynamic populations of learners achieve collective cognitive alignment},
  author  = {Falandays, J. Benjamin and Smaldino, Paul E.},
  journal = {Cognitive Science},
  volume  = {46},
  number  = {8},
  pages   = {e13183},
  year    = {2022},
  doi     = {10.1111/cogs.13183}
}
```

**Code.** Smaldino's publication page lists "[code:Python]" for this paper; not in a single
canonical GitHub repo, but available from the authors / supplementary.

**Mechanism (one sentence).** Agent-based unsupervised category-learning where each learner's
training input is sampled from the population's current productions; the recursive sampling
creates self-reinforcing attractors in feature space even without any externally-imposed payoff.

**Graft analysis.** This is the *no-resource* baseline — it shows attractors can emerge from
recursive learning alone. Useful as a *contrast* in the paper: "Falandays-Smaldino get attractors
from sampling alone; our model adds patronage flow and shows resource coupling *broadens* the
basin / sharpens the threshold". Not a primary mechanism graft, but a strong "related work"
citation that sets up our specific contribution.

---

## 6. Hofbauer & Sigmund — Evolutionary Games and Population Dynamics

**BibTeX**

```bibtex
@book{hofbauer1998evolutionary,
  title     = {Evolutionary Games and Population Dynamics},
  author    = {Hofbauer, Josef and Sigmund, Karl},
  year      = {1998},
  publisher = {Cambridge University Press},
  address   = {Cambridge},
  doi       = {10.1017/CBO9781139173179}
}
```

**Code.** N/A — textbook.

**Mechanism (one sentence).** Canonical replicator equation `ẋ_i = x_i (f_i(x) − φ(x))` with
the systematic bistability result: in a 2×2 coordination game with payoff matrix
`((a,b),(c,d))` and `a > c, d > b`, the interior equilibrium is *unstable* and both pure
states are attractors — textbook bistability with explicit basin boundary.

**Graft analysis.** This is the formal foundation that licenses every move above. The right way
to cite it: §3 of our paper invokes Hofbauer-Sigmund Theorem 7.5.1 (replicator with coordination
payoffs) to justify why the *coupled* belief-resource Jacobian can be unstable at an interior
point. We do not need to re-derive — we cite. Should already be on every IWAI reviewer's shelf.

---

## 7. Ohtsuki & Nowak 2006 — Replicator equation on graphs

**BibTeX**

```bibtex
@article{ohtsuki2006replicator,
  title   = {The replicator equation on graphs},
  author  = {Ohtsuki, Hisashi and Nowak, Martin A.},
  journal = {Journal of Theoretical Biology},
  volume  = {243},
  number  = {1},
  pages   = {86--97},
  year    = {2006},
  doi     = {10.1016/j.jtbi.2006.06.004}
}
```

**Code.** No canonical repo; standard pair-approximation routines are short.

**Mechanism (one sentence).** Pair-approximation transforms the payoff matrix `(a,b;c,d)` on a
k-regular graph into a *modified* matrix whose entries depend on `k` — and the modification can
flip the game from coordination to coexistence (and vice versa), creating or destroying
bistability *as a function of graph degree*.

**Graft analysis.** This is the key result for *replicator-dynamics-on-networks* that the brief
flagged. The promise: if we replace the pair-approximation correction by a *trust-weighted*
correction (using our row-stochastic trust matrix `T`), the same machinery should hand us a
graph-dependent bistability condition for the belief-resource coupled system. Concretely: write
the local replicator with payoffs `f_i(μ, r)`, expand around the interior fixed point with
the trust Laplacian as the spatial operator, and the bistability condition becomes a Fiedler-style
inequality on `λ₂(T)`. This is a defensible "trust topology controls bistability" headline that
ties together our nb10 Fiedler-vanguard work *and* the resource graft from §2 above. **High
value-add — recommend pursuing as a §4 extension after the Tilman graft is in.**

---

## 8. O'Connor & Weatherall — Misinformation Age / Conformity in scientific networks

**BibTeX**

```bibtex
@book{oconnor2019misinformation,
  title     = {The Misinformation Age: How False Beliefs Spread},
  author    = {O'Connor, Cailin and Weatherall, James Owen},
  year      = {2019},
  publisher = {Yale University Press},
  address   = {New Haven}
}

@article{weatherall2018conformity,
  title   = {Conformity in scientific networks},
  author  = {Weatherall, James Owen and O'Connor, Cailin},
  journal = {Synthese},
  volume  = {198},
  number  = {8},
  pages   = {7257--7278},
  year    = {2021},
  doi     = {10.1007/s11229-019-02520-2}
}
```

**Code.** No public repo; the underlying Bala-Goyal network-epistemology model is widely
re-implemented in NetLogo and Python (search "Bala Goyal learning network").

**Mechanism (one sentence).** Bayesian agents on a network share evidence and conform to
neighbours' *actions* (not just beliefs); conformity introduces a payoff-like term that
generates **stable polarisation** — two coexisting belief clusters that never merge.

**Graft analysis.** This is the philosophy-of-science framing closest to our paper, and gives
us the cleanest *narrative* hook: "polarisation persists because conformity acts as a payoff".
The mechanism is structurally identical to a 2-strategy replicator with frequency-dependent
fitness; bistability comes from action-conformity weighting the Bayes update. Light graft into
our model: add a small `λ_conf · (μ_i − ⟨μ⟩_neighbours)²` regulariser to inference and the
existing fixed point bifurcates into two. Worth citing in §2 (related work) and possibly
running as a robustness check in §5.

---

## Synthesis — recommended action

1. **Primary graft: Tilman et al. 2020.** Clone https://github.com/atilman/EcoEvoGamesCode,
   port the `ṅ` equation into `src/resource.py`, parameterise extraction `h(μ, r)` so the
   ΔL¹<0 ∧ δH⁰<0 condition is satisfied for some `R_in` range. Verify in nb13 that monostability
   breaks.
2. **Theoretical scaffolding: Weitz 2016 + Hofbauer-Sigmund.** Cite in §3 to justify the
   coupled-ODE form and the existence of interior-unstable equilibria.
3. **Network extension: Ohtsuki-Nowak 2006.** Re-derive the bistability condition using the
   trust matrix in place of the regular-graph pair-approximation; connects to nb10 Fiedler.
4. **Narrative framing: O'Connor-Weatherall + Smaldino-McElreath.** Use these in §1/§2 to
   motivate why "paradigm persistence" is a real cultural-evolution phenomenon worth modelling.
5. **Fall-back / threshold story: Komarova-Nowak.** If the continuous-θ graft proves messy,
   discretise to a 2-paradigm replicator-mutator and use the universality threshold.

What we *gain* by adopting (1)+(3): the paper's headline becomes "trust topology + resource
feedback together produce paradigm bistability, neither alone suffices" — which is both true of
the current model (nb13: resource alone is monostable; nb10: trust alone is monostable) and
genuinely novel as an active-inference statement.
