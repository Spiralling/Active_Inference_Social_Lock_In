# Lit Review 04 — Canonical Nonlinear Dynamics on Graphs

## Executive summary (3 sentences)

Our linear-consensus + Gaussian-conjugate substrate is provably monostable (negative-definite Jacobian, Hartman-Grobman); to recover bistability / hysteresis / paradigm-persistence we need a node-level nonlinear potential or a coupling-mediated symmetry breaking. The applied-math literature offers two clean mechanism libraries — **graph Allen-Cahn / Ginzburg-Landau** (Bertozzi, van Gennip, Boyd-Porter) and **saturating-nonlinearity opinion dynamics** (Bizyaeva-Franci-Leonard) — both of which compose with our pooling/inference loop with minimal architectural surgery. The Boyd-Porter-Bertozzi (2020) result is the *direct* point of entry for us because it shows MLE on an SBM is literally a discrete surface-tension problem, so an Allen-Cahn term added per node on our SBM substrate is the most theoretically defensible non-linearisation available.

---

## 1. Bertozzi & Flenner — "Diffuse Interface Models on Graphs for Classification of High-Dimensional Data"

```bibtex
@article{bertozzi2012diffuse,
  author  = {Bertozzi, Andrea L. and Flenner, Arjuna},
  title   = {Diffuse Interface Models on Graphs for Classification of High Dimensional Data},
  journal = {Multiscale Modeling \& Simulation},
  volume  = {10},
  number  = {3},
  pages   = {1090--1118},
  year    = {2012},
  doi     = {10.1137/11083109X}
}
```

**Code**: companion MATLAB code historically distributed by Bertozzi's UCLA group at https://www.math.ucla.edu/~bertozzi/ (no canonical GitHub mirror — say so explicitly). PyTorch / NumPy reimplementations exist as undergraduate-thesis tier code on GitHub but no canonical reference repo.

**Nonlinearity**: Per-node double-well potential `W(u) = (1/4)(u² − 1)²` added to graph-Laplacian smoothing, giving the gradient flow `du/dt = −εLu − (1/ε)W′(u)` with `W′(u) = u³ − u`. Two stable wells at u = ±1, unstable at u = 0.

**Graft analysis**: Add a cubic term to the post-pool mean update in `src/inference.py`. The natural place is between `precision_pool` and the trust update: replace the linear blend with `μ_new = μ_pool − η · (μ_pool³ − μ_pool)` (centred on the relevant paradigm well rather than 0). Stays inside the Gaussian-conjugate frame because precision propagation is untouched — only the *mean* picks up the nonlinearity. AIF reading: each agent has two preferred-prior basins (the "Newtonian" and "relativistic" priors); the cubic is a free-energy gradient toward whichever basin µ is closer to. Cleanest of all options; minimal code change.

---

## 2. van Gennip & Bertozzi — "Γ-convergence of graph Ginzburg-Landau functionals"

```bibtex
@article{vangennip2012gamma,
  author  = {van Gennip, Yves and Bertozzi, Andrea L.},
  title   = {{$\Gamma$}-convergence of graph {G}inzburg-{L}andau functionals},
  journal = {Advances in Differential Equations},
  volume  = {17},
  number  = {11--12},
  pages   = {1115--1180},
  year    = {2012},
  url     = {https://arxiv.org/abs/1204.5220}
}
```

**Code**: no public reference repo. Mathematical paper.

**Nonlinearity**: graph GL energy `E_ε(u) = (ε/2) u^T L u + (1/ε) Σ_i W(u_i)`. Same double-well as #1; this paper supplies the *theoretical justification* (Γ-limit is graph-cut / total-variation seminorm as ε → 0) for why the Allen-Cahn flow is the right gradient flow on a graph.

**Graft analysis**: Pure theory paper — we cite it as the *legitimisation* of #1 inside the paper, not as code. Modules unchanged; provides the rigorous footing for "graph Allen-Cahn approximates graph-cut", which is exactly the structural story we need on an SBM (intra-community pooling vs. inter-community cut).

---

## 3. Boyd, Porter & Bertozzi — "Stochastic Block Models are a Discrete Surface Tension"

```bibtex
@article{boyd2020sbm,
  author  = {Boyd, Zachary M. and Porter, Mason A. and Bertozzi, Andrea L.},
  title   = {Stochastic Block Models are a Discrete Surface Tension},
  journal = {Journal of Nonlinear Science},
  volume  = {30},
  number  = {5},
  pages   = {2429--2462},
  year    = {2020},
  doi     = {10.1007/s00332-019-09541-8},
  url     = {https://arxiv.org/abs/1806.02485}
}
```

**Code**: implementation walkthrough in the paper; no canonical GitHub. MATLAB MBO scheme is reconstructable from the appendix and is short (~50 lines). Boyd's github at https://github.com/zboyd2 has related material.

**Nonlinearity**: MBO threshold dynamics — alternate Laplacian diffusion (smoothing) with hard thresholding at 0. Equivalent to the small-ε limit of Allen-Cahn. The headline result: maximum-likelihood community detection on an SBM is mathematically identical to discrete surface tension.

**Graft analysis**: **THIS IS THE KEY REFERENCE FOR US.** We *already* use SBM. Boyd-Porter-Bertozzi gives us the math that says: if we add a double-well node potential on top of SBM-Laplacian pooling, we are running the gradient flow of an Allen-Cahn energy that limits to MLE community recovery. This justifies a per-node `μ ↦ μ − η(μ³ − μ)` step inside `src/inference.py` *as a Bayesian object*, not an ad-hoc nonlinearity: each well is a paradigm-prior whose posterior the agent could in principle have inferred. No new modules. Touches `inference.py` only.

---

## 4. Bizyaeva, Franci & Leonard — "Nonlinear Opinion Dynamics with Tunable Sensitivity"

```bibtex
@article{bizyaeva2023nonlinear,
  author  = {Bizyaeva, Anastasia and Franci, Alessio and Leonard, Naomi E.},
  title   = {Nonlinear Opinion Dynamics With Tunable Sensitivity},
  journal = {IEEE Transactions on Automatic Control},
  volume  = {68},
  number  = {3},
  pages   = {1415--1430},
  year    = {2023},
  doi     = {10.1109/TAC.2022.3159527},
  url     = {https://arxiv.org/abs/2009.04332}
}
```

**Code**: Leonard Lab papers historically release MATLAB demo scripts on https://naomi.princeton.edu/ ; no canonical GitHub repo for this paper. Reimplementation in NumPy/JAX is a half-day of work (the ODE is one line).

**Nonlinearity**: `dz_i/dt = −d·z_i + u·S(α·z_i + γ·Σ_j A_ij z_j) + b_i` with saturation `S(y) = tanh(y)` (or any odd sigmoid). The product `u·α` is the gain on self-reinforcement; `u·γ` is the gain on social reinforcement. **As `u` crosses a critical threshold (set by the Perron eigenvector of A), the symmetric equilibrium z = 0 loses stability via a pitchfork bifurcation.** With slight asymmetry, the pitchfork unfolds into a saddle-node — exactly the cusp-catastrophe structure the paper needs.

**Graft analysis**: Stays *very* close to our AIF framing. Read `z_i` as our posterior mean `μ_i`. The saturation `tanh(α μ_i + γ Σ_j w_ij μ_j)` *replaces* the row-stochastic linear pool in `precision_pool`. The attention/gain `u` becomes a *resource-coupled* parameter (cf. our `src/resource.py`) — so resource modulates not the eigenvalue spectrum but the *bifurcation parameter itself*. This is the closest match to the story the paper is trying to tell. Touches `inference.py` (pooling) and `policy.py` (resource → u coupling). Loses strict Gaussian-conjugacy in the pool step (the tanh breaks linear-Gaussian closure on the mean), but precision can still be propagated separately — call it "quasi-conjugate".

---

## 5. Srivastava, Moehlis & Bullo — "On Bifurcations in Nonlinear Consensus Networks"

```bibtex
@article{srivastava2011bifurcations,
  author  = {Srivastava, Vaibhav and Moehlis, Jeff and Bullo, Francesco},
  title   = {On Bifurcations in Nonlinear Consensus Networks},
  journal = {Journal of Nonlinear Science},
  volume  = {21},
  number  = {6},
  pages   = {875--895},
  year    = {2011},
  doi     = {10.1007/s00332-011-9103-4}
}
```

**Code**: no public repo. Theory paper.

**Nonlinearity**: General nonlinear consensus `dx_i/dt = Σ_j a_ij f(x_j − x_i)` with `f` odd, monotone but bounded. Classifies the bifurcations (pitchfork, saddle-node, Hopf) you can get by perturbing a *linear* consensus protocol with f.

**Graft analysis**: Contrast / framing reference. Tells us *what's possible* in the neighbourhood of our linear baseline. Use it in §3 as the citation that justifies "to get the bifurcation we want, we must replace pooling with an odd-bounded nonlinearity" — bridges the gap from our current linear pool to #4. No code change implied directly.

---

## 6. Strogatz — *Nonlinear Dynamics and Chaos* (textbook)

```bibtex
@book{strogatz2015nonlinear,
  author    = {Strogatz, Steven H.},
  title     = {Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering},
  edition   = {2nd},
  publisher = {Westview Press},
  year      = {2015}
}
```

**Code**: no canonical repo; many community notebooks reproduce all examples (search github for `strogatz nonlinear dynamics notebook`).

**Nonlinearity**: Ch. 3 (1D bifurcations: saddle-node, transcritical, pitchfork, hysteresis); Ch. 8 (2D bifurcations, including saddle-node-on-invariant-circle); Ch. 12 (Fisher's equation / front propagation). Ch. 3.6 "Imperfect Bifurcations and Catastrophes" is the cusp picture we need verbatim.

**Graft analysis**: Reference / pedagogy. Ch. 3.6 figure is the picture we should reproduce schematically in §4 to show *what we are claiming the model does* once we add the nonlinearity. No code change.

---

## 7. Kuznetsov — *Elements of Applied Bifurcation Theory*

```bibtex
@book{kuznetsov2004elements,
  author    = {Kuznetsov, Yuri A.},
  title     = {Elements of Applied Bifurcation Theory},
  edition   = {3rd},
  publisher = {Springer},
  series    = {Applied Mathematical Sciences},
  volume    = {112},
  year      = {2004},
  doi       = {10.1007/978-1-4757-3978-7}
}
```

**Code**: MatCont (https://sourceforge.net/projects/matcont/), AUTO (https://github.com/auto-07p/auto-07p), PyDSTool (https://github.com/robclewley/pydstool), BifurcationKit.jl (https://github.com/bifurcationkit/BifurcationKit.jl).

**Nonlinearity**: §3.5 covers the saddle-node bifurcation normal form `dx/dt = α − x²` and the cusp `dx/dt = α + βx − x³`. §5–7 cover continuation methods.

**Graft analysis**: If we get a candidate nonlinear model running, BifurcationKit.jl is the standard tool to *prove* the bifurcation diagram (continuation, two-parameter unfolding). Outside our JAX stack but exportable: dump the right-hand side, run continuation in Julia, ship the figure. No `src/` change.

---

## 8. Olfati-Saber & Murray — "Consensus Protocols for Networks of Dynamic Agents" (linear-baseline reference)

```bibtex
@inproceedings{olfati2003consensus,
  author    = {Olfati-Saber, Reza and Murray, Richard M.},
  title     = {Consensus Protocols for Networks of Dynamic Agents},
  booktitle = {Proceedings of the 2003 American Control Conference},
  volume    = {2},
  pages     = {951--956},
  year      = {2003},
  doi       = {10.1109/ACC.2003.1239709}
}
```

**Code**: pedagogical implementations everywhere; no canonical repo.

**Nonlinearity**: NONE. Linear protocol `dx_i/dt = Σ_j a_ij (x_j − x_i)`. Cited to make the contrast explicit: our current model is exactly this (in discrete time, with precision weights), and Olfati-Saber's theorems guarantee monostability — which is *why* our model is structurally monostable.

**Graft analysis**: The *negative* reference — cite to establish that "linear consensus ⟹ monostable" is the well-known structural result we have to break to get bistability. No code change.

---

## 9. Krause-Hegselmann / Deffuant — Bounded confidence opinion dynamics

```bibtex
@article{hegselmann2002opinion,
  author  = {Hegselmann, Rainer and Krause, Ulrich},
  title   = {Opinion Dynamics and Bounded Confidence: Models, Analysis and Simulation},
  journal = {Journal of Artificial Societies and Social Simulation},
  volume  = {5},
  number  = {3},
  year    = {2002},
  url     = {https://www.jasss.org/5/3/2.html}
}
```

**Code**: `ndlib` (https://github.com/GiulioRossetti/ndlib) implements Hegselmann-Krause and Deffuant in Python. Mesa-based replications abound on GitHub.

**Nonlinearity**: Threshold (non-smooth) — agent i averages over j only if `|x_i − x_j| < ε`. Produces multi-cluster bistable end states; closest mainstream-social-science analogue to what we want.

**Graft analysis**: Threshold gating in trust matrix `gamma`. Replace `gamma[i,j]` in `precision_pool` with `gamma[i,j] · 𝟙{|μ_i − μ_j| < ε_conf}`. Touches `trust.py` and `inference.py`. The non-smoothness is awkward inside the JAX/diff-everything style we're using (would need a smooth surrogate, e.g. sigmoid window), and the *Bayesian* interpretation is messy (why would a Bayesian agent ignore far-away neighbours?). Workable but theoretically less clean than #1 or #4.

---

## 10. Krapivsky & Redner — Mean-field Ising / Glauber dynamics on networks

```bibtex
@book{krapivsky2010kinetic,
  author    = {Krapivsky, Pavel L. and Redner, Sidney and Ben-Naim, Eli},
  title     = {A Kinetic View of Statistical Physics},
  publisher = {Cambridge University Press},
  year      = {2010},
  doi       = {10.1017/CBO9780511780516}
}
```

Plus, for the modern graph version:

```bibtex
@article{dommers2017ising,
  author  = {Dommers, Sander and Giardin{\`a}, Cristian and van der Hofstad, Remco},
  title   = {Ising critical exponents on random graphs},
  journal = {Communications in Mathematical Physics},
  volume  = {328},
  pages   = {355--395},
  year    = {2014},
  doi     = {10.1007/s00220-014-1992-2}
}
```

**Code**: `netket` (https://github.com/netket/netket) has Ising-on-graph examples; `graph-tool` Ising demos.

**Nonlinearity**: Discrete spins `s_i ∈ {±1}`, Glauber update with `β` (inverse temperature). Mean-field magnetisation satisfies `m = tanh(β·(J·m + h))` — same `tanh` self-consistency as Leonard #4 in steady state. On SBM at high β: bistability with magnetisation pinned per community; hysteresis under sweep of h.

**Graft analysis**: Would force discretising `μ` to ±1, abandoning continuous-θ. Useful as a *limit* of our model (write the partition function for a coarse-grained `μ → sgn(μ)` and recover Ising) but not as an implementation target. Requires throwing out the Gaussian-conjugate machinery in `inference.py`. **Cite for theoretical lineage, do not implement.**

---

## 11. Replicator dynamics on graphs — Madeo & Mocenni

```bibtex
@article{madeo2015replicator,
  author  = {Madeo, Dario and Mocenni, Chiara},
  title   = {Game Interactions and Dynamics on Networked Populations},
  journal = {IEEE Transactions on Automatic Control},
  volume  = {60},
  number  = {7},
  pages   = {1801--1810},
  year    = {2015},
  doi     = {10.1109/TAC.2014.2384755}
}
```

Plus the canonical text:

```bibtex
@book{sandholm2010population,
  author    = {Sandholm, William H.},
  title     = {Population Games and Evolutionary Dynamics},
  publisher = {MIT Press},
  year      = {2010}
}
```

**Code**: `EGTtools` (https://github.com/Socrats/EGTtools) implements network replicator in Python. `Nashpy` for game-theory back-end.

**Nonlinearity**: `dx_i^s / dt = x_i^s · (f_i^s(x) − Σ_r x_i^r f_i^r(x))` — payoff-difference times current share, hence intrinsically nonlinear (quadratic on the simplex).

**Graft analysis**: Requires re-interpreting our continuous `μ ∈ ℝ` as a *strategy share over discrete paradigms*. Conceptually a clean fit for "competing paradigms", but it abandons the regression-family Gaussian observation model entirely — you would replace `src/inference.py` and `src/world.py`. Far more invasive than #1 or #4.

---

## 12. Reaction-diffusion on networks — Nakao & Mikhailov

```bibtex
@article{nakao2010turing,
  author  = {Nakao, Hiroya and Mikhailov, Alexander S.},
  title   = {Turing Patterns in Network-Organized Activator-Inhibitor Systems},
  journal = {Nature Physics},
  volume  = {6},
  number  = {7},
  pages   = {544--550},
  year    = {2010},
  doi     = {10.1038/nphys1651}
}
```

Foundational Othmer-Scriven:

```bibtex
@article{othmer1971instability,
  author  = {Othmer, Hans G. and Scriven, L. E.},
  title   = {Instability and Dynamic Pattern in Cellular Networks},
  journal = {Journal of Theoretical Biology},
  volume  = {32},
  pages   = {507--537},
  year    = {1971},
  doi     = {10.1016/0022-5193(71)90154-8}
}
```

**Code**: Nakao group's site has MATLAB demos at https://www.nakao-net.com/ (no GitHub); `NetworkDynamics.jl` (https://github.com/JuliaDynamics/NetworkDynamics.jl) supports general reaction-diffusion on graphs.

**Nonlinearity**: Two species per node (activator `u`, inhibitor `v`) with bistable kinetics `f(u,v)`, `g(u,v)` plus graph-Laplacian diffusion. Produces multistability and Turing-like clustering on networks.

**Graft analysis**: We would need a second variable per node — natural candidate is `(μ_i, R_i)` where R is the resource. This is conceptually close to what we *are* doing (resource-coupled belief), but the nonlinear kinetics are the new ingredient. Requires hand-design of activator-inhibitor functional forms — less principled than Allen-Cahn (#1) for a Bayesian story. Touches `resource.py` + `inference.py`. Best to keep in reserve.

---

## 13. Catastrophe theory — Zeeman / Thom

```bibtex
@book{zeeman1977catastrophe,
  author    = {Zeeman, E. Christopher},
  title     = {Catastrophe Theory: Selected Papers, 1972--1977},
  publisher = {Addison-Wesley},
  year      = {1977}
}
```

```bibtex
@book{thom1975structural,
  author    = {Thom, Ren{\'e}},
  title     = {Structural Stability and Morphogenesis},
  publisher = {Benjamin},
  year      = {1975}
}
```

**Code**: not applicable; geometric / topological theory.

**Nonlinearity**: Cusp catastrophe normal form `V(x) = (1/4)x⁴ + (1/2)αx² + βx`; gradient flow `dx/dt = −(x³ + αx + β)`. Fold-line saddle-node bifurcations bound the cusp; inside the cusp region the system is bistable; sweeping `β` produces hysteresis.

**Graft analysis**: Provides the *framing language* for §4. Map `α` ↔ resource asymmetry / inflow ratio (our `R_in`), `β` ↔ external truth `θ*(t)`. The graph Allen-Cahn implementation (#1, #3) is literally a node-wise cusp gradient flow under a Laplacian coupling — so catastrophe theory is the right *language* and Allen-Cahn is the right *implementation*. No code change.

---

## Recommended reading priority for the paper

1. Boyd-Porter-Bertozzi 2020 (#3) — read first, it covers our exact setting (SBM).
2. Bizyaeva-Franci-Leonard 2023 (#4) — read second, it gives an alternative grafting route.
3. Bertozzi-Flenner 2012 (#1) + van Gennip-Bertozzi 2012 (#2) — for the formal Allen-Cahn machinery.
4. Strogatz Ch. 3.6 (#6) and Kuznetsov §3.5 (#7) — keep on the desk.
5. Srivastava-Moehlis-Bullo (#5) and Olfati-Saber (#8) — for the linear→nonlinear narrative bridge in §3.
