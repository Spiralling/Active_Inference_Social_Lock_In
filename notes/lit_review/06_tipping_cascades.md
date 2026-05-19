# 06. Tipping points, critical-mass cascades, threshold contagion — empirical bistability

**Scope.** Survey of social-tipping / threshold-cascade literature filtered for *empirically-validated* bistable mechanisms, i.e. mechanisms where a real experiment, observational dataset, or field study (not just a simulation) supports the existence of an abrupt transition or coexisting attractors. The goal is to identify candidate grafts onto our current monostable contraction substrate (`src/world.py`, `src/inference.py`, `src/policy.py`, `src/trust.py`) that preserve the Bayesian-agent skeleton.

**Exec summary (3 sentences).** The single strongest empirical anchor for bistable social transitions is the Centola et al. (2018) tipping-point experiment, replicated/refined by Andreoni-Nikiforakis-Siegenthaler (2021) with a structural threshold-model fit on 54 lab societies — both demonstrate a critical-mass discontinuity at ~25% committed minority. The cleanest mechanistic graft for our codebase is a *heterogeneous-threshold reweighting of the social-signal likelihood* (Granovetter / Watts microfoundation by Wiedermann et al. 2020), which is fully compatible with Bayesian agents because it modifies the *likelihood shape*, not the inference rule. Couzin-style quorum-sensing and Dakos/Scheffer's critical-slowing-down diagnostics provide independent corroboration that abrupt transitions in collective systems are real and that early-warning signatures can be measured — but those are diagnostic tools, not new mechanisms to graft.

**Key takeaway for the paper.** We do not need to abandon Bayesian agents to get bistability. Two structurally compatible routes: (a) make the social-signal likelihood *nonlinear in adopter fraction* (complex-contagion / threshold reweighting), which produces a saddle-node in the mean-field; (b) make trust update *threshold-gated* (signal-confirmed cascades only), which is closer to the empirical complex-contagion finding that affirmation from multiple independent sources is required.

---

## 1. Centola, Becker, Brackbill, Baronchelli (2018) — *Experimental evidence for tipping points in social convention*

**BibTeX.**
```bibtex
@article{centola2018experimental,
  title={Experimental evidence for tipping points in social convention},
  author={Centola, Damon and Becker, Joshua and Brackbill, Devon and Baronchelli, Andrea},
  journal={Science},
  volume={360},
  number={6393},
  pages={1116--1119},
  year={2018},
  publisher={American Association for the Advancement of Sciences},
  doi={10.1126/science.aas8827}
}
```

**Code.** No official lab repository for the *2018 experiment itself*. Related Centola-lab/collaborator code: `https://github.com/drguilbe/complexpaths` (Guilbeault & Centola 2021 *Nat Commun* topological measures for complex contagion; R + AddHealth data). The 2018 paper's experimental platform is custom and not publicly distributed.

**Mechanism (one sentence).** Coordination game with naming-game payoffs: a committed minority overturns the majority convention iff its fraction exceeds a critical value (~25%) set by a saddle-node bifurcation in the mean-field replicator dynamics on a memory-1 imitation rule.

**Empirical validation.** Online lab experiment, 10 groups × 20 participants each, financial incentive to coordinate on a linguistic label; a confederate "minority" of varying size attempts to overturn the norm. Observed sharp transition between 21% and 31% committed minority — direct empirical observation of a tipping point, with replications across sessions.

**Graft analysis.** Replace our current symmetric Gaussian likelihood pull on θ with a coordination-game payoff structure. In `src/inference.py`, the agent's likelihood over peer signals becomes a *winner-take-all majority rule* over a discrete label space, not a continuous Gaussian. This is a bigger surgery than option (b) below: we'd be moving from continuous θ to a discrete-label population, losing the natural connection to "continuous paradigm coordinate."

---

## 2. Andreoni, Nikiforakis, Siegenthaler (2021) — *Predicting social tipping and norm change in controlled experiments*

**BibTeX.**
```bibtex
@article{andreoni2021predicting,
  title={Predicting social tipping and norm change in controlled experiments},
  author={Andreoni, James and Nikiforakis, Nikos and Siegenthaler, Simon},
  journal={Proceedings of the National Academy of Sciences},
  volume={118},
  number={16},
  pages={e2014893118},
  year={2021},
  publisher={National Academy of Sciences},
  doi={10.1073/pnas.2014893118}
}
```

**Code / data.** Experimental data + instructions deposited at OpenICPSR `https://doi.org/10.3886/E134021V4`. No model-fitting code repository — model is analytical (threshold distribution → equilibrium), fit reported in SI.

**Mechanism (one sentence).** Each individual has a private threshold `t_i` (fraction of others who must deviate before they do); the *distribution* of thresholds determines whether the no-deviation and full-deviation equilibria are both stable (bistability) or only one is — i.e. classical Granovetter mechanism, but with thresholds estimated from individual-level willingness-to-pay data.

**Empirical validation.** 54 experimental societies × 20 participants. Authors *estimate the threshold distribution from incentive-compatible elicitations*, then *predict* which societies will tip under each intervention, then *test* that prediction. This is one of the very few studies that fits a structural bistable model and forecasts out-of-sample — the strongest evidence we have that the Granovetter mechanism is empirically real, not just stipulated.

**Graft analysis.** Add a `threshold_i` parameter per agent in `src/world.py`. In `src/inference.py`, the social-signal likelihood is *gated*: peer evidence enters the Bayesian update only if the fraction of peers with high posterior on the alternative paradigm exceeds `threshold_i`. This is a *minimally invasive* graft: agents remain Bayesian, but the likelihood-mapping from peer-state → evidence becomes nonlinear (sigmoid/step). The mean-field of this is provably bistable for sufficiently steep gating + heterogeneous thresholds. **Strong recommend.**

---

## 3. Granovetter (1978) — *Threshold models of collective behavior*

**BibTeX.**
```bibtex
@article{granovetter1978threshold,
  title={Threshold models of collective behavior},
  author={Granovetter, Mark},
  journal={American Journal of Sociology},
  volume={83},
  number={6},
  pages={1420--1443},
  year={1978},
  publisher={University of Chicago Press}
}
```

**Code.** None (1978 theoretical paper).

**Mechanism (one sentence).** Each actor has a private threshold = fraction of others who must adopt before they will adopt; the *distribution* of thresholds maps onto multiple stable adoption fractions — small shifts in the distribution can shift the system between equilibria.

**Empirical validation.** Original paper is theoretical (riot illustrations); validation came later — Andreoni et al. 2021 above; observational work on protests, fads, smoking cessation. This is the foundational citation, not the empirical evidence itself.

**Graft analysis.** Same as #2 — the threshold mechanism is what Andreoni et al. operationalised. Cite Granovetter as the theoretical antecedent.

---

## 4. Watts (2002) — *A simple model of global cascades on random networks*

**BibTeX.**
```bibtex
@article{watts2002simple,
  title={A simple model of global cascades on random networks},
  author={Watts, Duncan J.},
  journal={Proceedings of the National Academy of Sciences},
  volume={99},
  number={9},
  pages={5766--5771},
  year={2002},
  publisher={National Academy of Sciences},
  doi={10.1073/pnas.082090499}
}
```

**Code.** Multiple third-party Python ports: `https://github.com/benmaier/thresholdmodel` (continuous-time Watts on networkx); `https://github.com/georgeberry/thresholds`; `https://github.com/pik-copan/pycascades` (PIK tipping-cascade framework). No official Watts repo.

**Mechanism (one sentence).** Network version of Granovetter: nodes have heterogeneous absolute-fraction thresholds; bistability between no-cascade and global-cascade phases arises as a percolation transition in the *vulnerable-node subgraph* whose width depends on threshold variance.

**Empirical validation.** The 2002 paper itself is simulation-only. Empirical fit came via the 2018 Centola experiment and the 2021 Andreoni experiment. The Watts model is the canonical *network-aware* version of the mechanism Andreoni et al. validate.

**Graft analysis.** If we move to network heterogeneity (`src/world.py` already has community structure), Watts gives the natural mathematical form: threshold = absolute fraction of neighbours. Combine with #2's per-agent threshold sampling. The PIK `pycascades` package is the closest off-the-shelf reference implementation.

---

## 5. Wiedermann, Smith, Heitzig, Donges (2020) — *A network-based microfoundation of Granovetter's threshold model for social tipping*

**BibTeX.**
```bibtex
@article{wiedermann2020network,
  title={A network-based microfoundation of {G}ranovetter's threshold model for social tipping},
  author={Wiedermann, Marc and Smith, E. Keith and Heitzig, Jobst and Donges, Jonathan F.},
  journal={Scientific Reports},
  volume={10},
  number={1},
  pages={11202},
  year={2020},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-020-67102-6}
}
```

**Code.** PIK group maintains `https://github.com/pik-copan/pycascades` (general tipping-cascades framework) — this paper's specific code is not in a dedicated repo but the methods are implementable in pycascades.

**Mechanism (one sentence).** Population is split into "certain actors" (already active) and "potential actors" whose activation threshold is *derived* from a network-cascade model rather than postulated — yielding broad threshold distributions consistent with Granovetter's bistability requirement.

**Empirical validation.** Theoretical derivation; the validation is that the *shape of the derived threshold distribution* matches what social-movement theory and Andreoni-style elicitations actually find. Closest thing to a principled link from micro-Bayesian mechanics to Granovetter's macro postulate.

**Graft analysis.** This is the *theoretical bridge* paper for our project: it shows that you can derive Granovetter thresholds from a microfounded social model. Cite it to justify our graft #2 — we're not just adding a threshold, we're using a derivation that links it to the underlying agent-level dynamics. Direct relevance to keeping the IWAI framing principled.

---

## 6. Couzin, Krause, Franks, Levin (2005) + Couzin et al. (2011) — *Collective decision-making with hysteresis*

**BibTeX.**
```bibtex
@article{couzin2005effective,
  title={Effective leadership and decision-making in animal groups on the move},
  author={Couzin, Iain D. and Krause, Jens and Franks, Nigel R. and Levin, Simon A.},
  journal={Nature},
  volume={433},
  number={7025},
  pages={513--516},
  year={2005},
  doi={10.1038/nature03236}
}
@article{couzin2011uninformed,
  title={Uninformed individuals promote democratic consensus in animal groups},
  author={Couzin, Iain D. and Ioannou, Christos C. and Demirel, G{\"u}ven and Gross, Thilo and Torney, Colin J. and Hartnett, Andrew and Conradt, Larissa and Levin, Simon A. and Leonard, Naomi E.},
  journal={Science},
  volume={334},
  number={6062},
  pages={1578--1580},
  year={2011},
  doi={10.1126/science.1210280}
}
```

**Code.** No official lab GitHub for these specific papers. Couzin lab (now Konstanz/MPI) has released related collective-motion code but not a canonical 2005/2011 repo. Matt Sosna's "Behind the Scenes" walkthrough of Couzin 2011 (`https://mattsosna.com/BTS-Couzin2011/`) is the most accessible public re-implementation.

**Mechanism (one sentence).** Quorum-sensing decision rule: an individual switches direction (paradigm) when the fraction of neighbours pointing the other way exceeds a sharp sigmoid threshold — producing bistability and hysteresis ("collective memory") even when individuals have no memory.

**Empirical validation.** Live-animal experiments with fish schools (golden shiners), measured group-state transitions; computational results match observed transition statistics. Bistability is *observed*, not just simulated.

**Graft analysis.** Mathematically equivalent to graft #2 with a sigmoid (rather than step) gate. The Couzin work is our *strongest non-human empirical anchor* — useful for framing because the math is identical to threshold contagion but the substrate is biological, weakening any objection that the bistability is an artefact of human strategic behaviour.

---

## 7. Dakos, Scheffer, van Nes, Brovkin, Petoukhov, Held (2008) — *Slowing down as an early warning signal for abrupt climate change*

**BibTeX.**
```bibtex
@article{dakos2008slowing,
  title={Slowing down as an early warning signal for abrupt climate change},
  author={Dakos, Vasilis and Scheffer, Marten and van Nes, Egbert H. and Brovkin, Victor and Petoukhov, Vladimir and Held, Hermann},
  journal={Proceedings of the National Academy of Sciences},
  volume={105},
  number={38},
  pages={14308--14312},
  year={2008},
  doi={10.1073/pnas.0802430105}
}
@article{scheffer2009early,
  title={Early-warning signals for critical transitions},
  author={Scheffer, Marten and Bascompte, Jordi and Brock, William A. and Carpenter, Stephen R. and Dakos, Vasilis and Held, Hermann and van Nes, Egbert H. and Rietkerk, Max and Sugihara, George},
  journal={Nature},
  volume={461},
  number={7260},
  pages={53--59},
  year={2009},
  doi={10.1038/nature08227}
}
```

**Code.** R package `earlywarnings` on CRAN, maintained by Dakos: `https://github.com/earlywarningtoolbox/earlywarnings-R`. Production-quality.

**Mechanism (one sentence).** *Not a tipping mechanism* — a diagnostic: near any saddle-node bifurcation, the dominant eigenvalue of the linearised dynamics goes to zero, producing increased variance and lag-1 autocorrelation in fluctuations.

**Empirical validation.** Eight palaeoclimate records of past abrupt transitions; replicated in lake ecosystems, mood dynamics, financial crises.

**Graft analysis.** Not a graft — use this as a *test* for whether any mechanism we add actually produces a bifurcation. If we successfully introduce bistability via graft #2, we should observe CSD signatures (variance↑, AR(1)↑) in `src/observables.py` time series before the transition. Add this as a diagnostic in nb13.

---

## 8. Cinelli, De Francisci Morales, Galeazzi, Quattrociocchi, Starnini (2021) — *The echo chamber effect on social media*

**BibTeX.**
```bibtex
@article{cinelli2021echo,
  title={The echo chamber effect on social media},
  author={Cinelli, Matteo and De Francisci Morales, Gianmarco and Galeazzi, Alessandro and Quattrociocchi, Walter and Starnini, Michele},
  journal={Proceedings of the National Academy of Sciences},
  volume={118},
  number={9},
  pages={e2023301118},
  year={2021},
  doi={10.1073/pnas.2023301118}
}
```

**Code.** Authors have released related code in various per-paper repos (Quattrociocchi group on GitHub `https://github.com/CHANGES-PoliMi/`). No single canonical repo for this paper.

**Mechanism (one sentence).** Bounded-confidence opinion dynamics + homophilic rewiring produce coexistence of polarised clusters; bistability is between consensus and polarised attractors, controlled by an interaction-radius parameter.

**Empirical validation.** Observational only: 100M+ posts across Gab, Facebook, Reddit, Twitter, showing user clustering in homophilic communities for controversial topics. The *bistability claim* itself is from companion modelling work (Baumann et al. 2020 PRL), fit to the Cinelli data.

**Graft analysis.** Less direct than #2/#6: bounded-confidence is not naturally Bayesian. Would require either (a) replacing posterior update with a confidence-bound truncation, breaking the AIF skeleton, or (b) reinterpreting trust gain as a function of belief similarity, which is close to what `src/trust.py` already does but doesn't add bistability without further nonlinearity. Lower priority graft, but useful citation for "echo-chamber" framing.

---

## Summary table

| # | Paper | Empirical strength | Code availability | Bayesian-compatible graft? |
|---|-------|-------------------|-------------------|----------------------------|
| 1 | Centola 2018 | **Strong** (lab experiment, replicated) | Indirect (Guilbeault/Centola repo) | Hard — requires discrete labels |
| 2 | Andreoni 2021 | **Strongest** (54 societies, fit + forecast) | Data only (OpenICPSR) | **Yes — minimal surgery** |
| 3 | Granovetter 1978 | Foundational (theoretical) | N/A | Yes (via #2) |
| 4 | Watts 2002 | Simulation only | **Yes** (pycascades, thresholdmodel) | Yes (network version of #2) |
| 5 | Wiedermann 2020 | Theoretical bridge | pycascades-compatible | Yes — *the* citation for graft #2 |
| 6 | Couzin 2005/2011 | **Strong** (live animal expts) | Partial (Sosna walkthrough) | Yes (sigmoid version of #2) |
| 7 | Dakos/Scheffer | Diagnostic (validated across domains) | **Yes** (earlywarnings R pkg) | Not a graft — use as test |
| 8 | Cinelli 2021 | Observational (large-scale) | Partial (related repos) | Awkward — would break AIF skeleton |

## Recommended graft path

1. **Add per-agent threshold `t_i` in `src/world.py`** sampled from a heterogeneous distribution (Andreoni-style).
2. **In `src/inference.py`**, modify the social-signal likelihood so peer evidence enters the Bayesian update *gated by a sigmoid in the alternative-paradigm fraction among neighbours*, with steepness `β` and offset `t_i`. Bayesian inference itself is unchanged — only the likelihood shape is.
3. **Cite Wiedermann 2020** as the principled microfoundation that lets us derive (not just postulate) the threshold distribution from agent-level dynamics.
4. **Validate via CSD (Dakos)** — in `src/observables.py` compute lag-1 autocorrelation and variance of mean μ across the population near the transition; CSD signatures would confirm we now have a saddle-node, not a contraction.
5. **Frame with Couzin 2005/2011** as the cross-substrate empirical evidence that the mechanism is real beyond strategic humans.

The β → 0 limit recovers our current monostable contraction map, so this graft is a *strict generalisation*, useful for the discussion section ("our previous model is the linear-likelihood limit of...").
