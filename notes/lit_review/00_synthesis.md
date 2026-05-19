# Lit review synthesis — paradigm-dynamics references with working code

*Generated 2026-05-19 from 7 parallel angle-reviews (`01_…07_….md` in this folder).*
*Companion to `notes/dynamical_systems_diagnosis.tex`.*

---

## Executive verdict

The bistability probe (nb13) showed our substrate is structurally monostable — a contraction map. The seven-angle literature review surfaced **two equally viable strategic paths**, plus one experiment we should run regardless:

| Path | Headline claim | Substrate change | Cost |
|---|---|---|---|
| **A** | "Trust-mediated paradigm capture" *(original framing, repaired)* | Graft one nonlinearity (4 concrete candidates below) | 1–3 weeks |
| **B** | "Why paradigm shifts are rare/slow" *(reframed around declining disruption)* | None — keep monostable substrate | 3–5 days, mostly writing |
| **Both** | Concrete empirical falsifier from Azoulay et al. 2019 | None | 1 notebook |

The Azoulay falsifier — ablate the highest-trust node, measure whether outsider belief-mass on the alternative paradigm rises — is **runnable on current `src/` with no model changes** and is informative under both paths. Do it first.

---

## The strategic fork

### Path B is the surprise

Angle 7 (science-of-science empirics) flipped the framing. The dominant signal in modern empirical science-of-science is *not* sharp Kuhnian revolutions — it is **monotone incrementalisation and lock-in**:

- **Park, Leahey, Funk 2023 (Nature):** CD-index of papers fell 91.5–100% across fields 1945–2010; patents fell 78.7–91.5% from 1980. Petersen et al. 2024 critique it as a citation-inflation artefact; cite both honestly.
- **Wu, Wang, Evans 2019 (Nature):** small teams disrupt, big teams develop. Large teams dominate science. Pushes toward incumbency.
- **Foster, Rzhetsky, Evans 2015 (ASR):** scientists strongly bias toward "tradition" strategies even when "innovation" strategies have higher expected reward — the trust-mediated incumbency our linear model produces is *the* empirical observation, not a model failure.

A monostable, slow-relaxation attractor is the structurally *correct* shape for this picture. Our nb12 "B shifts faster than A but neither produces real bistability" finding becomes the empirical content — not a bug.

### Path A is more ambitious but needs a real choice

Four concrete mechanisms emerged across the four "math/modelling" angles (1, 2, 4, 5). All four are convergent in one respect: **the cheapest graft is a saturating/sigmoid nonlinearity in the social or resource channel, not in the Bayesian belief update**.

| # | Mechanism | Paper | Working code | `src/` cost | AIF-native? | What it lets us claim |
|---|---|---|---|---|---|---|
| A1 | tanh-saturated social coupling, pitchfork at critical attention gain | Bizyaeva, Franci, Leonard 2023, IEEE TAC | No official repo (~30 line reimpl) | 1 line `population.py` | Yes — gain = "precision on social prior" | "Genuine pitchfork in a Bayesian-coupled population, controlled by social-attention precision" |
| A2 | Cubic potential per node + Laplacian coupling = discrete Allen-Cahn | **Boyd, Porter, Bertozzi 2020, *J. Nonlin. Sci.*** | None directly; numpy implementation trivial | 1 line `inference.py` | **Yes — they prove MLE on SBM ≡ discrete Allen-Cahn, so cubic is the SBM log-likelihood gradient flow** | "Paradigm bistability is the SBM-Bayesian posterior, not an ad-hoc nonlinearity" |
| A3 | Softmax-over-neighbour-attention with precision γ; learned Dirichlet habit | Albarracin et al. 2022, *Entropy* | `infer-actively/pymdp` (Python, active) | New `attention.py` + rewrite `trust.py` | **Yes — only genuine bistability in the multi-agent AIF literature** | "Polarised echo-chambers vs consensus via precision modulation in the AIF social prior" |
| A4 | Eco-evolutionary game: `ṙ = ε₁(R_in − ρr) − ε₂ h(μ, r)`, constructive bistability condition | Tilman, Plotkin, Akçay 2020, *Nat. Commun.* | `atilman/EcoEvoGamesCode` (Python) | Rewrite `resource.py` from readout to ODE | Compatible — Hyland action selection intact | "Bistability requires belief–resource coupling AND a μ-dependent extraction nonlinearity" |

**Path A picking heuristic.** A2 is the lowest-code and best-theoretically-justified (the cubic is *Bayesian*, not an ad-hoc fold). A1 is the lowest-code in an AIF-direct sense (attention precision is a familiar AIF lever). A3 is the most ambitious but the only one with an established AIF pedigree of producing real bistability. A4 is the only one that genuinely upgrades resource from a readout to a dynamic state, which is what the paper's "coupled feedback loops" language has been promising.

### A bridge result that connects the two paths

**Ohtsuki & Nowak 2006, *J. Theor. Biol.*** — replicator equation on graphs with pair-approximation correction. The 2×2 game's bistability condition becomes *graph-degree-dependent*: rederive substituting our trust matrix `Γ` for the regular-graph operator and the result should reduce to a Fiedler inequality on `λ₂(Γ)`. **This directly connects nb10's vanguard finding ("isolated-vanguard overconfidence floods the mainstream") to the bistability condition under any of the Path A grafts.**

---

## The Azoulay falsifier (run-this-regardless experiment)

**Azoulay, Fons-Rosen, Graff Zivin 2019, *AER* — already in `refs.bib`.** Empirical finding: removal of a star scientist causes an 8.6% sustained boost in non-collaborator article flow into the deceased star's subfield, disproportionately from highly-cited outsiders.

**The experiment, runnable on current `src/`:**

1. Initialise SBM population at consensus on θ_A = 0, with one high-trust node (top-1% in aggregate incoming Γ).
2. Introduce θ\* = 1 (the alternative paradigm) and run T steps.
3. At t = T/2, *ablate the high-trust node* (remove it from the graph, redistribute its incoming trust).
4. Measure: (a) belief-mass of outsider community on θ\* before vs after ablation; (b) time-to-shift for outsider community in ablated vs non-ablated control; (c) Gini of resource flow before vs after.

**Outcome under Path B (monostable substrate):** if the model reproduces Azoulay's *direction* (outsider belief-mass on alternative paradigm rises faster post-ablation) we have a concrete empirical hit even without bistability — this becomes the paper's main empirical experiment.

**Outcome under Path A (post-graft):** the experiment becomes the comparative-static against which the graft is validated. Pre-graft (current code) gives the slow-tracking signature; post-graft should give *both* slow tracking *and* a genuine basin-switching event for some parameter range.

Cost: one notebook (`notebooks/15_azoulay_falsifier.ipynb`), ~1 day.

---

## The "don't reinvent the wheel" warning

**Rodriguez-Sickert et al. 2015, *PLOS ONE*** is the closest competitor — an ABM of paradigm shifts with social influence and a "conservative force." If we go Path A we need to read it carefully and articulate what our active-inference grounding adds that they didn't already do. The honest claim will be: (i) AIF formal grounding, (ii) explicit Bayesian trust matrix as the only learned relational object, and (iii) the Hyland λ·U frame — *not* the dynamical phenomenon itself, which they already produced.

---

## Recommended next steps, ordered

1. **(1 day) Read Park-Leahey-Funk 2023, Wu-Wang-Evans 2019, Azoulay 2019.** These three decide whether Path B is intellectually honest.
2. **(1 day) Build `notebooks/15_azoulay_falsifier.ipynb`** on current code. Outcome is informative regardless of A/B choice.
3. **(½ day) Path decision:** A or B, single sentence justification, committed to a fresh memory note.
4. **(A: 1–3 weeks / B: 3–5 days) Execute.**
   - Path A: prototype A2 (Boyd-Porter cubic) first — single line of code, theoretically Bayesian. Fall back to A1 (Bizyaeva tanh) if A2 doesn't produce a clean fold under our parameters.
   - Path B: rewrite §1, §4.5, §5 of `paper/main.tex` around "incumbency mechanism + Azoulay falsifier"; cite Park-Leahey-Funk, Wu-Wang-Evans, Foster-Rzhetsky-Evans as motivation; cite Petersen et al. critique as honesty signal.

---

## BibTeX entries to add to `refs.bib`

The individual angle files (`01_…07_….md`) have full BibTeX. The **high-priority adds** (any path) are:

```
@article{bizyaevaFranci2023NonlinearOpinionDynamics, ...}      # A1 / both paths
@article{boydPorterBertozzi2020SBMSurfaceTension, ...}         # A2 / both paths
@article{tilmanPlotkinAkcay2020EcoEvoGames, ...}              # A4 / both paths
@article{ohtsukiNowak2006ReplicatorGraphs, ...}                # bridge / both paths
@article{parkLeaheyFunk2023DecliningDisruption, ...}           # Path B headline
@article{petersenArroyavePammolli2024DisruptionArtifact, ...}  # Path B honesty
@article{wuWangEvans2019SmallTeams, ...}                       # Path B / both
@article{fosterRzhetskyEvans2015TraditionInnovation, ...}      # Path B
@article{andreoniNikiforakisSiegenthaler2021Tipping, ...}      # A1 / Path B
@article{centolaBecker2018TippingPoints, ...}                  # both paths
@article{oconnorWeatherall2019Misinformation, ...}             # A3 / Path B
@article{zollman2010TransientDiversity, ...}                   # both paths
@article{heinsDaCosta2024CollectiveSurprise, ...}              # already in refs.bib as heins2024CollectiveSurprise — check
@article{rodriguezSickert2015ParadigmShifts, ...}              # don't-reinvent-the-wheel
```

(The seven angle files have ready-to-paste BibTeX blocks. I'll dedupe and append in one pass after the path decision.)

---

## Cross-angle convergence (the strongest signal)

Five of the seven angles independently surfaced the same minimal nonlinearity recipe:

- Angle 1 (sociophysics): tanh-saturated social coupling
- Angle 2 (collective AIF): softmax-over-neighbour-attention with precision γ
- Angle 3 (epistemic networks): endogenous mistrust update / sigmoid-gated social signal
- Angle 4 (nonlinear dynamics): cubic node potential + Laplacian coupling (the same mathematical object as a softened threshold)
- Angle 6 (tipping empirics): heterogeneous Granovetter threshold likelihood

These are all *the same mechanism*, dressed in five different vocabularies: **a saturating nonlinearity in the social channel, parameterised by a gain that becomes a bifurcation parameter**. The Boyd-Porter-Bertozzi 2020 result is the punchline: this mechanism is not arbitrary, it is the *gradient flow of the SBM log-likelihood under Bayesian inference*. That is the cleanest Path A graft and the one that requires no defence against "ad-hoc" criticism.

Only Angle 7 (sci-of-sci empirics) and Angle 5 (resource-coupled fitness) point elsewhere — Angle 7 to Path B (reframe), Angle 5 to Path A4 (upgrade resource layer to a real dynamic state). Both are coherent, both substantive.

---

*End synthesis.*
