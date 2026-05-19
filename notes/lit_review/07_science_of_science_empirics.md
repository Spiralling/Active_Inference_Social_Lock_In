# 07 — Science-of-Science Empirics: What Paradigm Dynamics Actually Look Like

**Scope:** survey of empirical / quantitative-historical literature on paradigm dynamics, with an eye to (a) what our active-inference substrate must reproduce to be credible, (b) which simulation models already do this so we don't reinvent wheels, (c) anchor citations for §1 motivation and §6 discussion of the IWAI 2026 paper.

## Executive summary (3 sentences)

The dominant empirical pattern in modern science-of-science is **incrementalism**: scientists prefer tradition over innovation (Foster–Rzhetsky–Evans 2015), large teams develop while only small teams disrupt (Wu–Wang–Evans 2019), and the CD index of disruptiveness has declined for half a century (Park–Leahey–Funk 2023, though the trend is contested as a citation-inflation artefact). The cleanest causal evidence for *gatekeeping* — i.e., that paradigm persistence is *socially* enforced rather than just epistemically rational — is Azoulay–Fons-Rosen–Graff Zivin 2019: outsider entry into a star's subfield jumps ~8.6% after the star dies, with disproportionately highly-cited contributions. For our purposes the empirical literature points strongly to a *gradual / lock-in* picture punctuated by mostly-exogenous shocks (deaths, generational turnover), not the sharp bistable revolutions the original Kuhn rhetoric suggests — which is a *tailwind* for our monostable model if we reframe the headline.

---

## 1. Park, Leahey & Funk (2023) — "Papers and patents are becoming less disruptive over time"

**BibTeX**
```bibtex
@article{park2023LessDisruptive,
  author  = {Park, Michael and Leahey, Erin and Funk, Russell J.},
  title   = {Papers and patents are becoming less disruptive over time},
  journal = {Nature},
  volume  = {613},
  number  = {7942},
  pages   = {138--144},
  year    = {2023},
  doi     = {10.1038/s41586-022-05543-x}
}
```

**Code / data:** Replication code and CD-index implementation: <https://russellfunk.org/cdindex/> (Russell Funk's lab page). Underlying CD-index Python package: `cdindex` on PyPI / GitHub (Funk lab). Real and public.

**Empirical claim (1 sentence):** Using the CD index on 45M papers and 3.9M patents, mean disruption declined 91.5–100% across fields between 1945 and 2010 (78.7–91.5% for patents 1980–2010), with the pattern attributed to scientists relying on a narrower slice of prior knowledge.

**Mechanism proposed (1 sentence):** Cognitive-load / "burden of knowledge" pushes researchers to specialise, citing a denser core of standard references, which mechanically lowers disruption scores.

**Implication for our paper:** **Phenomenon-to-reproduce target.** This is the single strongest motivating claim for "paradigm persistence" in modern science-of-science literature. Cite in §1 as the empirical headline our active-inference model attempts to explain mechanistically; the model's "monostable attractor with slow relaxation" reading is *closer* to a declining-disruption picture than to a Kuhnian-revolution picture, so this paper helps us reframe. Engage with the criticism (see §3 below).

---

## 2. Petersen, Arroyave & Pammolli (2024/2025) — "The disruption index is biased by citation inflation"

**BibTeX**
```bibtex
@article{petersen2024CitationInflation,
  author  = {Petersen, Alexander M. and Arroyave, Felber and Pammolli, Fabio},
  title   = {The Disruption Index Suffers From Citation Inflation and Is Confounded by Shifts in Scholarly Citation Practice},
  journal = {Journal of Informetrics},
  volume  = {19},
  number  = {1},
  year    = {2025},
  eprint  = {2406.15311},
  archivePrefix = {arXiv},
  doi     = {10.1016/j.joi.2024.101635}
}
```

**Code / data:** arXiv version provides reanalysis scripts; no formal GitHub repo I could confirm — flag as unverified.

**Empirical claim:** The Park-Leahey-Funk declining-disruption trend is largely an artefact of growing reference-list lengths driving citation-network density upward, which mechanically forces the CD index toward zero independent of any innovation slowdown.

**Mechanism:** Citation inflation — denser networks → more shared references → lower CD scores; not a real epistemic change.

**Implication:** **Citation, defensive.** Cite alongside Park et al. to show we are aware the disruption-decline finding is contested. Use this to soften any claim that our model "explains the decline"; instead, frame the model as explaining the *underlying mechanism Park et al. claim to measure*, regardless of whether the metric itself holds up.

---

## 3. Azoulay, Fons-Rosen & Graff Zivin (2019) — "Does science advance one funeral at a time?"

**BibTeX (already in refs.bib as `azoulay2019StarsShadow` — keep as-is)**

**Code / data:** NBER WP 21788 supplementary materials; Azoulay's MIT page lists data appendices. No public GitHub for the main analysis — admin data on NIH grants and PubMed used. Flag as **no real repo**.

**Empirical claim:** After the premature death of one of 452 elite life scientists, article flow into the deceased's subfield by *non-collaborators* rises 8.6% on average, disproportionately highly-cited and from scientists previously outside the field; flow by *collaborators* drops.

**Mechanism:** Living stars suppress outsider entry via control over funding, editorial gatekeeping, and prestige-conferred deference; their death lifts the barrier.

**Implication:** **Empirical anchor.** This is the canonical causal-identification result for paradigm gatekeeping in modern science-of-science. Our trust matrix and λ·U preference machinery should, in principle, produce an analogous "remove a high-prestige node, outsider entry rises" comparative-static. Build §5 around a death-of-a-star ablation: ablate the highest-trust node in the network and check whether outsider belief-mass on alternative θ rises. This is a *concrete, falsifiable* experimental design we currently lack.

---

## 4. Foster, Rzhetsky & Evans (2015) — "Tradition and innovation in scientists' research strategies"

**BibTeX**
```bibtex
@article{foster2015TraditionInnovation,
  author  = {Foster, Jacob G. and Rzhetsky, Andrey and Evans, James A.},
  title   = {Tradition and Innovation in Scientists' Research Strategies},
  journal = {American Sociological Review},
  volume  = {80},
  number  = {5},
  pages   = {875--908},
  year    = {2015},
  doi     = {10.1177/0003122415601618}
}
```

**Code / data:** Data drawn from MEDLINE; KnowledgeLab at UChicago hosts adjacent code (<https://knowledgelab.org>). No single canonical repo for this paper that I could confirm — flag as partially available.

**Empirical claim:** Across millions of biomedical abstracts, the modal scientific strategy is "jump tradition" (deepening known chemical relations); genuinely innovative strategies are rarer despite higher expected per-paper impact, because institutional rewards favour incrementalism.

**Mechanism:** Bourdieusian field theory — risk-averse strategies dominate because the payoff structure (publication, tenure) rewards incremental moves; "essential tension" à la Kuhn made explicit.

**Implication:** **Empirical anchor + theoretical scaffolding.** This is the empirical realisation of Bourdieu's field theory (already in our refs.bib) at scale. It justifies modelling scientific choice as utility-driven with strong incumbency bias — exactly the structure our λ·U preference machinery is designed to capture. Cite in §2 (related work) and §4 (model motivation).

---

## 5. Wu, Wang & Evans (2019) — "Large teams develop and small teams disrupt"

**BibTeX**
```bibtex
@article{wu2019SmallTeamsDisrupt,
  author  = {Wu, Lingfei and Wang, Dashun and Evans, James A.},
  title   = {Large teams develop and small teams disrupt science and technology},
  journal = {Nature},
  volume  = {566},
  number  = {7744},
  pages   = {378--382},
  year    = {2019},
  doi     = {10.1038/s41586-019-0941-9}
}
```

**Code / data:** Project page <https://lingfeiwu.github.io/smallTeams/> with code links; Harvard Dataverse hosts D-scores for ~19.5M papers. **Real repo, public.**

**Empirical claim:** Team size correlates monotonically with CD index in the opposite direction expected from productivity arguments — small teams disrupt, large teams develop and consolidate; both regimes are necessary.

**Mechanism:** Large teams optimise execution of established research programs (read: paradigm internals); small teams take riskier conceptual gambles. Compatible with our reading that high-trust dense networks lock in paradigms.

**Implication:** **Citation for §1 motivation + potential ablation target.** Predicts that in our model, *connectivity / community-density* should suppress disruption. Worth checking whether reducing connectivity in our network (smaller "teams") reproduces this — a cheap experiment.

---

## 6. Sinatra, Wang, Deville, Song & Barabási (2016) — "Quantifying the evolution of individual scientific impact" (Q-model)

**BibTeX**
```bibtex
@article{sinatra2016QModel,
  author  = {Sinatra, Roberta and Wang, Dashun and Deville, Pierre and Song, Chaoming and Barab{\'a}si, Albert-L{\'a}szl{\'o}},
  title   = {Quantifying the evolution of individual scientific impact},
  journal = {Science},
  volume  = {354},
  number  = {6312},
  pages   = {aaf5239},
  year    = {2016},
  doi     = {10.1126/science.aaf5239}
}
```

**Code / data:** Companion code on <https://github.com/Barabasi-Lab> and Sinatra's personal page. Real, public.

**Empirical claim:** Across 2,887 physicists (and replicated in other fields) a scientist's highest-impact paper is uniformly distributed within their publication sequence ("random impact rule"); a single career-stable parameter Q multiplied with paper-level luck explains the impact distribution.

**Mechanism:** Productivity × individual ability Q × random luck; *no* role for paradigm-stage timing — i.e., breakthroughs are not predictably concentrated in revolutionary periods.

**Implication:** **Headwind for Kuhnian framing; citation for §6 discussion.** This empirically undercuts the "revolutionary periods produce all the great work" picture. It's compatible with a monostable-with-slow-relaxation reading (which is what our model produces) but explicitly *incompatible* with a sharp bistable-revolution reading. Use to defend our model's structural monostability rather than apologise for it.

---

## 7. Tria, Loreto, Servedio & Strogatz (2014) — "The dynamics of correlated novelties"

**BibTeX**
```bibtex
@article{tria2014CorrelatedNovelties,
  author  = {Tria, Francesca and Loreto, Vittorio and Servedio, Vito D. P. and Strogatz, Steven H.},
  title   = {The dynamics of correlated novelties},
  journal = {Scientific Reports},
  volume  = {4},
  pages   = {5890},
  year    = {2014},
  doi     = {10.1038/srep05890}
}
```

**Code / data:** Loreto group at Sapienza hosts code. No single canonical repo I could confirm — flag as partially available.

**Empirical claim:** Heaps' and Zipf's laws for novelty rates and novelty-rank distributions hold across Wikipedia edits, social-tag emergence, text-word sequences, and music listening — i.e., novelty arrives at a sublinear rate with strong correlation ("one new thing leads to another").

**Mechanism:** Polya-urn generalisation with "adjacent possible" — each novelty seeds new tokens, making the space of available next-states expand.

**Implication:** **Reinvent-the-wheel check + §2 citation.** This is the most prominent existing mathematical model of "how new ideas enter a population" and it does *not* produce bistability — it produces a Heaps-law growth curve. We are not duplicating their work because we model the *trust-mediated adoption* of novelties, not their generation. But cite to position our contribution: we sit downstream of novelty generation, in the adoption layer.

---

## 8. Rodriguez-Sickert, Cosmelli, Claro & Fuentes (2015) — "The Underlying Social Dynamics of Paradigm Shifts"

**BibTeX**
```bibtex
@article{rodriguezSickert2015ParadigmShifts,
  author  = {Rodriguez-Sickert, Carlos and Cosmelli, Diego and Claro, Francisco and Fuentes, Miguel Angel},
  title   = {The Underlying Social Dynamics of Paradigm Shifts},
  journal = {PLOS ONE},
  volume  = {10},
  number  = {9},
  pages   = {e0138172},
  year    = {2015},
  doi     = {10.1371/journal.pone.0138172}
}
```

**Code / data:** No public repo cited in the PLOS version that I could confirm; reproducible from paper text. Flag as **no real repo**.

**Empirical claim:** Agent-based simulation: combining individual search, social influence, and random experimentation reproduces both periods of normal science (marginal change) and paradigm shifts (radical change); paradigm shifts become more likely when each scientist places a small positive weight on peers' experience.

**Mechanism:** Conformity weight × individual experimentation × landscape ruggedness; "conservative force" from current-paradigm representatives modulates shift probability.

**Implication:** **Closest prior art — read carefully.** This is the most direct competitor: an ABM that reproduces Kuhnian dynamics with social influence. Our contribution must be differentiated by (a) active-inference / variational-free-energy formal grounding (their model is heuristic), (b) explicit trust matrix as Bayesian object, and (c) the λ·U Hyland frame for paradigm-change cost. Cite in §2 as the closest existing simulation work; in §6 argue our formal grounding is the value-add.

---

## 9. Zollman (2007) — "The Communication Structure of Epistemic Communities"

**BibTeX**
```bibtex
@article{zollman2007CommunicationStructure,
  author  = {Zollman, Kevin J. S.},
  title   = {The Communication Structure of Epistemic Communities},
  journal = {Philosophy of Science},
  volume  = {74},
  number  = {5},
  pages   = {574--587},
  year    = {2007},
  doi     = {10.1086/525605}
}
```

**Code / data:** Zollman's personal page hosts NetLogo / Python implementations of the bandit-on-network model in subsequent papers. Real, partially public.

**Empirical claim:** In bandit-on-network simulations, *less* communication (sparser graphs) yields *more reliable* collective truth-tracking, because dense networks lock onto early-evidence false consensus too fast (the "Zollman effect").

**Mechanism:** Premature convergence — high connectivity collapses exploration; sparseness preserves diversity long enough for evidence to accumulate.

**Implication:** **Mechanism citation + potential anchor for §5.** Directly relevant to our trust-network story: trust amplifies the Zollman effect by making high-trust edges weight observations even more. Cite in §2 and consider as a confirmatory experiment — does our model reproduce premature consensus lock-in on the false θ at high network density?

---

## 10. Price (1976) — "A general theory of bibliometric and other cumulative advantage processes"

**BibTeX**
```bibtex
@article{price1976CumulativeAdvantage,
  author  = {{de Solla Price}, Derek J.},
  title   = {A general theory of bibliometric and other cumulative advantage processes},
  journal = {Journal of the American Society for Information Science},
  volume  = {27},
  number  = {5},
  pages   = {292--306},
  year    = {1976},
  doi     = {10.1002/asi.4630270505}
}
```

**Code / data:** Historical paper; no code.

**Empirical claim:** Citation distributions follow a "success breeds success" cumulative-advantage process, unifying Bradford, Lotka, Pareto, and Zipf laws under one preferential-attachment-style model.

**Mechanism:** Preferential attachment in citation networks; later canonised in Barabási-Albert.

**Implication:** **Background citation for §2.** Justifies the structural prior that high-trust / high-prestige actors will accumulate ever-more weight — i.e., trust-matrix dynamics in our model are predicted to lock in by preferential-attachment-type feedback. Cite when motivating why trust does not equilibrate to uniform.

---

## 11. Funk & Owen-Smith (2017) — "A Dynamic Network Measure of Technological Change"

**BibTeX**
```bibtex
@article{funkOwenSmith2017CDIndex,
  author  = {Funk, Russell J. and Owen-Smith, Jason},
  title   = {A Dynamic Network Measure of Technological Change},
  journal = {Management Science},
  volume  = {63},
  number  = {3},
  pages   = {791--817},
  year    = {2017},
  doi     = {10.1287/mnsc.2015.2366}
}
```

**Code / data:** `cdindex` package by Funk lab — see entry 1. Real, public.

**Empirical claim:** Introduces the CD index that operationalises consolidation vs. disruption at the level of individual patents/papers via citation-network displacement; method paper.

**Mechanism:** Operational — no novel mechanism; provides the metric used by Wu et al. and Park et al.

**Implication:** **Methods citation.** Cite when motivating any discussion of disruptive vs. consolidative behaviour in our agents — gives us a metric we could compute on simulated networks.

---

## Models that already do something like ours

- **Rodriguez-Sickert et al. 2015** (#8) — closest prior art, paradigm-shift ABM with social influence. Differentiate ours via active-inference formalism + trust matrix.
- **Zollman 2007** (#9) — bandit-on-network with premature consensus; not paradigm-focused but mechanistically adjacent.
- **Tria et al. 2014** (#7) — novelty generation, not paradigm adoption; complementary, not competing.
- **Sinatra et al. 2016** (#6) — Q-model is individual-level statistical, not mechanistic; complementary.

## What our model *must* engage with (top 2)

1. **Park-Leahey-Funk 2023 declining-disruption** — the headline empirical claim that "science is becoming more incremental." This is a *tailwind* for our monostable, slow-relaxation framing: if real science is in a regime of declining disruption, then a model whose attractor is a single locked-in θ is a *feature*, not a bug. Frame in §1 / §6.
2. **Azoulay et al. 2019 funeral effect** — the cleanest causal evidence that paradigm persistence is socially enforced. Our model should reproduce this as an ablation: remove the highest-trust agent → outsider belief-mass on alternative θ rises. This becomes our §5 falsifier.

---
