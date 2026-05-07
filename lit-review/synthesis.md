# Synthesis: Values as Salience Priors — Cross-Cluster Integration

**Date:** 2026-05-04
**Source clusters:** lit-01 through lit-05 (active inference core, social/cultural AI, cultural evolution, cooperative AI norms, Bourdieu/power)
**Purpose:** Map the loop-structure thesis onto specific papers, identify the contribution gap, flag scoop risks, sketch IWAI 2026 paper.

---

## 1. The Loop-Structure, Mapped to the Literature

The thesis traces a self-referential loop. Below: which papers carry which step, and where the apparatus is missing.

```
community prior  ──► salience landscape ──► attention/copying ──► installed prior  ──► community prior
   (C-vector)         (precision γ)           (CRED-mediated)        (next gen)         (closes loop)
```

| Step | Formal apparatus exists in | Cultural-evolution analog in | Sociological frame in |
|------|---------------------------|------------------------------|------------------------|
| Values as priors (C) | Friston et al. 2015 (epistemic value); Torresan, Kanai & Baltieri **2025** (shape of C drives behavior) | Henrich & Gil-White 2001; Heyes 2018 (Cognitive Gadgets) | Bourdieu (habitus); Lukes 3rd face |
| Priors → precision | Hesp et al. 2021 ("Deeply Felt Affect": valence = expected precision); Parvizi-Wayne **2024** (preferences enslave attention) | — | Bourdieu (doxa); Vaisey 2009 (practical consciousness) |
| Precision → attention/salience | Feldman & Friston 2010; Parr & Friston 2017, 2019 (attention vs salience) | Chudek et al. 2012 (bystander gaze ~ precision parameter) | — |
| Attention → community-level copying | — | Henrich, Chudek & Boyd 2015 ("Big Man Mechanism"); Henrich 2009 (CREDs); Egozi & Ram 2024 (prestige-as-drift) | — |
| Community-level dynamics → norms | Albarracin et al. 2022 (epistemic communities); Heins et al. 2024 PNAS (collective surprise minimization); Constant et al. 2019 (deontic value as cued policy priors) | Boehm (leveling, gossip, attention-withdrawal) | Bourdieu (field, capital); Lukes (3rd face) |
| Norms → power asymmetry | Albarracin et al. **2025** Entropy 27(5):522 (power as α-precision) | — | Bourdieu (symbolic violence); Lukes |
| Norms → AI capture | Kulveit et al. 2025 (gradual disempowerment); Köster et al. 2022 PNAS (silly rules) | — | — |

**Empty cells are the gaps.** Two of them are exactly where this paper's contribution sits (see §4).

---

## 2. Citation Corrections (Apply Before Drafting)

The earlier conversation transcript contained two attribution errors that need fixing:

1. **Torresan, Kanai & Baltieri** — paper is **2025** (arXiv:2512.03293, December 2025), not 2024. It is the empirical demonstration that the *shape* of C (sharp vs broad) drives exploration-exploitation behavior. Closest existing analog to this paper's thesis.

2. **"Limanowski 2024 — How preferences enslave attention"** — author is **Parvizi-Wayne**, not Limanowski. Limanowski is heavily cited in it but is not the author. Limanowski's own analogous paper is **Limanowski & Friston 2018** ("'Seeing the dark': grounding phenomenal transparency and opacity in precision estimation for active inference"). Cite both Parvizi-Wayne 2024 (the argument we want) and Limanowski & Friston 2018 (the precursor).

3. **Albarracin et al. 2025 social-power paper** — has been published in **Entropy 27(5):522**, doi 10.3390/e27050522, in addition to arXiv:2501.19368. Use the journal cite. **David Hyland confirmed as third author.**

4. **Note on the social-power paper's content**: The abstract does NOT contain the explicit habitus=priors / field=precision-landscape / capital=salience-shaping mapping that the earlier conversation attributed to it. The full paper *does* formalize power as differential capacity to set α-precision in others' generative models, *does* explicitly engage Bourdieu (habitus, field, capital, doxa-via-scripts) and Foucault, but does **not** cite Lukes and barely treats symbolic violence. Operates at the **policy-precision layer**, not the **prior-preference content layer**.

---

## 3. Closest-Neighbor Papers — Must Engage, Cannot Skirt

These are the papers that are doing adjacent work; the IWAI submission will be evaluated against them. Engage explicitly:

1. **Albarracin, de Jager, Hyland & Manski 2025** (Entropy 27(5):522 / arXiv:2501.19368) — closest existing work. Power as α-precision. Bourdieu engagement. Differentiation must be made explicit: they operate at policy-precision, this paper operates at prior-preference content + adds Lukes' third face + adds CRED-mediated cultural transmission as the precision-installation mechanism. Without that table, reviewers will ask "isn't this just Albarracin 2025?"

2. **Hyland & Gavenčiak (IWAI 2024 best poster)** — partial scoop risk on the active-inference foundations side. Track down what this contained; submit only after confirming the gap remains.

3. **Oldenburg & Zhi-Xuan 2024 AAMAS** — Bayesian rule induction for normative MARL. Closest cognitively rich treatment in current MARL. Their "shared assumption" is the unmodelled prior this paper names as precision. Strongest dialogue partner for the cooperative-AI framing.

4. **Constant, Ramstead, Veissière, Friston, Kirchhoff 2019** ("Regimes of Expectations") — gives **deontic value** as a formal apparatus that can be lifted directly. Norm-following becomes Bayes-optimal action under environmentally cued policy priors. This is a *formal apparatus to borrow*, not just to cite.

5. **Heins et al. 2025** ("As One and Many", or successor) — group-level generative models are non-trivially related to individual ones. A culture is *not* the average of its members' priors. Refines the central claim and prevents naive aggregation.

6. **Köster, Hadfield-Menell, Hadfield, Leibo 2022 PNAS** ("Silly Rules") — silly rules pump up the precision of "follow norms" without committing to content. The most empirically suggestive cooperative-AI paper for a precision-substrate story.

7. **Vaisey 2009 AJS** — empirical dual-process backbone. Practical/habitus-level culture beats articulated values for behavioral prediction. Bridges the Lizardo–Vaisey–Ignatow cognitive-sociology thread to active inference; they stopped at dual-process language without crossing to predictive processing.

---

## 4. The Contribution Shape

Three converging gap-statements from the cluster reviews point at the same opening:

> **A.** "No existing work models C as *learned from a community*." (Cluster A)
> **B.** "I did not find any paper that *explicitly* models CREDs as precision-modulators." (Cluster C)
> **C.** "Albarracin 2025 operates at the policy-precision layer rather than the prior-preference-content layer; the 1-to-1 dictionary (regimes of attention = habitus, deontic cues = doxa, TTOM-asymmetry = symbolic violence) has not been published." (Cluster E)

The contribution that lives in the intersection of these gaps:

**A formal account of how prior preferences (the C-distribution) are installed in agents through community-level precision dynamics — specifically through prestige and CRED-mediated attentional inheritance — and how this dynamic constitutes norms and exposes a precision-capture vulnerability under AI mediation.**

The paper has four moves:

1. **Values are C** — recapitulate the Friston 2015 / Torresan 2025 / Hesp 2021 chain that establishes prior preferences as the formal substrate of values, with valence formally identified with expected precision (Hesp).
2. **Communities install C** — formalize CREDs and prestige cues as precision modulators on the channel from observed-other-behavior to one's own C. This is the genuinely new piece. Henrich's signaling logic gets translated into precision math; Chudek et al. 2012's empirical "bystander gaze ~ generalized precision parameter" finding becomes the empirical anchor.
3. **Norms are stabilized regions of high-precision shared C** — formal definition with stability conditions (when does the loop converge, when does it bifurcate into echo chambers à la Albarracin 2022, when does it dissolve). Engage Heins 2025 on the non-aggregation point.
4. **Power and AI capture** — Albarracin 2025's α-precision power is *one half* of the story (policy precision); this paper supplies the *other half* (preference-content priors). Lukes' third face becomes the political-theory hook neither cluster has formalized. Gradual disempowerment is the asymmetric capture of the C-installation channel.

---

## 5. Risks and Concerns

### Scoop risk
- **Hyland & Gavenčiak IWAI 2024 best poster** — read this before submitting. If they did the C-installation move, the contribution shifts.
- **Albarracin 2025** — is the closest neighbor. The paper must explicitly state the differentiation; reviewers will pattern-match aggressively.

### Framing risks
- **Ransom et al. 2020 (Cognition)** — empirical challenge to "attention = precision." Standard active-inference rebuttal (deep priors over emotional value set precision) is *exactly* this paper's thesis, so it's evidence *for*, but it must be engaged, not skirted.
- **Millidge, Tschantz & Buckley 2021** — EFE has known foundational issues. Cite defensively; don't lean on EFE more than necessary.
- **"Attention" vs "salience" conflation** — even AI insiders blur these. Maintain Parr & Friston 2017's distinction (attention = precision on current data; salience = expected information gain on future samples) from the first sentence. Non-negotiable for IWAI.

### Empirical risk
- The CRED-as-precision-modulator move has no prior formalization (Cluster C concern). The formalization must be assembled from Henrich's signaling logic + Friston-tradition precision math. Easiest unstuck-move: a small pymdp simulation extending Albarracin 2022's epistemic-communities setup with a prestige-cue mechanism that modulates precision on inter-agent observation.

---

## 6. Suggested IWAI 2026 Paper Skeleton (informed by clusters)

Mapping each section to the clusters that supply it:

| Section | Clusters drawn from | Key citations |
|---------|--------------------|--------------| 
| Intro: why current cooperative-AI norms framing is incomplete | D + E | Dafoe et al. 2020; Lukes; Köster et al. 2022 |
| Background: values as C; precision as attention | A | Friston 2015; Hesp 2021; Feldman & Friston 2010; Parr & Friston 2017; Torresan 2025; Parvizi-Wayne 2024 |
| Background: collective active inference | B | Veissière 2020; Albarracin 2022; Constant 2019; Heins 2024 |
| **Main contribution: norms as community-level precision structure (C-installation via prestige/CREDs)** | C + B + A | Henrich & Gil-White 2001; Henrich 2009 CREDs; Henrich/Chudek/Boyd 2015; Chudek 2012; Egozi & Ram 2024 + new formalism |
| Cooperative AI implications | D | Critch & Krueger 2020; Oldenburg & Zhi-Xuan 2024; Köster 2022; Kulveit 2025 |
| Demonstration | A + B | pymdp extension of Albarracin 2022; precision sweep + bifurcation behavior |
| Discussion: Bourdieu/Lukes bridge; gradual disempowerment | E + D | Bourdieu; Lukes; Albarracin 2025 (differentiation); Kulveit 2025 |

---

## 7. Open Questions for Jonas

1. **Read Albarracin 2025 (full PDF, not abstract) before drafting.** Cluster B's concern is real: the abstract doesn't show the Bourdieu mappings the earlier conversation claimed. The full paper might or might not — either way, the differentiation argument depends on knowing exactly what they did at the C-content layer vs the policy-precision layer.

2. **Track down Hyland & Gavenčiak IWAI 2024.** If they did the C-installation move, contribution-shape changes.

3. **Confirm the CRED-precision formalism is genuinely novel.** Cluster C's gap-statement is firm but worth a final sanity check by reading Vasil et al. 2020 ("we are the same kind of creature" prior) carefully — that paper might be closer to CRED-as-precision than the cluster review surfaced.

4. **Decide on simulation scope.** Minimum viable demonstration is a 20-agent pymdp simulation extending Albarracin 2022 with a prestige-cue precision-modulator. That clears the IWAI bar. Anything more ambitious is a different paper.

---

## 8. Files in this Lit Review

- [README.md](README.md) — project overview, cluster index
- [lit-01-active-inference-core.md](lit-01-active-inference-core.md) — formal AIF apparatus (Cluster A)
- [lit-02-social-cultural-active-inference.md](lit-02-social-cultural-active-inference.md) — TTOM, epistemic communities, social powers (Cluster B)
- [lit-03-cultural-evolution-prestige.md](lit-03-cultural-evolution-prestige.md) — prestige, CREDs, big-man, Heyes (Cluster C)
- [lit-04-cooperative-ai-norms.md](lit-04-cooperative-ai-norms.md) — cooperative AI, norms-as-equilibria, gradual disempowerment (Cluster D)
- [lit-05-bourdieu-power-sociology.md](lit-05-bourdieu-power-sociology.md) — Bourdieu, Lukes, cognitive sociology (Cluster E)
- **synthesis.md** — this file
