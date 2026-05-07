# Literature Review: Bourdieu, Lukes, Sociology of Power — Habitus, Fields, Preference-Shaping

**Cluster:** E (bourdieu-power-sociology)
**Date:** 2026-05-04
**Scope:** Sociological treatments of power that relate to preference-shaping, attention-shaping, and the production of social subjectivity. Anchored on Bourdieu (habitus / field / capital / doxa / symbolic violence), Lukes' three-faces-of-power (especially the third face — preference-shaping), with Foucault and Honneth as supporting voices. Special attention to recent attempts to formalize these in cognitive-science / active-inference vocabulary.

---

## Key Findings

### 1. The Bourdieu-to-active-inference bridge has already been built — partly well, partly thinly

The single most load-bearing finding for Jonas's paper is that **Albarracin, de Jager, Hyland & Manski (2025)** — "The Physics and Metaphysics of Social Powers" (arXiv:2501.19368) — is exactly the bridge paper Jonas suspects exists. It is *not* a hand-wave. The authors:

- Translate Bourdieu's **field** into a metaphor of force-relations on agents whose positions depend on capital, then reframe capital as **"possibilistic power" — the creation of the future** (i.e., differential capacity to shape state-spaces).
- Recast **habitus** as "heuristics comparable to satisficing / fast-and-frugal heuristics, formalizable through Active Inference."
- Recast **doxa** through scripts and narratives functioning as **shared generative models** that "normalize certain behavioral patterns by providing templates for 'appropriate' action, making deviations appear abnormal."
- Provide an explicit **precision-weighted policy-selection equation**: π* = arg min_π G(π) + α · precision(Q(s|o,π)), where α is the agent's capacity to shape the perceived reliability of others' beliefs. This is the formal load-bearing move — power as differential capacity to set α for others, i.e., to modulate precision-weighting in others' generative models.
- Foucault is explicitly engaged (esp. *The Subject and Power*, 1982) as the paper's framing question: a "new economy of power relations" cashed out as control of attention via active inference.
- Notable absence: **Lukes is not cited.** Bourdieu's symbolic violence is engaged only obliquely (through "coercive domination"). Distinctions between cultural / economic / social capital are collapsed.

**Implication for Jonas:** the paper does roughly what Jonas wants to do, but at the macro / power-systems level rather than the micro / values-as-priors level. Their α-on-precision is precisely the formal handle for "power = capacity to shape salience." Jonas should (a) cite this as the closest precedent, (b) differentiate his contribution by pushing one level deeper — into the **prior preference distribution P̃(o)** itself (values), where Albarracin et al. operate at the level of precision over policies rather than the construction of preferred outcomes. Lukes' third face plus a values-as-priors framing is the gap they leave open.

### 2. Cognitive sociology has been doing this since 2004 — but in dual-process terms, not active-inference terms

There is a substantial, organized, multi-decade research programme — call it the **Lizardo–Vaisey–Ignatow cluster** ("culture, cognition, and action" / "cognitive sociology") — that has explicitly tried to translate Bourdieu's habitus into cognitive-science vocabulary.

- **Lizardo (2004)** "The Cognitive Origins of Bourdieu's Habitus" *(JTSB)* traces habitus to Lévi-Strauss's structural anthropology and Piaget's developmental psychology, arguing the misreading of habitus as objectivist/reductionist is wrong; it is a generative scheme for transforming social structure.
- **Vaisey (2009)** "Motivation and Justification: A Dual-Process Model of Culture in Action" *(AJS, vol. 114)* gives the canonical dual-process operationalization: habitus ≈ System-1 / "intuitive mind" / practical consciousness; discursive justification ≈ System-2. Empirically demonstrates that practical (script-based) culture predicts behavior better than discursive moral reasoning.
- **Ignatow (2007, 2009)** uses embodied-cognition theory to argue habitus must be revised to make embodied moral schemas explicit; Bourdieu underweights the moral/affective dimension.
- **Lizardo et al. (2016)** "What Are Dual Process Models?" *(Sociological Theory)* gives the synthetic field-defining statement.

This is a real, sophisticated literature. **But it has stopped at dual-process / schema-theoretic vocabulary** and has not (mostly) made the move to predictive processing or active inference. It is positioned exactly where Jonas could draft on it without redoing the work — these sociologists have already established that habitus is cognitive, embodied, generative, and dispositional. Jonas just has to push them one step further: from "System 1 schemas" to "high-precision priors over expected observations."

### 3. There is an active-inference-to-culture pipeline running parallel that has not yet been joined to the Bourdieu pipeline

Independently of the cognitive sociologists, a sequence of papers from the **Friston / Constant / Ramstead / Veissière / Kirmayer** group has built the active-inference machinery for culture:

- **Ramstead, Veissière & Kirmayer (2016)** "Cultural Affordances: Scaffolding Local Worlds Through Shared Intentionality and Regimes of Attention" *(Frontiers in Psychology)* — introduces **regimes of attention** as patterned cultural practices that guide attentional styles. This is **almost the same construct as Bourdieu's habitus**, restated as patterned priors over what to notice.
- **Constant, Ramstead, Veissière & Friston (2019)** "Regimes of Expectations: An Active Inference Model of Social Conformity and Human Decision Making" *(Frontiers in Psychology)* — formalizes deontic value via Markov decision processes; "deontic cues" in the environment trigger high-precision policies. This is the formal model of how norms work as priors.
- **Veissière, Constant, Ramstead, Friston & Kirmayer (2020)** "Thinking Through Other Minds: A Variational Approach to Cognition and Culture" *(BBS target article)* — the canonical synthesis. TTOM = the process of inferring others' expectations and behaving accordingly. Cultural niches afford epistemic resources via shared regimes of attention.

The conceptual gap that Jonas can fill: **regimes of attention ≈ habitus, deontic cues ≈ doxa, TTOM ≈ symbolic violence (when asymmetric).** Neither group has explicitly made this mapping in print at the level Jonas is proposing — Albarracin et al. start to but only at the high-power-asymmetry end.

### 4. Lukes' third face is the right hook for "preference-shaping power" but has not been formalized

Lukes (1974/2005) articulates three dimensions of power: (1) overt decision power, (2) agenda-setting / non-decisionmaking (Bachrach & Baratz 1962), (3) preference-shaping — securing compliance "by shaping and controlling [others'] thoughts and desires." The 2005 second edition adds substantial engagement with Foucault and concedes ground on real-vs-perceived interests.

Recent commentary (Haugaard, Digeser, and the **Hayward & Lukes "Nobody to Shoot?" debate**) has tried to formalize the third face but mostly stays in conceptual / political-theory register. The Tandfonline 2015 paper "(Re)Conceptualising the third face of power: insights from Bourdieu and Foucault" (Journal of Political Power) is the most direct attempt to fuse Lukes with Bourdieu's symbolic violence and Foucault's disciplinary power — useful to cite but not formalized.

**This is where Jonas's contribution lands.** "Preference-shaping power" in Lukes-speak = "shaping the prior preference distribution P̃(o) and the precision weighting over it" in active-inference-speak. The third face is what Albarracin et al. are gesturing at without naming.

### 5. Surprising depth: there's already a habitus = generative model literature

The Drews (2021) JTSB paper "Connecting sleep, the neurocognitive memory system, and Bourdieu's habitus concept: Is sleep a generative force of the habitus?" is a small but strong signal — it argues sleep (memory consolidation, dream-simulation) is the mechanism by which habitus updates, treating habitus operationally as a memory/generative system. This is Bourdieu being read in the predictive-brain register without the Friston label.

Similarly, **Strand & Lizardo's "Hysteresis Effect: Theorizing Mismatch in Action"** uses control-system / mismatch language that is one synonym away from "prediction error."

---

## Must-Read Shortlist

### 1. Bourdieu, P. (1990). *The Logic of Practice* (R. Nice, Trans.). Stanford: Stanford University Press. [orig. *Le sens pratique*, 1980]

**Central claim:** Habitus is "systems of durable, transposable dispositions, structured structures predisposed to function as structuring structures" — i.e., generatively-acquired schemes that produce practices and perceptions appropriate to the field they were formed in. Habitus = embodied history forgotten as nature. The **logic of practice** is fast, fuzzy, time-bound, and oriented to a feel-for-the-game; it is *not* the rule-following of an analyst's reconstruction.

**Why it matters for our thesis:** The cleanest statement of habitus as a *generative scheme* — the very vocabulary cognitive scientists later use for hierarchical generative models. Bourdieu insists that practical action is not deliberate optimization but pre-reflective anticipation tuned to a field. Translate "feel for the game" → "low-free-energy trajectory through a culturally-shaped state space" and the bridge to active inference is short. Read alongside *Pascalian Meditations* (2000) for the symbolic-violence and misrecognition material.

**Formal apparatus:** None internal to Bourdieu, but the conceptual machinery (durable / transposable / structured-structuring / field / capital) maps onto: durable = parameter persistence; transposable = generalization across contexts; structured-structuring = generative model; field = state-space topology weighted by capital.

### 2. Lukes, S. (2005). *Power: A Radical View* (2nd expanded ed.). Basingstoke: Palgrave Macmillan. [1st ed. 1974]

**Central claim:** Power has three dimensions. (1) The pluralist face: A gets B to do what B otherwise wouldn't. (2) The Bachrach-Baratz face: A keeps issues off the agenda — the power of non-decision. (3) **The radical face: A shapes B's preferences themselves**, including B's perception of B's own interests. "The supreme exercise of power is to get [others] to have the desires you want them to have — that is, to secure their compliance by controlling their thoughts and desires." The 2005 second edition adds two new essays engaging Foucault and refining the concept of domination.

**Why it matters:** This is the load-bearing political-theory frame for our thesis. Translating Lukes' third face into active-inference vocabulary — preference-shaping = shaping prior preferences P̃(o) and precision over them — is the move our paper makes. Editions matter: cite *both* 1974 (original radical view) and 2005 (post-Foucault refinement and "real interests" rejoinder).

**Formal apparatus:** None (political theory). But the third face is the precise sociological construct active inference can render formal.

### 3. Albarracin, M., de Jager, S., Hyland, D., & Manski, S. G. (2025). The Physics and Metaphysics of Social Powers: Bridging Cognitive Processing and Social Dynamics. arXiv:2501.19368.

**Central claim:** Social power is a function of (a) capacity to attract others' attention to one's preferred policies, and (b) capacity to process information effectively. Power flows through scripts and narratives operating as shared generative models that direct agents' precision-weighting. Asymmetric belief-synchronization (low-power agents update toward high-power agents more than vice-versa) is the formal expression of power.

**Why it matters:** This is the existing precedent for *exactly* the kind of move Jonas's paper proposes, but operating at the macro / capital-and-attention level rather than the values-as-priors level. Differentiation point: their α modulates precision over policies; Jonas's framing pushes one stratum deeper, into what gets installed as the prior preference distribution itself.

**Formal apparatus:** Variational free energy F = D_KL[q(s)||p(s|o)] − ln p(o); expected free energy with epistemic + pragmatic value; **empowerment** as I(A_t,n; s_t+n); **belief synchronization** as KL(Q_i || Q_j); **deontic-modulated policy selection** P(π|o) ∝ exp(−G(π) + D(π,o)); **precision-weighted policy** π* = arg min_π G(π) + α · precision(Q(s|o,π)).

### 4. Vaisey, S. (2009). Motivation and Justification: A Dual-Process Model of Culture in Action. *American Journal of Sociology*, 114(6), 1675–1715.

**Central claim:** Culture operates in two registers — a fast, automatic, motivational "practical consciousness" register (≈ habitus) and a slow, reflective, post-hoc justificatory register. Empirically, the motivational register predicts behavior; discursive moral reasoning predicts what people *say* about behavior. Demonstrates this with NSYR panel data.

**Why it matters:** Provides the empirical backbone for "values function pre-reflectively" — exactly Jonas's claim. Vaisey shows that articulated values are post-hoc rationalizations and that the operationally efficacious "values" are the fast, automatic dispositions Bourdieu calls habitus. This is empirical support for treating values as priors rather than as deliberative inputs.

**Formal apparatus:** Operationalizes practical-consciousness culture via forced-choice "moral schemas" panel measure; uses regression to show choice-of-schema at t1 predicts behavior at t2 controlling for articulated values.

### 5. Veissière, S. P. L., Constant, A., Ramstead, M. J. D., Friston, K. J., & Kirmayer, L. J. (2020). Thinking through other minds: A variational approach to cognition and culture. *Behavioral and Brain Sciences*, 43, e90.

**Central claim:** Cultural learning is the active-inference process of inferring others' expectations and conforming to them ("Thinking Through Other Minds" — TTOM). Cultural niches scaffold cognition by providing **cultural affordances** and **regimes of attention** that selectively pattern attention and behavior. Norms operate via **deontic cues** that trigger high-precision policies.

**Why it matters:** The canonical active-inference treatment of culture and norms. Their "regimes of attention" is the constructive twin of Bourdieu's habitus; their "deontic cues" is the constructive twin of doxa; their TTOM is the constructive twin of symbolic conformity. Pulling these into explicit dialogue with Bourdieu (which they do not fully do) is precisely where the paper's contribution sits.

**Formal apparatus:** Variational free-energy minimization across hierarchical generative models; deontic value as a quasi-prior in policy selection; cultural niche construction as a multi-agent KL-minimization process.

### 6. Constant, A., Ramstead, M. J. D., Veissière, S. P. L., & Friston, K. J. (2019). Regimes of Expectations: An Active Inference Model of Social Conformity and Human Decision Making. *Frontiers in Psychology*, 10, 679.

**Central claim:** Social conformity emerges from the joint action of **deontic cues** (e.g., red traffic lights) and **regimes of expectation** encoded in neural hierarchies, phenotypes, and patterned sociocultural practices. The deontic value of an action is "the prosocial increment of expected free energy" attributable to its norm-conformity.

**Why it matters:** The crispest formal model of how social norms operate as priors. Read in conjunction with Veissière et al. 2020 (TTOM) and Ramstead et al. 2016 (Cultural Affordances) as the trilogy. Together they give the active-inference formalism for "norms = stabilized regions of shared high-precision prior preference" — Jonas's Section 4 thesis, near-fully spelled out.

**Formal apparatus:** Active-inference / Markov-decision-process formulation; deontic value formalized as a precision-weighted contribution to expected free energy in policy selection.

---

## Key Researchers

| Researcher | Affiliation | Relevance |
|---|---|---|
| **Pierre Bourdieu** (d. 2002) | Collège de France | Habitus, field, capital, doxa, symbolic violence — primary theorist |
| **Loïc Wacquant** | UC Berkeley (Sociology) | Bourdieu's collaborator; carnal sociology; "Putting Habitus in its Place" (2014); definitive contemporary expositor of habitus |
| **Steven Lukes** | NYU (Sociology) | *Power: A Radical View* (1974/2005); third-face-of-power |
| **Omar Lizardo** | UCLA (Sociology) | Cognitive origins of habitus (2004); dual-process models in sociology (2016, 2021); maintains the *culturecog* blog as community hub |
| **Stephen Vaisey** | Duke (Sociology) | Dual-process model of culture in action (2009 AJS); panel-data measurement of habitus |
| **Gabriel Ignatow** | Univ. of North Texas (Sociology) | Embodied cognition + habitus (2007, 2009); co-editor of *Oxford Handbook of Cognitive Sociology* |
| **Michael Strand** | Brandeis (Sociology) | Hysteresis (with Lizardo) — habitus / field mismatch as control-system disequilibrium |
| **Mahault Albarracin** | VERSES.ai / UQAM | Lead author on Albarracin et al. 2025 — primary active-inference-to-power bridge |
| **Sonia de Jager** | Erasmus / Noise Research Union | Co-author Albarracin et al. 2025 |
| **Maxwell Ramstead** | VERSES.ai / McGill | Cultural affordances; TTOM; bridges between active inference and 4E cognition |
| **Axel Constant** | Univ. of Sussex | Regimes of Expectations; deontic value formalization |
| **Samuel Veissière** | McGill (Psychiatry / Anthropology) | TTOM; cultural-cognitive niches |
| **Karl Friston** | UCL / VERSES.ai | Free-energy principle; underwrites all the active-inference-to-culture work |
| **Laurence Kirmayer** | McGill (Social & Transcultural Psychiatry) | Cultural psychiatry → active inference bridge |
| **Henning Drews** | Christian-Albrechts-Universität zu Kiel | Sleep / memory / habitus (2021 JTSB) — under-the-radar predictive-brain reading of Bourdieu |
| **Axel Honneth** | Columbia / Frankfurt School | Recognition theory; subject formation through normative perspective-taking |
| **Michel Foucault** (d. 1984) | Collège de France | Power/knowledge; disciplinary subject formation; "The Subject and Power" (1982) is the most-cited piece in active-inference-power papers |

---

## Surprising / Counterintuitive Findings

1. **The bridge has already been built — and it explicitly uses precision-weighting.** Albarracin et al. 2025 already formalize "social power = differential capacity to modulate the α-precision parameter in others' generative models." That is precisely the move Jonas is planning to make. This is both a gift (existing precedent to cite) and a flag (must clearly differentiate). The differentiation lever: they operate at the level of policies and capital-as-power-pooling; Jonas's *values-as-priors* framing operates one stratum deeper, on prior preference distributions themselves. Their paper does not engage Lukes; Jonas's emphasis on the third face plugs that gap.

2. **Cognitive sociologists got there 15 years earlier — but stopped at dual-process.** Lizardo (2004), Vaisey (2009), Ignatow (2007/2009) had already worked out that habitus is a generative cognitive scheme producing pre-reflective dispositions, and Vaisey 2009 has empirical panel data showing that practical (habitus-like) culture predicts behavior better than articulated values. They never crossed into Friston-style predictive processing. This means the most rigorous *empirical* support for the values-as-priors thesis is sitting in the *American Journal of Sociology* under a "dual process" label that the active-inference community hasn't read.

3. **The Friston-pipeline papers (Cultural Affordances → Regimes of Expectations → TTOM) are functionally a Bourdieu translation that doesn't cite Bourdieu enough.** Ramstead, Constant, Veissière, Kirmayer have spent 2016–2020 building "regimes of attention," "deontic cues," "cultural niches as scaffolds for shared prior expectations" — and the Bourdieu citation density across these papers is light. The Bourdieu→active-inference mapping is essentially open source: the constructs are isomorphic; nobody has been the one to publish the 1-to-1 dictionary. Jonas can do that in a single table and it will be a contribution.

4. **Sleep is a mechanism for habitus updating** (Drews 2021 JTSB). Treating sleep / memory consolidation as the substrate of habitus formation puts habitus squarely in the predictive-brain literature without invoking Friston — a useful complementary framing for skeptical sociology readers.

---

## Open Questions / Gaps

- **Lukes' third face has never been formalized in active-inference terms.** Albarracin et al. brush against it without naming it. Vaisey demonstrates it empirically without naming it. This is the single clearest open theoretical move.
- **The cognitive-sociology cluster (Lizardo, Vaisey, Ignatow) and the active-inference-culture cluster (Ramstead, Constant, Veissière) do not cite each other systematically.** Bridging them is an opportunity.
- **Symbolic violence has not been formalized.** This is the single hardest concept to render precisely in active-inference terms — it requires modeling how the *shaping* of priors itself is hidden from the agent (the misrecognition condition). Hypothesis: symbolic violence = inducing high-precision priors whose *origin* the agent's generative model has no representation of.
- **Power as differential precision-control** vs. **power as prior-shaping** is not yet disambiguated in the literature. These are two distinct mechanisms (one operates on second-order weighting, the other on first-order content) and our paper should be explicit about which it is claiming. Likely: power operates on both; values-as-priors emphasizes the latter.
- **Wacquant's carnal sociology** has not been linked to interoceptive / embodied predictive processing despite obvious parallels (Friston-Seth interoceptive inference). Adjacent gap.

---

## Adjacent Reads (Secondary Tier)

- **Bourdieu, P. (1977).** *Outline of a Theory of Practice* (R. Nice, Trans.). Cambridge: Cambridge University Press. [orig. 1972] — the first systematic articulation of habitus and doxa.
- **Bourdieu, P. (1984).** *Distinction: A Social Critique of the Judgement of Taste* (R. Nice, Trans.). Harvard University Press. [orig. 1979] — the empirical-quantitative deployment of the habitus / capital framework. Salience-distribution by class.
- **Bourdieu, P. (2000).** *Pascalian Meditations* (R. Nice, Trans.). Stanford University Press. [orig. 1997] — most compact statement of symbolic power and misrecognition.
- **Bachrach, P. & Baratz, M. S. (1962).** Two Faces of Power. *American Political Science Review*, 56(4), 947–952. — second face / agenda-setting; necessary background for Lukes.
- **Hayward, R. C. (Re)Conceptualising the third face of power: insights from Bourdieu and Foucault. *Journal of Political Power*, 8(3), 2015, 359–376. — explicit Bourdieu-Foucault-Lukes synthesis attempt; pre-cognitive-science.
- **Lizardo, O., Mowry, R., Sepulvado, B., Stoltz, D. S., Taylor, M. A., Van Ness, J., & Wood, M. (2016).** What Are Dual Process Models? Implications for Cultural Analysis in Sociology. *Sociological Theory*, 34(4), 287–310. — field-defining synthesis; useful for citing the dual-process tradition.
- **Lizardo, O. (2021).** Habit and the explanation of action. *Journal for the Theory of Social Behaviour*, 51(3), 391–411. — habit-as-cause; reconstructs the metaphysics of dispositional explanation.
- **Lizardo, O. (2021).** Culture, Cognition, and Internalization. *Sociological Forum*, 36(S1), 1177–1206.
- **Strand, M. & Lizardo, O. The Hysteresis Effect: Theorizing Mismatch in Action. *Journal for the Theory of Social Behaviour*, 47(2), 2017, 164–194. — control-system framing of habitus / field disjunction; close to prediction-error language.
- **Wacquant, L. (2014).** Homines in Extremis: What Fighting Scholars Teach Us about Habitus. *Body & Society*, 20(2), 3–17.
- **Wacquant, L. (2016).** A Concise Genealogy and Anatomy of Habitus. *The Sociological Review*, 64(1), 64–72. — definitive contemporary statement.
- **Drews, H. J. (2021).** Connecting sleep, the neurocognitive memory system, and Bourdieu's habitus concept. *Journal for the Theory of Social Behaviour*, 51(2), 217–233.
- **Ramstead, M. J. D., Veissière, S. P. L., & Kirmayer, L. J. (2016).** Cultural Affordances: Scaffolding Local Worlds Through Shared Intentionality and Regimes of Attention. *Frontiers in Psychology*, 7, 1090.
- **Foucault, M. (1982).** The Subject and Power. *Critical Inquiry*, 8(4), 777–795. — most-cited single Foucault essay in active-inference-power literature.
- **Foucault, M. (1977).** *Discipline and Punish: The Birth of the Prison* (A. Sheridan, Trans.). Pantheon. [orig. 1975] — disciplinary attention and subject formation.
- **Honneth, A. (1995).** *The Struggle for Recognition: The Moral Grammar of Social Conflicts* (J. Anderson, Trans.). Cambridge: Polity. [orig. 1992] — recognition as constitutive of subjectivity; cognitive-and-affective intertwining.
- **Colombo, M. (2014).** Maladaptive social norms, cultural progress, and the free-energy principle. *Behavioral and Brain Sciences*, 37(4), 411–412. — sharp critique of the free-energy account of norm conformity; useful counterweight.

---

## BibTeX Stubs

```bibtex
@book{bourdieu1977outline,
  author    = {Bourdieu, Pierre},
  title     = {Outline of a Theory of Practice},
  translator = {Nice, Richard},
  year      = {1977},
  publisher = {Cambridge University Press},
  address   = {Cambridge},
  note      = {Original work published 1972 as Esquisse d'une theorie de la pratique}
}

@book{bourdieu1984distinction,
  author    = {Bourdieu, Pierre},
  title     = {Distinction: A Social Critique of the Judgement of Taste},
  translator = {Nice, Richard},
  year      = {1984},
  publisher = {Harvard University Press},
  address   = {Cambridge, MA},
  note      = {Original work published 1979}
}

@book{bourdieu1990logic,
  author    = {Bourdieu, Pierre},
  title     = {The Logic of Practice},
  translator = {Nice, Richard},
  year      = {1990},
  publisher = {Stanford University Press},
  address   = {Stanford, CA},
  note      = {Original work published 1980 as Le sens pratique}
}

@book{bourdieu2000pascalian,
  author    = {Bourdieu, Pierre},
  title     = {Pascalian Meditations},
  translator = {Nice, Richard},
  year      = {2000},
  publisher = {Stanford University Press},
  address   = {Stanford, CA},
  note      = {Original work published 1997 as Meditations pascaliennes}
}

@book{lukes1974power,
  author    = {Lukes, Steven},
  title     = {Power: A Radical View},
  year      = {1974},
  publisher = {Macmillan},
  address   = {London},
  edition   = {1st}
}

@book{lukes2005power,
  author    = {Lukes, Steven},
  title     = {Power: A Radical View},
  year      = {2005},
  publisher = {Palgrave Macmillan},
  address   = {Basingstoke},
  edition   = {2nd, expanded},
  isbn      = {0-333-42092-6}
}

@article{bachrach1962two,
  author    = {Bachrach, Peter and Baratz, Morton S.},
  title     = {Two Faces of Power},
  journal   = {The American Political Science Review},
  volume    = {56},
  number    = {4},
  pages     = {947--952},
  year      = {1962},
  doi       = {10.2307/1952796}
}

@article{albarracin2025physics,
  author    = {Albarracin, Mahault and de Jager, Sonia and Hyland, David and Manski, Sarah Grace},
  title     = {The Physics and Metaphysics of Social Powers: Bridging Cognitive Processing and Social Dynamics, a New Perspective on Power through Active Inference},
  journal   = {arXiv preprint},
  year      = {2025},
  eprint    = {2501.19368},
  archivePrefix = {arXiv},
  primaryClass = {cs.CY}
}

@article{lizardo2004cognitive,
  author    = {Lizardo, Omar},
  title     = {The Cognitive Origins of Bourdieu's Habitus},
  journal   = {Journal for the Theory of Social Behaviour},
  volume    = {34},
  number    = {4},
  pages     = {375--401},
  year      = {2004},
  doi       = {10.1111/j.1468-5914.2004.00255.x}
}

@article{vaisey2009motivation,
  author    = {Vaisey, Stephen},
  title     = {Motivation and Justification: A Dual-Process Model of Culture in Action},
  journal   = {American Journal of Sociology},
  volume    = {114},
  number    = {6},
  pages     = {1675--1715},
  year      = {2009},
  doi       = {10.1086/597179}
}

@article{ignatow2007theories,
  author    = {Ignatow, Gabriel},
  title     = {Theories of Embodied Knowledge: New Directions for Cultural and Cognitive Sociology?},
  journal   = {Journal for the Theory of Social Behaviour},
  volume    = {37},
  number    = {2},
  pages     = {115--135},
  year      = {2007},
  doi       = {10.1111/j.1468-5914.2007.00328.x}
}

@article{ignatow2009why,
  author    = {Ignatow, Gabriel},
  title     = {Why the Sociology of Morality Needs Bourdieu's Habitus},
  journal   = {Sociological Inquiry},
  volume    = {79},
  number    = {1},
  pages     = {98--114},
  year      = {2009},
  doi       = {10.1111/j.1475-682X.2008.00273.x}
}

@article{lizardo2016what,
  author    = {Lizardo, Omar and Mowry, Robert and Sepulvado, Brandon and Stoltz, Dustin S. and Taylor, Marshall A. and Van Ness, Justin and Wood, Michael},
  title     = {What Are Dual Process Models? Implications for Cultural Analysis in Sociology},
  journal   = {Sociological Theory},
  volume    = {34},
  number    = {4},
  pages     = {287--310},
  year      = {2016},
  doi       = {10.1177/0735275116675900}
}

@article{lizardo2021habit,
  author    = {Lizardo, Omar},
  title     = {Habit and the Explanation of Action},
  journal   = {Journal for the Theory of Social Behaviour},
  volume    = {51},
  number    = {3},
  pages     = {391--411},
  year      = {2021},
  doi       = {10.1111/jtsb.12273}
}

@article{strand2017hysteresis,
  author    = {Strand, Michael and Lizardo, Omar},
  title     = {The Hysteresis Effect: Theorizing Mismatch in Action},
  journal   = {Journal for the Theory of Social Behaviour},
  volume    = {47},
  number    = {2},
  pages     = {164--194},
  year      = {2017},
  doi       = {10.1111/jtsb.12117}
}

@article{wacquant2014homines,
  author    = {Wacquant, Loic},
  title     = {Homines in Extremis: What Fighting Scholars Teach Us about Habitus},
  journal   = {Body \& Society},
  volume    = {20},
  number    = {2},
  pages     = {3--17},
  year      = {2014},
  doi       = {10.1177/1357034X13501348}
}

@article{wacquant2016concise,
  author    = {Wacquant, Loic},
  title     = {A Concise Genealogy and Anatomy of Habitus},
  journal   = {The Sociological Review},
  volume    = {64},
  number    = {1},
  pages     = {64--72},
  year      = {2016},
  doi       = {10.1111/1467-954X.12356}
}

@article{drews2021connecting,
  author    = {Drews, Henning Johannes},
  title     = {Connecting Sleep, the Neurocognitive Memory System, and Bourdieu's Habitus Concept: Is Sleep a Generative Force of the Habitus?},
  journal   = {Journal for the Theory of Social Behaviour},
  volume    = {51},
  number    = {2},
  pages     = {217--233},
  year      = {2021},
  doi       = {10.1111/jtsb.12268}
}

@article{ramstead2016cultural,
  author    = {Ramstead, Maxwell J. D. and Veissi{\`e}re, Samuel P. L. and Kirmayer, Laurence J.},
  title     = {Cultural Affordances: Scaffolding Local Worlds Through Shared Intentionality and Regimes of Attention},
  journal   = {Frontiers in Psychology},
  volume    = {7},
  pages     = {1090},
  year      = {2016},
  doi       = {10.3389/fpsyg.2016.01090}
}

@article{constant2019regimes,
  author    = {Constant, Axel and Ramstead, Maxwell J. D. and Veissi{\`e}re, Samuel P. L. and Friston, Karl J.},
  title     = {Regimes of Expectations: An Active Inference Model of Social Conformity and Human Decision Making},
  journal   = {Frontiers in Psychology},
  volume    = {10},
  pages     = {679},
  year      = {2019},
  doi       = {10.3389/fpsyg.2019.00679}
}

@article{veissiere2020thinking,
  author    = {Veissi{\`e}re, Samuel P. L. and Constant, Axel and Ramstead, Maxwell J. D. and Friston, Karl J. and Kirmayer, Laurence J.},
  title     = {Thinking Through Other Minds: A Variational Approach to Cognition and Culture},
  journal   = {Behavioral and Brain Sciences},
  volume    = {43},
  pages     = {e90},
  year      = {2020},
  doi       = {10.1017/S0140525X19001213}
}

@article{foucault1982subject,
  author    = {Foucault, Michel},
  title     = {The Subject and Power},
  journal   = {Critical Inquiry},
  volume    = {8},
  number    = {4},
  pages     = {777--795},
  year      = {1982},
  doi       = {10.1086/448181}
}

@book{foucault1977discipline,
  author    = {Foucault, Michel},
  title     = {Discipline and Punish: The Birth of the Prison},
  translator = {Sheridan, Alan},
  year      = {1977},
  publisher = {Pantheon},
  address   = {New York},
  note      = {Original work published 1975 as Surveiller et punir}
}

@book{honneth1995struggle,
  author    = {Honneth, Axel},
  title     = {The Struggle for Recognition: The Moral Grammar of Social Conflicts},
  translator = {Anderson, Joel},
  year      = {1995},
  publisher = {Polity},
  address   = {Cambridge},
  note      = {Original work published 1992 as Kampf um Anerkennung}
}

@article{hayward2015reconceptualising,
  author    = {Hayward, Clarissa Rile},
  title     = {(Re)Conceptualising the Third Face of Power: Insights from Bourdieu and Foucault},
  journal   = {Journal of Political Power},
  volume    = {8},
  number    = {3},
  pages     = {359--376},
  year      = {2015},
  doi       = {10.1080/2158379X.2015.1095845},
  note      = {Author attribution to be verified}
}

@article{colombo2014maladaptive,
  author    = {Colombo, Matteo},
  title     = {Maladaptive Social Norms, Cultural Progress, and the Free-Energy Principle},
  journal   = {Behavioral and Brain Sciences},
  volume    = {37},
  number    = {4},
  pages     = {411--412},
  year      = {2014},
  doi       = {10.1017/S0140525X13003075}
}
```

---

**Note on verification:** All anchor citations confirmed via primary sources or authoritative secondary sources. The Hayward 2015 *Journal of Political Power* article author attribution should be double-checked at proof time (the ResearchGate page returned 403; the most likely author is Clarissa Rile Hayward, given her separate book *De-Facing Power* on the same theme, but verify against the journal page directly). Editions on Lukes (1974 vs 2005) confirmed: cite both for completeness; the 2005 Palgrave second edition is the version most papers reference today and contains the post-Foucault refinements.
