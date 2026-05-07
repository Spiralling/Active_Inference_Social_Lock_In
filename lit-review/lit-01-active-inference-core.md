# Literature Review: Active Inference Core — Values as Priors, Precision as Attention

**Cluster:** A (active-inference-core)
**Date:** 2026-05-04
**Scope:** Formal apparatus connecting prior preferences (C distribution / log-preference vector), precision-weighting, attention, and salience in active inference.

---

## Anchor citation corrections (read first)

Two of the four anchor citations supplied by Jonas need correction. Verified against arXiv and Springer:

1. **Torresan, Kanai & Baltieri** is **2025**, not 2024 — arXiv:2512.03293, submitted 2 Dec 2025. The thesis is unaffected; just bibliographic.
2. **"How preferences enslave attention…"** is by **Darius Parvizi-Wayne**, *not* Jakub Limanowski. Phenomenology and the Cognitive Sciences (online Sept 2024), DOI 10.1007/s11097-024-10028-5. Limanowski is heavily cited inside the paper but is not the author. Limanowski's own related contributions are on interoceptive / motor attention (e.g., Limanowski & Friston 2018, "'Seeing the Dark': Grounding Phenomenal Transparency and Opacity in Precision Estimation for Active Inference," *Frontiers in Psychology*).

The other two anchors (Feldman & Friston 2010; Parr & Friston 2017) are correct exactly as supplied.

---

## Key Findings

### 1. Precision *is* the formal substrate of attention — but the story has cracks

Feldman & Friston (2010) is the canonical formal identification: in hierarchical predictive processing, attention = optimisation of the precision (inverse variance) on prediction errors at sensory and intermediate levels. Higher precision = louder error signal = greater influence on belief updating. Variational free energy is decomposed into a complexity term plus an accuracy term weighted by precision; modulating precision is mechanistically how the brain selects which channels of evidence dominate. This framing has been ported wholesale into the discrete-state POMDP active-inference machinery (Friston et al. 2017, *A Process Theory*; Smith, Friston & Whyte 2022, *J. Math. Psych.*) where precision parameters appear as gain on likelihood (sensory) precision, transition (state) precision, and policy (γ) precision. **Caveat for the thesis:** Ransom et al. (2020, *Cognition*) argue forcefully that affect-biased attention — orienting to emotionally significant stimuli even when their precision is *low* — is not captured by precision optimisation alone. This is a direct challenge to the "values shape precision allocation" picture as it currently stands, and we should engage with it rather than ignore it.

### 2. Attention ≠ salience: the Parr & Friston distinction is load-bearing

Parr & Friston (2017, *Sci. Rep.*) and the follow-up (Parr & Friston 2019, *Curr. Opin. Psych.*, "Attention or salience?") establish a sharp formal distinction inside active inference that the field still routinely conflates:
- **Attention** = precision over *current* sensory data (gain on the likelihood mapping, A-matrix in POMDP form).
- **Salience** = expected information gain on *future* observations under candidate policies — the epistemic component of expected free energy (EFE).

Salience is therefore prospective and policy-conditional; it asks "if I take this action, how much will my posterior tighten?" This matters enormously for our thesis: **what an agent finds salient is determined by where its posterior is wide, which is determined by its priors (including preferences)**. Salience is not a property of the world; it is a property of the prior-shaped uncertainty landscape.

### 3. The C-distribution (log-preferences) does not just rank outcomes — its *shape* drives exploration

Torresan, Kanai & Baltieri (2025, arXiv:2512.03293) is the most direct formal result for our paper. They compare four parameterisations of the preference distribution P(o) over observations:
- **Hard goals** (delta-like / very peaked C)
- **Soft goals** (broad C)
- with/without **goal shaping** (intermediate sub-goal preferences interpolating toward terminal preferences)

In a grid-world navigation task, **goal shaping yields best exploitative performance but degrades learning of the transition dynamics**. Hard goals push EFE toward pragmatic value at the expense of epistemic value; soft goals do the inverse. This is exactly the formal operationalisation our paper needs: *the form of the prior determines the exploration-exploitation balance, which determines what gets sampled, which determines what gets learned, which determines what becomes salient.* The C-distribution is not an inert ranking — it is a generative force on the entire perception-action loop.

### 4. EFE decomposes preferences and information into a single currency

Friston et al. (2015, "Active inference and epistemic value," *Cognitive Neuroscience*) and the *Process Theory* paper (Friston et al. 2017) establish the canonical decomposition of expected free energy:

`G(π) ≈ –E_q[log P(o|C)]   +   E_q[H(o|s)]   –   I(s;o|π)`
`     ≈ pragmatic value (risk) + ambiguity – epistemic value (salience)`

The first term is the negative log-preference under predicted outcomes — *this is where the C-distribution enters the action-selection objective*. The third term is mutual information between latent states and predicted observations, which is salience proper. **For our thesis, the crucial structural fact is that preferences and salience appear additively in the same objective.** They are not separate cognitive systems — they are dual aspects of one variational quantity. A community that shares preferences is a community that shares an EFE landscape, which is a community that shares a salience map.

### 5. "Exogenous" salience is a residual category, not a primitive

Parvizi-Wayne (2024, *Phenom. Cogn. Sci.*) makes the most explicit philosophical argument for our framing: under active inference, the endogenous/exogenous attention dichotomy collapses. What looks like bottom-up, stimulus-driven capture is in fact precision-allocation on prediction errors generated by *deep* generative-model expectations the agent already had. A loud noise is salient because the agent has a deep prior that loud noises predict events worth modelling. **There is no precision-free signal entering the system** — the agent's deep priors (which include preferences via the EFE objective) shape what gets weighted as "surprising enough to attend to." This is the strongest formal license for the paper's central move from "values are priors" to "values determine salience."

### 6. Precision is implemented neurally, with documented neuromodulator candidates

Schwartenbeck et al. (2015, *Cerebral Cortex*) and Friston et al. (2014, "The anatomy of choice," *Phil. Trans. B*) give the empirical link: dopamine encodes precision over policies (γ), with phasic dopamine bursts looking like Bayesian updates of expected confidence. Acetylcholine is the leading candidate for sensory precision; noradrenaline for state-transition precision (Hesp et al. 2021, *Neural Computation*, "Deeply Felt Affect" — extends this with valence as expected precision of the action model). For the paper, this means precision-allocation isn't a metaphor — it has a documented neuromodulatory substrate that ties values (deep priors) to attention (precision) to felt valence in a single mechanistic story.

---

## Must-Read Shortlist

### Prior preferences in active inference agents: soft, hard, and goal shaping — Torresan, Kanai & Baltieri (2025)
- **Venue:** arXiv preprint (Araya / Kanai's group at Sony CSL / ARAYA)
- **arXiv:** 2512.03293 (submitted 2 Dec 2025)
- **Central claim:** The functional form of the preference distribution P(o|C) — hard vs soft, shaped vs unshaped — determines an active-inference agent's exploration-exploitation profile. Goal shaping helps exploitation but hurts model learning.
- **Why it matters for our thesis:** Direct formal demonstration that "preferences" are not a scalar utility but a *distribution* whose shape has measurable behavioural consequences for what gets sampled and learned. This is the closest existing piece to the values-as-priors-shape-salience claim.
- **Formal apparatus to borrow:** Their parameterisation of soft vs hard preferences (entropy of C as a continuous parameter); their grid-world setup as a minimal demonstration we could mirror with norm-following agents.

### Attention, Uncertainty, and Free-Energy — Feldman & Friston (2010)
- **Venue:** *Frontiers in Human Neuroscience*, 4:215
- **DOI:** 10.3389/fnhum.2010.00215
- **Central claim:** Attention is the brain's inference of the precision (inverse uncertainty) of prediction errors at each cortical level; biased competition and spatial attention emerge from precision optimisation.
- **Why it matters for our thesis:** This is the foundational paper that licenses "attention = precision." Without it the claim "salience = precision-weighted prior" is just metaphor.
- **Formal apparatus to borrow:** The hierarchical generative model with precision as inverse variance on each level's prediction-error signal; the formal proof that maximising model evidence implies precision optimisation.

### Working memory, attention, and salience in active inference — Parr & Friston (2017)
- **Venue:** *Scientific Reports* 7:14678
- **DOI:** 10.1038/s41598-017-15249-0
- **Central claim:** Within active inference, "attention" and "salience" are *distinct* formal quantities — attention is sensory-precision optimisation; salience is expected information gain over future samples (the epistemic term of EFE). Working memory is precision-maintenance over hidden states.
- **Why it matters for our thesis:** The paper's whole architecture rests on this distinction. Salience-as-EIG is what makes "values shape what gets attended next" formally tight rather than hand-wavy.
- **Formal apparatus to borrow:** The decomposition of EFE into pragmatic and epistemic terms; the message-passing scheme for visual sampling that explicitly computes salience as a function of policy-conditioned uncertainty.

### How preferences enslave attention: calling into question the endogenous/exogenous dichotomy from an active inference perspective — Parvizi-Wayne (2024)
- **Venue:** *Phenomenology and the Cognitive Sciences*
- **DOI:** 10.1007/s11097-024-10028-5
- **Central claim:** The classical endo/exogenous attention split is untenable under active inference; "exogenous" capture is prediction-error driven and prediction errors are computed against deep priors (including preferences). All attention is preference-modulated.
- **Why it matters for our thesis:** This is the philosophical scaffolding for the paper's central move. It does the conceptual work of dissolving the "but some salience is just bottom-up" objection.
- **Formal apparatus to borrow:** The argument is largely conceptual — but it gives us a clean target citation when we make the same move and lets us avoid re-deriving it.

### Active inference and epistemic value — Friston, Rigoli, Ognibene, Mathys, FitzGerald & Pezzulo (2015)
- **Venue:** *Cognitive Neuroscience* 6(4):187–214
- **DOI:** 10.1080/17588928.2015.1020053
- **Central claim:** Expected free energy decomposes additively into pragmatic value (expected log-preference) and epistemic value (expected information gain / Bayesian surprise / salience). Optimal policies trade these off; epistemic dominates until uncertainty is resolved, then pragmatic takes over.
- **Why it matters for our thesis:** This is the load-bearing formal result that *preferences and salience are commensurable in a single objective*. The paper's framing depends on this being a single currency, not two.
- **Formal apparatus to borrow:** The EFE decomposition itself — write it down explicitly in the paper. Also their treatment of "Bayesian surprise" as a unifying notion that subsumes earlier salience accounts (Itti & Baldi).

### A step-by-step tutorial on active inference and its application to empirical data — Smith, Friston & Whyte (2022)
- **Venue:** *Journal of Mathematical Psychology* 107:102632
- **DOI:** 10.1016/j.jmp.2021.102632
- **Central claim:** Pedagogical: how to build POMDP active-inference models with A, B, C, D matrices, and how to fit them to data.
- **Why it matters for our thesis:** This is where the C-matrix is explicitly defined and discussed with worked examples. If we want anyone to read our paper and immediately know how to operationalise "values as the C-distribution," we cite this. It also notes that "preferences in active inference are *expectations* about observations" — the unusual feature that your prior preferences also serve as a Bayesian prior, which is precisely the conceptual move the whole values-as-priors thesis rests on.
- **Formal apparatus to borrow:** Their notation conventions; their explicit discussion of the dual role of C as preference and prior.

---

## Key Researchers

- **Karl Friston** (UCL / Wellcome Centre for Human Neuroimaging) — originator of the free-energy principle and active inference; co-author on virtually every load-bearing paper here.
- **Thomas Parr** (Oxford / Nuffield) — the attention/salience distinction is largely his contribution; clearest writer in the active-inference school.
- **Lancelot Da Costa** (Imperial / UCL) — formal underpinnings of EFE, POMDP active inference, and the relationship to optimal control / RL.
- **Noor Sajid** (UCL) — *Active Inference: Demystified and Compared*; works on the relationship between EFE formulations and reward.
- **Conor Heins** (Max Planck Konstanz / Verses) — lead author of pymdp; multi-agent active inference and tractable inference schemes.
- **Ryan Smith** (Laureate Institute / Tulsa) — the *Step-by-step Tutorial*; clinical applications of active inference; the most accessible expositor.
- **Manuel Baltieri** (ARAYA / Sony CSL Tokyo) — work on prior preferences and embodied agents; co-author of the Torresan et al. paper.
- **Beren Millidge** (Oxford / Conjecture) — formal critic of EFE foundations ("Whence the Expected Free Energy?"); RL-AI bridge.
- **Alexander Tschantz** (Sussex / Verses) — active inference + RL; *Reinforcement Learning Through Active Inference*.
- **Casper Hesp** (Amsterdam / Princeton) — affect and valence as expected precision (subjective fitness).
- **Philipp Schwartenbeck** (Tübingen / DZNE) — empirical neuroscience of precision and dopamine.
- **Jakob Hohwy** (Monash) — philosophical foundations of predictive processing; precision as the substrate of attention.
- **Andy Clark** (Sussex) — *Surfing Uncertainty*; the canonical philosophical exposition of precision-weighting.
- **Anil Seth** (Sussex) — interoceptive inference; precision in emotion and self-modelling.
- **Jakub Limanowski** (TU Dresden / UCL) — interoceptive and motor attention under active inference; phenomenal transparency as precision.
- **Madeleine Ransom** (UBC Okanagan) — philosophical critic; affect-biased attention as a counter-example to pure precision-as-attention.
- **Darius Parvizi-Wayne** (KCL / VERSES) — explicit collapse of endo/exogenous attention from an active-inference standpoint.

---

## Surprising / Counterintuitive Findings

1. **The hard/soft preference distinction has a measurable cost.** Torresan et al. (2025) show that sharply-peaked preference distributions (the natural way to encode "this is what I want") *actively damage* an agent's ability to learn its environment's transition dynamics, because they suppress the epistemic term of EFE. This means that a community of strongly value-aligned agents (sharp shared C) is, by mathematical necessity, a community that under-explores. **This is a non-obvious cost of normative coherence that the paper should make explicit** — norms-as-precision-allocation is not free.

2. **Affect-biased attention is a documented exception to "attention = precision."** Ransom et al. (2020, *Cognition*) show empirically that emotionally-salient stimuli capture attention even when their precision is low. The standard active-inference response (precision is set by deep priors over emotional value, so this is still precision optimisation) only works if you grant that "values" sit upstream of precision-setting — which is exactly what we want to argue. So the apparent challenge is actually evidence *for* our framing, but only if we cite it carefully and explicitly defend the move.

3. **Attention and salience are routinely conflated even by active-inference researchers.** Parr & Friston (2019) had to write a whole *Current Opinion* paper just to make this distinction stick, and most subsequent work still uses "attention" loosely. **Implication for our paper:** we need to be ruthlessly precise from the first definition — "salience" in our paper means EIG (Bayesian surprise), not gain on current input. Mixing these will get the paper rejected by anyone in the IWAI crowd.

4. **The EFE decomposition has known foundational issues.** Millidge, Tschantz & Buckley (2021, "Whence the Expected Free Energy?", *Neural Computation* 33(2):447) point out that EFE is *not* simply free energy of the future — it requires extra moves (e.g., a "generative model of preferences") whose justification is contested. If a reviewer leans on this, we need to be ready to say: our argument depends on the *decomposition* (preferences + salience commensurable in one objective), not on EFE being uniquely derivable.

---

## Open Questions / Gaps

- **No-one has formally connected the C-distribution to social/normative priors.** All existing work treats C as exogenously specified by the modeller. Our paper's contribution opportunity is exactly here: model C as itself a quantity *learned from a community*, with its sharpness reflecting the strength of normative pressure.
- **The "precision is attention" story has not been fully reconciled with affect-biased attention.** Ransom et al.'s critique is largely unanswered. A side-claim in our paper could note this and reframe affect as a deep prior over preferences — i.e., we *use* the conflict to motivate our framing.
- **Multi-agent precision dynamics are barely touched.** There is work on multi-agent active inference (Heins, Friston, Pitliya 2024 ecosystems-of-intelligence paper; Albarracin et al. on cultural niche construction) but very little on how *shared precision allocation* propagates through a group. This is squarely our paper's territory and adjacent clusters should pick this up.
- **Empirical neuroscience of "value as prior precision" is suggestive but not closed.** Dopamine encodes policy precision (Schwartenbeck 2015), but the link from "abstract value" to "precision parameter" is more theoretical than measured. We should be careful not to overclaim mechanistic grounding.

---

## Adjacent Reads (Secondary Tier)

- **Friston et al. (2017), *Active Inference: A Process Theory*, Neural Computation 29(1):1–49** — the canonical POMDP active-inference scheme; reference for notation.
- **Da Costa, Sajid, Parr, Friston & Smith (2023), *Reward Maximization Through Discrete Active Inference*, Neural Computation 35(5):807–852** — when active inference reproduces RL-optimal behaviour and when it doesn't.
- **Sajid, Ball, Parr & Friston (2021), *Active Inference: Demystified and Compared*, Neural Computation 33(3):674–712** — the cleanest comparison with RL; useful for explaining EFE to outsiders.
- **Friston et al. (2014), *The anatomy of choice: dopamine and decision-making*, Phil. Trans. B 369:20130481** — dopamine = expected precision over policies; the empirical anchor for "precision is real."
- **Hesp, Smith, Parr, Allen, Friston & Ramstead (2021), *Deeply Felt Affect*, Neural Computation 33(2):398–446** — valence as expected precision of the action model; direct bridge from values to precision.
- **Schwartenbeck, FitzGerald, Mathys, Dolan, Kronbichler & Friston (2015), *The Dopaminergic Midbrain Encodes the Expected Certainty about Desired Outcomes*, Cerebral Cortex 25(10):3434–45** — empirical confirmation of the precision-dopamine link.
- **Ransom, Fazelpour, Markovic, Kryklywy, Thompson & Todd (2020), *Affect-biased attention and predictive processing*, Cognition 203:104370** — the strongest published critique of precision-as-attention; must engage.
- **Heins et al. (2022), *pymdp: A Python library for active inference in discrete state spaces*, JOSS 7(73):4098** — software citation if we run any simulations.
- **Tschantz, Millidge, Seth & Buckley (2020), *Reinforcement Learning Through Active Inference*, arXiv:2002.12636** — useful for the RL-bridging audience.
- **Millidge, Tschantz & Buckley (2021), *Whence the Expected Free Energy?*, Neural Computation 33(2):447–482** — foundational critique; cite if challenged on EFE.
- **Limanowski & Friston (2018), *'Seeing the Dark': Grounding Phenomenal Transparency and Opacity in Precision Estimation for Active Inference*, Frontiers in Psychology 9:643** — Limanowski's actual contribution, not the misattributed "preferences enslave attention" paper.
- **Friston, Lin, Frith, Pezzulo, Hobson & Ondobaka (2017), *Active Inference, Curiosity and Insight*, Neural Computation 29(10):2633–2683** — directly engages epistemic drives and "why explore."

---

## BibTeX Stubs

```bibtex
@article{torresan2025prior,
  title={Prior preferences in active inference agents: soft, hard, and goal shaping},
  author={Torresan, Filippo and Kanai, Ryota and Baltieri, Manuel},
  journal={arXiv preprint arXiv:2512.03293},
  year={2025},
  url={https://arxiv.org/abs/2512.03293}
}

@article{feldman2010attention,
  title={Attention, uncertainty, and free-energy},
  author={Feldman, Harriet and Friston, Karl J.},
  journal={Frontiers in Human Neuroscience},
  volume={4},
  pages={215},
  year={2010},
  doi={10.3389/fnhum.2010.00215}
}

@article{parr2017working,
  title={Working memory, attention, and salience in active inference},
  author={Parr, Thomas and Friston, Karl J.},
  journal={Scientific Reports},
  volume={7},
  number={1},
  pages={14678},
  year={2017},
  doi={10.1038/s41598-017-15249-0}
}

@article{parvizi2024preferences,
  title={How preferences enslave attention: calling into question the endogenous/exogenous dichotomy from an active inference perspective},
  author={Parvizi-Wayne, Darius},
  journal={Phenomenology and the Cognitive Sciences},
  year={2024},
  doi={10.1007/s11097-024-10028-5}
}

@article{friston2015active,
  title={Active inference and epistemic value},
  author={Friston, Karl and Rigoli, Francesco and Ognibene, Dimitri and Mathys, Christoph and FitzGerald, Thomas and Pezzulo, Giovanni},
  journal={Cognitive Neuroscience},
  volume={6},
  number={4},
  pages={187--214},
  year={2015},
  doi={10.1080/17588928.2015.1020053}
}

@article{smith2022stepbystep,
  title={A step-by-step tutorial on active inference and its application to empirical data},
  author={Smith, Ryan and Friston, Karl J. and Whyte, Christopher J.},
  journal={Journal of Mathematical Psychology},
  volume={107},
  pages={102632},
  year={2022},
  doi={10.1016/j.jmp.2021.102632}
}

@article{parr2019attention,
  title={Attention or salience?},
  author={Parr, Thomas and Friston, Karl J.},
  journal={Current Opinion in Psychology},
  volume={29},
  pages={1--5},
  year={2019},
  doi={10.1016/j.copsyc.2018.10.006}
}

@article{friston2017processtheory,
  title={Active inference: a process theory},
  author={Friston, Karl and FitzGerald, Thomas and Rigoli, Francesco and Schwartenbeck, Philipp and Pezzulo, Giovanni},
  journal={Neural Computation},
  volume={29},
  number={1},
  pages={1--49},
  year={2017},
  doi={10.1162/NECO_a_00912}
}

@article{ransom2020affect,
  title={Affect-biased attention and predictive processing},
  author={Ransom, Madeleine and Fazelpour, Sina and Markovic, Jelena and Kryklywy, James and Thompson, Evan T. and Todd, Rebecca M.},
  journal={Cognition},
  volume={203},
  pages={104370},
  year={2020},
  doi={10.1016/j.cognition.2020.104370}
}

@article{hesp2021deeply,
  title={Deeply felt affect: the emergence of valence in deep active inference},
  author={Hesp, Casper and Smith, Ryan and Parr, Thomas and Allen, Micah and Friston, Karl J. and Ramstead, Maxwell J. D.},
  journal={Neural Computation},
  volume={33},
  number={2},
  pages={398--446},
  year={2021},
  doi={10.1162/neco_a_01341}
}

@article{millidge2021whence,
  title={Whence the expected free energy?},
  author={Millidge, Beren and Tschantz, Alexander and Buckley, Christopher L.},
  journal={Neural Computation},
  volume={33},
  number={2},
  pages={447--482},
  year={2021},
  doi={10.1162/neco_a_01354}
}

@article{dacosta2023reward,
  title={Reward maximization through discrete active inference},
  author={Da Costa, Lancelot and Sajid, Noor and Parr, Thomas and Friston, Karl and Smith, Ryan},
  journal={Neural Computation},
  volume={35},
  number={5},
  pages={807--852},
  year={2023},
  doi={10.1162/neco_a_01574}
}

@article{schwartenbeck2015dopaminergic,
  title={The dopaminergic midbrain encodes the expected certainty about desired outcomes},
  author={Schwartenbeck, Philipp and FitzGerald, Thomas H. B. and Mathys, Christoph and Dolan, Ray and Friston, Karl},
  journal={Cerebral Cortex},
  volume={25},
  number={10},
  pages={3434--3445},
  year={2015},
  doi={10.1093/cercor/bhu159}
}

@article{heins2022pymdp,
  title={pymdp: A {Python} library for active inference in discrete state spaces},
  author={Heins, Conor and Millidge, Beren and Demekas, Daphne and Klein, Brennan and Friston, Karl and Couzin, Iain D. and Tschantz, Alexander},
  journal={Journal of Open Source Software},
  volume={7},
  number={73},
  pages={4098},
  year={2022},
  doi={10.21105/joss.04098}
}

@article{limanowski2018seeing,
  title={'{Seeing} the dark': grounding phenomenal transparency and opacity in precision estimation for active inference},
  author={Limanowski, Jakub and Friston, Karl},
  journal={Frontiers in Psychology},
  volume={9},
  pages={643},
  year={2018},
  doi={10.3389/fpsyg.2018.00643}
}
```
