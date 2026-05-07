# Literature Review: Cooperative AI and Multi-Agent Norms — Norms-as-Equilibria, Norms-as-Precision

**Cluster:** D (cooperative-ai-norms)
**Date:** 2026-05-04
**Scope:** Cooperative AI as a research field, multi-agent norms in AI (norms-as-equilibria, Lewis conventions, normative MARL), multi-agent active inference, gradual disempowerment, AI participation in human normative life.

---

## Key Findings

The cooperative-AI literature treats norms in three nested but distinct registers, and the paper's "norms-as-precision" move lands in a real gap between them.

1. **Norms-as-equilibria (the dominant frame).** Following Lewis, Schelling, and the game-theoretic tradition imported into MARL, "norm" is operationalised as a self-enforcing equilibrium of behavior under coordination/social-dilemma pressure. The agenda paper (Dafoe et al. 2020) and the canonical normative-MARL papers (Köster, Hadfield-Menell et al. 2022; Vinitsky, Köster et al. 2023; Oldenburg & Zhi-Xuan 2024) all sit here. Norms exist when sanctioning behavior plus compliance behavior jointly persist at population scale. There is essentially no story about what a norm *is* in the head of a single agent beyond a learned policy gradient — the cognitive substrate is a black box.

2. **Norms-as-rules (the Bayesian-cognitive frame).** Oldenburg & Zhi-Xuan (2024) and Hadfield-Menell, Andrus, Hadfield (2019) push toward an explicit symbolic representation: a norm is a probabilistically inferred *rule* (obligative or prohibitive) that agents Bayesianly induce from observed compliance/sanction data. This is closer to the paper's instincts but still treats the rule as discrete content rather than as a precision-weighting structure over policies and observations.

3. **Norms-as-shared-generative-models (the active-inference frame, currently underspecified).** Heins et al. (2024 PNAS), Maisto, Donnarumma & Pezzulo (2022/2024 IEEE TSMCS), Friston et al. (2024 "Shared Protentions"), and Hyland, Gavenčiak et al. (IWAI 2024) are explicitly building multi-agent active inference but have *not yet*, to my reading, articulated norms as stabilized regions of shared *precision* (i.e., shared attention/salience allocation) on top of shared priors. They model shared *goals* (preferred-state distributions C) and shared *generative models* (A,B), but the precision-weighting (γ on policies, π over observations) is treated as agent-internal rather than as the locus of normativity.

**This is the gap the paper fills.** The contribution "norms-as-precision" appears genuinely available: the field has shared priors (some), shared models (Friston's protentions, Maisto's interactive inference), and shared sanctioning behavior (Leibo lineage), but no one has published the synthesis that says *the cognitive phenomenology of norms — feeling of facticity, surprise on violation, stickiness — is precisely the signature of high-precision shared priors over policy distributions, and that is what stabilizes equilibria from below*. The closest move is Hadfield-Menell's "legible normativity" / silly-rules argument, which is functionally adjacent (silly rules increase the *legibility* of the normative substrate, i.e., its precision-shaping power) but is framed in cultural-evolution rather than active-inference terms.

The gradual disempowerment paper (Kulveit et al. 2025) is load-bearing for the *stakes* argument: if norms are only equilibria, then AI agents that sit at equilibrium look fine; if norms are precision-allocations co-constituted by participants, then AI integration that fails to participate at the precision level can quietly hollow out the substrate even while the equilibrium looks intact. This is the paper's strongest external lever.

---

## Must-Read Shortlist

### 1. Dafoe, Hughes, Bachrach, Collins, McKee, Leibo, Larson, Graepel (2020) — "Open Problems in Cooperative AI"
- **Central claim.** Cooperative AI should be constituted as a field studying *cooperative intelligence*: the capacity of agents (AI–AI, AI–human, human–human-via-AI) to find joint mutually-beneficial courses of action. Cooperation rests on four pillars: understanding, communication, commitment, and institutions/norms.
- **Why it matters.** This is the agenda paper the IWAI 2026 / Cooperative AI Workshop submissions will be measured against. Its treatment of "institutions/norms" is the equilibrium-and-mechanism-design frame the paper is challenging — norms are the "social structure to promote cooperation, decentralized and informal (norms) or centralized and formal (legal systems)." Notice the abstraction-level: norms are exogenous coordination devices, not cognitive structures.
- **Formal apparatus.** None proper; agenda paper. References Schelling, Ostrom, Lewis, Axelrod. Multi-agent RL and mechanism design as primary tools.
- **arXiv:** 2012.08630.

### 2. Critch & Krueger (2020) — "AI Research Considerations for Human Existential Safety (ARCHES)"
- **Central claim.** Reframes safety research as a *multi-principal/multi-agent* problem rather than a single-principal alignment problem. Introduces "prepotence" as a property of AI systems whose collective action humanity cannot reverse. Identifies delegation, comprehension, and instruction problems across the (#humans × #AIs) Cartesian space.
- **Why it matters.** The Berkeley/CHAI lineage that gradual disempowerment extends. ARCHES is where the multi-agent governance frame becomes part of safety thinking. It explicitly flags that single-agent value-alignment is the wrong frame for the world we will be in.
- **Formal apparatus.** Taxonomy of (1×1, 1×N, N×1, N×N) human-AI configurations; scenario-based evaluation of research directions.
- **arXiv:** 2006.04948.

### 3. Heins, Millidge, da Costa, Mann, Friston, Couzin (2024) — "Collective behavior from surprise minimization" (PNAS)
- **Central claim.** Schooling/flocking phenomena (cohesion, milling, directed motion, response to predators) emerge naturally when each agent runs active inference with social observation channels — *no explicit social forces or alignment rules required*. Beliefs about uncertainty (precision over sensory channels) determine collective sensitivity and decision accuracy.
- **Why it matters.** This is the bridge paper from Cluster B and the existence proof that *precision parameters do collective work* in multi-agent active inference. The paper can lean on it: if precision over sensory channels already shapes collective phenomena in this rudimentary case, precision over *policy priors* (i.e., values) shaping shared norms is the natural extension. This is Heins's own implicit research program.
- **Formal apparatus.** Discrete-state pymdp models; agents share an A-matrix encoding "where are my neighbors" but maintain individual beliefs; γ-precision on policy selection is the key parameter.
- **arXiv:** 2307.14804; **DOI:** 10.1073/pnas.2320239121.

### 4. Köster, Hadfield-Menell, Everett, Weidinger, Hadfield, Leibo (2022) — "Spurious normativity enhances learning of compliance and enforcement behavior in artificial agents" (PNAS)
- **Central claim.** Adding *arbitrary* "silly rules" (taboos that impose punishment for behaviors with no welfare cost) into a multi-agent foraging environment *improves* compliance and enforcement learning on the welfare-relevant taboo. Silly rules increase the legibility of the rule-following substrate and hence the speed at which agents acquire normative competence at all.
- **Why it matters.** This is the closest thing in MARL to a result that "norms have a cognitive substrate that is irreducible to their welfare function." The paper can argue: silly rules work because they sharpen the *precision* of the agents' priors over the policy distribution "follow taboos" — they act as additional Bayesian evidence for the existence of the normative regime, not for any specific rule. This is a textbook precision-shaping story dressed in MARL clothing.
- **Formal apparatus.** Deep MARL in a melting-pot-style foraging environment; tracks compliance and third-party punishment as separate learnable behaviors.
- **DOI:** 10.1073/pnas.2106028118.

### 5. Oldenburg & Zhi-Xuan (2024) — "Learning and Sustaining Shared Normative Systems via Bayesian Rule Induction in Markov Games"
- **Central claim.** Agents who *assume* there is a shared norm system and Bayesianly induce its content from observed compliance and sanction can rapidly converge to common knowledge of the norms even when starting from divergent priors. The shared assumption itself is the engine of stability: it bootstraps common knowledge, which in turn enables newcomers to learn fast, which sustains the norm.
- **Why it matters.** This is the most cognitively rich treatment of norms in current cooperative AI and the most direct dialogue partner. Its weakness from the paper's perspective: norms are still discrete propositional rules, and the "shared assumption" is unmodelled prior content rather than precision allocation. The paper's move is: *replace "shared rule" with "shared high-precision prior over policy distributions," and you get the same convergence dynamics with the cognitive phenomenology built in.*
- **Formal apparatus.** Bayesian rule induction over obligative/prohibitive predicates in Markov games; AAMAS 2024.
- **arXiv:** 2402.13399.

### 6. Kulveit, Douglas, Ammann, Turan, Krueger, Duvenaud (2025) — "Gradual Disempowerment: Systemic Existential Risks from Incremental AI Development"
- **Central claim.** Even without explicit AI takeover, incremental AI integration into economy, culture, and state shifts incentive structures and erodes the implicit alignment that comes from human participation being load-bearing. Disempowerment is endogenous to integration, not a deviation from it. Loss may be effectively irreversible.
- **Why it matters.** Provides the "why care" for the paper. If norms are equilibria, AI agents that sit at the same equilibrium are unproblematic. If norms are precision-allocations co-constituted by participants who are part of the substrate — i.e., norms supervene on shared cognitive architecture — then AI agents can hollow out norms while the *behavior* still looks fine. This is the paper's empirical/political stake.
- **Formal apparatus.** Position paper; conceptual; ICML 2025 Position Track.
- **arXiv:** 2501.16946.

---

## Key Researchers

- **Joel Z. Leibo** (Google DeepMind) — central node of MARL-norms work; melting-pot benchmark; sanctioning lineage.
- **Dylan Hadfield-Menell** (MIT CSAIL) — CIRL, legible normativity, silly rules, incomplete contracting; the most theoretically articulate voice on norms in alignment.
- **Gillian K. Hadfield** (Johns Hopkins / Toronto) — law-and-economics framing of normative infrastructure; co-author on incomplete contracting and silly rules.
- **Allan Dafoe** (Google DeepMind, formerly GovAI/Oxford) — agenda-setter; cooperative AI framing; AI governance.
- **Andrew Critch** (CHAI Berkeley / Encultured AI) — multi-principal safety, boundaries/membranes, ARCHES.
- **Tan Zhi-Xuan** (MIT, now Stanford-adjacent) — Bayesian normative inference; cognitive-science-rooted treatment of agency; close to the paper's intellectual register.
- **Conor Heins** (Max Planck Konstanz / VERSES) — multi-agent active inference; pymdp; the most relevant active-inference researcher for this cluster.
- **Giovanni Pezzulo** (CNR Rome) — interactive inference; cooperative joint action; shared protentions.
- **Karl Friston** (UCL / VERSES) — shared protentions; ecosystems of intelligence; the philosophical scaffolding.
- **Jan Kulveit** (ACS Prague) — gradual disempowerment; multi-agent active inference foundations (with Hyland, Gavenčiak).
- **David Hyland & Tomáš Gavenčiak** (Oxford / ACS Prague) — game-theoretic foundations of multi-agent active inference; IWAI 2024 best poster.
- **Eugene Vinitsky** (NYU) — decentralized norm acquisition via public sanctions.
- **Andrea Baronchelli** (City St. George's) — emergent social conventions in LLM populations.

---

## Surprising / Counterintuitive Findings

1. **Silly rules are *useful*, not noise.** Köster et al. (2022) show that arbitrary, welfare-irrelevant taboos *improve* MARL agents' acquisition of compliance with welfare-relevant taboos. The mechanism appears to be that silly rules increase the legibility/precision of the rule-following regime as such. From the paper's frame: silly rules pump up the precision of the policy prior "follow norms" without committing to any specific content. This is a striking inversion of the assumption that good normative systems are minimal/efficient. **Strong empirical hook for "norms-as-precision."**

2. **LLM populations spontaneously generate social conventions and collective biases that are invisible at the individual-agent level.** Ashery, Aiello & Baronchelli (2025, Science Advances) show convention emergence by population round 15 in most populations of LLM agents, with collective biases that no individual agent expresses. Committed minorities can flip the convention. This is exactly the macro-phenomenology a precision-prior account predicts: shared precision allocations are population-level objects that don't reduce to single-agent properties, and they exhibit phase-transition dynamics under minority pressure.

3. **Multi-agent active inference papers consistently model shared *goals* (preferred-state C-vectors) and shared *generative models* (A,B), but treat precision (γ, π) as private.** This is the gap. Friston's "shared protentions" comes closest by formalising shared anticipation, but doesn't draw the connection to norm-stability and sanctioning. The paper can claim the synthesis: *shared C is "values"; shared precision over priors and policies is "norms"; sanctioning and surprise-on-violation are the behavioral signatures of the latter.* No one currently in the literature has named this directly, as far as my search reaches.

---

## Open Questions / Gaps

- **Where does sanctioning sit in an active-inference account of norms?** In MARL it is a learnable policy. In an active-inference account it should fall out as the natural action when high-precision priors over others' policies are violated — i.e., the sanctioning behavior is the prediction-error response. This needs to be formalised. (Köster et al. is the empirical anchor.)
- **Can multi-agent active inference recover Lewis-style coordination on its own?** Hyland, Gavenčiak et al. (IWAI 2024) is the relevant attempt. Worth following up post-publication.
- **Is "shared precision" a coherent notion when each agent's generative model is distinct?** This is a technical question Cluster A should answer. From this cluster's vantage, the question is: what observable signatures distinguish "shared precision allocation" from "behavioral equilibrium that looks the same"?
- **AI-as-cultural-learner vs AI-as-norm-participant.** LLMs trained on human text inherit normative content but do they participate in norm-maintenance (precision allocation, sanctioning, surprise)? Kosoy, Gopnik et al. (2024) and the ratchet-effect literature suggest *not yet*. Gradual disempowerment becomes much sharper if this gap is real.
- **Concordia Contest and similar benchmarks.** Cooperative AI Foundation is now running benchmarks (Concordia 2024) testing promise-keeping, reciprocity, sanctioning, partner choice. These could become the empirical testbed for a precision-prior account if anyone operationalises it.

---

## Adjacent Reads (Secondary Tier)

- **Hadfield-Menell, Russell, Dragan, Abbeel (2016) — Cooperative Inverse Reinforcement Learning.** The original CIRL paper; relevant as the "value alignment as cooperative game" frame the paper inherits and pushes against. arXiv:1606.03137.
- **Hadfield-Menell, Andrus, Hadfield (2019) — Legible Normativity for AI Alignment: The Value of Silly Rules.** AIES 2019; theoretical companion to the 2022 PNAS empirical paper. arXiv:1811.01267.
- **Hadfield-Menell & Hadfield (2019) — Incomplete Contracting and AI Alignment.** AIES 2019; legal/economic framing of the normative-infrastructure dimension. arXiv:1804.04268.
- **Vinitsky, Köster, Agapiou, Duéñez-Guzmán, Vezhnevets, Leibo (2023) — A learning agent that acquires social norms from public sanctions in decentralized multi-agent settings.** Collective Intelligence; arXiv:2106.09012. The decentralized-sanctioning result.
- **Friston et al. (2024) — Shared Protentions in Multi-Agent Active Inference.** The Husserlian-phenomenology + sheaf-theoretic framing of shared anticipation. UCL Discovery 10190889.
- **Maisto, Donnarumma, Pezzulo (2022/2024) — Interactive inference: a multi-agent model of cooperative joint actions.** IEEE TSMCS; arXiv:2210.13113. Sensorimotor communication and intention legibility in active-inference dyads.
- **Friston, Ramstead, Kiefer et al. (2024) — Designing ecosystems of intelligence from first principles.** Collective Intelligence journal; the VERSES big-picture vision. Useful as a framing reference.
- **Hyland, Gavenčiak, da Costa, Heins, Kovarik, Gutierrez, Wooldridge, Kulveit (IWAI 2024) — Toward a Game-Theoretic Foundation of Multi-Agent Active Inference.** Best Poster IWAI 2024. Bridges game theory and active inference; check whether this scoops or supports the paper's move.
- **Cordova, Taverner, Del Val, Argente (2024) — A systematic review of norm emergence in multi-agent systems.** PRISMA review of 304 papers 2005–2024; arXiv:2412.10609. Useful for citation triage.
- **Ashery, Aiello, Baronchelli (2025) — Emergent social conventions and collective bias in LLM populations.** Science Advances 11(20), eadu9368; arXiv:2410.08948. The LLM-population convention-emergence result.
- **Yiu, Kosoy, Gopnik (2024) — Transmission Versus Truth, Imitation Versus Innovation: What Children Can Do That Large Language and Language-and-Vision Models Cannot (Yet).** Perspectives on Psychological Science. The "AI is not yet a cultural learner" position.
- **Heyes (2018) — Cognitive Gadgets.** Background frame for cultural learning as cognitive infrastructure; relevant if the paper pushes the cultural-learner angle.

---

## BibTeX Stubs

```bibtex
@article{dafoe2020openproblems,
  title={Open Problems in Cooperative AI},
  author={Dafoe, Allan and Hughes, Edward and Bachrach, Yoram and Collins, Tantum and McKee, Kevin R. and Leibo, Joel Z. and Larson, Kate and Graepel, Thore},
  journal={arXiv preprint arXiv:2012.08630},
  year={2020}
}

@article{critch2020arches,
  title={{AI} Research Considerations for Human Existential Safety ({ARCHES})},
  author={Critch, Andrew and Krueger, David},
  journal={arXiv preprint arXiv:2006.04948},
  year={2020}
}

@article{heins2024collective,
  title={Collective behavior from surprise minimization},
  author={Heins, Conor and Millidge, Beren and da Costa, Lancelot and Mann, Richard P. and Friston, Karl and Couzin, Iain D.},
  journal={Proceedings of the National Academy of Sciences},
  volume={121},
  number={17},
  pages={e2320239121},
  year={2024},
  doi={10.1073/pnas.2320239121}
}

@article{koster2022spurious,
  title={Spurious normativity enhances learning of compliance and enforcement behavior in artificial agents},
  author={K{\"o}ster, Raphael and Hadfield-Menell, Dylan and Everett, Richard and Weidinger, Laura and Hadfield, Gillian K. and Leibo, Joel Z.},
  journal={Proceedings of the National Academy of Sciences},
  volume={119},
  number={3},
  pages={e2106028118},
  year={2022},
  doi={10.1073/pnas.2106028118}
}

@inproceedings{oldenburg2024learning,
  title={Learning and Sustaining Shared Normative Systems via {B}ayesian Rule Induction in {M}arkov Games},
  author={Oldenburg, Ninell and Zhi-Xuan, Tan},
  booktitle={Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year={2024},
  note={arXiv:2402.13399}
}

@article{kulveit2025gradual,
  title={Gradual Disempowerment: Systemic Existential Risks from Incremental {AI} Development},
  author={Kulveit, Jan and Douglas, Raymond and Ammann, Nora and Turan, Deger and Krueger, David and Duvenaud, David},
  journal={arXiv preprint arXiv:2501.16946},
  year={2025},
  note={ICML 2025 Position Track}
}

@inproceedings{hadfieldmenell2016cirl,
  title={Cooperative Inverse Reinforcement Learning},
  author={Hadfield-Menell, Dylan and Russell, Stuart J. and Abbeel, Pieter and Dragan, Anca},
  booktitle={Advances in Neural Information Processing Systems},
  volume={29},
  year={2016}
}

@inproceedings{hadfieldmenell2019legible,
  title={Legible Normativity for {AI} Alignment: The Value of Silly Rules},
  author={Hadfield-Menell, Dylan and Andrus, McKane and Hadfield, Gillian K.},
  booktitle={Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES)},
  year={2019},
  note={arXiv:1811.01267}
}

@inproceedings{hadfieldmenell2019incomplete,
  title={Incomplete Contracting and {AI} Alignment},
  author={Hadfield-Menell, Dylan and Hadfield, Gillian K.},
  booktitle={Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES)},
  year={2019},
  note={arXiv:1804.04268}
}

@article{vinitsky2023sanctions,
  title={A learning agent that acquires social norms from public sanctions in decentralized multi-agent settings},
  author={Vinitsky, Eugene and K{\"o}ster, Raphael and Agapiou, John P. and Du{\'e}{\~n}ez-Guzm{\'a}n, Edgar A. and Vezhnevets, Alexander S. and Leibo, Joel Z.},
  journal={Collective Intelligence},
  volume={2},
  number={2},
  year={2023},
  doi={10.1177/26339137231162025}
}

@article{maisto2024interactive,
  title={Interactive Inference: A Multi-Agent Model of Cooperative Joint Actions},
  author={Maisto, Domenico and Donnarumma, Francesco and Pezzulo, Giovanni},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  volume={54},
  number={2},
  year={2024},
  note={arXiv:2210.13113}
}

@article{friston2024protentions,
  title={Shared Protentions in Multi-Agent Active Inference},
  author={Friston, Karl J. and others},
  year={2024},
  note={UCL Discovery 10190889}
}

@inproceedings{hyland2024gametheoretic,
  title={Toward a Game-Theoretic Foundation of Multi-Agent Active Inference},
  author={Hyland, David and Gaven{\v{c}}iak, Tom{\'a}{\v{s}} and da Costa, Lancelot and Heins, Conor and Kovarik, Vojtech and Gutierrez, Julian and Wooldridge, Michael J. and Kulveit, Jan},
  booktitle={5th International Workshop on Active Inference (IWAI 2024)},
  year={2024},
  note={Best Poster}
}

@article{ashery2025emergent,
  title={Emergent social conventions and collective bias in {LLM} populations},
  author={Ashery, Ariel Flint and Aiello, Luca Maria and Baronchelli, Andrea},
  journal={Science Advances},
  volume={11},
  number={20},
  pages={eadu9368},
  year={2025},
  doi={10.1126/sciadv.adu9368}
}

@article{cordova2024systematic,
  title={A systematic review of norm emergence in multi-agent systems},
  author={Cordova, Carmengelys and Taverner, Joaquin and Del Val, Elena and Argente, Estefania},
  journal={arXiv preprint arXiv:2412.10609},
  year={2024}
}

@article{friston2024ecosystems,
  title={Designing ecosystems of intelligence from first principles},
  author={Friston, Karl J. and Ramstead, Maxwell J. D. and Kiefer, Alex B. and Tschantz, Alexander and Buckley, Christopher L. and Albarracin, Mahault and Pitliya, Riddhi J. and Heins, Conor and Klein, Brennan and Millidge, Beren and others},
  journal={Collective Intelligence},
  volume={3},
  number={1},
  year={2024},
  doi={10.1177/26339137231222481}
}
```
