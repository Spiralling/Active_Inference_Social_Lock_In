---
cluster: B
title: Social and Cultural Active Inference
date: 2026-05-04
---

# Literature Review: Social and Cultural Active Inference — Shared Priors, Epistemic Communities, Social Power

**Cluster:** B (social-cultural-active-inference)
**Date:** 2026-05-04
**Scope:** Active inference applied to multi-agent, cultural, and social phenomena. Coverage: Thinking Through Other Minds (TTOM) and cultural niche construction; epistemic communities and shared generative models; active-inference accounts of social power; cultural affordances as shared precision structures; multi-agent surprise minimization. Excludes: single-agent precision/attention apparatus (Cluster A); gene-culture coevolution (Cluster C); cooperative AI / multi-agent RL (Cluster D); pure Bourdieu sociology unconnected to active inference (Cluster E).

---

## Key Findings

### 1. The TTOM framework treats culture as a regime of shared precision over what to attend to
Veissière, Constant, Ramstead, Friston & Kirmayer (2020) argue that humans acquire shared habits, norms and expectations not by representing other minds atom-by-atom but by **immersive participation in patterned practices that selectively pattern attention**. The variational reframing makes "culture" do precision-weighting work: cultural niches mark some information channels as reliable (high gain, high precision) and others as not. This is the load-bearing claim for our thesis. "Thinking through" other minds means inferring expectations through the niche's pre-structured saliences, not running an explicit theory-of-mind module. The TTOM target article was published in Behavioral and Brain Sciences 43:e90 with extensive open peer commentary.

### 2. Epistemic communities are natural attractors of self-confirming sampling — echo chambers fall out of free-energy minimization
Albarracin, Demekas, Ramstead & Heins (2022) build a POMDP active-inference model in which agents sample information sources they believe align with their views, update beliefs, and act (post hashtags). The dynamics produce stable echo chambers without any malign actor: shared priors over what counts as a reliable source create coupled belief-updating that locks communities in. The clean result for our thesis: **shared priors create shared sampling policies, which create shared evidence, which entrench the priors.** That circular causality is exactly the structure we want for "norms as stabilized regions of shared high-precision prior preference."

### 3. Social power, in active-inference terms, is asymmetric ability to bias others' priors and policies
Albarracin, de Jager, Hyland & Manski (2025) — confirmed: David Hyland is the third author — argue that empowered agents act as policy-amplifiers: they attract attention from others (others' priors over policies become anchored to the empowered agent's outputs), they extend their effective state space by relying on others' computational labor, and this creates a positive feedback loop. **Caveat (important — see "Concerns" below):** the abstract does not in fact contain explicit Bourdieu mappings of "habitus = priors," "field = precision-weighted attention landscape," "capital = differential ability to shape salience." The paper frames power as scale-invariant (physical → social) and emphasizes information-processing capacity and policy reach. The Bourdieu mapping appears in the broader Albarracin/Ramstead literature but should not be over-attributed to this specific paper without reading the body.

### 4. Cultural affordances and "regimes of attention" pre-date and ground TTOM
Ramstead, Veissière & Kirmayer (2016) introduced the formative move: **a "regime of attention" is a shared style of allocating attentional resources** that marks certain information channels as reliable (i.e., assigns them high precision) — and culture is a multilevel scaffolding mechanism that installs such regimes. The 2016 paper gives the conceptual machinery; the 2020 BBS target article gives the variational formalization. This framing is the cleanest historical anchor for the values-as-priors-shape-salience claim.

### 5. Constant et al.'s "deontic value" formalizes social conformity as a third value alongside epistemic and pragmatic
Constant, Ramstead, Veissière & Friston (2019) introduce **deontic value**: the (shared) value of a policy endowed by a direct policy–outcome mapping ("what one should do, given the cue"). Deontic cues (traffic lights, social roles) are environmental structures that emerge from repeated collective behavior and reify the cultural group's preferences as Bayes-optimal action defaults. This is the most concrete formal apparatus in the literature for showing that **norms are shared priors over policies with high precision**, because deontic cues short-circuit individual epistemic foraging. Highly relevant for IWAI 2026.

### 6. Multi-agent active inference can be done formally — schooling fish are surprise-minimizers
Heins, Millidge, Da Costa, Mann, Friston & Couzin (2024, PNAS) demonstrate that classical collective phenomena (cohesion, milling, directed motion) emerge from agents minimizing variational free energy under a generative model that includes generalized coordinates of motion of conspecifics. This is the most rigorous existing demonstration that **shared (or coupled) generative models produce coordinated collective behavior** without explicit social-force rules. Important as proof-of-concept that the apparatus scales beyond the dyad.

### 7. Group-level generative models are non-trivially related to individual ones
Heins et al.'s "As One and Many" (2025, Entropy) shows that a collective with a Markov blanket can be modeled *as* a higher-order active inference agent with its own generative model — but the group-level model is not just a sum of individual models. Important for our thesis because it cautions against the naive picture where "norms = average of individual priors." Norms may be group-level priors that emerge non-trivially from coupling dynamics.

---

## Must-Read Shortlist

### Thinking Through Other Minds: A Variational Approach to Cognition and Culture — Veissière, Constant, Ramstead, Friston & Kirmayer (2020)
- **Venue:** Behavioral and Brain Sciences, 43:e90 (Target Article + Commentaries + Authors' Response)
- **DOI:** 10.1017/S0140525X19001213
- **Central claim:** Humans learn shared habits, norms and expectations through immersive participation in patterned cultural practices that selectively pattern attention; "TTOM" replaces internalist theory-of-mind with a variational, niche-mediated process.
- **Why it matters for our thesis:** Provides the canonical statement that *culture installs precision regimes*, which is one of the two main pillars of values-as-salience-priors. The BBS commentaries (Andrews, Spaulding, Vande Cruys & Heylighen, Gweon, Werner, Trafford, Albahari, Sterelny et al., among others) supply most of the field's critical objections — read them en bloc.
- **Formal apparatus to borrow:** TTOM as hierarchical generative model with cultural priors as hyperpriors; cultural learning as second-order inference about which information sources to weight.

### Epistemic Communities under Active Inference — Albarracin, Demekas, Ramstead & Heins (2022)
- **Venue:** Entropy 24(4):476
- **DOI:** 10.3390/e24040476
- **Central claim:** Echo chambers are natural attractors when agents minimize free energy under shared priors that bias source-sampling. Confirmation bias falls out of self-evidencing dynamics, not malice or stupidity.
- **Why it matters for our thesis:** Most direct existing model of how shared priors stabilize a community by closing the perception-action loop on confirmatory evidence. Norms-as-stabilized-priors gets a working simulation here.
- **Formal apparatus to borrow:** POMDP-based multi-agent simulation; "meta-beliefs" (beliefs about neighbors' beliefs) as hidden state factors; observable acts (hashtags) as the social-evidence channel.

### The Physics and Metaphysics of Social Powers: A New Perspective on Power through Active Inference — Albarracin, de Jager, Hyland & Manski (2025)
- **Venue:** arXiv preprint, also Entropy 27(5):522 (2025)
- **DOI / arXiv:** arXiv:2501.19368; doi 10.3390/e27050522
- **Central claim:** Social power = augmented capacity to harness computation toward desired ends, partly via attention-attraction (others' attention amplifies an empowered agent's effective policy reach) and partly via offloading computation to social allies. Self-reinforcing because empowerment increases information-processing buffer against vulnerabilities.
- **Why it matters for our thesis:** Closest published statement that **power is differential capacity to shape what others attend to and which policies they enact** — i.e., differential capacity to install priors and bias precision in others. Directly germane to "norms as stabilized shared priors maintained by power asymmetries."
- **Formal apparatus to borrow:** Power as a function of policy-space breadth × computational throughput × social attention received. Note: as discussed in concerns below, the explicit Bourdieu→AIF mapping (habitus/field/capital) may be more in the spirit of the paper than its letter.

### Cultural Affordances: Scaffolding Local Worlds Through Shared Intentionality and Regimes of Attention — Ramstead, Veissière & Kirmayer (2016)
- **Venue:** Frontiers in Psychology, 7:1090
- **DOI:** 10.3389/fpsyg.2016.01090
- **Central claim:** Cultural affordances are jointly attended environmental opportunities for action; "regimes of attention" are shared styles of allocating attention that mark some channels as especially reliable, gating perceptual and motor learning.
- **Why it matters for our thesis:** Earliest clean statement of the salience-priors-as-cultural-installation idea. The 2020 TTOM paper is its variational formalization; this is its conceptual seed. Cite both.
- **Formal apparatus to borrow:** Multilevel affordance learning; cultural transmission as patterning of joint attention that increases neuronal gain (≈ precision) on culturally privileged channels.

### Regimes of Expectations: An Active Inference Model of Social Conformity — Constant, Ramstead, Veissière & Friston (2019)
- **Venue:** Frontiers in Psychology, 10:679
- **DOI:** 10.3389/fpsyg.2019.00679
- **Central claim:** Introduces **deontic value** — a third value type alongside epistemic and pragmatic — that governs decision-making in deontically cued environments (e.g., traffic lights). Deontic cues emerge from repeated collective behavior and reify cultural-group preferences in environmental structure.
- **Why it matters for our thesis:** The most concrete formal proposal for *what kind of prior a norm is*. A norm is a deontic prior over policies, made high-precision by environmental and social cueing. We can lift this directly into the IWAI paper.
- **Formal apparatus to borrow:** DEEP model (Deontic, Epistemic, Pragmatic value); MDP setup with cued obligations; circular causality between cued behavior and cue salience.

### Collective Behavior from Surprise Minimization — Heins, Millidge, Da Costa, Mann, Friston & Couzin (2024)
- **Venue:** Proceedings of the National Academy of Sciences, 121(17):e2320239121
- **DOI:** 10.1073/pnas.2320239121
- **Central claim:** Collective phenomena (cohesion, milling, directed motion) emerge naturally from individual agents minimizing variational free energy under generative models of conspecific motion; recovers and generalizes classical self-propelled-particle (Vicsek-style) models as special cases.
- **Why it matters for our thesis:** The proof-of-concept that multi-agent active inference produces coordinated, norm-like collective patterns from coupled generative models — without explicit social-force rules and without explicit utility functions. Useful as the "this is not just talk" reference for IWAI reviewers.
- **Formal apparatus to borrow:** Generalized filtering with generalized coordinates of motion; precision over sensory channels (each conspecific is a sensor) determines coupling strength → analogue of social influence weighting.

---

## Key Researchers

- **Maxwell J. D. Ramstead** (VERSES Research Lab; McGill; UCL Wellcome Centre for Human Neuroimaging) — central architect of the cultural-active-inference programme; co-author on TTOM, cultural affordances, ecosystems of intelligence, and most adjacent papers.
- **Mahault Albarracin** (UQAM; Director of Product, VERSES Research Lab) — leading the social-power, epistemic-communities, narrative, and explainable-AI threads. The center of gravity for "social" active inference 2022–2026.
- **Karl J. Friston** (UCL; VERSES) — originator of the free-energy principle; senior author on virtually every paper in this cluster.
- **Samuel P. L. Veissière** (McGill; Department of Psychiatry) — anthropologist; originator with Kirmayer of "regimes of attention"; co-author TTOM.
- **Laurence J. Kirmayer** (McGill; Division of Social and Transcultural Psychiatry) — psychiatric anthropologist; bridges medical anthropology and active inference; senior co-author TTOM and "Active Inference and Social Actors."
- **Axel Constant** (University of Sussex; Macquarie) — formal modeling of cultural niche construction (deontic value, culturally patterned attention styles, extended active inference).
- **Conor Heins** (Max Planck Institute of Animal Behavior; Konstanz; VERSES) — multi-agent active inference, schooling-fish work, group-level generative models.
- **David Hyland** (University of Oxford) — confirmed third author on the 2025 social-power paper; multi-agent and game-theoretic active inference.
- **Sonia de Jager** (independent researcher; first/second-author social powers 2025) — phenomenology and political theory in active-inference register.
- **Sarah Grace Manski** (George Mason University) — political economy / social theory; co-author social powers 2025.
- **Inês Hipólito** (Macquarie; Humboldt) — enactive and extended active inference; social cognition; critical of theory-of-mind framings.
- **Michael Kirchhoff** (University of Wollongong) — extended/embodied active inference; co-author on extended active inference (Mind & Language 2022).
- **Jared Vasil** (Duke) — first author "A World Unto Itself" on communication as active inference.
- **Nabil Bouizegarene** (McGill) — first author on narrative as active inference (2024).
- **Iain D. Couzin** (Max Planck Konstanz) — collective animal behavior; senior on the schooling-fish PNAS paper.
- **Jacob E. Cheadle** (UT Austin) — sociology + active inference ("Active Inference and Social Actors," KZfSS 2024).

---

## Surprising / Counterintuitive Findings

1. **Echo chambers are not a pathology of bad epistemic agents — they are the equilibrium for *any* agents minimizing free energy under shared confirmatory priors.** Albarracin et al. 2022 show this in a clean simulation. The implication for our paper is sharp: stabilized shared high-precision priors *are* echo chambers when viewed as belief states, and *are* norms when viewed as policy distributions. The two are the same dynamical phenomenon at different levels of description.

2. **Norm-following may not require any additional cognitive machinery beyond Bayes-optimal action under deontic cues.** Constant et al.'s "deontic value" reduces social conformity to environmental cueing of policies the cultural group has already pre-committed to via repeated collective action. Conformity is not "following the rule" — it is sampling the next action from a pre-shaped posterior whose shape is held in the environment, not in the head. This eats a lot of social-cognition theorizing in one bite.

3. **Group-level generative models are non-trivially related to individual ones — a culture is not the average of its members.** "As One and Many" (Heins et al. 2025) shows the group-level generative model emerging from a Markov-blanketed collective can have parameters that no individual instantiates. This pushes back on naive aggregate-of-priors stories and suggests *norms are emergent group-level priors* with their own dynamics. (Worth flagging as a concern for our central claim — not a refutation, but a refinement.)

4. **Cultural priors over communication itself ground language exchange.** Vasil et al. (2020) argue humans carry an evolved adaptive prior that "we are the same kind of creature in the same kind of niche" — and the *relative precision* of this prior between two communicating agents constrains turn-taking and information flow. So precision-weighting isn't just intra-agent attention; it modulates *between-agent* communication. This is an underappreciated formal handle.

---

## Open Questions / Gaps

- **Mathematical operationalization of "shared" priors.** Most papers use "shared" loosely. Exceptions: Albarracin et al. (2024, "Shared Protentions") use sheaf/topos theory to formalize coherent worldviews across agents. Heins et al.'s "As One and Many" attempts the bridge from individual to group-level model. There is no canonical formalism yet — opportunity for IWAI 2026 to propose one.
- **Empirical grounding.** Almost everything in this cluster is theoretical or simulation-based. The schooling-fish PNAS paper is the clearest empirical anchor; cultural-priors work in humans remains mostly conceptual. Constant et al.'s 2021 paper on culturally patterned attention styles ("antique vase decorations" simulation) is a step toward empirical operationalization.
- **What distinguishes a *value* from a generic prior?** The literature uses "preference," "deontic value," "prior preference," and "C-matrix" semi-interchangeably. A clean conceptual contribution at IWAI would be: *values are priors that have been epistemically fenced (i.e., made resistant to evidential update) by being installed via cultural-affective rather than perceptual channels.* This is what we should argue.
- **Norm-violation dynamics.** The literature focuses on stabilization. Less work on how norms break — when does a high-precision shared prior become updatable? Albarracin's broader work touches this; Hipólito's enactive critique might supply ammunition.

---

## Adjacent Reads (Secondary Tier)

- **Constant, Clark, Kirchhoff & Friston (2022)** — "Extended active inference: Constructing predictive cognition beyond skulls." *Mind & Language*, 37:865–882. doi: 10.1111/mila.12330. — Cognitive niche construction as optimization of generative models; important for the "values are partly held in the environment" line.
- **Constant et al. (2021)** — "The Acquisition of Culturally Patterned Attention Styles Under Active Inference." *Frontiers in Neurorobotics*, 15:729665. doi: 10.3389/fnbot.2021.729665. — Simulation of how cultural artefacts (vase decorations) shape attention via state-dependent likelihood remapping.
- **Hipólito, Hutto et al. (2022)** — "Enactive-Dynamic Social Cognition and Active Inference." *Frontiers in Psychology*, 13:855074. doi: 10.3389/fpsyg.2022.855074. — Critical refinement that pushes back on theory-of-mind readings of TTOM.
- **Albarracin, Pitliya, St Clere Smithe, Friedman, Friston & Ramstead (2024)** — "Shared Protentions in Multi-Agent Active Inference." *Entropy*, 26(4):303. doi: 10.3390/e26040303. — Formalizes shared goals using sheaf/topos theory; phenomenology + AIF + category theory.
- **Heins et al. (2025)** — "As One and Many: Relating Individual and Emergent Group-Level Generative Models in Active Inference." *Entropy*, 27(2):143. doi: 10.3390/e27020143.
- **Heins, Millidge, Da Costa, Mann, Friston & Couzin (2024)** — "Collective Behavior from Surprise Minimization." *PNAS*, 121(17):e2320239121.
- **Vasil, Badcock, Constant, Friston & Ramstead (2020)** — "A World Unto Itself: Human Communication as Active Inference." *Frontiers in Psychology*, 11:417. doi: 10.3389/fpsyg.2020.00417.
- **Bouizegarene, Ramstead, Constant, Friston & Kirmayer (2024)** — "Narrative as Active Inference: An Integrative Account of Cognitive and Social Functions in Adaptation." *Frontiers in Psychology*, 15:1345480. doi: 10.3389/fpsyg.2024.1345480.
- **Cheadle, Davidson-Turner & Goosby (2024)** — "Active Inference and Social Actors: Towards a Neuro-Bio-Social Theory of Brains and Bodies in Their Worlds." *KZfSS Kölner Zeitschrift für Soziologie und Sozialpsychologie*, 76:317–350. doi: 10.1007/s11577-024-00936-4.
- **Friston, Ramstead, Albarracin et al. (2024)** — "Designing Ecosystems of Intelligence from First Principles." *Collective Intelligence*, 3(1). doi: 10.1177/26339137231222481. — Long white paper; section on shared generative models is what to read.
- **Albarracin, Hipólito et al. (2024)** — "Designing Explainable AI with Active Inference." Springer. arXiv:2306.04025. — Notable for the line "our confidence in our high-level beliefs is heavily influenced by culture at the expense of direct experience."
- **Vande Cruys & Heylighen (2020)** — "The dark side of thinking through other minds." *BBS* commentary. Useful for failure modes of TTOM (manipulation, mass formation).
- **Selected BBS commentaries on TTOM (2020)** — "Thinking with other minds" (Andrews/Spaulding-style critique); "Have we lost the thinker in other minds?"; "Encultured minds, not error-reduction minds"; "Skill-based engagement with a rich landscape of affordances as an alternative to TTOM"; "The dark side of TTOM" (Vande Cruys & Heylighen); "Importance of the TTOM process explored through motor correlates."

---

## BibTeX Stubs

```bibtex
@article{veissiere2020thinking,
  title   = {Thinking through other minds: A variational approach to cognition and culture},
  author  = {Veissi{\`e}re, Samuel P. L. and Constant, Axel and Ramstead, Maxwell J. D. and Friston, Karl J. and Kirmayer, Laurence J.},
  journal = {Behavioral and Brain Sciences},
  volume  = {43},
  pages   = {e90},
  year    = {2020},
  doi     = {10.1017/S0140525X19001213}
}

@article{albarracin2022epistemic,
  title   = {Epistemic Communities under Active Inference},
  author  = {Albarracin, Mahault and Demekas, Daphne and Ramstead, Maxwell J. D. and Heins, Conor},
  journal = {Entropy},
  volume  = {24},
  number  = {4},
  pages   = {476},
  year    = {2022},
  doi     = {10.3390/e24040476}
}

@article{albarracin2025physics,
  title   = {The Physics and Metaphysics of Social Powers: Bridging Cognitive Processing and Social Dynamics, a New Perspective on Power through Active Inference},
  author  = {Albarracin, Mahault and de Jager, Sonia and Hyland, David and Manski, Sarah Grace},
  journal = {Entropy},
  volume  = {27},
  number  = {5},
  pages   = {522},
  year    = {2025},
  doi     = {10.3390/e27050522},
  eprint  = {2501.19368},
  archivePrefix = {arXiv}
}

@article{ramstead2016cultural,
  title   = {Cultural Affordances: Scaffolding Local Worlds Through Shared Intentionality and Regimes of Attention},
  author  = {Ramstead, Maxwell J. D. and Veissi{\`e}re, Samuel P. L. and Kirmayer, Laurence J.},
  journal = {Frontiers in Psychology},
  volume  = {7},
  pages   = {1090},
  year    = {2016},
  doi     = {10.3389/fpsyg.2016.01090}
}

@article{constant2019regimes,
  title   = {Regimes of Expectations: An Active Inference Model of Social Conformity and Human Decision Making},
  author  = {Constant, Axel and Ramstead, Maxwell J. D. and Veissi{\`e}re, Samuel P. L. and Friston, Karl},
  journal = {Frontiers in Psychology},
  volume  = {10},
  pages   = {679},
  year    = {2019},
  doi     = {10.3389/fpsyg.2019.00679}
}

@article{heins2024collective,
  title   = {Collective behavior from surprise minimization},
  author  = {Heins, Conor and Millidge, Beren and Da Costa, Lancelot and Mann, Richard P. and Friston, Karl J. and Couzin, Iain D.},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {121},
  number  = {17},
  pages   = {e2320239121},
  year    = {2024},
  doi     = {10.1073/pnas.2320239121}
}

@article{constant2022extended,
  title   = {Extended active inference: Constructing predictive cognition beyond skulls},
  author  = {Constant, Axel and Clark, Andy and Kirchhoff, Michael and Friston, Karl J.},
  journal = {Mind \& Language},
  volume  = {37},
  number  = {5},
  pages   = {865--882},
  year    = {2022},
  doi     = {10.1111/mila.12330}
}

@article{constant2021acquisition,
  title   = {The Acquisition of Culturally Patterned Attention Styles Under Active Inference},
  author  = {Constant, Axel and Tschantz, Alexander and Millidge, Beren and Criado-Boado, Felipe and Martinez, Luis M. and M{\"u}eller, Johannes and Clark, Andy},
  journal = {Frontiers in Neurorobotics},
  volume  = {15},
  pages   = {729665},
  year    = {2021},
  doi     = {10.3389/fnbot.2021.729665}
}

@article{hipolito2022enactive,
  title   = {Enactive-Dynamic Social Cognition and Active Inference},
  author  = {Hip{\'o}lito, In{\^e}s and others},
  journal = {Frontiers in Psychology},
  volume  = {13},
  pages   = {855074},
  year    = {2022},
  doi     = {10.3389/fpsyg.2022.855074}
}

@article{albarracin2024shared,
  title   = {Shared Protentions in Multi-Agent Active Inference},
  author  = {Albarracin, Mahault and Pitliya, Riddhi J. and St Clere Smithe, Toby and Friedman, Daniel Ari and Friston, Karl and Ramstead, Maxwell J. D.},
  journal = {Entropy},
  volume  = {26},
  number  = {4},
  pages   = {303},
  year    = {2024},
  doi     = {10.3390/e26040303}
}

@article{heins2025asoneandmany,
  title   = {As One and Many: Relating Individual and Emergent Group-Level Generative Models in Active Inference},
  author  = {Heins, Conor and others},
  journal = {Entropy},
  volume  = {27},
  number  = {2},
  pages   = {143},
  year    = {2025},
  doi     = {10.3390/e27020143}
}

@article{vasil2020world,
  title   = {A World Unto Itself: Human Communication as Active Inference},
  author  = {Vasil, Jared and Badcock, Paul B. and Constant, Axel and Friston, Karl and Ramstead, Maxwell J. D.},
  journal = {Frontiers in Psychology},
  volume  = {11},
  pages   = {417},
  year    = {2020},
  doi     = {10.3389/fpsyg.2020.00417}
}

@article{bouizegarene2024narrative,
  title   = {Narrative as active inference: an integrative account of cognitive and social functions in adaptation},
  author  = {Bouizegarene, Nabil and Ramstead, Maxwell J. D. and Constant, Axel and Friston, Karl J. and Kirmayer, Laurence J.},
  journal = {Frontiers in Psychology},
  volume  = {15},
  pages   = {1345480},
  year    = {2024},
  doi     = {10.3389/fpsyg.2024.1345480}
}

@article{cheadle2024active,
  title   = {Active Inference and Social Actors: Towards a Neuro-Bio-Social Theory of Brains and Bodies in Their Worlds},
  author  = {Cheadle, Jacob E. and Davidson-Turner, K. J. and Goosby, Bridget J.},
  journal = {KZfSS K{\"o}lner Zeitschrift f{\"u}r Soziologie und Sozialpsychologie},
  volume  = {76},
  pages   = {317--350},
  year    = {2024},
  doi     = {10.1007/s11577-024-00936-4}
}

@article{friston2024designing,
  title   = {Designing ecosystems of intelligence from first principles},
  author  = {Friston, Karl J. and Ramstead, Maxwell J. D. and Kiefer, Alex B. and Tschantz, Alexander and Buckley, Christopher L. and Albarracin, Mahault and Pitliya, Riddhi J. and Heins, Conor and others},
  journal = {Collective Intelligence},
  volume  = {3},
  number  = {1},
  year    = {2024},
  doi     = {10.1177/26339137231222481}
}

@article{vandecruys2020dark,
  title   = {The dark side of thinking through other minds},
  author  = {Vande Cruys, Sander and Heylighen, Francis},
  journal = {Behavioral and Brain Sciences},
  volume  = {43},
  pages   = {e92},
  year    = {2020},
  note    = {Commentary on Veissi{\`e}re et al. 2020},
  doi     = {10.1017/S0140525X20000035}
}
```
