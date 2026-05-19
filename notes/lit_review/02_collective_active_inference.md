# Collective / Multi-Agent Active Inference: Bistability and Phase Transitions

*Lit review for IWAI 2026 paper — substrate diagnosis. Compiled 2026-05-19.*

## Exec summary (3 sentences)

Across the multi-agent AIF literature, **genuine bistability is rare**: most "collective phase transitions" in AIF papers are either (a) precision-tuned crossings between *distinct attractors of a contraction map* (Heins/Couzin 2024, schooling vs milling — same Gaussian substrate as ours, so it would NOT save us), or (b) bifurcations driven by **discrete categorical state spaces with multimodal C-vectors / preferences and softmax policy selection** (Albarracin 2022 echo chambers, pymdp-based). The two real candidates for *grafting bistability* onto our Gaussian-conjugate substrate are (i) Albarracin's epistemic-confirmation-bias mechanism, which requires a categorical POMDP rewrite, and (ii) Heins/Da Costa's precision-gated regime selection, which our substrate already has the ingredients for but apparently still collapses to a single attractor — so the lesson is that *precision tuning alone is not bistability* unless preferences themselves are multimodal.

---

## 1. Heins, Millidge, Da Costa, Mann, Friston, Couzin (2024) — "Collective behavior from surprise minimization"

**BibTeX**
```bibtex
@article{heins2024collective,
  title   = {Collective behavior from surprise minimization},
  author  = {Heins, Conor and Millidge, Beren and Da Costa, Lancelot and
             Mann, Richard P. and Friston, Karl J. and Couzin, Iain D.},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {121},
  number  = {17},
  pages   = {e2320239121},
  year    = {2024},
  doi     = {10.1073/pnas.2320239121}
}
```

**Code**: https://github.com/conorheins/collective_motion_actinf — Python (JAX) + Julia, 33 commits, companion to PNAS paper. Real, working.

**Mechanism in one sentence.** Each fish maintains a *continuous* Gaussian generative model over distances-to-neighbors using generalized coordinates of motion (position, velocity, acceleration); the *sensory precision* parameters Γ_z (amplitude) and λ_z (smoothness of noise) act as control parameters that switch the collective between polarized (schooling) and rotational (milling) regimes — but each regime is an attracting fixed point of the *individual-level* belief-update dynamics, not a separatrix-bounded basin of a Hopf/pitchfork bifurcation at the collective level.

**Graft analysis.** This is **the closest substrate to ours**: Gaussian-conjugate continuous-state inference, precision learning, no categorical hidden states. We would graft by (a) extending `src/world.py` to carry generalized coordinates (θ, θ̇, θ̈) instead of just θ, and (b) letting each agent's *sensory precision* Γ_z be a learned variable (currently our τ_obs is exogenous). The catch: their "phase transition" is *qualitative geometry of the attractor* (polarized line vs rotating ring), not bistability — the collective still has a single basin per precision setting. **This would not give us bistability; it would give us richer single-basin geometry.** Same trap our nb13 just diagnosed.

---

## 2. Waade, Olesen, Laursen, Nehrer, Heins, Friston, Mathys (2025) — "As One and Many: Group-Level Generative Models in Active Inference"

**BibTeX**
```bibtex
@article{waade2025asoneandmany,
  title   = {As One and Many: Relating Individual and Emergent Group-Level
             Generative Models in Active Inference},
  author  = {Waade, Peter Thestrup and Olesen, Christoffer Lundbak and
             Laursen, Jonathan Ehrenreich and Nehrer, Samuel William and
             Heins, Conor and Friston, Karl and Mathys, Christoph},
  journal = {Entropy},
  volume  = {27},
  number  = {2},
  pages   = {143},
  year    = {2025},
  doi     = {10.3390/e27020143}
}
```

**Code**: ActiveInference.jl — https://github.com/ComputationalPsychiatry/ActiveInference.jl. Julia, actively maintained. Companion to Nehrer et al. (Entropy 2025, 27(1):62).

**Mechanism.** Discrete categorical POMDP agents with a *group-level Markov blanket*; the group itself is recovered (via sampling-based parameter estimation) as a higher-order categorical generative model. No bistability is explicitly demonstrated; the contribution is *methodological* — how to read off the emergent group model from collective trajectories.

**Graft analysis.** Not directly useful as a bistability source, but **methodologically important**: gives us a principled way to define when our population *constitutes* a paradigm-agent (group-level Markov blanket). Would require rewriting `src/inference.py` for discrete states; not worth the cost unless we're already migrating to categorical. Cite as theoretical scaffolding for "what would it mean for a paradigm to be a generative model in its own right."

---

## 3. Albarracin, Demekas, Ramstead, Heins (2022) — "Epistemic Communities under Active Inference"

**BibTeX**
```bibtex
@article{albarracin2022epistemic,
  title   = {Epistemic Communities under Active Inference},
  author  = {Albarracin, Mahault and Demekas, Daphne and
             Ramstead, Maxwell J. D. and Heins, Conor},
  journal = {Entropy},
  volume  = {24},
  number  = {4},
  pages   = {476},
  year    = {2022},
  doi     = {10.3390/e24040476}
}
```

**Code**: pymdp (https://github.com/infer-actively/pymdp) — Python, actively maintained, JAX backend available. Paper does not link a *specific* repo with the experiment code; the simulations are pymdp scripts.

**Mechanism in one sentence.** **Discrete categorical POMDP** agents with (i) softmax over a precision-weighted *epistemic confirmation bias* term γ that biases neighbor-sampling toward agents with predicted-similar tweets, and (ii) a learned habit prior η over which neighbors to attend to — together this gives a **positive-feedback loop on attention that produces basin-switching between consensus and polarized states** as γ and the inverse social volatility ωSoc are swept (their Fig. 3 heatmaps).

**Graft analysis.** **This is the strongest candidate for actual bistability**, but it requires moving off our Gaussian conjugate substrate: the bistability lives in the *discrete attention/habit prior over neighbors*, multiplied through a softmax. To replicate without going fully categorical, we would:
- Keep continuous θ inference in `src/inference.py` (Gaussian-conjugate is fine).
- Replace our row-stochastic trust matrix with a **softmax over a learned habit prior with precision γ**: τ_ij ∝ exp(γ · ⟨expected_alignment_ij⟩) — this is roughly what `src/trust.py` does but *without* the softmax precision (we use linear pooling).
- Add the habit-prior update (a Dirichlet over attention controls) — a new tiny module `src/attention.py`.
- The softmax + Dirichlet positive feedback is what generates the bistability; *linear pooling cannot*, which explains nb12's "trust is decorative" finding directly.

**This is the load-bearing graft.** Bistability does NOT require categorical hidden states for θ; it requires a **softmax-precision-driven attention/trust update** with a learnable concentration prior. That can be bolted onto our continuous-θ substrate.

---

## 4. Albarracin, de Jager, Hyland, Manski (2025) — "The Physics and Metaphysics of Social Powers"

**BibTeX**
```bibtex
@article{albarracin2025socialpowers,
  title   = {The Physics and Metaphysics of Social Powers: Bridging
             Cognitive Processing and Social Dynamics, a New Perspective
             on Power through Active Inference},
  author  = {Albarracin, Mahault and de Jager, Sonia and
             Hyland, David and Manski, Sarah Grace},
  journal = {Entropy},
  year    = {2025},
  eprint  = {2501.19368},
  archivePrefix = {arXiv}
}
```

**Code**: No public repository. Conceptual/theoretical paper, no simulation.

**Mechanism.** Argues social power = attention-capture × information-processing capacity, mediated by *attentional scripts* (narratives, ideologies). **No formal phase-transition result**; the paper is a position piece building the conceptual bridge from individual AIF to social power, including Hyland's λ·U attentional framing that we already cite.

**Graft analysis.** Not a substrate or mechanism we can graft — it's our paper's *philosophical neighbor*, useful for §6 discussion to position our resource-coupled paradigm dynamics as an empirical implementation of their attention-capture-as-power claim. Cite, do not graft.

---

## 5. Constant, Ramstead, Veissière, Friston (2019) — "Regimes of Expectations"

**BibTeX**
```bibtex
@article{constant2019regimes,
  title   = {Regimes of Expectations: An Active Inference Model of
             Social Conformity and Human Decision Making},
  author  = {Constant, Axel and Ramstead, Maxwell J. D. and
             Veissi{\`e}re, Samuel P. L. and Friston, Karl J.},
  journal = {Frontiers in Psychology},
  volume  = {10},
  pages   = {679},
  year    = {2019},
  doi     = {10.3389/fpsyg.2019.00679}
}
```

**Code**: None public. Worked-example MDP, no released code.

**Mechanism.** Categorical MDP with *deontic value* — a preference prior over "what people like me do" — that makes conformity-favoring policies dominate via softmax-over-EFE. No explicit bifurcation; the regime structure is hand-set via the C-vector.

**Graft analysis.** Useful conceptually (formalizes "paradigm" as a deontic-value distribution) but not a runnable substrate. Categorical only. The take-home for us: a *multimodal preference distribution* C (e.g., two paradigm peaks) under softmax-EFE is the canonical AIF route to bistability — but it requires categorical θ.

---

## 6. Veissière, Constant, Ramstead, Friston, Kirmayer (2019) — "Thinking through other minds"

**BibTeX**
```bibtex
@article{veissiere2020thinking,
  title   = {Thinking through other minds: A variational approach to
             cognition and culture},
  author  = {Veissi{\`e}re, Samuel P. L. and Constant, Axel and
             Ramstead, Maxwell J. D. and Friston, Karl J. and
             Kirmayer, Laurence J.},
  journal = {Behavioral and Brain Sciences},
  volume  = {43},
  pages   = {e90},
  year    = {2020},
  doi     = {10.1017/S0140525X19001213}
}
```

**Code**: None. BBS target article — purely conceptual.

**Mechanism.** Argues cultural expectations are *learned hyperpriors* shared across a community via mutual modeling. No formal model, no bistability claim.

**Graft analysis.** Not a substrate. Cite in §2 (related work) to anchor the "paradigm as shared hyperprior" framing; do not attempt to graft.

---

## 7. Kaufmann, Cisneros-Velarde, Bullo et al. (2021, 2024) — Social-reality construction / norm acquisition AIF

**BibTeX (placeholder for the 2026 social-reality-aif paper)**
```bibtex
@misc{anon2026socialreality,
  title         = {Social Reality Construction via Active Inference:
                   Modeling the Dialectic of Conformity and Creativity},
  author        = {Anonymous},
  year          = {2026},
  eprint        = {2604.09026},
  archivePrefix = {arXiv}
}
```

**Code**: https://github.com/jemand-rkn/social-reality-aif — exists but lightly maintained; Python with neural-net VAE/f-GAN parameterization.

**Mechanism.** Continuous-vector representations (2-D observation, 4-D social) with VAE-parameterized likelihoods. Shows *temporal sequence* (conformity → creativity) rather than bistability between them; complementary, not oscillatory.

**Graft analysis.** Not a bistability source. Their continuous-vector + VAE machinery is heavier than we need and does not buy us multistability. Skip for grafting; cite as a "creative deviation" precedent in §6.

---

## 8. Friedman, Tschantz, Ramstead, Friston, Constant (2021) — "Active Inference Model of Collective Intelligence"

**BibTeX**
```bibtex
@article{friedman2021collective,
  title   = {An Active Inference Model of Collective Intelligence},
  author  = {Friedman, Daniel A. and Tschantz, Alexander and
             Ramstead, Maxwell J. D. and Friston, Karl and
             Constant, Axel},
  journal = {Entropy},
  volume  = {23},
  number  = {7},
  pages   = {830},
  year    = {2021},
  doi     = {10.3390/e23070830}
}
```

**Code**: No paper-specific repo; relies on pymdp.

**Mechanism.** Categorical POMDP agents progressing through "stepwise cognitive transitions" (baseline → ToM → goal alignment). The transitions are *between successive model architectures*, not between attractors of a single dynamical system — there is no formal bifurcation; alignment "emerges endogenously."

**Graft analysis.** Conceptually adjacent (alignment via interaction) but no bistability mechanism. Categorical-only. Cite in §2 for "collective intelligence from AIF agents" lineage; do not graft.

---

## Cross-cutting diagnosis

The pattern across all 8 works:

| Paper | Substrate | Real bistability? | Graftable to ours? |
|-------|-----------|-------------------|--------------------|
| Heins 2024 | Continuous Gaussian (generalized coords) | No — precision-tuned attractor geometry | Yes — same substrate, but won't give bistability |
| Waade 2025 | Categorical | No — methodological | No |
| **Albarracin 2022** | **Categorical + softmax habits** | **Yes — γ × η feedback loop** | **Yes — graft the softmax-trust + habit prior, keep continuous θ** |
| Albarracin 2025 | None (theory) | N/A | Discussion only |
| Constant 2019 | Categorical MDP | Latent (multimodal C) | Only if we go categorical |
| Veissière 2019 | None (BBS) | N/A | Citation only |
| social-reality-aif | Continuous + VAE | No — temporal sequence | Skip |
| Friedman 2021 | Categorical | No — architectural transitions | Citation only |

**The clean finding for our paper.** Bistability in AIF collectives in the literature comes from **categorical preferences + softmax precision over policies/attention**, not from continuous-state Gaussian inference. Our substrate's monostability (Jacobian sum-of-negative-definites) is *the generic outcome* of Gaussian-conjugate + linear pooling. To get bistability without abandoning continuous θ, we need a **nonlinear (softmax-precision) trust/attention update** — i.e., upgrade `src/trust.py` from linear row-stochastic pooling to softmax-over-expected-alignment with a learnable concentration γ. This is the Albarracin 2022 graft.
