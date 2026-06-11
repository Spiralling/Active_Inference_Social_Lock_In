# Referee Report — *Changing a Networked Mind: Paradigm Dynamics as Structure Learning over a Hidden World*

**Venue:** IWAI 2026 (International Workshop on Active Inference)
**Reviewed artifact:** `paper/changing_networked_mind.pdf` (24 pp.) against `paper/changing_networked_mind.tex` and the experiment harness in `experiments/`, `scripts/`, `results/`.
**Method:** multi-agent review — 6 independent reviewer lenses → adversarial verification of every major/critical finding (a second agent attempts to *defend the paper*) → synthesis. 16 agents, ~903K tokens.

**How to read severities:** each major/critical finding was stress-tested. `severity` is the lens reviewer's initial rating; **`adjusted severity`** is the rating after a defender tried to refute it. The adjusted value is the one to trust. Of 9 initial *major* findings, the defense **downgraded 7 to minor** (real but overstated) and **upheld 2**; **0 of 36 findings were fully refuted**.

---

## 1. Summary of contribution

The paper models a scientific community as a population of bounded active-inference agents that perform **Bayesian structure learning** over a *directed* network of commitments, rather than updating scalar credences over a fixed menu of hypotheses. Its central representational move is that a single propagation operator T = (I − A)⁻¹ generates **two fields** on the same graph: a *conservatism* field κ = T·1 (the carry-over cost of revising a commitment, given everything that depends on it) and a *conviction* field U = T·u (the value a commitment inherits from cherished observations). From this the authors argue that exploration ceases to be a hand-installed temperature term and becomes the epistemic drive to *expand* the model; that lock-in requires conviction **and** disconnection together; that a forgetting factor ω turns lock-in from automatic into "earned"; and that model expansion can discover an unconceived dimension only once the world forces it. They implement this and run a single-agent dark-energy net plus a population tracking a moving world, recovering the Ω_m–Λ "banana" as a Schur-complement residue. The paper is candid that the **Kuhnian** reading is really Lakatosian, and reports a clean **negative result** (§4.4): a conservatism-gated revision rate does *not* recover the predicted belt-first/core-last "staircase," which it traces to an operator-set (data-independent) Fisher deposit.

## 2. Recommendation

**Accept with minor revisions.**

The conceptual contribution is coherent, well-positioned, and unusually self-aware; the empirical section is well-instrumented and reproducible, and the candidly-reported negative result is a *strength*, not grounds for rejection. The one genuine formal defect (the lock-in derivation routed through the wrong quantity) and a cluster of fixable overclaims in the abstract/title are all addressable with local edits and do not threaten the paper's validity at a workshop. The bar to clear is honest scoping, not new results.

## 3. Top strengths

- **Intellectual honesty is exemplary.** The central negative result (§4.4, the staircase null) is foregrounded in the abstract, demonstrated with real null *and* positive controls in code, and diagnosed (contested edges accumulate to ~720 from the operator-set Fisher deposit while true coupling differences are ≤ ±0.46) rather than merely asserted. A model of how to report a failed prediction.
- **Reproducibility is real, not gestured at.** Every headline number checked reproduces from the saved summaries (κ_GR = 10.5 / κ_Λ = 2.85 → ratio "3.7"; U_Λ = 1.19 > U_GR = 0.45; r(κ,U) = 0.25; Schur 0.00 → +0.50; forgetting precision 10816 → 516; endogenous γ 0 → ~1.0), and the repo ships seeded runs, null/positive controls, and an array-level golden-hash harness.
- **The core linear algebra is sound.** T = (I−A)⁻¹ as a finite Neumann series on a nilpotent DAG, the value-tilted Gaussian posterior and its CGF (Eq. 6), the Schur arithmetic of Eq. 14, and the forgetting steady state λU/(1−ω) all check out, several verified numerically.
- **Positioning is careful and sharp.** The distinctions from network epistemology (scalar credence vs. structure), Thagard's ECHO (undirected coherence net, no carry-over cost), the Causal Attitude Network, and prior single-agent BMR work are accurate and fair; the §1/§2 framing is genuinely incisive.
- **Scholarship is accurate.** Spot-checked high-value attributions (Albarracin 2022 quote, Holmes 2000, Boantza & Gal 2011, Nguyen's bubble/chamber distinction, Zollman 2010, the ARD trio, Friston & Penny BMR) all hold; no fabricated references or material mis-attributions.

---

## 4. Major issues (survived adversarial verification)

### 4.1 The lock-in derivation pins the wrong quantity — `locking-zpp-wrong-direction`
**Severity:** major → **major (confirmed)** · confidence 0.80
**Location:** §3.1 lines 463–467 and 480–483 (the Z″(λ) paragraph and the γ→Z″ link); reused §4.2 lines 968–981.

The paper derives evidential lock-in from the claim that holding a disconfirming sensory precision ρ_k near zero makes the susceptibility **Z″(λ) = UᵀΣU** "driven to zero" along that direction. But Σ is the **honest posterior covariance** (Eq. 6, p(x|o,G) = N(μ,Σ)). Removing a channel's precision ρ_k → 0 *removes information*, so by Loewner monotonicity of inversion **UᵀΣU is non-decreasing — it goes up, never to zero**. Verified numerically: for a latent read by one channel, UᵀΣU rises 0.26 → 2.25 as ρ_dis goes 100 → 0; a 20,000-trial random-geometry sweep gives min[Z″(0) − Z″(ρ>0)] = 0.0 (Z″ never falls); the component of ΣU along the gated direction *grows* 0.01 → 2.0, so "contributes nothing to ΣU" is false.

What actually vanishes is the channel's **mean-update gain** ρ_k·(H_k·ΣU) — the likelihood leverage that pulls the posterior mean back against the value tilt λΣU (0.995 → 0.0 as ρ_k → 0). **The lock-in conclusion is real and salvageable through the mean channel; only the named quantity is backwards.**

**Fix:** Re-route the mechanism through the channel's likelihood/Fisher precision ρ_k (the gain on the posterior *mean*), not the curvature Z″. State lock-in as "the disconfirming channel's mean-update gain → 0, so the value tilt is unopposed," and either drop the "Z″ → 0" claim or correct it to "Z″ along the gated direction *rises* but carries no evidence." **This is the single change most worth making before camera-ready** — it is the one genuine formal defect and it sits on the load-bearing mechanism.

### 4.2 The "more than a redescription" objection is named but not defeated — `single-strongest-objection-redescription`
**Severity:** major → **major (partial)** · confidence 0.72
**Location:** §4 opening ("must show three things to be more than a redescription"); §4.4; §5.

The authors *self-nominate* the staircase as THE discriminator between the structural and a tuned-scalar account (§2.3: "the structural account predicts a separation … the model's sharpest prediction") — and that is exactly the prediction that fails (§4.4: the κ-gradient gate is indistinguishable from a gradient-free uniform slowdown and a shuffled-κ gate). They then let the "more than redescription" framing stand without explicitly arguing the *surviving* wins clear the bar.

**Defender's correction (why this is `partial`, not fully `confirmed`):** the case does **not** rest on the lock-in result alone. §4.1's Schur-banana recovery (r: 0.00 → +0.50) is a second discriminating structural prediction that *succeeds* and is genuinely inexpressible in scalar-credence models (a scalar has no Schur complement), as is the endogenous-γ result with its deaf-but-honest control. So the objection is real at the level of argumentative *architecture*, but the reviewer overstated it by claiming the case rests on lock-in "alone."

**Fix:** Add a paragraph in §4 or §5 explicitly arguing that Schur recovery + structured two-lever lock-in + endogenous γ *jointly* clear "more than redescription," so the failed staircase is not left carrying that burden by default. Also benchmark the node-waking/expansion claim (§4.3) against existing BMR change-point demos, or soften it.

---

## 5. Minor issues & polish

### Notation / reproducibility traps  *(flagged independently by two lenses)*
- **Transpose convention** (`transpose-convention-paper-vs-code` / `conviction-field-transpose-mismatch`, conf. 0.80): paper writes κ = T·1 and U = T·u, but the released code computes the *transpose* (column-sums of T, and Tᵀu), with docstrings that literally say `U = Tᵀu`. Numerically equivalent under the SEM convention B[child,parent], but a reader reproducing "T·1" with the code's T gets κ_GR = **1.0, not 10.5**. State the index convention of T explicitly once, reconcile Eqs. 2–3 with the code, and fix the docstrings.
- **"Linearly independent" via Pearson r** (`linear-independence-vs-correlation`, nitpick): r = 0.25 measures centered collinearity, not linear independence (which here is trivially true since u is not constant). Keep the exact algebraic claim; describe r = 0.25 as *de-correlation*, optionally report the angle/cosine.
- **Dirty working tree** (`manifests-dirty-reproducibility`, nitpick): all 17 result manifests record `git.dirty=true` at run time, so the pinned SHA (e851d11) plus the golden-hash file — not the commit alone — defines reproducibility. Regenerate from a clean commit or note this in the appendix.

### Math labelling precision
- **κ ≠ Schur residue** (`kappa-equals-schur-residue-overclaim`, minor): Appendix C identifies the scalar κ (per-node descendant mass) with "the Schur-complement residue" (an off-diagonal induced edge, −0.5). Same precision geometry, different objects. Soften to "governed by the same Schur geometry."
- **Marginalize vs. condition** (`marginalize-vs-condition-two-readings`, minor): "two readings of one complement" equates *marginalizing a common cause* (App. C, Eq. 14) with *conditioning on a common effect / collider* (§4.1 banana). Both induce a precision coupling but they are dual, not identical. One sentence acknowledging explaining-away vs. common-cause would make the unification accurate.

### Argument / framing  *(body is honest; the abstract briefly isn't)*
- **"Two fields, one operator" is definitional** (`two-fields-definitional`, major → **minor/partial**): independence of κ = T·1 and U = T·u is *entailed* by the construction (T invertible ⇒ Tu ∝ T1 iff u ∝ 1), and u is hand-chosen so u ⊄ 1; r = 0.25 is a deterministic property of chosen inputs on a net "chosen to expose the geometry." The paper does *not* present this as a surprise (it states the one-line condition), so it survives only as a **rhetorical overstatement**: the abstract lists "the two fields come out independent" beside genuinely contingent results, and line 806–807's "come apart on a network we did not draw to order" contradicts the admitted hand-tuning. **Fix:** mark the independence as a *structural corollary* of non-proportional sources; let the load-bearing claim be the shared-operator commitment and the 2×2 lock-in result.
- **Kuhn vs. Lakatos** (`kuhnian-arc-not-delivered`, major → **minor/partial**): title/abstract foreground a Kuhn-coded banner ("Paradigm Dynamics"; "The Kuhnian arc is recovered as a consequence") while the substantive payload is Lakatosian, and §5 concedes identifying carry-over with incommensurability is "the framing's most contestable move." Both the Lakatos lean and the staircase null are pre-announced in the abstract, so this is an **emphasis/substance mismatch**, not a self-contradiction. **Fix:** soften "the Kuhnian arc" to "the lock-in-and-release arc," or re-subtitle toward Lakatos/problemshift.
- **Toy-to-real leap** (`toy-to-real-community-leap`, minor): phlogiston/dark-energy nets are used as if the model explains them, but weights are hand-wired and the "Nothing here is evidence about cosmology" disclaimer arrives only at the end of §4.4. The banana is a generic two-parents-one-child Schur artifact. **Fix:** move the disclaimer to the first use, soften "recovered" to "reproduces the qualitative shape of."
- **Exploration "no longer free"** (`exploration-is-expansion-equivocation`, minor): the implementation reintroduces a free Poisson rate r (swept in Table 1) — ε was *relabeled*, not eliminated. Acknowledge the rate is still free and locate the principled gain in the *content* of the proposal, or tie the rate to expected free energy.
- **Conservatism is dynamically inert** (`honest-negative-undercuts-its-own-mechanism`, minor): §4.4 reveals κ "never enters the update" in the runnable model, so one of the two headline fields is, as implemented, a static descriptor. State plainly in the discussion that making κ dynamically load-bearing is the central open problem, not a peripheral caveat.

### Novelty / positioning
- **Multi-agent structure learning omitted** (`novelty-multiagent-structure-learning-omitted`, major → **minor/partial**): the absolute "no one has run structure learning across a population" (line 201) is falsifiable by an ML-adjacent referee (FedDAG, distributed multi-agent causal-network learning, PEnBayes). The paper's *actual* novelty hook (structural divergence among bounded, value-laden agents) is well-scoped and untouched by this — but the §1 sentence is an overclaim and the literature is uncited. **Fix:** scope to "a population of bounded, value-laden agents as a model of paradigm dynamics" and add a one-clause cite-and-distinguish (those optimize ground-truth recovery of a single true graph, not motivated persistence over divergent paradigms).
- **CAN is undirected** (`can-directed-overstated`, minor): "now realised on the same directed graph" overstructures continuity with CAN (Dalege 2016), an undirected Ising/MRF. Borrow the *posture* (connectivity = attitude strength), not the formalism.
- **Structural incommensurability hedge** (`structural-incommensurability-not-yet-demonstrated`, minor): mirror the §1 hedge ("still to be earned … expresses paradigm difference in posterior means") into the abstract, where it is briefly dropped.
- **Unify the directedness contrast** (`echo-can-directedness-unified-contrast`, nitpick): both ECHO and CAN are undirected; stating once that neither supports a directed propagation operator makes the novelty load-bearing in one stroke.

### Empirical wording
- **"~1/λ" overclaim** (`poisson-inverse-lambda-overclaim`, minor, conf. 0.85): the data (22.8 → 3.7 vs. 1/λ = 50 → 1) and the paper's own figure diverge — the delay is a Poisson waiting time on a ~7-step deterministic floor, horizon-truncated at low λ (28% never wake). Replace "~1/λ" with that accurate description; the qualitative point survives.
- **"ω=1 stuck" is metric-dependent** (`omega1-stuck-metric-ambiguity`, minor): the same lever-free ω=1 cell *reaches* the final theory under the tracking 2×2's nearest-end-theory readout. Add a clause clarifying "stuck" = fraction-on-current-true-theory.
- **Positive confirmation** (`negative-result-cause-confirmed`): the §4.4 staircase null *and* its stated cause (operator-set Fisher deposit) are both demonstrated by real controls, not asserted (controls lag −45/−45/−45; contested edges → ~720 vs. true ≤ ±0.46). Optionally cite the 720-vs-0.46 number explicitly in §4.4 to make the diagnosis concrete.

### Scholarship hygiene
- **5 orphan references** (`orphan-references`, minor, conf. 0.97): `jonard2025`, `griffiths2011ibp` (IBP), `williams2010pid`, `dacosta2020`, `acemoglu2011` are never cited in-body. Engage each in one sentence (IBP as the nonparametric prior over expansion; Acemoglu–Ozdaglar as the opinion-dynamics baseline; Jonard 2025 as the closest premature-convergence foil) or delete.
- **Nguyen year/metadata** (`nguyen-year-metadata`, minor): the entry pairs 2020 volume/pages (Episteme 17(2):141–161) with year "2018." Change the displayed year to 2020.
- **Jonard "et al."** (`jonard-et-al-author-list`, nitpick): spell out the three authors (Jonard, Reijula & Marengo) in the reference list.
- **MacKay 1992 for ARD** (`mackay-bayesian-interpolation-ard`, nitpick): "Bayesian interpolation" is the evidence-framework predecessor, an imprecise anchor for ARD specifically; lean on Neal 1996 + Tipping 2001.
- **Federated precision claim** (`federated-precision-claim-verbatim`, minor, conf. 0.50): "preference conflicts reducing group precision below single-agent levels" (citing friston2023federated, waade2025) is a sharp quantitative assertion not clearly verbatim in either source. Point to the specific result or soften to "naive aggregation can degrade collective precision."

### Clarity / presentation  *(rewriting-for-altitude, not restructuring)*
- **"From gating to γ" too compressed** (`gating-to-gamma-too-compressed`, major → **minor/partial**): the §3.1 chain (Eq. 4 → 5 → 6 → 7) is the conceptual hinge but ~50 lines of unbroken high-density prose. Add a plain-language punchline *before* the math: "value can shift *where* the agent believes (the mean), but only evidence can change *how sure* it is (the variance); when value silences a channel, no evidence on that channel can reach the agent." Consider moving the endogenous indicator w_k = exp(−g|U·H_k|) (currently §4.2) up next to Eq. 7.
- **"Order parameter" overloaded** (`order-parameter-overload`, major → **minor/partial**): the term names both γ (line 470) and m_S (line 758), and γ is then re-cast as a "phase boundary." Reserve "order parameter" for m_S; call γ the "lock-in/control parameter" (the paper already uses "lock-in parameter" at line 343).
- **Under-sized multi-panel figures** (`multipanel-figs-too-small`, major → **minor/partial**): Fig 8 is a 2-up of two multi-panel composites (~2 cm per panel, ~5pt embedded titles); Fig 6b at 0.32 width crams three category labels. Split Fig 8 into two figures, strip the matplotlib auto-titles (they duplicate captions), re-export ≥10pt, widen Fig 6b. The numbers are all in the text, so this is polish, not comprehension-blocking.
- **Undefined symbols** (`undefined-symbols-h-lambda2`, minor): define h (information-form vector) where it first appears (§4.2 line 886; currently only glossed at line 1052) and gloss λ₂ as "algebraic connectivity (Fiedler value)" on first use in Table 1.
- **§4.4 "three dials" vs. Table 1's five** (`abstract-dials-mismatch`, major → **minor/confirmed**): a direct, definite-count self-contradiction ("the three dials (ω, λ, sensing resolution)" vs. five swept dials, "sensing resolution" being Appendix-A-only). One-line edit: "the dials of Table 1 are swept, not inferred."
- **Over-packed captions** (`captions-overpacked`, minor): captions run 8–12 lines re-arguing the result with equation/section cross-refs. Trim to what-it-shows / what-each-element-means / one-line takeaway.
- **Aphoristic slogans** (`aphoristic-slogans-impede`, minor): keep the best one or two; place each italic claim *after* its supporting argument, not as a flourish that pre-empts it.
- **Appendix fit** (`appendices-uneven-fit`, minor): Appendix C (Schur) is load-bearing; Appendix A (coarse world) is an orphan supporting no main-text claim; Appendix B largely re-illustrates Fig 2. For a workshop, compress or cut A and B.
- **Over-stuffed abstract** (`abstract-overstuffed`, minor): a single 25-line block front-loading undefined jargon ("Schur residue"). Restructure into shorter sentences; lead with the one-sentence claim, then the two-field result, then headline findings, then the honest negative.
- **Forward-referencing** (`notation-forward-refs`, minor): reduce "precise where it bites" forward-deferrals to at most one; define each quantity at first substantive mention.

---

## 6. Per-lens verdict

| Lens | One-line takeaway |
|---|---|
| **Mathematical / formal correctness** | Core linear algebra sound and several non-trivial claims verified numerically; the one real defect is the lock-in derivation pinning Z″ = UᵀΣU, which moves the *wrong way* under ρ_k → 0 (fix via the mean-update gain). |
| **Core thesis / argumentative coherence** | Coherent and unusually self-aware, but the abstract over-promises: "two fields independent" is largely definitional and the self-nominated discriminating prediction (the staircase) is the negative one. |
| **Related work / novelty / positioning** | Careful and well-hedged; defensible novelty, but the absolute "no one has run structure learning across a population" overlooks federated/distributed causal-discovery and needs scoping. |
| **Empirical results / reproducibility** | Exceptionally well-instrumented and honest — every checked number reproduces, controls are in code, the negative result's cause is *demonstrated*; only notation / "~1/λ" / dirty-tree nitpicks. |
| **Clarity / structure / presentation** | Bones are good; aphoristic, compressed prose and under-sized multi-panel figures lose a mixed audience at the load-bearing moments — a rewriting-for-altitude job, not restructuring. |
| **Citation / scholarship accuracy** | Substantively accurate with no fabrications or mis-attributions; defects are hygiene — five orphan references, a Nguyen year/metadata mismatch, an "et al." that hides two authors. |

---

## Appendix — finding ledger

36 findings, 0 refuted. Severity shown as *initial → adjusted* where the two differ.

| ID | Lens | Sev. (init→adj) | Location | One-line |
|---|---|---|---|---|
| `locking-zpp-wrong-direction` | math | major | §3.1 L463–483; §4.2 L968–981 | Lock-in pins Z″=UᵀΣU; gating ρ_k→0 raises it, not lowers it. |
| `kappa-equals-schur-residue-overclaim` | math | minor | App. C L1287; §4.1 L866 | Scalar κ ≠ off-diagonal Schur fill-in; soften to "same geometry." |
| `transpose-convention-paper-vs-code` | math | minor | Eqs. 2–3 vs code L113–123 | Code computes Tᵀ; "T·1" gives κ_GR=1.0 not 10.5. |
| `marginalize-vs-condition-two-readings` | math | minor | App. C L1280; §4.1 L862–877 | Common-cause marginalization ≠ collider conditioning. |
| `linear-independence-vs-correlation` | math | nitpick | §3.1 L421; §4.1 L843 | r=0.25 is de-correlation, not linear independence. |
| `two-fields-definitional` | argument | major→minor | §2; Eqs. 2–3; §4.1 | Field independence is built-in; rhetorical overclaim only. |
| `kuhnian-arc-not-delivered` | argument | major→minor | Title/abstract vs §5 | Payload is Lakatos; Kuhn banner is emphasis mismatch. |
| `single-strongest-objection-redescription` | argument | major | §4; §4.4; §5 | "More than redescription" not explicitly won; staircase is the negative. |
| `toy-to-real-community-leap` | argument | minor | §1, §4.1–4.4 | Toy-net → real-community leap; disclaimer arrives late. |
| `exploration-is-expansion-equivocation` | argument | minor | §2; §3.2; §4.3 | "Exploration no longer free" but Poisson rate r is swept. |
| `honest-negative-undercuts-its-own-mechanism` | argument | minor | §4.4; §3.4 | κ is dynamically inert in the runnable model; elevate as open problem. |
| `novelty-multiagent-structure-learning-omitted` | novelty | major→minor | §1 L201; §2 L293 | Absolute "no one" overlooks FedDAG/distributed causal discovery. |
| `can-directed-overstated` | novelty | minor | §2 L304–308 | CAN is undirected Ising; "same directed graph" overstates continuity. |
| `decorative-citations-unengaged` | novelty | minor | bib vs body | IBP / Acemoglu / Jonard cited-but-unused; engage or drop. |
| `structural-incommensurability-not-yet-demonstrated` | novelty | minor | abstract vs §1 L222 | Mirror the §1 "still to be earned" hedge into the abstract. |
| `echo-can-directedness-unified-contrast` | novelty | nitpick | §1 L147–193 | Engagement accurate; unify the directedness contrast in one line. |
| `poisson-inverse-lambda-overclaim` | empirical | minor | §4.3, Fig 8 | "~1/λ" contradicted by data (22.8 vs 50); it's a floored waiting time. |
| `conviction-field-transpose-mismatch` | empirical | minor | Eqs. 1,4,9 vs code L113–124 | Same transpose issue, found independently by the empirical lens. |
| `omega1-stuck-metric-ambiguity` | empirical | minor | §4.2 vs Fig 5 | "ω=1 stuck" is metric-dependent; clarify "stuck"=fraction-on-current. |
| `manifests-dirty-reproducibility` | empirical | nitpick | results/*/manifest.json | All 17 manifests `dirty=true`; golden-hash defines reproducibility. |
| `negative-result-cause-confirmed` | empirical | minor (✓ positive) | §4.4, Fig 9 | Staircase null + cause are demonstrated by real controls (720 vs 0.46). |
| `abstract-dials-mismatch` | writing | major→minor | §4.4 L1084 vs Table 1 | "Three dials" vs five swept; one-line fix. |
| `order-parameter-overload` | writing | major→minor | L470, L758, L784 | "Order parameter" names both γ and m_S; rename γ. |
| `gating-to-gamma-too-compressed` | writing | major→minor | §3.1 L432–483 | Conceptual hinge is densest passage; add a plain punchline first. |
| `multipanel-figs-too-small` | writing | major→minor | Figs 5,6,8,9 | Fig 8 & 6b under-sized as placed; split, strip auto-titles, ≥10pt. |
| `undefined-symbols-h-lambda2` | writing | minor | §4.2 L886; Table 1 | Define h and gloss λ₂ (Fiedler value) on first use. |
| `captions-overpacked` | writing | minor | Figs 1,2,3, all results | Captions re-argue the result; trim to show/mean/takeaway. |
| `aphoristic-slogans-impede` | writing | minor | §2 L269; §5 L1114; §4.1 L869 | Italic slogans land before their premises; place after the argument. |
| `appendices-uneven-fit` | writing | minor | App. A/B/C | App. A is an orphan; App. B duplicates Fig 2; compress for a workshop. |
| `abstract-overstuffed` | writing | minor | abstract L88–113 | 25-line block front-loads undefined jargon; restructure. |
| `notation-forward-refs` | writing | minor | §3 throughout | Heavy forward-referencing forces a read-it-twice structure. |
| `orphan-references` | citations | minor | bib L1329,1393,1408,1426,1507 | 5 entries never cited in-body; engage or delete. |
| `nguyen-year-metadata` | citations | minor | bib `nguyen2018` | 2020 volume/pages tagged year 2018; change to 2020. |
| `jonard-et-al-author-list` | citations | nitpick | bib `jonard2025` | "et al." hides a 3-author work (Jonard, Reijula & Marengo). |
| `mackay-bayesian-interpolation-ard` | citations | nitpick | §3.2; bib `mackay1992` | "Bayesian interpolation" is imprecise for ARD; lean on Neal/Tipping. |
| `federated-precision-claim-verbatim` | citations | minor | §1 background | "Group precision below single-agent levels" not clearly verbatim in sources. |

---

*Generated by a multi-agent review (6 lenses + adversarial verification + synthesis) on 2026-06-10. Severities reflect post-verification (`adjusted`) ratings. Line numbers reference `paper/changing_networked_mind.tex`.*
