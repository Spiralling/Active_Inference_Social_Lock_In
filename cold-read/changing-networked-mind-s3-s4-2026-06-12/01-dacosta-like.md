---
type: cold-read-reaction
parent: "[[README]]"
draft: "Changing a Networked Mind §3–§4"
reader: dacosta-like
based_on: "a reader like Lancelot Da Costa (mathematical active inference theorist)"
scene: fresh-careful
attention_budget: high
read_completion: "complete — every sentence, with margin notes"
outcome: engaged
dispatched_at: 2026-06-12T00:00:00+02:00
status: complete
tags: [cold-read, reaction]
---

# Rigorous active-inference theorist — cold read of "Changing a Networked Mind §3–§4"

## The scene

Saturday, home office, two hours blocked, second coffee going cold on the left. Rain. I actually want someone to do multi-agent structure learning right — I've been saying this community needs it. Pad and pen out. I'm going to read this carefully and I am going to write down every place the algebra doesn't close.

## Reading it (stream of thought)

**3.1 opens well, up to a point.**

The phlogiston motivating example is good scene-setting, doesn't bother me. $G=(V,E,\Pi)$ — fine, this is a precision-weighted DAG, nothing controversial. The identification of $(I-A)^{-1}$ as the total-effects matrix is correctly attributed to Wright and Bollen, and the link to Katz-Bonacich is legitimate. I'm nodding. The sensory channel notation ($\rho_k$) is introduced without full specification of what $H_k$ is — is this a row of the observation matrix? I assume so, linear-Gaussian, fine, but write it down.

The propagation operator $T = (I-A)^{-1}$ is stated to be finite "on a DAG." That's correct — the Neumann series $\sum_k A^k$ terminates because a DAG has no cycles so $A$ is nilpotent — but this deserves one line saying so explicitly, because the convergence of the Neumann series for general $\|A\|<1$ is a different argument than nilpotency on a DAG. Readers may confuse them. Small thing but I'd note it.

$\kappa = T\mathbf{1}$ as revision cost — this is the claimed "recognition rather than positing" of conservatism. The logic is: centrality in the Katz sense measures how much mass downstream would have to re-equilibrate. That's a reasonable interpretive gloss, but it's not derived from the free-energy objective here. It's a property of the graphical model, and the connection to what the *agent* actually minimizes step by step (VFE) is not made explicit. I'm writing in the margin: "is $\kappa$ a prediction of VFE minimization or just a property of the prior graphical model? If the latter, what stops an agent doing something cheaper?"

**3.2 — this is where I slow down hard.**

Eq. (3): $\mathcal{F}[q] = D_{KL}[q\|p(x|o,G)] - \lambda\,\mathbb{E}_q[U^\top x]$. Fine — this is EFE with a utility term, the KL-regularized utility maximization form from Todorov and Ortega/Braun. The solution $q_\lambda(x) \propto p(x|o,G)\,e^{\lambda U^\top x}$ is correct by direct calculation in the Gaussian case, no issue.

They say the tilt is a "pure mean shift, displacing commitments by $\lambda\Sigma U$ and leaving the covariance untouched." Let me check this. If $p(x|o,G) = \mathcal{N}(\mu, \Sigma)$ then $q_\lambda(x) \propto \mathcal{N}(\mu,\Sigma)\cdot e^{\lambda U^\top x}$. Completing the square: $q_\lambda = \mathcal{N}(\mu + \lambda\Sigma U, \Sigma)$. Yes, that's right. Good. Covariance unchanged, mean shifted by $\lambda\Sigma U$. This is correct.

Now the conviction gate: $w_k = \exp(-g|U \cdot H_k|)$. Here is my problem. This expression appears in step 1 of the simulation loop (§3.5, "Observe"), described as "channel weights on disconfirming rows are set endogenously by the conviction gate." But where does this expression come from? The stated objective is Eq. (3). I see no derivation — not here, not referenced to an appendix — that produces $w_k = \exp(-g|U\cdot H_k|)$ as the optimal channel weighting under that objective. The text says "As conviction drives a disconfirming channel toward $\rho_k \to 0$ that gain vanishes," which is a consequence claim, not a derivation. It's saying: IF conviction gates the channel to zero precision THEN the channel has no effect. Yes, trivially. But why does conviction gate it to zero in that particular functional form? The exponential with parameter $g$ looks like it was chosen for convenience, not derived. This is the central mechanism of §4.1 and it is mechanistically unjustified. I'm underlining this three times.

The lock-in parameter $\gamma$ (Eq. 4) — the fraction of disconfirming sensory precision "gated by the core." The numerator has an indicator $\mathbb{1}[\rho_k\ \text{gated by the core}]$. What is the formal definition of "gated by the core"? Is this a threshold on $w_k$? On $|U\cdot H_k|$? This is not stated. The expression looks like a definition, not a derived quantity, but I cannot evaluate it without knowing what the indicator means.

The paragraph ending §3.2 says "A bias, not a resistance." I know what they mean by this — the distinction between mean-shift (bias) and precision-change (resistance/entrenchment) — and it's a legitimate and interesting distinction. But if the gate $w_k$ *lowers sensory precision*, then it IS a precision effect, not just a mean effect. The claim that motivated reasoning in this regime is "wishful, not dogmatic" with the gate in the loop seems to be simultaneously true and false depending on whether you count the gate as part of the agent's internal dynamics or as an external weighting. This distinction needs to be made more carefully. I'm not sure the authors have themselves sorted out whether the gate is inside or outside the VFE minimization.

**3.3 — structurally honest.**

The asymmetry between reduction (closed-form, Savage-Dickey) and expansion (requires new inversion, not closed-form) is correctly described. The Friston et al. 2018 BMR citation is appropriate. The AIC/ARD parameterization via $\alpha_e$ is standard. The Poisson arrival rate for expansion candidates — I accept this as a reasonable modelling commitment, and they acknowledge it as such. The Stanford problem of unconceived alternatives is correctly named. The appendix treatment of the rate as a belief is a nice touch, though I'll want to see if it's actually done or just promised.

One issue: they say the Bayes factor for expansion can be evaluated from "the residual the old paradigm absorbed... visible as a coherent pattern in prediction error; when a Poisson proposal arrives, its *direction* is read from that pattern's leading eigenvector." This is operationally described in the simulation loop (§3.5 step 4) but is it derived from the stated model? The leading eigenvector of prediction error as a proposal direction sounds like a heuristic, not a consequence of VFE minimization over $G$. I want a citation or a derivation, not a phrase.

**3.4 — fine, nothing to object to.**

Exponential forgetting $\Pi \leftarrow \Pi_0 + \omega(\Pi-\Pi_0)$ is the standard Bayesian forgetting/drift prior, correctly described. The saturation claim (tilt saturates at $\lambda U/(1-\omega)$) — let me think. If precision accumulates as $\Pi_t = \Pi_0 + \sum_{s=0}^{t}\omega^{t-s}\delta_s$, then in steady state with constant per-step input $\delta$ this is $\Pi_0 + \delta/(1-\omega)$. Similarly the mean tilt saturates. Yes, that formula is right. And the competition between saturated conviction tilt and saturated evidence is the right way to frame the threshold. This paragraph earns my trust.

**3.5 — the loop.**

The simulation loop is clearly and explicitly stated, which I appreciate. Six steps, no hand-waving about "updating beliefs." But:

Step 1 ("Observe"): channel weights set by $w_k = \exp(-g|U\cdot H_k|)$. Again — this is not derived, just restated. The parameter $g$ is introduced here with no prior on it, no sensitivity analysis announced, nothing. It's a free parameter with enormous leverage on the main result of §4.1 and it's just... set by hand. Same problem as §3.2.

Step 3 ("Crisis check"): "Both protective belt edges scoring $\Delta F + \lambda_i\Delta U > 0$ is a *crisis*." The crisis criterion involves $\lambda_i$ (individual conviction) and $\Delta U$ (conviction-value change of the prune). Is $\Delta U$ closed-form under BMR? They claim both $\Delta F$ and $\Delta U$ are "equally closed-form" and "the Savage-Dickey identity already exhibits" the latter. This is plausible for the linear-Gaussian case but I want to see it — it's not obvious that the reduced posterior's mean has a clean closed-form expression derivable from the BMR identity in general. This is exactly the kind of claim that gets quietly wrong in implementation.

The footnote "An appendix derives both checks from the single tilted objective" — this appendix is doing a lot of work and I cannot evaluate whether the derivations are honest without seeing it.

**Section 4 — the theorems.**

"Pooling is a contraction" — yes, DeGroot dynamics with fixed content-independent weights on a connected graph converge to consensus. This is the classical result. The paper states it correctly and the attribution to DeGroot 1974 is right. The identification of the information-form fusion as exactly DeGroot dynamics is the right move.

"Gaussian conflict resolves by compromise" — the citation chain (Dawid 1973, O'Hagan 1979, O'Hagan and Pericchi 2012) is correct. The claim is that under Gaussian posterior, conflict between two sources produces confident compromise, not a fat-tailed or bimodal result. This is exactly right and well-known in Bayesian robustness theory. Good.

But here is my frustration: the paper calls these "two classical theorems" and says "Two classical theorems... say the sweep could not have come out otherwise." These are theorems *about* the regime, invoked to explain simulation results. This is fine and correct usage. But in §4.1 the result "92% convert at inter=0.005" is presented as a consequence of these theorems. Is it? The theorems say consensus must happen eventually on a connected graph, and that Gaussian conflict resolves by compromise. They do not say the transition is sharp at inter≈0.005, or that 92% is the right number at that threshold, or that the gate parameter $g$ doesn't matter above that threshold. The sharpness of the transition, the 92%, the "barely changes the picture" for doubling conviction — these are simulation numbers, not theorem consequences. The paper is blurring "the sweep is consistent with what the theorems predict" with "the theorems explain the sweep." The former is true; the latter overstates.

§4.2 "incommensurability is a property of the pooling protocol" — this is the best result in the paper. The Genest-Zidek 1986 citation is correct and well-chosen. The distinction between delta-pooling (absence read as confident zero) and abstention-pooling (absence skipped in the average) is a genuine conceptual contribution and is stated clearly. I believe this result. It has the shape of a real theorem: the protocol determines the support of the pooled belief, and a concept outside the support cannot survive averaging. This could probably be stated as a proper proposition with proof, and I would like that.

**Table 2.** The table caption is doing a lot of work. "None of which opens the consensus engine" — I want a clearer statement of what "opens" means operationally. The cell entries are very dense and I'm not confident I'm reading the Hawkes process result ("waking dispersion 64→2 steps") correctly. These are interesting mechanisms but they feel under-reported in the main text.

**Robustness paragraph.** "The arc is robust across forgetting and observation noise ($\omega\in[0.9,1.0]$, $\sigma_o\in[0.25,1.0]$)." The forgetting range only covers the last 10% of possible forgetting rates. Why? A community with $\omega=0.5$ forgets half its accumulated precision every round — is the model qualitatively different there? The narrow range is suspicious and unexplained.

## After (the collapse)

- **How far I actually read:** Every sentence. Full read. Pad has about two pages of margin notes.

- **What I think they want from me / my takeaway:** They want me to accept that they've given a principled, derivation-honest model of paradigm change and shown (a) that Gaussian pooling is a null model that provably can't produce Kuhnian dynamics, and (b) that what escapes the null requires specific structural features (gate, abstention protocol). The takeaway they're pushing is: the simulation confirms classical theorems plus identifies the protocol-sensitivity of discovery survival.

- **Did I believe it? What lost me:** I believe the architecture in broad strokes. The Gaussian regime is correctly scoped, the theoretical guarantees invoked are real and correctly attributed, the forgetting mechanics are sound, the abstention-pooling distinction is sharp and I think it's right. What I do not believe — or rather, what I cannot evaluate — is the conviction gate $w_k = \exp(-g|U\cdot H_k|)$. This expression is introduced in §3.2, used in the simulation loop as if it were a consequence of Eq. (3), and never derived. It has a free parameter $g$ that is presumably swept but I don't see the sensitivity analysis reported for §4.1. If the 92% conversion threshold depends on $g$, the theorem framing of the result is wrong. Additionally: the "leading eigenvector as expansion direction" in step 4 is similarly a heuristic that could have been a derivation. The crisis check's claim that $\Delta U$ is closed-form via BMR is stated but not exhibited. These three items are the paper's derivational debts and they are not small.

- **What I'd actually do next:** Major revision. The contribution is real — the abstention-pooling result especially is worth publishing. But I want: (1) a proper derivation of the gate $w_k$ from the stated objective (Eq. 3), or an explicit acknowledgement that it is a modelling commitment with a corresponding sensitivity analysis showing results are qualitatively stable across $g$; (2) a displayed closed-form expression for $\Delta U$ under BMR, or a proof sketch; (3) a proposition with proof — or at minimum a proof sketch — for the abstention-pooling survival result; (4) an explanation of why $\omega \in [0.9, 1.0]$ is the right range to test and what happens outside it; (5) cleaner separation of "simulation consistent with theorems" versus "theorems imply simulation results." The paper is currently writing the latter when it means the former.

- **Would it land differently if I were fresh / less busy?** No — these are mathematical questions, not attention-budget questions. I'd find the same gaps on any re-read.

---

**Outcome: engaged, full read — one genuine contribution found (abstention-pooling), one central mechanism (the gate) never derived from the stated objective, overstated theorem-to-simulation inference throughout.**
