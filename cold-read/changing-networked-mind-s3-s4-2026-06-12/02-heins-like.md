---
type: cold-read-reaction
parent: "[[README]]"
draft: "Changing a Networked Mind §3–§4"
reader: heins-like
based_on: "a reader like Conor Heins (multi-agent active inference / belief sharing)"
scene: distracted-typical
attention_budget: medium
read_completion: "Abstract fully, figures and bold claims in §4 closely, §3 partially (jumped around, skipped equation blocks)"
outcome: engaged
dispatched_at: 2026-06-12T00:00:00+02:00
status: complete
tags: [cold-read, reaction]
---

# Multi-agent active inference practitioner — cold read of "Changing a Networked Mind §3–§4"

## The scene

Regional train, laptop on tray table, Slack nagging about a deadline. Opened this because someone in the group mentioned it. Midway through a coffee I didn't finish. The title hooked me but I've been burned by philosophy-of-science papers that bury a two-line model in four pages of Kuhn. I'm going figures-first and if nothing grabs me in five minutes this goes in the "maybe later" folder.

## Reading it (stream of thought)

Abstract. OK. "Conflict resolves into confident compromise." That's... wait, that's basically what we showed, isn't it? Belief sharing is a curse because pooling drives you to consensus. But they're framing it as a theorem of the regime. Fine, let me see if they actually prove it or just assert it.

Skip to figures. Where are they.

Figure 1 — two fields on a paradigm network. Core/belt structure, revision cost versus conviction field. OK that's a nice schematic. The T1 versus Tu decomposition is clean. I like that they're explicit that these are independent vectors. Actually — that's a non-obvious point. If u is proportional to 1 they collapse. That's worth keeping.

Figure 2 — population model. Trust graph, SBM, amber channels, structural proposals. OK this is the architecture I care about. "What passes along trust channels is a structural proposal" — they're sharing graph structure, not just beliefs. That's the extension I'm actually curious about. Does it work or does it just average the graphs away too?

Now I want to see the results before reading §3. Jumping to §4.

"Pooling is a contraction." Yes — DeGroot dynamics, we know this. "Gaussian conflict resolves by compromise." Dawid 1973, O'Hagan. Yes, yes, I know. So they're treating their whole model as a null regime, which is honest. I actually respect that framing more than I expected to. They're not claiming to explain paradigm shifts, they're characterizing *why the defaults fail*.

Figure 3 — the inter-community threshold. 92% convert at inter = 0.005. That's two orders of magnitude weaker than internal trust. And it's a sharp boundary. OK this is a real result. The variance spike at the boundary is the right diagnostic — that's not noise, that's a bifurcation signature. Good. They know what they're doing numerically.

"An agent can silence its own instruments; it cannot silence its neighbours'." That's a clean sentence. That's the one I'd quote. The gating mechanism is local but evidence propagation is not. So the gate is useless once there's any social contact. Makes sense — the gate operates on the agent's own channel weights, but incoming posterior information from trusted peers arrives precision-weighted by trust, not by the receiver's conviction. Yeah.

Figure 4 — abstention pooling versus delta pooling. OK *this* is the result I needed. Staggered survival is zero under delta pooling at every rate. Under abstention pooling it tracks discovery. The framing as "absence read as maximally certain it is zero" versus abstention — that's the Genest & Zidek point and I've been wondering if anyone had actually run this in an active inference context. They have. Green line tracks the dashed line (discovery fraction) pretty cleanly.

Wait — 32 seeds per cell. That's fine for a sweep this size. Are they seeded? Abstract says all code ships seeded. OK.

Table 2 — four escapes. Crisis-to-discovery 55→29 steps with adequacy-inferred rate. Social excitation (Hawkes) waking dispersion 64→2 steps, staggered survival 0→0.88–0.95. That's the big one. Synchrony outruns the averaging. So the conclusion is: you can't fix the averaging by going through it, you have to route around it temporally. That's interesting and I think it's right.

Going back to §3 now because I want to check the fusion step. §3.5.

"information form counterpart of trust-weighted averaging in opinion dynamics (DeGroot 1974)"— yes, precision-weighted pooling, standard. "what is not standard is that structural edits spread the same way" — OK so graph topology propagates as proposals scored by Bayes factor at the receiver. That's the bit I want to understand.

Reading step 5 more carefully. "Posteriors and structural edits are pooled across the trust graph... Whether agents pool over all dimensions or only over those they jointly represent is a protocol choice with consequences (§4.2)." Right, that's the delta vs. abstention distinction. So the mechanism is explicit.

Equation 3 — the tilted objective. I've seen this form before (Ortega & Braun), nothing exotic. The lock-in parameter equation 4 — fraction of disconfirming precision silenced. OK. The gate formula w_k = exp(-g |U · H_k|) is clean. Continuous relaxation of binary channel silencing, which is sensible.

Checking §3.3 on expansion — Poisson arrivals, Bayes factor accept test. The direction read from residual leading eigenvector. That's a nice touch. The residual from the pruned model pointing toward the missing dimension. They're claiming the residual is informative about what's missing. I'd want to see that in a figure but I don't see one. Maybe the appendix.

Train's slowing. Skip robustness paragraph — looks like a parameter table I can read later if I'm using this setup. The "null and positive controls" claim — I'll trust that until I look at the code.

## After (the collapse)

- **How far I actually read:** Abstract in full, §4 closely (all figures, bold sentences, Table 2), §3.5 in detail, §3.1–§3.4 partially (equations skimmed, prose read at half-speed for the key mechanisms).
- **What I think they want from me / my takeaway:** They want to show that the Gaussian/pooling defaults are provably a null regime for paradigm change, characterize the boundary conditions, and locate two specific failure modes (gate corrosion by social contact, delta pooling killing staggered discovery). The takeaway I actually walked away with: abstention pooling versus delta pooling is the operationally important distinction, and the social excitation result is the strongest finding.
- **Did I believe it? What lost me:** I believe the pooling theorem — that's just DeGroot, they're right to frame it as inherited. I believe Figure 3, the threshold result is plausible and the variance spike is good evidence it's a sharp boundary. Figure 4 I believe. What I'm less sure about: the expansion direction from residual eigenvector — that's doing real work and there's no figure for it. The adequacy-inferred rate in Table 2 is compressed into one number, I'd want to see the distribution. The appendix with the posterior rate treatment sounds interesting but it's not in front of me.
- **What I'd actually do next:** Pull the code, check the seed setup and the abstention pooling implementation. The delta/abstention distinction is directly relevant to work we're doing on federated inference. Probably mention to the group. If the code is clean I'd cite it in the methods section of our current paper and reference Figure 4 specifically.
- **Would it land differently if I were fresh / less busy?:** Yes — I'd actually read §3.1–§3.3 linearly and probably appreciate the path-analytic framing more than I did skimming it; the T = (I-A)^{-1} propagation operator is a clean unification I half-appreciated but didn't sit with.
