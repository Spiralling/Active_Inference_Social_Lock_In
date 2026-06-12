---
type: cold-read
draft: "Changing a Networked Mind — §3 (The model) + §4 (Results)"
draft_path: "C:\\GitHub\\Paradigm_Shift_Act_Inf\\paper\\changing_networked_mind_v3.tex"
audience: "IWAI 2026 reviewers + adjacent philosophy-of-science readers"
intended_ask: "The story is open-endedness: Gaussian mean aggregation in a community leads to non-divergence/closedness, and you need other ways of propagating information. Building toward open-ended multi-agent systems and open-ended science. And — the main thing — the reader believes the model construction is proper."
readers: 3
dispatched_at: 2026-06-12T00:00:00+02:00
completed_at: 2026-06-12T00:00:00+02:00
status: complete
tags: [cold-read]
---

# Cold read: "Changing a Networked Mind" §3–§4

**Intended audience:** IWAI 2026 reviewers and adjacent philosophy-of-science / multi-agent readers.
**What you wanted them to take away:** The open-endedness story (Gaussian averaging forecloses divergence; other propagation is needed), and — main thing — belief that the model construction is proper.

## The panel
| # | Reader | Based on | Scene | Attention | Outcome |
|---|--------|----------|-------|-----------|---------|
| 01 | [[01-dacosta-like]] | a reader like Lancelot Da Costa (mathematical AIF theory) | fresh-careful | high | engaged (full read, 2 pages of margin notes) |
| 02 | [[02-heins-like]] | a reader like Conor Heins (multi-agent AIF / belief sharing) | distracted-typical | medium | engaged (~80%, figures-first) |
| 03 | [[03-zollman-like]] | a reader like Kevin Zollman (network epistemology) | busy-hostile | low | engaged (read through §4.2 closely — did not bounce) |

## Intended vs. received

**The story landed on all three readers — including the hostile one.** Each reconstructed the intended open-endedness arc in their own vocabulary:

- **Da Costa-like** received: "Gaussian pooling is a null model that provably can't produce Kuhnian dynamics; what escapes requires specific structural features (gate, abstention protocol)." Very close to intended.
- **Heins-like** received: "abstention vs. delta pooling is the operationally important distinction; you can't fix the averaging by going through it, you have to route around it." This is the open-endedness thesis in practitioner terms. He would pull the code and cite Fig. 4.
- **Zollman-like** received: "credence dynamics presupposes the support; model-class extension is a dynamic our framework doesn't have, and the pooling protocol is where it bites." The exact intended claim, conceded by the reader with the strongest motive to reject it.

**But the main ask — "the construction is proper" — fails with exactly the reader who checks.** The careful theorist believes the architecture broadly (operator, tilt algebra verified by hand, forgetting/saturation verified, BMR correctly used) but identifies three derivational debts:
1. **The conviction gate** $w_k=\exp(-g|U\cdot H_k|)$ — load-bearing for the entire §4.1 result, presented as if it follows from the tilted objective (Eq. 3), never derived, with a free parameter $g$ and no reported sensitivity analysis. "Underlining this three times."
2. **The closed-form $\Delta U$ claim** in the crisis check — stated, never exhibited; "exactly the kind of claim that gets quietly wrong in implementation."
3. **The leading-eigenvector expansion direction** — a heuristic written as if it were a consequence of the model.

Plus two empirical-honesty flags: the unexplained $\omega\in[0.9,1.0]$ robustness range ("only the last 10% of possible forgetting rates — suspicious"), and the undefined indicator in Eq. 4 ("gated by the core" — threshold on what?). Verdict instinct: **major revision**, contribution real.

## What survived every read

- **The abstention-pooling result (Fig. 4) is unanimously the contribution.** The theorist wants it stated as a proper proposition with proof; the practitioner will cite it; the philosopher concedes "the first thing in this paper that I could not have produced, and I know it." This is the robust core — protect it and promote it.
- **The null-regime framing reads as honest to everyone**, including the reader predisposed to call it physics-envy: "they're not claiming to explain paradigm shifts, they're characterizing why the defaults fail... that's the right move."
- **The literature trust signals work hard.** Dawid 1973, Genest & Zidek, Wright/Bollen all got checked in-head by two readers and passed — each correct citation visibly bought goodwill ("They went all the way back. I'm slightly less annoyed.").
- The sentence "An agent can silence its own instruments; it cannot silence its neighbours'" is the quotable line — the practitioner pulled it out verbatim.

## What only the careful reader saw

Everything in the "derivational debts" list above — the gate, $\Delta U$, the eigenvector heuristic, the $\omega$ range, the Eq. 4 indicator. The skimming readers took the gate formula at face value ("clean, a continuous relaxation, sensible") — meaning at IWAI you will get away with it with most readers *and be killed by the one mathematical reviewer*, which workshops reliably assign at least one of.

Also careful-reader-only: the **theorem-to-simulation blurring**. "The theorems say consensus must happen eventually; they do not say the transition is sharp at inter≈0.005 or that 92% is the number. The paper writes 'the theorems explain the sweep' when it means 'the sweep is consistent with the theorems.'" The hostile reader made the cousin complaint: the contraction theorem "is not a contribution; it's a foil — used correctly, but don't present deploying it as novel."

## Where readers diverged

- **The §4 theorems-first opening:** the practitioner respected it, the philosopher half-rolled his eyes at the contraction half ("I teach this to undergraduates") but credited the framing, the theorist accepted the theorems but rejected the implication arrow. Tag: partly priors (Zollman), partly **writing-fixable** — one sentence distinguishing "the theorems bound what can happen; the sweep measures where" from "the theorems imply the numbers" resolves both complaints.
- **The "cannot represent" abstract claim:** only the philosopher hit it, and hard — "the honest version is 'cannot represent *without additional apparatus*'." Tag: **writing-fixable**, one word-level change, high goodwill payoff with exactly the community the intro picks a fight with.
- **§3 reading depth:** the practitioner skipped the equations and was fine; the theorist read everything and found the gaps. The model section's prose carries skimmers; its math must carry checkers. Currently it carries the former and not fully the latter.

## If you change one thing

**Settle the conviction gate's epistemic status.** Either derive $w_k=\exp(-g|U\cdot H_k|)$ from the tilted objective (or from a stated auxiliary principle), or explicitly declare it a modelling commitment in §3.2 and report a $g$-sensitivity panel for the §4.1 result. It is the single mechanism standing between you and the goal you named as primary ("the reader believes the construction is proper"): the careful reviewer's whole major-revision verdict pivots on it, and it contaminates trust in the §4.1 numbers ("if the 92% threshold depends on $g$, the theorem framing of the result is wrong").

Runner-up fixes, in order of leverage: (2) state the abstention-pooling survival result as a proposition with a proof sketch — it upgrades the unanimous favorite from finding to theorem; (3) soften "cannot represent" → "cannot represent without extending the apparatus"; (4) one sentence separating theorem-bounds from sweep-measurements in the §4 opening; (5) explain or widen the $\omega$ range; (6) define the Eq. 4 indicator.
