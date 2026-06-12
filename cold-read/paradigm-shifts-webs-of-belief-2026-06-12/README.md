---
type: cold-read
draft: "Paradigm Shifts: Motivated Inference on Webs of Belief (v5)"
draft_path: "C:\\GitHub\\Paradigm_Shift_Act_Inf\\paper\\changing_networked_mind_v5.pdf"
audience: "IWAI 2026 reviewers (community-based double-blind; active inference community)"
intended_ask: "Accept for publication — a reviewer should judge it novel, sound, and well-scoped enough to accept"
readers: 3
dispatched_at: 2026-06-12T19:19:02+02:00
completed_at: 2026-06-12T19:35:00+02:00
status: complete
tags: [cold-read]
---

# Cold read: Paradigm Shifts: Motivated Inference on Webs of Belief (v5)

**Intended audience:** IWAI 2026 reviewers (Madrid, Oct 2026; community-based double-blind; Springer CCIS).
**What you wanted them to take away:** Accept for publication.

Same panel as the v3 read (2026-06-12, `cold-read/changing-networked-mind-v3-2026-06-12/`), fresh scene draws, readers cold — no knowledge of v3 or the earlier panel.

## The panel

| # | Reader | Based on | Scene | Attention | Outcome |
|---|--------|----------|-------|-----------|---------|
| 01 | [[01-albarracin-like-careful]] | a reader like Mahault Albarracin | fresh-careful | high | engaged — all 28pp, appendices verified against claims; **accept (7)** |
| 02 | [[02-robotics-postdoc-typical]] | composite Ghent/Delft robotics postdoc | distracted-typical | medium | engaged-skim — abstract, all figures, §3.4, §4; no appendices; **weak/borderline accept (6)** |
| 03 | [[03-buckley-like-hostile]] | a reader like Christopher Buckley | busy-hostile | low | engaged — ~18/28pp non-linearly, **read the appendices voluntarily**; **weak accept tending accept (6)** |

Simulated panel verdict: **accept-leaning across the board (7/6/6), no rejects.** Versus the v3 panel (6/6/weak-reject-pending-appendices), every blocker named there is gone: the appendices are in and two readers verified derivations against them; nobody objected to "theorem" labeling (the contraction is now attributed to DeGroot and called a closed form); the rivals-are-given caveat was found and accepted by all three.

## Intended vs. received

- **Careful reader (01):** received the thesis exactly as intended and upgraded to a clear accept. Verified Eq. 3 against Appendix A, the crisis-check ledger against D.2 (Eqs. 10–11), the Schur residue against D.3, and the wiring-gate normalization against C — all held. Her one genuine flaw: **[2] Çatal et al. and [46] shared protentions are ghost references** — in the bibliography, never cited in the text (verified against the LaTeX source: neither key is ever `\cite`'d). For a reader from that lineage this reads as "citation appearance, not engagement," and it's the difference between her 7 and an 8.
- **Modal reviewer (02):** got the full contribution from abstract + figures + §3.4 alone and judged it a presentable 15-minute IWAI talk. Walked away believing the loop is reimplementable. What he took away that you may not have intended: the crisis threshold "felt hand-wired for the phlogiston topology," and he wanted one missing comparison — why not a mixture-of-Gaussians at the community level as the obvious alternative way to hold multiple hypotheses? Lost the §3.2 Savage–Dickey thread to an interruption and never recovered it.
- **Hostile reader (03):** the headline of this whole read. He braced at "Paradigm shifts are a significant driver…" ("oh good, another Kuhn paper"), and then the paper systematically dismantled his prior: DeGroot named, "whatever the parameters" backed by the entrywise inequality, prototype numbers flagged paragraph-by-paragraph in Table 1, and — the moment that stopped him cold — Appendix C reporting **"This refuted our own prediction"**: "That is not what most FEP papers do. I read it twice." He read 12 pages of appendix *by choice* on a Saturday morning. Final lean: "every specific claim I tried to catch them overclaiming, they had already caveated."

**The gap, such as it remains:** small and concentrated. You wanted accept; you got 7/6/6. What's left between this and 8/7/7 is: ghost references, the contraction claim not being a labelled Proposition in the main text, seed-consistency for the N=16 Kuhn-cycle trace, and the most honest material (Appendix C's scope paragraph, Appendix B's lone-discoverer result) living where two of three readers never go.

## What survived every read

- **The thesis ladder** — averaging is a contraction, so structural diversity cannot survive contact under the default; a posterior over rival structures restores it; the Kuhn cycle reads off the belief state. All three reproduced it accurately, including the skimmer.
- **Fig. 3 (three-architecture log-log)** — "the money shot" (02), "a genuine result figure" (03), the two-orders-of-magnitude gap noted by 01.
- **Fig. 4 (Kuhn cycle, crisis as entropy peak)** — the accept-clinching figure for 01 ("this is the figure... why I would say accept"), "actually nice... not obviously faked" (03), "clean operationalization" (02).
- **The honesty register, now structural rather than rhetorical** — "Boundaries, stated plainly," the byte-identical-defaults controls, Table 1's above/below-the-rule bookkeeping, D.5's "what remains a modelling commitment," and the self-refutation report. This is what converted the hostile read; in v3 it was a sentence, in v5 it's an architecture.
- **The rivals-are-given scope choice** — surfaced in §3.2 main text and accepted by all three readers as honest scoping rather than a hidden weakness (in v3 it was the modal reviewer's chief complaint about the abstract).

## What only the careful reader saw

- **The ghost references** ([2], [46]) — invisible to a skimmer, glaring to the one reviewer most likely to be assigned this paper. Verified real.
- **The Fig. 4 right-axis ambiguity** — H(q(m)) tops out at 1.0 but max entropy for K=2 is log 2 ≈ 0.693 nats; is it normalized by log K? The axis label doesn't say.
- **D.2's sharpest insight** — λΔU_e *shrinks* as evidence accumulates, so a data-rich paradigm is barely protected at the decision threshold and lock-in must come from upstream attention-gating. She calls it "the paper's most important theoretical insight" and wants it in the main text; it currently lives only in Appendix D.
- **The dissenting-minority question** — the model could track the fate of agents who held the old paradigm after the flip (conviction is per-agent), and never asks. A one-sentence future-work flag would pre-empt the question from any [1]-lineage reviewer.
- **The lone-discoverer/protocol result (Fig. 7, Appendix B)** and its line "incommensurability as a population-level force is a property of the protocol, not of the concepts" — flagged as one of the paper's best, buried on p.21.

## Where readers diverged

- **Appendix reach.** The careful and hostile readers both lived in the appendices and came away more convinced; the modal reviewer never opened them and so took "no bar is tuned" on faith and kept a residual "feels hand-wired" suspicion. *Writing-fixable at low cost:* the hostile reader's own condition — summarize Appendix C's "Scope, stated plainly" in one paragraph at the end of §4 — fixes exactly this for the reviewer who won't flip pages.
- **§1–2 philosophy.** Careful reader: earned, correctly cited (down to preferring Kuhn 1983 over 1962 for incommensurability). Modal reviewer: "skipped most of it and lost nothing for the technical result," wants it trimmed. *Priors, mostly* — but the modal reviewer is the median, so §1–2 is still carrying cost with the audience's center of mass.
- **What's still missing as a baseline.** Only the modal reviewer asked for a mixture-of-Gaussians community-level comparison; only the hostile reader asked for seed-consistency on Fig. 4; only the careful reader asked for the minority's fate. Three different "one more thing"s — none fatal, all cheap to pre-empt with a sentence each.

## If you change one thing

**Cite or cut the ghost references.** `catal2024` and `albarracin2024protentions` are in the bibliography and never invoked (verified in the source). Given IWAI's community-based review, the probability that an actual reviewer is from exactly that lineage is high, and it's the single finding that read as a *flaw* rather than a wish — everything else on the panel's lists is upgrade, not repair. One sentence in §4.1 ("the hazard is recognized within active inference itself [2]" — which v3 had and v5 dropped) and one in the §5 collective-intelligence paragraph for [46] would convert both into engagement.

Close behind, in order of score-per-minute: (1) label the entrywise contraction claim as a Proposition in §4.1; (2) one paragraph at the end of §4 summarizing Appendix C's scope commitments; (3) Fig. 4 — state seeds and normalize (or label) the entropy axis; (4) one sentence in §3.3 re-stating that the K candidates are given (it's in §3.2; restate where rivals are defined).
