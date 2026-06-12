---
type: cold-read
draft: "Changing a Networked Mind: Paradigm Dynamics as Structure Learning over a Hidden World (v3)"
draft_path: "C:\\GitHub\\Paradigm_Shift_Act_Inf\\paper\\changing_networked_mind_v3.pdf"
audience: "IWAI 2026 reviewers (community-based double-blind; active inference community)"
intended_ask: "Accept for publication — a reviewer should judge it novel, sound, and well-scoped enough to accept"
readers: 3
dispatched_at: 2026-06-12T16:17:03+02:00
completed_at: 2026-06-12T16:25:00+02:00
status: complete
tags: [cold-read]
---

# Cold read: Changing a Networked Mind (v3)

**Intended audience:** IWAI 2026 reviewers (Madrid, Oct 2026; community-based double-blind review; Springer CCIS).
**What you wanted them to take away:** Accept for publication.

## The panel

| # | Reader | Based on | Scene | Attention | Outcome |
|---|--------|----------|-------|-----------|---------|
| 01 | [[01-albarracin-like-careful]] | a reader like Mahault Albarracin (VERSES; social AIF, refs [1][2][15]) | fresh-careful | high | engaged — read everything twice; **weak accept leaning accept (6)** |
| 02 | [[02-robotics-postdoc-typical]] | composite Ghent/Delft robotics postdoc (modal IWAI reviewer) | distracted-typical | medium | skimmed — abstract, equations, figures, results; skipped §1–2, §5; **weak accept (6)** |
| 03 | [[03-buckley-like-hostile]] | a reader like Christopher Buckley (Sussex; anti-overclaiming school) | busy-hostile | low | engaged (nearly bounced at the abstract) — **weak reject *pending appendices***, "best-scoped FEP-meets-social-epistemology paper in two years, sunk by the absence of its own appendices" |

Simulated panel verdict: **borderline**. Two weak accepts and one conditional weak reject. The reject is not intellectual — it is a verifiability objection that one mechanical fix removes.

## Intended vs. received

The intended ask was acceptance. What every reader received was the *content* you wanted them to receive — and then docked the paper on logistics:

- **Careful reader (01):** restated the thesis almost exactly as the abstract intends ("under the field's own defaults, conflict provably resolves into confident compromise; two departures restore doubt-as-posterior and pluralism-as-fixed-point"). Verdict 6, withheld from a 7 by: missing appendices, 4–6 pages over format, and the K-wirings-are-given assumption surfacing only on p13 when it is a founding assumption of the representational layer.
- **Modal reviewer (02):** got the thesis right from abstract + figures + §3.7 alone — the paper survives a skim structurally. But walked away believing the abstract *oversells* the rival mechanism: "it implies the mechanism can grow a new hypothesis, which it cannot without a supplied candidate set." Wants that asterisk in the abstract. Also: read Result 1 as "we verified a contraction theorem we already cited" and Result 2's variance-inflation figure as the actual novelty, undersold.
- **Hostile reader (03):** pattern-matched "another Kuhn paper" within ten seconds of the literary abstract and was reaching for reject. Two things reversed him: the variance-inflation figure (behavior the baseline theorem rules out — "I stop wanting to reject it"), and the honesty sentences ("DeGroot dynamics in information form"; "nothing exotic has been smuggled in" — "That is the sentence I needed. They know what they have."). What he could not get past: "these are theorems of the regime" with no appendix in the PDF. That is, verbatim, the failure mode he has a standing policy of rejecting.

**The gap:** you wanted "accept"; the ideas earned it from all three readers, but the *artifact* — missing appendices, 18 pages against a 10–12 norm, an abstract that hides the supplied-candidates caveat — is what each reader priced their score on. The intellectual work is done; the packaging is what's blocking the ask.

## What survived every read

- **The one-line thesis.** "Averaging refines a frame; it cannot grow a new one" / gates buy time, rivals buy walls — all three readers reproduced it correctly, including the skimmer. The compression of the contribution into that ladder works.
- **The variance-inflation result (crisis as suspended confidence; Fig. 4, p14).** Unanimously the strongest thing in the paper. Careful: "the figure I will cite in my review." Modal: "the most convincing panel... novel and interesting and under-sold." Hostile: "the best figure in the paper... shows something behaviorally distinct that I couldn't have derived by staring at the equations." It converted the hostile reader mid-read.
- **The honesty register.** Naming the pooling as DeGroot, the gate as classical robust Bayes (O'Hagan), the demos as "deliberately small... existence proofs," and the "nothing exotic has been smuggled in" paragraph — every reader independently flagged these as trust-building, and for the hostile reader they were decisive. This register is an asset; protect it in any revision.
- **The §3.7 six-step loop.** Both the modal and hostile readers used it as their anchor; the modal reviewer judged it reimplementable, which for an IWAI engineer is the bar.
- **The three-curve experiment figure (Fig. 3, p13)** read cleanly even to the skimmer: disconnected/connected/gated with the right null control.

## What only the careful reader saw

At risk with real reviewers, since only the fresh-careful read reached these:

- **The conviction field U = Tu as a genuinely new result on top of Hyland & Albarracin [15]** — the same operator on two linearly independent sources, so conservatism and conviction decorrelate. The careful reader called it "genuinely new... not derivative"; the other two never registered it as a contribution distinct from [15].
- **Incommensurability-as-state** (p17) — "the best theoretical contribution," in her words, and she had to reach §5 to find it; the modal reviewer read only §5's headings, the hostile reader skimmed past it.
- **The citation positioning of [1], [2], [15]** — used as building blocks rather than wallpaper, checked and approved. Only a reader from that lineage would have verified this; it bought real goodwill, but only with that one reader.
- **"A population of structure learners is only as open-ended as its pooling permits"** (p17–18) — she flagged it as the sentence that will land with the IWAI audience and suggested moving it toward the abstract. Buried where it is, two of three readers never saw it.

## Where readers diverged

- **Value of Result 1 (homogeneity effect).** Careful reader: theorems made visible, fair comparison, provable inequality. Modal reviewer: "we verified a known theorem in our specific setup, not a surprise." *Split is part priors, part writing* — an engineer will always discount a contraction-theorem result, but the paper could pre-empt it by framing Result 1 explicitly as the measured *boundary* (the 10%-vs-94% numbers) rather than as the collapse itself.
- **The abstract.** Careful reader: dense but correct. Modal: "trying to front-load four results and two theorems and a claim about Kuhn simultaneously"; had to read it twice. Hostile: "almost a page... literary in a way that immediately makes me suspect the math will be thin" — it nearly caused a bounce before page 2. *Writing-fixable:* the abstract's length and register are actively selecting against the busy reader, who at IWAI is the modal reader.
- **Verdict mechanics.** The two readers who could not verify the theorems but trusted the framing gave 6; the reader whose identity is built on verifying gave weak-reject-pending-appendices. *Not priors — artifact:* all three converge on accept once the appendices ship with the PDF.
- **Small details that only bit the skimmers:** the "(fro)" unit on the Fig. 3 y-axis read as an undefined unit (the modal reviewer transcribed it as "flo" and was annoyed — spell out "Frobenius"); whether λ is fixed or inferred is unclear from §3.2 alone.

## If you change one thing

**Ship the appendices in the reviewed PDF — or demote "theorem" to "proposition (proof in supplement)" everywhere the proof isn't present.** It is the single fix that flips the hostile reader from weak reject to weak accept ("resubmit with A, B, C present and I will read them"), and it removes the only caveat both other readers attached to their 6s. Every other fix (cut §1–2 toward 12–14 pages; one abstract sentence stating the K candidate wirings are supplied, not discovered; shorten the abstract itself) raises scores — this one removes the stated grounds for rejection.
