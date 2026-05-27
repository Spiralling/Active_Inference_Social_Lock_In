# Call notes — 2026-05-28

Mahault, Jonas, David. ~40 min call.

---

## Agreed paper structure: ablation build-up

David proposed structuring the paper as incremental complexity:

1. **Fig 1 — Schematic.** Model architecture: agents on a network, the
   substrates, the coupling.

2. **Fig 2 — Base model.** Simplest version: agents + trust-weighted message
   passing + motivated belief updates. No resources, no POMDP actions. Show
   belief trajectories. Prove the minimal substrate is sufficient for basic
   dynamics.

3. **Fig 3 — Phase diagram (headline).** Trust coupling vs. paradigm inertia.
   The regime boundary between adaptation and lock-in.

4. **Fig 4 — Ablation.** What the resource layer adds that the base can't
   reproduce. The dynamic that only appears once resources are in.

Key principle: "the simpler the model, the stronger the contribution."
Reviewers will ask why we need each layer — the ablation answers that.

---

## Simplest model (Jonas + David, London, Thu-Sat)

The minimum model for Fig 2:
- Agents on a network (Watts-Strogatz)
- Message passing: trust-weighted mixture of neighbors' beliefs
- Belief update: motivated variational update from Hyland & Albarracin (2025)
- Trust: static or co-evolving with prediction error

David's framing: "It's basically our paper minus the multi-agent test."
Jonas: "Can we get confirmation bias and lock-in from just the update equation?"

If the simplest model already shows lock-in → great, the resource extension
becomes the ablation showing what it adds.
If it just drifts → that absence motivates the full model.

---

## Binary context debate

Mahault presented the hierarchical context model (c=0 normal science, c=1
crisis). Jonas asked: "Why does the context variable need to be binary?"

Mahault's answer: principled AIF requires carrying the context posterior
forward through a proper transition model. Without it, you lose evidence
accumulation about normal-science vs crisis.

Jonas's counter-suggestion: instead of binary c, use a richer belief network
per agent where different beliefs have different centrality/stickiness. This
would give graded transitions naturally and connects to the abstract's
"structured network of interdependent beliefs."

**Open question:** Can we replace the binary context with a continuous
belief-revision cost (lambda) that modulates update strength? This would
recover the Hyland & Albarracin (2025) formalism directly at the multi-agent
level.

---

## Resources debate

David skeptical about resources being essential to the story. His position:
start with information dynamics alone, see what falls out. If the simplest
model already produces interesting dynamics, resources become "nice to have"
not "load-bearing."

Mahault's position: resources are what makes social structure gate exploration.
Without r, trust topology is cosmetic. The cost barrier 1/(r - r_min) is
what makes agents unable to afford informative experiments when starved of
network patronage. Resource alone monostable, trust alone monostable, coupled
= bistable.

**Resolution:** run the stripped-down version first (Fig 2). If it's "too
smooth," that's the argument FOR keeping resources (Fig 4). Either way, the
contrast does the work.

---

## Work split

- **Jonas + David:** Minimal model from London (Thu-Sat). "Me and David are
  probably gonna work together from London tomorrow."
- **Mahault:** Extensions — resource layer, continuous context, phase diagrams.
  "I'll work off my own extensions and do the rest."
- **Jonas:** Owns first paper draft. Open to major revisions.
- **Next call:** Tuesday (Jonas scheduling). "Could we do Sunday or Tuesday?"
  Settled on Tuesday, time TBD.

---

## Timeline

- Thu-Sat: Jonas+David minimal model; Mahault extensions
- Sat afternoon: Jonas free to focus fully (other commitment ends)
- Sun/Mon: Results should be in
- Tuesday: Call to sync on results + figure plan
- ~June 1: Start writing
- June 7: Deadline

Mahault: "We should have a working draft a week before June 7 for writing and
iteration without the three-day no-sleep sprint."

---

## Action items for Mahault

1. Continue working on extensions (resource coupling, trust learning, phase diagrams)
2. Prepare the ablation: what does the resource layer add that base can't do?
3. Think about Jonas's suggestion re: richer belief networks vs binary context
4. Be ready to show the contrast (Fig 2 vs Fig 4) at Tuesday's call
