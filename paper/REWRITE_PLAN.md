# Rewrite plan — *The Changing Networked Mind*

Reframe, reorder, tighten around one question and one clean conceptual core.
Prose and references are reused; experiments are **downstream and plastic** —
each is redesigned to test one beat of the locked story. Audience: IWAI /
active-inference (cut scaffolding they don't need; AIF machinery stays
unmodified; name the generative-model commitments openly).

---

## 0. Core story (LOCKED)

A paradigm is not a credence over a fixed menu of theories. It is a **directed
network of interdependent commitments** whose *structure* a community must learn
while using it. Model paradigm change as **structure learning over that network**
— the network carrying belief *and* value — across a population that shares and
**locally tests** candidate structural revisions. Then Kuhn's arc — normal
science, anomaly, crisis, revolution, lock-in — is a **consequence**, not a
stipulation.

The chain: *normal science* reduces within the current structure, observing at
the belt. *Anomaly*: because the core is interior (no direct observable),
structural prediction error about it **accrues by propagation** from the belt and
from peers' proposals, unannounced. *Crisis*: accrued core-residual crosses the
threshold that forces a structural edit. *Revolution*: expansion edits the core
and pays the carry-over cost. *Lock-in*: if value is high enough it silences the
channels carrying the disconfirming signal, so the residual never accrues — and
in a population this is durable only when conviction is shielded by
disconnection.

---

## 1. Conceptual spine — two fields + network geometry (NO attention mechanism)

Two posited fields on one operator `T = (I−A)⁻¹`, plus the geometry that's
already definitional. Never conflate them:

- **Stuckness** `κ = T·𝟙` (+ accumulated precision): how hard a commitment is to
  move. Position + history. **Derived.**
- **Value** `U = T·u`: how much it is wanted-true. Attachment. Independent of κ.
  Source `u` exogenous.
- **Network geometry** (belt = in contact with the world; core = interior): this
  is what sets the **timescale**, and it is *not a new quantity* — it falls out of
  what a paradigm net is. The core is slow because change reaches it only through
  the graph. (This replaces the earlier "attention mechanism" — dropped.)

Jobs: **stuckness sets how slowly the core moves once pushed; value sets whether
you let it be pushed; geometry sets whether the push ever arrives.**

---

## 2. Modeling-choices ledger (state these AS choices in the paper)

**Settled posits:** linear-Gaussian couplings on a DAG; honest agents (lock-in is
never a dishonesty artifact); exogenous utility `u`; derived conservatism κ.

**Objective — myopic VFE, Poisson arrival, ΔF acceptance (LOCKED).** Agents
minimize *current* VFE (local myopic modelers, not EFE planners). Genuinely-new
structural candidates *arrive* at a Poisson rate; each is *evaluated* by Bayesian
model reduction (ΔF, closed form). The honest claim: **the evaluation of a
structural change is derived (ΔF = epistemic value of the edit); its arrival is a
rate, and it must be, because one cannot plan toward a dimension one has not
conceived.** The Poisson rate is therefore not a cop-out but the correct model of
serendipity — the same Stanford unconceived-alternatives point that is the paper's
frontier. Rewrite every "EFE drives exploration" line to this.

**Core moved by propagation (LOCKED).** Paradigm cores are *interior* nodes with
**no direct observable**; they are revised only by propagated structural error
from the belt and from peers. (This is the single commitment §7.7 violated — there
the operator read the core directly, so it moved first and the staircase
inverted.)

**Social move — "try it yourself" (LOCKED, load-bearing).** Agents do **not**
fuse peer proposals weighted by trust. They *instantiate* a peer's proposed
revision and score it by ΔF **against their own structure and data**, then adopt
or reject. Consequences:
  1. Same node, different basement → different ΔF → communities **diverge in
     edges**, not just means → structural incommensurability, the flagship claim.
  2. Local testing **preserves** heterogeneity where naive precision *fusion*
     destroyed it (the recurring no-pool washout). This is the fix, not a flourish.
  Trust precision `π_s` now gates *whether you bother to test a source's proposal*
  (bubble = untested-because-absent; echo chamber = present-but-π_s≈0).

**Cost of exploring — rate, not resource (LOCKED).** A time/rate cost (you can
only propose/test every so often) = the Poisson rate above. Resource budgets are
out of scope (already scoped out earlier in the project); they add accounting for
no story gain.

---

## 3. Section-by-section reframe (against the actual §§1–9)

- **§1 The question (Kuhn doorway).** Add the missing framing paragraph (drafted:
  "least well modelled" → threads into the phlogiston open). Keep the two-debts
  gap (exploration posited / menu fixed). Refs unchanged.
  **NEW Fig. 1:** the Kuhn arc as a diagram, our mechanism annotated underneath
  each phase.
- **§2 What a paradigm is.** Keep "a menu has no hidden interior" + "structure of
  reality is hidden." Fold the six-desiderata checklist into one sentence.
- **§3 Structure learning.** Reduction = exploit (cheap, BMR); expansion = explore
  (costly). Carry-over κ derived. **Promote the explicit modeling-choices ledger
  (§2 above) to a named subsection here.** Demote the Schur worked-example to a
  short payoff (move the banana to §7).
- **§4 The value field (HINGE).** U = T·u, decoupled from κ. State once, here, the
  three-jobs sentence. Keep the cumulant/λ-tilt machinery but lead with the
  concept.
- **§5 Population + local testing.** Rewrite the social model from fusion →
  **trust-gated local ΔF testing**. Bubble vs echo chamber re-expressed on π_s.
  This is where the divergence mechanism lives.
- **§6 Running the model.** Keep the order parameter; reframe the staircase as a
  prediction the *interior-core* design now tests honestly.
- **§7 Results.** See §4 below.
- **§8/§9 Prior work + Discussion.** Kuhn corollary, Stanford frontier (now the
  *reason* exploration is a rate), incommensurability = Schur residue + the
  local-testing divergence reading. Schur banana payoff lands here.

---

## 4. Experiment redesign (each beat → one clean test; reruns expected)

- **Two fields real & independent** → κ vs U on the dark-energy net. *(have it:
  fig_two_fields_dag, fig_schur_banana)*
- **Structural divergence via local testing (NEW FLAGSHIP).** Two+ communities,
  different basements, sharing proposals they each ΔF-test locally. Measure
  edge-divergence (not means). Predicted: same proposal, opposite verdicts →
  communities end on different *structures*. Contrast against the
  fusion-homogenizes control. *(de-risks the layer-3 claim the project never
  tested)*
- **Observability staircase (RERUN).** Core = interior, no direct H row; belt
  observed; structural error reaches core only by propagation. Does belt-first /
  core-last finally appear? This is the honest version of the §7.7 test.
- **Crisis → expansion (node-wake), re-pointed at the core.** Residual accrues at
  the unobserved core, crosses threshold, wakes the dimension. *(adapt fig_node_wake)*
- **Lock-in = value silences the channel (endogenous γ).** *(have it:
  fig_endogenous_gamma)*
- **Population: conviction + disconnection.** *(have it: fig_tracking_2x2,
  fig_diversity — but re-examine under local-testing dynamics, since fusion was
  the old substrate)*

> Note: moving from fusion → local testing is the substrate change that touches
> the most existing results. Re-run the population experiments on it; expect the
> heterogeneity/divergence results to *strengthen* (no more washout).

---

## 5. Drafting order

1. §1 frame + new opener + Kuhn-arc figure stub.
2. §3 modeling-choices ledger (get the posited/derived/myopic commitments exact).
3. §4 hinge (κ/U three-jobs).
4. §5 population + local-testing rewrite (the substrate change).
5. §6/§7 results, reruns, staircase + divergence.
6. §2 background trim; §8/§9 discussion; Fig. 1 art; reference + length pass.

## 6. Open (carry defaults unless flagged)

- Poisson rate value / whether arrival is per-agent or per-community.
- Edge-divergence metric for the flagship test (coupling-mass disagreement vs
  graph-edit distance).
- Whether "try it yourself" fully replaces fusion or is the headline with fusion
  as the washout control.
