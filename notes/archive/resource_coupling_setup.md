# ARCHIVED — Resource-coupling setup (paradigm dynamics)

**Archived 2026-05-20.** Superseded by the belief-utility (Hyland) direction agreed in a
45-min conversation with David Hyland (2026-05-20): the categorical-POMDP rebuild **drops
resource dynamics**, and *belief utilities* now play the symmetry-breaking / "paradigm
capture" role that resource was meant to play. This note preserves the resource design so
it can be revived if needed.

**Code status:** `src/resource.py` (`flow_from_trust`, `inflow_share`, `flow_step`,
`cost_x`) is **NOT deleted** — it remains part of the *continuous linear-consensus baseline*
(`src/population.py` and `src/policy.py` import it; `notes/dynamical_systems_diagnosis.tex`
relies on that baseline). Only the resource *coupling design* for the new POMDP rebuild is
archived here.

---

## 1. The resource-coupled paradigm model (paper framing, 2026-05-18)

Coupled **belief–resource** system. Each agent carries a scalar resource state `r_i` on top
of the Gaussian posterior; resources flow on the graph and close the loop between belief,
action, world, and reward (Bourdieu/Kuhn: belief follows the resource gradient).

**Two inflow channels (both in the baseline, not future work):**
- **Patronage** `W_{ji}(t+1) ∝ exp(−D_KL[q_i ‖ q_j])` — agents allocate to those whose beliefs they share (gatekeeper logic).
- **Meritocratic** `η^merit_i ∝ exp(−ε_ii)` — driven by self-surprisal (same quantity already computed for the trust update).

**Three couplings back into belief dynamics:**
- Resource-gated experiment cost `c(x; r_i)` that diverges when `r_i` falls below a discriminating experiment's cost.
- Expected-resource-gain policy term `κ · ΔE[r_i | x]`.
- λ-scaled socially-induced utility `U(q_i; {q_j, r_j}_j)` pulling the posterior toward high-`r` neighbours.

**Three closed feedback loops on one spine:** L1 epistemic (existing AIF), L2 trust (existing
multi-agent), L3 resource (new). None closes on its own.

**Experimental axes (resource version):** A — Γ–W alignment; B — patronage centralisation ×
meritocratic share `R_merit/R_in` (the prescriptive progress-studies axis); C —
Fisher-information–coupled experiment cost (fitness-valley effect).

## 2. The POMDP-rebuild resource plan (design-note §5, dropped 2026-05-20)

**Decision (now reverted):** resource modulates the social **message precision** `κ_i` (not
the exogenous inflow share). Rationale at the time: it touches the validated bifurcation
parameter directly — rich/high-resource agents broadcast with higher effective precision, so
resource inequality tilts which basin the population falls into ("resource → paradigm-capture").

Concretely: `κ_{j→i} = κ_0 (1 + g · r̃_j)`, with `r̃_j` agent `j`'s resource share and `g`
the coupling gain; the resource recursion would reuse `src/resource.py` with `W` derived from
the trust matrix `T`. The inflow-share channel was deferred.

## 3. Why dropped, and how to revive

**Why dropped:** (i) nb14 showed the continuous resource layer is a *slaved readout* — ρ≡0.9
across the entire `(R_in, α_flow)` box, no independent dynamical mode; (ii) David's reframing
is simpler and more faithful to the sociology of science — paradigm *commitment / motivated
reasoning* (belief utility), not money, is what entrenches paradigms; (iii) **belief-utility
heterogeneity now supplies the symmetry-breaking field** that turns the NB15 symmetric
pitchfork (polarization) into capture, which is exactly what `resource→κ` was for.

**How to revive:** resource could re-enter later as a *second-order modulator* of belief
utility or message precision (the `κ_{j→i} = κ_0(1 + g·r̃_j)` form above), once the
belief-utility core is validated. Treat it as an extension, not a load-bearing mechanism.

See [[project_paper_scope]] and [[project_pomdp_rebuild_decision]] (memory) for the current
belief-utility direction.
