"""The full Kuhnian revolution on the phlogiston world: population + BMR + BME, one loop.

Everything the paper develops, finally in one model:

* **Population** (``run_simulation``): N agents on a two-community SBM ("open" vs "dogmatic" --
  different prune thresholds ``lam_i`` and different conviction-gate strengths via per-agent
  ``u`` scaling), fusing precision over the trust graph, observing the gravimetric channel of a
  DETERMINISTIC truth schedule that flips from the phlogiston regime to the oxygen regime at
  ``t_shift`` (the precision balance arrives; calx is heavier).
* **Crisis = the model's own ledger** (no hand-made accumulator, no proxy threshold): per agent
  and edge, the prune is scored on the unified move ledger ``Delta F + lam_i * Delta U`` -- the
  Savage-Dickey prune evidence plus ``lam_i`` times the CLOSED-FORM conviction-value change
  ``Delta U_e = U_i . (mu_reduced - mu)``, where ``U_i`` is the agent's current conviction field
  (``dual_field.conviction_field``) and ``mu_reduced`` the reduced posterior mean the
  Savage-Dickey identity already exhibits. This is the tilted objective (eq:tilted) applied to
  a structural move: pruning a conviction-protective edge costs value (``Delta U < 0``), so the
  evidence must overcome ``lam_i * |Delta U|`` -- the former ``lam_i * v_e`` threshold with the
  proxy ``v_e = |U_p| + |U_c|`` replaced by the quantity it stood in for. Crisis = both
  protective BELT edges score ``> 0``.
* **Revolution = Bayesian model reduction APPLIED**: at crisis the agent's forgetting anchor is
  replaced by the CPD-space reduced prior (``over.prune_edge`` composed over every edge its own
  ``Delta F`` flags -- the same reduction family the evidence scored) with the phlogiston hub
  block released (mean-preserving precision scaling).
* **Discovery = Bayesian model expansion**: the basis carries an 11th, genuinely unconceived
  ``oxygen`` slot, pinned (huge self-precision, zero couplings, no observation row -- a hidden
  hub, like phlogiston). Post-crisis, on each Poisson-arriving proposal, the agent reads its
  windowed node-space prediction errors on the disagreement nodes
  (``action.residual_from_errors``), proposes a hub (``action.propose_hub`` -- the windowed
  errors fix the proposal's DIRECTION; the floor strength is telemetry, not a gate), and scores
  the wake with the model log Bayes factor (``action.expansion_score``) -- the SOLE accept test.
  There is no residual-floor trigger and no sustain requirement: the Poisson rate models that
  one cannot plan toward an unconceived dimension, and the Bayes factor decides whether the
  residual loads on the proposed hub. Acceptance borders the oxygen slot in place (the agent's
  current net AND its anchor), replicating ``action.wake_hub``'s math in the pre-allocated
  dimension. Precision fusion then carries the awakened couplings to neighbours: the social
  diffusion of a new concept.

The pinned slot keeps every shape fixed (the population engine is jitted), and because it is
exactly decoupled with ``h = 0`` in all six Savage-Dickey inputs, the 14-edge prune read-out is
numerically unchanged until the moment an agent wires it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np
import jax.numpy as jnp

from src.structural import graphs, observables
from src.structural.action import expansion_score, propose_hub, residual_from_errors
from src.structural.belief import GaussianBeliefNet
from src.structural.candidates import CandidateConfig, search_candidates
from src.structural.dual_field import conviction_field
from src.structural.models.kuhn_cycle import scale_block
from src.structural.phlogiston import DISAGREEMENT_NODES, HUB, StructuralConfig, conviction_u
from src.structural.proposal_rate import ProposalRateState, RateConfig
from src.structural.simulation import (AgentSpec, ConvictionDynamics, Scenario, _dF_NE,
                                       _dU_NE, run_simulation)
from src.structural.scenarios import build_substrate, phlogiston_scenario, BELT

OXYGEN = "oxygen"
PIN = 1.0e4                  # pinned-slot self-precision (float32-safe; replaced on wake)
HUB_SELF_PREC = 2.0          # the woken hub's self-precision (matches cfg.hub_self_prec)
WINDOW = 30                  # trailing prediction-error window the proposal DIRECTION is read from
WARMUP = 5
FREE_PREC = 1e-2             # the dormant hub's free prior precision (matches expansion_score)


# ----------------------------------------------------------------------
# the padded (10 -> 11) scenario: phlogiston + a pinned, unconceived oxygen slot
# ----------------------------------------------------------------------

def _pad_mat(M: np.ndarray, pin: float = PIN, n: int = 1) -> np.ndarray:
    """(..., 10, 10) -> (..., 10+n, 10+n) with ``n`` decoupled pinned diagonal slots."""
    M = np.asarray(M, dtype=np.float64)
    out = np.zeros(M.shape[:-2] + (M.shape[-2] + n, M.shape[-1] + n), dtype=M.dtype)
    out[..., :-n, :-n] = M
    for s in range(n):
        out[..., -n + s, -n + s] = pin
    return out


def _pad_vec(v: np.ndarray, n: int = 1) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape[:-1] + (v.shape[-1] + n,), dtype=v.dtype)
    out[..., :-n] = v
    return out


def padded_scenario(cfg: StructuralConfig, *, conviction: float = 2.0, n_slots: int = 1):
    """Build the phlogiston scenario, then pad every basis-dimension array with ``n_slots``
    pinned, genuinely unconceived slots (the first is named oxygen; the rest are anonymous
    ``slot_1``... for the multi-candidate search to wire rival hubs into). Returns
    ``(scenario, substrate)`` -- the substrate's CPD net ``over`` is what the crisis-time
    reduction composes ``prune_edge`` on. Because every slot is exactly decoupled with
    ``h = 0`` in all six Savage-Dickey inputs, the 14-edge prune read-out is numerically
    unchanged for any ``n_slots`` (the pin-cancellation regression in
    ``tests/test_candidates.py``)."""
    sub = build_substrate(cfg, conviction=conviction)
    scn = phlogiston_scenario(cfg, sub=sub, conviction=conviction)
    slot_names = (OXYGEN,) + tuple(f"slot_{m}" for m in range(1, n_slots))
    names11 = tuple(scn.names) + slot_names
    H11 = np.concatenate([np.asarray(scn.H), np.zeros((scn.m, n_slots))], axis=1)
    order_fn = partial(_order11, names=names11, mass_nodes=DISAGREEMENT_NODES,
                       mu_phlog=cfg.mu_phlog_mass, mu_oxy=cfg.mu_oxy_mass)
    scn11 = Scenario(
        name="kuhn_phlogiston",
        names=names11,
        H=jnp.asarray(H11),
        disc_rows=scn.disc_rows,
        phis=jnp.asarray(_pad_vec(np.asarray(scn.phis), n_slots)),
        epoch_t=scn.epoch_t,
        n_epochs=1,
        sigma_o=scn.sigma_o,
        Pi_base=jnp.asarray(_pad_mat(scn.Pi_base, n=n_slots)),
        h_base=jnp.asarray(_pad_vec(scn.h_base, n_slots)),
        Pi0=jnp.asarray(_pad_mat(scn.Pi0, n=n_slots)),
        h0=jnp.asarray(_pad_vec(scn.h0, n_slots)),
        Pi0r=jnp.asarray(_pad_mat(scn.Pi0r, n=n_slots)),
        h0r=jnp.asarray(_pad_vec(scn.h0r, n_slots)),
        v_e=scn.v_e,
        edges=scn.edges,
        belt_ix=scn.belt_ix,
        u=jnp.asarray(_pad_vec(scn.u, n_slots)),
        conviction_alpha=scn.conviction_alpha,
        order_fn=order_fn,
        lstar=scn.lstar,
    )
    return scn11, sub


def _order11(Pi, h, *, names, mass_nodes, mu_phlog, mu_oxy):
    return observables.order_parameter(Pi, h, names, mass_nodes, mu_phlog, mu_oxy)


# ----------------------------------------------------------------------
# the revolution hook: BMR-triggered crisis + BME discovery, per agent
# ----------------------------------------------------------------------

@dataclass
class RevolutionTelemetry:
    crisis_step: np.ndarray | None = None      # (N,) first step both belt edges scored > 0; -1
    expand_step: np.ndarray | None = None      # (N,) first step oxygen bordered; -1
    dF_belt_t: list = field(default_factory=list)     # (T, N) min belt-edge Delta F
    score_belt_t: list = field(default_factory=list)  # (T, N) min belt-edge ledger score
    floor_t: list = field(default_factory=list)       # (T, N) residual-floor strength (telemetry)
    oxy_coupling_t: list = field(default_factory=list)  # (T, N) ||oxygen couplings||
    n_pruned: np.ndarray | None = None          # (N,) edges pruned at crisis
    rate_t: list = field(default_factory=list)        # (T, N) per-agent proposal rate (rate_cfg)
    attempt_t: list = field(default_factory=list)     # (T, N) bool, this step's arrival draw
    prune_back_step: np.ndarray | None = None         # (N,) first step a wake was re-pinned; -1
    hub_dF_prune_t: list = field(default_factory=list)   # (T, N) audit: re-scored wake dF
    #                                                      (prune verdict when <= 0)
    precrisis_dF_t: list = field(default_factory=list)   # (T, N) pre-crisis expansion dF
    candidate_log: list = field(default_factory=list)    # per-candidate dicts (cand_cfg)


def make_revolution_hook(*, scn: Scenario, sub, lam: np.ndarray,
                         hub_release: float = 0.25,
                         window: int = WINDOW,
                         proposal_rate: float | None = 0.08, seed: int = 0,
                         enable_reduction: bool = True, enable_expansion: bool = True,
                         warmup: int = WARMUP,
                         rate_cfg: RateConfig | None = None,
                         trust_W: np.ndarray | None = None,
                         emit_fuse_mask: bool = False,
                         wake_credit: bool = False, prune_grace: int = 10,
                         prune_patience: int = 3, credit_decay: float = 0.85,
                         allow_precrisis_apply: bool = False,
                         precrisis_check: bool = False,
                         cand_cfg=None):
    """Returns ``(hook, telemetry)``.

    Crisis & reduction: agent i scores each prune on the unified ledger,
    ``score[i, e] = Delta F[i, e] + lam_i * Delta U[i, e]`` (both halves closed-form, vs the
    STATIC padded reference -- valid for agents whose anchor is still untouched). BOTH belt
    edges scoring ``> 0`` is crisis => anchor <- CPD-reduced prior over ALL edges whose own
    score flags them, hub block released.
    Expansion: post-crisis, on each Poisson arrival, windowed node-space errors on the
    disagreement nodes -> hub proposal (direction); ``expansion_score`` (the model log Bayes
    factor) is the sole accept test => border the oxygen slot in the agent's current net AND
    anchor. ``proposal_rate=None`` = deterministic (attempt every eligible step) -- the
    deterministic-environment headline mode.

    ``wake_credit`` (opt-in) is the wake-then-prune cycle the limitations name as the
    principled remedy for the Bayes factor's repeated-look false-discovery rate: the wake is
    accepted on the FULL ledger ``Delta F + Delta G > 0`` (curiosity pays for entertaining
    the structure on credit), and the woken hub is then AUDITED every round after
    ``prune_grace`` steps in the same currency that scored it -- the bordered-model log
    Bayes factor re-evaluated on the agent's current beliefs against its pre-wake anchor,
    PLUS the epistemic credit, which now EXPIRES: the ``k``-th audit's verdict is
    ``Delta F_now + Delta G_now * credit_decay**k <= 0`` => prune vote. A genuine hub
    accumulates positive evidence before the credit runs out; a noise wake never does, and
    ``prune_patience`` consecutive prune votes re-pin the slot exactly. Re-pinned agents
    become eligible to wake again -- entertaining is paid by (expiring) curiosity, keeping
    by evidence. ``allow_precrisis_apply`` lifts the crisis-first eligibility gate (used by
    the null-world false-discovery experiment, where no crisis ever fires).

    ``precrisis_check`` (opt-in, pure measurement): runs the identical proposal + ledger
    pipeline for every PRE-crisis agent each step WITHOUT applying it, recording the model
    log Bayes factor in ``precrisis_dF_t``. This upgrades the enforced crisis-before-
    expansion ordering from a modelling commitment to a result: the ledger itself rejects
    pre-crisis wakes (the old paradigm still absorbs the residual).

    ``rate_cfg`` (optional) replaces the single-float Poisson gate with the principled rate
    of ``proposal_rate.py``: per-agent rates that can ride the trust graph (Hawkes excitation
    on neighbours' discoveries -- ``trust_W`` required) and/or track the agent's own residual
    floor as an updatable inadequacy belief (adaptive). When set, ``proposal_rate`` is
    ignored, the residual floor is computed for EVERY un-expanded agent each step (the
    adaptive modes consume it; legacy telemetry only filled it post-crisis), and the arrival
    draws come from a SEPARATE RNG stream (``900_002 + seed``) so the legacy stream is
    untouched. ``None`` (default) keeps the legacy path byte-identical."""
    names10 = sub.over.names
    d10 = len(names10)
    n_slots = len(scn.names) - d10
    slot_names = tuple(scn.names[d10:])
    idx10 = {n: i for i, n in enumerate(names10)}
    dis_idx = np.array([idx10[n] for n in DISAGREEMENT_NODES])       # basis indices (10-dim)
    dis_rows = dis_idx - 1                                           # H direct row of node i is i-1
    belt = np.asarray(scn.belt_ix, dtype=int)
    edges = scn.edges
    Pi0, h0 = scn.Pi0[0], scn.h0[0]
    Pi0r, h0r = scn.Pi0r[0], scn.h0r[0]
    lam = np.asarray(lam, dtype=np.float64)
    arr_rng = np.random.default_rng(900_001 + seed)                  # separate from world RNG
    gate_rng = np.random.default_rng(900_002 + seed)                 # rate_cfg arrival stream

    tele = RevolutionTelemetry()
    state: dict = {"err": [], "anchor_cache": {}, "rate_state": None,
                   "awake": None, "wake_time": None, "audit_count": None,
                   "wake_book": {}}    # i -> (prewake anchor 10-block Pi/h, proposal,
    #                                          coupling scale, wired slot index)

    def _net10(Pi_i, h_i):
        return GaussianBeliefNet(Pi=jnp.asarray(Pi_i[:d10, :d10]),
                                 h=jnp.asarray(h_i[:d10]), names=names10)

    def _reduced_anchor(pruned: tuple):
        """CPD-space reduction composed over the flagged edges + hub release, padded."""
        if pruned not in state["anchor_cache"]:
            net = sub.over
            for (p, c) in pruned:
                net = net.prune_edge(c, p)
            info = net.to_info()
            Pi_r = _pad_mat(np.asarray(info.Pi), n=n_slots)
            h_r = _pad_vec(np.asarray(info.h), n_slots)
            Pi_r, h_r = scale_block(Pi_r, h_r, np.array([idx10[HUB]]), hub_release)
            state["anchor_cache"][pruned] = (Pi_r, h_r)
        return state["anchor_cache"][pruned]

    def hook(t, Pi, h, Pi_prior, h_prior, w_obs, o):
        N = Pi.shape[0]
        if tele.crisis_step is None:
            tele.crisis_step = np.full(N, -1, dtype=int)
            tele.expand_step = np.full(N, -1, dtype=int)
            tele.prune_back_step = np.full(N, -1, dtype=int)
            tele.n_pruned = np.zeros(N, dtype=int)
            state["awake"] = np.zeros(N, dtype=bool)     # == expand_step >= 0 unless re-pinned
            state["wake_time"] = np.full(N, -1, dtype=int)
            state["audit_count"] = np.zeros(N, dtype=int)
            state["audit_n"] = np.zeros(N, dtype=int)    # audits since wake (credit decay)
            state["cand_done"] = np.zeros(N, dtype=bool)  # accepted a candidate arrival
        awake = state["awake"]

        mu = np.linalg.solve(Pi, h[..., None])[..., 0]               # (N, 11)
        err = o[:, dis_rows] - mu[:, dis_idx]                        # (N, k) node-space errors
        state["err"].append(err)
        if len(state["err"]) > window:
            state["err"].pop(0)

        # ---- crisis read-out: the model's own ledger, score = Delta F + lam * Delta U. Both
        # halves are read vs the static refs: exact for untouched anchors, interpretable
        # (evidence/value vs the ORIGINAL paradigm) for reduced ones, and INVALID once the
        # oxygen slot is wired (the pinned-block cancellation breaks) -- so expanded agents
        # are masked to NaN in the telemetry. ----
        Pi_j, h_j = jnp.asarray(Pi), jnp.asarray(h)
        dF = np.asarray(_dF_NE(Pi_j, h_j, Pi0, h0, Pi0r, h0r))       # (N, E)
        U = conviction_field(Pi_j, h_j, scn.names, scn.u, scn.conviction_alpha)
        dU = np.asarray(_dU_NE(Pi_j, h_j, Pi0, h0, Pi0r, h0r, U))    # (N, E)
        score = dF + lam[:, None] * dU
        dF_belt = dF[:, belt].min(axis=1)
        dF_belt[awake] = np.nan                  # pinned cancellation broken while wired
        tele.dF_belt_t.append(dF_belt)
        score_belt = score[:, belt].min(axis=1)
        score_belt[awake] = np.nan
        tele.score_belt_t.append(score_belt)
        upd_Pi_prior = upd_h_prior = upd_Pi = None

        if enable_reduction and t >= warmup:
            flagged = score > 0.0                                    # (N, E) ledger accept
            crisis_now = flagged[:, belt].all(axis=1) & (tele.crisis_step < 0)
            if crisis_now.any():
                upd_Pi_prior = Pi_prior.copy()
                upd_h_prior = h_prior.copy()
                for i in np.nonzero(crisis_now)[0]:
                    pruned = tuple(e for k, e in enumerate(edges) if flagged[i, k])
                    Pi_r, h_r = _reduced_anchor(pruned)
                    upd_Pi_prior[i] = Pi_r
                    upd_h_prior[i] = h_r
                    tele.crisis_step[i] = t
                    tele.n_pruned[i] = len(pruned)

        # ---- expansion: Poisson-arriving hub proposal -> Bayes-factor test (the sole gate).
        # The windowed errors only fix the proposal's DIRECTION (and the floor telemetry);
        # whether the residual loads on the hub is decided by ``expansion_score``. Strictly
        # post-crisis: at the crisis step itself the window still holds pre-reduction errors,
        # so the first attempt is the following step. ----
        floors = np.zeros(N)
        win_ok = enable_expansion and len(state["err"]) >= min(window, 10)
        E = np.stack(state["err"], axis=0) if win_ok else None       # (T, N, k)

        # ---- the principled rate (opt-in): per-agent rates from the Hawkes/adaptive state,
        # advanced once per step. Floors are computed for EVERY un-expanded agent (the
        # adaptive belief tracks inadequacy pre-crisis too); attempt draws (one per agent
        # per step, eligibility-independent) come from the dedicated gate stream. ----
        attempts = None
        props: dict = {}
        if rate_cfg is not None:
            if state["rate_state"] is None:
                state["rate_state"] = ProposalRateState(rate_cfg, N, trust_W)
            rstate = state["rate_state"]
            if win_ok:
                for i in range(N):
                    if awake[i]:
                        continue
                    net_i = _net10(Pi[i], h[i])
                    R = residual_from_errors(jnp.asarray(E[:, i, :]))
                    prop = propose_hub(net_i, DISAGREEMENT_NODES, residual=R)
                    props[i] = (net_i, prop)
                    floors[i] = prop.strength
            rates = rstate.step(t, floors)
            attempts = gate_rng.random(N) < rstate.attempt_probs(rates)
            tele.rate_t.append(rates)
            tele.attempt_t.append(attempts)

        upd_h = None
        if win_ok:
            for i in range(N):
                if (tele.crisis_step[i] < 0 and not allow_precrisis_apply) \
                        or tele.crisis_step[i] == t or awake[i] \
                        or (cand_cfg is not None and state["cand_done"][i]):
                    continue
                if rate_cfg is not None:
                    net_i, prop = props[i]
                    if not attempts[i]:
                        continue
                else:
                    net_i = _net10(Pi[i], h[i])
                    R = residual_from_errors(jnp.asarray(E[:, i, :]))
                    prop = propose_hub(net_i, DISAGREEMENT_NODES, residual=R)
                    floors[i] = prop.strength
                    if proposal_rate is not None and arr_rng.random() >= proposal_rate:
                        continue
                anchor_Pi = upd_Pi_prior[i] if upd_Pi_prior is not None else Pi_prior[i]
                anchor_h = upd_h_prior[i] if upd_h_prior is not None else h_prior[i]

                # ---- multi-candidate search (opt-in): several hubs (top-k eigenpairs,
                # one pre-allocated slot each) + new couplings among existing commitments
                # (largest residual entries), ALL priced on the same ledger; the best
                # positive (or every positive) candidate is applied. ----
                if cand_cfg is not None:
                    R_cand = residual_from_errors(jnp.asarray(E[:, i, :]))
                    cands = search_candidates(net_i, _net10(anchor_Pi, anchor_h),
                                              DISAGREEMENT_NODES, R_cand, cand_cfg,
                                              slot_names, hub_self_prec=HUB_SELF_PREC)
                    applied_hub = applied_any = False
                    for c in cands:
                        entry = dict(t=int(t), agent=int(i), kind=c.kind, target=c.target,
                                     dF=float(c.score.delta_F), dG=float(c.score.delta_G),
                                     accepted=False)
                        tele.candidate_log.append(entry)
                        if c.score.delta_F <= 0.0:
                            continue
                        if cand_cfg.accept == "best" and applied_any:
                            continue
                        if upd_Pi is None:
                            upd_Pi = Pi.copy()
                        if upd_Pi_prior is None:
                            upd_Pi_prior = Pi_prior.copy()
                            upd_h_prior = h_prior.copy()
                        if c.kind == "hub":
                            slot = d10 + slot_names.index(c.target)
                            cfull = np.zeros(d10)
                            cfull[dis_idx] = (c.score.detail["coupling_scale"]
                                              * np.asarray(c.payload.pattern))
                            for M in (upd_Pi, upd_Pi_prior):
                                M[i, slot, :d10] = cfull
                                M[i, :d10, slot] = cfull
                                M[i, slot, slot] = HUB_SELF_PREC
                            applied_hub = True
                            if wake_credit:      # register with the audit (the cycle)
                                state["wake_book"][i] = (
                                    np.array(anchor_Pi[:d10, :d10], dtype=np.float64),
                                    np.array(anchor_h[:d10], dtype=np.float64),
                                    c.payload,
                                    float(c.score.detail["coupling_scale"]), slot)
                        else:                                        # coupling
                            ca, cb, mag = c.payload
                            ai, bi = idx10[ca], idx10[cb]
                            for M in (upd_Pi, upd_Pi_prior):
                                M[i, ai, bi] += mag
                                M[i, bi, ai] += mag
                        entry["accepted"] = True
                        applied_any = True
                    if applied_any:
                        state["cand_done"][i] = True
                        if applied_hub:
                            awake[i] = True              # a wired slot breaks the pin read-out
                            state["wake_time"][i] = t
                            state["audit_count"][i] = 0
                            state["audit_n"][i] = 0
                            if state["rate_state"] is not None:
                                state["rate_state"].record_discovery(i, t)
                        if tele.expand_step[i] < 0:
                            tele.expand_step[i] = t
                    continue

                ms = expansion_score(net_i, _net10(anchor_Pi, anchor_h), prop, OXYGEN,
                                     hub_self_prec=HUB_SELF_PREC)
                # default: accept on the model log Bayes factor ALONE -- the data must hold
                # the woken hub (ms.accept folds in the epistemic gain delta_G, which is
                # positive for any new latent and pays for the attempt, not the keep).
                # wake_credit: accept on the FULL ledger Delta F + Delta G > 0 -- curiosity
                # wakes the hub on credit, and the per-round Savage-Dickey audit below is
                # what prunes a spurious wake back (the wake-then-prune cycle).
                accept = (ms.score > 0.0) if wake_credit else (ms.delta_F > 0.0)
                if not accept:
                    continue
                if wake_credit:                  # the audit re-asks this exact question
                    state["wake_book"][i] = (
                        np.array(anchor_Pi[:d10, :d10], dtype=np.float64),
                        np.array(anchor_h[:d10], dtype=np.float64),
                        prop, float(ms.detail["coupling_scale"]), d10)
                c10 = np.zeros(d10)
                c10[dis_idx] = ms.detail["coupling_scale"] * np.asarray(prop.pattern)
                if upd_Pi is None:
                    upd_Pi = Pi.copy()
                if upd_Pi_prior is None:
                    upd_Pi_prior = Pi_prior.copy()
                    upd_h_prior = h_prior.copy()
                for M in (upd_Pi, upd_Pi_prior):
                    M[i, d10, :d10] = c10
                    M[i, :d10, d10] = c10
                    M[i, d10, d10] = HUB_SELF_PREC                   # replace the PIN
                awake[i] = True
                state["wake_time"][i] = t
                state["audit_count"][i] = 0
                state["audit_n"][i] = 0
                if tele.expand_step[i] < 0:
                    tele.expand_step[i] = t                          # first wake (telemetry)
                if state["rate_state"] is not None:
                    state["rate_state"].record_discovery(i, t)
        tele.floor_t.append(floors)

        # ---- the wake-then-prune audit: every woken hub is re-scored each round in the
        # SAME currency that priced the wake -- the bordered-model log Bayes factor of the
        # agent's CURRENT (conditioned, 10-dim) beliefs against its PRE-wake anchor at the
        # stored coupling. (A pure Savage-Dickey prune of the wired anchor cannot work: the
        # wake wrote the coupling into the prior, and the slot has no observation row, so the
        # hub is held only by the data's mean displacement along its pattern -- which is
        # exactly what this score reads.) A genuine hub keeps earning the evidence as long as
        # the world keeps displacing the means; a spurious one's score decays to <= 0, and
        # ``prune_patience`` consecutive prune verdicts after ``prune_grace`` steps re-pin
        # the slot exactly (the agent may later wake again -- the cycle).
        if wake_credit:
            dFp = np.full(N, np.nan)
            for i in np.nonzero(awake)[0]:
                if t - state["wake_time"][i] < prune_grace or i not in state["wake_book"]:
                    continue
                aPi10, ah10, prop_i, scale_i, slot_i = state["wake_book"][i]
                ms_now = expansion_score(
                    _net10(Pi[i], h[i]),
                    GaussianBeliefNet(Pi=jnp.asarray(aPi10), h=jnp.asarray(ah10),
                                      names=names10),
                    prop_i, OXYGEN, hub_self_prec=HUB_SELF_PREC, coupling_scale=scale_i)
                # the verdict is the wake's own ledger with EXPIRING credit: the epistemic
                # gain paid for entertaining the structure, but it decays per audit, so the
                # data must take over before the credit runs out. (Without the decay the
                # credit covers any spurious wake forever; without the credit, the few
                # consolidation steps right after a genuine wake read as prune votes.)
                credit = float(ms_now.delta_G) * (credit_decay ** int(state["audit_n"][i]))
                state["audit_n"][i] += 1
                dFp[i] = ms_now.delta_F + credit
                state["audit_count"][i] = state["audit_count"][i] + 1 \
                    if dFp[i] <= 0.0 else 0
                if state["audit_count"][i] >= prune_patience:
                    if upd_Pi is None:
                        upd_Pi = Pi.copy()
                    if upd_Pi_prior is None:
                        upd_Pi_prior = Pi_prior.copy()
                        upd_h_prior = h_prior.copy()
                    if upd_h is None:
                        upd_h = h.copy()
                    for M in (upd_Pi, upd_Pi_prior):
                        M[i, slot_i, :] = 0.0
                        M[i, :, slot_i] = 0.0
                        M[i, slot_i, slot_i] = PIN                   # the original pinned form
                    upd_h[i, slot_i] = 0.0
                    upd_h_prior[i, slot_i] = 0.0
                    awake[i] = False
                    state["audit_count"][i] = 0
                    state["cand_done"][i] = False    # the cycle: eligible to wake again
                    del state["wake_book"][i]
                    if tele.prune_back_step[i] < 0:
                        tele.prune_back_step[i] = t
            tele.hub_dF_prune_t.append(dFp)

        # ---- the pre-crisis measurement: would the ledger accept an expansion BEFORE the
        # agent's own crisis? Run the identical pipeline without applying it. ----
        if precrisis_check and win_ok:
            pre_dF = np.full(N, np.nan)
            for i in range(N):
                if tele.crisis_step[i] >= 0 or awake[i]:
                    continue
                if i in props:
                    net_i, prop = props[i]
                else:
                    net_i = _net10(Pi[i], h[i])
                    R = residual_from_errors(jnp.asarray(E[:, i, :]))
                    prop = propose_hub(net_i, DISAGREEMENT_NODES, residual=R)
                ms = expansion_score(net_i, _net10(Pi_prior[i], h_prior[i]), prop, OXYGEN,
                                     hub_self_prec=HUB_SELF_PREC)
                pre_dF[i] = ms.delta_F
            tele.precrisis_dF_t.append(pre_dF)
        # EFFECTIVE oxygen coupling ||Pi[slot,:]|| / sqrt(Pi[slot,slot]): a fusion-borrowed
        # row next to a still-pinned diagonal is functionally inert and must not count.
        tele.oxy_coupling_t.append(
            np.linalg.norm(Pi[:, d10, :d10], axis=1) / np.sqrt(np.abs(Pi[:, d10, d10]) + 1e-12))

        out = {}
        if upd_Pi is not None:
            out["Pi"] = upd_Pi
        if upd_h is not None:
            out["h"] = upd_h
        if upd_Pi_prior is not None:
            out["Pi_prior"] = upd_Pi_prior
            out["h_prior"] = upd_h_prior
        if emit_fuse_mask:
            # who represents the oxygen dimension (consumed by posterior_masked fusion):
            # the 10 conceived dims for everyone, the slot only while the agent holds it.
            mask = np.ones((N, d10 + n_slots))
            mask[:, d10:] = awake.astype(float)[:, None]
            out["fuse_mask"] = mask
        return out or None

    return hook, tele


# ----------------------------------------------------------------------
# one population run
# ----------------------------------------------------------------------

def single_run(*, N: int = 80, inter: float = 0.05, intra: float = 0.4,
               frac_open: float = 0.5,
               lam_open: float = 0.10, lam_dogma: float = 0.35,
               gate_strength: float | np.ndarray = 1.0, gate_mode: str = "conviction",
               s_open: float = 0.3, s_dogma: float = 3.0,
               omega: float = 0.97, sigma_o: float = 0.5,
               t_shift: int = 40, t_reverse: int | None = None, n_steps: int = 180,
               proposal_rate: float | None = 0.08,
               hub_release: float = 0.25,
               enable_reduction: bool = True, enable_expansion: bool = True,
               rate_mode: str = "poisson", hawkes_beta: float = 0.0,
               hawkes_tau: float = 10.0, rate_kappa: float = 0.0,
               rate_floor_ema: float = 0.1, rate_floor_ref: float = 0.0,
               rate_max: float = 1.0,
               fuse_mode: str = "posterior",
               wake_credit: bool = False, prune_grace: int = 10, prune_patience: int = 3,
               credit_decay: float = 0.85, allow_precrisis_apply: bool = False,
               precrisis_check: bool = False,
               n_slots: int = 1, k_hubs: int = 1, top_couplings: int = 0,
               candidate_accept: str = "best",
               conviction_eps: float = 0.0, conviction_decay: float = 0.01,
               entrenchment_mode: str = "fisher_diag", conviction_gmax: float = 10.0,
               social_nu: float | None = None,
               trust_memory: float | None = None,
               seed: int = 0, snapshot_every: int = 2) -> dict:
    """N agents, two equal communities (open vs dogmatic), through the deterministic regime
    flip. ``proposal_rate=None`` => deterministic expansion attempts (headline mode);
    a float => Poisson-arriving proposals (discovery as a waiting time).

    ``rate_mode != "poisson"`` switches the arrival process to the principled rate of
    ``proposal_rate.py`` (per-agent, on a separate RNG stream): ``"hawkes"`` adds the
    mutually-exciting term on the trust graph (``hawkes_beta`` / ``hawkes_tau``),
    ``"adaptive"`` makes the base rate an updatable belief tracking the agent's own residual
    floor (``rate_kappa`` / ``rate_floor_ema`` / ``rate_floor_ref``), ``"hawkes_adaptive"``
    composes both. ``proposal_rate`` then sets the base rate ``r0`` and the results gain
    ``rate_tn`` / ``attempt_tn`` (T, N). The default is byte-identical to the legacy path."""
    cfg = StructuralConfig(sigma_o=sigma_o, t_shift=t_shift, n_steps=n_steps,
                           regime_schedule=("step" if t_reverse is None else "reversal"),
                           t_reverse=t_reverse,
                           observation_operator="relational")
    scn, sub = padded_scenario(cfg, n_slots=n_slots)
    cand_cfg = None
    if k_hubs != 1 or top_couplings != 0:
        cand_cfg = CandidateConfig(k_hubs=k_hubs, top_couplings=top_couplings,
                                   accept=candidate_accept)
    n_open = max(1, min(N - 1, round(N * frac_open)))   # the vanguard can be a minority
    sizes = [n_open, N - n_open]
    graph = graphs.community(sizes, intra=intra, inter=inter, seed=seed)
    community = np.concatenate([np.zeros(sizes[0], dtype=int), np.ones(sizes[1], dtype=int)])

    lam = np.where(community == 0, lam_open, lam_dogma)
    scale = np.where(community == 0, s_open, s_dogma)
    u11 = np.asarray(scn.u)                                          # (11,)
    u_agent = scale[:, None] * u11[None, :]                          # per-agent gate strength
    w = np.ones((N, scn.m))
    spec = AgentSpec(w_obs=jnp.asarray(w), lam=jnp.asarray(lam),
                     u_agent=jnp.asarray(u_agent, dtype=jnp.float32))

    rate_cfg = trust_W = None
    if rate_mode != "poisson":
        rate_cfg = RateConfig(
            mode=rate_mode,
            r0=(proposal_rate if proposal_rate is not None else 0.08),
            hawkes_beta=hawkes_beta, hawkes_tau=hawkes_tau,
            kappa=rate_kappa, floor_ema=rate_floor_ema, floor_ref=rate_floor_ref,
            r_max=rate_max)
        trust_W = np.asarray(graph.trust_W())

    hook, tele = make_revolution_hook(
        scn=scn, sub=sub, lam=lam, hub_release=hub_release,
        proposal_rate=proposal_rate, seed=seed,
        enable_reduction=enable_reduction, enable_expansion=enable_expansion,
        rate_cfg=rate_cfg, trust_W=trust_W,
        emit_fuse_mask=(fuse_mode == "posterior_masked"),
        wake_credit=wake_credit, prune_grace=prune_grace, prune_patience=prune_patience,
        credit_decay=credit_decay, allow_precrisis_apply=allow_precrisis_apply,
        precrisis_check=precrisis_check, cand_cfg=cand_cfg)
    conv_dyn = None
    if conviction_eps > 0.0:
        conv_dyn = ConvictionDynamics(eps=conviction_eps, decay=conviction_decay,
                                      entrenchment=entrenchment_mode,
                                      g_max=conviction_gmax)
    # CONTENT-GATED TRUST (opt-in): disagreement on the mass-law (disagreement) nodes
    # gates the fusion weights via reliability.social_gamma. None => static W, byte-identical.
    social_idx = (tuple(scn.names.index(n) for n in DISAGREEMENT_NODES)
                  if social_nu is not None else None)
    r = run_simulation(scn, graph, spec, fuse_mode=fuse_mode, forgetting=omega,
                       endogenous_gamma=bool(np.any(np.asarray(gate_strength) > 0)),
                       gate_strength=gate_strength, gate_mode=gate_mode,
                       social_nu=social_nu, social_idx=social_idx,
                       trust_memory=trust_memory,
                       host_hook=hook, conviction_dynamics=conv_dyn,
                       snapshot_every=snapshot_every, seed=seed)

    out = dict(r)
    out["community"] = community
    out["lam"] = lam
    out["v_e_belt"] = np.asarray(scn.v_e)[0][list(scn.belt_ix)]
    out["t_shift"] = t_shift
    out["crisis_step"] = tele.crisis_step
    out["expand_step"] = tele.expand_step
    out["n_pruned"] = tele.n_pruned
    out["dF_belt_tn"] = np.stack(tele.dF_belt_t)                     # (T, N)
    out["score_belt_tn"] = np.stack(tele.score_belt_t)               # (T, N) ledger score
    out["floor_tn"] = np.stack(tele.floor_t)
    out["oxy_coupling_tn"] = np.stack(tele.oxy_coupling_t)
    if tele.rate_t:                                                  # rate_cfg telemetry
        out["rate_tn"] = np.stack(tele.rate_t)                       # (T, N)
        out["attempt_tn"] = np.stack(tele.attempt_t)                 # (T, N) bool
    if wake_credit:
        out["prune_back_step"] = tele.prune_back_step                # (N,)
        out["hub_dF_prune_tn"] = np.stack(tele.hub_dF_prune_t)       # (T', N) audit trace
    if tele.precrisis_dF_t:
        out["precrisis_dF_tn"] = np.stack(tele.precrisis_dF_t)       # (T', N)
    if cand_cfg is not None:
        out["candidate_log"] = tele.candidate_log                    # list of dicts
    out["names"] = scn.names
    # per-community oxygen index over snapshots (means-level conversion read-out)
    S = r["snap_Pi"].shape[0]
    oxy = np.zeros((S, 2))
    for s in range(S):
        mu = np.linalg.solve(r["snap_Pi"][s], r["snap_h"][s][..., None])[..., 0]
        ix = [scn.names.index(n) for n in DISAGREEMENT_NODES]
        val = mu[:, ix].mean(axis=1)
        oxy_idx = np.clip((val - (-1.0)) / 2.0, 0.0, 1.0)            # mu_phlog=-1 -> 0, mu_oxy=1 -> 1
        for c in (0, 1):
            oxy[s, c] = oxy_idx[community == c].mean()
    out["oxy_index_sc"] = oxy
    return out
