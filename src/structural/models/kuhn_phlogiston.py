"""The full Kuhnian revolution on the phlogiston world: population + BMR + BME, one loop.

Everything the paper develops, finally in one model:

* **Population** (``run_simulation``): N agents on a two-community SBM ("open" vs "dogmatic" --
  different prune thresholds ``lam_i`` and different conviction-gate strengths via per-agent
  ``u`` scaling), fusing precision over the trust graph, observing the gravimetric channel of a
  DETERMINISTIC truth schedule that flips from the phlogiston regime to the oxygen regime at
  ``t_shift`` (the precision balance arrives; calx is heavier).
* **Crisis = the model's own evidence** (no hand-made accumulator): per agent, the Savage-Dickey
  prune evidence ``Delta F`` on the two protective BELT edges crossing the conviction protection
  ``lam_i * v_e`` -- the engine's ``revolted`` criterion, now an act rather than a read-out.
* **Revolution = Bayesian model reduction APPLIED**: at crisis the agent's forgetting anchor is
  replaced by the CPD-space reduced prior (``over.prune_edge`` composed over every edge its own
  ``Delta F`` flags -- the same reduction family the evidence scored) with the phlogiston hub
  block released (mean-preserving precision scaling).
* **Discovery = Bayesian model expansion**: the basis carries an 11th, genuinely unconceived
  ``oxygen`` slot, pinned (huge self-precision, zero couplings, no observation row -- a hidden
  hub, like phlogiston). Post-crisis, each agent reads its windowed node-space prediction errors
  on the disagreement nodes (``action.residual_from_errors``), proposes a hub
  (``action.propose_hub``), and -- on a sustained residual floor AND a Poisson-arriving proposal
  -- scores the wake with the model log Bayes factor (``action.expansion_score``). Acceptance
  borders the oxygen slot in place (the agent's current net AND its anchor), replicating
  ``action.wake_hub``'s math in the pre-allocated dimension. Precision fusion then carries the
  awakened couplings to neighbours: the social diffusion of a new concept.

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
from src.structural.models.kuhn_cycle import scale_block
from src.structural.phlogiston import DISAGREEMENT_NODES, HUB, StructuralConfig, conviction_u
from src.structural.simulation import AgentSpec, Scenario, _dF_NE, run_simulation
from src.structural.scenarios import build_substrate, phlogiston_scenario, BELT

OXYGEN = "oxygen"
PIN = 1.0e4                  # pinned-slot self-precision (float32-safe; replaced on wake)
HUB_SELF_PREC = 2.0          # the woken hub's self-precision (matches cfg.hub_self_prec)
WINDOW = 30                  # trailing prediction-error window for the residual floor
SUSTAIN = 5                  # floor must hold this many consecutive steps
WARMUP = 5


# ----------------------------------------------------------------------
# the padded (10 -> 11) scenario: phlogiston + a pinned, unconceived oxygen slot
# ----------------------------------------------------------------------

def _pad_mat(M: np.ndarray, pin: float = PIN) -> np.ndarray:
    """(..., 10, 10) -> (..., 11, 11) with a decoupled pinned diagonal slot."""
    M = np.asarray(M, dtype=np.float64)
    out = np.zeros(M.shape[:-2] + (M.shape[-2] + 1, M.shape[-1] + 1), dtype=M.dtype)
    out[..., :-1, :-1] = M
    out[..., -1, -1] = pin
    return out


def _pad_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape[:-1] + (v.shape[-1] + 1,), dtype=v.dtype)
    out[..., :-1] = v
    return out


def padded_scenario(cfg: StructuralConfig, *, conviction: float = 2.0):
    """Build the phlogiston scenario, then pad every basis-dimension array with the pinned
    oxygen slot. Returns ``(scenario, substrate)`` -- the substrate's CPD net ``over`` is what
    the crisis-time reduction composes ``prune_edge`` on."""
    sub = build_substrate(cfg, conviction=conviction)
    scn = phlogiston_scenario(cfg, sub=sub, conviction=conviction)
    names11 = tuple(scn.names) + (OXYGEN,)
    H11 = np.concatenate([np.asarray(scn.H), np.zeros((scn.m, 1))], axis=1)
    order_fn = partial(_order11, names=names11, mass_nodes=DISAGREEMENT_NODES,
                       mu_phlog=cfg.mu_phlog_mass, mu_oxy=cfg.mu_oxy_mass)
    scn11 = Scenario(
        name="kuhn_phlogiston",
        names=names11,
        H=jnp.asarray(H11),
        disc_rows=scn.disc_rows,
        phis=jnp.asarray(_pad_vec(np.asarray(scn.phis))),
        epoch_t=scn.epoch_t,
        n_epochs=1,
        sigma_o=scn.sigma_o,
        Pi_base=jnp.asarray(_pad_mat(scn.Pi_base)),
        h_base=jnp.asarray(_pad_vec(scn.h_base)),
        Pi0=jnp.asarray(_pad_mat(scn.Pi0)),
        h0=jnp.asarray(_pad_vec(scn.h0)),
        Pi0r=jnp.asarray(_pad_mat(scn.Pi0r)),
        h0r=jnp.asarray(_pad_vec(scn.h0r)),
        v_e=scn.v_e,
        edges=scn.edges,
        belt_ix=scn.belt_ix,
        u=jnp.asarray(_pad_vec(scn.u)),
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
    crisis_step: np.ndarray | None = None      # (N,) first step both belt edges flagged; -1
    expand_step: np.ndarray | None = None      # (N,) first step oxygen bordered; -1
    dF_belt_t: list = field(default_factory=list)     # (T, N) min belt-edge Delta F
    floor_t: list = field(default_factory=list)       # (T, N) residual-floor strength
    oxy_coupling_t: list = field(default_factory=list)  # (T, N) ||oxygen couplings||
    n_pruned: np.ndarray | None = None          # (N,) edges pruned at crisis


def make_revolution_hook(*, scn: Scenario, sub, lam: np.ndarray,
                         hub_release: float = 0.25,
                         trigger: float = 1.5, window: int = WINDOW, sustain: int = SUSTAIN,
                         proposal_rate: float | None = 0.08, seed: int = 0,
                         enable_reduction: bool = True, enable_expansion: bool = True,
                         warmup: int = WARMUP):
    """Returns ``(hook, telemetry)``.

    Crisis & reduction: agent i's ``Delta F`` (vs the STATIC padded reference -- valid for
    agents whose anchor is still untouched) exceeds ``lam_i * v_e`` on BOTH belt edges =>
    anchor <- CPD-reduced prior over ALL its flagged edges, hub block released.
    Expansion: post-crisis, windowed node-space errors on the disagreement nodes -> hub
    proposal; sustained floor + Poisson arrival -> ``expansion_score``; accept => border the
    oxygen slot in the agent's current net AND anchor. ``proposal_rate=None`` = deterministic
    (attempt every eligible step) -- the deterministic-environment headline mode."""
    names10 = sub.over.names
    d10 = len(names10)
    idx10 = {n: i for i, n in enumerate(names10)}
    dis_idx = np.array([idx10[n] for n in DISAGREEMENT_NODES])       # basis indices (10-dim)
    dis_rows = dis_idx - 1                                           # H direct row of node i is i-1
    belt = np.asarray(scn.belt_ix, dtype=int)
    v_e = np.asarray(scn.v_e)[0]                                     # (E,)
    edges = scn.edges
    Pi0, h0 = scn.Pi0[0], scn.h0[0]
    Pi0r, h0r = scn.Pi0r[0], scn.h0r[0]
    lam = np.asarray(lam, dtype=np.float64)
    arr_rng = np.random.default_rng(900_001 + seed)                  # separate from world RNG

    tele = RevolutionTelemetry()
    state: dict = {"err": [], "streak": None, "anchor_cache": {}}

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
            Pi_r = _pad_mat(np.asarray(info.Pi))
            h_r = _pad_vec(np.asarray(info.h))
            Pi_r, h_r = scale_block(Pi_r, h_r, np.array([idx10[HUB]]), hub_release)
            state["anchor_cache"][pruned] = (Pi_r, h_r)
        return state["anchor_cache"][pruned]

    def hook(t, Pi, h, Pi_prior, h_prior, w_obs, o):
        N = Pi.shape[0]
        if tele.crisis_step is None:
            tele.crisis_step = np.full(N, -1, dtype=int)
            tele.expand_step = np.full(N, -1, dtype=int)
            tele.n_pruned = np.zeros(N, dtype=int)
            state["streak"] = np.zeros(N, dtype=int)

        mu = np.linalg.solve(Pi, h[..., None])[..., 0]               # (N, 11)
        err = o[:, dis_rows] - mu[:, dis_idx]                        # (N, k) node-space errors
        state["err"].append(err)
        if len(state["err"]) > window:
            state["err"].pop(0)

        # ---- crisis read-out: the model's own prune evidence. The static refs are exact for
        # untouched anchors, interpretable (evidence vs the ORIGINAL paradigm) for reduced
        # ones, and INVALID once the oxygen slot is wired (the pinned-block cancellation
        # breaks) -- so expanded agents are masked to NaN in the telemetry. ----
        dF = np.asarray(_dF_NE(jnp.asarray(Pi), jnp.asarray(h), Pi0, h0, Pi0r, h0r))  # (N,E)
        dF_belt = dF[:, belt].min(axis=1)
        dF_belt[tele.expand_step >= 0] = np.nan
        tele.dF_belt_t.append(dF_belt)
        upd_Pi_prior = upd_h_prior = upd_Pi = None

        if enable_reduction and t >= warmup:
            flagged = dF > (lam[:, None] * v_e[None, :])             # (N, E)
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

        # ---- expansion: residual floor -> hub proposal -> Poisson-gated Bayes-factor test ----
        floors = np.zeros(N)
        if enable_expansion and len(state["err"]) >= min(window, 10):
            E = np.stack(state["err"], axis=0)                       # (T, N, k)
            for i in range(N):
                if tele.crisis_step[i] < 0 or tele.expand_step[i] >= 0:
                    continue
                net_i = _net10(Pi[i], h[i])
                R = residual_from_errors(jnp.asarray(E[:, i, :]))
                prop = propose_hub(net_i, DISAGREEMENT_NODES, residual=R)
                floors[i] = prop.strength
                state["streak"][i] = state["streak"][i] + 1 if prop.strength >= trigger else 0
                if state["streak"][i] < sustain:
                    continue
                if proposal_rate is not None and arr_rng.random() >= proposal_rate:
                    continue
                anchor_Pi = upd_Pi_prior[i] if upd_Pi_prior is not None else Pi_prior[i]
                anchor_h = upd_h_prior[i] if upd_h_prior is not None else h_prior[i]
                ms = expansion_score(net_i, _net10(anchor_Pi, anchor_h), prop, OXYGEN,
                                     hub_self_prec=HUB_SELF_PREC)
                if not ms.accept:
                    continue
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
                tele.expand_step[i] = t
        tele.floor_t.append(floors)
        # EFFECTIVE oxygen coupling ||Pi[slot,:]|| / sqrt(Pi[slot,slot]): a fusion-borrowed
        # row next to a still-pinned diagonal is functionally inert and must not count.
        tele.oxy_coupling_t.append(
            np.linalg.norm(Pi[:, d10, :d10], axis=1) / np.sqrt(np.abs(Pi[:, d10, d10]) + 1e-12))

        out = {}
        if upd_Pi is not None:
            out["Pi"] = upd_Pi
        if upd_Pi_prior is not None:
            out["Pi_prior"] = upd_Pi_prior
            out["h_prior"] = upd_h_prior
        return out or None

    return hook, tele


# ----------------------------------------------------------------------
# one population run
# ----------------------------------------------------------------------

def single_run(*, N: int = 80, inter: float = 0.05, intra: float = 0.4,
               frac_open: float = 0.5,
               lam_open: float = 0.10, lam_dogma: float = 0.35,
               gate_strength: float = 1.0, s_open: float = 0.3, s_dogma: float = 3.0,
               omega: float = 0.97, sigma_o: float = 0.5,
               t_shift: int = 40, n_steps: int = 180,
               trigger: float = 1.5, proposal_rate: float | None = 0.08,
               hub_release: float = 0.25,
               enable_reduction: bool = True, enable_expansion: bool = True,
               seed: int = 0, snapshot_every: int = 2) -> dict:
    """N agents, two equal communities (open vs dogmatic), through the deterministic regime
    flip. ``proposal_rate=None`` => deterministic expansion attempts (headline mode);
    a float => Poisson-arriving proposals (discovery as a waiting time)."""
    cfg = StructuralConfig(sigma_o=sigma_o, t_shift=t_shift, n_steps=n_steps,
                           regime_schedule="step", observation_operator="relational")
    scn, sub = padded_scenario(cfg)
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

    hook, tele = make_revolution_hook(
        scn=scn, sub=sub, lam=lam, hub_release=hub_release, trigger=trigger,
        proposal_rate=proposal_rate, seed=seed,
        enable_reduction=enable_reduction, enable_expansion=enable_expansion)
    r = run_simulation(scn, graph, spec, fuse_mode="posterior", forgetting=omega,
                       endogenous_gamma=(gate_strength > 0), gate_strength=gate_strength,
                       host_hook=hook, snapshot_every=snapshot_every, seed=seed)

    out = dict(r)
    out["community"] = community
    out["lam"] = lam
    out["v_e_belt"] = np.asarray(scn.v_e)[0][list(scn.belt_ix)]
    out["t_shift"] = t_shift
    out["crisis_step"] = tele.crisis_step
    out["expand_step"] = tele.expand_step
    out["n_pruned"] = tele.n_pruned
    out["dF_belt_tn"] = np.stack(tele.dF_belt_t)                     # (T, N)
    out["floor_tn"] = np.stack(tele.floor_t)
    out["oxy_coupling_tn"] = np.stack(tele.oxy_coupling_t)
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
