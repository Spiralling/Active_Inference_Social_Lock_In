"""The Kuhn cycle, endogenously: core observability + anomaly-driven crisis.

This is the design REWRITE_PLAN.md specifies to fix the paper's honest negative (Sec. 4.4):
the belt-first/core-last staircase failed because every node was DIRECTLY observed -- the
operator-set Fisher deposit flattens any conservatism geometry. Here the geometry is put where
the paper says it belongs: in *observability*.

* **Core = the three theory commitments.** Their direct observation rows are masked
  (``AgentSpec.w_obs = 0``), so evidence reaches the core ONLY through the three relational
  ``balance(commitment) = commitment - anomaly1 - anomaly2`` rows -- by propagation from the
  observed belt (the two anomaly nodes + law_coherence), never directly. The core prior is
  additionally stiffened (high precision = the paradigm's hard core).
* **Normal science.** Under the incumbent epoch the belt tracks the data and the balance rows
  are satisfied; the residual on the disconfirming channel stays at noise level.
* **Anomaly accumulation.** When the world switches theory (epoch change at ``t1``), the belt
  re-fits its direct reads quickly, but the pinned core now CONTRADICTS the balance rows: a
  persistent residual appears on the disconfirming channel that no belt adjustment can absorb.
  Each agent integrates it: ``A_t = rho A_{t-1} + ||w_disc * (o_disc - H_disc mu)||^2``.
  The residual is weighted by the agent's OWN effective channel weights: a conviction-gated
  agent (endogenous gamma) silences exactly the rows that feed its crisis detector.
* **Crisis -> revolution.** When ``A_t`` crosses ``theta`` the agent's protective belt has
  failed: the *forgetting anchor's* core-block precision is released (a mean-preserving
  congruence scaling), the core becomes revisable, and the already-present relational evidence
  re-aligns it over the next ~1/(1-omega) steps. Nothing is scripted: the timing of the crisis
  and the revolution are products of the dynamics.
* **Lock-in as a phase.** With conviction gating strong enough, the anomaly is *recorded by
  the world but never seen by the agent* (the unweighted accumulator climbs, the weighted one
  does not), theta is never crossed, and the population stays in degenerate normal science
  forever -- the Kuhnian loss-of-crisis failure mode.

Everything runs through the unmodified engine (``run_simulation`` + its ``host_hook``).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from src.structural import graphs
from src.structural.simulation import AgentSpec, run_simulation
from src.structural.scenarios import (
    COSMOLOGY_EPOCHS,
    _ANOMALIES,
    _COMMITMENTS,
    cosmology_scenario,
    cosmology_theory_means,
    cosmology_utility_toward,
)

#: The Kuhnian hard core: the three theory commitments. (NOT ``basis.core_nodes``, which in
#: the landscape preset names the *observed* coherence/anomaly nodes -- the opposite reading.)
KUHN_CORE = _COMMITMENTS
KUHN_BELT = _ANOMALIES


# ----------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------

def core_indices(names: tuple[str, ...]) -> np.ndarray:
    return np.array([names.index(n) for n in KUHN_CORE], dtype=int)


def belt_indices(names: tuple[str, ...]) -> np.ndarray:
    return np.array([names.index(n) for n in KUHN_BELT], dtype=int)


def masked_w_obs(n_agents: int, m: int, core_idx: np.ndarray) -> np.ndarray:
    """(N, m) observation weights with the core nodes' DIRECT rows masked to zero. Rows 0..d-1
    of the cosmology H are the direct reads in basis order, so direct row of node i is row i;
    the relational balance rows (d..m-1) stay open."""
    w = np.ones((n_agents, m), dtype=np.float64)
    w[:, core_idx] = 0.0
    return w


def scale_block(Pi: np.ndarray, h: np.ndarray, idx: np.ndarray, scale: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """Mean-preserving congruence scaling of a node block: ``Pi' = D Pi D`` with
    ``D = diag(sqrt(scale) on idx, 1 elsewhere)`` and ``h' = Pi' mu`` so the implied mean is
    unchanged. PD-safe (congruence with a positive diagonal). ``scale > 1`` stiffens the block
    (the paradigm's hard core); ``scale < 1`` releases it (the crisis move)."""
    d = Pi.shape[-1]
    D = np.ones(d)
    D[idx] = np.sqrt(scale)
    mu = np.linalg.solve(Pi, h)
    Pi2 = D[:, None] * Pi * D[None, :]
    return Pi2, Pi2 @ mu


# ----------------------------------------------------------------------
# the crisis hook
# ----------------------------------------------------------------------

@dataclass
class KuhnTelemetry:
    A_t: list = field(default_factory=list)          # (T, N) weighted anomaly accumulator
    A_raw_t: list = field(default_factory=list)      # (T, N) unweighted (what the world says)
    crisis_step: np.ndarray | None = None            # (N,) step of crisis, -1 = never


def make_kuhn_hook(*, H: np.ndarray, disc_rows: tuple[int, ...], core_idx: np.ndarray,
                   rho: float = 0.95, theta: float = np.inf, release: float = 0.05,
                   warmup: int = 5) -> tuple:
    """Returns ``(hook, telemetry)``. The hook accumulates the gamma-gated disconfirming
    residual per agent and, on first crossing ``theta`` (after ``warmup``), releases that
    agent's forgetting-anchor core precision by ``release`` -- the core becomes revisable and
    forgetting + the standing relational evidence do the rest. ``theta = inf`` never fires
    (the no-crisis control)."""
    H_disc = np.asarray(H)[list(disc_rows)]          # (n_disc, d)
    tele = KuhnTelemetry()
    state = {"A": None, "A_raw": None}

    def hook(t, Pi, h, Pi_prior, h_prior, w_obs, o):
        N = Pi.shape[0]
        if state["A"] is None:
            state["A"] = np.zeros(N)
            state["A_raw"] = np.zeros(N)
            tele.crisis_step = np.full(N, -1, dtype=int)
        mu = np.linalg.solve(Pi, h[..., None])[..., 0]            # (N, d)
        r = o[:, list(disc_rows)] - mu @ H_disc.T                 # (N, n_disc)
        w = w_obs[:, list(disc_rows)]                             # effective (gamma-gated)
        state["A"] = rho * state["A"] + np.sum((w * r) ** 2, axis=1)
        state["A_raw"] = rho * state["A_raw"] + np.sum(r ** 2, axis=1)
        tele.A_t.append(state["A"].copy())
        tele.A_raw_t.append(state["A_raw"].copy())
        if not np.isfinite(theta) or t < warmup:
            return None
        fresh = (state["A"] > theta) & (tele.crisis_step < 0)
        if not fresh.any():
            return None
        tele.crisis_step[fresh] = t
        Pi_p, h_p = Pi_prior.copy(), h_prior.copy()
        for i in np.nonzero(fresh)[0]:
            Pi_p[i], h_p[i] = scale_block(Pi_p[i], h_p[i], core_idx, release)
        return {"Pi_prior": Pi_p, "h_prior": h_p}

    return hook, tele


# ----------------------------------------------------------------------
# progress curves (the staircase read-out)
# ----------------------------------------------------------------------

def progress_curves(snap_Pi: np.ndarray, snap_h: np.ndarray, names: tuple[str, ...],
                    epoch_from: int = 0, epoch_to: int = 1) -> dict:
    """Population-mean revision progress on the old->new theory axis, stratified by shell:
    ``p_S(t) = clip(<mu_S(t) - mu0_S, mu1_S - mu0_S> / ||mu1_S - mu0_S||^2, 0, 1)``.
    0 = still the old theory's marginal means on that shell, 1 = arrived at the new one's."""
    theory_mu = cosmology_theory_means()                          # (E, d)
    mu0, mu1 = theory_mu[epoch_from], theory_mu[epoch_to]
    S = snap_Pi.shape[0]
    mu_t = np.stack([np.linalg.solve(snap_Pi[s], snap_h[s][..., None])[..., 0].mean(axis=0)
                     for s in range(S)])                          # (S, d)
    out = {}
    for label, idx in (("belt", belt_indices(names)), ("core", core_indices(names))):
        delta = mu1[idx] - mu0[idx]
        denom = float(delta @ delta)
        out[label] = np.clip((mu_t[:, idx] - mu0[idx]) @ delta / (denom + 1e-12), 0.0, 1.0)
        out[f"{label}_axis_norm"] = float(np.sqrt(denom))
    return out


def crossing_time(progress: np.ndarray, snap_t: np.ndarray, level: float = 0.5) -> int:
    """First STEP (not snapshot index) at which the progress curve crosses ``level``;
    -1 if never."""
    hit = np.nonzero(progress >= level)[0]
    return int(snap_t[hit[0]]) if len(hit) else -1


# ----------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------

def single_run(*, gate_strength: float = 0.0, theta: float = np.inf,
               seed: int = 0, N: int = 24, n_steps: int = 160, t1: int = 60,
               omega: float = 0.9, sigma_o: float = 0.5,
               core_stiffness: float = 8.0, release: float = 0.05, rho: float = 0.95,
               core_observed: bool = False, snapshot_every: int = 2) -> dict:
    """One population through one epoch switch (epoch 0 -> 1 at ``t1``; the third epoch is
    pushed past the horizon). ``core_observed=True`` is the full-observability control: no
    mask, no stiffening, no crisis -- the configuration whose missing staircase is the paper's
    Sec. 4.4 null."""
    scn = cosmology_scenario(n_steps=n_steps, t1=t1, t2=10 * n_steps, sigma_o=sigma_o)
    names, d, m = scn.names, scn.dim, scn.m
    core_idx = core_indices(names)

    if not core_observed:
        Pi_base, h_base = scale_block(np.asarray(scn.Pi_base), np.asarray(scn.h_base),
                                      core_idx, core_stiffness)
        scn = dataclasses.replace(scn, Pi_base=jnp.asarray(Pi_base), h_base=jnp.asarray(h_base))
        w = masked_w_obs(N, m, core_idx)
    else:
        w = np.ones((N, m))

    u = np.tile(np.asarray(cosmology_utility_toward(COSMOLOGY_EPOCHS[0])), (N, 1))
    spec = AgentSpec(w_obs=jnp.asarray(w), lam=jnp.full((N,), 0.2),
                     u_agent=jnp.asarray(u, dtype=jnp.float32))
    graph = graphs.complete(N)

    hook, tele = (make_kuhn_hook(H=np.asarray(scn.H), disc_rows=scn.disc_rows,
                                 core_idx=core_idx, rho=rho, theta=theta, release=release)
                  if not core_observed else (None, None))
    r = run_simulation(scn, graph, spec, forgetting=omega,
                       endogenous_gamma=(gate_strength > 0), gate_strength=gate_strength,
                       host_hook=hook, snapshot_every=snapshot_every, seed=seed)

    prog = progress_curves(r["snap_Pi"], r["snap_h"], names)
    out = dict(r)
    out["belt_prog_t"] = prog["belt"]
    out["core_prog_t"] = prog["core"]
    out["t_belt"] = crossing_time(prog["belt"], r["snap_t"])
    out["t_core"] = crossing_time(prog["core"], r["snap_t"])
    out["t1"] = t1
    out["names"] = names
    if tele is not None:
        out["A_tn"] = np.stack(tele.A_t)                          # (T, N)
        out["A_raw_tn"] = np.stack(tele.A_raw_t)
        out["crisis_step"] = tele.crisis_step
    return out
