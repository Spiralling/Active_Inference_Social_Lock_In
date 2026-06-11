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
* **Anomaly accumulation (telemetry).** When the world switches theory (epoch change at
  ``t1``), the belt re-fits its direct reads quickly, but the pinned core now CONTRADICTS the
  balance rows: a persistent residual appears on the disconfirming channel that no belt
  adjustment can absorb. Each agent integrates it as TELEMETRY:
  ``A_t = rho A_{t-1} + ||w_disc * (o_disc - H_disc mu)||^2`` (weighted by the agent's OWN
  effective channel weights, so the gated-vs-raw contrast remains visible).
* **Crisis -> revolution = the move ledger.** The crisis decision is not a hand-made
  accumulator threshold: each step the RELEASE of the core block (the mean-preserving
  congruence scaling of the forgetting anchor) is scored on the unified move ledger,
  ``score = Delta F + lam * Delta U``, with ``Delta F`` the Savage-Dickey evidence for
  swapping the agent's anchor for its core-released version under the agent's own accumulated
  likelihood, and ``Delta U`` the closed-form conviction-value change of that swap (the
  released posterior mean drifts off the cherished theory, so ``Delta U < 0``). ``score > 0``
  is crisis: the anchor is released, the core becomes revisable, and the standing relational
  evidence re-aligns it over the next ~1/(1-omega) steps. Nothing is scripted, and there is no
  free threshold: ``lam`` is the same conviction temperature as everywhere else in the paper.
* **Lock-in as a phase.** With conviction gating strong enough, the anomaly is *recorded by
  the world but never seen by the agent* (the raw accumulator climbs; the agent's gated
  deposit -- hence its own ``Delta F`` -- does not), the ledger never goes positive, and the
  population stays in degenerate normal science forever -- the Kuhnian loss-of-crisis
  failure mode.

Everything runs through the unmodified engine (``run_simulation`` + its ``host_hook``).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

from src.structural import graphs, linalg
from src.structural.dual_field import conviction_field
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
    score_t: list = field(default_factory=list)      # (T, N) release ledger dF + lam dU
    crisis_step: np.ndarray | None = None            # (N,) step of crisis, -1 = never


@jax.jit
def _release_ledger(Pi, h, Pi0, h0, Pi0r, h0r, U):
    """Per-agent ledger of the core-RELEASE move, ``(dF, dU)`` each (N,).

    ``(Pi0, h0)`` are the agents' current forgetting anchors, ``(Pi0r, h0r)`` their
    core-released versions (``scale_block``). ``dF`` is the Savage-Dickey evidence for the
    anchor swap under each agent's own accumulated likelihood ``(Pi - Pi0, h - h0)``; ``dU``
    is the closed-form conviction-value change, ``U_i . (mu_released - mu)``, read from the
    released posterior the same identity exhibits. The crisis rule is
    ``dF + lam * dU > 0`` -- the unified move ledger on a prior-release instead of an
    edge prune."""
    def one(P, q, P0, q0, Pr, qr, u):
        dF = linalg.savage_dickey(P, q, P0, q0, Pr, qr)[0]
        mu = jnp.linalg.solve(P, q[..., None])[..., 0]
        mu_r = jnp.linalg.solve(Pr + (P - P0), (qr + (q - q0))[..., None])[..., 0]
        return dF, u @ (mu_r - mu)
    return jax.vmap(one)(Pi, h, Pi0, h0, Pi0r, h0r, U)


def make_kuhn_hook(*, H: np.ndarray, disc_rows: tuple[int, ...], core_idx: np.ndarray,
                   names: tuple[str, ...], u_agent: np.ndarray, conviction_alpha: float,
                   lam: float = 0.2, rho: float = 0.95, release: float = 0.05,
                   crisis: bool = True, warmup: int = 15) -> tuple:
    """Returns ``(hook, telemetry)``. Each step the hook scores the core-release move on the
    unified ledger, ``score_i = dF_i + lam * dU_i`` (see ``_release_ledger``); the first
    ``score_i > 0`` (after ``warmup``) is agent i's crisis: its forgetting-anchor core
    precision is released by ``release`` -- the core becomes revisable and forgetting + the
    standing relational evidence do the rest. The gamma-gated and raw anomaly accumulators are
    kept as TELEMETRY (the gated-vs-world contrast of the lock-in story); they no longer gate
    anything. ``crisis=False`` never releases (the no-crisis control). ``warmup`` is the
    initialization guard: in the first ~dozen steps the small-sample ``Delta U`` noise is
    amplified by large ``lam`` and can cross zero spuriously; by ``t ~ 15`` the ledger has
    settled."""
    H_disc = np.asarray(H)[list(disc_rows)]          # (n_disc, d)
    u_agent = np.asarray(u_agent)
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

        # ---- the release ledger: dF + lam * dU vs each agent's CURRENT anchor ----
        Pi_rel = np.empty_like(np.asarray(Pi_prior))
        h_rel = np.empty_like(np.asarray(h_prior))
        for i in range(N):
            Pi_rel[i], h_rel[i] = scale_block(np.asarray(Pi_prior[i]), np.asarray(h_prior[i]),
                                              core_idx, release)
        U = conviction_field(jnp.asarray(Pi), jnp.asarray(h), names,
                             jnp.asarray(u_agent), conviction_alpha)
        dF, dU = _release_ledger(jnp.asarray(Pi), jnp.asarray(h),
                                 jnp.asarray(Pi_prior), jnp.asarray(h_prior),
                                 jnp.asarray(Pi_rel), jnp.asarray(h_rel), U)
        score = np.asarray(dF) + lam * np.asarray(dU)
        # an already-released agent's ledger keeps scoring a FURTHER release; mask to NaN
        score[tele.crisis_step >= 0] = np.nan
        tele.score_t.append(score)
        if not crisis or t < warmup:
            return None
        with np.errstate(invalid="ignore"):
            fresh = (score > 0.0) & (tele.crisis_step < 0)
        if not fresh.any():
            return None
        tele.crisis_step[fresh] = t
        Pi_p, h_p = np.asarray(Pi_prior).copy(), np.asarray(h_prior).copy()
        for i in np.nonzero(fresh)[0]:
            Pi_p[i], h_p[i] = Pi_rel[i], h_rel[i]
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

def single_run(*, gate_strength: float = 0.0, lam: float = 0.2, crisis: bool = True,
               seed: int = 0, N: int = 24, n_steps: int = 160, t1: int = 60,
               omega: float = 0.9, sigma_o: float = 0.5,
               core_stiffness: float = 8.0, release: float = 0.05, rho: float = 0.95,
               core_observed: bool = False, snapshot_every: int = 2) -> dict:
    """One population through one epoch switch (epoch 0 -> 1 at ``t1``; the third epoch is
    pushed past the horizon). ``lam`` is the conviction temperature on the release ledger
    (crisis iff ``dF + lam dU > 0``); ``crisis=False`` is the no-crisis control.
    ``core_observed=True`` is the full-observability control: no mask, no stiffening, no
    crisis -- the configuration whose missing staircase is the paper's Sec. 4.4 null."""
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
    spec = AgentSpec(w_obs=jnp.asarray(w), lam=jnp.full((N,), lam),
                     u_agent=jnp.asarray(u, dtype=jnp.float32))
    graph = graphs.complete(N)

    hook, tele = (make_kuhn_hook(H=np.asarray(scn.H), disc_rows=scn.disc_rows,
                                 core_idx=core_idx, names=names, u_agent=u,
                                 conviction_alpha=scn.conviction_alpha,
                                 lam=lam, rho=rho, release=release, crisis=crisis)
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
        out["score_tn"] = np.stack(tele.score_t)                  # (T, N) release ledger
        out["crisis_step"] = tele.crisis_step
    return out
