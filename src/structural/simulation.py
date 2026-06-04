"""ONE reusable population-simulation engine; environments plug in as ``Scenario`` presets.

This is the de-tangled lift of ``scripts/run_multiagent_topology.py`` (nb43). There the
run-loop, the *substrate* (which over-wired Bayes net, which contested edges, what conviction
protection ``v_e``) and the *world* (which truth at each step) were welded into one function
``run_world``. Here they are split:

  * the **invariant** -- ``N`` agents on a trust graph who each step FUSE precision over the
    graph (``step.fuse``: ``Pi_i <- sum_j W_ij Pi_j``) then weighted-OBSERVE the world and
    deposit Fisher information -- lives once, in :func:`run_simulation`;
  * the **variant** -- node basis, the per-step world truth ``phi_at(t)``, the observation
    operator ``H``, the per-epoch reference priors the structural prune is read against, and
    the conviction utility -- is supplied by a :class:`Scenario`;
  * the **population heterogeneity** (per-agent self-censorship, conviction threshold, prior
    stiffness, and the conviction tilt strength) is a :class:`AgentSpec`.

So nb43's phlogiston experiment is ``run_simulation(phlogiston_scenario(...), graph, spec)``
and the new changing-cosmology experiment is the SAME call with ``cosmology_scenario(...)`` --
a different environment in the same engine, exactly the "underlying run-simulation class,
define new environments within it" the design asks for. The three jit kernels
(``_deposit_all`` / ``_dF_NE`` / ``_fuse_observe``) are lifted **verbatim** from nb43 so the
phlogiston scenario reproduces nb43's results **byte-for-byte** (the regression gate); the
only additions are (i) per-epoch reference priors for the snapshot read-out (a single epoch
recovers nb43) and (ii) an OPTIONAL per-agent conviction tilt ``h <- h + tilt_i U_i`` (off by
default -> byte-identical).

The per-step loop is an explicit Python loop (not ``jax.lax.scan``) -- mirroring nb43 -- because
the conviction-gated prune is a *variable* read-out (per agent, per edge, thresholded by the
per-epoch ``lambda_i v_e``) and host syncs are kept out of the loop (snapshots only).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from src.structural import graphs, linalg, shells
from src.structural.step import fuse
from src.structural.world import sample_o, fisher_deposit_weighted
from src.structural.dual_field import conviction_field
from src.structural.bayesnet import from_info as bn_from_info, to_info as bn_to_info

EPS = 1e-12


# ----------------------------------------------------------------------
# The environment contract: a Scenario.
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """Everything :func:`run_simulation` needs that is *world-specific*.

    The dynamics consume the first block; the snapshot read-out consumes the per-epoch
    reference priors; the conviction tilt consumes ``u`` / ``conviction_alpha``.

    Dynamics
      ``names``      : length-``d`` node basis (the shared fusion basis; order IS the matrix
                       basis).
      ``H``          : (m, d) observation operator. A *relational* row (a combination of
                       nodes) deposits off-diagonal Fisher so the edges LEARN; a node row is
                       a direct read. Held constant over the run (the world's *truth* moves,
                       not the operator).
      ``disc_rows``  : indices of the rows of ``H`` that carry the *disconfirming* channel --
                       the ones a self-censoring agent down-weights (see ``AgentSpec.w_obs``).
      ``phis``       : (T, d) the world truth ``phi_at(t)`` precomputed per step.
      ``epoch_t``    : (T,) the epoch index of each step (``0`` for a single-epoch world).
      ``n_epochs``   : number of distinct epochs ``E``.
      ``sigma_o``    : observation noise std.

    Initial belief
      ``Pi_base`` / ``h_base`` : (d, d) / (d,) the prior every agent starts from (scaled
                       per-agent by ``AgentSpec.precision_scale``). For phlogiston this is the
                       over-wired incumbent; for cosmology, epoch-0's theory.

    Per-epoch read-out reference (the prune is *re-armed* against the current epoch)
      ``Pi0`` / ``h0``   : (E, d, d) / (E, d) the reference prior of each epoch.
      ``Pi0r`` / ``h0r`` : (E, n_edges, d, d) / (E, n_edges, d) the per-edge *reduced* priors
                       (one contested edge pinned out), stacked so the Savage-Dickey prune
                       read-out vmaps over agents AND edges.
      ``v_e``        : (E, n_edges) conviction protection per edge per epoch.
      ``edges``      : the ``n_edges`` contested ``(parent, child)`` couplings (fixed basis
                       across epochs).
      ``belt_ix``    : indices (within ``edges``) of the BELT edges -- an agent has "revolted"
                       when it has pruned all of them.

    Conviction
      ``u``                : (d,) base intrinsic utility (propagated to ``U = T u`` per agent).
      ``conviction_alpha`` : propagation strength of the conviction field.

    Read-out
      ``order_fn`` : ``(Pi (N,d,d), h (N,d)) -> scalar`` the population order parameter.
      ``lstar``    : a reference conviction scale (reported, not used in the dynamics).
    """

    name: str
    names: tuple[str, ...]
    H: jax.Array
    disc_rows: tuple[int, ...]
    phis: jax.Array
    epoch_t: tuple[int, ...]
    n_epochs: int
    sigma_o: float
    Pi_base: jax.Array
    h_base: jax.Array
    Pi0: jax.Array
    h0: jax.Array
    Pi0r: jax.Array
    h0r: jax.Array
    v_e: jax.Array
    edges: tuple[tuple[str, str], ...]
    belt_ix: tuple[int, ...]
    u: jax.Array
    conviction_alpha: float
    order_fn: Callable
    lstar: float = 0.0

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def m(self) -> int:
        return int(self.H.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.phis.shape[0])

    @property
    def n_edges(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class AgentSpec:
    """Per-agent population heterogeneity (everything that is NOT the shared world).

    ``w_obs``           : (N, m) per-agent observation weights over the rows of ``H`` -- a
                          self-censoring agent down-weights ``disc_rows`` (it gathers *no*
                          information on the channel its current model would have refuted).
    ``lam``             : (N,) conviction threshold for the prune read-out: edge ``e`` is
                          pruned by agent ``i`` iff ``Delta F[i, e] > lam_i v_e``.
    ``precision_scale`` : (N,) prior precision (stiffness / conservatism) scale on
                          ``Pi_base`` -- a stiff prior moves last. ``None`` => 1 (the
                          homogeneous broadcast, byte-identical to nb43).
    ``tilt``            : (N,) per-agent conviction-tilt strength ``lambda`` for the lever
                          ``h <- h + lambda_i U_i`` applied each step. ``None`` => no tilt
                          (byte-identical to nb43).
    ``u_agent``         : (N, d) per-agent intrinsic utility for the tilt (each community can
                          favour its OWN theory). ``None`` => broadcast the scenario's ``u``.
    """

    w_obs: jax.Array
    lam: jax.Array
    precision_scale: jax.Array | None = None
    tilt: jax.Array | None = None
    u_agent: jax.Array | None = None


# ----------------------------------------------------------------------
# Vmapped + jit'd kernels -- lifted VERBATIM from nb43 (host syncs out of the loop).
# ----------------------------------------------------------------------

@jax.jit
def _deposit_all(H, phi, keys, sigma_o, w_obs):
    """One round of evidence for every agent (vectorised): agent ``i`` draws
    ``o_i = H phi + eps`` (own noise) and deposits ``J_i = H^T diag(w_i) H / sigma^2``,
    with ``w_i`` down-weighting the disconfirming rows by its self-censorship.
    ``keys`` (N,2), ``w_obs`` (N,m). Returns ``(J_all, j_all)`` (N,d,d),(N,d)."""
    def one(key, w):
        o = sample_o(H, phi, sigma_o, key)
        return fisher_deposit_weighted(H, o, sigma_o, w)
    return jax.vmap(one)(keys, w_obs)


@jax.jit
def _dF_NE(Pi, h, Pi0, h0, Pi0r, h0r):
    """Per-agent, per-edge prune evidence ``Delta F[i, e]`` (N, E) via Savage-Dickey.

    The shared likelihood is inferred as ``Pi_i - Pi0`` (the agent's diffused accumulated
    evidence), so this reads the prune evidence of each FUSED agent against the (current
    epoch's) reference prior. Double vmap: over edges (the per-edge reduced prior) and over
    agents."""
    def per_edge(Pr, hr):
        return jax.vmap(lambda P, q: linalg.savage_dickey(P, q, Pi0, h0, Pr, hr)[0])(Pi, h)
    return jax.vmap(per_edge)(Pi0r, h0r).T            # (N, E)


@jax.jit
def _cpd_fuse(Pi, h, W):
    """Fuse in DIRECTED-CPD space: communicate the Bayes NET, not the precision.

    Each agent's belief is read back as its directed Bayes net via the exact CPD<->precision
    bijection (``bayesnet.from_info`` -> per-node parent weights ``B``, intercepts ``b``,
    residual variances ``s``); the CPDs are then trust-weight-averaged over the graph and
    recompiled (``bayesnet.to_info``). Because the bijection is nonlinear, averaging the
    directed CPD edge weights ``B`` is genuinely different from averaging the symmetric
    precision ``Pi`` (the ``posterior`` mode) -- this is "agents share their Bayes nets".
    Requires each agent's ``Pi`` positive-definite (true for prior + Fisher), and a shared
    parents-before-children node order (the fixed scenario basis)."""
    B, b, s = jax.vmap(bn_from_info)(Pi, h)               # (N,d,d),(N,d),(N,d)
    B_f = jnp.einsum("ij,jab->iab", W, B)                 # average directed edge weights
    b_f = W @ b
    s_f = W @ s                                           # variances stay positive
    return jax.vmap(bn_to_info)(B_f, b_f, s_f)


@partial(jax.jit, static_argnames=("mode",))
def _fuse_observe(Pi, h, J_all, j_all, W, Woff, eta, mode):
    """One round of the population dynamics for the chosen ``fuse_mode``.

    ``posterior``    : fuse whole nets (``Pi_i <- sum_j W_ij Pi_j``) THEN add own deposit --
                       multi-hop diffusion; identical priors fuse idempotently, so the prior
                       is preserved and only the deposits diffuse (the load-bearing mode).
    ``bayesnet``     : communicate the directed Bayes net -- average the per-agent CPDs
                       (``B``, ``b``, ``s``) over the graph (``_cpd_fuse``) THEN add own deposit.
                       Shares the directed structure rather than the symmetric precision.
    ``deposit_pool`` : never fuse the net; pool only this step's deposits
                       (``Pi_i += sum_j W_ij J_j``) -- 1-hop, prior never averaged in.
    ``deposit_keep`` : keep your own deposit whole + ``eta * sum_{j!=i} What_ij J_j``
                       (``Woff`` row-stochastic off-diagonal) -- the cleanest no-wash control.
    """
    if mode == "posterior":
        Pi_f, h_f = fuse(Pi, h, W)
        return Pi_f + J_all, h_f + j_all
    if mode == "bayesnet":
        Pi_f, h_f = _cpd_fuse(Pi, h, W)
        return Pi_f + J_all, h_f + j_all
    if mode == "deposit_pool":
        return (Pi + jnp.einsum("ij,jab->iab", W, J_all), h + W @ j_all)
    # deposit_keep
    return (Pi + J_all + eta * jnp.einsum("ij,jab->iab", Woff, J_all),
            h + j_all + eta * (Woff @ j_all))


# ----------------------------------------------------------------------
# THE ENGINE.
# ----------------------------------------------------------------------

def run_simulation(scenario: Scenario, graph: graphs.Graph, spec: AgentSpec, *,
                   fuse_mode: str = "posterior", eta: float = 0.5,
                   forgetting: float = 1.0,
                   endogenous_gamma: bool = False, gate_strength: float = 1.0,
                   w_floor: float = 0.0, deposit_gate: jax.Array | None = None,
                   snapshot_every: int = 5, seed: int = 0) -> dict:
    """Simulate ``N = graph.n`` agents pooling precision over ``graph`` in ``scenario``'s
    world, reading out the conviction-gated structural prune per agent each snapshot.

    Each round: FUSE the agents' belief nets over the trust graph (``fuse_mode``), then each
    agent draws an observation of the current world truth ``scenario.phis[t]`` and deposits
    its (self-censorship-weighted) Fisher information. An OPTIONAL conviction tilt
    ``h <- h + tilt_i U_i`` follows (``spec.tilt`` -> the "block revision" lever). The prune
    is a *read-out* of where each fused agent lands -- it never mutates the net -- re-armed
    against the *current epoch's* reference prior so "structure that fit epoch 0 but not
    epoch 1" reads as stale.

    Returns a dict of host arrays (snapshots every ``snapshot_every`` steps + the final step);
    see the keys at the bottom. ``snap_Pi`` / ``snap_h`` carry the full per-agent nets so the
    scenario-specific read-outs (theory tracking, structural match) are computed host-side."""
    N, d, m = graph.n, scenario.dim, scenario.m
    n_steps, n_edges, n_epochs = scenario.n_steps, scenario.n_edges, scenario.n_epochs

    W = graph.trust_W()                                  # (N,N) row-stochastic
    Woff = np.asarray(W) * (1.0 - np.eye(N))
    Woff = jnp.asarray(Woff / (Woff.sum(axis=1, keepdims=True) + EPS))

    # initial belief: the scenario's base prior, scaled per agent by conservatism.
    if spec.precision_scale is None:
        Pi = jnp.broadcast_to(scenario.Pi_base, (N, d, d))
        h = jnp.broadcast_to(scenario.h_base, (N, d))
    else:
        ps = jnp.asarray(spec.precision_scale)           # (N,)
        Pi = ps[:, None, None] * scenario.Pi_base[None]  # (N,d,d) stiffer prior, same mean
        h = ps[:, None] * scenario.h_base[None]          # (N,d)

    # The forgetting anchor: each agent's *prior* (Pi, h at t=0). With forgetting < 1 the
    # accumulated evidence (everything beyond the prior) relaxes back toward it each step --
    # an exponential forgetting factor omega in (0, 1] on the accumulated concentration
    # (paper Sec. 3.3), so a once-heavy epoch's evidence leaks at rate omega and the system
    # can re-track a moving world instead of ratcheting. omega = 1 => no relaxation (the
    # accumulation default; byte-identical to the original engine / nb43).
    Pi_prior, h_prior = Pi, h
    omega = float(forgetting)

    w_obs = jnp.asarray(spec.w_obs)
    lam = np.asarray(spec.lam)                            # (N,)
    tilt = None if spec.tilt is None else jnp.asarray(spec.tilt)
    u_agent = scenario.u if spec.u_agent is None else jnp.asarray(spec.u_agent)
    H, phis, sigma_o = scenario.H, scenario.phis, float(scenario.sigma_o)

    # ENDOGENOUS gamma (E2): the conviction field silences its OWN disconfirming channels.
    # Each step w_obs on the disconfirming rows is recomputed from a saturating map of the
    # conviction projected onto those rows, w_disc = exp(-gate_strength |U . H_disc|), so high
    # conviction drives the disconfirming sensory weight (hence gamma) up endogenously. Off by
    # default => w_obs static => byte-identical. The DEPOSIT GATE (E1): a per-node revision-rate
    # gate g (d,) scales the Fisher deposit PD-preservingly (J <- G^.5 J G^.5, j <- G^.5 j) so
    # high-conservatism nodes integrate evidence slower. None => byte-identical.
    w_obs_base = w_obs
    disc_arr = np.asarray(scenario.disc_rows, dtype=int)
    disc_idx = jnp.asarray(disc_arr)
    gate_sqrt = None if deposit_gate is None else jnp.sqrt(jnp.asarray(deposit_gate))
    Pi0, h0, Pi0r, h0r, v_e = (scenario.Pi0, scenario.h0, scenario.Pi0r,
                               scenario.h0r, np.asarray(scenario.v_e))
    epoch_t = scenario.epoch_t
    belt_ix = np.asarray(scenario.belt_ix, dtype=int)

    key = jax.random.PRNGKey(seed)
    snap_t, kept_t, revolted_t, m_t, snap_Pi, snap_h, gamma_t = [], [], [], [], [], [], []
    last_dF = last_pruned = None

    def snapshot(t):
        e = int(epoch_t[t])
        dF = np.asarray(_dF_NE(Pi, h, Pi0[e], h0[e], Pi0r[e], h0r[e]))   # (N,E)
        protect = lam[:, None] * v_e[e][None, :]                          # (N,E)
        pruned = dF > protect
        kept_t.append((~pruned).sum(axis=1))
        revolted_t.append(pruned[:, belt_ix].all(axis=1))
        m_t.append(float(scenario.order_fn(Pi, h)))
        snap_Pi.append(np.asarray(Pi))
        snap_h.append(np.asarray(h))
        # realised gamma = mean fraction of disconfirming sensory weight removed (eq:gamma)
        gamma_t.append(float(np.mean(1.0 - np.asarray(w_obs)[:, disc_arr]))
                       if disc_arr.size else 0.0)
        snap_t.append(t)
        return dF, pruned

    for t in range(n_steps):
        phi = phis[t]
        if omega < 1.0:                       # relax accumulated evidence toward the prior
            Pi = Pi_prior + omega * (Pi - Pi_prior)
            h = h_prior + omega * (h - h_prior)
        if endogenous_gamma and disc_arr.size:   # conviction silences its own disc channels (E2)
            Ug = conviction_field(Pi, h, scenario.names, u_agent, scenario.conviction_alpha)
            proj = jnp.abs(Ug @ H[disc_idx].T)                          # (N, n_disc)
            w_disc = jnp.clip(jnp.exp(-gate_strength * proj), w_floor, 1.0)
            w_obs = w_obs_base.at[:, disc_idx].set(w_disc)
        key, sk = jax.random.split(key)
        keys = jax.random.split(sk, N)
        J_all, j_all = _deposit_all(H, phi, keys, sigma_o, w_obs)
        if gate_sqrt is not None:                # conservatism gates the revision rate (E1)
            J_all = gate_sqrt[None, :, None] * J_all * gate_sqrt[None, None, :]
            j_all = gate_sqrt[None, :] * j_all
        Pi, h = _fuse_observe(Pi, h, J_all, j_all, W, Woff, eta, fuse_mode)
        if tilt is not None:                            # the conviction lever (off by default)
            U = conviction_field(Pi, h, scenario.names, u_agent, scenario.conviction_alpha)
            h = h + tilt[:, None] * U
        if (t % snapshot_every == 0) or (t == n_steps - 1):
            last_dF, last_pruned = snapshot(t)

    snap_Pi_arr = np.stack(snap_Pi)                                       # (S,N,d,d)
    disagreement_t = shells.residual_disagreement(snap_Pi_arr)           # (S,)

    return {
        "snap_t": np.asarray(snap_t),
        "kept_t": np.stack(kept_t),                       # (S, N)
        "revolted_t": np.stack(revolted_t),               # (S, N) bool
        "m_t": np.asarray(m_t),                           # (S,)
        "gamma_t": np.asarray(gamma_t),                   # (S,) realised disc-weight removed
        "disagreement_t": np.asarray(disagreement_t),     # (S,)
        "snap_Pi": snap_Pi_arr,                           # (S, N, d, d)
        "snap_h": np.stack(snap_h),                       # (S, N, d)
        "final_revolted_fraction": float(np.stack(revolted_t)[-1].mean()),
        "final_committed": np.asarray(~last_pruned),      # (N, E) kept
        "final_dF": np.asarray(last_dF),                  # (N, E)
        "lambda2": graphs.algebraic_connectivity(graph),
        "mean_degree": float(np.asarray(graph.A).sum(axis=1).mean()),
        "membership": (None if graph.membership is None
                       else np.asarray(graph.membership)),
        "lam": lam,
        "graph_kind": graph.kind,
        "fuse_mode": fuse_mode,
        "edges": [f"{p}->{c}" for (p, c) in scenario.edges],
        "belt_ix": np.asarray(scenario.belt_ix),
        "v_e": v_e[0],                                    # epoch-0 (nb43 single-epoch compat)
        "v_e_epochs": v_e,                                # (E, n_edges) full
        "epoch_t": np.asarray(epoch_t),
        "lstar": scenario.lstar,
        "n_edges": n_edges,
    }
