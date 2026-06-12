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
the conviction-gated prune is a *variable* read-out (per agent, per edge, scored on the unified
move ledger ``dF + lambda dU``) and host syncs are kept out of the loop (snapshots only).
NOTE on the regression gate: the three jit kernels and the belief DYNAMICS remain byte-identical
to nb43; the prune READ-OUT (``kept_t`` / ``revolted_t``) was upgraded from the proxy threshold
``dF > lambda v_e`` to the closed-form ledger ``dF + lambda dU > 0`` (see ``_dU_NE``), so those
two telemetry arrays differ from nb43 by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from src.structural import graphs, linalg, shells
from src.structural import reliability as rel
from src.structural.step import fuse, trust_weights
from src.structural.world import sample_o, fisher_deposit_weighted
from src.structural.dual_field import conviction_field, cost_field_live
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

    Applied reduction (only consumed when ``run_simulation(bmr_every=...)`` is set)
      ``reduce_fn`` : ``(epoch: int, flags: (n_edges,) bool) -> (Pi_red (d,d), h_red (d,))``
                      the JOINT reduced reference prior with all flagged edges removed at
                      once. Needed because per-edge reductions need not compose: when two
                      contested edges share a child (the phlogiston belt), summing the
                      single-edge prior deltas double-subtracts the co-parent fill-in term.
                      ``None`` => the engine falls back to the per-edge delta sum
                      ``Pi0[e] + sum_flagged (Pi0r[e,k] - Pi0[e])``, which is exact iff the
                      per-edge reductions are disjoint (true for cosmology's
                      ``zero_edge_prior`` edits).
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
    reduce_fn: Callable | None = None

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
class ConvictionDynamics:
    """Dynamic intrinsic utility ``u`` (the Lakatos closure of the conviction field).

    The paper's limitations name the commitment: "the conviction source ``u`` is exogenous
    and static. Letting ``u`` accrete onto structurally entrenched commitments -- value
    aligning with conservatism as a programme matures -- would recover Lakatos's progressive
    and degenerating problemshifts." This is that mechanism, opt-in: each step a per-agent,
    per-node gain state ``g`` (N, d) integrates the agent's own ENTRENCHMENT,

        g <- clip(g + eps * entrenchment - decay * g, 0, g_max),
        u_eff = u_base * (1 + g),

    sign-preserving (value accretes onto what the agent already values, where its structure
    is loaded), so ``eps = 0`` (or ``None``) is exactly the static field. Entrenchment is
    read from the agent's own beliefs:

      ``fisher_diag`` : the accumulated evidence deposit ``diag(Pi - Pi_prior)``, max-
                        normalized per agent -- commitments the programme has piled data on;
      ``degree``      : the off-diagonal mass of ``|Pi|`` -- commitments much depends on
                        (the conservatism geometry itself).

    The clip ``g_max`` bounds the accretion x endogenous-gate (E2) feedback loop: dynamic
    conviction feeds both the gate and the ledger's ``Delta U``, and an unbounded gain would
    run away."""

    eps: float = 0.0
    decay: float = 0.01
    entrenchment: str = "fisher_diag"
    g_max: float = 10.0

    def __post_init__(self):
        if self.entrenchment not in ("fisher_diag", "degree"):
            raise ValueError(f"entrenchment must be 'fisher_diag'|'degree', "
                             f"got {self.entrenchment!r}")


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
    ``keys`` (N,2), ``w_obs`` (N,m). Returns ``(J_all, j_all, o_all)``
    (N,d,d),(N,d),(N,m) -- the raw observations are returned so host-side read-outs
    (e.g. an anomaly accumulator) can see what the agents saw; the RNG draws are
    unchanged, so dynamics are byte-identical to the two-output version."""
    def one(key, w):
        o = sample_o(H, phi, sigma_o, key)
        J, j = fisher_deposit_weighted(H, o, sigma_o, w)
        return J, j, o
    return jax.vmap(one)(keys, w_obs)


@jax.jit
def _deposit_all_gated(H, phi, keys, sigma_o, w_obs, Pi, h, nu):
    """``_deposit_all`` with per-channel INFERRED RELIABILITY -- the Student-t
    scale-mixture gate of ``reliability.channel_reliability``: each agent's channel
    weights are multiplied by ``lambda_k = (nu+1)/(nu + z_k^2)``, the posterior
    reliability of channel ``k`` given the agent's OWN surprise at this step's
    observation. On-prediction channels keep full weight; a gross outlier is
    discounted ``~ nu/z^2`` (inferred unreliable, not averaged in).

    The mirror of ``step._fuse_then_observe`` / ``step._observe_pooled``
    (``cfg.reliability_nu``), so the mechanism lives ONCE in ``reliability.py`` and
    both engines wire it. One reading difference, stated honestly: here ``z^2`` is
    read against the agent's own PRE-FUSION belief (this engine computes deposits
    before it fuses) -- identical to step.py's observation-sharing mode; step.py's
    posterior mode reads the post-fusion belief, one fusion later. The RNG draws
    are identical to ``_deposit_all``, so the same seed sees the same world."""
    def one(key, w, Pi_i, h_i):
        o = sample_o(H, phi, sigma_o, key)
        lam_k = rel.channel_reliability(Pi_i, h_i, H, o, sigma_o, nu)
        J, j = fisher_deposit_weighted(H, o, sigma_o, w * lam_k)
        return J, j, o
    return jax.vmap(one)(keys, w_obs, Pi, h)


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
def _dU_NE(Pi, h, Pi0, h0, Pi0r, h0r, U):
    """Per-agent, per-edge conviction-value change of the prune, ``Delta U[i, e]`` (N, E).

    The companion of ``_dF_NE`` on the unified move ledger ``score = Delta F + lam * Delta U``:
    where ``_dF_NE`` reads the evidence half, this reads the value half in the same closed
    form. The Savage-Dickey identity already exhibits the reduced posterior -- shared
    likelihood deposit ``J = Pi - Pi0``, ``j = h - h0``, reduced posterior
    ``(Pi0r_e + J, h0r_e + j)`` -- so the value change of pruning edge ``e`` is one solve:

        Delta U[i, e] = U_i . (mu_reduced(i, e) - mu(i)),

    with ``U_i`` the agent's conviction field (``dual_field.conviction_field``). Pruning a
    conviction-protective edge moves the mean away from the conviction (``Delta U < 0``), so
    the ledger reproduces the old ``Delta F > lam * v_e`` form with the actual value loss in
    place of the proxy ``v_e = |U_p| + |U_c|``. Double vmap mirrors ``_dF_NE``."""
    mu = jnp.linalg.solve(Pi, h[..., None])[..., 0]                  # (N, d)
    def per_edge(Pr, hr):
        def one(P, q, m, u):
            mu_r = jnp.linalg.solve(Pr + (P - Pi0), (hr + (q - h0))[..., None])[..., 0]
            return u @ (mu_r - m)
        return jax.vmap(one)(Pi, h, mu, U)
    return jax.vmap(per_edge)(Pi0r, h0r).T                           # (N, E)


def _joint_reduced(scenario: Scenario, e: int, F: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """The JOINT reduced reference prior per agent for one applied-BMR round.

    ``F`` (N, n_edges) bool: which contested edges each agent's ledger flags for
    removal. Returns ``(R (N,d,d), r (N,d))`` -- for each agent, epoch ``e``'s
    reference prior with ALL its flagged edges removed at once, via
    ``scenario.reduce_fn`` (exact: recompiles the masked CPD net) or the per-edge
    delta-sum fallback (exact only for disjoint reductions -- see ``Scenario``).
    Host numpy; flag rows are deduplicated (``np.unique``) because agents share
    flag patterns heavily, so the cost is (unique patterns) x one small compile."""
    Pi0e, h0e = np.asarray(scenario.Pi0[e]), np.asarray(scenario.h0[e])
    Pi0re, h0re = np.asarray(scenario.Pi0r[e]), np.asarray(scenario.h0r[e])
    uniq, inv = np.unique(F, axis=0, return_inverse=True)
    R_u = np.empty((len(uniq),) + Pi0e.shape)
    r_u = np.empty((len(uniq),) + h0e.shape)
    for k, row in enumerate(uniq):
        if not row.any():
            R_u[k], r_u[k] = Pi0e, h0e
        elif scenario.reduce_fn is not None:
            Rk, rk = scenario.reduce_fn(e, row)
            R_u[k], r_u[k] = np.asarray(Rk), np.asarray(rk)
        else:
            R_u[k] = Pi0e + (Pi0re[row] - Pi0e[None]).sum(axis=0)
            r_u[k] = h0e + (h0re[row] - h0e[None]).sum(axis=0)
    return R_u[inv], r_u[inv]


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


@jax.jit
def _fuse_masked(Pi, h, J_all, j_all, W, mask):
    """Dimension-aware posterior fusion: pool ONLY over agents that represent a dimension.

    ``mask`` (N, d): 1 where the dimension is awake/represented for that agent, 0 where it
    is an unconceived (pinned) slot. The paper's limitations section flags that plain
    ``posterior`` fusion pools a pinned slot's huge self-precision as if it were a firmly
    held "no such thing", crushing a lone discoverer's coupling; this mode is the named
    alternative -- a fusion scheme that EXCLUDES unrepresented dimensions:

      * sender side: agent ``j`` contributes to entry ``(a, b)`` only if it represents both
        dimensions (``M_j[a,b] = mask_j[a] mask_j[b]``), and the weights are renormalized
        per-entry over the representing senders, so an awake slot is never dragged toward a
        neighbour's pin;
      * receiver side: where agent ``i`` itself does not represent the entry, it keeps its
        OWN value -- a pinned slot stays pinned (one cannot receive a concept one does not
        contain; the concept itself still spreads through the proposal channel, not through
        averaging).

    With an all-ones mask every denominator is 1 and this reduces exactly to the
    ``posterior`` mode (``fuse`` + deposit). NOTE: the per-entry renormalization makes the
    fused matrix a different convex mix per entry, which does not automatically preserve
    positive-definiteness; ``tests/test_fuse_masked.py`` checks PD along a staggered wake
    trajectory (the asleep-receiver case is exactly block-diagonal and safe)."""
    EPSm = 1e-12
    M = mask[:, :, None] * mask[:, None, :]                       # (N,d,d) sender entries
    num = jnp.einsum("ij,jab->iab", W, M * Pi)
    den = jnp.einsum("ij,jab->iab", W, M)
    fused = jnp.where(den > EPSm, num / (den + EPSm), Pi)
    Pi_f = jnp.where(M > 0, fused, Pi)                            # receiver guard
    num_h = W @ (mask * h)
    den_h = W @ mask
    fused_h = jnp.where(den_h > EPSm, num_h / (den_h + EPSm), h)
    h_f = jnp.where(mask > 0, fused_h, h)
    return Pi_f + J_all, h_f + j_all


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
                   w_floor: float = 0.0, gate_mode: str = "conviction",
                   deposit_gate: jax.Array | None = None,
                   social_nu: float | None = None,
                   social_idx: tuple[int, ...] | None = None,
                   social_gate: str = "means",
                   social_pairs: tuple | None = None,
                   social_scale_floor: float = 0.1,
                   trust_memory: float | None = None,
                   reliability_nu: float | None = None,
                   bmr_every: int | None = None,
                   host_hook: Callable | None = None,
                   conviction_dynamics: ConvictionDynamics | None = None,
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

    ``reliability_nu`` (optional) turns on per-channel INFERRED RELIABILITY: each
    agent's deposit weights are multiplied by the Student-t reliability
    ``(nu+1)/(nu+z_k^2)`` of its own surprise at this step's observation
    (``reliability.channel_reliability`` -- the same gate the step.py engine wires
    via ``cfg.reliability_nu``; see ``_deposit_all_gated`` for the one reading
    difference). ``None`` (default) is byte-identical.

    ``gate_mode`` selects WHAT the endogenous sensory gate (E2) reads
    (``reliability.gate_weights``): ``"conviction"`` (default, byte-identical) silences a
    disconfirming channel by how directly it bears on what the agent VALUES
    (``|U . H_disc|``); ``"cost"`` silences it by the re-equilibration mass of the
    commitments it addresses (``kappa . |H_disc|``, the live revision-cost field
    ``dual_field.cost_field_live``) -- wishful vs dogmatic lock-in, separated.

    ``trust_memory`` (optional) makes the content-gated trust a TRACK RECORD rather than
    an instantaneous read: the pairwise disagreement statistic is accumulated as an EMA
    ``Z2_t = omega_T Z2_{t-1} + (1 - omega_T) z2_t`` upstream of the Student-t weight
    (``reliability.trust_memory_update``), so a neighbour who has KEPT disagreeing stays
    discounted after one agreeable step and trust is re-earned at the memory's timescale.
    Requires ``social_nu``; works under both ``social_gate`` modes. ``None`` (default) is
    byte-identical.

    ``bmr_every`` (optional) makes the reduction APPLIED rather than read-out -- the
    paper's crisis check actually executed: every ``bmr_every`` steps each agent scores
    the contested edges on the SAME ledger the snapshot reads (``dF + lam_i dU > 0``)
    and its belief is replaced by the exact Savage-Dickey reduced posterior for the
    flagged set -- the joint reduced reference prior (``Scenario.reduce_fn``) plus the
    agent's untouched accumulated deposit ``Pi - Pi0``. The forgetting anchor receives
    the same (precision-scaled) edit, so with ``forgetting < 1`` a pruned coupling
    relaxes toward ZERO, not back toward the old prior. Semantics, stated honestly:
    marginal per-edge scores, joint application (exactly the paper's crisis-check
    reading); the application is a toggle-delta -- re-evaluated each round from the
    current net, so a cleared flag RESTORES the prior edge (structure is reversible;
    regrowth on supporting evidence is allowed). Because the deposit is inferred as
    ``Pi - Pi0`` against the UNREDUCED reference, a live edit biases its own future
    score (mild hysteresis). ``None`` (default) runs zero new code -- byte-identical.

    ``host_hook`` (optional): called once per step AFTER fuse/observe/tilt and BEFORE the
    snapshot, as ``host_hook(t, Pi, h, Pi_prior, h_prior, w_obs, o)`` with numpy arrays
    (``o`` (N,m) = this step's raw observations). It may return a dict replacing any of
    ``{"Pi", "h", "Pi_prior", "h_prior"}`` -- e.g. an endogenous-crisis mechanism that
    releases the forgetting anchor's core precision when an anomaly accumulator crosses a
    threshold. ``None`` (default) keeps the loop host-sync-free and byte-identical.

    Returns a dict of host arrays (snapshots every ``snapshot_every`` steps + the final step);
    see the keys at the bottom. ``snap_Pi`` / ``snap_h`` carry the full per-agent nets so the
    scenario-specific read-outs (theory tracking, structural match) are computed host-side."""
    N, d, m = graph.n, scenario.dim, scenario.m
    n_steps, n_edges, n_epochs = scenario.n_steps, scenario.n_edges, scenario.n_epochs

    W = graph.trust_W()                                  # (N,N) row-stochastic
    Woff = np.asarray(W) * (1.0 - np.eye(N))
    Woff = jnp.asarray(Woff / (Woff.sum(axis=1, keepdims=True) + EPS))

    # CONTENT-GATED TRUST (opt-in; mirrors step._gated_W): with ``social_nu`` set, the
    # fusion weights are recomputed each round from the Student-t trust precisions
    # ``gamma_ij = (nu_s+1)/(nu_s+z_ij^2)`` over the STATIC graph support -- the topology
    # is respected, but agents who have diverged stop averaging with each other (camps).
    # WHAT the gate reads is ``social_gate``:
    #   "means"     (default) : disagreement of the posterior MEANS on the ``social_idx``
    #                           nodes (``reliability.social_gamma``) -- what opinions you hold;
    #   "structure"           : disagreement of the contested COUPLINGS ``Pi[a, c]`` over
    #                           ``social_pairs`` (``reliability.pairwise_structure_z2``) --
    #                           trust by HOW YOU WIRE THE WORLD, not what opinions or values
    #                           you hold. The gate that can protect a pluralism living in the
    #                           wiring when both camps agree in means.
    # ``social_nu=None`` (DEFAULT) keeps the static ``W`` above, byte-identical.
    A_self = social_ix = social_pairs_arr = None
    if social_nu is not None:
        if social_gate not in ("means", "structure"):
            raise ValueError(f"social_gate must be 'means'|'structure', got {social_gate!r}")
        if social_gate == "structure":
            if social_pairs is None:
                raise ValueError("social_gate='structure' requires social_pairs (the "
                                 "contested (i, j) couplings whose divergence gates trust)")
            social_pairs_arr = jnp.asarray(np.asarray(social_pairs, dtype=int))
        else:
            if not social_idx:
                raise ValueError("social_nu requires social_idx (node indices whose "
                                 "disagreement gates trust)")
            social_ix = jnp.asarray(np.asarray(social_idx, dtype=int))
        A_self = jnp.asarray((np.asarray(graph.A) > 0).astype(float) + np.eye(N))

    if gate_mode not in ("conviction", "cost"):
        raise ValueError(f"gate_mode must be 'conviction'|'cost', got {gate_mode!r}")
    if trust_memory is not None:
        if social_nu is None:
            raise ValueError("trust_memory requires social_nu (it accumulates the "
                             "trust gate's z2 statistic)")
        if not 0.0 < float(trust_memory) < 1.0:
            raise ValueError(f"trust_memory must be in (0, 1), got {trust_memory}")
    Z2_mem = None                          # trust track record (EMA of pairwise z2)

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
    # DYNAMIC conviction (Lakatos accretion, opt-in): the per-agent gain state g (N, d)
    # integrates each agent's own entrenchment; u_agent becomes u_base * (1 + g). None =>
    # u_agent stays the static field above, byte-identical.
    u_base = jnp.broadcast_to(u_agent, (N, d)) if conviction_dynamics is not None else None
    g_gain = jnp.zeros((N, d)) if conviction_dynamics is not None else None
    u_gain_t: list = []
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
    Pi0, h0, Pi0r, h0r = scenario.Pi0, scenario.h0, scenario.Pi0r, scenario.h0r
    v_e = np.asarray(scenario.v_e)        # kept as an OUTPUT (substrate property); the prune
    # read-out itself no longer thresholds on it -- see the ledger in ``snapshot``.
    epoch_t = scenario.epoch_t
    belt_ix = np.asarray(scenario.belt_ix, dtype=int)

    key = jax.random.PRNGKey(seed)
    fuse_mask = None                       # (N, d) representation mask (posterior_masked)
    snap_t, kept_t, revolted_t, m_t, snap_Pi, snap_h, gamma_t = [], [], [], [], [], [], []
    last_dF = last_pruned = None

    # APPLIED BMR state (opt-in; see the docstring): the live toggle-delta each agent's
    # net (and anchor) currently carries, and the flags that produced it.
    bmr_dPi = bmr_dh = bmr_flags = ps_arr = None
    applied_t: list = []
    if bmr_every is not None:
        bmr_dPi = np.zeros((N, d, d))
        bmr_dh = np.zeros((N, d))
        bmr_flags = np.zeros((N, n_edges), dtype=bool)
        ps_arr = (np.ones(N) if spec.precision_scale is None
                  else np.asarray(spec.precision_scale, dtype=float))

    def snapshot(t):
        e = int(epoch_t[t])
        dF = np.asarray(_dF_NE(Pi, h, Pi0[e], h0[e], Pi0r[e], h0r[e]))   # (N,E)
        # the prune read-out is the unified move ledger, score = dF + lam * dU, with dU the
        # CLOSED-FORM conviction-value change of each prune under the agent's own conviction
        # field (the former proxy threshold lam * v_e replaced by the quantity it stood for).
        U = conviction_field(Pi, h, scenario.names, u_agent, scenario.conviction_alpha)
        dU = np.asarray(_dU_NE(Pi, h, Pi0[e], h0[e], Pi0r[e], h0r[e], U))  # (N,E)
        pruned = (dF + lam[:, None] * dU) > 0.0
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
        if conviction_dynamics is not None:   # Lakatos accretion: value follows entrenchment
            cd = conviction_dynamics
            if cd.entrenchment == "fisher_diag":
                ent = jnp.diagonal(Pi - Pi_prior, axis1=1, axis2=2)        # (N, d) deposit
            else:                              # "degree": off-diagonal structural load
                ent = jnp.abs(Pi).sum(axis=2) - jnp.abs(
                    jnp.diagonal(Pi, axis1=1, axis2=2))
            ent = jnp.maximum(ent, 0.0)
            ent = ent / (ent.max(axis=1, keepdims=True) + EPS)             # per-agent norm
            g_gain = jnp.clip(g_gain + cd.eps * ent - cd.decay * g_gain, 0.0, cd.g_max)
            u_agent = u_base * (1.0 + g_gain)
        if endogenous_gamma and disc_arr.size:   # the sensory gate (E2): what closes a channel
            # is gate_mode -- conviction (wishful: |U.H|) or cost (dogmatic: kappa.|H|).
            Ug = (None if gate_mode == "cost" else
                  conviction_field(Pi, h, scenario.names, u_agent, scenario.conviction_alpha))
            kap = (None if gate_mode == "conviction" else
                   cost_field_live(Pi, scenario.conviction_alpha))
            w_disc = rel.gate_weights(Ug, kap, H[disc_idx], gate_mode, gate_strength, w_floor)
            w_obs = w_obs_base.at[:, disc_idx].set(w_disc)
        key, sk = jax.random.split(key)
        keys = jax.random.split(sk, N)
        if reliability_nu is not None:   # inferred per-channel reliability (Student-t)
            J_all, j_all, o_all = _deposit_all_gated(H, phi, keys, sigma_o, w_obs,
                                                     Pi, h, reliability_nu)
        else:
            J_all, j_all, o_all = _deposit_all(H, phi, keys, sigma_o, w_obs)
        if gate_sqrt is not None:                # conservatism gates the revision rate (E1)
            J_all = gate_sqrt[None, :, None] * J_all * gate_sqrt[None, None, :]
            j_all = gate_sqrt[None, :] * j_all
        W_t, Woff_t = W, Woff
        if social_nu is not None:            # content-gated trust: this round's W from gamma
            if social_gate == "structure":
                z2 = rel.pairwise_structure_z2(Pi, social_pairs_arr, social_scale_floor)
            else:
                z2 = rel.social_z2(Pi, h, social_ix)
            if trust_memory is not None:     # trust as a track record (EMA on z2)
                Z2_mem = rel.trust_memory_update(Z2_mem, z2, trust_memory)
                z2 = Z2_mem
            gamma = rel.student_t_weight(z2, social_nu)
            W_t = trust_weights(A_self, gamma)
            Wo = W_t * (1.0 - jnp.eye(N))
            Woff_t = Wo / (Wo.sum(axis=1, keepdims=True) + EPS)
        if fuse_mode == "posterior_masked":
            mk = jnp.ones((N, d)) if fuse_mask is None else jnp.asarray(fuse_mask)
            Pi, h = _fuse_masked(Pi, h, J_all, j_all, W_t, mk)
        else:
            Pi, h = _fuse_observe(Pi, h, J_all, j_all, W_t, Woff_t, eta, fuse_mode)
        if tilt is not None:                            # the conviction lever (off by default)
            U = conviction_field(Pi, h, scenario.names, u_agent, scenario.conviction_alpha)
            h = h + tilt[:, None] * U
        if host_hook is not None:                # host-side per-step logic (e.g. crisis)
            upd = host_hook(t, np.asarray(Pi), np.asarray(h), np.asarray(Pi_prior),
                            np.asarray(h_prior), np.asarray(w_obs), np.asarray(o_all))
            if upd:
                # "fuse_mask" (N, d): which dimensions each agent represents -- consumed
                # by next step's fuse_mode="posterior_masked" (the hook owns wake state).
                if "fuse_mask" in upd:
                    fuse_mask = np.asarray(upd.pop("fuse_mask"))
            if upd:
                Pi = jnp.asarray(upd.get("Pi", Pi))
                h = jnp.asarray(upd.get("h", h))
                Pi_prior = jnp.asarray(upd.get("Pi_prior", Pi_prior))
                h_prior = jnp.asarray(upd.get("h_prior", h_prior))
        if bmr_every is not None and t > 0 and t % bmr_every == 0:
            # APPLY the reduction (the crisis check executed): score on the same ledger
            # the snapshot reads, then swap in the exact Savage-Dickey reduced posterior
            # (joint reduced prior + untouched deposit) as a toggle-delta, anchor included.
            e = int(epoch_t[t])
            dF_a = np.asarray(_dF_NE(Pi, h, Pi0[e], h0[e], Pi0r[e], h0r[e]))     # (N,E)
            U_a = conviction_field(Pi, h, scenario.names, u_agent,
                                   scenario.conviction_alpha)
            dU_a = np.asarray(_dU_NE(Pi, h, Pi0[e], h0[e], Pi0r[e], h0r[e], U_a))
            F = (dF_a + lam[:, None] * dU_a) > 0.0                               # (N,E)
            R, r = _joint_reduced(scenario, e, F)                       # (N,d,d),(N,d)
            dPi_new = R - np.asarray(Pi0[e])
            dh_new = r - np.asarray(h0[e])
            Pi = jnp.asarray(np.asarray(Pi) + (dPi_new - bmr_dPi))
            h = jnp.asarray(np.asarray(h) + (dh_new - bmr_dh))
            # the anchor gets the SAME (precision-scaled) edit: omega < 1 then relaxes a
            # pruned coupling toward 0, not back toward the old prior value.
            Pi_prior = jnp.asarray(np.asarray(Pi_prior)
                                   + ps_arr[:, None, None] * (dPi_new - bmr_dPi))
            h_prior = jnp.asarray(np.asarray(h_prior)
                                  + ps_arr[:, None] * (dh_new - bmr_dh))
            bmr_dPi, bmr_dh, bmr_flags = dPi_new, dh_new, F
        if (t % snapshot_every == 0) or (t == n_steps - 1):
            last_dF, last_pruned = snapshot(t)
            if conviction_dynamics is not None:
                u_gain_t.append(np.asarray(g_gain))
            if bmr_every is not None:
                applied_t.append(bmr_flags.copy())

    snap_Pi_arr = np.stack(snap_Pi)                                       # (S,N,d,d)
    disagreement_t = shells.residual_disagreement(snap_Pi_arr)           # (S,)

    extra = {}
    if conviction_dynamics is not None:
        extra["u_gain_t"] = np.stack(u_gain_t)            # (S, N, d) accreted gain
    if bmr_every is not None:
        extra["applied_pruned_t"] = np.stack(applied_t)   # (S, N, E) bool, applied state
        extra["bmr_every"] = bmr_every
    return {
        **extra,
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
