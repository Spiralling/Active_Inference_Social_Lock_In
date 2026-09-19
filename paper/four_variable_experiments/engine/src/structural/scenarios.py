"""Environment presets for the reusable :mod:`src.structural.simulation` engine.

Two environments, one engine:

  * :func:`phlogiston_scenario` -- the single-epoch over-wired phlogiston paradigm of nb43
    (built from :func:`build_substrate`, which is lifted verbatim from
    ``scripts/run_multiagent_topology.py`` so that script can re-export it and nb43's import
    is unchanged). Fed through ``run_simulation`` it reproduces nb43 byte-for-byte.

  * :func:`cosmology_scenario` -- a NEW changing-cosmology world: three genuinely different
    cosmology theories (dark-matter -> modified-gravity -> scale-variant laws, the existing
    ``landscape_presets`` Bayes-net presets) that the world cycles through in three epochs.
    Agents must re-grow their net structure to track each successive theory. Because each
    preset couples the anomalies to a *different* commitment, "which edges should be kept"
    changes each epoch -- so a population that over-pruned in epoch 0 has lost the structure
    epoch 1 needs.

The cosmology read-outs the notebook actually plots (which theory each community currently
holds; structural match to the *current* theory) are computed host-side from the engine's
``snap_h`` / ``snap_Pi`` using the helpers here (:func:`cosmology_theory_means`,
:func:`cosmology_true_couplings`, :func:`closest_theory`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from src.structural import observables, linalg
from src.structural.bayesnet import LinearGaussianBN
from src.structural.dual_field import PrecisionUtilityNet
from src.structural.belief import GaussianBeliefNet
from src.structural.phlogiston import (
    StructuralConfig, phlogiston_bn, gravimetric_H, gravimetric_rows,
    phi_true_at, conviction_u, balanced_lambda, DISAGREEMENT_NODES,
)
from src.structural.landscape_presets import (
    LandscapeBasis, cosmology_basis,
    bayesnet_preset_dark_matter, bayesnet_preset_modified_gravity,
    bayesnet_preset_scale_variant_laws,
)
from src.structural.simulation import Scenario


# ======================================================================
# PHLOGISTON (nb43) -- the substrate + scenario.
# ======================================================================

# The 4 spurious edges that over-wire the incumbent paradigm (identical to nb38/nb42):
# extra parent->child couplings the data will (or will not) license pruning. (parent, child).
DEFAULT_SPURIOUS = (
    ("combustion_releases", "mass_change_sign", 0.6),
    ("calcination_releases", "reduction_with_charcoal", 0.6),
    ("air_has_capacity", "respiration_like_combustion", 0.6),
    ("combustion_releases", "calx_heavier_than_metal", 0.5),
)

# The contested belt: the two NATIVE phlogiston mass-law edges (highest conviction
# protection v_e). "Revolted" = an agent has pruned BOTH -> it has eliminated the
# phlogiston mass-law reading. (parent, child) in phlogiston_bn's CPD direction.
BELT = (
    ("mass_change_sign", "calx_heavier_than_metal"),
    ("gas_consumed", "calx_heavier_than_metal"),
)


@dataclass(frozen=True)
class Substrate:
    """The over-wired phlogiston paradigm and everything derived once from it (nb43)."""
    over: LinearGaussianBN                 # the over-wired incumbent BN (CPD form)
    Pi0: jnp.ndarray                       # (d,d) compiled prior precision
    h0: jnp.ndarray                        # (d,)  compiled prior potential
    edges: tuple[tuple[str, str], ...]     # all (parent, child) edges of `over`
    v_e: np.ndarray                        # (E,) conviction protection per edge
    belt_ix: tuple[int, ...]               # indices of the BELT edges within `edges`
    Pi0r: jnp.ndarray                      # (E,d,d) per-edge reduced prior precision
    h0r: jnp.ndarray                       # (E,d)   per-edge reduced prior potential
    lstar: float                           # balanced lambda* (reference conviction scale)
    Hg: jnp.ndarray                        # (m,d) gravimetric (relational) operator
    disc_rows: tuple[int, ...]             # rows of Hg that carry the disconfirming channel


def build_substrate(cfg: StructuralConfig, *, conviction: float = 2.0,
                    over_wiring=DEFAULT_SPURIOUS) -> Substrate:
    """The over-wired incumbent paradigm and everything derived once from it.

    ``conviction`` is the CPD stiffness of ``phlogiston_bn`` (>=1 keeps the compiled prior
    positive-definite -- no indefinite-hub trap), NOT the per-agent prune threshold
    ``lambda_i``. The conviction protection ``v_e = |U_parent| + |U_child|`` is the
    propagated intrinsic-utility magnitude (``U = T u``; magnitude is mean-independent, so
    it does not flip sign after the regime shift). The per-edge reduced priors (one edge
    pinned out) are precomputed and stacked so the prune readout vmaps over agents AND edges.
    """
    bn = phlogiston_bn(cfg, conviction=conviction)
    names = bn.names
    idx = {n: i for i, n in enumerate(names)}
    d = bn.dim

    B = np.asarray(bn.B).copy()
    for p, c, w in over_wiring:
        B[idx[c], idx[p]] = w
    over = LinearGaussianBN(B=jnp.asarray(B), b=bn.b, s=bn.s, names=names)
    gbn = over.to_info()
    Pi0, h0 = gbn.Pi, gbn.h

    # every directed edge present in B, as (parent, child) name tuples (nb42 convention)
    edges = tuple((names[a], names[b])
                  for b in range(d) for a in range(d) if abs(B[b, a]) > 1e-9)

    # conviction protection v_e from the propagated utility magnitude
    U = np.abs(np.asarray(PrecisionUtilityNet(
        names=names, Pi=Pi0, h=h0,
        u=conviction_u(cfg, "phlogiston"), alpha=0.5).effective_utility()))
    v_e = np.array([U[idx[p]] + U[idx[c]] for (p, c) in edges])

    belt_ix = tuple(edges.index(e) for e in BELT)

    # per-edge reduced prior (prune parent->child): over.prune_edge(child, parent)
    Pi0r_list, h0r_list = [], []
    for (p, c) in edges:
        red = over.prune_edge(c, p).to_info()
        Pi0r_list.append(red.Pi)
        h0r_list.append(red.h)
    Pi0r = jnp.stack(Pi0r_list)
    h0r = jnp.stack(h0r_list)

    Hg = gravimetric_H(cfg)
    rows = gravimetric_rows(cfg)
    disc_rows = tuple(i for i, r in enumerate(rows)
                      if ("mass_balance" in r) or (r in DISAGREEMENT_NODES))

    return Substrate(over=over, Pi0=Pi0, h0=h0, edges=edges, v_e=v_e,
                     belt_ix=belt_ix, Pi0r=Pi0r, h0r=h0r,
                     lstar=balanced_lambda(cfg, "phlogiston"),
                     Hg=Hg, disc_rows=disc_rows)


def _phlog_order_parameter(Pi, h, *, names, mass_nodes, mu_phlog, mu_oxy):
    return observables.order_parameter(Pi, h, names, mass_nodes, mu_phlog, mu_oxy)


def phlogiston_scenario(cfg: StructuralConfig, *, sub: Substrate | None = None,
                        conviction: float = 2.0,
                        over_wiring=DEFAULT_SPURIOUS) -> Scenario:
    """Wrap the nb43 phlogiston substrate as a single-epoch :class:`Scenario`.

    Fed through ``run_simulation`` with the vanguard ``AgentSpec`` (built in
    ``scripts/run_multiagent_topology.run_world``) this reproduces nb43 byte-for-byte: a
    single epoch (``epoch_t`` all 0), the over-wired prior as both the agents' initial belief
    and the prune reference, the gravimetric relational operator, and the oxygen-index order
    parameter."""
    if sub is None:
        sub = build_substrate(cfg, conviction=conviction, over_wiring=over_wiring)
    names = sub.over.names
    n_steps = cfg.n_steps
    phis = jnp.stack([phi_true_at(cfg, t) for t in range(n_steps)])       # (T, d)
    order_fn = partial(_phlog_order_parameter, names=names,
                       mass_nodes=DISAGREEMENT_NODES,
                       mu_phlog=cfg.mu_phlog_mass, mu_oxy=cfg.mu_oxy_mass)

    # APPLIED-BMR joint reducer (``run_simulation(bmr_every=...)``): the reference prior
    # with ALL flagged edges removed AT ONCE -- mask the CPD weights, recompile (PD by
    # construction). One-hot flags reproduce ``sub.Pi0r[k]`` / ``h0r[k]`` exactly (it is
    # the same ``prune_edge`` -> ``to_info`` operation); the joint compile is REQUIRED
    # here because the belt edges share the child ``calx_heavier_than_metal``, so summing
    # per-edge prior deltas would double-subtract the co-parent fill-in term.
    idx = {n: i for i, n in enumerate(names)}
    edge_ci = tuple((idx[c], idx[p]) for (p, c) in sub.edges)   # B[child, parent] slots
    B_full = np.asarray(sub.over.B)

    def _reduce(epoch: int, flags) -> tuple[jnp.ndarray, jnp.ndarray]:
        B = B_full.copy()
        for k, on in enumerate(np.asarray(flags, dtype=bool)):
            if on:
                c, p = edge_ci[k]
                B[c, p] = 0.0
        red = LinearGaussianBN(B=jnp.asarray(B), b=sub.over.b, s=sub.over.s,
                               names=names).to_info()
        return red.Pi, red.h

    return Scenario(
        name="phlogiston",
        names=names,
        H=sub.Hg,
        disc_rows=sub.disc_rows,
        phis=phis,
        epoch_t=tuple([0] * n_steps),
        n_epochs=1,
        sigma_o=float(cfg.sigma_o),
        Pi_base=sub.Pi0,
        h_base=sub.h0,
        Pi0=sub.Pi0[None],                                               # (1,d,d)
        h0=sub.h0[None],                                                 # (1,d)
        Pi0r=sub.Pi0r[None],                                            # (1,E,d,d)
        h0r=sub.h0r[None],                                              # (1,E,d)
        v_e=jnp.asarray(sub.v_e)[None],                                 # (1,E)
        edges=sub.edges,
        belt_ix=sub.belt_ix,
        u=conviction_u(cfg, "phlogiston"),
        conviction_alpha=0.5,
        order_fn=order_fn,
        lstar=sub.lstar,
        reduce_fn=_reduce,
    )


# ======================================================================
# COSMOLOGY -- a world that changes theory three times.
# ======================================================================

# The three epochs, in order, mapped to the existing landscape presets (which genuinely
# differ in coupling AND mean). dark-matter -> modified-gravity -> scale-variant/dark-energy.
COSMOLOGY_EPOCHS: tuple[str, ...] = ("dark_matter", "modified_gravity", "scale_variant_laws")

_PRESET_BUILDERS = {
    "dark_matter": bayesnet_preset_dark_matter,
    "modified_gravity": bayesnet_preset_modified_gravity,
    "scale_variant_laws": bayesnet_preset_scale_variant_laws,
}

# The two anomaly nodes and the three rival commitments (the cosmology "belt").
_ANOMALIES = ("rotation_curve_anomaly", "large_scale_structure_anomaly")
_COMMITMENTS = ("dark_matter_commitment", "modified_gravity_commitment",
                "scale_variant_laws_commitment")
# The contested couplings we track: each anomaly -> each rival commitment. Which of these the
# data should hold changes every epoch (each preset couples the anomalies to a *different*
# commitment), so an over-pruned population has lost the structure the next epoch needs.
COSMOLOGY_EDGES: tuple[tuple[str, str], ...] = tuple(
    (a, c) for c in _COMMITMENTS for a in _ANOMALIES)


def cosmology_presets(basis: LandscapeBasis | None = None) -> list:
    basis = cosmology_basis() if basis is None else basis
    return [_PRESET_BUILDERS[name](basis) for name in COSMOLOGY_EPOCHS]


def cosmology_theory_means(basis: LandscapeBasis | None = None) -> np.ndarray:
    """(E, d) the marginal mean of each epoch's theory -- the candidate theories a community's
    belief is projected onto for the "which theory does it hold" read-out."""
    nets = [p.belief_net for p in cosmology_presets(basis)]
    return np.stack([np.asarray(linalg.info_mean(n.Pi, n.h)) for n in nets])


def cosmology_true_couplings(basis: LandscapeBasis | None = None
                             ) -> np.ndarray:
    """(E, n_edges) the true off-diagonal precision ``Pi[a, c]`` of each contested edge under
    each epoch's theory -- the comparand for the structural-match read-out (how close a
    community's *learned* couplings are to the current true theory)."""
    basis = cosmology_basis() if basis is None else basis
    nets = [p.belief_net for p in cosmology_presets(basis)]
    idx = {n: i for i, n in enumerate(basis.names)}
    out = np.zeros((len(nets), len(COSMOLOGY_EDGES)))
    for e, net in enumerate(nets):
        Pi = np.asarray(net.Pi)
        for k, (a, c) in enumerate(COSMOLOGY_EDGES):
            out[e, k] = Pi[idx[a], idx[c]]
    return out


def closest_theory(means: np.ndarray, theory_mus: np.ndarray) -> np.ndarray:
    """Nearest candidate-theory index for each belief mean. ``means`` (..., d), ``theory_mus``
    (E, d). Returns int array of shape ``means.shape[:-1]``: which of the E theories each belief
    sits closest to (Euclidean on the commitment basis). The headline "tracking" read-out: it
    should follow the true epoch through the transitions, or freeze on an earlier theory."""
    means = np.asarray(means)
    d = means.shape[-1]
    flat = means.reshape(-1, d)
    dists = np.linalg.norm(flat[:, None, :] - theory_mus[None, :, :], axis=2)  # (M, E)
    return np.argmin(dists, axis=1).reshape(means.shape[:-1])


def theory_alignment(means: np.ndarray, theory_mus: np.ndarray) -> np.ndarray:
    """Soft per-theory alignment (..., E): a softmax-style weight over candidate theories from
    the (negative) squared distance, so a community's *graded* drift between theories is visible
    (not just the arg-min). Rows sum to 1."""
    means = np.asarray(means)
    d = means.shape[-1]
    flat = means.reshape(-1, d)
    d2 = ((flat[:, None, :] - theory_mus[None, :, :]) ** 2).sum(axis=2)        # (M, E)
    scale = np.median(d2) + 1e-9
    w = np.exp(-d2 / scale)
    w = w / w.sum(axis=1, keepdims=True)
    return w.reshape(means.shape[:-1] + (theory_mus.shape[0],))


def cosmology_H(basis: LandscapeBasis | None = None) -> jnp.ndarray:
    """The relational observation operator: a direct (level) row on every node PLUS one
    relational *anomaly-balance* row per rival commitment, ``commitment - rotation_curve -
    large_scale_structure`` -- reading whether that commitment accounts for both anomalies.
    The relational rows deposit OFF-DIAGONAL Fisher, so the contested anomaly<->commitment
    couplings LEARN over the rollout (the precondition for structural re-tracking). Same
    constructor as ``phlogiston.gravimetric_H``."""
    from src.structural.bayesnet import relational_operator
    basis = cosmology_basis() if basis is None else basis
    rels = [{n: 1.0} for n in basis.names]                     # direct level reads
    for c in _COMMITMENTS:                                     # one anomaly-balance per rival
        rels.append({c: 1.0, _ANOMALIES[0]: -1.0, _ANOMALIES[1]: -1.0})
    return relational_operator(basis.names, rels)


def cosmology_rows(basis: LandscapeBasis | None = None) -> tuple[str, ...]:
    basis = cosmology_basis() if basis is None else basis
    return tuple(basis.names) + tuple(f"balance({c})" for c in _COMMITMENTS)


def cosmology_utility_toward(theory: str, basis: LandscapeBasis | None = None,
                             coherence: float = 0.5) -> jnp.ndarray:
    """The intrinsic utility ``u`` (d,) of a community committed to ``theory``: +1 on that
    theory's commitment, -1 on the rivals, ``coherence`` on ``law_coherence``. Propagated to
    ``U = T u`` and added to the potential each step, this is the "block revision" lever -- a
    committed community keeps pulling its belief back toward its favoured theory."""
    basis = cosmology_basis() if basis is None else basis
    idx = {n: i for i, n in enumerate(basis.names)}
    home = f"{theory}_commitment"
    u = np.zeros((len(basis.names),), dtype=np.float32)
    u[idx["law_coherence"]] = coherence
    for c in _COMMITMENTS:
        u[idx[c]] = 1.0 if c == home else -1.0
    return jnp.asarray(u)


def _cosmology_order_parameter(Pi, h, *, incumbent_idx):
    """Population-mean posterior value of the incumbent (dark-matter) commitment -- a coarse
    scalar diagnostic (it falls as the population leaves the incumbent). The *headline*
    tracking is computed host-side via ``closest_theory``; this is just the engine's ``m_t``."""
    mu = jax.vmap(jnp.linalg.solve)(Pi, h)            # (N, d)
    return mu[:, incumbent_idx].mean()


def cosmology_scenario(*, n_steps: int = 180, t1: int = 60, t2: int = 120,
                       sigma_o: float = 0.5, conviction_alpha: float = 0.5,
                       incumbent_theory: str = "dark_matter",
                       basis: LandscapeBasis | None = None) -> Scenario:
    """The 3-epoch changing-cosmology :class:`Scenario`.

    World truth ``phi_at(t)`` is the *current epoch's preset mean*: dark-matter on ``[0, t1)``,
    modified-gravity on ``[t1, t2)``, scale-variant laws on ``[t2, n_steps)``. The observation
    operator is :func:`cosmology_H` (relational anomaly-balance rows so the contested couplings
    learn). Agents start from the incumbent (dark-matter) theory prior; the prune read-out is
    re-armed against the *current epoch's* theory prior, so structure that fit epoch 0 but not
    epoch 1 reads as stale. ``reduce_fn`` is left ``None``: the contested reductions are
    disjoint ``zero_edge_prior`` edits, so the engine's per-edge delta-sum fallback for
    applied BMR (``bmr_every``) is already exact here. The conviction utility favours the incumbent (the bloc that gets
    "stuck"); per-community direction/strength is supplied via the ``AgentSpec``."""
    basis = cosmology_basis() if basis is None else basis
    names = tuple(basis.names)
    d = len(names)
    idx = {n: i for i, n in enumerate(names)}
    presets = cosmology_presets(basis)
    nets = [p.belief_net for p in presets]                     # GaussianBeliefNet per epoch

    # epoch schedule + world truth (each epoch's preset mean)
    def epoch_of(t: int) -> int:
        return 0 if t < t1 else (1 if t < t2 else 2)
    epoch_t = tuple(epoch_of(t) for t in range(n_steps))
    theory_mu = cosmology_theory_means(basis)                  # (E, d)
    phis = jnp.asarray(np.stack([theory_mu[epoch_of(t)] for t in range(n_steps)]))  # (T,d)

    # per-epoch reference priors + per-edge reduced priors (one contested coupling pinned out)
    Pi0 = jnp.stack([n.Pi for n in nets])                      # (E, d, d)
    h0 = jnp.stack([n.h for n in nets])                        # (E, d)
    edge_ij = [(idx[a], idx[c]) for (a, c) in COSMOLOGY_EDGES]
    Pi0r_e, h0r_e, v_e_e = [], [], []
    u_incumbent = cosmology_utility_toward(incumbent_theory, basis)
    for e, net in enumerate(nets):
        Pi_e, h_e = net.Pi, net.h
        Pi0r_e.append(jnp.stack([linalg.zero_edge_prior(Pi_e, i, j) for (i, j) in edge_ij]))
        h0r_e.append(jnp.broadcast_to(h_e, (len(edge_ij),) + h_e.shape))
        U = np.abs(np.asarray(PrecisionUtilityNet(
            names=names, Pi=Pi_e, h=h_e, u=u_incumbent,
            alpha=conviction_alpha).effective_utility()))
        v_e_e.append(np.array([U[i] + U[j] for (i, j) in edge_ij]))
    Pi0r = jnp.stack(Pi0r_e)                                   # (E, n_edges, d, d)
    h0r = jnp.stack(h0r_e)                                     # (E, n_edges, d)
    v_e = jnp.asarray(np.stack(v_e_e))                         # (E, n_edges)

    # disconfirming rows: the relational anomaly-balance rows (the contested channel)
    rows = cosmology_rows(basis)
    disc_rows = tuple(i for i, r in enumerate(rows) if r.startswith("balance("))

    # belt = the incumbent (dark-matter) anomaly couplings -> "revolted" = abandoned them
    belt_ix = tuple(k for k, (a, c) in enumerate(COSMOLOGY_EDGES)
                    if c == f"{incumbent_theory}_commitment")

    order_fn = partial(_cosmology_order_parameter,
                       incumbent_idx=idx[f"{incumbent_theory}_commitment"])

    return Scenario(
        name="cosmology",
        names=names,
        H=cosmology_H(basis),
        disc_rows=disc_rows,
        phis=phis,
        epoch_t=epoch_t,
        n_epochs=3,
        sigma_o=float(sigma_o),
        Pi_base=nets[0].Pi,                                    # start at the incumbent theory
        h_base=nets[0].h,
        Pi0=Pi0,
        h0=h0,
        Pi0r=Pi0r,
        h0r=h0r,
        v_e=v_e,
        edges=COSMOLOGY_EDGES,
        belt_ix=belt_ix,
        u=u_incumbent,
        conviction_alpha=conviction_alpha,
        order_fn=order_fn,
        lstar=0.0,
    )
