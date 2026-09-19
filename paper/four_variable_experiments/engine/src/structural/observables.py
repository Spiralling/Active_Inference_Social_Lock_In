"""Post-hoc order parameters for the structural experiment.

The Gaussian-structural sibling of ``src/pomdp/observables.py`` and
``src/observables.py``: read-only scalars computed from belief nets / running
log-evidences, never part of the dynamics.

  - ``oxygen_index``    : where a single belief sits on the phlogiston(0)->oxygen(1)
                          axis, read off its posterior mean on the mass-law nodes.
  - ``order_parameter`` : the population-mean oxygen index -- the curve that
                          crosses 1/2 at the population paradigm shift (the
                          paper's transition figure).
  - ``evidence_gap``    : log p(o|oxygen) - log p(o|phlogiston); its sign is the
                          held paradigm (the upper-envelope crossing).
  - ``carryover_mass``  : Frobenius norm of a hub's Schur fill-in -- the literal
                          "importance = fill-in" number.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.structural.belief import GaussianBeliefNet
from src.structural.bmr import carryover
from src.structural import linalg


def _name_indices(names: tuple[str, ...], subset: tuple[str, ...]) -> jax.Array:
    """Integer indices of ``subset`` within the basis ``names``."""
    return jnp.asarray([names.index(n) for n in subset])


def oxygen_index(mean: jax.Array, names: tuple[str, ...],
                 mass_nodes: tuple[str, ...],
                 mu_phlog: float, mu_oxy: float) -> jax.Array:
    """Map a belief's posterior mean on the mass-law nodes to a [0, 1] axis:
    0 = the phlogiston reading (calx lighter, ``mu_phlog``), 1 = the oxygen
    reading (calx heavier, ``mu_oxy``). Averaged over the mass nodes and clipped.
    """
    idx = _name_indices(names, mass_nodes)
    m = mean[idx].mean()
    frac = (m - mu_phlog) / (mu_oxy - mu_phlog)
    return jnp.clip(frac, 0.0, 1.0)


def population_means(Pi: jax.Array, h: jax.Array) -> jax.Array:
    """Posterior means for a stack of beliefs. ``Pi`` (N, d, d), ``h`` (N, d)
    -> (N, d), solving ``Pi_i mu_i = h_i`` per agent (never inverting)."""
    return jax.vmap(jnp.linalg.solve)(Pi, h)


def order_parameter(Pi: jax.Array, h: jax.Array, names: tuple[str, ...],
                    mass_nodes: tuple[str, ...],
                    mu_phlog: float, mu_oxy: float) -> jax.Array:
    """Population order parameter m(t): the mean oxygen index across agents.
    Starts near 0 (everyone phlogiston), approaches 1 as the population
    reorganises to oxygen. Crosses 1/2 at the population paradigm shift.
    """
    means = population_means(Pi, h)                       # (N, d)
    idx = _name_indices(names, mass_nodes)
    m = means[:, idx].mean(axis=1)                        # (N,)
    frac = (m - mu_phlog) / (mu_oxy - mu_phlog)
    return jnp.clip(frac, 0.0, 1.0).mean()


def evidence_gap(logZ_phlog: float, logZ_oxy: float) -> float:
    """log Bayes factor oxygen-over-phlogiston. > 0 => oxygen is the held
    paradigm (upper envelope); the sign crossing is the paradigm shift."""
    return float(logZ_oxy - logZ_phlog)


def held_paradigm(running_logZ: dict[str, float]) -> str:
    """The candidate paradigm currently on top of the evidence envelope."""
    return max(running_logZ, key=running_logZ.get)


def carryover_mass(net: GaussianBeliefNet, hub: tuple[str, ...]) -> float:
    """Frobenius norm of the hub's Schur fill-in ``Pi_ab Pi_bb^{-1} Pi_ba`` --
    the literal importance of the hub (how much re-coupling its removal forces
    among its neighbours). Large for a core hub, small for a belt node."""
    return float(jnp.linalg.norm(carryover(net, hub)))


# ----------------------------------------------------------------------
# Convergence / suppression time -- post-hoc read-offs of an m(t) trajectory.
# These operate on a host array (the (n_steps,) order-parameter trace returned
# by ``run_trace`` / ``run_bridge``); they are NOT part of the dynamics. Because
# the model is monostable (every population eventually relaxes to the truth), the
# interesting quantity is the *transient*: how long a wrong paradigm is held
# before the community converges. ``settling_time`` measures exactly that.
# ----------------------------------------------------------------------

def settling_time(m_t, asymptote: float | None = None,
                  tol: float = 0.05, k: int = 5) -> int:
    """First step from which ``m_t`` stays within ``tol`` of its asymptote for at
    least ``k`` consecutive steps -- the convergence / suppression time. A long
    settling time is a long-held wrong paradigm (Model A); a short one is a
    community that converged quickly (Model B after bridging).

    ``asymptote`` defaults to the trajectory's final value ``m_t[-1]`` (the
    realised attractor). Returns ``len(m_t)`` if the trajectory never settles to
    that band within the horizon -- i.e. "did not converge in the run" (log the
    horizon when reporting this, per the no-silent-caps rule)."""
    m = np.asarray(m_t, dtype=float)
    T = m.shape[0]
    a = float(m[-1]) if asymptote is None else float(asymptote)
    within = np.abs(m - a) <= tol                              # (T,) bool
    if k <= 0:
        raise ValueError("k must be >= 1")
    for i in range(T - k + 1):
        if within[i:i + k].all():
            return i
    return T


def time_to_half(m_t, level: float = 0.5) -> int:
    """First step at which ``m_t`` reaches ``level`` (default 1/2, the population
    paradigm-shift crossing). Returns ``len(m_t)`` if it never reaches ``level``
    within the horizon (e.g. a fully suppressed Model A that stays below 1/2)."""
    m = np.asarray(m_t, dtype=float)
    reached = np.nonzero(m >= level)[0]
    return int(reached[0]) if reached.size else int(m.shape[0])


# ----------------------------------------------------------------------
# Strain: WHERE is the model misspecified? -- node-splitting conflict measures
# (Presanis et al. 2013) as read-only diagnostics on a belief net. The idea:
# split the information feeding a node (or flowing across an edge) into two
# independent sources, and score the calibrated discrepancy of the two implied
# marginals, z^2 = delta^T (Sigma_1 + Sigma_2)^{-1} delta -- for 1-D Gaussian
# marginals simply (mu_1 - mu_2)^2 / (var_1 + var_2). z^2 ~ 1 is consistency;
# z^2 >> 1 localizes the conflict. ``edge_strain`` scores exactly the edit BMR
# would make (zero one prior edge, Savage-Dickey deposit swap), so it can
# *target* the existing pruning machinery (action.reduction_score) instead of
# scanning all edges. No dynamics here -- pure read-offs.
# ----------------------------------------------------------------------


def node_marginal(Pi: jax.Array, h: jax.Array, v: int
                  ) -> tuple[jax.Array, jax.Array]:
    """Exact 1-D marginal ``(mu_v, var_v)`` of node ``v``: ``schur_marginalize``
    onto the singleton ``{v}`` (all other nodes integrated out). ``v`` is a
    host-side int (these are post-hoc read-offs, not scan bodies)."""
    d = Pi.shape[-1]
    keep = jnp.asarray([v], dtype=jnp.int32)
    drop = jnp.asarray([i for i in range(d) if i != v], dtype=jnp.int32)
    Pi_m, h_m = linalg.schur_marginalize(Pi, h, keep, drop)
    var = 1.0 / Pi_m[0, 0]
    return h_m[0] * var, var


def gaussian_conflict_z2(mu1, var1, mu2, var2):
    """The calibrated conflict between two 1-D Gaussian sources for the same
    quantity: ``z^2 = (mu1 - mu2)^2 / (var1 + var2)`` -- the node-splitting
    discrepancy statistic (Presanis et al. 2013), ~chi^2_1 under consistency."""
    return (mu1 - mu2) ** 2 / (var1 + var2)


def node_split_strain(Pi_rest: jax.Array, h_rest: jax.Array,
                      prec_direct: float, mean_direct: float, v: int):
    """Node-splitting strain at node ``v``: the conflict between what the REST of
    the model implies about ``v`` (the marginal of ``(Pi_rest, h_rest)``, which
    must exclude the direct channel's deposit) and the DIRECT evidence channel
    summarised as ``N(mean_direct, 1/prec_direct)``. Large => the model's web and
    the node's own data disagree -- the misspecification lives at (or near) ``v``."""
    mu_r, var_r = node_marginal(Pi_rest, h_rest, v)
    return gaussian_conflict_z2(mu_r, var_r, mean_direct, 1.0 / prec_direct)


def edge_strain(Pi_post: jax.Array, h_post: jax.Array,
                Pi0: jax.Array, h0: jax.Array, u: int, v: int):
    """Edge strain (the zero-edge variant): how much the prior edge ``(u, v)`` is
    *fighting the data*, scored as the calibrated displacement of the endpoint
    marginals when the edge is removed from the prior under the SAME deposit.

    The reduced posterior is the Savage-Dickey swap (no logZ needed):
    ``Pi_red = zero_edge_prior(Pi0, u, v) + (Pi_post - Pi0)``, ``h_red = h_post``
    (the edge edit touches only ``Pi``). The strain is the max over the two
    endpoints of ``gaussian_conflict_z2`` between the full and reduced marginals.
    Consistent data leave the marginals where they were (z^2 < 1); a misspecified
    edge that drags an endpoint against its evidence snaps back on removal
    (z^2 >> 1). Scores exactly the edit ``bmr.prune_edge_prior`` /
    ``action.reduction_score`` would price -- so strain *targets* BMR."""
    Pi_red = linalg.zero_edge_prior(Pi0, u, v) + (Pi_post - Pi0)
    z = [gaussian_conflict_z2(*node_marginal(Pi_post, h_post, w),
                              *node_marginal(Pi_red, h_post, w))
         for w in (u, v)]
    return jnp.maximum(z[0], z[1])
