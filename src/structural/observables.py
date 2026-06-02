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
