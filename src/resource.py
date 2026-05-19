"""Resource layer for the trust-centric coupled model (PDF §3).

The trust matrix Gamma is the only Bayesian-learned relational object; W and
eta are *deterministic readouts* of Gamma and are NOT carried as state by
the population:

    W = row-normalise(Gamma)                  # flow weights j -> i
        W_{ji} = gamma_{ji} / sum_k gamma_{jk}
        (i.e. each row of W is what fraction of j's resource flows to each
        target; rows sum to 1)

    eta_i = (sum_j gamma_{ji}) / (sum_{j,k} gamma_{jk})
        aggregate-incoming-trust column-sum, normalised across the
        population to a probability vector. Used as the exogenous share.

Resource recursion (Eq. flow-step):

    r_i(t+1) = (1 - delta) * r_i(t)
             + alpha * sum_j W_{ji} * r_j(t)
             - c(x_i; r_i)
             + R_in * eta_i

Two conservation invariants follow by construction (and are tested in
Tier-1 #5 and #6):

  - With delta = 0, c = 0, alpha = 0: sum_i r_i grows by exactly R_in per step.
  - With delta = 0, c = 0, R_in = 0: sum_i r_i is preserved iff alpha = 1
    AND W is column-stochastic. In general alpha is a fraction of outflow
    along row-stochastic W, so net mass change per step is
    (alpha - 1) * sum_i r_i when alpha != 1. Plan test #6 calls alpha the
    "fixed-fraction outflow", so we implement the mass-preserving form:
    r_i(t+1) inherits (1 - alpha) of its own resource plus alpha * inflow.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


EPS = 1e-12


def flow_from_trust(gamma: jax.Array) -> jax.Array:
    """Row-normalise Gamma to obtain the flow matrix W.

    W[j, i] is the fraction of j's outflowing resource that goes to i.
    Each row of W sums to 1 by construction.
    """
    row_sum = gamma.sum(axis=1, keepdims=True) + EPS
    return gamma / row_sum


def inflow_share(gamma: jax.Array) -> jax.Array:
    """Aggregate-incoming-trust column-sum, normalised to a probability vector.

    eta_i = (sum_j gamma_{ji}) / (sum_{j,k} gamma_{jk}).

    Encodes "how much of the total trust mass in the population points at i" —
    the share of the exogenous inflow R_in that agent i receives each step.
    """
    col_sum = gamma.sum(axis=0)
    total = col_sum.sum() + EPS
    return col_sum / total


def cost_x(x: jax.Array, r: jax.Array, h1_squared_over_sigma2: jax.Array,
           c0: float, r_min: float, barrier_eps: float,
           fisher_cost_steepness: float) -> jax.Array:
    """Apparatus cost c(x; r_i) = c0 * fisher_cost_steepness * h1(x)^2/sigma^2 * phi(r).

    barrier: phi(r) = 1 / max(r - r_min, barrier_eps).

    Monotone decreasing in r (rich agents pay less per Fisher unit), divergent
    as r -> r_min, never NaN. Inputs are broadcastable: pass r of shape (N,)
    and h1_squared_over_sigma2 of shape (N,) for the agent-realised cost; or
    pass shape (N, G) for the policy-time per-grid evaluation.
    """
    phi = 1.0 / jnp.maximum(r - r_min, barrier_eps)
    return c0 * fisher_cost_steepness * h1_squared_over_sigma2 * phi


def flow_step(r: jax.Array, W: jax.Array, eta: jax.Array,
              c_x: jax.Array, R_in: float,
              alpha_flow: float, delta_decay: float) -> jax.Array:
    """One step of the resource recursion.

    r        : (N,) current resources
    W        : (N, N) row-normalised trust flow matrix; W[j, i] = j -> i
    eta      : (N,) aggregate-incoming-trust shares (sum to 1)
    c_x      : (N,) realised cost c(x_i; r_i) for each agent
    R_in     : exogenous inflow scale (scalar)
    alpha    : internal-flow fraction (scalar)
    delta    : geometric decay (scalar)

    The form used here is mass-preserving when delta = 0, c = 0, R_in = 0:
    each agent keeps (1 - alpha) of its own resource and receives alpha
    times the trust-weighted inflow from neighbours (sum of W[:, i] * r,
    which is W.T @ r).
    """
    inflow = W.T @ r                                            # (N,) sum_j W[j,i] r_j
    r_new = ((1.0 - delta_decay) * r
             - alpha_flow * r                                   # outflow
             + alpha_flow * inflow                              # weighted inflow
             - c_x
             + R_in * eta)
    return r_new


def gini(r: jax.Array) -> jax.Array:
    """Gini coefficient of the resource distribution, vectorised.

    G = (sum_i sum_j |r_i - r_j|) / (2 * N * sum_i r_i).
    """
    abs_diff = jnp.abs(r[:, None] - r[None, :]).sum()
    denom = 2.0 * r.shape[0] * r.sum() + EPS
    return abs_diff / denom


def resource_weighted_mean(values: jax.Array, r: jax.Array) -> jax.Array:
    """sum_i (r_i / sum_k r_k) * values_i — population mean weighted by resource share."""
    w = r / (r.sum() + EPS)
    return (w * values).sum()
