"""Pure conjugate Gaussian inference under the regression family
o ~ N(h0(x) + theta * h1(x), sigma^2).

Two updates and one log-density utility:

- ``private_update``: standard linear-Gaussian Bayes step. After observing
  (x, o), an agent with prior q(theta) = N(mu, 1/tau) gets
      tau' = tau + h1(x)^2 / sigma^2
      mu'  = (tau * mu + h1(x) * (o - h0(x)) / sigma^2) / tau'

- ``precision_pool``: precision-weighted average over a neighbourhood
  (PDF eq 5). With row-normalised weights w_ij,
      tau_pool = sum_j w_ij * tau_j
      mu_pool  = sum_j w_ij * tau_j * mu_j / tau_pool

- ``log_predictive``: log density of the predictive p(o | x; mu, tau).
  The predictive is N(h0(x) + mu * h1(x), sigma^2 + h1(x)^2 / tau).
  Used both by trust (to score neighbours' predictions of o_i) and as
  self-surprisal (i scoring its own predictive on its own observation).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import WorldConfig
from src.world import h0, h1


EPS = 1e-12


def private_update(mu: jax.Array, tau: jax.Array,
                   x: jax.Array, o: jax.Array,
                   cfg: WorldConfig,
                   posterior_rho: float = 1.0,
                   prior_mu: float = 0.0,
                   prior_tau: float = 1.0) -> tuple[jax.Array, jax.Array]:
    """One Bayesian step per agent. All inputs are (N,).

    ``posterior_rho`` applies exponential forgetting that reverts the
    carried-forward posterior toward a fixed prior (``prior_mu``,
    ``prior_tau``) — the agent's anchoring paradigm. The prior carried into
    this step is the information-form convex blend

        tau_carry        = rho * tau + (1 - rho) * prior_tau
        tau_carry * mu_c = rho * tau * mu + (1 - rho) * prior_tau * prior_mu

    so a component that goes unobserved relaxes geometrically toward
    ``prior_mu`` (rate 1 - rho), while observations pull it toward the
    truth — a standing tension between evidence and the paradigm prior.

    At ``posterior_rho = 1.0`` (default) the blend collapses to the
    standard ratcheting conjugate-Gaussian update: precision only grows and
    the mean never reverts, so ``prior_mu`` / ``prior_tau`` have no effect.
    """
    h1x = h1(x, cfg)
    h0x = h0(x, cfg)
    sigma2 = cfg.sigma ** 2
    delta_tau = h1x ** 2 / sigma2
    tau_carry = posterior_rho * tau + (1.0 - posterior_rho) * prior_tau
    tau_mu_carry = (posterior_rho * tau * mu
                    + (1.0 - posterior_rho) * prior_tau * prior_mu)
    tau_new = tau_carry + delta_tau
    mu_new = (tau_mu_carry + h1x * (o - h0x) / sigma2) / (tau_new + EPS)
    return mu_new, tau_new


def precision_pool(mu_priv: jax.Array, tau_priv: jax.Array,
                   gamma: jax.Array, mask: jax.Array
                   ) -> tuple[jax.Array, jax.Array]:
    """Row-stochastic pooling over the closed neighbourhood.

    ``mu_priv``, ``tau_priv`` : (N,) private posteriors after own observation.
    ``gamma``                  : (N, N) trust matrix, gamma[i, j] = i's trust in j.
    ``mask``                   : (N, N) closed-neighbourhood mask = A_adj + I.

    Returns (mu_pool, tau_pool), each (N,). Each agent i pools the j∈{i}∪N(i)
    private posteriors with weights w_ij = gamma_ij / sum_k gamma_ik.
    """
    weights_raw = gamma * mask                          # (N, N)
    row_sum = weights_raw.sum(axis=1, keepdims=True)    # (N, 1)
    w = weights_raw / (row_sum + EPS)                   # (N, N), row-stochastic

    # tau_pool[i] = sum_j w_ij * tau_priv[j]
    tau_pool = (w * tau_priv[None, :]).sum(axis=1)      # (N,)
    # mu_pool[i] = sum_j w_ij * tau_priv[j] * mu_priv[j] / tau_pool[i]
    weighted_mean_num = (w * (tau_priv * mu_priv)[None, :]).sum(axis=1)
    mu_pool = weighted_mean_num / (tau_pool + EPS)
    return mu_pool, tau_pool


def log_predictive(mu: jax.Array, tau: jax.Array,
                   x: jax.Array, o: jax.Array,
                   cfg: WorldConfig) -> jax.Array:
    """Log p(o | x; mu, tau) under the predictive of an agent with posterior
    q(theta) = N(mu, 1/tau).

    Predictive: o | x ~ N(h0(x) + mu h1(x), sigma^2 + h1(x)^2 / tau).
    Returns scalar log-density per element of x (broadcasts).
    """
    h1x = h1(x, cfg)
    h0x = h0(x, cfg)
    pred_mean = h0x + mu * h1x
    pred_var = cfg.sigma ** 2 + (h1x ** 2) / (tau + EPS)
    log_norm = -0.5 * jnp.log(2.0 * jnp.pi * pred_var)
    log_kern = -0.5 * (o - pred_mean) ** 2 / pred_var
    return log_norm + log_kern
