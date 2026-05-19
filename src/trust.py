"""Gamma-conjugate Bayesian precision learning for the trust matrix
(PDF §2.3, eqs 6/7/8).

Two pure functions:

- ``surprisal_matrix``: ε_ij = −log q_j^priv(o_i | x_i) for j ≠ i (the
  surprisal that j's private posterior assigned to i's actual observation)
  and ε_ii = −log p_env(o_i | x_i, mu_i) for the self-loop (i's own
  predictive evaluated on its own observation). Both use ``log_predictive``
  from inference.py — the env model is just the predictive evaluated at
  the agent's own posterior mean.

- ``gamma_conjugate_step``: applies the per-edge update
      alpha_ij ← rho * alpha_ij + 1
      beta_ij  ← rho * beta_ij  + epsilon_ij
      gamma_ij ← alpha_ij / beta_ij
  with row-normalisation for numerical stability. When ``learning=False``,
  alpha and beta are returned unchanged (and gamma is recomputed from the
  unchanged hyperparameters — a no-op).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import TrustConfig, WorldConfig
from src.inference import log_predictive


EPS = 1e-12


def surprisal_matrix(mu_priv: jax.Array, tau_priv: jax.Array,
                     x_obs: jax.Array, o_obs: jax.Array,
                     mask_self_adj: jax.Array,
                     world_cfg: WorldConfig) -> jax.Array:
    """Compute ε_ij = −log q_j^priv(o_i | x_i) for every (i, j) in the
    closed neighbourhood (mask_self_adj is A + I).

    ``mu_priv``, ``tau_priv`` : (N,) j's private posterior parameters.
    ``x_obs``, ``o_obs``       : (N,) i's chosen x and resulting observation.
    ``mask_self_adj``          : (N, N) edge mask including diagonal.

    Returns ``epsilon`` : (N, N) masked surprisal matrix.

    Broadcasting: for fixed (x_i, o_i) and varying j's posterior, we compute
    log_predictive with mu = mu_priv[j], tau = tau_priv[j], x = x_obs[i],
    o = o_obs[i]. The result is shaped (N_i, N_j).
    """
    # log_predictive(mu, tau, x, o, cfg) — broadcastable over leading axes.
    # We want shape (N, N) where row i is i's observation, column j is j's posterior.
    mu_b = mu_priv[None, :]              # (1, N)
    tau_b = tau_priv[None, :]            # (1, N)
    x_b = x_obs[:, None]                 # (N, 1)
    o_b = o_obs[:, None]                 # (N, 1)
    log_p = log_predictive(mu_b, tau_b, x_b, o_b, world_cfg)  # (N, N)
    epsilon = -log_p
    return epsilon * mask_self_adj


def gamma_conjugate_step(alpha: jax.Array, beta: jax.Array,
                         epsilon: jax.Array, mask_self_adj: jax.Array,
                         cfg: TrustConfig
                         ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One step of the Gamma-conjugate update.

    ``mask_self_adj`` carries A + I so the self-loop participates in
    trust accumulation (PDF §2: γ_ii encodes self-trust on direct env
    observation).

    Returns (alpha_new, beta_new, gamma_new). gamma is re-derived from
    alpha/beta whether or not learning is enabled; row-normalisation is
    applied for numerical stability (the social-pool weights are
    scale-invariant, so the normalisation does not change downstream
    behaviour but keeps absolute magnitudes bounded over long runs).
    """
    if cfg.learning:
        alpha_new = cfg.rho * alpha + mask_self_adj
        beta_new = cfg.rho * beta + epsilon
    else:
        alpha_new = alpha
        beta_new = beta

    gamma_raw = (alpha_new / (beta_new + EPS)) * mask_self_adj   # (N, N)
    row_sum = gamma_raw.sum(axis=1, keepdims=True) + EPS
    gamma_new = (gamma_raw / row_sum) * mask_self_adj
    return alpha_new, beta_new, gamma_new
