"""Hyland-style cost-of-mind-change term U(q_i; {q_j, r_j}_j).

The trust-centric model retains a single non-Bayesian piece in the posterior
update: a soft pull toward a *socially-induced target* shaped by neighbours'
posteriors, weighted by their resource. The target is computed once per
step and combined with the data-driven posterior by precision-weighted
blending.

Two pure functions:

- ``socially_induced_U(mu, tau, r, A_self) -> (mu_tgt, tau_tgt)``
    A closed-form Gaussian target per agent: a precision-weighted mean of
    neighbours' (mu_j, tau_j), with each neighbour additionally re-weighted
    by its resource share r_j / sum_k r_k inside i's closed neighbourhood.
    Resource thus modulates which neighbours "count" toward the target —
    the rich-and-confident dominate.

- ``lambda_modified_update(mu_data, tau_data, mu_tgt, tau_tgt, lambda_mc)
    -> (mu', tau')``
    Precision-weighted blend:
        tau' = tau_data + lambda_mc * tau_tgt
        mu'  = (tau_data * mu_data + lambda_mc * tau_tgt * mu_tgt) / tau'
    At lambda_mc = 0 this is the identity on (mu_data, tau_data) (Tier-1 #8).
    At lambda_mc -> infinity the agent's posterior collapses to the target.

Notes
-----
* This module reads ``r`` but ``inference.py`` does not — the contract that
  data-driven Bayes and resource-modulated social pressure are conceptually
  distinct is enforced by module boundary.
* The target is *closed-form* (no MC, no iteration) — JAX-pure and JIT-friendly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


EPS = 1e-12


def socially_induced_U(mu: jax.Array, tau: jax.Array, r: jax.Array,
                       A_self: jax.Array
                       ) -> tuple[jax.Array, jax.Array]:
    """Resource-weighted Gaussian target per agent.

    mu, tau, r : (N,) carrier-forward posteriors and current resource.
    A_self     : (N, N) closed-neighbourhood mask = A + I.

    For each i, the target is

        w_j        = r_j / sum_{k in N+[i]} r_k                # resource share
        tau_tgt_i  = sum_j (A_self[i, j] * w_j * tau_j)
        mu_tgt_i   = (sum_j A_self[i, j] * w_j * tau_j * mu_j) / tau_tgt_i

    Reduces to the standard precision-pool target when r is uniform.
    """
    mask_r = A_self * r[None, :]                                # (N, N) rows of r restricted to closed nbhd
    row_total_r = mask_r.sum(axis=1, keepdims=True) + EPS       # (N, 1)
    w = mask_r / row_total_r                                    # (N, N) resource shares per row

    tau_tgt = (w * tau[None, :]).sum(axis=1)                    # (N,)
    num = (w * (tau * mu)[None, :]).sum(axis=1)                 # (N,)
    mu_tgt = num / (tau_tgt + EPS)
    return mu_tgt, tau_tgt


def lambda_modified_update(mu_data: jax.Array, tau_data: jax.Array,
                           mu_tgt: jax.Array, tau_tgt: jax.Array,
                           lambda_mc: float
                           ) -> tuple[jax.Array, jax.Array]:
    """Precision-weighted blend of the Bayes posterior and the social target.

        tau' = tau_data + lambda_mc * tau_tgt
        mu'  = (tau_data * mu_data + lambda_mc * tau_tgt * mu_tgt) / tau'

    At lambda_mc = 0 returns (mu_data, tau_data) exactly.
    """
    tau_new = tau_data + lambda_mc * tau_tgt
    num = tau_data * mu_data + lambda_mc * tau_tgt * mu_tgt
    mu_new = num / (tau_new + EPS)
    return mu_new, tau_new
