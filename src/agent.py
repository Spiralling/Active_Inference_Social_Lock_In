"""Per-agent update functions for the multi-feature salience-prior model.

Each agent carries a salience prior C_i ∈ ℝ^R parameterising R independent
Bernoulli features φ_r via p(φ_r=1 | c) = σ(c). All updates here operate
per feature; vmap across features and agents lives in ``Population``.

The three pieces are, in paper notation:

  Eq (3) — posterior mixture in distribution space:
      q_i^{(t+1)}(o) ∝ (1 - d_i) p_env(o | obs_i; γ_env)
                       + d_i Σ_j w_ij q_j^{(t)}(o)

  Eq (5) — per-norm C-update with asymmetric KL cost of mind-change:
      C_{i,r}^{(t+1)} = argmin_c [ KL(q_i(φ_r) ‖ p(φ_r|c))
                                 + λ_{i,r} KL(p(φ_r|c) ‖ p(φ_r|C_{i,r}^(t))) ]
      Closed form for Bernoulli (Hyland & Albarracin 2025 App. A):
          C_{i,r}^{(t+1)} = C_{i,r}^{(t)} + λ_{i,r}⁻¹ · logit q_i(φ_r=1)

  Eq (7) — multiplicative trust update from prediction error
      (implemented in ``Population``).

All functions are pure JAX and ``vmap``ed inside ``Population``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Floor for λ to avoid division-by-zero when sweeping λ→0 ("pure evidence").
LAMBDA_MIN = 1e-3
# Small ε for log-space safety.
EPS = 1e-12


def neighbour_pool_feature(
    gamma_row: jax.Array, A_row: jax.Array, q_all_feature: jax.Array
) -> jax.Array:
    """Trust-weighted average of neighbours' previous posteriors on one feature.

    ``gamma_row``        : (N,) precisions on incoming edges to agent i.
    ``A_row``            : (N,) 0/1 adjacency mask for agent i.
    ``q_all_feature``    : (N, 2) all agents' previous Bernoulli posteriors
                           on a single feature.
    Returns (2,) — the trust-weighted aggregate Bernoulli for agent i.
    """
    weights = gamma_row * A_row
    z = weights.sum() + EPS
    return (weights[:, None] * q_all_feature).sum(axis=0) / z


def p_env_bernoulli(obs_bit: jax.Array, gamma_env: jax.Array) -> jax.Array:
    """Bernoulli env-channel posterior under fixed channel precision γ_env.

    Given an observed bit ``obs_bit ∈ {0, 1}``, the env channel returns a
    Bernoulli probability with σ(γ_env) on the observed bit and σ(-γ_env)
    on the other. As γ_env → ∞, the posterior collapses to one-hot at obs.

    ``obs_bit``: scalar (or batched) int / 0-1 array.
    ``gamma_env``: scalar.
    Returns shape (..., 2): [P(φ=0), P(φ=1)].
    """
    p_correct = jax.nn.sigmoid(gamma_env)
    p_wrong = 1.0 - p_correct
    p_one = jnp.where(obs_bit == 1, p_correct, p_wrong)
    return jnp.stack([1.0 - p_one, p_one], axis=-1)


def p_env_uniform(shape: tuple[int, ...] = (2,)) -> jax.Array:
    """Uniform Bernoulli — used for non-live features (no env signal this step)."""
    return jnp.full(shape, 0.5)


def posterior_mixture(
    p_env: jax.Array, p_social: jax.Array, d: jax.Array
) -> jax.Array:
    """Eq (3): linear mixture of env and social Bernoulli posteriors.

    ``p_env``    : (..., 2) Bernoulli over the feature.
    ``p_social`` : (..., 2) Bernoulli — trust-weighted neighbour pool.
    ``d``        : scalar (broadcastable) — delegation factor in [0, 1].
    Returns (..., 2). No re-normalisation needed if both inputs are
    proper distributions (linear combination of stochastic vectors stays
    stochastic), but we clip and renormalise for numerical safety.
    """
    mixed = (1.0 - d) * p_env + d * p_social
    mixed = jnp.clip(mixed, EPS, 1.0)
    return mixed / mixed.sum(axis=-1, keepdims=True)


def update_C_feature(
    C_old_r: jax.Array, q_r: jax.Array, lambda_r: jax.Array
) -> jax.Array:
    """Closed-form Eq (5) update for a single Bernoulli feature.

    ``C_old_r`` : scalar — previous logit on feature r.
    ``q_r``     : (2,) — current posterior on feature r.
    ``lambda_r``: scalar — per-norm stubbornness λ_{i,r} ≥ 0.

    Derivation: argmin_c [ KL(q ‖ Bern(σ(c))) + λ KL(Bern(σ(c)) ‖ Bern(σ(C_old))) ].
    Setting ∂/∂c = 0 for Bernoulli → c* = C_old + λ⁻¹ · logit q(φ=1).

    λ → 0  : c* = logit q (Bayes-optimal projection of q onto the family).
    λ → ∞  : c* = C_old (frozen prior).
    """
    inv_lambda = 1.0 / jnp.maximum(lambda_r, LAMBDA_MIN)
    logit_q = jnp.log(q_r[1] + EPS) - jnp.log(q_r[0] + EPS)
    return C_old_r + inv_lambda * logit_q


def act_for_situation(
    C_row: jax.Array, sigma_t: jax.Array, key: jax.Array
) -> jax.Array:
    """Sample one binary action under the live feature's preference.

    ``C_row``  : (R,) the agent's salience prior.
    ``sigma_t``: scalar int — live feature index this step.
    ``key``    : PRNG key.
    Returns scalar int in {0, 1}.

    Action is Bernoulli(σ(C[σ_t])): the agent acts on its preference
    weight for the currently rewarded feature.
    """
    p_one = jax.nn.sigmoid(C_row[sigma_t])
    u = jax.random.uniform(key)
    return (u < p_one).astype(jnp.int32)
