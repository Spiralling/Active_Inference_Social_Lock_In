"""Active-inference experimental design.

Two objectives, selected by ``PolicyConfig.objective``:

1. ``"eig_minus_cost"`` — the v2 informational baseline, preserved verbatim
   for Tier-3 regression against notebooks 08–11:
        pi_i(x) ∝ exp(beta_exp * (EIG_i(x) − c(x; cost_slope, cost_kind)))

2. ``"resource_gain"`` — the trust-centric refactor. Agents sample x to
   maximise the expected change in their own resource:
        pi_i(x) ∝ exp(beta_exp * Ê[Δr_i | x])
   where Ê[Δr_i | x] decomposes as (PDF Eq. expected-resource-gain)
        α · Σ_j E[W_{ji}(t+1) | x] · r_j(t)        # internal flow
      + R_in · E[η_i(t+1) | x]                       # exogenous share
      − c(x; r_i)                                    # cost (Fisher-coupled barrier)
      − δ · r_i(t)                                   # decay

   E[W(t+1) | x] and E[η(t+1) | x] are recovered from a linearised
   projection of Γ(t+1) under the not-yet-observed o ~ q_i^priv(o | x):

        γ_{ji}(t+1) ≈ γ_{ji}(t) · (1 - β_{ji}^{-1}
                                  · (H(p_i, p_j; x) - ρβ_{ji} / (ρβ_{ji} + 1)))

   where H(p_i, p_j; x) is the closed-form Gaussian cross-entropy of i's
   predictive against j's. Information gain enters *instrumentally*: agents
   that probe high-h1(x) regions sharpen their predictives, which sharpens
   neighbours' cross-entropies, which shifts W and η.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import PolicyConfig, ResourceConfig, TrustConfig, WorldConfig
from src.resource import cost_x as resource_cost_x
from src.world import h0, h1


EPS = 1e-12


# ----------------------------------------------------------------------
# v2 baseline: information-gain / fixed-cost policy. Untouched.
# ----------------------------------------------------------------------


def expected_info_gain(x: jax.Array, tau_i: jax.Array,
                       world_cfg: WorldConfig) -> jax.Array:
    """EIG_i(x) = ½ log(1 + h1(x)^2 / (sigma^2 tau_i)). Output (N, G)."""
    h1x = h1(x, world_cfg)
    sigma2 = world_cfg.sigma ** 2
    ratio = (h1x ** 2)[None, :] / (sigma2 * tau_i[:, None] + EPS)
    return 0.5 * jnp.log1p(ratio)


def cost(x: jax.Array, cfg: PolicyConfig) -> jax.Array:
    """Legacy cost. Used only by the v2-baseline objective path."""
    abs_x = jnp.abs(x)
    if cfg.cost_kind == "abs":
        return cfg.cost_slope * abs_x
    if cfg.cost_kind == "quadratic":
        return cfg.cost_slope * abs_x ** 2
    if cfg.cost_kind == "exponential":
        return jnp.expm1(cfg.cost_slope * abs_x)
    raise ValueError(f"Unknown cost_kind: {cfg.cost_kind!r}")


def softmax_choose_x(tau: jax.Array, key: jax.Array,
                     policy_cfg: PolicyConfig,
                     world_cfg: WorldConfig) -> jax.Array:
    """v2 baseline. Sample x per agent from softmax(beta_exp * (EIG − cost))."""
    x_grid = jnp.asarray(policy_cfg.x_grid)
    eig = expected_info_gain(x_grid, tau, world_cfg)
    c = cost(x_grid, policy_cfg)
    efe = eig - c[None, :]
    logits = policy_cfg.beta_exp * efe
    keys = jax.random.split(key, tau.shape[0])
    idx = jax.vmap(lambda k, row: jax.random.categorical(k, row))(keys, logits)
    return x_grid[idx]


# ----------------------------------------------------------------------
# Trust-centric: resource-gain policy with closed-form Ê[Δr | x].
# ----------------------------------------------------------------------


def _predictive_moments(x_grid: jax.Array, mu: jax.Array, tau: jax.Array,
                        world_cfg: WorldConfig, post_update: bool = False
                        ) -> tuple[jax.Array, jax.Array]:
    """Predictive mean and variance per (agent, grid point).

    Returns (m, s2) each of shape (N, G):
        m[i, g]  = h0(x_g) + mu_i * h1(x_g)
        s2[i, g] = sigma^2 + h1(x_g)^2 / tau_i_eff

    If ``post_update`` is True, tau_i_eff is the data-updated precision
    tau_i + h1(x_g)^2 / sigma^2 — the moment that the *actual* step will
    score surprisals against. Mean uses current mu (E[mu_data | x] = mu by
    the conjugate posterior property), which is exact in expectation.
    """
    h0x = h0(x_grid, world_cfg)[None, :]                    # (1, G)
    h1x = h1(x_grid, world_cfg)[None, :]                    # (1, G)
    sigma2 = world_cfg.sigma ** 2
    m = h0x + mu[:, None] * h1x                             # (N, G)
    if post_update:
        tau_eff = tau[:, None] + (h1x ** 2) / sigma2
    else:
        tau_eff = tau[:, None]
    s2 = sigma2 + (h1x ** 2) / (tau_eff + EPS)              # (N, G)
    return m, s2


def expected_cross_surprisal(mu: jax.Array, tau: jax.Array,
                             x_grid: jax.Array,
                             world_cfg: WorldConfig) -> jax.Array:
    """E_{o~p_i}[-log p_j(o | x)] per (i, j, g).

    Closed-form Gaussian cross-entropy on POST-UPDATE predictive moments
    (mu_i, tau_i + h1(x_i)^2/sigma^2). The actual step computes the
    surprisal matrix against post-update private posteriors; using
    post-update tau here matches that channel to leading order in
    Var[mu_data]:

        H(p_i, p_j; x) = ½ log(2π s_j^2) + (s_i^2 + (m_i - m_j)^2) / (2 s_j^2).
    """
    m, s2 = _predictive_moments(x_grid, mu, tau, world_cfg, post_update=True)
    m_i = m[:, None, :]                                      # (N, 1, G)
    s2_i = s2[:, None, :]                                    # (N, 1, G)
    m_j = m[None, :, :]                                      # (1, N, G)
    s2_j = s2[None, :, :]                                    # (1, N, G)
    half_log = 0.5 * jnp.log(2.0 * jnp.pi * s2_j + EPS)
    quadratic = (s2_i + (m_i - m_j) ** 2) / (2.0 * s2_j + EPS)
    return half_log + quadratic                              # (N, N, G)


def project_gamma_under_x(alpha_trust: jax.Array, beta_trust: jax.Array,
                          gamma: jax.Array,
                          E_eps: jax.Array, mask_self_adj: jax.Array,
                          trust_cfg: TrustConfig) -> jax.Array:
    """Linearised projection of Γ(t+1) given each candidate x_g.

    The exact one-step update is
        alpha_ji' = rho * alpha_ji + 1
        beta_ji'  = rho * beta_ji  + eps_ji
        gamma_ji' = alpha_ji' / beta_ji'

    Taking E[eps_ji | x] = H(p_i, p_j; x), the expected post-step gamma is
        E[gamma_ji'] ≈ (rho * alpha + 1) / (rho * beta + E[eps_ji])
    which to first order in 1/(rho * beta) is

        E[gamma_ji'] ≈ gamma_ji * (1 - beta_ji^{-1} * (E[eps] - rho*beta/(rho*beta+1)))

    (the plan's first-order form). We compute the closed-form ratio
    directly, which is no more expensive and removes the linearisation
    bias.
    """
    rho = trust_cfg.rho
    alpha_new = rho * alpha_trust + mask_self_adj            # (N, N) — broadcast as (N, N, 1)
    # E_eps has shape (N, N, G); promote alpha_new, beta_trust to (N, N, 1)
    beta_new = rho * beta_trust[:, :, None] + E_eps          # (N, N, G)
    alpha_b = alpha_new[:, :, None]                          # (N, N, 1)
    gamma_proj = alpha_b / (beta_new + EPS)                  # (N, N, G)
    mask_b = mask_self_adj[:, :, None]
    return gamma_proj * mask_b


def W_and_eta_from_gamma(gamma_xyg: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-grid-point (W, eta) from a projected gamma tensor of shape (N, N, G).

    W_{j, i, g}   = gamma[j, i, g] / sum_k gamma[j, k, g]      # row-normalise per j
    eta_{i, g}    = (sum_j gamma[j, i, g]) / sum_{j,k} gamma[j, k, g]
    """
    row_sum = gamma_xyg.sum(axis=1, keepdims=True) + EPS     # (N, 1, G)
    W = gamma_xyg / row_sum                                  # (N, N, G)

    col_sum = gamma_xyg.sum(axis=0)                          # (N, G)
    total = col_sum.sum(axis=0, keepdims=True) + EPS         # (1, G)
    eta = col_sum / total                                    # (N, G)
    return W, eta


def expected_resource_gain(mu: jax.Array, tau: jax.Array, r: jax.Array,
                           alpha_trust: jax.Array, beta_trust: jax.Array,
                           gamma: jax.Array, mask_self_adj: jax.Array,
                           x_grid: jax.Array,
                           world_cfg: WorldConfig,
                           trust_cfg: TrustConfig,
                           resource_cfg: ResourceConfig) -> jax.Array:
    """Compute Ê[Δr_i | x] of shape (N, G).

    Following the mass-preserving form of ``resource.flow_step``
    ((1-δ-α)·r + α·(W.T @ r) - c + R_in·η):

      Δr_i = − (δ + α) · r_i                       (decay + outflow)
           + α · Σ_j W_{j,i}(t+1) · r_j(t)         (trust-weighted inflow)
           − c(x_i; r_i)                           (apparatus cost)
           + R_in · η_i(t+1)                       (exogenous share)

    The outflow term −α·r_i is required for self-consistency with
    ``resource.flow_step`` — without it, the policy systematically
    over-estimates the inflow-dominated regime by exactly α·r.
    """
    h1x_sq_over_sigma2 = (h1(x_grid, world_cfg) ** 2) / (world_cfg.sigma ** 2 + EPS)  # (G,)
    h1x_sq_over_sigma2_NG = jnp.broadcast_to(h1x_sq_over_sigma2[None, :],
                                              (mu.shape[0], x_grid.shape[0]))
    r_NG = jnp.broadcast_to(r[:, None], (mu.shape[0], x_grid.shape[0]))

    term_cost = resource_cost_x(
        x_grid, r_NG, h1x_sq_over_sigma2_NG,
        c0=resource_cfg.c0, r_min=resource_cfg.r_min,
        barrier_eps=resource_cfg.barrier_eps,
        fisher_cost_steepness=resource_cfg.fisher_cost_steepness,
    )  # (N, G)

    term_self = (resource_cfg.delta_decay + resource_cfg.alpha_flow) * r[:, None]  # (N, 1)

    E_eps = expected_cross_surprisal(mu, tau, x_grid, world_cfg)   # (N, N, G)
    gamma_proj = project_gamma_under_x(
        alpha_trust, beta_trust, gamma, E_eps, mask_self_adj, trust_cfg,
    )                                                        # (N, N, G)
    W_proj, eta_proj = W_and_eta_from_gamma(gamma_proj)      # (N, N, G), (N, G)

    # term_flow[i, g] = sum_j W_proj[j, i, g] * r[j]
    term_flow = resource_cfg.alpha_flow * (W_proj * r[:, None, None]).sum(axis=0)  # (N, G)
    term_inflow = resource_cfg.R_in * eta_proj                # (N, G)

    return term_flow + term_inflow - term_cost - term_self


def resource_gain_policy(mu: jax.Array, tau: jax.Array, r: jax.Array,
                         alpha_trust: jax.Array, beta_trust: jax.Array,
                         gamma: jax.Array, mask_self_adj: jax.Array,
                         key: jax.Array,
                         policy_cfg: PolicyConfig, world_cfg: WorldConfig,
                         trust_cfg: TrustConfig,
                         resource_cfg: ResourceConfig) -> jax.Array:
    """Sample one x per agent under the trust-centric resource-gain objective."""
    x_grid = jnp.asarray(policy_cfg.x_grid)
    gain = expected_resource_gain(
        mu, tau, r, alpha_trust, beta_trust, gamma, mask_self_adj,
        x_grid, world_cfg, trust_cfg, resource_cfg,
    )                                                        # (N, G)
    logits = policy_cfg.beta_exp * gain
    keys = jax.random.split(key, mu.shape[0])
    idx = jax.vmap(lambda k, row: jax.random.categorical(k, row))(keys, logits)
    return x_grid[idx]
