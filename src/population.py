"""Population: thin orchestration of the trust-centric per-step cycle.

Composes pure modules:
  policy        — choose x_i (v2 EIG−cost OR new resource-gain objective)
  world         — sample o_i from N(h0 + theta* h1, sigma^2)
  inference     — private Bayes update
  utility       — social target + lambda-modified blend (Hyland λ·U term)
  inference     — precision-weighted social pool
  trust         — Gamma-conjugate update on (alpha, beta) per edge
  resource      — flow_from_trust → W, inflow_share → eta, flow_step → r

Per-step cycle (PDF §3, trust-centric refactor):
  1. choose experiment x_i^(t) via softmax-policy (objective: PolicyConfig)
  2. sample environment o_i ~ N(h0(x) + theta*(t) h1(x), sigma^2)
  3. form private posterior q_i^data from (x_i, o_i)
  4. compute socially-induced target (mu_tgt, tau_tgt) from neighbours weighted
     by resource; blend with q_i^data via lambda_mc (λ·U term)
  5. precision-pool the blended private posterior over the closed neighbourhood
  6. surprisal eps_ij; Gamma-conjugate update of (alpha, beta), gamma'
  7. W = row-normalise(gamma'); eta = col-sum-normalise(gamma'); r-step
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src import inference, policy, resource, trust, utility, world
from src.config import ModelConfig
from src.network import build_adjacency


class Population(eqx.Module):
    """Per-agent (mu, tau, r); per-edge (alpha, beta, gamma) trust state.

    W and eta are NOT carried as state — they are computed each step as
    deterministic readouts of gamma (architectural decision #1 of the plan).
    """

    cfg: ModelConfig = eqx.field(static=True)
    A_adj: jax.Array          # (N, N) raw adjacency, zero diagonal
    A_self_adj: jax.Array     # (N, N) adjacency + identity
    mu: jax.Array             # (N,) posterior mean over theta
    tau: jax.Array            # (N,) posterior precision
    alpha: jax.Array          # (N, N) Gamma shape, masked by A_self_adj
    beta: jax.Array           # (N, N) Gamma rate,  masked by A_self_adj
    gamma: jax.Array          # (N, N) alpha/beta, row-normalised
    r: jax.Array              # (N,) endogenous resource
    key: jax.Array

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def init(cls, cfg: ModelConfig, key: jax.Array) -> "Population":
        N = cfg.n_agents

        A_np = build_adjacency(
            n_agents=N,
            mean_degree=cfg.network.mean_degree,
            rewiring_p=cfg.network.rewiring_p,
            seed=cfg.seed,
            kind=cfg.network.kind,
            society_membership=None,
            intra_prob=cfg.network.intra_prob,
            inter_prob=cfg.network.inter_prob,
        )
        A_adj = jnp.asarray(A_np)
        A_self_adj = A_adj + jnp.eye(N, dtype=A_adj.dtype)

        mu = jnp.full((N,), cfg.mu_0)
        tau = jnp.full((N,), cfg.tau_0)

        n0 = cfg.trust.prior_n0
        beta0 = cfg.trust.prior_n0 * cfg.trust.prior_eps0
        alpha = n0 * A_self_adj
        beta_ = beta0 * A_self_adj
        gamma_raw = (alpha / (beta_ + 1e-12)) * A_self_adj
        row_sum = gamma_raw.sum(axis=1, keepdims=True) + 1e-12
        gamma = (gamma_raw / row_sum) * A_self_adj

        r0 = jnp.full((N,), cfg.resource.r0)

        return cls(
            cfg=cfg, A_adj=A_adj, A_self_adj=A_self_adj,
            mu=mu, tau=tau, alpha=alpha, beta=beta_, gamma=gamma,
            r=r0, key=key,
        )

    def __repr__(self) -> str:
        N = self.cfg.n_agents
        n_edges = int((self.A_adj != 0).sum() // 2)
        return (
            f"Population(N={N}, edges={n_edges}, "
            f"mu_mean={float(self.mu.mean()):+.3f}, "
            f"tau_mean={float(self.tau.mean()):.3f}, "
            f"gamma_mean={float(self.gamma[self.A_self_adj > 0].mean()):.4f}, "
            f"r_mean={float(self.r.mean()):.3f})"
        )

    # ------------------------------------------------------------------
    # One-step API
    # ------------------------------------------------------------------

    def step(self, t: int | jax.Array) -> tuple["Population", dict]:
        """One round of the per-step cycle. Returns (new_pop, step_out)."""
        new_pop, x_chosen, o_obs, theta_star = _step_jit(
            self, jnp.asarray(t, dtype=jnp.int32),
        )
        return new_pop, dict(
            theta_star=float(np.asarray(theta_star)),
            x_chosen=np.asarray(x_chosen),
            o_obs=np.asarray(o_obs),
        )


# ----------------------------------------------------------------------
# JIT-compiled core
# ----------------------------------------------------------------------


@eqx.filter_jit
def _step_jit(pop: "Population", t: jax.Array
              ) -> tuple["Population", jax.Array, jax.Array, jax.Array]:
    cfg = pop.cfg

    # 1. Policy: each agent picks x_i.
    key_pol, key_obs, key_next = jax.random.split(pop.key, 3)
    if cfg.policy.objective == "eig_minus_cost":
        x_chosen = policy.softmax_choose_x(pop.tau, key_pol, cfg.policy, cfg.world)
    elif cfg.policy.objective == "resource_gain":
        x_chosen = policy.resource_gain_policy(
            pop.mu, pop.tau, pop.r,
            pop.alpha, pop.beta, pop.gamma, pop.A_self_adj,
            key_pol, cfg.policy, cfg.world, cfg.trust, cfg.resource,
        )
    else:
        raise ValueError(f"Unknown policy.objective: {cfg.policy.objective!r}")

    # 2. Environment.
    theta_star = world.theta_schedule(t, cfg.world)
    o_obs = world.sample_o(x_chosen, theta_star, cfg.world, key_obs)

    # 3. Private Bayes update (data-driven).
    mu_data, tau_data = inference.private_update(
        pop.mu, pop.tau, x_chosen, o_obs, cfg.world,
        cfg.inference.posterior_rho, cfg.mu_0, cfg.tau_0,
    )

    # 4. λ·U term: socially-induced target + Hyland blend.
    mu_tgt, tau_tgt = utility.socially_induced_U(
        pop.mu, pop.tau, pop.r, pop.A_self_adj,
    )
    mu_priv, tau_priv = utility.lambda_modified_update(
        mu_data, tau_data, mu_tgt, tau_tgt, cfg.utility.lambda_mc,
    )

    # 5. Precision-pool with current gamma row.
    mu_pool, tau_pool = inference.precision_pool(
        mu_priv, tau_priv, pop.gamma, pop.A_self_adj,
    )

    # 6. Surprisal matrix + Gamma-conjugate trust update.
    epsilon = trust.surprisal_matrix(
        mu_priv, tau_priv, x_chosen, o_obs, pop.A_self_adj, cfg.world,
    )
    alpha_new, beta_new, gamma_new = trust.gamma_conjugate_step(
        pop.alpha, pop.beta, epsilon, pop.A_self_adj, cfg.trust,
    )

    # 7. Resource step: W and eta are deterministic readouts of gamma_new.
    W_new = resource.flow_from_trust(gamma_new)
    eta_new = resource.inflow_share(gamma_new)
    h1x_sq_over_sigma2 = (world.h1(x_chosen, cfg.world) ** 2) / (cfg.world.sigma ** 2 + 1e-12)
    c_x = resource.cost_x(
        x_chosen, pop.r, h1x_sq_over_sigma2,
        c0=cfg.resource.c0, r_min=cfg.resource.r_min,
        barrier_eps=cfg.resource.barrier_eps,
        fisher_cost_steepness=cfg.resource.fisher_cost_steepness,
    )
    r_new = resource.flow_step(
        pop.r, W_new, eta_new, c_x,
        R_in=cfg.resource.R_in,
        alpha_flow=cfg.resource.alpha_flow,
        delta_decay=cfg.resource.delta_decay,
    )

    new_pop = eqx.tree_at(
        lambda p: (p.mu, p.tau, p.alpha, p.beta, p.gamma, p.r, p.key),
        pop,
        (mu_pool, tau_pool, alpha_new, beta_new, gamma_new, r_new, key_next),
    )
    return new_pop, x_chosen, o_obs, theta_star
