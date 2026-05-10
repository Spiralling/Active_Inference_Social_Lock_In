"""Population: per-agent multi-feature salience priors plus inter-agent
coupling state.

Aligned with paper §4 (Algorithm 1):
  - Each step samples a situation σ_t (= live feature index this step).
  - Agents act on their preference for the live feature.
  - Posterior mixture (Eq 3) combines env channel (only for live feature)
    with trust-weighted social pool (every feature).
  - Per-norm closed-form C-update (Eq 5).
  - Trust update via prediction error on the live feature (Eq 7).

A ``pymdp.Agent`` instance is retained as a *symbolic* placeholder so the
active-inference idiom (``A``, ``B``, ``C``, ``D``, ``infer_states``) is
visible to AIF reviewers; the multi-feature model state lives in dedicated
``equinox.Module`` fields (``C``, ``q_post``, ``gamma``, ``lambda_``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pymdp.agent import Agent as PymdpAgent

from src.agent import (
    EPS,
    act_for_situation,
    neighbour_pool_feature,
    p_env_bernoulli,
    p_env_uniform,
    posterior_mixture,
    update_C_feature,
)
from src.config import ModelConfig, sample_lambda
from src.environment import CoordinationEnvironment
from src.network import build_adjacency


def _make_pymdp_agent(cfg: ModelConfig) -> PymdpAgent:
    """Symbolic pymdp.Agent — a 1-modality, 2-state placeholder.

    The model's actual salience prior lives in ``Population.C`` (shape
    (N, R)). This placeholder is kept so reviewers scanning the code
    see ``isinstance(agent, PymdpAgent)`` and the standard A/B/C/D
    surface, but no math runs through it.
    """
    N = cfg.n_agents
    n_obs = 2
    A = [jnp.broadcast_to(jnp.eye(n_obs)[None], (N, n_obs, n_obs))]
    B = [jnp.broadcast_to(jnp.eye(n_obs)[None, :, :, None], (N, n_obs, n_obs, 1))]
    C = [jnp.zeros((N, n_obs))]
    D = [jnp.full((N, n_obs), 1.0 / n_obs)]
    return PymdpAgent(A=A, B=B, C=C, D=D, batch_size=N, num_iter=1)


class Population(eqx.Module):
    """Multi-feature salience-prior population with learned trust precisions."""

    cfg: ModelConfig = eqx.field(static=True)

    agent: PymdpAgent          # symbolic AIF placeholder (see _make_pymdp_agent)
    A_adj: jax.Array           # (N, N) adjacency 0/1, zero diagonal
    gamma: jax.Array           # (N, N) trust precisions, masked by A_adj
    d: jax.Array               # (N,) delegation factors
    lambda_: jax.Array         # (N, R) per-norm stubbornness λ_{i,r}
    C: jax.Array               # (N, R) salience prior logits
    q_post: jax.Array          # (N, R, 2) per-feature Bernoulli posteriors
    key: jax.Array

    @classmethod
    def init(cls, cfg: ModelConfig, key: jax.Array) -> "Population":
        N, R = cfg.n_agents, cfg.n_features

        A_np = build_adjacency(
            n_agents=N, mean_degree=cfg.mean_degree,
            rewiring_p=cfg.rewiring_p, seed=cfg.seed, kind=cfg.network_kind,
        )
        A_adj = jnp.asarray(A_np)

        k_d, k_lam, k_main = jax.random.split(key, 3)
        d = jax.random.beta(k_d, cfg.delegation_alpha, cfg.delegation_beta, (N,))
        lambda_ = sample_lambda(cfg.lambda_dist, N, R, cfg.lambda_scope, k_lam)

        agent = _make_pymdp_agent(cfg)
        C = jnp.full((N, R), cfg.init_C_strength)
        q_post = jnp.full((N, R, 2), 0.5)
        gamma = cfg.init_gamma * A_adj

        return cls(
            cfg=cfg, agent=agent, A_adj=A_adj, gamma=gamma,
            d=d, lambda_=lambda_, C=C, q_post=q_post, key=k_main,
        )

    def __repr__(self) -> str:
        N, R = self.cfg.n_agents, self.cfg.n_features
        n_edges = int((self.A_adj != 0).sum() // 2)
        return (
            f"Population(N={N}, R={R}, edges={n_edges}, "
            f"d_mean={float(self.d.mean()):.3f}, "
            f"lambda_mean={float(self.lambda_.mean()):.3f}, "
            f"gamma_mean={float(self.gamma.sum() / max(self.A_adj.sum(), 1)):.3f}, "
            f"C_std={float(jnp.std(self.C)):.3f})"
        )

    # ------------------------------------------------------------------
    # one-step API — pure functional updates
    # ------------------------------------------------------------------

    def act(self, sigma_t: int | jax.Array) -> tuple["Population", jax.Array]:
        """Sample one binary action per agent for the live feature σ_t."""
        return _act_jit(self, jnp.asarray(sigma_t, dtype=jnp.int32))

    def observe(
        self, sigma_t: int | jax.Array, true_value: int | jax.Array
    ) -> "Population":
        """Apply the four updates (Algorithm 1 lines 5–9):

        1. p_env: Bernoulli(σ(γ_env)) on observed bit for the live feature;
           uniform for non-live features.
        2. p_social: trust-weighted neighbour pool per feature.
        3. q ← linear mixture (Eq 3); C ← closed-form (Eq 5).
        4. γ ← multiplicative decay on prediction error (Eq 7).
        """
        return _observe_jit(
            self,
            jnp.asarray(sigma_t, dtype=jnp.int32),
            jnp.asarray(true_value, dtype=jnp.int32),
        )

    def step(
        self,
        env: CoordinationEnvironment,
    ) -> tuple["Population", dict]:
        """Drive one full env-pop timestep. Returns (new_pop, env_out dict).

        Order matches paper Algorithm 1: situation drawn first, agents act
        conditional on situation, env evaluates and evolves, agents observe.
        """
        sigma_t = env.draw_situation()
        pop_acted, actions = self.act(sigma_t)
        env_out = env.step(np.asarray(actions), situation=sigma_t)
        new_pop = pop_acted.observe(sigma_t, int(env_out["true_value"]))
        return new_pop, env_out


# ----------------------------------------------------------------------
# JIT-compiled cores. The ``Population`` methods are thin wrappers so JAX
# transforms (jit, vmap, grad) cache by static cfg + shape signatures.
# ----------------------------------------------------------------------


@eqx.filter_jit
def _act_jit(pop: "Population", sigma_t: jax.Array) -> tuple["Population", jax.Array]:
    new_key, sub = jax.random.split(pop.key)
    keys = jax.random.split(sub, pop.cfg.n_agents)
    actions = jax.vmap(act_for_situation, in_axes=(0, None, 0))(pop.C, sigma_t, keys)
    return eqx.tree_at(lambda p: p.key, pop, new_key), actions


@eqx.filter_jit
def _observe_jit(
    pop: "Population", sigma_t: jax.Array, true_value: jax.Array
) -> "Population":
    """Per-step belief, preference and trust updates."""
    cfg = pop.cfg
    N, R = cfg.n_agents, cfg.n_features
    gamma_env = jnp.asarray(cfg.gamma_env)

    # -------------------------------------------------------------- #
    # 1. p_env per feature.
    #    Live feature σ_t: Bernoulli softened with γ_env on true_value.
    #    Non-live features: uniform [0.5, 0.5] (no env evidence this step).
    # -------------------------------------------------------------- #
    live_p = p_env_bernoulli(true_value, gamma_env)                # (2,)
    feature_idx = jnp.arange(R)
    is_live = (feature_idx == sigma_t)[:, None]                   # (R, 1)
    p_env_per_feature = jnp.where(is_live, live_p[None, :], p_env_uniform((R, 2)))
    p_env = jnp.broadcast_to(p_env_per_feature[None, :, :], (N, R, 2))

    # -------------------------------------------------------------- #
    # 2. p_social: trust-weighted pool of neighbours' previous q_post,
    #    per feature. q_post has shape (N, R, 2); transpose to (R, N, 2)
    #    so the inner vmap pools across agents per feature.
    # -------------------------------------------------------------- #
    q_by_feature = jnp.transpose(pop.q_post, (1, 0, 2))           # (R, N, 2)

    def pool_one_agent_one_feature(gamma_row, A_row, q_feature):
        return neighbour_pool_feature(gamma_row, A_row, q_feature)

    # vmap inner: across features at fixed agent
    # vmap outer: across agents
    def pool_one_agent(gamma_row, A_row):
        return jax.vmap(pool_one_agent_one_feature, in_axes=(None, None, 0))(
            gamma_row, A_row, q_by_feature
        )                                                          # (R, 2)

    p_social = jax.vmap(pool_one_agent, in_axes=(0, 0))(pop.gamma, pop.A_adj)  # (N, R, 2)

    # -------------------------------------------------------------- #
    # 3. Linear posterior mixture (Eq 3) and closed-form C-update (Eq 5).
    # -------------------------------------------------------------- #
    d_b = pop.d[:, None, None]                                     # (N, 1, 1)
    q_new = posterior_mixture(p_env, p_social, d_b)                # (N, R, 2)

    def update_C_one_agent(C_row, q_row, lambda_row):
        return jax.vmap(update_C_feature, in_axes=(0, 0, 0))(C_row, q_row, lambda_row)

    C_new = jax.vmap(update_C_one_agent, in_axes=(0, 0, 0))(
        pop.C, q_new, pop.lambda_,
    )                                                              # (N, R)

    # -------------------------------------------------------------- #
    # 4. Trust update (Eq 7) — uses *previous* q_post on the live feature
    #    (each neighbour j's track record before this step's update).
    #       ε_ij = -ln q_j^{(t)}(φ_{σ_t} = true_value)
    #       γ_ij ← γ_ij · exp(-η_γ · ε_ij)
    # -------------------------------------------------------------- #
    q_live_prev = pop.q_post[:, sigma_t, :]                        # (N, 2)
    p_true = q_live_prev[:, true_value] + EPS                      # (N,)
    eps_per_neighbour = -jnp.log(p_true)                            # (N,)
    decay = jnp.exp(-cfg.eta_gamma * eps_per_neighbour)             # (N,)
    gamma_new = pop.gamma * decay[None, :] * pop.A_adj              # (N, N)

    return eqx.tree_at(
        lambda p: (p.q_post, p.gamma, p.C),
        pop,
        (q_new, gamma_new, C_new),
    )
