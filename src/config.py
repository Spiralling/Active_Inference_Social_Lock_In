"""Configuration for the multi-agent active-inference coupling layer.

Mirrors the dataclass style of ``src/environment.py``. ``ModelConfig`` holds
every population-level knob; ``build()`` is the single factory that constructs
both an environment and a population with consistent ``n_agents`` and seeds.

Aligned with Paper §4 (multi-feature salience priors, per-norm cost of
mind-change).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from src.environment import CoordinationEnvironment, EnvConfig


@dataclass(frozen=True)
class ModelConfig:
    """Knobs for the agent layer.

    Notation follows ``paper/sections/04_model.tex``:
      - C_i ∈ R^R: salience prior over R Bernoulli features φ_r.
      - gamma_ij: edge precision (trust on incoming edge j -> i).
      - d_i: delegation factor — weight on social vs. own observation.
      - lambda_{i,r}: per-norm cost of mind-change.
    """

    n_agents: int = 300
    n_features: int = 4

    # Network topology.
    network_kind: str = "scale_free"          # "scale_free" (Barabási–Albert) or "watts_strogatz"
    mean_degree: int = 6                       # WS k; for BA, m = mean_degree // 2
    rewiring_p: float = 0.1                    # WS only

    # Delegation factor d_i ~ Beta(α, β) — fraction of evidence weight on
    # social channel; (1-d_i) on the env channel.
    delegation_alpha: float = 7.0
    delegation_beta: float = 3.0

    # Per-norm stubbornness λ_{i,r}, sampled per (agent, norm) or per agent.
    lambda_dist: tuple[str, dict[str, Any]] = field(
        default_factory=lambda: ("constant", {"value": 1.0})
    )
    lambda_scope: str = "per_agent"            # "per_agent" or "per_norm"

    # Initial salience prior strength: C_i = init_C_strength · 1 (uniform).
    init_C_strength: float = 0.0

    # Initial inter-agent precision γ_ij on every existing edge.
    init_gamma: float = 1.0

    # Environmental channel precision γ_env (Eq 3): logit on Bernoulli
    # likelihood of feature given observed bit. σ(γ_env) = prob assigned to
    # the observed bit. Default 2.0 → σ(2)≈0.88.
    gamma_env: float = 2.0

    # Trust-update learning rate η_γ in Eq 7.
    eta_gamma: float = 0.1

    seed: int = 0


def build(
    cfg: ModelConfig | None = None,
    env_cfg: EnvConfig | None = None,
) -> tuple[CoordinationEnvironment, "Population"]:
    """Construct (environment, population) with consistent shapes and seed."""
    from src.population import Population

    cfg = cfg or ModelConfig()
    env_cfg = env_cfg or EnvConfig(
        n_agents=cfg.n_agents,
        n_features=cfg.n_features,
        seed=cfg.seed,
    )

    if env_cfg.n_agents != cfg.n_agents:
        raise ValueError(
            f"env_cfg.n_agents ({env_cfg.n_agents}) != cfg.n_agents ({cfg.n_agents})"
        )
    if env_cfg.n_features != cfg.n_features:
        raise ValueError(
            f"env_cfg.n_features ({env_cfg.n_features}) != cfg.n_features ({cfg.n_features})"
        )

    env = CoordinationEnvironment(env_cfg)
    pop = Population.init(cfg, jax.random.PRNGKey(cfg.seed))
    return env, pop


def sample_lambda(
    spec: tuple[str, dict[str, Any]],
    n_agents: int,
    n_features: int,
    scope: str,
    key: jax.Array,
) -> jax.Array:
    """Sample per-(agent, norm) λ values. Returns shape (n_agents, n_features).

    ``scope='per_agent'`` draws (n_agents,) and broadcasts across norms — an
    agent has one stubbornness level applied to every norm. ``scope='per_norm'``
    draws (n_agents, n_features) i.i.d. — an agent can be loose on one norm
    and rigid on another; this is the multi-norm regime that motivates Eq (5).

    Supported distribution kinds:
      - "constant"  {"value": v}
      - "uniform"   {"lo": a, "hi": b}
      - "lognormal" {"mu": m, "sigma": s}
      - "beta"      {"alpha": a, "beta": b, "scale": s}
      - "bimodal"   {"low": a, "high": b, "p_low": p}
    """
    if scope == "per_agent":
        sample_shape: tuple[int, ...] = (n_agents,)
    elif scope == "per_norm":
        sample_shape = (n_agents, n_features)
    else:
        raise ValueError(f"Unknown lambda_scope: {scope!r}")

    kind, params = spec
    if kind == "constant":
        out = jnp.full(sample_shape, float(params["value"]))
    elif kind == "uniform":
        out = jax.random.uniform(
            key, sample_shape,
            minval=float(params["lo"]), maxval=float(params["hi"]),
        )
    elif kind == "lognormal":
        out = jnp.exp(
            jax.random.normal(key, sample_shape) * float(params["sigma"])
            + float(params["mu"])
        )
    elif kind == "beta":
        scale = float(params.get("scale", 1.0))
        out = scale * jax.random.beta(
            key, float(params["alpha"]), float(params["beta"]), sample_shape,
        )
    elif kind == "bimodal":
        u = jax.random.uniform(key, sample_shape)
        out = jnp.where(u < float(params["p_low"]), float(params["low"]), float(params["high"]))
    else:
        raise ValueError(f"Unknown lambda_dist kind: {kind!r}")

    if scope == "per_agent":
        out = jnp.broadcast_to(out[:, None], (n_agents, n_features))
    return out
