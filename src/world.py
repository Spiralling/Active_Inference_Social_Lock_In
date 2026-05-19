"""The parameterised world (PDF §1.2).

Observations follow a regression family
    o | x, theta ~ N(h0(x) + theta * h1(x), sigma^2)
where x is the experimental setup (action), h0 is the shared baseline,
h1 is the discriminating function, and theta indexes theories.

For the relativistic-correction case (PDF Figure 1):
    h0(x) = x / 2          (Newtonian / Galilean baseline)
    h1(x) = x^3            (cubic correction term)
    theta = 0              (Newtonian)
    theta = theta_star     (relativistic)

The Fisher information I(theta; x) = h1(x)^2 / sigma^2 determines the
discriminating regime; the paradigm-consistent regime is where I is small.

This module is pure — no JAX modules, no side effects beyond the PRNG
threading that JAX makes explicit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import WorldConfig


def h0(x: jax.Array, cfg: WorldConfig) -> jax.Array:
    """Shared baseline: predictions all candidate theories agree on at small x."""
    if cfg.h0_kind == "linear_half":
        return 0.5 * x
    if cfg.h0_kind == "linear":
        return x
    if cfg.h0_kind == "zero":
        return jnp.zeros_like(x)
    raise ValueError(f"Unknown h0_kind: {cfg.h0_kind!r}")


def h1(x: jax.Array, cfg: WorldConfig) -> jax.Array:
    """Discriminating function. Its shape sets the geometry of the
    paradigm-crisis regime (PDF §1.2).

    ``power``     : h1(x) = x^k. Used in the paper with k=3.
    ``window``    : Gaussian bump at x0 with width w — localised crisis.
    ``threshold`` : indicator [x > xc] — phase-transition-like.
    ``oscillatory``: sin(omega * x) — periodic anomalies.
    """
    if cfg.h1_kind == "power":
        return x ** cfg.h1_k
    if cfg.h1_kind == "window":
        return jnp.exp(-((x - cfg.h1_window_x0) ** 2) / (cfg.h1_window_w ** 2))
    if cfg.h1_kind == "threshold":
        return (x > cfg.h1_threshold_xc).astype(x.dtype)
    if cfg.h1_kind == "oscillatory":
        return jnp.sin(cfg.h1_omega * x)
    raise ValueError(f"Unknown h1_kind: {cfg.h1_kind!r}")


def fisher_info(x: jax.Array, cfg: WorldConfig) -> jax.Array:
    """I(theta; x) = h1(x)^2 / sigma^2 (PDF eq 3). Identical for all theta
    in this family — linear-in-theta means Fisher information is parameter-
    free along theta."""
    return h1(x, cfg) ** 2 / (cfg.sigma ** 2)


def sample_o(x: jax.Array, theta_star: jax.Array, cfg: WorldConfig,
             key: jax.Array) -> jax.Array:
    """Sample one observation per agent.

    ``x`` is (N,); returns (N,). Variance is sigma^2 per draw, independent
    across agents.
    """
    mean = h0(x, cfg) + theta_star * h1(x, cfg)
    noise = jax.random.normal(key, x.shape) * cfg.sigma
    return mean + noise


def theta_schedule(t: int | jax.Array, cfg: WorldConfig) -> jax.Array:
    """The truth theta*(t) under the three supported schedules.

    ``step``     : theta_star_pre for t < t_shift, theta_star_post for t >= t_shift.
    ``reversal`` : pre -> post at t_shift, then post -> pre at t_reverse.
                   Used for hysteresis-loop probes.
    ``ramp``     : linear interpolation from pre to post over
                   [t_shift, t_shift + T_ramp].
    """
    t_arr = jnp.asarray(t, dtype=jnp.float32)
    pre = jnp.asarray(cfg.theta_star_pre, dtype=jnp.float32)
    post = jnp.asarray(cfg.theta_star_post, dtype=jnp.float32)

    if cfg.schedule == "step":
        return jnp.where(t_arr < cfg.schedule_t_shift, pre, post)

    if cfg.schedule == "reversal":
        t_rev = cfg.schedule_t_reverse
        if t_rev is None:
            raise ValueError("reversal schedule requires schedule_t_reverse")
        return jnp.where(
            t_arr < cfg.schedule_t_shift, pre,
            jnp.where(t_arr < t_rev, post, pre),
        )

    if cfg.schedule == "ramp":
        T = cfg.schedule_ramp_T
        frac = jnp.clip((t_arr - cfg.schedule_t_shift) / max(T, 1), 0.0, 1.0)
        return pre + frac * (post - pre)

    raise ValueError(f"Unknown schedule: {cfg.schedule!r}")
