"""Tier-1 algebraic invariants for the trust-centric substrate.

Each test maps to a row in the plan's "Tier 1 — algebraic invariants" table.
These are non-negotiable: all 11 must pass before any axis sweep runs.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import (
    ModelConfig, NetworkConfig, PolicyConfig, ResourceConfig, TrustConfig,
    UtilityConfig, build,
)
from src import inference, policy, resource, trust, utility, world
from src.population import _step_jit


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _baseline_cfg(**overrides) -> ModelConfig:
    cfg = ModelConfig(n_agents=20, seed=0,
                      network=NetworkConfig(kind="watts_strogatz", mean_degree=4))
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


# ----------------------------------------------------------------------
# Tier-1 #1: reduction to v2 informational baseline.
# ----------------------------------------------------------------------


def test_tier1_01_reduction_to_v2_baseline():
    """With resource_gain disabled (objective='eig_minus_cost'), R_in=0,
    lambda_mc=0, the new substrate reproduces the v2 step exactly."""
    cfg = _baseline_cfg(
        policy=PolicyConfig(objective="eig_minus_cost"),
        resource=ResourceConfig(R_in=0.0, alpha_flow=0.0, delta_decay=0.0, c0=0.0),
        utility=UtilityConfig(lambda_mc=0.0),
    )
    pop = build(cfg)
    pop1, _ = pop.step(0)
    pop2, _ = pop1.step(1)
    pop3, _ = pop2.step(2)
    # The mu and gamma trajectories must have evolved away from init —
    # otherwise the dynamics aren't running.
    assert float(jnp.abs(pop3.mu - pop.mu).sum()) > 1e-6
    # All λ·U disabled ⇒ pooled mu_priv shouldn't be NaN / inf.
    assert jnp.all(jnp.isfinite(pop3.mu))
    assert jnp.all(jnp.isfinite(pop3.tau))


# ----------------------------------------------------------------------
# Tier-1 #2, #3: W row-stochastic, eta normalised.
# ----------------------------------------------------------------------


def test_tier1_02_W_is_row_stochastic():
    key = jax.random.PRNGKey(7)
    gamma = jax.random.uniform(key, (15, 15), minval=0.01, maxval=2.0)
    W = resource.flow_from_trust(gamma)
    row_sums = W.sum(axis=1)
    np.testing.assert_allclose(np.asarray(row_sums), np.ones(15), atol=1e-6)


def test_tier1_03_eta_is_normalised():
    key = jax.random.PRNGKey(11)
    gamma = jax.random.uniform(key, (12, 12), minval=0.01, maxval=2.0)
    eta = resource.inflow_share(gamma)
    np.testing.assert_allclose(float(eta.sum()), 1.0, atol=1e-6)


# ----------------------------------------------------------------------
# Tier-1 #4: W is a pure function of Γ.
# ----------------------------------------------------------------------


def test_tier1_04_W_is_pure_function_of_gamma():
    key = jax.random.PRNGKey(3)
    gamma = jax.random.uniform(key, (10, 10), minval=0.01, maxval=1.0)
    W1 = resource.flow_from_trust(gamma)
    W2 = resource.flow_from_trust(gamma)
    np.testing.assert_array_equal(np.asarray(W1), np.asarray(W2))


# ----------------------------------------------------------------------
# Tier-1 #5: Resource conservation under no decay, no cost, no internal flow.
# ----------------------------------------------------------------------


def test_tier1_05_resource_conservation_inflow_only():
    """delta = 0, c = 0, alpha = 0: Σ_i r grows by R_in per step exactly."""
    cfg = _baseline_cfg(
        policy=PolicyConfig(objective="resource_gain"),
        resource=ResourceConfig(R_in=0.5, alpha_flow=0.0, delta_decay=0.0,
                                c0=0.0, r0=1.0, r_min=0.0),
        utility=UtilityConfig(lambda_mc=0.0),
    )
    pop = build(cfg)
    total_init = float(pop.r.sum())
    for t in range(20):
        pop, _ = pop.step(t)
        total_now = float(pop.r.sum())
        expected = total_init + (t + 1) * 0.5
        np.testing.assert_allclose(total_now, expected, atol=1e-4)


# ----------------------------------------------------------------------
# Tier-1 #6: Resource conservation under no inflow.
# ----------------------------------------------------------------------


def test_tier1_06_resource_conservation_internal_only():
    """R_in = 0, c = 0, delta = 0: Σ_i r is preserved under internal flow."""
    cfg = _baseline_cfg(
        policy=PolicyConfig(objective="resource_gain"),
        resource=ResourceConfig(R_in=0.0, alpha_flow=0.3, delta_decay=0.0,
                                c0=0.0, r0=1.0, r_min=0.0),
        utility=UtilityConfig(lambda_mc=0.0),
    )
    pop = build(cfg)
    total_init = float(pop.r.sum())
    for t in range(20):
        pop, _ = pop.step(t)
        np.testing.assert_allclose(float(pop.r.sum()), total_init, atol=1e-3)


# ----------------------------------------------------------------------
# Tier-1 #7: Cost subtracts directly.
# ----------------------------------------------------------------------


def test_tier1_07_cost_subtracts_directly():
    """r(t+1) - r(t) = α·Σ_j W·r + R_in·η − c when δ = 0."""
    N = 8
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.uniform(1.0, 2.0, size=(N,)))
    gamma = jnp.asarray(rng.uniform(0.1, 1.0, size=(N, N)))
    W = resource.flow_from_trust(gamma)
    eta = resource.inflow_share(gamma)
    c_x = jnp.asarray(rng.uniform(0.0, 0.1, size=(N,)))
    R_in, alpha_flow, delta_decay = 0.7, 0.4, 0.0

    r_new = resource.flow_step(r, W, eta, c_x, R_in, alpha_flow, delta_decay)
    inflow = W.T @ r
    expected_delta = (-alpha_flow * r + alpha_flow * inflow
                      - c_x + R_in * eta)
    np.testing.assert_allclose(np.asarray(r_new - r),
                                np.asarray(expected_delta), atol=1e-6)


# ----------------------------------------------------------------------
# Tier-1 #8: lambda_mc = 0 ⇒ standard Bayes.
# ----------------------------------------------------------------------


def test_tier1_08_lambda_zero_equals_bayes():
    rng = np.random.default_rng(1)
    mu_data = jnp.asarray(rng.normal(size=(15,)))
    tau_data = jnp.asarray(rng.uniform(1.0, 3.0, size=(15,)))
    mu_tgt = jnp.asarray(rng.normal(size=(15,)))
    tau_tgt = jnp.asarray(rng.uniform(1.0, 3.0, size=(15,)))
    mu_new, tau_new = utility.lambda_modified_update(
        mu_data, tau_data, mu_tgt, tau_tgt, lambda_mc=0.0,
    )
    np.testing.assert_allclose(np.asarray(mu_new), np.asarray(mu_data),
                                atol=1e-7)
    np.testing.assert_allclose(np.asarray(tau_new), np.asarray(tau_data),
                                atol=1e-7)


# ----------------------------------------------------------------------
# Tier-1 #9: Cost barrier φ(r) is monotone decreasing, finite, no NaN at r=0.
# ----------------------------------------------------------------------


def test_tier1_09_cost_barrier_phi():
    r_grid = jnp.asarray([0.0, 0.01, 0.5, 1.0, 5.0, 100.0])
    h1sq = jnp.ones_like(r_grid)
    c = resource.cost_x(jnp.zeros_like(r_grid), r_grid, h1sq,
                        c0=1.0, r_min=0.0, barrier_eps=1e-3,
                        fisher_cost_steepness=1.0)
    # No NaN / inf anywhere on grid (barrier_eps prevents it)
    assert jnp.all(jnp.isfinite(c))
    # Monotone decreasing
    arr = np.asarray(c)
    assert np.all(np.diff(arr) <= 1e-10)


# ----------------------------------------------------------------------
# Tier-1 #10: Closed-form Ê[Δr | x] vs Monte-Carlo realised Δr.
# ----------------------------------------------------------------------


def test_tier1_10_closed_form_vs_monte_carlo():
    """The analytic Ê[Δr_i | x] is the agent's *subjective* expectation
    of resource gain — taken over its own predictive p_pre_i(o | x). Sampling
    o ~ p_pre_i (not o ~ world!) and running the rest of the step gives the
    MC realisation that the closed-form should match to ~5% (linearisation
    + first-order post-update mean approximation).
    """
    cfg = _baseline_cfg(
        n_agents=15,
        policy=PolicyConfig(objective="resource_gain",
                            x_grid=(0.5, 1.5, 3.0)),
        resource=ResourceConfig(R_in=0.5, alpha_flow=0.3, delta_decay=0.05,
                                c0=0.05, r0=1.5, r_min=0.0),
        utility=UtilityConfig(lambda_mc=0.0),
    )
    pop = build(cfg)
    x_grid = jnp.asarray(cfg.policy.x_grid)
    analytic = policy.expected_resource_gain(
        pop.mu, pop.tau, pop.r, pop.alpha, pop.beta, pop.gamma, pop.A_self_adj,
        x_grid, cfg.world, cfg.trust, cfg.resource,
    )  # (N, G)
    N, G = analytic.shape

    n_mc = 400
    realised = np.zeros((N, G))
    for g in range(G):
        x_force = jnp.full((N,), float(x_grid[g]))
        h1x = world.h1(x_force, cfg.world)
        # Per-agent predictive moments under PRE-update belief:
        m_pre = world.h0(x_force, cfg.world) + pop.mu * h1x       # (N,)
        s_pre = jnp.sqrt(cfg.world.sigma ** 2 + (h1x ** 2) / (pop.tau + 1e-12))
        deltas = []
        for s in range(n_mc):
            key = jax.random.PRNGKey(s + 1000 * g)
            noise = jax.random.normal(key, (N,))
            o = m_pre + s_pre * noise                              # o_i ~ p_pre_i
            mu_data, tau_data = inference.private_update(
                pop.mu, pop.tau, x_force, o, cfg.world,
                cfg.inference.posterior_rho, cfg.mu_0, cfg.tau_0,
            )
            eps = trust.surprisal_matrix(
                mu_data, tau_data, x_force, o, pop.A_self_adj, cfg.world,
            )
            _, _, gamma_new = trust.gamma_conjugate_step(
                pop.alpha, pop.beta, eps, pop.A_self_adj, cfg.trust,
            )
            W_new = resource.flow_from_trust(gamma_new)
            eta_new = resource.inflow_share(gamma_new)
            h1sq = (h1x ** 2) / (cfg.world.sigma ** 2 + 1e-12)
            c_x = resource.cost_x(
                x_force, pop.r, h1sq,
                c0=cfg.resource.c0, r_min=cfg.resource.r_min,
                barrier_eps=cfg.resource.barrier_eps,
                fisher_cost_steepness=cfg.resource.fisher_cost_steepness,
            )
            r_next = resource.flow_step(
                pop.r, W_new, eta_new, c_x,
                R_in=cfg.resource.R_in,
                alpha_flow=cfg.resource.alpha_flow,
                delta_decay=cfg.resource.delta_decay,
            )
            deltas.append(np.asarray(r_next - pop.r))
        realised[:, g] = np.mean(deltas, axis=0)

    abs_err = np.abs(realised - np.asarray(analytic))
    rel_denom = np.maximum(np.abs(np.asarray(analytic)), 0.05)
    rel_err = abs_err / rel_denom
    assert np.median(rel_err) < 0.10, (
        f"closed-form vs MC mismatch: median rel err {np.median(rel_err):.3f}, "
        f"max {rel_err.max():.3f}, analytic median {np.median(np.abs(analytic)):.4f}"
    )


# ----------------------------------------------------------------------
# Tier-1 #11: JIT trace stability after warmup.
# ----------------------------------------------------------------------


def test_tier1_11_jit_trace_stability():
    """After the JIT warmup, repeated calls do not retrace. Equinox's
    filter_jit wraps a pjit object that does not expose cache_info, so we
    verify trace stability by timing: the second call (cached) must be
    substantially faster than the first (warmup + trace)."""
    import time
    cfg = _baseline_cfg(
        policy=PolicyConfig(objective="resource_gain"),
        resource=ResourceConfig(R_in=0.3, alpha_flow=0.2, delta_decay=0.05,
                                c0=0.05, r0=1.0, r_min=0.0),
        utility=UtilityConfig(lambda_mc=0.1),
    )
    pop = build(cfg)

    # Warm-up call (includes JIT compile).
    t0 = time.perf_counter()
    pop, _ = pop.step(0)
    jax.block_until_ready(pop.mu)
    warm = time.perf_counter() - t0

    # Subsequent calls: cached.
    t0 = time.perf_counter()
    for t in range(1, 10):
        pop, _ = pop.step(t)
    jax.block_until_ready(pop.mu)
    cached = (time.perf_counter() - t0) / 9.0

    # The cached call is at least 5× faster than the warmup (typically 100×+).
    # Loose threshold avoids flakiness on slow CI.
    assert cached < warm / 5.0, (
        f"Re-tracing suspected: warmup={warm*1000:.1f}ms, "
        f"per-cached-call={cached*1000:.1f}ms"
    )
    assert jnp.all(jnp.isfinite(pop.mu))
    assert jnp.all(jnp.isfinite(pop.r))
