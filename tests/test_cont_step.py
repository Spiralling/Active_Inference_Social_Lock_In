"""Tests for the continuous-lambda model (cont_step.py)."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.pomdp.gen_model import PomdpConfig, build_generative_model
from src.pomdp.agent_pop import infer_state
from src.pomdp.cont_step import (
    ContConfig, tempered_infer, update_lambda, run_cont,
)


def _pomdp(**kw):
    base = dict(x_grid=(0.1, 0.3, 0.5, 0.8, 1.0), true_paradigm=1,
                q_reliability=0.85)
    base.update(kw)
    return PomdpConfig(**base)


def _cfg(**kw):
    pomdp_kw = kw.pop("pomdp_kw", {})
    base = dict(pomdp=_pomdp(**pomdp_kw), n_agents=40, n_steps=60, seed=42,
                obs_x_index=4, lambda_init=1.0)
    base.update(kw)
    return ContConfig(**base)


# ======================================================================
# Tempered inference algebra
# ======================================================================

class TestTemperedInfer:

    def test_lambda_one_matches_standard(self):
        """At lambda=1, tempered_infer should match agent_pop.infer_state."""
        gm = build_generative_model(_pomdp())
        q = jnp.array([0.4, 0.6])
        o = jax.nn.one_hot(2, 5)
        o_soc = jnp.array([0.5, 0.5])
        action = jnp.int32(3)

        standard = infer_state(q, gm["A_world"], o, action,
                               gm["A_social"], o_soc, jnp.float32(1.0))
        tempered = tempered_infer(q, gm["A_world"], o, action,
                                  gm["A_social"], o_soc, jnp.float32(1.0),
                                  jnp.float32(1.0))
        np.testing.assert_allclose(
            np.asarray(tempered), np.asarray(standard), atol=1e-5)

    def test_high_lambda_freezes_beliefs(self):
        """At high lambda, world evidence is negligible — posterior ≈ prior."""
        gm = build_generative_model(_pomdp())
        q = jnp.array([0.7, 0.3])
        o = jax.nn.one_hot(0, 5)
        o_soc = jnp.array([0.5, 0.5])
        action = jnp.int32(4)

        post = tempered_infer(q, gm["A_world"], o, action,
                              gm["A_social"], o_soc, jnp.float32(0.0),
                              jnp.float32(100.0))
        np.testing.assert_allclose(np.asarray(post)[0], 0.7, atol=0.05)

    def test_low_lambda_amplifies_evidence(self):
        """At lambda < 1, evidence is overweighted."""
        gm = build_generative_model(_pomdp())
        q = jnp.array([0.5, 0.5])
        o = jax.nn.one_hot(3, 5)
        o_soc = jnp.array([0.5, 0.5])
        action = jnp.int32(4)

        post_std = tempered_infer(q, gm["A_world"], o, action,
                                  gm["A_social"], o_soc, jnp.float32(0.0),
                                  jnp.float32(1.0))
        post_amp = tempered_infer(q, gm["A_world"], o, action,
                                  gm["A_social"], o_soc, jnp.float32(0.0),
                                  jnp.float32(0.5))
        # amplified posterior should be more extreme
        max_std = float(jnp.max(post_std))
        max_amp = float(jnp.max(post_amp))
        assert max_amp >= max_std - 0.01


import jax


# ======================================================================
# Lambda dynamics
# ======================================================================

class TestLambdaDynamics:

    def test_large_update_increases_lambda(self):
        """Big KL → lambda goes up (more conservative)."""
        lam = jnp.array([1.5, 1.5])
        q_old = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        q_new = jnp.array([[0.9, 0.1], [0.5, 0.5]])
        lam_new = update_lambda(lam, q_new, q_old, eta=0.5,
                                kl_target=0.01, lam_min=0.5, lam_max=5.0)
        assert float(lam_new[0]) > float(lam[0])

    def test_small_update_decreases_lambda(self):
        """Small KL (below target) → lambda drifts down."""
        lam = jnp.array([2.0])
        q_old = jnp.array([[0.5, 0.5]])
        q_new = jnp.array([[0.501, 0.499]])
        lam_new = update_lambda(lam, q_new, q_old, eta=0.5,
                                kl_target=0.05, lam_min=0.5, lam_max=5.0)
        assert float(lam_new[0]) < float(lam[0])

    def test_lambda_clipped(self):
        lam = jnp.array([0.5, 5.0])
        q_old = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        q_new = jnp.array([[0.5, 0.5], [0.99, 0.01]])
        lam_new = update_lambda(lam, q_new, q_old, eta=1.0,
                                kl_target=0.01, lam_min=0.5, lam_max=5.0)
        assert float(lam_new[0]) >= 0.5
        assert float(lam_new[1]) <= 5.0


# ======================================================================
# Behavioral tests
# ======================================================================

class TestBehavior:

    def test_standard_bayes_converges(self):
        """lambda=1 (standard Bayes) should converge to truth."""
        cfg = _cfg(lambda_init=1.0, eta_lambda=0.0, n_steps=100)
        out = run_cont(cfg)
        assert out["mean_qB"][-1] > 0.75

    def test_conservative_slows_convergence(self):
        """lambda=3 should be slower than lambda=1."""
        out_std = run_cont(_cfg(lambda_init=1.0, eta_lambda=0.0, n_steps=80))
        out_con = run_cont(_cfg(lambda_init=3.0, eta_lambda=0.0, n_steps=80))
        assert out_std["mean_qB"].sum() > out_con["mean_qB"].sum()

    def test_lambda_dynamics_produce_capture(self):
        """With social coupling + adaptive lambda, wrong-leaning population
        should lock in: updates are costly → lambda rises → less responsive."""
        cfg = ContConfig(
            pomdp=_pomdp(q_reliability=0.65),
            lambda_init=2.0,
            eta_lambda=0.3,
            kl_target=0.01,
            n_agents=60, n_steps=200,
            social_mask=1.0, obs_x_index=4, seed=7,
        )
        D = np.tile([0.65, 0.35], (60, 1))
        out = run_cont(cfg, D_per_agent=D)
        assert out["mean_qB"][-1] < 0.5, (
            f"Expected capture; got {out['mean_qB'][-1]:.3f}")

    def test_low_lambda_escapes(self):
        """Same conditions but lambda=1 + no adaptation → truth wins."""
        cfg = ContConfig(
            pomdp=_pomdp(q_reliability=0.65),
            lambda_init=1.0,
            eta_lambda=0.0,
            n_agents=60, n_steps=200,
            social_mask=1.0, obs_x_index=4, seed=7,
        )
        D = np.tile([0.65, 0.35], (60, 1))
        out = run_cont(cfg, D_per_agent=D)
        assert out["mean_qB"][-1] > 0.5, (
            f"Expected truth; got {out['mean_qB'][-1]:.3f}")


# ======================================================================
# Integration
# ======================================================================

class TestIntegration:

    def test_shapes(self):
        cfg = _cfg(n_agents=20, n_steps=30)
        out = run_cont(cfg)
        assert out["mean_qB"].shape == (30,)
        assert out["mean_lambda"].shape == (30,)
        assert out["final_q"].shape == (20, 2)
        assert out["final_lam"].shape == (20,)

    def test_deterministic(self):
        cfg = _cfg(n_agents=20, n_steps=30, seed=99)
        out1 = run_cont(cfg)
        out2 = run_cont(cfg)
        np.testing.assert_array_equal(out1["mean_qB"], out2["mean_qB"])

    def test_resource_coupling_runs(self):
        """Smoke test: resource coupling doesn't crash."""
        cfg = _cfg(n_agents=20, n_steps=30, resource_coupling=True,
                   lambda_init=1.0, eta_lambda=0.0)
        out = run_cont(cfg)
        assert "mean_r" in out
        assert out["mean_r"].shape == (30,)
        assert out["final_r"].shape == (20,)
        assert np.all(np.isfinite(out["mean_r"]))

    def test_theta_schedule_wired(self):
        from src.config import WorldConfig
        wc = WorldConfig(schedule="step", schedule_t_shift=20,
                         theta_star_pre=0.0, theta_star_post=1.0)
        cfg = _cfg(pomdp_kw=dict(world=wc), use_theta_schedule=True,
                   n_steps=40, lambda_init=1.0, eta_lambda=0.0)
        out = run_cont(cfg)
        assert out["theta_star_trace"][0] == 0.0
        assert out["theta_star_trace"][39] == 1.0
