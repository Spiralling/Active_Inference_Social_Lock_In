"""Tests for the simplified theory-laden model (hierarchical context with proper AIF)."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.pomdp.gen_model import PomdpConfig, build_generative_model
from src.pomdp.simple_step import (
    SimpleConfig,
    build_joint_A_world,
    build_joint_A_social,
    build_B_c,
    transition_joint,
    marginalize_theta,
    marginalize_context,
    run_simple,
)


def _pomdp(**kw):
    base = dict(x_grid=(0.1, 0.3, 0.5, 0.8, 1.0), true_paradigm=1,
                q_reliability=0.85)
    base.update(kw)
    return PomdpConfig(**base)


def _cfg(**kw):
    pomdp_kw = kw.pop("pomdp_kw", {})
    base = dict(pomdp=_pomdp(**pomdp_kw), n_agents=40, n_steps=60, seed=42,
                obs_x_index=4)
    base.update(kw)
    return SimpleConfig(**base)


# ======================================================================
# Joint model algebra
# ======================================================================

class TestJointAlgebra:

    def test_joint_A_world_columns_normalized(self):
        gm = build_generative_model(_pomdp())
        A_j = build_joint_A_world(gm["A_world"], K=2)
        col_sums = np.asarray(A_j.sum(axis=0))
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-6)

    def test_joint_A_world_c0_paradigm_independent(self):
        gm = build_generative_model(_pomdp())
        K, C = 2, 2
        A_j = np.asarray(build_joint_A_world(gm["A_world"], K))
        col_c0_theta0 = A_j[:, 0 * C + 0, :]
        col_c0_theta1 = A_j[:, 1 * C + 0, :]
        np.testing.assert_allclose(col_c0_theta0, col_c0_theta1, atol=1e-6)

    def test_joint_A_world_c1_matches_original(self):
        gm = build_generative_model(_pomdp())
        K, C = 2, 2
        A_orig = np.asarray(gm["A_world"])
        A_j = np.asarray(build_joint_A_world(gm["A_world"], K))
        for k in range(K):
            np.testing.assert_allclose(
                A_j[:, k * C + 1, :], A_orig[:, k, :], atol=1e-5)

    def test_joint_A_social_c_independent(self):
        gm = build_generative_model(_pomdp())
        K, C = 2, 2
        A_soc = np.asarray(build_joint_A_social(gm["A_social"], K))
        for k in range(K):
            np.testing.assert_allclose(
                A_soc[:, k * C + 0], A_soc[:, k * C + 1], atol=1e-8)


# ======================================================================
# Transition model algebra
# ======================================================================

class TestTransition:

    def test_B_c_columns_normalized(self):
        for eps_c in [0.01, 0.05, 0.2]:
            for eps_r in [0.1, 0.3, 0.5]:
                B = np.asarray(build_B_c(eps_c, eps_r))
                np.testing.assert_allclose(B.sum(axis=0), 1.0, atol=1e-10)

    def test_transition_preserves_normalization(self):
        B_c = build_B_c(0.05, 0.30)
        q = jnp.array([[0.2, 0.3, 0.1, 0.4],
                        [0.1, 0.1, 0.3, 0.5]])
        q_t = transition_joint(q, B_c, K=2, C=2)
        sums = np.asarray(q_t.sum(axis=1))
        np.testing.assert_allclose(sums, 1.0, atol=1e-7)

    def test_transition_identity_at_trivial_B(self):
        """B_c = identity should not change the distribution."""
        B_c = build_B_c(eps_crisis=0.0, eps_resolve=0.0)
        q = jnp.array([[0.2, 0.3, 0.1, 0.4]])
        q_t = transition_joint(q, B_c, K=2, C=2)
        np.testing.assert_allclose(np.asarray(q_t), np.asarray(q), atol=1e-7)

    def test_transition_drives_toward_stationary(self):
        """Repeated transition should converge to stationary distribution of B_c."""
        B_c = build_B_c(eps_crisis=0.05, eps_resolve=0.30)
        # stationary: pi_0/pi_1 = eps_resolve/eps_crisis = 6
        pi_c = np.array([0.30 / 0.35, 0.05 / 0.35])

        q = jnp.array([[0.0, 0.5, 0.0, 0.5]])  # all mass on c=1
        for _ in range(200):
            q = transition_joint(q, B_c, K=2, C=2)
        q_c = np.asarray(marginalize_context(q, K=2, C=2)).ravel()
        np.testing.assert_allclose(q_c, pi_c, atol=0.02)

    def test_marginalize_recovers_theta(self):
        q_theta = jnp.array([[0.3, 0.7], [0.6, 0.4]])
        D_c = jnp.array([0.8, 0.2])
        q_joint = (q_theta[:, :, None] * D_c[None, None, :]).reshape(2, 4)
        recovered = marginalize_theta(q_joint, K=2, C=2)
        np.testing.assert_allclose(
            np.asarray(recovered), np.asarray(q_theta), atol=1e-7)


# ======================================================================
# Principled AIF dynamics
# ======================================================================

class TestPrincipledDynamics:

    def test_context_learns_from_evidence(self):
        """With low eps_resolve, anomalous evidence should push agents into
        crisis (c=1) and they should STAY there, updating paradigm beliefs."""
        cfg = SimpleConfig(
            pomdp=_pomdp(q_reliability=0.55),
            eps_crisis=0.10,
            eps_resolve=0.05,     # low: crises persist
            D_c_normal=0.9,
            n_agents=40,
            n_steps=100,
            obs_x_index=4,
            social_mask=0.0,      # no social — pure evidence
            seed=42,
        )
        out = run_simple(cfg)
        # agents should shift from c=0 (normal) toward c=1 (crisis)
        # because true paradigm 1 generates discriminating evidence
        assert out["mean_c_normal"][-1] < out["mean_c_normal"][0], (
            "evidence should shift agents out of normal science")

    def test_high_resolve_maintains_normal_science(self):
        """With high eps_resolve, agents snap back to normal science each step,
        staying theory-laden and slow to update paradigm beliefs."""
        cfg_low = SimpleConfig(
            pomdp=_pomdp(q_reliability=0.55),
            eps_crisis=0.05, eps_resolve=0.05,
            D_c_normal=0.9,
            n_agents=40, n_steps=80,
            obs_x_index=3, social_mask=0.0, seed=42,
        )
        cfg_high = SimpleConfig(
            pomdp=_pomdp(q_reliability=0.55),
            eps_crisis=0.05, eps_resolve=0.80,
            D_c_normal=0.9,
            n_agents=40, n_steps=80,
            obs_x_index=3, social_mask=0.0, seed=42,
        )
        out_low = run_simple(cfg_low)
        out_high = run_simple(cfg_high)
        cum_low = out_low["mean_qB"].sum()
        cum_high = out_high["mean_qB"].sum()
        assert cum_low > cum_high, (
            "low eps_resolve (persistent crisis) should converge faster "
            "than high eps_resolve (strong paradigm inertia)")


# ======================================================================
# Behavioral tests
# ======================================================================

class TestBehavior:

    def test_open_agents_converge_to_truth(self):
        """Low eps_resolve + no social -> agents enter crisis, learn truth."""
        cfg = _cfg(eps_crisis=0.10, eps_resolve=0.05,
                   D_c_normal=0.9, n_steps=100)
        out = run_simple(cfg)
        assert out["mean_qB"][-1] > 0.70

    def test_fully_insulated_freezes(self):
        """eps_crisis=0 + eps_resolve=1 + no social -> stuck in normal science."""
        cfg = _cfg(eps_crisis=0.0, eps_resolve=1.0,
                   D_c_normal=1.0, social_mask=0.0, n_steps=80)
        D_init = np.tile([0.7, 0.3], (cfg.n_agents, 1))
        out = run_simple(cfg, D_per_agent=D_init)
        drift = abs(out["final_q"].mean(axis=0)[0] - 0.7)
        assert drift < 0.1, f"beliefs drifted by {drift:.3f} despite full insulation"

    def test_paradigm_capture_via_inertia(self):
        """High eps_resolve + social coupling + wrong initial lean -> capture."""
        cfg = SimpleConfig(
            pomdp=_pomdp(q_reliability=0.65),
            eps_crisis=0.03,
            eps_resolve=0.70,     # strong paradigm inertia
            D_c_normal=0.95,
            n_agents=60,
            n_steps=200,
            social_mask=1.0,
            obs_x_index=4,
            seed=7,
        )
        D_init = np.tile([0.60, 0.40], (cfg.n_agents, 1))
        out = run_simple(cfg, D_per_agent=D_init)
        assert out["mean_qB"][-1] < 0.45, (
            f"Expected capture; got mean_qB={out['mean_qB'][-1]:.3f}")

    def test_low_inertia_escapes_capture(self):
        """Same conditions but high crisis rate + low resolve -> evidence wins."""
        cfg = SimpleConfig(
            pomdp=_pomdp(q_reliability=0.65),
            eps_crisis=0.30,      # high: agents readily detect anomalies
            eps_resolve=0.01,     # low: crises persist -> open to evidence
            D_c_normal=0.95,
            n_agents=60,
            n_steps=200,
            social_mask=1.0,
            obs_x_index=4,
            seed=7,
        )
        D_init = np.tile([0.60, 0.40], (cfg.n_agents, 1))
        out = run_simple(cfg, D_per_agent=D_init)
        assert out["mean_qB"][-1] > 0.55, (
            f"Expected truth; got mean_qB={out['mean_qB'][-1]:.3f}")


# ======================================================================
# Integration: drift + shapes + determinism
# ======================================================================

class TestIntegration:

    def test_theta_schedule_wired(self):
        from src.config import WorldConfig
        wc = WorldConfig(schedule="step", schedule_t_shift=30,
                         theta_star_pre=0.0, theta_star_post=1.0)
        cfg = _cfg(
            pomdp_kw=dict(world=wc),
            use_theta_schedule=True, n_steps=60)
        out = run_simple(cfg)
        assert out["theta_star_trace"][0] == 0.0
        assert out["theta_star_trace"][59] == 1.0

    def test_run_simple_shapes(self):
        cfg = _cfg(n_agents=20, n_steps=30)
        out = run_simple(cfg)
        assert out["mean_qB"].shape == (30,)
        assert out["occ_B"].shape == (30,)
        assert out["final_q"].shape == (20, 2)
        assert out["final_q_joint"].shape == (20, 4)
        assert out["mean_c_normal"].shape == (30,)
        assert out["T"].shape == (20, 20)
        assert out["theta_star_trace"].shape == (30,)
        assert len(out["infos"]) == 30

    def test_deterministic_seed(self):
        cfg = _cfg(n_agents=20, n_steps=30, seed=123)
        out1 = run_simple(cfg)
        out2 = run_simple(cfg)
        np.testing.assert_array_equal(out1["mean_qB"], out2["mean_qB"])
        np.testing.assert_array_equal(out1["final_q"], out2["final_q"])
