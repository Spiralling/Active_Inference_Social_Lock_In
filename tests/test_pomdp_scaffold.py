"""Phase-1 tests for the categorical epistemic-POMDP scaffold (src/pomdp/).

Mirrors the tier-1/tier-2 style of the continuous-substrate tests. These guard
the generative-model algebra, the belief-utility EFE contract (incl. the
proven vacuity of the linear form), the inter-agent step, and the central
structural finding (the martingale wall).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.pomdp.gen_model import (
    PomdpConfig, build_generative_model, build_A_world, build_B,
    discriminability,
)
from src.pomdp.agent_pop import efe_terms, policy_posterior
from src.pomdp import step as S, observables as O


def _cfg(**kw):
    base = dict(x_grid=(0.1, 0.3, 0.5, 0.8, 1.0), true_paradigm=1,
                q_reliability=0.85, gamma_policy=2.0)
    base.update(kw)
    return PomdpConfig(**base)


# ----------------------------------------------------------------------
# Generative-model algebra
# ----------------------------------------------------------------------

def test_A_world_columns_normalized():
    cfg = _cfg()
    A = np.asarray(build_A_world(cfg))
    assert A.shape == (cfg.n_o, cfg.n_paradigms, len(cfg.x_grid))
    assert np.allclose(A.sum(axis=0), 1.0, atol=1e-6)


def test_B_is_identity():
    cfg = _cfg()
    assert np.allclose(np.asarray(build_B(cfg)), np.eye(cfg.n_paradigms))


def test_discriminability_monotone_in_x():
    # h1(x)=x^3 => higher-x experiments separate the paradigms more
    d = discriminability(_cfg())
    assert np.all(np.diff(d) >= -1e-9)
    assert d[0] < d[-1]


def test_A_social_reliability():
    cfg = _cfg(q_reliability=0.8)
    As = np.asarray(build_generative_model(cfg)["A_social"])
    assert np.allclose(As, np.array([[0.8, 0.2], [0.2, 0.8]]), atol=1e-6)
    assert np.allclose(As.sum(axis=0), 1.0)


# ----------------------------------------------------------------------
# Belief-utility EFE contract
# ----------------------------------------------------------------------

def test_efe_regression_flat_U_equals_pure():
    """beta_U=0 and flat U give the same policy as pure (epistemic+pragmatic)."""
    cfg = _cfg()
    gm = build_generative_model(cfg)
    q = jnp.array([0.5, 0.5])
    base, _ = policy_posterior(q, gm["A_world"], gm["C"], jnp.zeros(2), 0.0, cfg.gamma_policy)
    flat, _ = policy_posterior(q, gm["A_world"], gm["C"], jnp.zeros(2), 1.0, cfg.gamma_policy,
                               bu_mode="confirm")
    assert np.allclose(np.asarray(base), np.asarray(flat), atol=1e-6)


def test_linear_belief_utility_is_vacuous():
    """The linear expected-belief term is constant across actions (martingale)."""
    cfg = _cfg()
    gm = build_generative_model(cfg)
    q = jnp.array([0.5, 0.5])
    t = efe_terms(q, gm["A_world"], gm["C"], jnp.array([0.0, 3.0]), beta_U=1.0,
                  bu_mode="expected_belief")
    bu = np.asarray(t["belief_utility"])
    assert np.allclose(bu, bu[0], atol=1e-6)   # identical across experiments


def test_confirm_belief_utility_is_nonvacuous():
    """The convex confirm term DOES shift the policy."""
    cfg = _cfg()
    gm = build_generative_model(cfg)
    q = jnp.array([0.5, 0.5])
    flat, _ = policy_posterior(q, gm["A_world"], gm["C"], jnp.zeros(2), 3.0, cfg.gamma_policy,
                               bu_mode="confirm")
    con, _ = policy_posterior(q, gm["A_world"], gm["C"], jnp.array([0.0, 3.0]), 3.0,
                              cfg.gamma_policy, bu_mode="confirm")
    assert not np.allclose(np.asarray(flat), np.asarray(con), atol=1e-3)


def test_plan_tilt_regresses_to_pure_at_beta0():
    cfg = _cfg()
    gm = build_generative_model(cfg)
    q = jnp.array([0.45, 0.55])
    a, _ = policy_posterior(q, gm["A_world"], gm["C"], jnp.array([1.0, 0.0]), 0.0,
                            cfg.gamma_policy, bu_mode="plan_tilt")
    b, _ = policy_posterior(q, gm["A_world"], gm["C"], jnp.zeros(2), 0.0, cfg.gamma_policy)
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-6)


# ----------------------------------------------------------------------
# Inter-agent step + the structural finding
# ----------------------------------------------------------------------

def test_step_runs_and_shapes():
    cfg = _cfg()
    out = S.run(cfg, n_agents=40, n_steps=10, seed=0)
    assert out["mean_qB"].shape == (10,)
    assert out["final_q"].shape == (40, cfg.n_paradigms)
    assert np.allclose(np.asarray(out["final_q"]).sum(axis=1), 1.0, atol=1e-5)


def test_honest_neutral_converges_to_truth():
    """Truth=B, neutral start, no belief-utility -> population learns the truth."""
    cfg = _cfg()
    N = 120
    D = np.tile([0.5, 0.5], (N, 1))
    out = S.run(cfg, N, 80, D_per_agent=D, bu_mode="plan_tilt", seed=1)
    assert out["mean_qB"][-1] > 0.8


def test_martingale_wall_belief_utility_cannot_flip_basin():
    """The central finding: from a neutral start, belief-utility toward the WRONG
    paradigm cannot capture it; honest evidence still wins."""
    cfg = _cfg(beta_U=8.0)
    N = 120
    U = np.tile([1.0, 0.0], (N, 1))      # wants wrong paradigm A
    D = np.tile([0.5, 0.5], (N, 1))
    out = S.run(cfg, N, 100, U_per_agent=U, D_per_agent=D, bu_mode="plan_tilt", seed=7)
    summ = O.trajectory_summary(out, cfg)
    assert summ["mean_belief_true"] > 0.8
    assert not summ["capture_against_evidence"]


def test_social_fold_initial_condition_locks_basin():
    """Starting leaning the wrong paradigm A + strong social -> lock-in to A
    (capture is social-fold + initial-condition, independent of belief-utility)."""
    cfg = _cfg(q_reliability=0.9)
    N = 120
    D = np.tile([0.65, 0.35], (N, 1))    # lean wrong paradigm A
    out = S.run(cfg, N, 100, D_per_agent=D, bu_mode="plan_tilt", seed=2)
    # population should NOT have reached the truth (basin held A)
    assert out["mean_qB"][-1] < 0.5
