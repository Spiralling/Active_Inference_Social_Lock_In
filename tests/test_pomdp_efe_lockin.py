"""Tests for EFE experiment-selection lock-in (the principled nb19 mechanism).

The notebook-19 rebuild replaces the old hand-built attention gate with the
genuine active-inference mechanism: experiment choice is the EFE policy
``softmax(gamma * [epistemic(EIG) + pragmatic - cost])``. Self-censorship is
*derived* — a confident agent's expected information gain on the decisive
experiment shrinks (it predicts the outcome under its own paradigm), so a per-
experiment cost makes that experiment net-negative for confident agents only.

These tests guard, in order:
  1. regression — cost=0 / cost_scale=0 is byte-identical to the pre-cost EFE;
  2. belief-dependence of EIG — the decisive experiment's salience falls as the
     agent gets confident;
  3. the cost mechanism — a positive cost suppresses the decisive experiment for
     a CONFIDENT agent but not an UNCERTAIN one (confidence is the deciding var);
  4. monostability without cost — a confident-WRONG isolated agent still
     converges to the truth (the nb13/nb18 baseline) when cost=0;
  5. lock-in WITH cost in the commitment (greedy) limit — a confident-wrong agent
     past the conviction threshold stays permanently wrong;
  6. the sampled/greedy distinction — finite-gamma sampling escapes (long-horizon
     only), argmax is what makes it permanent;
  7. the honest-Bayes / martingale contract is untouched by the cost term.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.pomdp.gen_model import (
    PomdpConfig, build_generative_model, build_cost,
)
from src.pomdp.agent_pop import efe_terms, policy_posterior, sample_action
from src.pomdp import step as S, observables as O


XG = (0.1, 0.5, 1.0, 2.0, 3.0)


def _cfg(**kw):
    base = dict(x_grid=XG, true_paradigm=1, q_reliability=0.85, gamma_policy=4.0)
    base.update(kw)
    return PomdpConfig(**base)


# ----------------------------------------------------------------------
# 1. Regression: zero cost is the pure epistemic+pragmatic EFE
# ----------------------------------------------------------------------

def test_build_cost_zero_scale_is_zero():
    gm = build_generative_model(_cfg(cost_scale=0.0))
    assert np.allclose(np.asarray(gm["cost"]), 0.0)


def test_efe_cost_none_equals_cost_zero_vector():
    cfg = _cfg(cost_scale=0.0)
    gm = build_generative_model(cfg)
    q = jnp.array([0.4, 0.6])
    none = efe_terms(q, gm["A_world"], gm["C"], jnp.zeros(2), 0.0, cost=None)
    zero = efe_terms(q, gm["A_world"], gm["C"], jnp.zeros(2), 0.0,
                     cost=jnp.zeros(len(XG)))
    assert np.allclose(np.asarray(none["neg_efe"]), np.asarray(zero["neg_efe"]))


@pytest.mark.parametrize("bu_mode", ["confirm", "plan_tilt"])
def test_cost_zero_regression_matches_no_cost(bu_mode):
    """cost_scale=0 leaves the policy identical to the pre-cost EFE."""
    cfg = _cfg(cost_scale=0.0)
    gm = build_generative_model(cfg)
    q = jnp.array([0.45, 0.55])
    U = jnp.array([1.0, 0.0])
    with_cost, _ = policy_posterior(q, gm["A_world"], gm["C"], U, 2.0,
                                    cfg.gamma_policy, bu_mode=bu_mode,
                                    cost=gm["cost"])
    no_cost, _ = policy_posterior(q, gm["A_world"], gm["C"], U, 2.0,
                                  cfg.gamma_policy, bu_mode=bu_mode, cost=None)
    assert np.allclose(np.asarray(with_cost), np.asarray(no_cost), atol=1e-6)


def test_normalised_discriminability_cost_unit_max():
    """The discriminability cost base is normalised so the decisive experiment
    has unit base cost — cost_scale then reads in EFE (nat) units."""
    cost = np.asarray(build_cost(_cfg(cost_scale=0.7, cost_kind="discriminability")))
    assert np.isclose(cost.max(), 0.7, atol=1e-6)


# ----------------------------------------------------------------------
# 2. EIG is belief-dependent: salience of the decisive experiment falls
#    as the agent grows confident
# ----------------------------------------------------------------------

def test_eig_on_decisive_experiment_shrinks_with_confidence():
    cfg = _cfg(cost_scale=0.0)
    gm = build_generative_model(cfg)
    a_star = int(np.argmax(np.asarray(gm["d"])))   # most discriminating experiment

    def eig_decisive(qB):
        q = jnp.array([1 - qB, qB])
        t = efe_terms(q, gm["A_world"], gm["C"], jnp.zeros(2), 0.0, cost=None)
        return float(np.asarray(t["epistemic"])[a_star])

    e_unsure = eig_decisive(0.5)
    e_sure = eig_decisive(0.99)
    assert e_unsure > e_sure                       # confidence shrinks salience
    assert e_sure < 0.5 * e_unsure                 # and substantially


# ----------------------------------------------------------------------
# 3. The cost mechanism: confidence (not raw cost) flips the decisive choice
# ----------------------------------------------------------------------

def test_cost_suppresses_decisive_experiment_for_confident_only():
    """With a positive cost, an UNCERTAIN agent still prefers the decisive
    experiment while a CONFIDENT one abandons it — self-censorship is derived
    from belief, not imposed by a gate."""
    cfg = _cfg(cost_scale=0.4, gamma_policy=8.0)
    gm = build_generative_model(cfg)
    a_star = O.discriminating_experiment_index(gm["d"])

    def p_decisive(qB):
        q = jnp.array([1 - qB, qB])
        q_pi, _ = policy_posterior(q, gm["A_world"], gm["C"], jnp.zeros(2), 0.0,
                                   cfg.gamma_policy, bu_mode="plan_tilt",
                                   cost=gm["cost"])
        return float(np.asarray(q_pi)[a_star])

    # uncertain agent keeps meaningful mass on the decisive experiment;
    # a confident (wrong) agent self-censors it almost entirely.
    assert p_decisive(0.5) > p_decisive(0.99)
    assert p_decisive(0.999) < 0.5 * p_decisive(0.5)


# ----------------------------------------------------------------------
# 4. Without cost the substrate is monostable: confident-wrong -> truth
# ----------------------------------------------------------------------

def _run_confident_wrong(cost_scale, greedy, qB0, n=200, N=40, gamma=8.0,
                         social_mask=0.0, seed=0):
    cfg = _cfg(cost_scale=cost_scale, gamma_policy=gamma)
    D = np.tile([1 - qB0, qB0], (N, 1))            # start believing wrong paradigm A
    return S.run(cfg, N, n, D_per_agent=D, social_mask=social_mask,
                 bu_mode="plan_tilt", greedy=greedy, seed=seed)


def test_zero_cost_confident_wrong_converges_to_truth():
    out = _run_confident_wrong(cost_scale=0.0, greedy=True, qB0=0.005)
    summ = O.lockin_summary(out, true_paradigm=1)
    assert summ["converged_truth"]
    assert not summ["locked_wrong"]


# ----------------------------------------------------------------------
# 5. With cost + commitment (greedy), a confident-wrong agent locks in
# ----------------------------------------------------------------------

def test_cost_greedy_confident_wrong_locks_in():
    out = _run_confident_wrong(cost_scale=0.4, greedy=True, qB0=0.005)
    summ = O.lockin_summary(out, true_paradigm=1)
    assert summ["locked_wrong"]                    # never reaches the truth
    assert summ["crossed_at"] is None              # the shift is prevented
    # and self-censorship is visible: policy mass on the decisive experiment ~0
    assert out["p_discrim"][-1] < 0.05


def test_cost_greedy_neutral_start_still_finds_truth():
    """The locked basin is reached only from a confident-wrong start: a neutral
    agent (high EIG, runs the decisive experiment) still converges, so cost does
    not simply freeze everyone — it opens a *second* basin (genuine bistability),
    not a global stop."""
    cfg = _cfg(cost_scale=0.4, gamma_policy=8.0)
    N = 40
    D = np.tile([0.5, 0.5], (N, 1))
    out = S.run(cfg, N, 200, D_per_agent=D, social_mask=0.0,
                bu_mode="plan_tilt", greedy=True, seed=0)
    assert O.lockin_summary(out, true_paradigm=1)["converged_truth"]


def test_conviction_threshold_separates_basins():
    """Permanent lock-in requires the initial conviction to clear a threshold:
    a mildly-wrong agent escapes, a strongly-wrong one is captured (same cost)."""
    mild = _run_confident_wrong(cost_scale=0.4, greedy=True, qB0=0.08)
    deep = _run_confident_wrong(cost_scale=0.4, greedy=True, qB0=0.01)
    assert O.lockin_summary(mild, true_paradigm=1)["converged_truth"]
    assert O.lockin_summary(deep, true_paradigm=1)["locked_wrong"]


# ----------------------------------------------------------------------
# 6. Sampled vs greedy: finite-gamma escapes, argmax makes it permanent
# ----------------------------------------------------------------------

def test_sampled_policy_escapes_even_when_greedy_locks():
    """At finite gamma the softmax flattens as the confident agent's EIG shrinks
    on all experiments, so policy mass on the decisive experiment stays bounded
    away from zero and the agent eventually samples it and is freed. Only the
    argmax (commitment / gamma->inf) limit is strictly permanent."""
    qB0 = 0.005
    sampled = _run_confident_wrong(cost_scale=0.4, greedy=False, qB0=qB0, gamma=8.0)
    greedy = _run_confident_wrong(cost_scale=0.4, greedy=True, qB0=qB0, gamma=8.0)
    assert O.lockin_summary(sampled, true_paradigm=1)["converged_truth"]
    assert O.lockin_summary(greedy, true_paradigm=1)["locked_wrong"]


# ----------------------------------------------------------------------
# 6b. The compiled rollout reproduces the eager loop exactly
# ----------------------------------------------------------------------

def test_run_fast_matches_run():
    """``run_fast`` (lax.scan) must compute the identical dynamics as the eager
    ``run`` loop — same EFE+cost policy, same greedy selection, same Bayes
    update — so the phase-sweep speedup is not a behavioural shortcut."""
    cfg = _cfg(cost_scale=0.4, gamma_policy=8.0)
    N = 24
    D = np.tile([0.99, 0.01], (N, 1))
    slow = S.run(cfg, N, 120, D_per_agent=D, social_mask=0.0,
                 bu_mode="plan_tilt", greedy=True, seed=0)
    fast = S.run_fast(cfg, N, 120, D_per_agent=D, social_mask=0.0,
                      bu_mode="plan_tilt", greedy=True, seed=0)
    assert np.allclose(slow["mean_qB"], fast["mean_qB"], atol=1e-6)
    assert np.allclose(slow["p_discrim"], fast["p_discrim"], atol=1e-6)
    assert np.allclose(slow["chosen_d"], fast["chosen_d"], atol=1e-6)
    assert np.allclose(slow["final_q"], fast["final_q"], atol=1e-6)


# ----------------------------------------------------------------------
# 7. The honest-Bayes contract is untouched by the cost term
# ----------------------------------------------------------------------

def test_cost_does_not_touch_belief_update_martingale():
    """Cost lives in the policy (data acquisition), never the belief update.
    From a neutral start with cost, honest evidence still wins — the martingale
    wall is respected; lock-in came from biased *selection*, not biased updating."""
    cfg = _cfg(cost_scale=0.4, beta_U=8.0, gamma_policy=4.0)
    N = 60
    U = np.tile([1.0, 0.0], (N, 1))                # wants the wrong paradigm A
    D = np.tile([0.5, 0.5], (N, 1))                # but starts neutral
    out = S.run(cfg, N, 120, U_per_agent=U, D_per_agent=D,
                bu_mode="plan_tilt", greedy=False, seed=7)
    summ = O.trajectory_summary(out, cfg)
    assert summ["mean_belief_true"] > 0.8
    assert not summ["capture_against_evidence"]
