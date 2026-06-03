"""Tests for dual-field utility scoring (src/structural/dual_field.py)."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from src.structural.belief import GaussianBeliefNet
from src.structural.bmr import bmr_prune, prune_edge_prior
from src.structural.dual_field import PrecisionUtilityNet, score_edit_with_utility


def _three_node_fixture():
    names = ("a", "b", "c")
    Pi = jnp.array(
        [
            [2.44340095, 0.31552731, -0.41184977],
            [0.31552731, 2.42869670, -0.62499969],
            [-0.41184977, -0.62499969, 5.09262175],
        ]
    )
    h = jnp.array([-1.26542147, -0.62327446, 0.04132598])
    u = jnp.array([0.90347018, 0.09401230, -0.74349925])
    return names, Pi, h, u


def test_utility_operator_row_stochastic_and_non_negative():
    names, Pi, h, u = _three_node_fixture()
    net = PrecisionUtilityNet(names=names, Pi=Pi, h=h, u=u)
    W = np.asarray(net.utility_operator())
    assert np.all(W >= -1e-12)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-8)


def test_effective_utility_equals_intrinsic_when_alpha_zero():
    names, Pi, h, u = _three_node_fixture()
    net = PrecisionUtilityNet(names=names, Pi=Pi, h=h, u=u, alpha=0.0)
    assert np.allclose(np.asarray(net.effective_utility()), np.asarray(u), atol=1e-8)


def test_cost_field_invariant_to_intrinsic_utility():
    names, Pi, h, u = _three_node_fixture()
    net_a = PrecisionUtilityNet(names=names, Pi=Pi, h=h, u=u)
    net_b = PrecisionUtilityNet(names=names, Pi=Pi, h=h, u=-3.0 * u)
    assert np.allclose(np.asarray(net_a.cost_field()), np.asarray(net_b.cost_field()), atol=1e-8)


def test_beta_zero_recovers_bmr_decision_and_delta_F():
    names, Pi0, h0, u = _three_node_fixture()
    Pi_post = Pi0 + jnp.array(
        [
            [0.03499385, -0.09510590, -0.08342804],
            [-0.09510590, 0.39114820, 0.07262132],
            [-0.08342804, 0.07262132, 0.47429307],
        ]
    )
    h_post = jnp.array([-0.58218974, -0.95587180, 0.21708101])

    prior = GaussianBeliefNet(Pi=Pi0, h=h0, names=names)
    post = GaussianBeliefNet(Pi=Pi_post, h=h_post, names=names)
    reduced_prior = prune_edge_prior(prior, ("a", "b"))

    bmr_out = bmr_prune(post, prior, reduced_prior)
    out = score_edit_with_utility(post, prior, reduced_prior, u, beta_u=0.0)

    assert np.isclose(float(out.delta_J), float(out.delta_F), atol=1e-8)
    assert np.isclose(float(out.delta_F), float(bmr_out["delta_F"]), atol=1e-8)
    assert out.favour_reduced == bool(bmr_out["favour_reduced"])


def test_increasing_beta_u_can_flip_decision():
    names, Pi0, h0, u = _three_node_fixture()
    Pi_post = jnp.array(
        [
            [2.47839480, 0.22042141, -0.49527781],
            [0.22042141, 2.81984490, -0.55237837],
            [-0.49527781, -0.55237837, 5.56691482],
        ]
    )
    h_post = jnp.array([-0.58218974, -0.95587180, 0.21708101])

    prior = GaussianBeliefNet(Pi=Pi0, h=h0, names=names)
    post = GaussianBeliefNet(Pi=Pi_post, h=h_post, names=names)
    reduced_prior = prune_edge_prior(prior, ("a", "b"))

    low_beta = score_edit_with_utility(post, prior, reduced_prior, u, beta_u=0.0)
    high_beta = score_edit_with_utility(post, prior, reduced_prior, u, beta_u=3.0)

    assert float(low_beta.delta_F) < 0.0
    assert float(low_beta.delta_U) > 0.0
    assert low_beta.favour_reduced is False
    assert high_beta.favour_reduced is True
