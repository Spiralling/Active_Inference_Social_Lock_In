"""Tests for inferred reliability (lambda), content-gated trust (gamma) and edge strain.

Layer: structural · Guards: src/structural/reliability.py, the lambda/gamma wiring in
src/structural/step.py, and the strain read-offs in src/structural/observables.py ·
Map: tests/README.md

Four contracts:

  1. the Student-t reliability weight is exact at z=0, monotone in surprise, and
     an inferred outlier's deposit is strictly discounted;
  2. the social gate severs trust between divergent agents while respecting the
     carried W's support (rows stay stochastic, off-support stays zero);
  3. edge strain is calibrated (consistent data => z^2 < 1; a planted conflict
     lights up the conflicted edge by an order of magnitude);
  4. the new cfg gates default OFF bitwise (``None`` == the legacy dynamics,
     same PRNG) and are LIVE when set (outputs actually change).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.structural.phlogiston import StructuralConfig
from src.structural import linalg
from src.structural import observables as obs
from src.structural import reliability as rel
from src.structural import step as S
from src.structural.step import trust_weights

KEY = jax.random.PRNGKey(0)
NU = 4.0


# ----------------------------------------------------------------------
# 1. lambda sanity: the Student-t gate on a small PD net.
# ----------------------------------------------------------------------

def test_student_t_weight_at_zero_exact():
    assert float(rel.student_t_weight(jnp.asarray(0.0), NU)) == (NU + 1.0) / NU


def test_student_t_weight_monotone_decreasing():
    z2 = jnp.linspace(0.0, 50.0, 101)
    w = np.asarray(rel.student_t_weight(z2, NU))
    assert (np.diff(w) < 0).all()


def _two_node_belief():
    Pi = jnp.asarray([[2.0, -0.5], [-0.5, 2.0]])
    mu = jnp.asarray([0.5, -0.5])
    return Pi, Pi @ mu, mu


def test_channel_reliability_on_prediction_and_outlier():
    Pi, h, mu = _two_node_belief()
    H = jnp.eye(2)
    sigma_o = 1.0

    # on-prediction observation: z^2 = 0 on both channels => the exact maximum.
    lam = rel.channel_reliability(Pi, h, H, H @ mu, sigma_o, NU)
    assert np.allclose(np.asarray(lam), (NU + 1.0) / NU, atol=1e-6)

    # 10-sd outlier on channel 0 only: lambda_0 collapses, lambda_1 keeps max.
    sd0 = float(jnp.sqrt(rel.predictive_channel_variance(Pi, H, sigma_o)[0]))
    o = (H @ mu).at[0].add(10.0 * sd0)
    lam = rel.channel_reliability(Pi, h, H, o, sigma_o, NU)
    assert float(lam[0]) < 0.1 < float(lam[1])
    assert float(lam[1]) == pytest.approx((NU + 1.0) / NU, abs=1e-6)

    # the inferred-unreliable channel's deposit is strictly smaller with the
    # mechanism on (same observation either way).
    J_off, _ = linalg.fisher_deposit_weighted(H, o, sigma_o, jnp.ones(2))
    J_on, _ = linalg.fisher_deposit_weighted(H, o, sigma_o, jnp.ones(2) * lam)
    assert float(J_on[0, 0]) < float(J_off[0, 0])


# ----------------------------------------------------------------------
# 2. gamma gating: divergent agents lose trust; support is respected.
# ----------------------------------------------------------------------

def test_social_gamma_severs_divergent_agent():
    # N=3 complete graph on a 2-node basis; agent 2 shifted 5 sd on each node.
    Pi = jnp.broadcast_to(4.0 * jnp.eye(2), (3, 2, 2))
    var_pair = 1.0 / 4.0 + 1.0 / 4.0          # denom of the pairwise z^2 per node
    shift = 5.0 * float(np.sqrt(var_pair))
    mu = jnp.asarray([[0.0, 0.0], [0.0, 0.0], [shift, shift]])
    h = jnp.einsum("nab,nb->na", Pi, mu)

    gamma = rel.social_gamma(Pi, h, jnp.asarray([0, 1]), nu_s=1.0)
    assert float(gamma[0, 2]) < 0.1 * float(gamma[0, 1])
    # symmetric, max trust on the diagonal ((nu_s+1)/nu_s).
    assert np.allclose(np.asarray(gamma), np.asarray(gamma).T, atol=1e-6)
    assert np.allclose(np.diag(np.asarray(gamma)), 2.0, atol=1e-6)


def test_gated_W_row_stochastic_and_support_respected():
    groups = [{"count": 3, "stance": -2.0, "prec_scale": 10.0},
              {"count": 3, "stance": 2.0, "prec_scale": 10.0}]
    cfg = StructuralConfig(n_agents=6, n_steps=5, social_nu=1.0)
    ring = sum(jnp.roll(jnp.eye(6), k, axis=1) for k in (-1, 1))
    W = trust_weights(ring + jnp.eye(6))
    state = S.init_state(cfg, KEY, groups, W_override=W)

    Wg = S._gated_W(state.Pi, state.h, W, cfg)
    assert np.allclose(np.asarray(Wg).sum(axis=1), 1.0, atol=1e-6)
    assert np.asarray(Wg)[np.asarray(W) == 0.0].max() == 0.0
    # the gate actually moved weight: cross-group neighbours are down-weighted.
    assert not np.allclose(np.asarray(Wg), np.asarray(W), atol=1e-6)


# ----------------------------------------------------------------------
# 3. strain calibration on a 3-node chain.
# ----------------------------------------------------------------------

def test_gaussian_conflict_z2_exact():
    assert float(obs.gaussian_conflict_z2(0.0, 1.0, 2.0, 1.0)) == 2.0


def _chain(d0: float, b: float):
    Pi0 = d0 * jnp.eye(3)
    Pi0 = Pi0.at[0, 1].set(b).at[1, 0].set(b).at[1, 2].set(b).at[2, 1].set(b)
    return Pi0, jnp.zeros(3)


def _chain_posterior(Pi0, h0, phi, sigma_o, T):
    H = jnp.eye(3)
    J, j = linalg.fisher_deposit(H, H @ phi, sigma_o)   # noise-free o = H @ phi
    return Pi0 + T * J, h0 + T * j


def test_edge_strain_consistent_vs_planted_conflict():
    # strong chain (PD: min eig = 6 - 3.5*sqrt(2) > 0) so the edge carries real load.
    Pi0, h0 = _chain(6.0, -3.5)
    assert float(jnp.linalg.eigvalsh(Pi0)[0]) > 0.0

    # consistent data: both edges' removal barely moves the marginals.
    Pc, hc = _chain_posterior(Pi0, h0, jnp.asarray([1.0, 1.0, 1.0]), 1.0, 12)
    assert float(obs.edge_strain(Pc, hc, Pi0, h0, 0, 1)) < 1.0
    assert float(obs.edge_strain(Pc, hc, Pi0, h0, 1, 2)) < 1.0

    # planted conflict on A (10 sd against the A-B edge's pull): the conflicted
    # edge lights up by an order of magnitude, the off-conflict edge stays cold.
    Pk, hk = _chain_posterior(Pi0, h0, jnp.asarray([-10.0, 1.0, 1.0]), 1.0, 12)
    s_ab = float(obs.edge_strain(Pk, hk, Pi0, h0, 0, 1))
    s_bc = float(obs.edge_strain(Pk, hk, Pi0, h0, 1, 2))
    assert s_ab > 10.0
    assert s_ab > 5.0 * s_bc


def test_node_marginal_matches_dense_inverse():
    Pi0, h0 = _chain(2.0, -0.8)
    h = Pi0 @ jnp.asarray([0.3, -0.2, 0.7])
    Sigma = np.linalg.inv(np.asarray(Pi0))
    mu = Sigma @ np.asarray(h)
    for v in range(3):
        m, var = obs.node_marginal(Pi0, h, v)
        assert float(m) == pytest.approx(mu[v], abs=1e-5)
        assert float(var) == pytest.approx(Sigma[v, v], abs=1e-5)


# ----------------------------------------------------------------------
# 4. byte-identity of the default path; the gates are live when set.
# ----------------------------------------------------------------------

GROUPS = [{"count": 4, "stance": -2.0, "prec_scale": 10.0},
          {"count": 4, "stance": 2.0, "prec_scale": 10.0}]


def test_defaults_bitwise_equal_explicit_nones():
    cfg_a = StructuralConfig(n_agents=8, n_steps=10)
    cfg_b = StructuralConfig(n_agents=8, n_steps=10,
                             reliability_nu=None, social_nu=None)

    fin_a = S.run_final(cfg_a, S.init_state(cfg_a, jax.random.PRNGKey(0)))
    fin_b = S.run_final(cfg_b, S.init_state(cfg_b, jax.random.PRNGKey(0)))
    assert np.array_equal(np.asarray(fin_a.Pi), np.asarray(fin_b.Pi))
    assert np.array_equal(np.asarray(fin_a.h), np.asarray(fin_b.h))

    ms_a, gw_a = S.run_trace(cfg_a, S.init_state(cfg_a, jax.random.PRNGKey(0)))
    ms_b, gw_b = S.run_trace(cfg_b, S.init_state(cfg_b, jax.random.PRNGKey(0)))
    assert np.array_equal(np.asarray(ms_a), np.asarray(ms_b))
    assert np.array_equal(np.asarray(gw_a), np.asarray(gw_b))


def test_reliability_gate_is_live():
    cfg0 = StructuralConfig(n_agents=8, n_steps=10)
    cfg1 = dataclasses.replace(cfg0, reliability_nu=4.0)
    ref = S.run_final(cfg0, S.init_state(cfg0, jax.random.PRNGKey(0)))
    out = S.run_final(cfg1, S.init_state(cfg1, jax.random.PRNGKey(0)))
    assert not np.allclose(np.asarray(out.h), np.asarray(ref.h), atol=1e-8)


def test_social_gate_is_live_with_heterogeneous_groups():
    cfg0 = StructuralConfig(n_agents=8, n_steps=10)
    cfg1 = dataclasses.replace(cfg0, social_nu=1.0)
    ref = S.run_final(cfg0, S.init_state(cfg0, jax.random.PRNGKey(0), GROUPS))
    out = S.run_final(cfg1, S.init_state(cfg1, jax.random.PRNGKey(0), GROUPS))
    assert not np.allclose(np.asarray(out.h), np.asarray(ref.h), atol=1e-8)


def test_reliability_gate_in_observation_sharing_mode():
    cfg0 = StructuralConfig(n_agents=8, n_steps=10, sharing_mode="observation")
    cfg1 = dataclasses.replace(cfg0, reliability_nu=4.0)
    ref = S.run_final(cfg0, S.init_state(cfg0, jax.random.PRNGKey(0)))
    out = S.run_final(cfg1, S.init_state(cfg1, jax.random.PRNGKey(0)))
    assert not np.allclose(np.asarray(out.h), np.asarray(ref.h), atol=1e-8)
