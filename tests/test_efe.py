"""Hand-checked tests for the EFE attention functional (src/structural/efe.py).

Pencil-and-paper asserts on tiny populations. The two contracts that matter:
  - the two prediction-error drivers compute the closed forms in the docstring;
  - ``efe_channel_precision`` reduces to its pragmatic baseline at zero epistemic/social
    weight, and the *social* term re-opens a channel the baseline suppressed when a trusted
    neighbour disagrees (the self-sealing-breaker).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from src.structural import efe


def test_conditional_precision_reads_diagonal():
    Pi = jnp.array([[2.0, 0.5], [0.5, 4.0]])
    cp = efe.conditional_precision(Pi, jnp.array([0, 1]))
    assert np.allclose(np.asarray(cp), [2.0, 4.0])


def test_predictive_variance_is_inverse_diagonal():
    Pi = jnp.array([[2.0, 0.5], [0.5, 4.0]])
    pv = efe.predictive_variance(Pi, jnp.array([0, 1]))
    assert np.allclose(np.asarray(pv), [0.5, 0.25])


def test_predictive_variance_stacked():
    Pi = jnp.stack([jnp.array([[2.0, 0.0], [0.0, 1.0]]),
                    jnp.array([[5.0, 0.0], [0.0, 4.0]])])      # (2,2,2)
    pv = efe.predictive_variance(Pi, jnp.array([0, 1]))
    assert pv.shape == (2, 2)
    assert np.allclose(np.asarray(pv), [[0.5, 1.0], [0.2, 0.25]])


def test_social_surprisal_by_hand():
    # 2 agents, measure node 0. Agent 0 believes mu0[0]=0, agent 1 believes mu1[0]=1.
    mu = jnp.array([[0.0, 9.0], [1.0, 9.0]])                   # (2,2)
    Pi = jnp.stack([jnp.array([[3.0, 0.0], [0.0, 1.0]]),
                    jnp.array([[7.0, 0.0], [0.0, 1.0]])])      # (2,2,2); Pi_j[0,0] = 3, 7
    W = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    pe = efe.social_surprisal(mu, Pi, W, jnp.array([0]))       # (2,1)
    # PE[0,0] = W00*Pi0[0,0]*(0-0)^2 + W01*Pi1[0,0]*(1-0)^2 = 0.5*7*1 = 3.5
    # PE[1,0] = W10*Pi0[0,0]*(0-1)^2 + W11*Pi1[0,0]*(1-1)^2 = 0.5*3*1 = 1.5
    assert np.allclose(np.asarray(pe[:, 0]), [3.5, 1.5])


def test_social_surprisal_zero_when_agreeing():
    mu = jnp.array([[0.5, 0.0], [0.5, 0.0]])                   # identical beliefs
    Pi = jnp.stack([jnp.eye(2), jnp.eye(2)])
    W = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    pe = efe.social_surprisal(mu, Pi, W, jnp.array([0]))
    assert np.allclose(np.asarray(pe), 0.0)


def test_efe_reduces_to_base_rho_at_zero_weight():
    Pi = jnp.stack([jnp.array([[2.0, 0.3], [0.3, 1.5]]) for _ in range(3)])
    h = jnp.zeros((3, 2))
    W = jnp.full((3, 3), 1 / 3)
    base = jnp.array([0.4, 0.9])                               # <= rho_max
    rho = efe.efe_channel_precision(Pi, h, W, base, jnp.array([0, 1]),
                                    epistemic_weight=0.0, social_weight=0.0, rho_max=1.0)
    assert rho.shape == (3, 2)
    assert np.allclose(np.asarray(rho), np.broadcast_to(np.asarray(base), (3, 2)))


def test_social_term_reopens_suppressed_channel():
    # Channel 0 is fully suppressed by the paradigm (base_rho[0] ~ 0). A confident, trusted
    # neighbour who disagrees on node 0 should drive rho[*,0] back up above the baseline.
    mu = jnp.array([[0.0, 0.0], [2.0, 0.0]])                   # agent 1 disagrees on node 0
    Pi = jnp.stack([jnp.array([[1.0, 0.0], [0.0, 1.0]]),
                    jnp.array([[8.0, 0.0], [0.0, 1.0]])])      # neighbour 1 is confident
    h = jnp.einsum('nij,nj->ni', Pi, mu)                       # h = Pi @ mu so solve recovers mu
    W = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    base = jnp.array([0.01, 0.01])
    rho_off = efe.efe_channel_precision(Pi, h, W, base, jnp.array([0, 1]),
                                        epistemic_weight=0.0, social_weight=0.0)
    rho_on = efe.efe_channel_precision(Pi, h, W, base, jnp.array([0, 1]),
                                       epistemic_weight=0.0, social_weight=1.0)
    # social term off: stays at the suppressed baseline.
    assert np.allclose(np.asarray(rho_off[:, 0]), 0.01)
    # social term on: agent 0 (whose confident neighbour disagrees on node 0) re-opens channel 0.
    assert float(rho_on[0, 0]) > 0.3
    # node 1 (no disagreement) stays suppressed even with the social term on.
    assert float(rho_on[0, 1]) < 0.05


def test_efe_zero_weights_equals_derived_in_dynamics():
    """The back-compat anchor at the dynamics level: a full rollout in efe mode with
    both drives off is byte-identical to derived mode."""
    import dataclasses
    import jax
    from src.structural.phlogiston import StructuralConfig
    from src.structural import step as S

    key = jax.random.PRNGKey(0)
    cfg_d = StructuralConfig(n_agents=12, n_steps=20, precision_mode="derived",
                             core_governance=40.0)
    cfg_e = dataclasses.replace(cfg_d, precision_mode="efe",
                                epistemic_weight=0.0, social_weight=0.0)
    fd = S.run_final(cfg_d, S.init_state(cfg_d, key))
    fe = S.run_final(cfg_e, S.init_state(cfg_e, key))
    assert jnp.allclose(fd.Pi, fe.Pi)
    assert jnp.allclose(fd.h, fe.h)


def test_efe_clips_at_rho_max():
    mu = jnp.array([[0.0, 0.0], [50.0, 0.0]])                  # huge disagreement
    Pi = jnp.stack([jnp.eye(2), 10.0 * jnp.eye(2)])
    h = jnp.einsum('nij,nj->ni', Pi, mu)
    W = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    base = jnp.array([0.5, 0.5])
    rho = efe.efe_channel_precision(Pi, h, W, base, jnp.array([0, 1]),
                                    epistemic_weight=1.0, social_weight=1.0, rho_max=1.0)
    assert np.all(np.asarray(rho) <= 1.0 + 1e-6)
