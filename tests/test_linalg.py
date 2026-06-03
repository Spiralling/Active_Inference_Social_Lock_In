"""Hand-checked tests for the reusable LA kernel (src/structural/linalg.py).

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/linalg.py ·
Map: tests/README.md

Pure-array, pencil-and-paper asserts on tiny nets -- these pin the *numbers*, not
just self-consistency, so they double as a readable spec of the information-form
identities. (The belief/bmr wrappers are covered separately in test_structural.py;
here we exercise linalg directly.)
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from src.structural import linalg as la


# ----------------------------------------------------------------------
# log-evidence, mean, cov.
# ----------------------------------------------------------------------

def test_log_evidence_1d_by_hand():
    # d=1: Pi=[[2]], h=[1].
    #   logZ = 1/2 ( h^2/Pi - log Pi + log 2pi )
    #        = 1/2 ( 0.5 - log 2 + log 2pi ) = 1/2 ( 0.5 + log pi )
    Pi = jnp.array([[2.0]]); h = jnp.array([1.0])
    expected = 0.5 * (0.5 + np.log(np.pi))
    assert np.isclose(float(la.log_evidence(Pi, h)), expected, atol=1e-6)


def test_mean_and_cov_2x2():
    Pi = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    h = jnp.array([1.0, 0.0])
    Pi_np = np.array([[2.0, 0.5], [0.5, 1.0]])
    assert np.allclose(np.asarray(la.info_mean(Pi, h)),
                       np.linalg.solve(Pi_np, [1.0, 0.0]))
    assert np.allclose(np.asarray(la.info_cov(Pi)), np.linalg.inv(Pi_np))


# ----------------------------------------------------------------------
# learn / fuse = addition.
# ----------------------------------------------------------------------

def test_add_information_is_addition():
    Pi = jnp.eye(2); h = jnp.zeros(2)
    J = jnp.array([[1.0, 0.0], [0.0, 3.0]]); j = jnp.array([2.0, 5.0])
    Pi2, h2 = la.add_information(Pi, h, J, j)
    assert np.allclose(np.asarray(Pi2), [[2.0, 0.0], [0.0, 4.0]])
    assert np.allclose(np.asarray(h2), [2.0, 5.0])


def test_fuse_stack_row_stochastic_average():
    # two agents, identity beliefs; W averages them 50/50 -> still identity.
    Pi = jnp.stack([jnp.eye(2), 3.0 * jnp.eye(2)])           # (2,2,2)
    h = jnp.stack([jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])])
    W = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    Pi_f, h_f = la.fuse_stack(Pi, h, W)
    # each fused agent = 0.5*I + 0.5*3I = 2I ; h = 0.5*(1,0)+0.5*(0,1) = (0.5,0.5)
    assert np.allclose(np.asarray(Pi_f[0]), 2.0 * np.eye(2))
    assert np.allclose(np.asarray(h_f[0]), [0.5, 0.5])


def test_row_stochastic():
    raw = jnp.array([[1.0, 1.0, 0.0], [0.0, 2.0, 2.0]])
    W = la.row_stochastic(raw)
    assert np.allclose(np.asarray(W.sum(axis=1)), [1.0, 1.0])
    assert np.allclose(np.asarray(W[0]), [0.5, 0.5, 0.0])


# ----------------------------------------------------------------------
# fisher deposit.
# ----------------------------------------------------------------------

def test_fisher_deposit_by_hand():
    H = jnp.array([[1.0, 0.0], [0.0, 2.0]]); o = jnp.array([3.0, 1.0])
    J, j = la.fisher_deposit(H, o, sigma_o=1.0)
    # J = H^T H = diag(1,4); j = H^T o = (3, 2)
    assert np.allclose(np.asarray(J), [[1.0, 0.0], [0.0, 4.0]])
    assert np.allclose(np.asarray(j), [3.0, 2.0])


def test_fisher_deposit_weighted_zero_skips_channel():
    H = jnp.array([[1.0, 0.0], [0.0, 2.0]]); o = jnp.array([3.0, 1.0])
    # weight 0 on channel 1 -> only channel 0 contributes.
    J, j = la.fisher_deposit_weighted(H, o, 1.0, jnp.array([1.0, 0.0]))
    assert np.allclose(np.asarray(J), [[1.0, 0.0], [0.0, 0.0]])
    assert np.allclose(np.asarray(j), [3.0, 0.0])
    # all-ones recovers the plain deposit.
    J1, j1 = la.fisher_deposit_weighted(H, o, 1.0, jnp.ones(2))
    Jp, jp = la.fisher_deposit(H, o, 1.0)
    assert np.allclose(np.asarray(J1), np.asarray(Jp))


# ----------------------------------------------------------------------
# Schur complement / carry-over / conditioning.
# ----------------------------------------------------------------------

def test_schur_complement_2x2_by_hand():
    # Pi = [[a,b],[b,c]]; drop node 1 -> Pi_marg = a - b^2/c.
    a, b, c = 2.0, 0.5, 1.0
    Pi = jnp.array([[a, b], [b, c]]); h = jnp.array([1.0, 4.0])
    keep = jnp.array([0]); drop = jnp.array([1])
    Pi_m, h_m = la.schur_marginalize(Pi, h, keep, drop)
    assert np.isclose(float(Pi_m[0, 0]), a - b * b / c)
    # h_marg = h_a - Pi_ab Pi_bb^-1 h_b = 1 - 0.5*(1/1)*4 = -1
    assert np.isclose(float(h_m[0]), 1.0 - b / c * 4.0)


def test_carryover_fillin_is_the_subtracted_term():
    a, b, c = 2.0, 0.5, 1.0
    Pi = jnp.array([[a, b], [b, c]])
    keep = jnp.array([0]); drop = jnp.array([1])
    fill = la.carryover_fillin(Pi, keep, drop)
    assert np.isclose(float(fill[0, 0]), b * b / c)


def test_condition_restricts_block():
    Pi = jnp.array([[2.0, 0.5], [0.5, 1.0]]); h = jnp.array([1.0, 4.0])
    Pi_c, h_c = la.condition(Pi, h, jnp.array([0]))
    assert np.allclose(np.asarray(Pi_c), [[2.0]])
    assert np.allclose(np.asarray(h_c), [1.0])


def test_border_then_schur_returns_original():
    # border a 1-node net with a new coordinate, then marginalize it out.
    Pi = jnp.array([[2.0]]); h = jnp.array([1.0])
    Pi_b, h_b = la.border(Pi, h, couplings=jnp.array([0.3]),
                          Pi_diag=1.5, h_new=0.0)
    assert Pi_b.shape == (2, 2)
    Pi_m, h_m = la.schur_marginalize(Pi_b, h_b, jnp.array([0]), jnp.array([1]))
    # Schur leaves the fill-in 0.3^2/1.5 behind on the survivor.
    assert np.isclose(float(Pi_m[0, 0]), 2.0 - 0.3 ** 2 / 1.5)


# ----------------------------------------------------------------------
# BMR Savage-Dickey.
# ----------------------------------------------------------------------

def test_savage_dickey_matches_four_log_evidences():
    rng = np.random.default_rng(3)
    def spd(d):
        M = rng.standard_normal((d, d)); return M @ M.T + np.eye(d)
    Pi0 = jnp.asarray(spd(2)); h0 = jnp.asarray(rng.standard_normal(2))
    J = jnp.asarray(spd(2)); j = jnp.asarray(rng.standard_normal(2))
    Pi_post, h_post = Pi0 + J, h0 + j
    Pi0r = jnp.asarray(spd(2)); h0r = jnp.asarray(rng.standard_normal(2))

    dF, Pi_rp, h_rp = la.savage_dickey(Pi_post, h_post, Pi0, h0, Pi0r, h0r)
    ref = ((float(la.log_evidence(Pi_rp, h_rp)) - float(la.log_evidence(Pi0r, h0r)))
           - (float(la.log_evidence(Pi_post, h_post)) - float(la.log_evidence(Pi0, h0))))
    assert np.isclose(float(dF), ref, atol=1e-5)
    # the reduced posterior shares the likelihood deposit J,j.
    assert np.allclose(np.asarray(Pi_rp), np.asarray(Pi0r + J))


def test_spike_prior_pins_node():
    Pi = jnp.array([[2.0, 0.5], [0.5, 1.0]]); h = jnp.array([1.0, 1.0])
    Pi2, h2 = la.spike_prior(Pi, h, jnp.array([1]), value=2.0, spike=1e6)
    # node 1 decoupled, diagonal spiked, mean pinned to 2.
    assert np.isclose(float(Pi2[1, 1]), 1e6)
    assert np.isclose(float(Pi2[0, 1]), 0.0) and np.isclose(float(Pi2[1, 0]), 0.0)
    assert np.isclose(float(h2[1]) / float(Pi2[1, 1]), 2.0)
    # incumbent block untouched.
    assert np.isclose(float(Pi2[0, 0]), 2.0)
