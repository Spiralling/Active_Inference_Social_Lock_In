"""Tests for the represented-rival layer (q(phi, m): mixtures over candidate wirings).

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/rival.py
(log_predictive, the prequential identity, model_posterior/pool_log_weights,
fuse_within_frame, anchored_forget, gaussian_kl, m_gated_weights, run_rival).

The load-bearing checks: the per-observation score is the exact Gaussian predictive
density (vs a hand-rolled reference); its running sum telescopes to the closed-form
marginal likelihood (the prequential identity against ``linalg.log_evidence``); K = 1
reduces to the plain engine; and the Demo-A acid miniature -- the Gaussian pool's
blurred wiring is FROZEN while the rival agent's model posterior locks onto the truth.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.structural import rival as rv
from src.structural import linalg
from src.structural.step import fuse
from src.structural.bayesnet import LinearGaussianBN


def _pd(d, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    return jnp.asarray(A @ A.T + d * scale * np.eye(d), dtype=jnp.float32)


def _demo_nets():
    """M1: direct A->C (0.8); M2: chain A->B->C (0.7, 0.7). Zero-mean, PD."""
    names = ("a", "b", "c")
    B1 = jnp.zeros((3, 3)).at[2, 0].set(0.8)
    B2 = jnp.zeros((3, 3)).at[1, 0].set(0.7).at[2, 1].set(0.7)
    z, s = jnp.zeros(3), jnp.ones(3)
    n1 = LinearGaussianBN(B=B1, b=z, s=s, names=names).to_info()
    n2 = LinearGaussianBN(B=B2, b=z, s=s, names=names).to_info()
    return n1, n2


# ----------------------------------------------------------------------
# 1. log_predictive == the Gaussian predictive density (hand-rolled reference).
# ----------------------------------------------------------------------

def test_log_predictive_matches_reference():
    d, m = 4, 3
    Pi = _pd(d, 1)
    h = jnp.asarray(np.random.default_rng(2).normal(size=d), dtype=jnp.float32)
    H = jnp.asarray(np.random.default_rng(3).normal(size=(m, d)), dtype=jnp.float32)
    o = jnp.asarray([0.3, -1.2, 0.5])
    sigma = 0.7
    got = float(rv.log_predictive(Pi, h, H, o, sigma))
    # reference: dense numpy
    Pi_n, H_n = np.asarray(Pi, float), np.asarray(H, float)
    mu = np.linalg.solve(Pi_n, np.asarray(h, float))
    C = sigma ** 2 * np.eye(m) + H_n @ np.linalg.inv(Pi_n) @ H_n.T
    r = np.asarray(o, float) - H_n @ mu
    ref = -0.5 * (m * np.log(2 * np.pi) + np.linalg.slogdet(C)[1]
                  + r @ np.linalg.solve(C, r))
    assert got == pytest.approx(ref, rel=1e-4)


# ----------------------------------------------------------------------
# 2. The prequential identity: running score telescopes to the model evidence.
# ----------------------------------------------------------------------

def test_prequential_telescopes_to_log_evidence():
    """sum_t log_predictive(Pi_t, h_t, o_t) ==
    [logZ(Pi_T, h_T) - logZ(Pi_0, h_0)] - sum|o|^2/2s^2 - (Tm/2)log(2 pi s^2)."""
    d, m, T, sigma = 3, 2, 4, 0.6
    rng = np.random.default_rng(7)
    Pi = _pd(d, 5)
    h = jnp.asarray(rng.normal(size=d), dtype=jnp.float32)
    H = jnp.asarray(rng.normal(size=(m, d)), dtype=jnp.float32)
    Pi0, h0 = Pi, h
    J = H.T @ H / sigma ** 2
    total, sq = 0.0, 0.0
    for t in range(T):
        o = jnp.asarray(rng.normal(size=m), dtype=jnp.float32)
        total += float(rv.log_predictive(Pi, h, H, o, sigma))
        sq += float(o @ o)
        Pi = Pi + J
        h = h + H.T @ o / sigma ** 2
    closed = (float(linalg.log_evidence(Pi, h)) - float(linalg.log_evidence(Pi0, h0))
              - sq / (2 * sigma ** 2) - (T * m / 2) * np.log(2 * np.pi * sigma ** 2))
    assert total == pytest.approx(closed, rel=1e-3, abs=1e-2)


# ----------------------------------------------------------------------
# 3. Model-posterior numerics.
# ----------------------------------------------------------------------

def test_model_posterior_stable_and_shift_invariant():
    L = jnp.asarray([[1e4, 1e4 - 2.0], [-1e4, -1e4 + 1.0]])
    q = np.asarray(rv.model_posterior(L))
    assert np.isfinite(q).all() and np.allclose(q.sum(axis=1), 1.0)
    q_shift = np.asarray(rv.model_posterior(L + 123.4))
    assert np.allclose(q, q_shift, atol=1e-6)


# ----------------------------------------------------------------------
# 4. K = 1 hypothesis-aligned fusion == step.fuse.
# ----------------------------------------------------------------------

def test_fuse_within_frame_k1_equals_step_fuse():
    N, d = 4, 3
    Pi = jnp.stack([_pd(d, s) for s in range(N)])[:, None]          # (N,1,d,d)
    h = jnp.asarray(np.random.default_rng(0).normal(size=(N, 1, d)),
                    dtype=jnp.float32)
    W = jnp.asarray(np.full((N, N), 1.0 / N), dtype=jnp.float32)
    Pi_f, h_f = rv.fuse_within_frame(Pi, h, W)
    Pi_ref, h_ref = fuse(Pi[:, 0], h[:, 0], W)
    assert np.allclose(Pi_f[:, 0], Pi_ref, atol=1e-5)
    assert np.allclose(h_f[:, 0], h_ref, atol=1e-5)


# ----------------------------------------------------------------------
# 5. Identical candidates stay exactly tied.
# ----------------------------------------------------------------------

def test_identical_candidates_stay_uniform():
    net, _ = _demo_nets()
    cfg = rv.RivalConfig(Pi0=jnp.stack([net.Pi, net.Pi]),
                         h0=jnp.stack([net.h, net.h]),
                         H=jnp.eye(3), sigma_o=0.5, omega=0.9, alpha_m=0.05)
    W = jnp.ones((2, 2)) / 2.0
    A = jnp.ones((2, 2))
    r = rv.run_rival(cfg, W, A, jnp.ones((2, 3)), jnp.zeros((10, 3)), seed=0)
    assert np.abs(r["L_t"][:, :, 0] - r["L_t"][:, :, 1]).max() < 1e-4
    assert np.allclose(r["q_t"][-1], 0.5, atol=1e-4)


# ----------------------------------------------------------------------
# 6. Gaussian KL properties.
# ----------------------------------------------------------------------

def test_gaussian_kl_properties():
    Pi1, Pi2 = _pd(3, 1), _pd(3, 2)
    h1 = jnp.asarray([0.5, -0.2, 1.0])
    h2 = jnp.asarray([-0.3, 0.8, 0.1])
    assert float(rv.gaussian_kl(Pi1, h1, Pi1, h1)) == pytest.approx(0.0, abs=1e-4)
    assert float(rv.gaussian_kl(Pi1, h1, Pi2, h2)) > 0.0
    # 1-d closed form: KL = 0.5 (s1/s2 + (m1-m2)^2/s2 - 1 + ln(s2/s1)), s = variance
    p1, p2, q1, q2 = 2.0, 0.5, 1.0, -0.4                  # precisions, potentials
    m1, m2, s1, s2 = q1 / p1, q2 / p2, 1 / p1, 1 / p2
    ref = 0.5 * (s1 / s2 + (m1 - m2) ** 2 / s2 - 1 + np.log(s2 / s1))
    got = float(rv.gaussian_kl(jnp.asarray([[p1]]), jnp.asarray([q1]),
                               jnp.asarray([[p2]]), jnp.asarray([q2])))
    assert got == pytest.approx(ref, rel=1e-4)


# ----------------------------------------------------------------------
# 7. Engine smoke on the cosmology menu.
# ----------------------------------------------------------------------

def test_engine_smoke_cosmology_menu():
    from src.structural.scenarios import cosmology_presets
    nets = [p.belief_net for p in cosmology_presets()]
    cfg = rv.RivalConfig(Pi0=jnp.stack([n.Pi for n in nets]),
                         h0=jnp.stack([n.h for n in nets]),
                         H=jnp.eye(6), sigma_o=0.5, omega=0.9, alpha_m=0.02)
    N = 4
    W = jnp.ones((N, N)) / N
    A = jnp.ones((N, N))
    phis = jnp.zeros((20, 6))
    r = rv.run_rival(cfg, W, A, jnp.ones((N, 6)), phis, seed=1)
    assert r["L_t"].shape == (20, N, 3) and r["mu_eff_t"].shape == (20, N, 6)
    assert np.isfinite(r["L_t"]).all() and np.isfinite(r["mu_eff_t"]).all()
    assert np.allclose(r["q_t"].sum(axis=-1), 1.0, atol=1e-5)
    assert all(np.linalg.eigvalsh(P).min() > 0 for P in r["Pi"][:, 0])


# ----------------------------------------------------------------------
# 8. Anchored forgetting.
# ----------------------------------------------------------------------

def test_anchored_forget_noop_and_convergence():
    net1, net2 = _demo_nets()
    Pi0 = jnp.stack([net1.Pi, net2.Pi])
    h0 = jnp.stack([net1.h, net2.h])
    Pi = Pi0[None] + 3.0
    h = h0[None] + 1.0
    P1, q1 = rv.anchored_forget(Pi, h, Pi0, h0, 1.0)
    assert np.allclose(P1, Pi) and np.allclose(q1, h)
    P, q = Pi, h
    for _ in range(200):
        P, q = rv.anchored_forget(P, q, Pi0, h0, 0.9)
    assert np.allclose(P, Pi0[None], atol=1e-5) and np.allclose(q, h0[None], atol=1e-5)


# ----------------------------------------------------------------------
# 9. The m-gate: row-stochastic, support-respecting.
# ----------------------------------------------------------------------

def test_m_gated_weights_rows_and_support():
    L = jnp.asarray([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0]])   # two camps in m-space
    A_self = jnp.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    Wm = np.asarray(rv.m_gated_weights(L, A_self, nu_m=1.0))
    assert np.allclose(Wm.sum(axis=1), 1.0, atol=1e-5)
    assert Wm[0, 2] == 0.0 and Wm[2, 0] == 0.0              # support respected
    assert Wm[1, 0] > Wm[1, 2]                              # same-camp trusted more


# ----------------------------------------------------------------------
# 10. Demo-A acid miniature: the pool's blur is frozen; the rival locks on.
# ----------------------------------------------------------------------

def test_demo_a_miniature_blur_vs_lock():
    net1, net2 = _demo_nets()
    rng = np.random.default_rng(0)
    Sigma1 = np.linalg.inv(np.asarray(net1.Pi, float))
    chol = np.linalg.cholesky(Sigma1)
    phis = jnp.asarray((chol @ rng.normal(size=(3, 150))).T, dtype=jnp.float32)
    W = jnp.ones((1, 1))
    A = jnp.ones((1, 1))
    w = jnp.ones((1, 3))

    # rival agent: q(M1) must lock on (the score reads the corr structure).
    # omega SMALL is load-bearing: heavy within-frame parameter memory lets BOTH
    # wirings explain the data (accumulated diagonal deposits swamp the structural
    # part of the predictive covariance) -- short memory keeps structure visible.
    cfg = rv.RivalConfig(Pi0=jnp.stack([net1.Pi, net2.Pi]),
                         h0=jnp.stack([net1.h, net2.h]),
                         H=jnp.eye(3), sigma_o=0.5, omega=0.3, alpha_m=0.0)
    r = rv.run_rival(cfg, W, A, w, phis, seed=3)
    assert r["q_t"][-1, 0, 0] > 0.9, r["q_t"][-1]

    # Gaussian pool baseline (K=1, the averaged prior): H = I deposits are diagonal,
    # so the blurred off-diagonals are FROZEN at exactly half strength forever
    Pi_pool = 0.5 * (net1.Pi + net2.Pi)
    h_pool = 0.5 * (net1.h + net2.h)
    cfg_b = rv.RivalConfig(Pi0=Pi_pool[None], h0=h_pool[None],
                           H=jnp.eye(3), sigma_o=0.5, omega=1.0)
    rb = rv.run_rival(cfg_b, W, A, w, phis, seed=3)
    off = ~np.eye(3, dtype=bool)
    assert np.allclose(rb["Pi"][0, 0][off], np.asarray(Pi_pool)[off], atol=1e-4)
