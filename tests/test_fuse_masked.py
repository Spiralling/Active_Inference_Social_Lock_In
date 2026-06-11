"""Tests for dimension-aware fusion (``fuse_mode="posterior_masked"``).

Layer: structural · Guards: src/structural/simulation.py (_fuse_masked + mask threading).

The masked mode must (i) reduce exactly to the ``posterior`` mode under an all-ones mask,
(ii) keep a pinned receiver's slot pinned (no concept import through averaging), (iii) never
drag an awake sender's slot toward a neighbour's pin (the per-entry renormalization), and
(iv) stay positive-definite along a staggered-wake trajectory (the plan's flagged numerical
risk -- per-entry convex mixes are not automatically PD, so it is CHECKED, not assumed).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.structural.simulation import _fuse_masked, _fuse_observe
from src.structural.step import trust_weights

PIN = 1.0e4


def _stack(rng, n, d, pinned: list[bool]):
    """A stack of PD nets; agents flagged pinned carry a decoupled PIN slot at dim d-1."""
    Pi = np.zeros((n, d, d))
    h = rng.normal(size=(n, d))
    for i in range(n):
        M = rng.normal(size=(d, d))
        Pi[i] = M @ M.T + d * np.eye(d)
        if pinned[i]:
            Pi[i, -1, :] = Pi[i, :, -1] = 0.0
            Pi[i, -1, -1] = PIN
            h[i, -1] = 0.0
    return jnp.asarray(Pi), jnp.asarray(h)


def _W(n):
    return trust_weights(jnp.ones((n, n)))          # complete graph + self


def test_all_ones_mask_reproduces_posterior_mode():
    rng = np.random.default_rng(0)
    n, d = 5, 6
    Pi, h = _stack(rng, n, d, [False] * n)
    J = jnp.zeros((n, d, d)); j = jnp.zeros((n, d))
    W = _W(n)
    Woff = jnp.zeros((n, n))
    Pi_m, h_m = _fuse_masked(Pi, h, J, j, W, jnp.ones((n, d)))
    Pi_p, h_p = _fuse_observe(Pi, h, J, j, W, Woff, 0.5, "posterior")
    np.testing.assert_allclose(np.asarray(Pi_m), np.asarray(Pi_p), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.asarray(h_m), np.asarray(h_p), rtol=1e-6, atol=1e-8)


def test_pinned_receiver_keeps_its_pin():
    """An agent that does not represent the slot keeps its own pinned entries exactly --
    a concept cannot arrive through averaging."""
    rng = np.random.default_rng(1)
    n, d = 4, 5
    pinned = [True, False, False, False]
    Pi, h = _stack(rng, n, d, pinned)
    mask = jnp.asarray(np.array([[1.0] * (d - 1) + [0.0 if p else 1.0] for p in pinned]))
    Pi_f, h_f = _fuse_masked(Pi, h, jnp.zeros((n, d, d)), jnp.zeros((n, d)), _W(n), mask)
    assert float(Pi_f[0, -1, -1]) == pytest.approx(PIN)
    np.testing.assert_allclose(np.asarray(Pi_f[0, -1, :-1]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(Pi_f[0, :-1, -1]), 0.0, atol=1e-12)
    assert float(h_f[0, -1]) == pytest.approx(0.0)


def test_awake_sender_not_dragged_by_pinned_neighbours():
    """The lone awake agent's slot entries survive fusion exactly (it is the only
    representing sender, so the renormalized weight on itself is 1) -- the crushing the
    plain posterior mode inflicts is gone."""
    rng = np.random.default_rng(2)
    n, d = 4, 5
    pinned = [False, True, True, True]                # agent 0 woke the slot
    Pi, h = _stack(rng, n, d, pinned)
    mask = jnp.asarray(np.array([[1.0] * (d - 1) + [0.0 if p else 1.0] for p in pinned]))
    W = _W(n)
    Pi_f, h_f = _fuse_masked(Pi, h, jnp.zeros((n, d, d)), jnp.zeros((n, d)), W, mask)
    np.testing.assert_allclose(np.asarray(Pi_f[0, -1, :]), np.asarray(Pi[0, -1, :]),
                               rtol=1e-6, atol=1e-10)
    assert float(h_f[0, -1]) == pytest.approx(float(h[0, -1]), rel=1e-6)
    # while the conceived block still fuses normally (everyone represents it)
    blk = jnp.einsum("ij,jab->iab", W, Pi)[0, :-1, :-1]
    np.testing.assert_allclose(np.asarray(Pi_f[0, :-1, :-1]), np.asarray(blk),
                               rtol=1e-6, atol=1e-8)
    # ...and the pinned agents' fused diag is NOT inflated by the awake agent's pin-free
    # slot: their own PIN is kept verbatim (receiver guard), no 1e4/N wash.
    for i in (1, 2, 3):
        assert float(Pi_f[i, -1, -1]) == pytest.approx(PIN)


def test_pd_along_a_staggered_wake_trajectory():
    """The flagged numerical risk: per-entry renormalization is not a single convex
    combination. Walk a staggered wake (agents waking one at a time, fusing every step,
    depositing noise Fisher) and require every fused matrix to stay PD throughout."""
    rng = np.random.default_rng(3)
    n, d = 6, 7
    pinned = [True] * n
    Pi, h = _stack(rng, n, d, pinned)
    W = _W(n)
    coupling = rng.normal(size=d - 1)
    coupling = 0.6 * coupling / np.linalg.norm(coupling)
    for step in range(30):
        if step % 4 == 0 and step // 4 < n:           # wake one more agent
            i = step // 4
            pinned[i] = False
            Pi_np = np.asarray(Pi).copy()
            Pi_np[i, -1, :-1] = coupling
            Pi_np[i, :-1, -1] = coupling
            Pi_np[i, -1, -1] = 2.0
            Pi = jnp.asarray(Pi_np)
        mask = jnp.asarray(np.array(
            [[1.0] * (d - 1) + [0.0 if p else 1.0] for p in pinned]))
        Jd = np.zeros((n, d, d))
        Jd[:, : d - 1, : d - 1] = 0.05 * np.eye(d - 1)[None]
        Pi, h = _fuse_masked(Pi, h, jnp.asarray(Jd), jnp.zeros((n, d)), W, mask)
        evs = np.linalg.eigvalsh(np.asarray(Pi))
        assert (evs > 0).all(), f"lost PD at step {step}: min eig {evs.min():.3e}"


def test_single_run_masked_mode_runs_and_survives_staggered():
    """End-to-end: under posterior_masked the staggered (Poisson) discovery is no longer
    crushed -- lone discoverers keep an effective oxygen coupling."""
    from src.structural.models.kuhn_phlogiston import single_run
    kw = dict(N=16, inter=0.0, n_steps=120, t_shift=30, proposal_rate=0.05,
              snapshot_every=20, seed=3)
    r_plain = single_run(fuse_mode="posterior", **kw)
    r_mask = single_run(fuse_mode="posterior_masked", **kw)
    o = r_mask["community"] == 0
    es_m = r_mask["expand_step"][o]
    if not (es_m >= 0).any():                          # tiny run: no discovery this seed
        pytest.skip("no discovery at this seed/size")
    surv_mask = (r_mask["oxy_coupling_tn"][-1][o] > 0.02).mean()
    surv_plain = (r_plain["oxy_coupling_tn"][-1][r_plain["community"] == 0] > 0.02).mean()
    assert surv_mask > surv_plain, (surv_mask, surv_plain)
    assert np.isfinite(r_mask["oxy_coupling_tn"]).all()
