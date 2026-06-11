"""Tests for the multi-candidate structure search (the candidate space, searched).

Layer: structural · Guards: src/structural/action.py (propose_hubs_topk, coupling_candidates,
coupling_score), src/structural/candidates.py (search_candidates), and the multi-slot padding
of models/kuhn_phlogiston.padded_scenario (the pin-cancellation regression).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.structural.action import (coupling_candidates, coupling_score, propose_hub,
                                   propose_hubs_topk)
from src.structural.belief import GaussianBeliefNet


def _net(d, seed=0, names=None):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(d, d))
    return GaussianBeliefNet(Pi=jnp.asarray(M @ M.T + d * np.eye(d)),
                             h=jnp.asarray(rng.normal(size=d)),
                             names=names or tuple(f"n{i}" for i in range(d)))


def test_topk_orders_by_strength_and_recovers_the_dominant_cause():
    """The top-k candidates come back in descending |eigenvalue| order with the dominant
    planted cause first (the diagonal-zeroing keeps only the coupling structure, so the
    SECOND cause competes with the dominant cause's rank-complement artifacts -- which is
    why discrimination is left to the ledger, see test_search_accepts_only_the_true_hub)."""
    k = 5
    v1 = np.array([1.0, -1.0, 1.0, 0.0, 0.0]) / np.sqrt(3)
    v2 = np.array([0.0, 0.0, 0.0, 1.0, -1.0]) / np.sqrt(2)
    R = 3.0 * np.outer(v1, v1) + 1.2 * np.outer(v2, v2)
    net = _net(k)
    props = propose_hubs_topk(net, net.names, residual=jnp.asarray(R), k=3)
    assert len(props) == 3
    assert props[0].strength >= props[1].strength >= props[2].strength
    assert abs(float(np.asarray(props[0].pattern) @ v1)) > 0.95
    # k=1 reproduces the legacy single proposal exactly
    legacy = propose_hub(net, net.names, residual=jnp.asarray(R))
    assert props[0].strength == pytest.approx(legacy.strength)
    np.testing.assert_allclose(np.asarray(props[0].pattern), np.asarray(legacy.pattern))


def test_coupling_candidates_rank_by_magnitude():
    nbrs = ("a", "b", "c", "d")
    R = np.zeros((4, 4))
    R[0, 1] = R[1, 0] = 0.5
    R[2, 3] = R[3, 2] = -2.0
    cands = coupling_candidates(jnp.asarray(R), nbrs, top_m=2)
    assert cands[0][:2] == ("c", "d") and cands[0][2] == pytest.approx(-2.0)
    assert cands[1][:2] == ("a", "b") and cands[1][2] == pytest.approx(0.5)


def test_coupling_score_accepts_planted_rejects_null():
    """An edge the data hold (the posterior mean genuinely co-displaced) must score
    dF > 0; a zero-residual edge must wire nothing and be declined."""
    d = 4
    names = tuple(f"n{i}" for i in range(d))
    Pi0 = jnp.asarray(3.0 * np.eye(d))
    prior = GaussianBeliefNet(Pi=Pi0, h=jnp.zeros(d), names=names)
    # deposit evidence that nodes 0,1 move together: J with strong (0,1) structure
    J = np.zeros((d, d)); J[0, 0] = J[1, 1] = 4.0; J[0, 1] = J[1, 0] = -3.6
    j = np.zeros(d); j[0] = j[1] = 6.0
    post = GaussianBeliefNet(Pi=Pi0 + jnp.asarray(J), h=jnp.asarray(j), names=names)
    ms = coupling_score(post, prior, ("n0", "n1"), magnitude=-0.8)
    assert ms.delta_F > 0 and ms.accept
    ms_null = coupling_score(post, prior, ("n2", "n3"), magnitude=0.0)
    assert not ms_null.accept and ms_null.detail["magnitude"] == 0.0
    # PD safety: a magnitude far past the bound is capped, never improper
    ms_big = coupling_score(post, prior, ("n0", "n1"), magnitude=-50.0)
    assert np.isfinite(ms_big.delta_F)


def test_multislot_padding_leaves_the_ledger_unchanged():
    """The pin-cancellation regression: extra pinned slots must leave the crisis machinery
    numerically unchanged (float32 roundoff only) and the discovery unchanged exactly."""
    from src.structural.models.kuhn_phlogiston import single_run
    kw = dict(N=12, inter=0.0, n_steps=80, t_shift=30, proposal_rate=None,
              snapshot_every=20, seed=0)
    r1 = single_run(n_slots=1, **kw)
    r3 = single_run(n_slots=3, **kw)
    np.testing.assert_array_equal(r1["crisis_step"], r3["crisis_step"])
    np.testing.assert_array_equal(r1["expand_step"], r3["expand_step"])
    d = np.abs(r1["dF_belt_tn"] - r3["dF_belt_tn"])
    assert np.nanmax(d) < 1e-3, f"pin cancellation violated: {np.nanmax(d)}"


def test_search_accepts_only_the_true_hub():
    """End-to-end: with 3 slots, 3 hub candidates and 3 coupling candidates per arrival,
    the oxygen direction must be the only accepted candidate; rivals and spurious couplings
    must score dF <= 0 on the same ledger."""
    from src.structural.models.kuhn_phlogiston import single_run
    r = single_run(N=12, inter=0.0, n_steps=80, t_shift=30, proposal_rate=None,
                   n_slots=3, k_hubs=3, top_couplings=3, snapshot_every=20, seed=0)
    log = r["candidate_log"]
    assert log, "candidates must be evaluated"
    accepted = {e["target"] for e in log if e["accepted"]}
    assert accepted == {"oxygen"}
    assert all(e["dF"] <= 0 for e in log if not e["accepted"])
    o = r["community"] == 0
    assert (r["expand_step"][o] >= 0).all()
