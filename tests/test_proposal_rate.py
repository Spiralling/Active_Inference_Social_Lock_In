"""Tests for the principled proposal rate (Hawkes social excitation + adaptive hyperprior).

Layer: structural · Guards: src/structural/proposal_rate.py, and the rate_cfg plumbing in
src/structural/models/kuhn_phlogiston.py (the legacy Poisson path must stay byte-identical).

The recursive O(N) kernel must equal the appendix equation written out as an explicit sum
over discovery events; beta = 0 must collapse to the constant base rate; the adaptive belief
must rise monotonically with a persistent residual floor and decay when the model goes quiet.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.structural.proposal_rate import ProposalRateState, RateConfig, explicit_rates


def _ring_trust(n: int) -> np.ndarray:
    """Row-stochastic trust over a ring + self-loop (a small stand-in for graph.trust_W())."""
    A = np.eye(n)
    for i in range(n):
        A[i, (i - 1) % n] = A[i, (i + 1) % n] = 1.0
    return A / A.sum(axis=1, keepdims=True)


def test_recursive_kernel_matches_explicit_sum():
    n, T = 6, 40
    cfg = RateConfig(mode="hawkes", r0=0.02, hawkes_beta=0.5, hawkes_tau=7.0, r_max=10.0)
    W = _ring_trust(n)
    state = ProposalRateState(cfg, n, W)
    events = [(0, 3), (2, 3), (4, 10), (0, 25)]          # staggered discoveries
    by_step = {}
    for (i, t) in events:
        by_step.setdefault(t, []).append(i)
    for t in range(T):
        r = state.step(t)
        r_ref = explicit_rates(cfg, n, W, [(i, tj) for (i, tj) in events if tj < t], t)
        np.testing.assert_allclose(r, r_ref, rtol=1e-12, atol=1e-12)
        for i in by_step.get(t, []):
            state.record_discovery(i, t)


def test_beta_zero_is_constant_base_rate():
    n = 5
    cfg = RateConfig(mode="hawkes", r0=0.08, hawkes_beta=0.0)
    state = ProposalRateState(cfg, n, _ring_trust(n))
    state.record_discovery(2, 0)                          # a bump of size 0
    for t in range(1, 20):
        np.testing.assert_allclose(state.step(t), np.full(n, 0.08))


def test_excitation_decays_back_to_base():
    n = 4
    cfg = RateConfig(mode="hawkes", r0=0.01, hawkes_beta=1.0, hawkes_tau=3.0, r_max=10.0)
    state = ProposalRateState(cfg, n, _ring_trust(n))
    state.record_discovery(0, 0)
    r1 = state.step(1)
    assert r1.max() > cfg.r0                              # neighbours are excited...
    for t in range(2, 80):
        r = state.step(t)
    np.testing.assert_allclose(r, np.full(n, cfg.r0), atol=1e-6)   # ...then it fades


def test_trust_weights_localize_the_excitation():
    """Only trusted neighbours (and the discoverer itself) feel the bump."""
    n = 8
    cfg = RateConfig(mode="hawkes", r0=0.0, hawkes_beta=1.0, hawkes_tau=5.0, r_max=10.0)
    state = ProposalRateState(cfg, n, _ring_trust(n))
    state.record_discovery(0, 0)
    r = state.step(1)
    touched = {n - 1, 0, 1}                               # ring neighbourhood of agent 0
    for i in range(n):
        assert (r[i] > 0) == (i in touched)


def test_adaptive_rate_tracks_the_floor_and_clips():
    n = 3
    cfg = RateConfig(mode="adaptive", r0=0.02, kappa=0.5, floor_ema=0.3,
                     floor_ref=0.1, r_max=0.4)
    state = ProposalRateState(cfg, n)
    floors = np.array([0.0, 0.5, 50.0])                   # quiet / loaded / huge
    r_prev = state.step(0, floors)
    for t in range(1, 30):
        r = state.step(t, floors)
        assert (r >= r_prev - 1e-12).all()                # persistent floor: monotone rise
        r_prev = r
    assert r[0] == pytest.approx(cfg.r0)                  # quiet model stays at base
    assert r[1] > cfg.r0
    assert r[2] == pytest.approx(cfg.r_max)               # the clip holds
    for t in range(30, 200):                              # the model goes quiet -> decay
        r = state.step(t, np.zeros(n))
    np.testing.assert_allclose(r, np.full(n, cfg.r0), atol=1e-6)


def test_adaptive_reference_matches_recursion():
    n, T = 4, 25
    cfg = RateConfig(mode="hawkes_adaptive", r0=0.01, hawkes_beta=0.4, hawkes_tau=6.0,
                     kappa=0.8, floor_ema=0.2, floor_ref=0.05, r_max=5.0)
    W = _ring_trust(n)
    rng = np.random.default_rng(0)
    floors = rng.uniform(0.0, 0.6, size=(T, n))
    state = ProposalRateState(cfg, n, W)
    events = [(1, 5), (3, 12)]
    for t in range(T):
        r = state.step(t, floors[t])
        r_ref = explicit_rates(cfg, n, W, [(i, tj) for (i, tj) in events if tj < t], t,
                               floor_history=floors[: t + 1])
        np.testing.assert_allclose(r, r_ref, rtol=1e-10, atol=1e-12)
        for (i, tj) in events:
            if tj == t:
                state.record_discovery(i, t)


def test_hawkes_mode_requires_trust():
    with pytest.raises(ValueError):
        ProposalRateState(RateConfig(mode="hawkes"), 4, None)
    with pytest.raises(ValueError):
        RateConfig(mode="bogus")


def test_attempt_probs_are_one_minus_exp():
    state = ProposalRateState(RateConfig(mode="adaptive", kappa=0.0), 2)
    np.testing.assert_allclose(state.attempt_probs(np.array([0.0, 0.1])),
                               [0.0, 1.0 - np.exp(-0.1)])


def test_legacy_poisson_path_is_unchanged_by_the_new_kwargs():
    """single_run(rate_mode='poisson') must be bit-identical to the pre-change behaviour:
    the new kwargs default to a no-op and the gate stream is never consumed."""
    from src.structural.models.kuhn_phlogiston import single_run
    kw = dict(N=12, inter=0.0, n_steps=60, t_shift=20, proposal_rate=0.2,
              seed=0, snapshot_every=10)
    r_legacy = single_run(**kw)
    r_again = single_run(rate_mode="poisson", **kw)
    np.testing.assert_array_equal(r_legacy["expand_step"], r_again["expand_step"])
    np.testing.assert_array_equal(r_legacy["crisis_step"], r_again["crisis_step"])
    np.testing.assert_allclose(r_legacy["oxy_coupling_tn"], r_again["oxy_coupling_tn"])
    assert "rate_tn" not in r_legacy and "rate_tn" not in r_again


def test_hawkes_run_produces_rate_telemetry():
    from src.structural.models.kuhn_phlogiston import single_run
    r = single_run(N=12, inter=0.0, n_steps=60, t_shift=20, proposal_rate=0.05,
                   rate_mode="hawkes", hawkes_beta=0.5, hawkes_tau=8.0,
                   seed=0, snapshot_every=10)
    assert r["rate_tn"].shape == (60, 12)
    assert r["attempt_tn"].dtype == bool
    assert (r["rate_tn"] >= 0.05 - 1e-12).all()           # excitation only ever adds
