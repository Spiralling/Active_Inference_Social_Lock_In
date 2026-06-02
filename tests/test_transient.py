"""Tests for the transient-study additions (the suppression-vs-enablement plan).

Three things are exercised:

  * the **forgetting removal** is complete (the rejected mechanism is gone, and
    its absence is byte-identical to the old default ``forget_rho=1.0`` path);
  * the **time-varying-network rollout** (``run_trace_schedule`` /
    ``Network.run_bridge``) reduces to the static rollout on the two limits that
    pin it down -- a constant schedule equals ``run_trace``, and a bridge with
    ``t_incubate=0`` equals an always-coupled run;
  * the **convergence-time observables** (``settling_time`` / ``time_to_half``)
    return the right indices on hand-checked trajectories.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.config import NetworkConfig
from src.structural.phlogiston import StructuralConfig
from src.structural import step as S
from src.structural import observables as obs
from src.structural.kernel import Network


KEY = jax.random.PRNGKey(0)


# ----------------------------------------------------------------------
# Forgetting removal.
# ----------------------------------------------------------------------

def test_forgetting_fields_removed():
    """The rejected forgetting knobs are gone from the config surface."""
    fields = {f.name for f in dataclasses.fields(StructuralConfig)}
    assert "forget_rho" not in fields
    assert "forget_kappa" not in fields


# ----------------------------------------------------------------------
# Time-varying-network rollout: the two equivalence limits.
# ----------------------------------------------------------------------

def test_schedule_constant_equals_run_trace():
    """A constant per-step schedule (``W_seq[t] == state.W``) must reproduce the
    static ``run_trace`` exactly -- the additive code path carries no drift."""
    cfg = StructuralConfig(n_agents=12, n_steps=20, precision_mode="derived",
                           core_governance=30.0)
    state = S.init_state(cfg, KEY)
    ms_ref, gw_ref = S.run_trace(cfg, state)

    W_seq = jnp.broadcast_to(state.W, (cfg.n_steps,) + state.W.shape)
    ms, gw = S.run_trace_schedule(cfg, state, W_seq)

    assert np.allclose(np.asarray(ms), np.asarray(ms_ref), atol=1e-6)
    assert np.allclose(np.asarray(gw), np.asarray(gw_ref), atol=1e-6)


def test_index_schedule_constant_equals_run_trace_index():
    cfg = StructuralConfig(n_agents=10, n_steps=16, precision_mode="heuristic",
                           experiment_bias=0.4)
    state = S.init_state(cfg, KEY)
    ox_ref = S.run_trace_index(cfg, state)

    W_seq = jnp.broadcast_to(state.W, (cfg.n_steps,) + state.W.shape)
    ox = S.run_trace_index_schedule(cfg, state, W_seq)

    assert np.allclose(np.asarray(ox), np.asarray(ox_ref), atol=1e-6)


def _two_society_groups(n_each: int = 8) -> list[dict]:
    """A mainstream phlogiston community + an oxygen (challenger) community."""
    return [{"count": n_each, "paradigm": "phlogiston", "stance": -1.0},
            {"count": n_each, "paradigm": "oxygen"}]


def test_bridge_t0_equals_always_coupled():
    """``run_bridge(t_incubate=0, inter_prob=p)`` on a split (disconnected) init
    must equal a plain ``run_trace`` on a network wired coupled at the same
    ``inter_prob`` -- the bridge opens immediately, so there is no incubation and
    the two trust graphs (rebuilt with the same seed/intra_prob) coincide."""
    groups = _two_society_groups(8)
    inter = 0.3
    base_net = NetworkConfig(kind="planted_sbm", intra_prob=0.4, inter_prob=0.0)

    split = StructuralConfig(n_agents=16, n_steps=18, network=base_net)
    coupled = dataclasses.replace(
        split, network=dataclasses.replace(base_net, inter_prob=inter))

    ms_ref, gw_ref = Network.init(coupled, KEY, groups).run_trace()
    ms, gw = Network.init(split, KEY, groups).run_bridge(0, inter, groups)

    assert np.allclose(ms, ms_ref, atol=1e-6)
    assert np.allclose(gw, gw_ref, atol=1e-6)


def test_bridge_incubation_delays_contact():
    """Sanity: with a non-zero incubation window the disconnected communities only
    meet at ``t_incubate``, so the pre-bridge population trajectory matches the
    fully-isolated run up to that step, then diverges once the bridge opens."""
    groups = _two_society_groups(8)
    net_cfg = NetworkConfig(kind="planted_sbm", intra_prob=0.4, inter_prob=0.0)
    cfg = StructuralConfig(n_agents=16, n_steps=24, network=net_cfg)

    net = Network.init(cfg, KEY, groups)
    t_inc = 10
    ms_split, _ = net.run_trace()                       # never bridged (inter=0)
    ms_bridge, _ = net.run_bridge(t_inc, 0.3, groups)

    # identical while both are still split, then the bridge run departs.
    assert np.allclose(ms_bridge[:t_inc], ms_split[:t_inc], atol=1e-6)
    assert not np.allclose(ms_bridge[-1], ms_split[-1], atol=1e-3)


def test_bridge_index_shape_and_t0():
    groups = _two_society_groups(8)
    cfg = StructuralConfig(
        n_agents=16, n_steps=14,
        network=NetworkConfig(kind="planted_sbm", intra_prob=0.4, inter_prob=0.0))
    ox = Network.init(cfg, KEY, groups).run_bridge_index(0, 0.3, groups)
    assert ox.shape == (cfg.n_steps, cfg.n_agents)
    assert np.all((ox >= 0.0) & (ox <= 1.0))


# ----------------------------------------------------------------------
# Convergence-time observables on hand-checked trajectories.
# ----------------------------------------------------------------------

def test_settling_time_basic():
    # within tol of asymptote=1 from index 3 onward, for k=3 consecutive steps.
    m = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert obs.settling_time(m, asymptote=1.0, tol=0.05, k=3) == 3


def test_settling_time_default_asymptote_constant():
    m = np.full(10, 0.5)
    assert obs.settling_time(m, tol=0.01, k=4) == 0


def test_settling_time_never_settles_returns_horizon():
    m = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])     # asymptote=m[-1]=1
    assert obs.settling_time(m, tol=0.05, k=2) == len(m)


def test_time_to_half_crossing():
    m = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    assert obs.time_to_half(m) == 3


def test_time_to_half_never_reaches():
    m = np.array([0.0, 0.1, 0.2, 0.3])
    assert obs.time_to_half(m) == len(m)


def test_time_to_half_immediate():
    m = np.array([0.5, 0.6, 0.7])
    assert obs.time_to_half(m) == 0


def test_suppression_longer_than_enablement():
    """The science check in miniature: a strongly-biased, self-sealing dense field
    (Model A) holds the wrong paradigm longer than a split that lets the challenger
    incubate then bridge (Model B). Both converge (monostable), A is slower."""
    groups = _two_society_groups(12)            # challenger = its own community
    # Model A: one dense homogeneous field, strong incumbent bias, challenger
    # scattered in as a minority; self-sealing on.
    a_groups = [{"count": 18, "paradigm": "phlogiston", "stance": -1.0,
                 "prec_scale": 3.0},
                {"count": 6, "paradigm": "oxygen"}]
    cfg_a = StructuralConfig(
        n_agents=24, n_steps=120, precision_mode="derived", core_governance=100.0,
        network=NetworkConfig(kind="watts_strogatz", mean_degree=6))
    ms_a, _ = Network.init(cfg_a, KEY, a_groups).run_trace()

    # Model B: challenger as its own community, bridged after incubation.
    cfg_b = StructuralConfig(
        n_agents=24, n_steps=120, precision_mode="derived", core_governance=100.0,
        network=NetworkConfig(kind="planted_sbm", intra_prob=0.5, inter_prob=0.0))
    ms_b, _ = Network.init(cfg_b, KEY, groups).run_bridge(30, 0.3, groups)

    st_a = obs.settling_time(ms_a, tol=0.05, k=5)
    st_b = obs.settling_time(ms_b, tol=0.05, k=5)
    # both reach a stable band within the horizon...
    assert st_a < len(ms_a) and st_b < len(ms_b)
    # ...and the suppressed dense field settles no sooner than the split.
    assert st_a >= st_b
