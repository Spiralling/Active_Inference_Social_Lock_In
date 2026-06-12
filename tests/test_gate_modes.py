"""Tests for the sensory gate modes (conviction vs cost) and earned trust (EMA memory).

Layer: structural (THE PAPER OBJECT) · Guards: reliability.gate_weights,
reliability.social_z2, reliability.trust_memory_update, dual_field.cost_field_live,
simulation.run_simulation(gate_mode=..., trust_memory=...).

Two hypotheses about lock-in, separated: the CONVICTION gate silences a channel because
the agent values what it threatens (wishful); the COST gate because revising what the
channel addresses would force the whole web to re-equilibrate (dogmatic). ``U = T u``
and ``kappa = T 1 - 1`` are independent fields, so the two gates must be able to
disagree about which channel to close. Earned trust: gamma read from an EMA'd z2 track
record instead of the instantaneous disagreement. Defaults must stay byte-identical.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.structural import graphs
from src.structural import reliability as rel
from src.structural.dual_field import PrecisionUtilityNet, cost_field_live
from src.structural.phlogiston import StructuralConfig
from src.structural.scenarios import phlogiston_scenario
from src.structural.simulation import run_simulation, AgentSpec


def _spec(scn, n, lam=0.2):
    return AgentSpec(w_obs=jnp.ones((n, scn.m)), lam=jnp.full((n,), float(lam)))


def _belt_nodes(scn):
    idx = {nm: i for i, nm in enumerate(scn.names)}
    return (idx["mass_change_sign"], idx["gas_consumed"],
            idx["calx_heavier_than_metal"])


# ----------------------------------------------------------------------
# 1. defaults: gate_mode/trust_memory off => byte-identical contracts.
# ----------------------------------------------------------------------

def test_gate_mode_default_byte_identical():
    scn = phlogiston_scenario(StructuralConfig(n_steps=12, t_shift=4))
    n = 5
    g = graphs.complete(n)
    spec = _spec(scn, n)
    kw = dict(forgetting=0.92, snapshot_every=4, seed=0,
              endogenous_gamma=True, gate_strength=2.0)
    r0 = run_simulation(scn, g, spec, **kw)
    r1 = run_simulation(scn, g, spec, **kw, gate_mode="conviction")
    assert np.array_equal(r0["snap_Pi"], r1["snap_Pi"])
    assert np.array_equal(r0["snap_h"], r1["snap_h"])
    assert np.array_equal(r0["gamma_t"], r1["gamma_t"])


def test_gate_mode_inert_when_gate_off():
    """With the sensory gate off, gate_mode must be dead code."""
    scn = phlogiston_scenario(StructuralConfig(n_steps=12, t_shift=4))
    n = 4
    g = graphs.complete(n)
    spec = _spec(scn, n)
    kw = dict(forgetting=0.92, snapshot_every=4, seed=0, endogenous_gamma=False)
    r0 = run_simulation(scn, g, spec, **kw)
    r1 = run_simulation(scn, g, spec, **kw, gate_mode="cost")
    assert np.array_equal(r0["snap_Pi"], r1["snap_Pi"])
    assert np.array_equal(r0["snap_h"], r1["snap_h"])


def test_gate_weights_conviction_matches_legacy_expression():
    """The 'conviction' branch must be op-for-op the legacy inline gate."""
    rng = np.random.default_rng(0)
    U = jnp.asarray(rng.normal(size=(6, 11)))
    H_disc = jnp.asarray(rng.normal(size=(3, 11)))
    for g, w_floor in [(1.0, 0.0), (3.0, 0.05)]:
        got = rel.gate_weights(U, None, H_disc, "conviction", g, w_floor)
        legacy = jnp.clip(jnp.exp(-g * jnp.abs(U @ H_disc.T)), w_floor, 1.0)
        assert np.array_equal(np.asarray(got), np.asarray(legacy))


def test_gate_weights_per_agent_strength():
    """gate_strength may be per-agent (N,): the cost projection is community-blind,
    so per-community gating asymmetry in cost mode enters through the strength."""
    rng = np.random.default_rng(2)
    kappa = jnp.asarray(rng.uniform(0.5, 2.0, size=(4, 6)))
    H_disc = jnp.asarray(rng.normal(size=(2, 6)))
    g = jnp.asarray([0.0, 0.5, 1.0, 5.0])
    w = np.asarray(rel.gate_weights(None, kappa, H_disc, "cost", g, 0.0))
    w_scalar = np.asarray(rel.gate_weights(None, kappa, H_disc, "cost", 1.0, 0.0))
    assert np.allclose(w[0], 1.0)                      # zero strength: no silencing
    assert np.array_equal(w[2], w_scalar[2])           # row at g=1 matches the scalar gate
    assert (w[3] <= w[1] + 1e-12).all()                # stronger gate silences more


def test_social_z2_matches_social_gamma():
    """social_z2 + student_t_weight must reproduce social_gamma exactly (the means-mode
    social block was restructured around the exposed z2 statistic)."""
    rng = np.random.default_rng(1)
    n, d = 5, 7
    A = rng.normal(size=(n, d, d))
    Pi = jnp.asarray(A @ np.transpose(A, (0, 2, 1)) + 3.0 * np.eye(d))
    h = jnp.asarray(rng.normal(size=(n, d)))
    ix = jnp.asarray([1, 4])
    nu = 1.5
    got = rel.student_t_weight(rel.social_z2(Pi, h, ix), nu)
    assert np.array_equal(np.asarray(got), np.asarray(rel.social_gamma(Pi, h, ix, nu)))


def test_validation_errors():
    scn = phlogiston_scenario(StructuralConfig(n_steps=4))
    n = 3
    g = graphs.complete(n)
    spec = _spec(scn, n)
    with pytest.raises(ValueError, match="gate_mode"):
        run_simulation(scn, g, spec, gate_mode="bogus")
    with pytest.raises(ValueError, match="requires social_nu"):
        run_simulation(scn, g, spec, trust_memory=0.9)
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        run_simulation(scn, g, spec, social_nu=1.0,
                       social_idx=tuple(_belt_nodes(scn)), trust_memory=1.5)


# ----------------------------------------------------------------------
# 2. the live cost field: informative where the row-stochastic T 1 is constant.
# ----------------------------------------------------------------------

def test_cost_field_live_nonconstant_hub_vs_belt():
    """kappa must read the structure: the hub (max off-diagonal coupling mass)
    carries the largest re-equilibration cost; the field is nonnegative and far
    from constant -- exactly what the row-stochastic T 1 cannot deliver."""
    scn = phlogiston_scenario(StructuralConfig(n_steps=4))
    Pi0 = np.asarray(scn.Pi0[0])
    kappa = np.asarray(cost_field_live(jnp.asarray(Pi0)[None]))[0]
    off = np.abs(Pi0) * (1.0 - np.eye(Pi0.shape[0]))
    hub = int(off.sum(axis=1).argmax())
    assert kappa.argmax() == hub
    assert (kappa[hub] > kappa[np.arange(len(kappa)) != hub]).all()
    assert kappa.min() >= -1e-9
    assert np.ptp(kappa) > 0.1
    belt = _belt_nodes(scn)
    assert (kappa[hub] > kappa[list(belt)]).all()


def test_cost_field_live_isolated_zero():
    """A node with no couplings -- e.g. a pinned, not-yet-awakened slot -- must
    read exactly 0; a fully diagonal net reads 0 everywhere."""
    d = 5
    Pi_diag = jnp.asarray(np.diag(np.full(d, 4.0)))[None]
    assert np.allclose(np.asarray(cost_field_live(Pi_diag))[0], 0.0, atol=1e-9)

    Pi = np.diag(np.full(d, 4.0))
    Pi[0, 1] = Pi[1, 0] = 2.0
    Pi[1, 2] = Pi[2, 1] = 1.5          # nodes 3, 4 isolated
    kappa = np.asarray(cost_field_live(jnp.asarray(Pi)[None]))[0]
    assert kappa[3] == pytest.approx(0.0, abs=1e-9)
    assert kappa[4] == pytest.approx(0.0, abs=1e-9)
    assert kappa[1] > kappa[0] > 0.0   # the 2-coupling node costs most


# ----------------------------------------------------------------------
# 3. the two gates disagree: the Figure-1 decoupling, operational.
# ----------------------------------------------------------------------

def test_cost_gate_decouples_from_conviction():
    """central-yet-value-neutral vs cheap-yet-strongly-held: with H_disc = I the
    conviction gate closes the valued leaf's channel hardest, the cost gate the
    hub's -- the two modes must pick different channels."""
    names = ("hub", "leaf1", "leaf2")
    Pi = jnp.asarray(np.array([[4.0, 2.0, 2.0],
                               [2.0, 4.0, 0.0],
                               [2.0, 0.0, 4.0]]))
    u = jnp.asarray(np.array([0.0, 0.0, 1.0]))        # value lives on leaf2 only
    U = PrecisionUtilityNet(names=names, Pi=Pi, h=jnp.zeros(3), u=u,
                            alpha=0.5).effective_utility()[None]      # (1, 3)
    kappa = cost_field_live(Pi[None])                                  # (1, 3)
    H_disc = jnp.eye(3)
    w_conv = np.asarray(rel.gate_weights(U, None, H_disc, "conviction", 1.0, 0.0))[0]
    w_cost = np.asarray(rel.gate_weights(None, kappa, H_disc, "cost", 1.0, 0.0))[0]
    assert w_conv.argmin() == 2        # wishful: silence what bears on the valued node
    assert w_cost.argmin() == 0        # dogmatic: silence what tugs the hub
    assert w_conv.argmin() != w_cost.argmin()


def test_cost_gate_live_in_simulation():
    scn = phlogiston_scenario(StructuralConfig(n_steps=16, t_shift=4))
    n = 5
    g = graphs.complete(n)
    spec = _spec(scn, n)
    kw = dict(forgetting=0.92, snapshot_every=4, seed=0,
              endogenous_gamma=True, gate_strength=2.0)
    r_conv = run_simulation(scn, g, spec, **kw, gate_mode="conviction")
    r_cost = run_simulation(scn, g, spec, **kw, gate_mode="cost")
    assert not np.array_equal(r_conv["snap_Pi"], r_cost["snap_Pi"])
    assert np.isfinite(r_cost["snap_Pi"]).all() and np.isfinite(r_cost["snap_h"]).all()
    # the cost gate actually silences something (realised gamma > 0 at some snapshot)
    assert max(r_cost["gamma_t"]) > 0.0


# ----------------------------------------------------------------------
# 4. earned trust: the EMA track record vs the instantaneous read.
# ----------------------------------------------------------------------

def test_trust_memory_persistence_and_slow_recovery():
    """After sustained disagreement ends, the remembered gamma must stay discounted
    while the instantaneous gamma snaps back; the record then decays monotonically."""
    nu, omega_T = 1.0, 0.9
    Z2 = None
    for _ in range(20):                                   # sustained disagreement
        Z2 = rel.trust_memory_update(Z2, jnp.asarray(10.0), omega_T)
    gamma_mem_onset = float(rel.student_t_weight(Z2, nu))
    gamma_inst = float(rel.student_t_weight(jnp.asarray(0.0), nu))   # z2 = 0 now
    assert gamma_mem_onset < 0.5 * gamma_inst             # discount persists

    trail = []
    for _ in range(30):                                   # agreement from here on
        Z2 = rel.trust_memory_update(Z2, jnp.asarray(0.0), omega_T)
        trail.append(float(Z2))
    assert all(a > b for a, b in zip(trail, trail[1:]))   # monotone recovery
    assert trail[-1] < 0.5                                # trust is re-earned

    # omega_T -> 0 recovers the instantaneous statistic after the first step
    assert float(rel.trust_memory_update(jnp.asarray(5.0), jnp.asarray(1.0), 1e-9)) \
        == pytest.approx(1.0, abs=1e-6)


def test_trust_memory_live_both_social_gates():
    scn = phlogiston_scenario(StructuralConfig(n_steps=16, t_shift=4))
    n = 5
    g = graphs.complete(n)
    spec = _spec(scn, n)
    m_i, g_i, c_i = _belt_nodes(scn)
    base = dict(forgetting=0.92, snapshot_every=4, seed=0, social_nu=1.0)

    means = dict(base, social_idx=(m_i, g_i, c_i))
    r_inst = run_simulation(scn, g, spec, **means)
    r_mem = run_simulation(scn, g, spec, **means, trust_memory=0.9)
    assert not np.array_equal(r_inst["snap_Pi"], r_mem["snap_Pi"])
    assert np.isfinite(r_mem["snap_Pi"]).all() and np.isfinite(r_mem["snap_h"]).all()

    # structure mode needs wirings that actually diverge: the Gaussian Fisher deposit
    # never sees the observation, so homogeneous agents keep IDENTICAL Pi forever and
    # the wiring z2 is exactly 0 (nothing for the memory to remember). Heterogeneous
    # channel weights make the deposits -- hence the couplings -- differ.
    w_het = np.ones((n, scn.m))
    w_het[3:, np.asarray(scn.disc_rows)] = 0.1
    spec_het = AgentSpec(w_obs=jnp.asarray(w_het), lam=jnp.full((n,), 0.2))
    struct = dict(base, social_gate="structure",
                  social_pairs=((m_i, c_i), (g_i, c_i)))
    s_inst = run_simulation(scn, g, spec_het, **struct)
    s_mem = run_simulation(scn, g, spec_het, **struct, trust_memory=0.9)
    assert not np.array_equal(s_inst["snap_Pi"], s_mem["snap_Pi"])
    assert np.isfinite(s_mem["snap_Pi"]).all() and np.isfinite(s_mem["snap_h"]).all()
