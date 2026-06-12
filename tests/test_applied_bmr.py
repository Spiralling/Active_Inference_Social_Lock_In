"""Tests for APPLIED Bayesian Model Reduction (in-loop structural pruning).

Layer: structural (THE PAPER OBJECT) · Guards: simulation.run_simulation(bmr_every=...),
simulation._joint_reduced, the phlogiston Scenario.reduce_fn (joint CPD reducer).

The crisis check executed: every ``bmr_every`` steps each agent's ledger-flagged edges
are removed by swapping in the exact Savage-Dickey reduced posterior (joint reduced
reference prior + untouched deposit), anchor included. Defaults must stay byte-identical.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.structural import graphs
from src.structural.phlogiston import StructuralConfig
from src.structural.scenarios import phlogiston_scenario, cosmology_scenario
from src.structural.simulation import run_simulation, AgentSpec, _joint_reduced


def _spec(scn, n, lam=0.2):
    return AgentSpec(w_obs=jnp.ones((n, scn.m)), lam=jnp.full((n,), float(lam)))


def _belt_nodes(scn):
    idx = {nm: i for i, nm in enumerate(scn.names)}
    return (idx["mass_change_sign"], idx["gas_consumed"],
            idx["calx_heavier_than_metal"])


# ----------------------------------------------------------------------
# 1. defaults byte-identical; out-of-range cadence applies nothing.
# ----------------------------------------------------------------------

def test_defaults_byte_identical_and_inert_cadence():
    scn = phlogiston_scenario(StructuralConfig(n_steps=12, t_shift=4))
    n = 5
    g = graphs.complete(n)
    spec = _spec(scn, n)
    r0 = run_simulation(scn, g, spec, forgetting=0.92, snapshot_every=4, seed=0)
    r1 = run_simulation(scn, g, spec, forgetting=0.92, snapshot_every=4, seed=0,
                        bmr_every=None)
    assert np.array_equal(r0["snap_Pi"], r1["snap_Pi"])
    assert np.array_equal(r0["snap_h"], r1["snap_h"])
    assert "applied_pruned_t" not in r0

    # cadence beyond the horizon: BMR on but never fires -> dynamics byte-identical,
    # telemetry recorded all-False.
    r2 = run_simulation(scn, g, spec, forgetting=0.92, snapshot_every=4, seed=0,
                        bmr_every=1000)
    assert np.array_equal(r0["snap_Pi"], r2["snap_Pi"])
    assert np.array_equal(r0["snap_h"], r2["snap_h"])
    assert r2["applied_pruned_t"].shape == (r0["snap_Pi"].shape[0], n, scn.n_edges)
    assert not r2["applied_pruned_t"].any()
    assert r2["bmr_every"] == 1000


# ----------------------------------------------------------------------
# 2. the joint reducer: one-hot == per-edge reduced prior; joint != delta-sum.
# ----------------------------------------------------------------------

def test_reduce_fn_one_hot_matches_per_edge_priors():
    scn = phlogiston_scenario(StructuralConfig(n_steps=4))
    Pi0r, h0r = np.asarray(scn.Pi0r[0]), np.asarray(scn.h0r[0])
    h0 = np.asarray(scn.h0[0])
    # the phlogiston prune's h-replacement is NOT the identity (recompiled CPD net)
    assert np.abs(h0r - h0[None]).max() > 1e-6
    for k in range(scn.n_edges):
        flags = np.zeros(scn.n_edges, dtype=bool)
        flags[k] = True
        R, r = scn.reduce_fn(0, flags)
        assert np.allclose(np.asarray(R), Pi0r[k], atol=1e-5)
        assert np.allclose(np.asarray(r), h0r[k], atol=1e-5)


def test_joint_belt_reduce_differs_from_delta_sum():
    """The belt edges share the child calx_heavier_than_metal: summing per-edge prior
    deltas double-subtracts the co-parent fill-in B_m B_g / s_c, planting a spurious
    coupling between the two parents. The joint CPD recompile removes it exactly --
    the reason Scenario.reduce_fn exists."""
    scn = phlogiston_scenario(StructuralConfig(n_steps=4))
    m_i, g_i, c_i = _belt_nodes(scn)
    Pi0 = np.asarray(scn.Pi0[0])
    Pi0r = np.asarray(scn.Pi0r[0])
    belt = np.asarray(scn.belt_ix)
    flags = np.zeros(scn.n_edges, dtype=bool)
    flags[belt] = True
    Rj, _ = scn.reduce_fn(0, flags)
    Rj = np.asarray(Rj)
    # both belt couplings removed exactly
    assert Rj[m_i, c_i] == pytest.approx(0.0, abs=1e-6)
    assert Rj[g_i, c_i] == pytest.approx(0.0, abs=1e-6)
    # joint reduce kills the co-parent fill-in too; the delta-sum gets it WRONG
    R_sum = Pi0 + (Pi0r[belt] - Pi0[None]).sum(axis=0)
    assert Rj[m_i, g_i] == pytest.approx(0.0, abs=1e-6)
    assert abs(R_sum[m_i, g_i]) > 0.5                  # the double-subtracted term
    assert not np.allclose(Rj, R_sum, atol=1e-3)
    # and the joint reduced prior is PD (compiled CPD net)
    assert np.linalg.eigvalsh(Rj).min() > 0.0


def test_joint_reduced_fallback_exact_for_disjoint_entries():
    """Cosmology has reduce_fn=None; its per-edge reductions zero DISTINCT matrix
    entries, so the engine's delta-sum fallback is exact for any flag combination."""
    scn = cosmology_scenario(n_steps=8, t1=3, t2=6)
    assert scn.reduce_fn is None
    E = scn.n_edges
    F = np.zeros((2, E), dtype=bool)
    F[0, 0] = F[0, 3] = True                            # two edges, distinct entries
    R, r = _joint_reduced(scn, 0, F)
    Pi0, h0 = np.asarray(scn.Pi0[0]), np.asarray(scn.h0[0])
    Pi0r = np.asarray(scn.Pi0r[0])
    expect = Pi0 + (Pi0r[0] - Pi0) + (Pi0r[3] - Pi0)
    assert np.allclose(R[0], expect)
    assert np.allclose(R[1], Pi0) and np.allclose(r[1], h0)


# ----------------------------------------------------------------------
# 3. application semantics: flagged structure actually leaves the net.
# ----------------------------------------------------------------------

def test_applied_prune_removes_belt_structure():
    """Oxygen world from t=0, lam ~ 0 (pure evidence), self-censorship keeping the
    deposit window small: the belt must get flagged AND the applied run's belt coupling
    must end far below the read-out run's (the flags now feed back into the net)."""
    cfg = StructuralConfig(n_steps=40, t_shift=0)
    scn = phlogiston_scenario(cfg)
    n = 6
    g = graphs.complete(n)
    w = np.ones((n, scn.m))
    w[:, np.asarray(scn.disc_rows)] = 0.05              # small deposit window: the
    # residual coupling after a prune is the KEPT deposit (exact Savage-Dickey
    # semantics), so the absolute check below needs the window well under the prior
    spec = AgentSpec(w_obs=jnp.asarray(w), lam=jnp.full((n,), 1e-3))
    kw = dict(forgetting=0.92, snapshot_every=5, seed=1)
    r_read = run_simulation(scn, g, spec, **kw)
    r_app = run_simulation(scn, g, spec, **kw, bmr_every=4)

    m_i, g_i, c_i = _belt_nodes(scn)
    belt = np.asarray(scn.belt_ix)
    assert r_app["applied_pruned_t"][-1][:, belt].all(), \
        "belt should be flagged-and-applied under oxygen data at lam ~ 0"
    bc_read = np.abs(r_read["snap_Pi"][-1][:, [m_i, g_i], c_i]).mean()
    bc_app = np.abs(r_app["snap_Pi"][-1][:, [m_i, g_i], c_i]).mean()
    assert bc_app < 0.5 * bc_read, (bc_app, bc_read)
    # anchor edit: the pruned coupling relaxes toward 0, not the old prior (~1.8)
    assert bc_app < 0.9
    # deposit kept: the final net sits ABOVE the (per-agent joint) reduced prior on
    # the diagonal -- the prune removed prior structure, never the accumulated data
    R_fin, _ = _joint_reduced(scn, 0, r_app["applied_pruned_t"][-1])
    assert (np.diagonal(r_app["snap_Pi"][-1], axis1=1, axis2=2)
            >= np.diagonal(R_fin, axis1=1, axis2=2) - 1e-5).all()
    # numerics: finite and PD after repeated application
    assert np.isfinite(r_app["snap_Pi"]).all() and np.isfinite(r_app["snap_h"]).all()
    assert min(np.linalg.eigvalsh(P).min() for P in r_app["snap_Pi"][-1]) > 0.0


def test_applied_smoke_shapes_and_health():
    cfg = StructuralConfig(n_steps=20, t_shift=8)
    scn = phlogiston_scenario(cfg)
    n = 4
    spec = _spec(scn, n, lam=0.1)
    r = run_simulation(scn, graphs.complete(n), spec, forgetting=0.92,
                       snapshot_every=4, seed=0, bmr_every=5)
    S = r["snap_Pi"].shape[0]
    assert r["applied_pruned_t"].shape == (S, n, scn.n_edges)
    assert r["applied_pruned_t"].dtype == bool
    assert np.isfinite(r["snap_Pi"]).all() and np.isfinite(r["snap_h"]).all()
