"""Tests for the structure-level measurement layer.

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/structure_metrics.py
(coupling_vector, edge_set, spectral_signature, pairwise_distance, dispersion_trace,
consensus_union_report).

These are post-hoc, host-numpy read-offs of belief-net stacks -- never part of the
dynamics -- so the tests are exact-arithmetic unit checks on a two-wiring fixture:
net A holds edge (0,1)@3.0, net B holds edge (2,3)@3.0 (the mirror wiring), the
population is two agents of each. Every expected number below is computed by hand.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.structural import structure_metrics as sm
from src.structural.dual_field import PrecisionUtilityNet


PAIRS = ((0, 1), (2, 3))


def _net(edge, weight=3.0, d=4):
    Pi = 5.0 * np.eye(d)
    i, j = edge
    Pi[i, j] = Pi[j, i] = weight
    return Pi


@pytest.fixture
def fixture():
    A, B = _net((0, 1)), _net((2, 3))
    stack = np.stack([A, A, B, B])
    membership = np.array([0, 0, 1, 1])
    return A, B, stack, membership


# ----------------------------------------------------------------------
# 1. coupling_vector: shape passthrough + exact values.
# ----------------------------------------------------------------------

def test_coupling_vector_values_and_shapes(fixture):
    A, _, stack, _ = fixture
    assert np.array_equal(sm.coupling_vector(A, PAIRS), [3.0, 0.0])
    assert sm.coupling_vector(stack, PAIRS).shape == (4, 2)
    traj = np.stack([stack, stack])                       # (T=2, N=4, d, d)
    out = sm.coupling_vector(traj, PAIRS)
    assert out.shape == (2, 4, 2)
    assert np.array_equal(out[0], out[1])


# ----------------------------------------------------------------------
# 2. pairwise_distance, kind='fro'.
# ----------------------------------------------------------------------

def test_fro_distance_exact(fixture):
    A, B, stack, _ = fixture
    D = sm.pairwise_distance(stack, kind="fro", pairs=PAIRS)
    assert D.shape == (4, 4)
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0.0)
    # A holds [3, 0], B holds [0, 3] -> ||A - B|| = 3 sqrt(2)
    assert np.isclose(D[0, 2], 3.0 * np.sqrt(2.0))
    assert np.isclose(D[0, 1], 0.0)


# ----------------------------------------------------------------------
# 3. pairwise_distance, kind='jaccard'.
# ----------------------------------------------------------------------

def test_jaccard_distance(fixture):
    _, _, stack, _ = fixture
    D = sm.pairwise_distance(stack, kind="jaccard", pairs=PAIRS, threshold=1.0)
    assert np.isclose(D[0, 2], 1.0)                       # disjoint edge sets
    assert np.isclose(D[0, 1], 0.0)                       # identical edge sets
    # threshold above every coupling: both sets empty -> distance 0 (agree on absence)
    D_empty = sm.pairwise_distance(stack, kind="jaccard", pairs=PAIRS, threshold=10.0)
    assert np.allclose(D_empty, 0.0)


# ----------------------------------------------------------------------
# 4. spectral_signature: permutation invariance + the operator mirror.
# ----------------------------------------------------------------------

def test_spectral_permutation_invariant_and_mirror_blind(fixture):
    A, B, _, _ = fixture
    perm = np.array([2, 3, 0, 1])
    A_perm = A[perm][:, perm]
    assert np.allclose(sm.spectral_signature(A), sm.spectral_signature(A_perm))
    # mirror wirings (same shape, different nodes) read as 0 -- documented behavior
    assert np.allclose(sm.spectral_signature(A), sm.spectral_signature(B))
    # a net holding BOTH edges is a genuinely different shape
    both = _net((0, 1))
    both[2, 3] = both[3, 2] = 3.0
    D = sm.pairwise_distance(np.stack([A, both]), kind="spectral")
    assert D[0, 1] > 0.1


def test_spectral_signature_matches_utility_operator(fixture):
    A, _, _, _ = fixture
    net = PrecisionUtilityNet(names=("a", "b", "c", "d"), Pi=jnp.asarray(A),
                              h=jnp.zeros(4), u=jnp.zeros(4))
    ev = np.sort(np.linalg.eigvals(np.asarray(net.utility_operator())).real)[::-1]
    assert np.allclose(sm.spectral_signature(A), ev)


# ----------------------------------------------------------------------
# 5. dispersion_trace: the within/cross decomposition, exact on T=1.
# ----------------------------------------------------------------------

def test_dispersion_trace_exact(fixture):
    _, _, stack, membership = fixture
    tr = sm.dispersion_trace(stack[None], membership, kind="fro", pairs=PAIRS)
    d_ab = 3.0 * np.sqrt(2.0)
    assert np.allclose(tr["within"], [[0.0, 0.0]])
    assert np.allclose(tr["cross"], [d_ab])               # 4 cross pairs, all d_ab
    assert np.allclose(tr["overall"], [4.0 * d_ab / 6.0])  # 6 unordered pairs


# ----------------------------------------------------------------------
# 6. consensus_union_report: union surplus, exact dilution, the theorem-let.
# ----------------------------------------------------------------------

def test_consensus_union_report_exact(fixture):
    A, B, _, _ = fixture
    rep = sm.consensus_union_report(np.stack([A, B]), pairs=PAIRS, threshold=1.0)
    assert np.array_equal(rep["per_agent_counts"], [1, 1])
    assert rep["union_count"] == 2 and rep["best_individual_count"] == 1
    assert rep["union_coverage_ratio"] == 2.0
    assert np.allclose(rep["consensus_couplings"], [1.5, 1.5])
    assert rep["dilution_ratio"] == 0.5                    # |mean| / max = 1.5 / 3.0
    assert rep["edges_lost"] == frozenset()                # 1.5 still > threshold 1.0
    assert rep["synergy_edges"] == frozenset()


def test_consensus_loses_edges_but_never_creates(fixture):
    A, B, _, _ = fixture
    rep = sm.consensus_union_report(np.stack([A, B]), pairs=PAIRS, threshold=2.0)
    assert rep["consensus_edges"] == frozenset()           # 1.5 < 2.0: both diluted away
    assert rep["edges_lost"] == frozenset({(0, 1), (2, 3)})
    # the theorem-let: |mean_i Pi_i[e]| <= max_i |Pi_i[e]| -- no synergy, ever
    assert rep["synergy_edges"] == frozenset()


def test_consensus_mean_when_h_given(fixture):
    A, B, _, _ = fixture
    h = np.stack([np.ones(4), 3.0 * np.ones(4)])
    rep = sm.consensus_union_report(np.stack([A, B]), h_stack=h, pairs=PAIRS)
    Pi_cons = 0.5 * (A + B)
    assert np.allclose(rep["consensus_mean"], np.linalg.solve(Pi_cons, 2.0 * np.ones(4)))


# ----------------------------------------------------------------------
# 7. the structure-reading trust gate statistic (reliability.pairwise_structure_z2).
# ----------------------------------------------------------------------

def test_pairwise_structure_z2_gate(fixture):
    from src.structural import reliability as rel
    from src.structural.step import trust_weights

    A, B, stack, _ = fixture
    pairs = jnp.asarray(np.asarray(PAIRS, dtype=int))
    z2 = np.asarray(rel.pairwise_structure_z2(jnp.asarray(np.stack([A, B])), pairs))
    # per coupling: (3-0)^2 / max((9+0)/2, floor^2) = 2; summed over both -> 4 exactly
    assert np.isclose(z2[0, 1], 4.0)
    assert np.allclose(np.diag(z2), 0.0)
    # on the (A, A, B, B) population: within-camp trust > cross-camp trust
    gam = np.asarray(rel.student_t_weight(
        rel.pairwise_structure_z2(jnp.asarray(stack), pairs), 0.5))
    assert gam[0, 1] > gam[0, 2]
    W = np.asarray(trust_weights(jnp.ones((4, 4)), jnp.asarray(gam)))
    assert np.allclose(W.sum(axis=1), 1.0)


# ----------------------------------------------------------------------
# 8. engine: defaults byte-identical; structure-gate run healthy.
# ----------------------------------------------------------------------

def test_engine_defaults_byte_identical_and_structure_gate_smoke():
    from src.structural import graphs
    from src.structural.simulation import run_simulation, AgentSpec
    from src.structural.scenarios import cosmology_scenario, COSMOLOGY_EDGES

    n = 6
    scn = cosmology_scenario(n_steps=12, t1=4, t2=8, sigma_o=0.5)
    idx = {nm: i for i, nm in enumerate(scn.names)}
    edges_ij = tuple((idx[a], idx[c]) for (a, c) in COSMOLOGY_EDGES)
    g = graphs.complete(n)
    spec = AgentSpec(w_obs=jnp.ones((n, scn.m)), lam=jnp.full((n,), 0.2))

    r0 = run_simulation(scn, g, spec, forgetting=0.9, snapshot_every=4, seed=0)
    r1 = run_simulation(scn, g, spec, forgetting=0.9, snapshot_every=4, seed=0,
                        social_gate="means", social_pairs=None, social_scale_floor=0.1)
    assert np.array_equal(r0["snap_Pi"], r1["snap_Pi"])
    assert np.array_equal(r0["snap_h"], r1["snap_h"])

    r2 = run_simulation(scn, g, spec, forgetting=0.9, snapshot_every=4, seed=0,
                        social_nu=0.5, social_gate="structure", social_pairs=edges_ij)
    assert np.isfinite(r2["snap_Pi"]).all() and np.isfinite(r2["snap_h"]).all()
    assert r2["snap_Pi"].shape[1:] == (n, scn.dim, scn.dim)
    assert r2["snap_h"].shape[1:] == (n, scn.dim)

    with pytest.raises(ValueError, match="social_pairs"):
        run_simulation(scn, g, spec, snapshot_every=4, seed=0,
                       social_nu=0.5, social_gate="structure")
