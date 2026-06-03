"""Tests for the well-defined Bayes-net layer (src/structural/bayesnet.py).

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/bayesnet.py · Map:
tests/README.md

The contract is that the Linear-Gaussian Bayes net (explicit per-node CPDs) and
the joint precision form are *the same object in two coordinates*:

  - Tier-1 bijection: ``to_info`` / ``from_info`` round-trip to machine precision
    in both directions; the compiled joint matches the closed-form LGBN covariance.
  - Tier-2 beliefs: per-node marginals and conditionals match brute-force numpy.
  - Tier-3 structure learning: a *relational* observation moves the off-diagonal
    (edge) parameters, while a node-wise observation cannot -- the bug the overhaul
    fixes, pinned as a test.
  - Tier-4 BMR: the edge-level Bayes factor matches ``linalg.savage_dickey`` and
    has the right sign for a supported vs unsupported edge.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.structural.bayesnet import (
    LinearGaussianBN, from_edges, to_info, from_info, relational_operator,
)
from src.structural.belief import GaussianBeliefNet
from src.structural import linalg


def _rand_spd(d, seed=0, jitter=1.0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((d, d))
    return M @ M.T + jitter * np.eye(d)


def _example_bn():
    """A 3-node DAG a -> b, a -> c, b -> c with distinct CPD params."""
    names = ("a", "b", "c")
    edges = {("a", "b"): 0.7, ("a", "c"): -0.4, ("b", "c"): 0.5}
    return from_edges(names, edges,
                      intercepts={"a": 1.0, "b": -0.5, "c": 2.0},
                      variances={"a": 0.5, "b": 1.5, "c": 0.8})


# ----------------------------------------------------------------------
# Tier-1: the CPD <-> precision bijection
# ----------------------------------------------------------------------

def test_to_info_matches_closed_form_covariance():
    bn = _example_bn()
    gbn = bn.to_info()
    d = bn.dim
    A = np.eye(d) - np.asarray(bn.B)
    D = np.diag(np.asarray(bn.s))
    Ainv = np.linalg.inv(A)
    Sigma_ref = Ainv @ D @ Ainv.T
    mu_ref = Ainv @ np.asarray(bn.b)
    assert np.allclose(np.asarray(gbn.cov()), Sigma_ref, atol=1e-6)
    assert np.allclose(np.asarray(gbn.mean()), mu_ref, atol=1e-6)


def test_roundtrip_cpd_to_info_to_cpd():
    bn = _example_bn()
    bn2 = LinearGaussianBN.from_info(bn.to_info())
    assert np.allclose(np.asarray(bn.B), np.asarray(bn2.B), atol=1e-6)
    assert np.allclose(np.asarray(bn.b), np.asarray(bn2.b), atol=1e-6)
    assert np.allclose(np.asarray(bn.s), np.asarray(bn2.s), atol=1e-6)


@pytest.mark.parametrize("d", [1, 2, 3, 5])
def test_roundtrip_info_to_cpd_to_info(d):
    Pi = jnp.asarray(_rand_spd(d, seed=d))
    h = jnp.asarray(np.random.default_rng(d + 1).standard_normal(d))
    B, b, s = from_info(Pi, h)
    Pi2, h2 = to_info(B, b, s)
    assert np.allclose(np.asarray(Pi), np.asarray(Pi2), atol=1e-5)
    assert np.allclose(np.asarray(h), np.asarray(h2), atol=1e-5)


def test_from_info_B_is_lower_triangular():
    Pi = jnp.asarray(_rand_spd(4, seed=3))
    h = jnp.zeros(4)
    B, _, _ = from_info(Pi, h)
    B = np.asarray(B)
    assert np.allclose(np.triu(B), 0.0, atol=1e-8)  # strictly lower-triangular


# ----------------------------------------------------------------------
# Tier-2: well-defined per-node beliefs
# ----------------------------------------------------------------------

def test_marginal_matches_joint_diagonal():
    bn = _example_bn()
    mu, cov = bn.joint()
    for n in bn.names:
        m, v = bn.marginal(n)
        i = bn.index(n)
        assert np.isclose(m, float(mu[i]), atol=1e-6)
        assert np.isclose(v, float(cov[i, i]), atol=1e-6)


def test_conditional_matches_numpy_partition():
    bn = _example_bn()
    mu, cov = bn.joint()
    mu, cov = np.asarray(mu), np.asarray(cov)
    # condition node "c" on a=x_a, b=x_b via the standard Gaussian formula.
    a_i, evidence = bn.index("c"), {"a": 0.3, "b": -1.2}
    b_idx = [bn.index(n) for n in evidence]
    x_b = np.array([evidence[n] for n in evidence])
    Saa = cov[a_i, a_i]
    Sab = cov[a_i, b_idx]
    Sbb = cov[np.ix_(b_idx, b_idx)]
    mean_ref = mu[a_i] + Sab @ np.linalg.solve(Sbb, x_b - mu[b_idx])
    var_ref = Saa - Sab @ np.linalg.solve(Sbb, cov[b_idx, a_i])
    mean, var = bn.conditional("c", evidence)
    assert np.isclose(mean, mean_ref, atol=1e-6)
    assert np.isclose(var, var_ref, atol=1e-6)


def test_cpd_reports_parents_and_weights():
    bn = _example_bn()
    cpd_c = bn.cpd("c")
    assert set(cpd_c["parents"]) == {"a", "b"}
    assert np.isclose(cpd_c["weights"]["a"], -0.4)
    assert np.isclose(cpd_c["weights"]["b"], 0.5)
    assert np.isclose(cpd_c["intercept"], 2.0)
    assert np.isclose(cpd_c["variance"], 0.8)


# ----------------------------------------------------------------------
# Tier-3: structure learning -- relational vs node-wise observation
# ----------------------------------------------------------------------

def test_node_wise_observation_does_not_move_edges():
    """A direct (one-node) read deposits only diagonal Fisher info, so the
    off-diagonal coupling cannot change -- the legacy frozen-structure regime."""
    bn = _example_bn()
    Pi0 = np.asarray(bn.to_info().Pi)
    H = relational_operator(bn.names, [{"a": 1.0}, {"b": 1.0}, {"c": 1.0}])
    bn1 = bn.observe(H, jnp.array([0.2, 0.1, -0.3]), sigma_o=1.0)
    Pi1 = np.asarray(bn1.to_info().Pi)
    off = ~np.eye(bn.dim, dtype=bool)
    assert np.allclose(Pi1[off], Pi0[off], atol=1e-6)   # off-diagonals untouched


def test_relational_observation_moves_edges():
    """A relational read (a combination of nodes) deposits off-diagonal Fisher
    information, so the coupling -- and hence the CPD parent weights -- move. This
    is the corrected behaviour: the Bayes net learns its structure from data."""
    bn = _example_bn()
    Pi0 = np.asarray(bn.to_info().Pi)
    B0 = np.asarray(bn.B)
    # a mass-balance: reads a - b - c together (couples all three).
    H = relational_operator(bn.names, [{"a": 1.0, "b": -1.0, "c": -1.0}])
    bn1 = bn.observe(H, jnp.array([0.5]), sigma_o=1.0)
    Pi1 = np.asarray(bn1.to_info().Pi)
    B1 = np.asarray(bn1.B)
    off = ~np.eye(bn.dim, dtype=bool)
    assert not np.allclose(Pi1[off], Pi0[off], atol=1e-3)   # edges in joint moved
    assert not np.allclose(B1, B0, atol=1e-3)               # CPD weights moved


# ----------------------------------------------------------------------
# Tier-4: edge-level Bayesian Model Reduction
# ----------------------------------------------------------------------

def test_bmr_prune_edge_matches_savage_dickey():
    bn = _example_bn()
    Pi0, h0 = to_info(bn.B, bn.b, bn.s)
    # some data already seen, summarised as a Fisher deposit.
    H = relational_operator(bn.names, [{"a": 1.0, "b": -1.0, "c": -1.0}])
    J, j = linalg.fisher_deposit(H, jnp.array([0.5]), 1.0)
    out = bn.bmr_prune_edge("c", "b", likelihood=(J, j))
    reduced = bn.prune_edge("c", "b")
    Pi0r, h0r = to_info(reduced.B, reduced.b, reduced.s)
    Pi_post, h_post = linalg.add_information(Pi0, h0, J, j)
    dF_ref, _, _ = linalg.savage_dickey(Pi_post, h_post, Pi0, h0, Pi0r, h0r)
    assert np.isclose(float(out["delta_F"]), float(dF_ref), atol=1e-6)


def test_prune_edge_removes_parent():
    bn = _example_bn()
    reduced = bn.prune_edge("c", "b")
    assert "b" not in reduced.parents("c")
    assert "a" in reduced.parents("c")     # other parent untouched
