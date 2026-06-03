"""Tests for the first-class graph layer (src/structural/graphs.py).

Layer: structural · Guards: src/structural/graphs.py · Map: tests/README.md

Two contracts:
  1. **Validity** -- every constructor returns a symmetric 0/1 adjacency with a
     zero diagonal, the right size, and the right structural signature (a complete
     graph is (n-1)-regular, a ring is degree-k, a community graph disconnects at
     ``inter=0``, ...).
  2. **Legacy equivalence** -- the families that already existed in
     ``build_adjacency`` (scale_free / watts_strogatz / planted_sbm) are byte-
     identical when reached through the new ``graphs.*`` constructors and through
     ``graph_from_config``, so the rewrite changed the *interface*, not the graphs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.network import build_adjacency
from src.config import NetworkConfig
from src.structural import graphs
from src.structural.graphs import Graph


def _is_valid_adjacency(A: np.ndarray, n: int) -> None:
    assert A.shape == (n, n)
    assert np.array_equal(A, A.T)                       # symmetric
    assert np.all(np.diag(A) == 0.0)                    # zero diagonal
    assert set(np.unique(A)).issubset({0.0, 1.0})       # binary


# ----------------------------------------------------------------------
# Validity of every constructor.
# ----------------------------------------------------------------------

def test_erdos_renyi_valid_and_degree():
    g = graphs.erdos_renyi(60, p=0.2, seed=1)
    _is_valid_adjacency(g.A, 60)
    assert g.kind == "erdos_renyi"
    # mean degree ~ p (n-1)
    assert abs(g.A.sum(1).mean() - 0.2 * 59) < 6.0


def test_erdos_renyi_mean_degree_target():
    g = graphs.erdos_renyi(80, mean_degree=6, seed=2)
    _is_valid_adjacency(g.A, 80)
    assert abs(g.A.sum(1).mean() - 6.0) < 3.0


def test_erdos_renyi_needs_p_or_degree():
    with pytest.raises(ValueError):
        graphs.erdos_renyi(10)


def test_scale_free_alias_and_validity():
    g = graphs.scale_free(40, mean_degree=4, seed=0)
    _is_valid_adjacency(g.A, 40)
    assert g.kind == "scale_free"
    assert graphs.scale_free is graphs.barabasi_albert


def test_watts_strogatz_validity():
    g = graphs.watts_strogatz(40, mean_degree=4, rewiring_p=0.1, seed=0)
    _is_valid_adjacency(g.A, 40)


def test_ring_is_k_regular():
    g = graphs.ring(20, mean_degree=4)
    _is_valid_adjacency(g.A, 20)
    assert np.all(g.A.sum(1) == 4)                      # exactly k-regular


def test_complete_is_full_and_uniform_W():
    g = graphs.complete(12)
    _is_valid_adjacency(g.A, 12)
    assert np.all(g.A.sum(1) == 11)                     # (n-1)-regular
    W = np.asarray(g.trust_W())
    assert np.allclose(W, np.full((12, 12), 1 / 12))    # closed nbhd = everyone


def test_lattice_validity_and_degree():
    g = graphs.lattice(5, 5)
    _is_valid_adjacency(g.A, 25)
    # interior nodes have degree 4, corners 2, edges 3 -> mean in (2, 4)
    assert 2.0 < g.A.sum(1).mean() < 4.0


# ----------------------------------------------------------------------
# Community graph: membership, disconnection, re-bridging.
# ----------------------------------------------------------------------

def test_community_membership_contiguous():
    g = graphs.community([5, 7, 4], intra=0.5, inter=0.05, seed=0)
    _is_valid_adjacency(g.A, 16)
    assert g.n_blocks == 3
    assert list(g.membership) == [0]*5 + [1]*7 + [2]*4


def test_community_disconnects_at_inter_zero():
    g = graphs.community([10, 10], intra=0.4, inter=0.0, seed=0)
    cross = g.A[:10, 10:]
    assert cross.sum() == 0.0                           # genuinely disconnected
    W = np.asarray(g.trust_W())
    assert W[:10, 10:].sum() == 0.0                     # zero cross-community trust


def test_with_bridge_adds_cross_edges_only():
    g0 = graphs.community([10, 10], intra=0.4, inter=0.0, seed=3)
    g1 = g0.with_bridge(0.3)
    assert g1.membership is not None and g1.n_blocks == 2
    # bridge opens cross-community edges
    assert g1.A[:10, 10:].sum() > 0.0


def test_with_bridge_requires_membership():
    with pytest.raises(ValueError):
        graphs.complete(8).with_bridge(0.2)


def test_isolated_removes_all_edges():
    g = graphs.community([6, 6], intra=0.5, inter=0.1, seed=0).isolated()
    assert g.A.sum() == 0.0
    assert np.allclose(np.asarray(g.trust_W()), np.eye(12))  # W = identity
    assert g.membership is not None                     # membership preserved


# ----------------------------------------------------------------------
# Legacy equivalence: new interface, identical graphs.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["scale_free", "watts_strogatz"])
def test_constructor_matches_build_adjacency(kind):
    ctor = graphs.scale_free if kind == "scale_free" else graphs.watts_strogatz
    g = ctor(50, mean_degree=4, seed=7) if kind == "scale_free" \
        else ctor(50, mean_degree=4, rewiring_p=0.1, seed=7)
    ref = build_adjacency(50, mean_degree=4, rewiring_p=0.1, seed=7, kind=kind) \
        if kind == "watts_strogatz" \
        else build_adjacency(50, mean_degree=4, rewiring_p=0.0, seed=7, kind=kind)
    assert np.array_equal(g.A, ref)


def test_community_matches_legacy_sbm():
    membership = np.array([0]*12 + [1]*12)
    ref = build_adjacency(24, mean_degree=0, rewiring_p=0.0, seed=5,
                          kind="planted_sbm", society_membership=membership,
                          intra_prob=0.3, inter_prob=0.05)
    g = graphs.community([12, 12], intra=0.3, inter=0.05, seed=5)
    assert np.array_equal(g.A, ref)


def test_graph_from_config_legacy_path():
    nc = NetworkConfig(kind="watts_strogatz", mean_degree=4, rewiring_p=0.1)
    g = graphs.graph_from_config(nc, 30, seed=0)
    ref = build_adjacency(30, mean_degree=4, rewiring_p=0.1, seed=0,
                          kind="watts_strogatz")
    assert np.array_equal(g.A, ref)


def test_graph_from_config_lattice_requires_square():
    nc = NetworkConfig(kind="lattice")
    with pytest.raises(ValueError):
        graphs.graph_from_config(nc, 10, seed=0)        # 10 is not a perfect square


# ----------------------------------------------------------------------
# Graph hashing: stable static key (metadata only, not array values).
# ----------------------------------------------------------------------

def test_graph_hash_ignores_array_values():
    a = graphs.erdos_renyi(20, mean_degree=4, seed=1)
    b = graphs.erdos_renyi(20, mean_degree=4, seed=999)   # different A, same shape
    assert hash(a) == hash(b) and a == b                  # same compiled-rollout key
    c = graphs.erdos_renyi(21, mean_degree=4, seed=1)     # different n
    assert a != c


def test_trust_W_is_row_stochastic():
    for g in [graphs.erdos_renyi(30, mean_degree=4, seed=0),
              graphs.scale_free(30, mean_degree=4, seed=0),
              graphs.community([15, 15], intra=0.4, inter=0.05, seed=0)]:
        W = np.asarray(g.trust_W())
        assert np.allclose(W.sum(1), 1.0)
