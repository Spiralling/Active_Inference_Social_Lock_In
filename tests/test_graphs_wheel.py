"""Tests for the wheel constructor (the missing Zollman topology).

Layer: structural · Guards: src/structural/graphs.py::wheel.

The wheel is the intermediate-connectivity reference between the cycle (ring) and the
complete graph; the network-epistemology speed/accuracy figures ride on the Fiedler-
value ordering ring < wheel < complete, so that is pinned here alongside the basic
shape (n nodes, a hub of degree n-1).
"""

from __future__ import annotations

import numpy as np

from src.structural import graphs


def test_wheel_shape_and_hub():
    n = 8
    g = graphs.wheel(n)
    A = np.asarray(g.A)
    assert A.shape == (n, n)
    assert np.allclose(A, A.T) and np.allclose(np.diag(A), 0.0)
    assert g.kind == "wheel"
    # node 0 is the hub: connected to all n-1 rim nodes
    assert int(A[0].sum()) == n - 1
    # rim nodes have degree 3 (two cycle neighbours + the hub)
    assert all(int(A[i].sum()) == 3 for i in range(1, n))


def test_wheel_connectivity_between_ring_and_complete():
    n = 12
    l_ring = graphs.algebraic_connectivity(graphs.ring(n))
    l_wheel = graphs.algebraic_connectivity(graphs.wheel(n))
    l_complete = graphs.algebraic_connectivity(graphs.complete(n))
    assert l_ring < l_wheel < l_complete
