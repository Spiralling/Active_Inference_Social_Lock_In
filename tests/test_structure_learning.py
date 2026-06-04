"""Tests for active structure learning: edge read-out + expand/reduce moves (plan P7).

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/shells.py (edge_count_trace),
src/structural/linalg.py (border / schur_marginalize as the expansion/reduction pair).

The paper's engine is structure learning by EXPANSION (add a node, real inversion) and
REDUCTION (prune in closed form by BMR). These tests pin the edge-count read-out that feeds
the edge-edit timeline, and the reserved-slot expand-then-reduce round-trip (the scan-safe
way to add a node: border into a slot, marginalize it back).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.structural import shells, linalg


def test_edge_count_trace_tracks_growth():
    T, N, d = 12, 3, 4
    Pi_t = np.tile(np.eye(d), (T, N, 1, 1)).astype(float)
    for t in range(T):                       # one off-diagonal edge grows past threshold
        Pi_t[t, :, 0, 1] = Pi_t[t, :, 1, 0] = 0.03 * t
    counts, delta = shells.edge_count_trace(Pi_t, threshold=0.15)
    assert counts.shape == (T,) and delta.shape == (T,)
    assert counts[0] == 0 and counts[-1] >= 1            # edge appears as it accumulates
    assert delta.sum() == pytest.approx(counts[-1] - counts[0])


def test_reserved_slot_disconnected_is_a_noop():
    """The reserved-slot expansion mechanism: a node added DISCONNECTED (couplings=0, a
    vague slot) is structurally inert -- marginalizing it returns the original net exactly.
    Data (relational deposits) then *activate* the slot by depositing its couplings, which
    is when the expansion becomes real (next test). This is what makes a fixed-d superset
    scan-safe: the inactive slot does nothing until the data wire it in."""
    d = 5
    rng = np.random.default_rng(0)
    M = rng.normal(size=(d, d))
    Pi = jnp.asarray(M @ M.T + d * np.eye(d))            # PD
    h = jnp.asarray(rng.normal(size=d))

    Pi2, h2 = linalg.border(Pi, h, jnp.zeros(d), Pi_diag=4.0, h_new=0.0)
    assert Pi2.shape == (d + 1, d + 1) and h2.shape == (d + 1,)

    keep, drop = jnp.arange(d), jnp.asarray([d])
    Pi_back, h_back = linalg.schur_marginalize(Pi2, h2, keep, drop)
    assert jnp.allclose(Pi_back, Pi, atol=1e-5)
    assert jnp.allclose(h_back, h, atol=1e-5)


def test_expansion_costs_a_real_inversion_reduction_is_cheap():
    """Sanity on the asymmetry the paper turns on: marginalizing the bordered node leaves a
    carry-over fill-in among its neighbours (the Schur residual) when its couplings are
    non-zero -- structure was genuinely added, not free."""
    d = 4
    Pi = jnp.asarray(np.eye(d) * 3.0)
    h = jnp.zeros(d)
    couplings = jnp.asarray([1.0, -1.0, 0.0, 0.0])       # the new node binds nodes 0,1
    Pi2, h2 = linalg.border(Pi, h, couplings, Pi_diag=2.0)
    fill = linalg.carryover_fillin(Pi2, keep=jnp.arange(d), drop=jnp.asarray([d]))
    assert float(jnp.linalg.norm(fill)) > 1e-6           # a real coupling was deposited
