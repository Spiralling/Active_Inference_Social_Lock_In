"""Tests for the stratified order-parameter read-outs (plan P2).

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/shells.py
(conservatism_split, residual_disagreement, assign_shells/shell_curves wiring,
agent_conservatism).

These are post-hoc, host-numpy read-offs of a population rollout -- the Eq. (13)
belt/core decomposition m_S(t) and the "residual structural disagreement" measured
quantity. They are never part of the dynamics, so the tests are pure-numpy unit checks
of the metrics plus one rollout smoke test.
"""

from __future__ import annotations

import numpy as np
import jax
import pytest

from src.structural.phlogiston import StructuralConfig
from src.structural import shells, step as S


KEY = jax.random.PRNGKey(0)


# ----------------------------------------------------------------------
# conservatism_split: tau + the honest bimodality number.
# ----------------------------------------------------------------------

def test_conservatism_split_tau_is_median():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    tau, _ = shells.conservatism_split(v)
    assert tau == float(np.median(v))


def test_conservatism_split_separates_bimodal_from_unimodal():
    rng = np.random.default_rng(0)
    unimodal = rng.normal(0.0, 1.0, 4000)
    bimodal = np.concatenate([rng.normal(-3.0, 0.3, 2000),
                              rng.normal(+3.0, 0.3, 2000)])
    _, bc_uni = shells.conservatism_split(unimodal)
    _, bc_bi = shells.conservatism_split(bimodal)
    assert bc_uni < 0.555 < bc_bi          # the classic Sarle cutoff


def test_conservatism_split_constant_is_degenerate():
    tau, bc = shells.conservatism_split(np.full(10, 2.5))
    assert tau == 2.5 and bc == 0.0


# ----------------------------------------------------------------------
# assign_shells + shell_curves: the belt-first/core-last decomposition.
# ----------------------------------------------------------------------

def test_shell_curves_belt_leads_core():
    """A synthetic population where the low-conservatism block reorganises early and the
    high-conservatism block late must yield a belt curve that crosses 1/2 before core --
    the staircase read-out plumbing."""
    N, T = 20, 30
    cons = np.concatenate([np.full(N // 2, 0.1), np.full(N // 2, 5.0)])  # belt, core
    shell_id, labels = shells.assign_shells(cons, n_shells=2, labels=("belt", "core"))
    idx_t = np.zeros((T, N))
    for i in range(N):
        t0 = 3 if cons[i] < 1.0 else 20         # belt ramps early, core late
        idx_t[:, i] = np.clip((np.arange(T) - t0) / 3.0, 0.0, 1.0)
    curves = shells.shell_curves(idx_t, shell_id, labels)
    assert set(curves) == {"belt", "core"}
    belt_cross = int(np.argmax(curves["belt"] >= 0.5))
    core_cross = int(np.argmax(curves["core"] >= 0.5))
    assert belt_cross < core_cross


def test_agent_conservatism_precision_orders_by_prior_mass():
    Pi0 = np.stack([0.5 * np.eye(4), 5.0 * np.eye(4)])      # (2, 4, 4)
    c = shells.agent_conservatism(Pi0, kind="precision")
    assert c[1] > c[0]                                       # stiffer prior = more conservative


# ----------------------------------------------------------------------
# residual_disagreement: the consensus / lock-in signature.
# ----------------------------------------------------------------------

def test_residual_disagreement_identical_agents_is_zero():
    Pi_t = np.tile(np.eye(3), (5, 4, 1, 1))                 # T=5, N=4 identical nets
    assert np.allclose(shells.residual_disagreement(Pi_t), 0.0)


def test_residual_disagreement_falls_when_converging():
    """Agents starting at different precisions and relaxing to a common one must drive
    the dispersion down (consensus), robust to all of them growing together."""
    T, N, d = 12, 4, 3
    Pi_t = np.zeros((T, N, d, d))
    for t in range(T):
        for i in range(N):
            scale = (1.0 + i) - i * (t / (T - 1))           # -> 1.0 for all as t -> T-1
            Pi_t[t, i] = scale * np.eye(d)
    dz = shells.residual_disagreement(Pi_t)
    assert dz[-1] < dz[0] and dz[-1] < 1e-6                  # reaches consensus


def test_residual_disagreement_shape_on_rollout():
    cfg = StructuralConfig(n_agents=12, n_steps=20)
    Pi_t, _ = S.run_trace_net(cfg, S.init_state(cfg, KEY))
    dz = shells.residual_disagreement(Pi_t)
    assert dz.shape == (cfg.n_steps,)
    assert np.all(dz >= 0.0) and np.all(np.isfinite(dz))
