"""Tests for the gamma lock-in phase transition (plan P3).

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/precision.py
(channel_precision_H_stack), src/structural/shells.py (core_stall), and the
gamma -> evidential-lock-in behaviour of the population loop on the relational substrate.

The paper's §6 headline prediction: graded convergence to the truth at low gamma, and a
phase transition into EVIDENTIAL LOCK-IN as gamma (the disconfirming-precision-silenced
fraction, Eq. 9) rises -- honest agents whose truth is reachable nonetheless stall, because
the channel that would refute them has been driven to zero precision. The lock-in is a
genuine bistability (two absorbing states picked out by the initial condition), not slow
relaxation, because a silenced channel deposits exactly zero Fisher.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.config import NetworkConfig
from src.structural.phlogiston import StructuralConfig
from src.structural import step as S, shells, precision as P, phlogiston as ph


KEY = jax.random.PRNGKey(0)


def _cfg(gov, **kw):
    base = dict(n_agents=48, n_steps=120, t_shift=0, observation_operator="relational",
                precision_mode="derived", core_governance=gov,
                network=NetworkConfig(kind="watts_strogatz", mean_degree=4, rewiring_p=0.1))
    base.update(kw)
    return StructuralConfig(**base)


def _final_m(gov, stance, **kw):
    cfg = _cfg(gov, **kw)
    groups = [{"count": cfg.n_agents, "paradigm": "phlogiston", "stance": stance}]
    ms, _ = S.run_trace(cfg, S.init_state(cfg, KEY, groups=groups))
    return float(ms[-1])


# ----------------------------------------------------------------------
# shells.core_stall -- the phase-diagram cell classifier.
# ----------------------------------------------------------------------

def test_core_stall_converged_vs_locked():
    converged = np.linspace(0.0, 1.0, 50)
    locked = np.full(50, 0.08)
    s_c, f_c, p_c = shells.core_stall(converged)
    s_l, f_l, p_l = shells.core_stall(locked)
    assert not s_c and f_c == pytest.approx(1.0)
    assert s_l and p_l < 0.5


# ----------------------------------------------------------------------
# channel_precision_H_stack -- per-agent adaptive gamma on relational rows.
# ----------------------------------------------------------------------

def test_H_stack_zero_governance_is_rho_max():
    cfg = StructuralConfig()
    H = ph.gravimetric_H(cfg)
    Pi = jnp.stack([ph.paradigm_field(cfg).Pi] * 4)         # (4, d, d)
    c = cfg.node_names.index(cfg.core_node)
    rho = P.channel_precision_H_stack(Pi, c, H, core_governance=0.0, rho_max=1.0)
    assert rho.shape == (4, H.shape[0])
    assert jnp.allclose(rho, 1.0)


def test_H_stack_entrenchment_silences_more():
    """A stiffer (higher-precision) agent has a larger coupling-to-core cost, hence a
    smaller rho on the disconfirming channel -- entrenchment self-silences."""
    cfg = StructuralConfig()
    H = ph.gravimetric_H(cfg)
    c = cfg.node_names.index(cfg.core_node)
    base = ph.paradigm_field(cfg).Pi
    Pi = jnp.stack([base, 5.0 * base])                      # agent 1 is 5x more entrenched
    rho = P.channel_precision_H_stack(Pi, c, H, core_governance=50.0, rho_max=1.0)
    # the mass-balance (disconfirming) row is the last; stiffer agent => smaller rho there
    assert float(rho[1, -1]) < float(rho[0, -1])


# ----------------------------------------------------------------------
# The headline behaviour: gamma drives a monostable -> bistable transition.
# ----------------------------------------------------------------------

def test_low_gamma_converges_to_truth():
    """At gamma = 0 the population reaches the reachable truth (world held at oxygen)."""
    assert _final_m(0.0, stance=-1.0) > 0.8


def test_high_gamma_locks_in_and_is_bistable():
    """At large gamma the disconfirming channel is silenced: a population started at the
    phlogiston pole STALLS there, while one started at oxygen stays at oxygen -- two
    absorbing states (bistable lock-in), the truth reachable yet not reached."""
    m_phlog = _final_m(400.0, stance=-1.0)     # start wrong
    m_oxy = _final_m(400.0, stance=+1.0)       # start right
    assert m_phlog < 0.4                        # locked on the wrong paradigm
    assert m_oxy > 0.8                          # stays on the truth
    assert (m_oxy - m_phlog) > 0.4              # the two ICs do not agree => bistable


def test_conservatism_lowers_lock_in_threshold():
    """A stiffer core locks at a lower gamma: at a fixed moderate gamma, raising the
    conservatism (prior precision) scale pushes the population from converged into stalled."""
    m_soft = _final_m(60.0, stance=-1.0, n_agents=48,
                      network=NetworkConfig(kind="watts_strogatz"))
    # rebuild with a stiff population at the same gamma
    cfg = _cfg(60.0)
    groups = [{"count": cfg.n_agents, "paradigm": "phlogiston",
               "stance": cfg.mu_phlog_mass, "prec_scale": 8.0}]
    ms, _ = S.run_trace(cfg, S.init_state(cfg, KEY, groups=groups))
    m_stiff = float(ms[-1])
    assert m_stiff < m_soft                      # stiffer => closer to / into lock-in
