"""Tests for the conviction field U = T u and the value-tilt (plan P4).

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/phlogiston.py
(conviction_u, conviction_field, balanced_lambda) and the conviction-tilt branch of
step._transition.

The conviction field is the paper's SECOND field: the same propagation operator T that
gives carry-over kappa = T 1 also carries the intrinsic utility u, giving U = T u. The
motivated posterior q_lambda prop p(s|o) e^{lambda U(s)} is, for a linear utility on a
Gaussian, one shift of the potential h <- h + lambda U. The belief-utility knob is JUST a
balance: lambda ~ balanced_lambda so neither the evidence nor the conviction dominates.
These tests pin (a) the default (no tilt) is byte-identical, (b) the field has the right
sign/shape, (c) the tilt moves belief toward the valued paradigm, monotonically.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.config import NetworkConfig
from src.structural.phlogiston import StructuralConfig
from src.structural import step as S, phlogiston as ph


KEY = jax.random.PRNGKey(0)
NET = NetworkConfig(kind="watts_strogatz", mean_degree=4, rewiring_p=0.1)


def _cfg(**kw):
    base = dict(n_agents=40, n_steps=120, t_shift=0, observation_operator="relational",
                precision_mode="derived", core_governance=0.0, network=NET)
    base.update(kw)
    return StructuralConfig(**base)


def _final_m(cfg):
    g = [{"count": cfg.n_agents, "paradigm": "phlogiston", "stance": cfg.mu_phlog_mass}]
    ms, _ = S.run_trace(cfg, S.init_state(cfg, KEY, groups=g))
    return float(ms[-1])


# ----------------------------------------------------------------------
# conviction_u: signed value toward a paradigm.
# ----------------------------------------------------------------------

def test_conviction_u_signs():
    cfg = StructuralConfig()
    idx = {n: i for i, n in enumerate(cfg.node_names)}
    up = ph.conviction_u(cfg, "phlogiston")
    uo = ph.conviction_u(cfg, "oxygen")
    un = ph.conviction_u(cfg, "neutral")
    for n in ph.DISAGREEMENT_NODES:
        assert float(up[idx[n]]) < 0 and float(uo[idx[n]]) > 0
    assert jnp.allclose(un, 0.0)


# ----------------------------------------------------------------------
# conviction_field: shape + propagation reuse of dual_field.
# ----------------------------------------------------------------------

def test_conviction_field_shape_and_nonzero():
    cfg = StructuralConfig()
    net = ph.phlogiston_prior(cfg)
    u = ph.conviction_u(cfg, "phlogiston")
    U = ph.conviction_field(net.Pi[None], net.h[None], cfg.node_names, u, 0.5)
    assert U.shape == (1, len(cfg.node_names))
    idx = cfg.node_names.index("calx_heavier_than_metal")
    assert float(U[0, idx]) < 0.0                # value toward phlogiston (calx lighter)


def test_balanced_lambda_positive_finite():
    lam = ph.balanced_lambda(_cfg(), toward="phlogiston")
    assert np.isfinite(lam) and lam > 0.0


# ----------------------------------------------------------------------
# the tilt: default is byte-identical; nonzero tilt moves belief by value.
# ----------------------------------------------------------------------

def test_zero_tilt_is_pure_evidence_default():
    """conviction_tilt=0.0 (the default) leaves the update untouched: a run with the field
    explicitly 0 equals a run that never set it (the guard skips the tilt code)."""
    cfg_default = _cfg()
    cfg_zero = _cfg(conviction_tilt=0.0, conviction_toward="phlogiston")
    assert _final_m(cfg_default) == pytest.approx(_final_m(cfg_zero), abs=1e-6)


def test_tilt_toward_phlogiston_drags_belief_down():
    """A conviction toward phlogiston (the wrong paradigm) pulls the final belief away from
    the oxygen truth, below the pure-evidence value."""
    lam = ph.balanced_lambda(_cfg(), toward="phlogiston")
    m_evidence = _final_m(_cfg(conviction_tilt=0.0))
    m_tilted = _final_m(_cfg(conviction_tilt=lam, conviction_toward="phlogiston"))
    assert m_tilted < m_evidence - 0.1


def test_tilt_is_monotone_in_strength():
    """Increasing the tilt toward phlogiston monotonically lowers the final oxygen index --
    stronger conviction, more motivated pull (eventually value-driven lock-in)."""
    lam = ph.balanced_lambda(_cfg(), toward="phlogiston")
    ms = [_final_m(_cfg(conviction_tilt=f * lam, conviction_toward="phlogiston"))
          for f in (0.0, 0.5, 1.0, 2.0)]
    assert all(ms[i + 1] <= ms[i] + 1e-6 for i in range(len(ms) - 1))
    assert ms[0] - ms[-1] > 0.3                  # the field has a real, large effect


def test_tilt_direction_matters():
    """Tilting toward oxygen (the truth) does not drag belief below the evidence-only run;
    tilting toward phlogiston does -- the sign of the conviction sets the direction."""
    lam = ph.balanced_lambda(_cfg(), toward="phlogiston")
    m_evidence = _final_m(_cfg(conviction_tilt=0.0))
    m_oxy = _final_m(_cfg(conviction_tilt=lam, conviction_toward="oxygen"))
    m_phlog = _final_m(_cfg(conviction_tilt=lam, conviction_toward="phlogiston"))
    assert m_phlog < m_evidence
    assert m_oxy >= m_evidence - 0.05
