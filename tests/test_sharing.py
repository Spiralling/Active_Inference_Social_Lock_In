"""Tests for the sharing-mode axis: share posteriors vs raw observations (plan P6).

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/step.py
(_fuse_then_observe / _observe_pooled / _transition dispatch), cfg.sharing_mode.

The paper keeps explicit the axis of WHAT a peer transmits across the trust graph: a whole
belief (a *conclusion*, prior bias and all) or just its *raw observations* (signal). "Sharing
conclusions propagates bias along with signal." These tests pin (a) 'posterior' (the default)
is byte-identical to the original dynamics, (b) 'observation' runs, and (c) a stubborn biased
bloc contaminates the rest ONLY when conclusions are shared.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.structural.phlogiston import StructuralConfig
from src.structural import step as S, graphs as G


KEY = jax.random.PRNGKey(0)


def test_posterior_is_default_and_byte_identical():
    assert StructuralConfig().sharing_mode == "posterior"
    cfg_def = StructuralConfig(n_agents=10, n_steps=14, observation_operator="relational")
    cfg_post = dataclasses.replace(cfg_def, sharing_mode="posterior")
    ms_d, _ = S.run_trace(cfg_def, S.init_state(cfg_def, KEY))
    ms_p, _ = S.run_trace(cfg_post, S.init_state(cfg_post, KEY))
    assert np.allclose(np.asarray(ms_d), np.asarray(ms_p))


def test_observation_mode_runs_finite():
    cfg = dataclasses.replace(StructuralConfig(), n_agents=8, n_steps=12,
                              observation_operator="relational", sharing_mode="observation")
    out = S.run_final(cfg, S.init_state(cfg, KEY))
    assert out.Pi.shape == (8, len(cfg.node_names), len(cfg.node_names))
    assert jnp.all(jnp.isfinite(out.Pi)) and jnp.all(jnp.isfinite(out.h))


def _seekers_final(mode, n_steps=70):
    """Final belief of the truth-seekers when wired (complete graph, world held at the
    oxygen truth) to a bloc of stubborn high-precision phlogiston zealots."""
    g = G.complete(30)
    cfg = dataclasses.replace(StructuralConfig(), n_agents=30, n_steps=n_steps, t_shift=0,
                              observation_operator="relational", precision_mode="derived",
                              sharing_mode=mode)
    groups = [{"count": 15, "paradigm": "phlogiston", "stance": cfg.mu_phlog_mass, "prec_scale": 300.0},
              {"count": 15, "paradigm": "phlogiston", "stance": cfg.mu_phlog_mass, "prec_scale": 1.0}]
    st = S.init_state(cfg, KEY, groups=groups, W_override=g.trust_W())
    return float(np.asarray(S.run_trace_index(cfg, st))[:, 15:].mean(axis=1)[-1])


def test_bias_propagates_only_when_sharing_conclusions():
    m_obs = _seekers_final("observation")
    m_post = _seekers_final("posterior")
    assert m_obs > 0.8                      # observation: seekers reach the truth (signal only)
    assert m_post < m_obs - 0.2             # posterior: the zealots' prior bias drags them down
