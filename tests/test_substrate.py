"""Tests for the observation-operator substrate migration (plan P1).

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/phlogiston.py
(observation_operator / observation_rows / disagreement_row_mask /
derived_channel_precision), src/structural/precision.py (channel_precision_H),
src/structural/step.py (_transition operator selector), src/structural/shells.py
(edge_trace).

The paper's §3 turns on structure *learning*: the data must be able to move the
paradigm's edges. The legacy substrate observes through ``H_observable`` (one row per
node => DIAGONAL Fisher => the off-diagonal couplings never move -- frozen edges). The
corrected substrate observes through ``gravimetric_H`` (a relational mass-balance row =>
OFF-DIAGONAL Fisher => the belt edges learn). These tests pin two things:

  1. ``observation_operator='node'`` is byte-identical to the legacy behaviour (default),
     so nb18-29 and the rest of the suite are untouched.
  2. ``'relational'`` actually deposits off-diagonal Fisher (the belt edge moves), and the
     gamma machinery (channel_precision_H) is well-defined on it and reduces to the node
     form on direct rows.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.structural.phlogiston import StructuralConfig
from src.structural import step as S
from src.structural import phlogiston as ph
from src.structural import precision as P
from src.structural import shells


KEY = jax.random.PRNGKey(0)
EDGE = ("calx_heavier_than_metal", "mass_change_sign")   # a belt mass-balance coupling


# ----------------------------------------------------------------------
# 1. node mode is the default and byte-identical to the legacy operator.
# ----------------------------------------------------------------------

def test_default_is_node_operator():
    assert StructuralConfig().observation_operator == "node"


def test_node_operator_is_H_observable():
    cfg = StructuralConfig()
    assert jnp.array_equal(ph.observation_operator(cfg), ph.H_observable(cfg))
    assert ph.observation_rows(cfg) == ph.measured_nodes(cfg)


def test_node_disagreement_mask_byte_identical():
    """The operator-aware mask must reproduce the legacy hand-computed mask in node
    mode (the regression anchor for attention_weights / run_trace grav read-outs)."""
    cfg = StructuralConfig()
    legacy = jnp.asarray([1.0 if n in ph.DISAGREEMENT_NODES else 0.0
                          for n in ph.measured_nodes(cfg)])
    assert jnp.array_equal(ph.disagreement_row_mask(cfg), legacy)


@pytest.mark.parametrize("mode", ["heuristic", "derived", "derived_live"])
def test_node_run_trace_unchanged_shape_and_determinism(mode):
    cfg = StructuralConfig(n_agents=10, n_steps=12, precision_mode=mode,
                           core_governance=30.0, experiment_bias=0.4)
    ms1, gw1 = S.run_trace(cfg, S.init_state(cfg, KEY))
    ms2, gw2 = S.run_trace(cfg, S.init_state(cfg, KEY))
    assert ms1.shape == (cfg.n_steps,)
    assert np.allclose(np.asarray(ms1), np.asarray(ms2))   # pure => deterministic
    assert np.allclose(np.asarray(gw1), np.asarray(gw2))


# ----------------------------------------------------------------------
# 2. relational mode: extra mass-balance row + it is flagged disconfirming.
# ----------------------------------------------------------------------

def _rel(cfg, **kw):
    return dataclasses.replace(cfg, observation_operator="relational", **kw)


def test_relational_operator_is_gravimetric_with_extra_row():
    cfg = StructuralConfig()
    H_node = ph.observation_operator(cfg)
    H_rel = ph.observation_operator(_rel(cfg))
    assert H_rel.shape[0] == H_node.shape[0] + 1            # one extra mass-balance row
    assert jnp.array_equal(H_rel, ph.gravimetric_H(cfg))
    assert ph.observation_rows(_rel(cfg)) == ph.gravimetric_rows(cfg)


def test_relational_mass_balance_row_is_disconfirming():
    cfg = _rel(StructuralConfig())
    mask = ph.disagreement_row_mask(cfg)
    rows = ph.observation_rows(cfg)
    assert mask.shape[0] == len(rows)
    # the mass-balance row (last, a combination not a bare node) is flagged 1
    assert float(mask[-1]) == 1.0
    assert rows[-1] not in cfg.node_names


def test_bad_operator_raises():
    with pytest.raises(ValueError):
        ph.observation_operator(dataclasses.replace(StructuralConfig(),
                                                    observation_operator="bogus"))


# ----------------------------------------------------------------------
# 3. the headline P1 claim: relational deposits OFF-DIAGONAL Fisher -> edges move,
#    node mode leaves the same coupling frozen.
# ----------------------------------------------------------------------

def test_relational_moves_edges_node_freezes_them():
    cfg = StructuralConfig(n_agents=8, n_steps=20)
    Pi_node, _ = S.run_trace_net(cfg, S.init_state(cfg, KEY))
    Pi_rel, _ = S.run_trace_net(_rel(cfg), S.init_state(_rel(cfg), KEY))

    node_edge = shells.edge_trace(Pi_node, cfg, [EDGE])[EDGE]
    rel_edge = shells.edge_trace(Pi_rel, cfg, [EDGE])[EDGE]

    # the coupling is absent in the prior (hub couples to each, but not calx<->mass directly)
    assert abs(node_edge[0]) < 1e-6
    # node operator: diagonal Fisher only -> the edge stays pinned at the prior (~0)
    assert np.max(np.abs(node_edge)) < 1e-6
    # relational operator: off-diagonal Fisher accumulates -> the edge moves substantially
    assert np.abs(rel_edge[-1]) > 0.5
    # and it grows monotonically in magnitude (every step deposits the same off-diagonal)
    assert np.abs(rel_edge[-1]) > np.abs(rel_edge[len(rel_edge) // 2]) > np.abs(rel_edge[1])


# ----------------------------------------------------------------------
# 4. gamma machinery on the relational substrate (channel_precision_H).
# ----------------------------------------------------------------------

def test_channel_precision_H_reduces_to_node_form_on_direct_rows():
    """The relational cost (H[k].Pi[:,c])^2/(H[k].Pi.H[k]) must equal the node cost
    Pi[c,v]^2/Pi[v,v] when H is the unit-row operator -- so the node and relational
    derivations agree on their shared direct rows."""
    cfg = StructuralConfig(core_governance=25.0)
    net = ph.paradigm_field(cfg)
    H_node = ph.H_observable(cfg)
    rho_H = P.channel_precision_H(net, cfg.core_node, H_node, cfg.core_governance,
                                  cfg.rho_max)
    rho_node = P.channel_precision(net, cfg.core_node, ph.measured_nodes(cfg),
                                   cfg.core_governance, cfg.rho_max, "carryover")
    assert jnp.allclose(rho_H, rho_node, atol=1e-6)


def test_derived_channel_precision_relational_shape_and_governance():
    cfg = _rel(StructuralConfig(core_governance=50.0))
    rho = ph.derived_channel_precision(cfg)
    assert rho.shape[0] == len(ph.observation_rows(cfg))   # one rho per relational row
    # governance silences the mass-balance channel: rho < rho_max on the disconfirming row
    assert float(rho[-1]) < cfg.rho_max
    # at g=0 it is the unbiased deposit (all rho_max)
    rho0 = ph.derived_channel_precision(_rel(StructuralConfig(core_governance=0.0)))
    assert jnp.allclose(rho0, 1.0, atol=1e-6)


@pytest.mark.parametrize("mode", ["heuristic", "derived", "derived_live"])
def test_relational_runs_in_supported_modes(mode):
    cfg = _rel(StructuralConfig(n_agents=6, n_steps=10, precision_mode=mode,
                                core_governance=40.0))
    out = S.run_final(cfg, S.init_state(cfg, KEY))
    assert out.Pi.shape == (cfg.n_agents, len(cfg.node_names), len(cfg.node_names))
    assert jnp.all(jnp.isfinite(out.Pi))


def test_relational_efe_mode_raises():
    """'efe' is the one precision mode not yet wired for the relational operator (its
    drives index measured_nodes); it must fail loudly, not with a shape error."""
    cfg = _rel(StructuralConfig(n_agents=4, n_steps=4, precision_mode="efe"))
    with pytest.raises(NotImplementedError):
        S.run_final(cfg, S.init_state(cfg, KEY))
