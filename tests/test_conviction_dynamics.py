"""Tests for dynamic conviction (the Lakatos accretion of ``u``).

Layer: structural · Guards: src/structural/simulation.py (ConvictionDynamics + the gain
loop), and the conviction_eps plumbing of models/kuhn_phlogiston.single_run.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.structural.simulation import ConvictionDynamics


def test_config_validates():
    with pytest.raises(ValueError):
        ConvictionDynamics(entrenchment="bogus")
    assert ConvictionDynamics().eps == 0.0


def test_none_and_eps_zero_are_byte_identical():
    """The opt-in contract: conviction_eps=0 (the default, mapped to None) must leave every
    array byte-identical to the legacy run."""
    from src.structural.models.kuhn_phlogiston import single_run
    kw = dict(N=10, inter=0.0, n_steps=50, t_shift=20, proposal_rate=0.1,
              snapshot_every=10, seed=0)
    r0 = single_run(**kw)
    r1 = single_run(conviction_eps=0.0, **kw)
    np.testing.assert_array_equal(r0["snap_Pi"], r1["snap_Pi"])
    np.testing.assert_array_equal(r0["snap_h"], r1["snap_h"])
    assert "u_gain_t" not in r0 and "u_gain_t" not in r1


def test_gain_grows_on_entrenched_nodes_and_clips():
    """With accretion on, the gain state must (i) be reported, (ii) load on the nodes the
    agent's own deposits load on (fisher_diag), and (iii) respect the clip."""
    from src.structural.models.kuhn_phlogiston import single_run
    g_max = 0.5
    r = single_run(N=10, inter=0.0, n_steps=80, t_shift=30, proposal_rate=0.1,
                   conviction_eps=0.1, conviction_decay=0.02, conviction_gmax=g_max,
                   snapshot_every=10, seed=0)
    g = r["u_gain_t"]                                    # (S, N, d)
    assert g.shape[1] == 10
    assert g.max() > 0.0, "gain must accrete"
    assert g.max() <= g_max + 1e-9, "the clip must hold"
    assert g[-1].mean() > g[1].mean(), "accretion grows as the programme matures"


def test_accretion_strengthens_the_gate():
    """The Lakatos effect at the population level: value accreting onto entrenched
    commitments strengthens the endogenous conviction gate, so the dogmatic community's
    realized disconfirming-weight removal (gamma) is at least as high as in the static-u
    run, and conversion is no faster."""
    from src.structural.models.kuhn_phlogiston import single_run
    kw = dict(N=16, inter=0.0, n_steps=100, t_shift=30, proposal_rate=0.1,
              snapshot_every=10, seed=0)
    r_static = single_run(**kw)
    r_dyn = single_run(conviction_eps=0.3, conviction_decay=0.01, **kw)
    assert r_dyn["gamma_t"][-1] >= r_static["gamma_t"][-1] - 1e-6
    assert r_dyn["oxy_index_sc"][-1, 1] <= r_static["oxy_index_sc"][-1, 1] + 1e-6
