"""Tests for the inferred-reliability channel gate in the Scenario engine.

Layer: structural (THE PAPER OBJECT) · Guards: simulation.run_simulation(reliability_nu),
simulation._deposit_all_gated.

The port that converges the two engines: ``reliability.channel_reliability`` (the
Student-t scale-mixture gate on per-channel surprise) was previously wired only into
the step.py scan engine (``cfg.reliability_nu``); ``run_simulation(reliability_nu=...)``
now wires the SAME shared function. Defaults must stay byte-identical, ``nu -> inf``
must recover the ungated dynamics, and a world that conflicts with the agents' beliefs
must be discounted relative to one that matches them.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from src.structural import graphs
from src.structural.phlogiston import StructuralConfig
from src.structural.scenarios import phlogiston_scenario
from src.structural.simulation import run_simulation, AgentSpec


def _run(scn, n=5, seed=0, stiff=None, **kw):
    spec = AgentSpec(w_obs=jnp.ones((n, scn.m)), lam=jnp.full((n,), 0.2),
                     precision_scale=(None if stiff is None
                                      else jnp.full((n,), float(stiff))))
    return run_simulation(scn, graphs.complete(n), spec, forgetting=1.0,
                          snapshot_every=4, seed=seed, **kw)


def test_defaults_byte_identical():
    scn = phlogiston_scenario(StructuralConfig(n_steps=10, t_shift=4))
    r0 = _run(scn)
    r1 = _run(scn, reliability_nu=None)
    assert np.array_equal(r0["snap_Pi"], r1["snap_Pi"])
    assert np.array_equal(r0["snap_h"], r1["snap_h"])


def test_nu_inf_recovers_ungated():
    """lambda_k = (nu+1)/(nu+z^2) -> 1 as nu -> inf, so a huge nu must reproduce the
    ungated dynamics to numerical tolerance (same seed => same observations)."""
    scn = phlogiston_scenario(StructuralConfig(n_steps=10, t_shift=4))
    r0 = _run(scn)
    r1 = _run(scn, reliability_nu=1e9)
    assert np.allclose(r0["snap_Pi"], r1["snap_Pi"], rtol=1e-4, atol=1e-4)
    assert np.allclose(r0["snap_h"], r1["snap_h"], rtol=1e-4, atol=1e-3)


def test_conflicting_world_is_discounted():
    """The gate reads SURPRISE, channel by channel. Read the accumulated deposit on
    the DISAGREEMENT (mass-law) nodes -- the only nodes whose truth differs between
    the regimes: when the world matches the agents' prior (phlogiston regime
    throughout) those channels are on-prediction and keep ~full weight; when it
    conflicts from t=0 (oxygen regime) the same channels are surprising and the
    Student-t gate discounts their deposit. STIFF priors (precision_scale=100) keep
    the belief from adapting within the horizon, so the conflict stays surprising:
    the gate reads PER-STEP surprise, and once a belief has adapted, a sustained
    consistent signal is correctly no longer an outlier (persistent-conflict memory
    is the deposit-EMA machinery of experiments/abc_conflict.py, deliberately not
    this gate). Ratios to the ungated run -- robust to the overall deposit scale."""
    from src.structural.phlogiston import DISAGREEMENT_NODES

    def gain(t_shift, nu):
        scn = phlogiston_scenario(StructuralConfig(n_steps=16, t_shift=t_shift))
        ix = np.asarray([scn.names.index(n) for n in DISAGREEMENT_NODES])
        r = _run(scn, stiff=100.0, **({} if nu is None else dict(reliability_nu=nu)))
        dep = r["snap_Pi"][-1] - 100.0 * np.asarray(scn.Pi0[0])[None]
        return dep[:, ix, ix].mean()

    match = gain(t_shift=1000, nu=1.0) / gain(t_shift=1000, nu=None)
    conflict = gain(t_shift=0, nu=1.0) / gain(t_shift=0, nu=None)
    # measured (seed 0, deterministic): match ~ 1.21 (unsurprising channels are mildly
    # UP-weighted -- the (nu+1)/nu ceiling), conflict ~ 0.66 -- a 0.55x relative discount.
    assert match > 0.9, f"on-prediction channels should keep ~full weight ({match:.2f})"
    assert conflict < 0.65 * match, \
        f"conflicting channels should be clearly discounted ({conflict:.2f} vs {match:.2f})"


def test_gated_smoke_finite():
    scn = phlogiston_scenario(StructuralConfig(n_steps=12, t_shift=4))
    r = _run(scn, reliability_nu=0.5)
    assert np.isfinite(r["snap_Pi"]).all() and np.isfinite(r["snap_h"]).all()
