"""Tests for the wake-then-prune cycle and the pre-crisis expansion check.

Layer: structural · Guards: the ``wake_credit`` / ``credit_decay`` / ``allow_precrisis_apply``
/ ``precrisis_check`` machinery of ``models/kuhn_phlogiston.make_revolution_hook``.

The cycle the appendix names as the principled false-discovery remedy: a wake is accepted on
the full ledger (``Delta F + Delta G > 0`` -- curiosity pays for entertaining the structure
on credit), and the credit EXPIRES -- the k-th audit's verdict is
``Delta F_now + Delta G_now * decay^k``, so a genuine hub must start paying its own way in
evidence before the credit runs out, while a noise wake is re-pinned exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.structural.models.kuhn_phlogiston import PIN, single_run

KW = dict(N=12, inter=0.0, n_steps=200, snapshot_every=20, proposal_rate=0.15,
          wake_credit=True, prune_grace=15, prune_patience=5, credit_decay=0.85)


def test_null_world_credit_wakes_are_pruned_back():
    """No regime flip: the residual is pure noise, yet the epistemic credit wakes hubs
    (curiosity entertains structures). The expiring-credit audit must prune essentially all
    of them back -- the controlled false-discovery rate."""
    r = single_run(t_shift=10_000, allow_precrisis_apply=True, seed=0, **KW)
    es, ps = r["expand_step"], r["prune_back_step"]
    woke = es >= 0
    assert woke.mean() > 0.5, "the credit should wake hubs even on noise"
    assert (ps[woke] >= 0).all(), "every noise wake must be pruned back"
    assert (ps[woke] > es[woke]).all()


def test_repin_restores_the_exact_pinned_form():
    """A pruned-back agent's slot must return to the original pinned form (zero row/col,
    PIN diagonal, zero potential) in both its net and its anchor -- so the 14-edge crisis
    ledger read-out is valid again."""
    r = single_run(t_shift=10_000, allow_precrisis_apply=True, seed=0, **KW)
    ps = r["prune_back_step"]
    woke_and_pruned = (r["expand_step"] >= 0) & (ps >= 0)
    assert woke_and_pruned.any()
    Pi_final = r["snap_Pi"][-1]                       # (N, 11, 11)
    h_final = r["snap_h"][-1]
    d10 = Pi_final.shape[-1] - 1
    # agents whose final state is re-pinned (no later re-wake held at the end)
    eff = r["oxy_coupling_tn"][-1]
    for i in np.nonzero(woke_and_pruned & (eff < 1e-6))[0]:
        assert Pi_final[i, d10, d10] == pytest.approx(PIN)
        np.testing.assert_allclose(Pi_final[i, d10, :d10], 0.0, atol=1e-10)
        assert abs(h_final[i, d10]) < 1e-8


def test_flip_world_wakes_survive_the_audit():
    """Genuine structure: the post-crisis residual loads on the hub, so the wake earns its
    keep before the credit expires -- no prune-backs."""
    r = single_run(t_shift=30, seed=0, **KW)
    o = r["community"] == 0
    es, ps = r["expand_step"][o], r["prune_back_step"][o]
    assert (es >= 0).mean() > 0.8
    assert (ps >= 0).mean() == 0.0
    # and the audit trace itself is positive once the consolidation steps pass
    tr = r["hub_dF_prune_tn"][:, o]
    last = tr[np.isfinite(tr)]
    assert last[-20:].min() > 0.0


def test_precrisis_check_is_pure_measurement_and_rejects():
    """The pre-crisis discovery check: the ledger must reject expansion before the agent's
    own crisis (the residual is still absorbed by the intact paradigm), and switching the
    measurement on must not perturb the dynamics."""
    kw = dict(N=12, inter=0.0, n_steps=120, t_shift=30, proposal_rate=0.1,
              snapshot_every=20)
    r_off = single_run(seed=0, **kw)
    r_on = single_run(precrisis_check=True, seed=0, **kw)
    np.testing.assert_array_equal(r_off["crisis_step"], r_on["crisis_step"])
    np.testing.assert_array_equal(r_off["expand_step"], r_on["expand_step"])
    np.testing.assert_allclose(r_off["snap_Pi"][-1], r_on["snap_Pi"][-1])
    pre = r_on["precrisis_dF_tn"]
    assert np.isfinite(pre).any(), "pre-crisis agents must be measured"
    frac_pos = (pre > 0).sum() / np.isfinite(pre).sum()
    assert frac_pos < 1e-3 and np.nanmax(pre) < 0.05, \
        "the ledger must reject pre-crisis wakes (rare crossings marginal at most)"
