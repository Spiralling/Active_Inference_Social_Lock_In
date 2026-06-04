"""Tests for the action layer: the three structural moves on one ledger.

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/action.py
(inverse-Schur proposal, wake/score/select), built on bmr.py (Savage-Dickey) and
belief.border (the reserved-slot expansion of test_structure_learning.py).

The claims pinned here:
  * the residual block's leading eigenpair LOCALIZES a planted common cause (Eq. 2 read
    backwards) and its strength tracks the floor height (the trigger);
  * waking a hub earns model evidence (delta_F > 0) iff the data carry a residual that
    LOADS on the hub, and is declined when the neighbours are uncoupled (no floor);
  * a strong conviction tilt VETOES an evidence-favoured wake (the lock-in mechanism);
  * select_action arbitrates expand / reduce / sample.

The faithful fixture marginalizes a true hidden hub out of a 4-node joint, so the observed
3-node net carries a residual whose means AND couplings are consistent (a real common cause,
not a synthetic mismatch) -- the only setting in which "wake the hub" is a well-posed move.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from src.structural import action, bmr
from src.structural.belief import GaussianBeliefNet


def _common_cause_obs(g: float = 1.2, mu_z: float = 2.0):
    """Observed 3-node net after marginalizing a hidden hub ``z`` that drives ``a, b, c``
    (SEM ``x = A x + ε``, children load on ``z`` with weight ``g``, hub mean ``mu_z``).
    The marginal carries negative precision off-diagonals (the common-cause fingerprint)
    and means pulled along ``[+,+,+]`` -- a real residual to wake a hub into."""
    A = np.zeros((4, 4))
    A[0, 3] = A[1, 3] = A[2, 3] = g
    I = np.eye(4)
    Pi = (I - A).T @ I @ (I - A)
    mu = np.array([g * mu_z, g * mu_z, g * mu_z, mu_z])
    net = GaussianBeliefNet(Pi=jnp.asarray(Pi), h=jnp.asarray(Pi @ mu),
                            names=("a", "b", "c", "z"))
    return bmr.schur_marginalize(net, ("z",))


def _diag_prior(diag, names=("a", "b", "c")):
    d = len(names)
    return GaussianBeliefNet(Pi=jnp.asarray(np.diag(np.full(d, diag))),
                             h=jnp.zeros(d), names=tuple(names))


# ----------------------------------------------------------------------
# 1. The inverse-Schur proposal localizes a planted common cause.
# ----------------------------------------------------------------------

def test_proposal_recovers_common_cause_pattern():
    obs3 = _common_cause_obs()
    prop = action.propose_hub(obs3, ("a", "b", "c"))
    v = np.asarray(prop.pattern)
    assert prop.strength > 0.0                              # a floor is present
    # the leading eigenvector is the all-same-sign common-cause direction (Eq. 2).
    assert np.allclose(np.abs(v), np.abs(v[0]), atol=1e-3)  # equal weight on a,b,c
    assert v[0] * v[1] > 0 and v[1] * v[2] > 0              # same sign across the children


def test_strength_tracks_floor_height_and_vanishes_without_one():
    weak = action.propose_hub(_common_cause_obs(g=0.6), ("a", "b", "c"))
    strong = action.propose_hub(_common_cause_obs(g=1.4), ("a", "b", "c"))
    assert strong.strength > weak.strength > 0.0           # taller floor => bigger trigger
    # diagonal (independent) net: no induced coupling => no hub wants to be drawn.
    flat = action.propose_hub(_diag_prior(2.0), ("a", "b", "c"))
    assert flat.strength < 1e-6


# ----------------------------------------------------------------------
# 2. Wake earns evidence iff the data carry a residual that loads on the hub.
# ----------------------------------------------------------------------

def test_wake_favoured_on_real_residual_declined_without_one():
    obs3 = _common_cause_obs()
    prior = _diag_prior(float(np.diag(np.asarray(obs3.Pi))[0]))   # parent holds them indep.
    post = GaussianBeliefNet(Pi=obs3.Pi, h=obs3.h, names=("a", "b", "c"))

    prop = action.propose_hub(post, ("a", "b", "c"))
    s = action.expansion_score(post, prior, prop, "z")
    assert s.delta_F > 0.0 and s.delta_G >= 0.0            # the residual loads on the hub
    assert s.accept                                        # => WAKE

    # control: an independent posterior has no residual -> wire nothing, decline.
    post_ind = GaussianBeliefNet(Pi=jnp.asarray(np.diag([2., 2., 2.])),
                                 h=jnp.asarray([2., 2., 2.]), names=("a", "b", "c"))
    prop0 = action.propose_hub(post_ind, ("a", "b", "c"))
    s0 = action.expansion_score(post_ind, prior, prop0, "z")
    assert prop0.strength < 1e-6
    assert not s0.accept                                   # nothing to wake


def test_wake_keeps_the_bordered_net_positive_definite():
    """The PD-safe coupling scale: the woken (d+1) joint must stay a proper Gaussian, else
    its log-evidence is meaningless. A hub wired at the default scale keeps Π PD."""
    obs3 = _common_cause_obs(g=1.4)
    post = GaussianBeliefNet(Pi=obs3.Pi, h=obs3.h, names=("a", "b", "c"))
    prop = action.propose_hub(post, ("a", "b", "c"))
    woken = action.wake_hub(post, prop, "z", hub_self_prec=2.0)
    eigvals = np.linalg.eigvalsh(np.asarray(woken.Pi))
    assert eigvals.min() > 0.0                              # bordered net stays PD


# ----------------------------------------------------------------------
# 3. The conviction veto: a tilt rejects an evidence-favoured wake.
# ----------------------------------------------------------------------

def test_conviction_tilt_vetoes_an_evidence_favoured_wake():
    obs3 = _common_cause_obs()
    prior = _diag_prior(float(np.diag(np.asarray(obs3.Pi))[0]))
    post = GaussianBeliefNet(Pi=obs3.Pi, h=obs3.h, names=("a", "b", "c"))
    prop = action.propose_hub(post, ("a", "b", "c"))

    base = action.expansion_score(post, prior, prop, "z")
    assert base.accept                                     # evidence favours the wake

    # a utility opposing the common-cause mean direction makes waking value-negative.
    u = jnp.asarray([-1., -1., -1., 0.])
    probe = action.expansion_score(post, prior, prop, "z", u_full=u, tilt=1.0)
    u_full = u if probe.delta_U < 0 else -u
    dU = abs(probe.delta_U)
    assert dU > 1e-6                                        # the wake does move value

    big_tilt = 2.0 * base.score / dU                       # enough to overcome ΔF + ΔG
    vetoed = action.expansion_score(post, prior, prop, "z", u_full=u_full, tilt=big_tilt)
    assert vetoed.delta_F > 0.0                            # evidence STILL favours it
    assert not vetoed.accept                               # but conviction vetoes the wake


# ----------------------------------------------------------------------
# 4. Arbitration over the three moves.
# ----------------------------------------------------------------------

def test_select_action_expands_reduces_samples():
    obs3 = _common_cause_obs()
    diag0 = float(np.diag(np.asarray(obs3.Pi))[0])

    # EXPAND: a real common-cause residual the diagonal parent cannot account for.
    prior = _diag_prior(diag0)
    post = GaussianBeliefNet(Pi=obs3.Pi, h=obs3.h, names=("a", "b", "c"))
    out = action.select_action(post, prior, ("a", "b", "c"), "z",
                               prune_edges=(("a", "b"),))
    assert out["action"] == "expand"

    # REDUCE: the prior asserts an a-b edge the (independent) posterior does not support;
    # no residual coupling present to expand into.
    prior_e = GaussianBeliefNet(
        Pi=jnp.asarray(np.array([[3., 0.9, 0.], [0.9, 3., 0.], [0., 0., 3.]])),
        h=jnp.zeros(3), names=("a", "b", "c"))
    post_e = GaussianBeliefNet(Pi=jnp.asarray(np.diag([4., 4., 4.])),
                               h=jnp.asarray([1., -1., 0.]), names=("a", "b", "c"))
    out_e = action.select_action(post_e, prior_e, ("a", "b", "c"), "z",
                                 prune_edges=(("a", "b"),))
    assert out_e["action"] == "reduce"

    # SAMPLE: no residual to expand into, no supported edge to prune.
    prior_s = _diag_prior(3.0)
    post_s = GaussianBeliefNet(Pi=jnp.asarray(np.diag([4., 4., 4.])),
                               h=jnp.asarray([0.5, 0.5, 0.5]), names=("a", "b", "c"))
    out_s = action.select_action(post_s, prior_s, ("a", "b", "c"), "z",
                                 prune_edges=(("b", "c"),))
    assert out_s["action"] == "sample"
