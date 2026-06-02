"""Tests for the structural Gaussian Bayes-net model (src/structural/).

Two tiers, mirroring the continuous-substrate and POMDP suites:
  - Tier-1 algebra: the closed-form belief operations, checked against the
    worked numbers in ``notes/bmr_feynman_linear_algebra.tex`` and against
    brute-force numpy/scipy.
  - Tier-2 BMR / dynamics: the closed-form Bayes factor, the evidence-drift
    re-ordering across the regime shift, and the purity of the multi-agent step.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.structural.belief import (
    GaussianBeliefNet, vague_prior, add_fisher, combine, border,
)
from src.structural.bmr import (
    log_evidence, schur_marginalize, condition, carryover, bmr_prune,
    prune_node_prior, prune_edge_prior,
)
from src.structural import phlogiston as ph
from src.structural.agent import evidence_race, evidence_race_weighted
from src.structural import step as S
from src.structural import observables as obs
from src.structural import precision as P
from src.structural.world import sample_o, fisher_deposit, fisher_deposit_weighted


def _rand_spd(d, seed=0, jitter=1.0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((d, d))
    return M @ M.T + jitter * np.eye(d)


# ----------------------------------------------------------------------
# Tier-1: algebra
# ----------------------------------------------------------------------

@pytest.mark.parametrize("d", [1, 2, 3])
def test_log_evidence_matches_numpy(d):
    P = _rand_spd(d, seed=d)
    h = np.random.default_rng(d + 7).standard_normal(d)
    net = GaussianBeliefNet(Pi=jnp.asarray(P), h=jnp.asarray(h),
                            names=tuple(f"n{i}" for i in range(d)))
    ref = 0.5 * (h @ np.linalg.solve(P, h) - np.linalg.slogdet(P)[1]
                 + d * np.log(2 * np.pi))
    assert np.isclose(float(log_evidence(net)), ref, atol=1e-4)


def test_log_evidence_matches_brute_force_integral():
    # 1-D: Z = int exp(-1/2 Pi x^2 + h x) dx, compared to the closed form.
    Pi, h = 2.0, 0.7
    xs = np.linspace(-40, 40, 400001)
    integrand = np.exp(-0.5 * Pi * xs**2 + h * xs)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapz removed in numpy 2.x
    logZ_num = np.log(trapezoid(integrand, xs))
    net = GaussianBeliefNet(Pi=jnp.asarray([[Pi]]), h=jnp.asarray([h]),
                            names=("x",))
    assert np.isclose(float(log_evidence(net)), logZ_num, atol=1e-3)


def test_schur_star_matches_note_numbers():
    # note Fig. star / Appendix A: [[2,0,1],[0,2,1],[1,1,2]] drop hub -> [[1.5,-0.5],[-0.5,1.5]]
    net = GaussianBeliefNet(Pi=jnp.asarray([[2., 0, 1], [0, 2, 1], [1, 1, 2]]),
                            h=jnp.zeros(3), names=("x1", "x2", "b"))
    m = schur_marginalize(net, ("b",))
    assert m.names == ("x1", "x2")
    assert np.allclose(np.asarray(m.Pi),
                       np.array([[1.5, -0.5], [-0.5, 1.5]]), atol=1e-9)


def test_border_then_condition_is_identity():
    # condition (delete the row) is the exact adjoint of border.
    base = GaussianBeliefNet(Pi=jnp.asarray(_rand_spd(3, seed=2)),
                             h=jnp.asarray([0.3, -0.1, 0.5]),
                             names=("a", "b", "c"))
    bd = border(base, "new", Pi_diag=3.0, couplings=jnp.asarray([0.4, -0.2, 0.1]),
                h_new=0.9)
    back = condition(bd, ("new",))
    assert back.names == base.names
    assert np.allclose(np.asarray(back.Pi), np.asarray(base.Pi), atol=1e-8)
    assert np.allclose(np.asarray(back.h), np.asarray(base.h), atol=1e-8)


def test_border_leaves_incumbent_block_untouched():
    base = GaussianBeliefNet(Pi=jnp.asarray(_rand_spd(3, seed=5)),
                             h=jnp.asarray([1.0, 2.0, 3.0]), names=("a", "b", "c"))
    bd = border(base, "new", Pi_diag=2.0, couplings=jnp.asarray([0.5, 0.5, 0.5]))
    assert np.allclose(np.asarray(bd.Pi[:3, :3]), np.asarray(base.Pi))
    assert np.allclose(np.asarray(bd.h[:3]), np.asarray(base.h))


def test_border_then_schur_yields_carryover():
    # marginalizing the just-added node leaves Pi_aa - c d^{-1} c^T (the ghost).
    base = GaussianBeliefNet(Pi=jnp.asarray([[2., 0.], [0., 2.]]),
                             h=jnp.zeros(2), names=("x1", "x2"))
    bd = border(base, "b", Pi_diag=2.0, couplings=jnp.asarray([1., 1.]))
    m = schur_marginalize(bd, ("b",))
    assert np.allclose(np.asarray(m.Pi),
                       np.array([[1.5, -0.5], [-0.5, 1.5]]), atol=1e-9)


def test_additive_fusion_of_independent_evidence():
    # one agent seeing all data == two agents each seeing half, then combined.
    names = ("a", "b")
    prior = vague_prior(names, kappa=1.0)
    J1 = jnp.asarray(_rand_spd(2, seed=11, jitter=0.1))
    J2 = jnp.asarray(_rand_spd(2, seed=12, jitter=0.1))
    j1 = jnp.asarray([0.5, -0.3]); j2 = jnp.asarray([0.1, 0.7])
    full = add_fisher(add_fisher(prior, J1, j1), J2, j2)
    half1 = add_fisher(prior, J1, j1)
    half2 = add_fisher(prior, J2, j2)
    fused = combine(half1, half2)
    # fused double-counts the shared prior; subtract one prior to compare.
    fused_Pi = fused.Pi - prior.Pi
    fused_h = fused.h - prior.h
    assert np.allclose(np.asarray(fused_Pi), np.asarray(full.Pi))
    assert np.allclose(np.asarray(fused_h), np.asarray(full.h))


def test_fuse_reduces_to_precision_pool_scalar():
    # the matrix fuse on 1x1 nets must equal src/inference.py:precision_pool.
    from src.inference import precision_pool
    N = 3
    taus = jnp.asarray([1.0, 2.0, 4.0])
    mus = jnp.asarray([0.5, -0.5, 1.0])
    gamma = jnp.asarray([[1., 2., 0.], [2., 1., 1.], [0., 1., 1.]])
    mask = jnp.ones((N, N))
    mu_pool, tau_pool = precision_pool(mus, taus, gamma, mask)

    # build 1x1 belief nets (Pi=tau, h=tau*mu), fuse, read back mean & precision.
    Pi = taus.reshape(N, 1, 1)
    h = (taus * mus).reshape(N, 1)
    W = S.trust_weights(mask, gamma)
    Pi_f, h_f = S.fuse(Pi, h, W)
    tau_f = Pi_f[:, 0, 0]
    mu_f = h_f[:, 0] / tau_f
    assert np.allclose(np.asarray(tau_f), np.asarray(tau_pool), atol=1e-5)
    assert np.allclose(np.asarray(mu_f), np.asarray(mu_pool), atol=1e-5)


def test_vague_prior_invertible_and_negligible():
    names = ("a", "b", "c")
    vp = vague_prior(names, kappa=1e-3)
    assert np.linalg.slogdet(np.asarray(vp.Pi))[0] == 1.0     # PD, invertible
    sharp = GaussianBeliefNet(Pi=jnp.asarray(_rand_spd(3, seed=3, jitter=5.0)),
                              h=jnp.asarray([1., 1., 1.]), names=names)
    fused = combine(sharp, vp)
    assert np.allclose(np.asarray(fused.mean()), np.asarray(sharp.mean()), atol=1e-2)


# ----------------------------------------------------------------------
# Tier-2: BMR / dynamics
# ----------------------------------------------------------------------

def _cgf(net, t):
    """Gaussian cumulant generating function K(t) = mu^T t + 1/2 t^T Sigma t."""
    mu = np.asarray(net.mean())
    Sigma = np.linalg.inv(np.asarray(net.Pi))
    t = np.asarray(t)
    return float(mu @ t + 0.5 * t @ Sigma @ t)


def test_bmr_prune_matches_mgf_mean_shift():
    # For a pure-tilt (mean-shift) reduced prior tilde_p/p propto exp(t^T phi),
    # the proper model log Bayes factor is K_post(t) - K_prior(t): the note's
    # "Bayes factor = M_post(t)" omits the prior renormalization, which the
    # closed-form model comparison in bmr_prune correctly retains.
    d = 3
    names = tuple(f"n{i}" for i in range(d))
    Pi0 = jnp.asarray(_rand_spd(d, seed=21))
    h0 = jnp.asarray([0.2, -0.4, 0.6])
    prior = GaussianBeliefNet(Pi=Pi0, h=h0, names=names)
    J = jnp.asarray(_rand_spd(d, seed=22, jitter=0.5))
    j = jnp.asarray([1.0, 0.5, -0.5])
    post = add_fisher(prior, J, j)
    t = jnp.asarray([0.3, -0.2, 0.1])
    reduced = GaussianBeliefNet(Pi=Pi0, h=h0 + t, names=names)
    out = bmr_prune(post, prior, reduced)
    expected = _cgf(post, t) - _cgf(prior, t)
    assert np.isclose(float(out["delta_F"]), expected, atol=1e-4)


def test_bmr_prune_reduced_posterior_matches_refit():
    # tilde_post must equal a direct refit: reduced prior + the shared Fisher.
    d = 3
    names = tuple(f"n{i}" for i in range(d))
    prior = GaussianBeliefNet(Pi=jnp.asarray(_rand_spd(d, seed=31)),
                              h=jnp.asarray([0.1, 0.2, 0.3]), names=names)
    J = jnp.asarray(_rand_spd(d, seed=32, jitter=0.3)); j = jnp.asarray([0.4, -0.1, 0.2])
    post = add_fisher(prior, J, j)
    reduced = prune_edge_prior(prior, (names[0], names[1]))
    out = bmr_prune(post, prior, reduced)
    refit = add_fisher(reduced, J, j)
    assert np.allclose(np.asarray(out["tilde_post"].Pi), np.asarray(refit.Pi))
    assert np.allclose(np.asarray(out["tilde_post"].h), np.asarray(refit.h))


def test_schur_of_hub_reproduces_oxygen_support():
    # the carry-over edges the phlogiston hub leaves behind must be present in
    # the oxygen prior (decision #2: support match, oxygen adds gravimetric extra).
    cfg = ph.StructuralConfig()
    surv = schur_marginalize(ph.phlogiston_prior(cfg), (ph.HUB,))
    oxy = ph.oxygen_prior(cfg)
    s_names = surv.names
    surv_support = np.abs(np.asarray(surv.Pi)) > 1e-6
    for ia, na in enumerate(s_names):
        for ib, nb in enumerate(s_names):
            if surv_support[ia, ib]:
                io, jo = oxy.names.index(na), oxy.names.index(nb)
                assert abs(float(oxy.Pi[io, jo])) > 1e-6, f"missing edge {na}-{nb}"


def test_carryover_mass_hub_exceeds_belt():
    cfg = ph.StructuralConfig()
    phlog = ph.phlogiston_prior(cfg)
    hub = obs.carryover_mass(phlog, (ph.HUB,))
    belt = obs.carryover_mass(phlog, ("calx_heavier_than_metal",))
    assert hub > belt
    assert hub > 1.0 and belt < 1e-6


def test_bmr_sign_flips_across_regime():
    # phlogiston's commitment (calx lighter) is vindicated on average in its own
    # regime (delta_F > 0, a modest Occam refund) and decisively refuted in the
    # oxygen regime (delta_F strongly < 0). The vindication is small-amplitude
    # so the near-zero crossings around the boundary are noise; the robust facts
    # are the windowed-mean signs, the decisive post-shift falsification, and
    # that once refuted it stays refuted.
    cfg = ph.StructuralConfig(n_steps=120, t_shift=40)
    out = evidence_race(cfg, jax.random.PRNGKey(0))
    dF = np.asarray(out["delta_F"])
    assert dF[5:cfg.t_shift].mean() > 0          # vindicated in its own regime
    assert dF[-20:].mean() < -10                 # decisively refuted, late
    crossings = np.where(np.diff(np.sign(dF)) != 0)[0]
    assert crossings[-1] >= cfg.t_shift          # decisive falsification is post-shift
    assert np.all(dF[crossings[-1] + 1:] < 0)    # once refuted, stays refuted


def test_evidence_drift_reorders_across_regime():
    # phlogiston leads on evidence in its regime; oxygen leads late.
    cfg = ph.StructuralConfig(n_steps=120, t_shift=40)
    out = evidence_race(cfg, jax.random.PRNGKey(1))
    gap = np.asarray(out["gap"])            # logZ_oxy - logZ_phlog
    assert gap[30] < 0                      # phlogiston ahead in its regime
    assert gap[-1] > 0                      # oxygen ahead at the end


def test_structural_step_is_pure():
    # same key -> identical output (purity contract, mirrors the POMDP step).
    cfg = ph.StructuralConfig(n_agents=20, n_steps=10)
    s1 = S.init_state(cfg, jax.random.PRNGKey(7))
    s2 = S.init_state(cfg, jax.random.PRNGKey(7))
    a1, o1 = S.step(s1, cfg, 0)
    a2, o2 = S.step(s2, cfg, 0)
    assert np.allclose(np.asarray(a1.Pi), np.asarray(a2.Pi))
    assert np.allclose(np.asarray(a1.h), np.asarray(a2.h))
    assert o1["order_parameter"] == o2["order_parameter"]


def _order_traj(state, cfg):
    # population order parameter m(t) over a full rollout (notebook's run_collect).
    xs = []
    for tt in range(cfg.n_steps):
        state, out = S.step(state, cfg, tt)
        xs.append(out["order_parameter"])
    return np.asarray(xs)


def test_heterogeneous_init_breaks_symmetry():
    # Heterogeneous priors (stance x confidence) make the precision-weighted
    # fusion consequential: unlike the homogeneous run (fused == isolated), the
    # fused and isolated order-parameter trajectories now differ.
    import dataclasses
    cfg = ph.StructuralConfig(n_agents=40, n_steps=60, t_shift=20)
    schools = [
        {"count": 20, "stance": -1.5, "prec_scale": 2.0, "label": "dogmatic phlog"},
        {"count": 20, "stance": +0.6, "prec_scale": 1.0, "label": "proto-oxygen"},
    ]
    key = jax.random.PRNGKey(0)

    # (a) per-agent priors are genuinely heterogeneous across the group boundary.
    s_het = S.init_state(cfg, key, groups=schools)
    assert not np.allclose(np.asarray(s_het.Pi[0]), np.asarray(s_het.Pi[-1]))
    assert not np.allclose(np.asarray(s_het.h[0]), np.asarray(s_het.h[-1]))
    # agents inside a group stay identical (broadcast within block).
    assert np.allclose(np.asarray(s_het.Pi[0]), np.asarray(s_het.Pi[19]))

    # (b) The social channel becomes consequential under heterogeneity: the
    # fused-vs-isolated gap is far larger than in the homogeneous control (where
    # fusion only averages out per-agent observation noise -- "essentially the
    # identity on the mean trajectory", per nb18's homogeneous finding).
    m_fused = _order_traj(S.init_state(cfg, key, groups=schools), cfg)
    s_iso = dataclasses.replace(
        S.init_state(cfg, key, groups=schools), W=jnp.eye(cfg.n_agents))
    m_iso = _order_traj(s_iso, cfg)
    het_gap = np.abs(m_fused - m_iso).max()

    m_fused_h = _order_traj(S.init_state(cfg, key), cfg)
    s_iso_h = dataclasses.replace(
        S.init_state(cfg, key), W=jnp.eye(cfg.n_agents))
    m_iso_h = _order_traj(s_iso_h, cfg)
    homo_gap = np.abs(m_fused_h - m_iso_h).max()

    assert het_gap > 3 * homo_gap        # heterogeneity makes fusion bite
    assert het_gap > 0.1                 # and the effect is sizeable


def test_heterogeneous_init_count_mismatch_raises():
    cfg = ph.StructuralConfig(n_agents=40)
    bad = [{"count": 10, "stance": -1.0}, {"count": 20, "stance": 1.0}]
    with pytest.raises(ValueError):
        S.init_state(cfg, jax.random.PRNGKey(0), groups=bad)


def _crosses_half(m):
    return np.any(np.diff(np.sign(np.asarray(m) - 0.5)) != 0)


def test_precision_is_the_lock_in_knob():
    # A low-precision phlogiston bloc concedes (crosses 1/2 after the world flips);
    # a high-precision one is never overwhelmed within the horizon -- lock-in.
    cfg = ph.StructuralConfig(n_agents=20, n_steps=120, t_shift=40)
    key = jax.random.PRNGKey(0)
    m_soft = _order_traj(
        S.init_state(cfg, key, groups=[{"count": 20, "stance": -2.0, "prec_scale": 1.0}]), cfg)
    m_hard = _order_traj(
        S.init_state(cfg, key, groups=[{"count": 20, "stance": -2.0, "prec_scale": 200.0}]), cfg)
    assert _crosses_half(m_soft)          # soft prior concedes to the evidence
    assert not _crosses_half(m_hard)      # entrenched prior refuses to update
    assert m_hard[-1] < 0.2               # and stays deep in phlogiston


def test_entrenched_minority_captures_population():
    # In isolation the open majority flips; precision-weighted fusion lets a small
    # high-precision minority drag the population back below the shift threshold.
    import dataclasses
    cfg = ph.StructuralConfig(n_agents=60, n_steps=120, t_shift=40)
    key = jax.random.PRNGKey(0)
    schools = [
        {"count": 15, "stance": -2.0, "prec_scale": 120.0},   # entrenched minority
        {"count": 25, "stance":  0.0, "prec_scale": 0.5},     # open agnostic
        {"count": 20, "stance": +0.4, "prec_scale": 1.0},     # proto-oxygenist
    ]
    m_fused = _order_traj(S.init_state(cfg, key, groups=schools), cfg)
    m_iso = _order_traj(
        dataclasses.replace(S.init_state(cfg, key, groups=schools),
                            W=jnp.eye(cfg.n_agents)), cfg)
    assert _crosses_half(m_iso)           # without communication the majority flips
    assert not _crosses_half(m_fused)     # communication-driven capture prevents the shift
    assert m_fused[-1] < 0.5


# --- biased experiment selection (theory-laden observation) -----------------

def test_unbiased_attention_is_all_ones():
    # experiment_bias=0 must recover the plain deposit -> weights are all 1.
    cfg = ph.StructuralConfig(experiment_bias=0.0)
    oxy = jnp.array([0.0, 0.3, 0.7, 1.0])
    W = ph.attention_weights(oxy, cfg)
    assert np.allclose(np.asarray(W), 1.0)


def test_bias_gate_skips_gravimetric_when_holding_phlogiston():
    # A phlogiston-holding agent (oxy<1/2) skips the disagreement experiments at
    # bias=1; an oxygen-holding agent (oxy>1/2) still runs them.
    cfg = ph.StructuralConfig(experiment_bias=1.0, bias_sharpness=12.0)
    mask = np.asarray(ph.disagreement_row_mask(cfg)).astype(bool)
    W = np.asarray(ph.attention_weights(jnp.array([0.0, 1.0]), cfg))
    assert np.all(W[0, mask] < 0.05)       # committed phlogistonist: refuting assay off
    assert np.allclose(W[0, ~mask], 1.0)   # agreement assays always run
    assert np.all(W[1, mask] > 0.95)       # oxygen-holder runs the refuting assay


def test_run_final_matches_step_loop():
    # The scan fast-forward must agree with the python step loop bit-for-bit.
    cfg = ph.StructuralConfig(n_agents=12, n_steps=15, experiment_bias=0.6)
    key = jax.random.PRNGKey(3)
    s0 = S.init_state(cfg, key, groups=[{"count": 12, "stance": -1.0, "prec_scale": 5.0}])
    s = s0
    for t in range(cfg.n_steps):
        s, _ = S.step(s, cfg, t)
    fin = S.run_final(cfg, s0)
    assert np.allclose(np.asarray(s.Pi), np.asarray(fin.Pi), atol=1e-5)
    assert np.allclose(np.asarray(s.h), np.asarray(fin.h), atol=1e-5)


def _m_final(cfg, state):
    fin = S.run_final(cfg, state)
    return float(obs.order_parameter(fin.Pi, fin.h, cfg.node_names,
                 ph.DISAGREEMENT_NODES, cfg.mu_phlog_mass, cfg.mu_oxy_mass))


def test_biased_selection_makes_lock_in_permanent():
    # At a long horizon the unbiased entrenched bloc eventually concedes (no
    # forgetting -> monostable); biased selection makes the lock-in permanent.
    key = jax.random.PRNGKey(0)
    grp = [{"count": 12, "stance": -1.5, "prec_scale": 30.0}]
    cfg_unbiased = ph.StructuralConfig(n_agents=12, n_steps=500, t_shift=40,
                                       experiment_bias=0.0)
    cfg_biased = ph.StructuralConfig(n_agents=12, n_steps=500, t_shift=40,
                                     experiment_bias=1.0, bias_sharpness=12.0)
    assert _m_final(cfg_unbiased, S.init_state(cfg_unbiased, key, groups=grp)) > 0.5
    assert _m_final(cfg_biased, S.init_state(cfg_biased, key, groups=grp)) < 0.1


# ----------------------------------------------------------------------
# Tier-3: derived evidential precision (rho_k, the dual of carry-over).
# ----------------------------------------------------------------------

# A conviction-bearing PD paradigm: base_prec/hub_self_prec large enough that the
# common-cause hub prior is positive-definite (2*hub_self_prec > sum hub couplings
# = 6.4), so the order-parameter solve is well-posed and lock-in is clean.
def _pd_cfg(g, n_steps=160, mode="derived", **kw):
    return ph.StructuralConfig(
        n_agents=12, n_steps=n_steps, t_shift=40,
        base_prec=5.0, hub_self_prec=10.0, mu_phlog_mass=-1.5,
        precision_mode=mode, core_governance=float(g), **kw)


def test_rho_from_cost_g0_is_rho_max():
    # The back-compat anchor: zero governance leaves every channel at full gain.
    cost = jnp.asarray([0.0, 0.64, 3.0, 100.0])
    assert np.allclose(np.asarray(P.rho_from_cost(cost, 0.0, rho_max=1.0)), 1.0)
    assert np.allclose(np.asarray(P.rho_from_cost(cost, 0.0, rho_max=0.7)), 0.7)


def test_rho_max_for_core_independent_node():
    # A node with zero coupling to the core costs nothing to revise -> rho = rho_max
    # at any governance (hand-built block-diagonal net makes the independence exact).
    net = GaussianBeliefNet(
        Pi=jnp.asarray([[2.0, 0.8, 0.0],
                        [0.8, 1.0, 0.0],
                        [0.0, 0.0, 1.0]]),
        h=jnp.zeros(3), names=("core", "bound", "free"))
    cost = P.core_coupling(net, "core", ("bound", "free"), cost_kind="carryover")
    assert float(cost[1]) == pytest.approx(0.0)             # free: Pi[core,free]=0
    assert float(cost[0]) > 0.0                             # bound: Pi[core,bound]!=0
    for g in (1.0, 10.0, 1000.0):
        rho = P.channel_precision(net, "core", ("bound", "free"), g)
        assert float(rho[1]) == pytest.approx(1.0)          # free channel never governed
        assert float(rho[0]) < 1.0                          # bound channel governed


def test_rho_monotone_decreasing_in_governance():
    # On the paradigm field, rho on the mass-law channels falls strictly and -> 0 as
    # core_governance grows (the dual of carry-over: cost is Pi[core,v]^2/Pi[v,v] > 0).
    cfg = _pd_cfg(0.0)
    meas = ph.measured_nodes(cfg)
    cols = [meas.index(n) for n in ph.DISAGREEMENT_NODES]
    prev = None
    for g in [0.0, 1.0, 10.0, 100.0, 1000.0, 1e5]:
        rho = np.asarray(ph.derived_channel_precision(
            __import__("dataclasses").replace(cfg, core_governance=float(g))))
        mass = rho[cols]
        if g == 0.0:
            assert np.allclose(mass, 1.0)
        if prev is not None:
            assert np.all(mass < prev - 1e-9)               # strictly decreasing
        prev = mass
    assert np.all(mass < 1e-3)                              # -> 0 at strong governance


def test_derived_transition_reduces_to_plain_deposit_at_g0():
    # precision_mode='derived', g=0 must be byte-identical to the unbiased heuristic
    # path (both are the plain unweighted Fisher deposit).
    key = jax.random.PRNGKey(1)
    base = dict(n_agents=10, n_steps=40, t_shift=15,
                base_prec=5.0, hub_self_prec=10.0, mu_phlog_mass=-1.5)
    cfg_h = ph.StructuralConfig(**base, precision_mode="heuristic", experiment_bias=0.0)
    cfg_d = ph.StructuralConfig(**base, precision_mode="derived", core_governance=0.0)
    fin_h = S.run_final(cfg_h, S.init_state(cfg_h, key))
    fin_d = S.run_final(cfg_d, S.init_state(cfg_d, key))
    assert np.allclose(np.asarray(fin_h.Pi), np.asarray(fin_d.Pi), atol=1e-6)
    assert np.allclose(np.asarray(fin_h.h), np.asarray(fin_d.h), atol=1e-6)


def test_mass_channels_governed_relative_to_least_coupled():
    # The disagreement (mass-law) channels are governed: with the anomaly bound, the
    # three mass-law channels all have rho < 1 at positive governance.
    cfg = _pd_cfg(50.0)
    meas = ph.measured_nodes(cfg)
    rho = np.asarray(ph.derived_channel_precision(cfg))
    for n in ph.DISAGREEMENT_NODES:
        assert rho[meas.index(n)] < 0.999


def test_derived_lock_in_stalls_below_half():
    # The GO/NO-GO of the design: strong governance stalls m(t) below 1/2 after the
    # regime flip (mass channels silenced -> belief frozen at the phlogiston prior),
    # while g=0 reorganises and crosses 1/2 (graded). Monotone in between.
    m0 = _m_final(_pd_cfg(0.0), S.init_state(_pd_cfg(0.0), jax.random.PRNGKey(0)))
    mlock = _m_final(_pd_cfg(3000.0), S.init_state(_pd_cfg(3000.0), jax.random.PRNGKey(0)))
    assert m0 > 0.5                                        # graded reorganization
    assert mlock < 0.1                                     # evidential lock-in


def test_anomaly_binding_deepens_seal():
    # Leaving the anomaly unbound (anomaly_coupling=0) leaves an ungoverned channel
    # through which the rival leaks, so the seal is shallower than when it is bound.
    g = 1000.0
    cfg_bound = _pd_cfg(g)                                  # anomaly bound at hub_coupling
    cfg_leak = _pd_cfg(g, anomaly_coupling=0.0)
    m_bound = _m_final(cfg_bound, S.init_state(cfg_bound, jax.random.PRNGKey(0)))
    m_leak = _m_final(cfg_leak, S.init_state(cfg_leak, jax.random.PRNGKey(0)))
    assert m_bound < m_leak                                # binding seals deeper


def test_evidence_race_weighted_ones_recovers_unweighted():
    # rho = ones must reproduce the plain evidence_race trajectories exactly.
    cfg = ph.StructuralConfig(n_steps=60, t_shift=20)
    key = jax.random.PRNGKey(7)
    H = ph.H_observable(cfg)
    ref = evidence_race(cfg, key)
    got = evidence_race_weighted(cfg, key, jnp.ones(H.shape[0]))
    for k in ("gap", "delta_F", "logZ_oxy", "logZ_phlog"):
        assert np.allclose(np.asarray(ref[k]), np.asarray(got[k]), atol=1e-5)


def test_multiplicative_bias_gap_flat_under_incumbent_R():
    # The oxygen-over-phlogiston evidence gap GROWS after the regime flip under a
    # paradigm-neutral common R (all rho=1), but stays ~flat under the incumbent's
    # R(G) (the rival's anomaly lands on a silenced channel). Both candidates share
    # the same weighted Fisher, so the bias is purely in R.
    cfg = ph.StructuralConfig(n_steps=120, t_shift=40, base_prec=5.0,
                              hub_self_prec=10.0, mu_phlog_mass=-1.5,
                              precision_mode="derived", core_governance=3000.0)
    key = jax.random.PRNGKey(0)
    H = ph.H_observable(cfg)
    gap_neutral = np.asarray(evidence_race_weighted(cfg, key, jnp.ones(H.shape[0]))["gap"])
    gap_incumbent = np.asarray(evidence_race_weighted(
        cfg, key, ph.derived_channel_precision(cfg))["gap"])
    ts = cfg.t_shift
    grow_neutral = gap_neutral[-1] - gap_neutral[ts]
    grow_incumbent = gap_incumbent[-1] - gap_incumbent[ts]
    assert grow_neutral > 5.0                              # disconfirmation accumulates
    assert grow_incumbent < 0.5 * grow_neutral             # silenced at the source


def test_bubble_vs_chamber_on_identical_data():
    # Same world, same seed. BUBBLE = the disagreement channels are physically absent
    # (rows removed from H): zero mass-node Fisher, m stalls -- but RESTORING the rows
    # lets info flow. ECHO CHAMBER = the channels are present and sampled but silenced
    # (rho~0): the data arrive yet ~zero mass-node Fisher is deposited and m stalls,
    # and re-showing the data changes nothing. The distinguishing scalar is the
    # cumulative Fisher deposited on the mass nodes under a restore/re-expose action.
    cfg = ph.StructuralConfig(n_steps=80, t_shift=20, base_prec=5.0,
                              hub_self_prec=10.0, mu_phlog_mass=-1.5)
    key = jax.random.PRNGKey(0)
    H = np.asarray(ph.H_observable(cfg))
    meas = ph.measured_nodes(cfg)
    dis_rows = [meas.index(n) for n in ph.DISAGREEMENT_NODES]
    mass_idx = [cfg.node_names.index(n) for n in ph.DISAGREEMENT_NODES]
    rho_chamber = np.asarray(ph.derived_channel_precision(
        __import__("dataclasses").replace(cfg, precision_mode="derived",
                                          core_governance=3000.0)))
    H_bubble = H.copy(); H_bubble[dis_rows] = 0.0           # channels absent

    def cumulative_mass_fisher(use_H, rho, expose=True):
        prior = ph.phlogiston_prior(cfg)
        Pi = np.asarray(prior.Pi); k = key; tot = 0.0
        for t in range(cfg.n_steps):
            phi = ph.phi_true_at(cfg, t)
            k, sub = jax.random.split(k)
            o = sample_o(jnp.asarray(use_H), phi, cfg.sigma_o, sub)
            if rho is None:
                J, _ = fisher_deposit(jnp.asarray(use_H), o, cfg.sigma_o)
            else:
                J, _ = fisher_deposit_weighted(jnp.asarray(use_H), o, cfg.sigma_o,
                                               jnp.asarray(rho))
            J = np.asarray(J)
            tot += float(np.sum(np.diag(J)[mass_idx])) if expose else 0.0
        return tot

    f_bubble = cumulative_mass_fisher(H_bubble, None)       # rows absent
    f_chamber = cumulative_mass_fisher(H, rho_chamber)      # rows present, gain off
    f_restored = cumulative_mass_fisher(H, None)            # bubble's channel restored
    assert f_bubble == pytest.approx(0.0, abs=1e-6)         # bubble: no mass Fisher
    assert f_chamber < 0.05 * f_restored                   # chamber: ~no mass Fisher
    assert f_restored > 1.0                                 # restoring the channel -> info flows


def test_run_trace_precision_returns_disagree_rho():
    # The additive 3-tuple: third element is (n_steps, n_disagree); for derived mode it
    # is constant in t and equals the derived rho on the disagreement rows; g=0 -> 1.
    cfg = _pd_cfg(100.0, n_steps=30)
    key = jax.random.PRNGKey(0)
    ms, gws, rho_dis = S.run_trace_precision(cfg, S.init_state(cfg, key))
    n_dis = len(ph.DISAGREEMENT_NODES)
    assert np.asarray(rho_dis).shape == (cfg.n_steps, n_dis)
    meas = ph.measured_nodes(cfg)
    cols = [meas.index(n) for n in ph.DISAGREEMENT_NODES]
    expect = np.asarray(ph.derived_channel_precision(cfg))[cols]
    assert np.allclose(np.asarray(rho_dis)[0], expect, atol=1e-5)
    assert np.allclose(np.asarray(rho_dis)[-1], expect, atol=1e-5)   # constant in t
    cfg0 = _pd_cfg(0.0, n_steps=10)
    _, _, rho0 = S.run_trace_precision(cfg0, S.init_state(cfg0, key))
    assert np.allclose(np.asarray(rho0), 1.0)                        # g=0 -> full gain
