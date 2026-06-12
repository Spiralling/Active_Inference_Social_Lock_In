"""The Kuhn cycle with inferred channel reliability: denial, crisis, late revolution.

In the fixed-precision regime the population's conversion through the regime flip
is a smooth sigmoid: anomalous data counts at face value from the first step, so
the order parameter just slides. Historically that is wrong twice over -- the
chemists first doubted the gravimetric MEASUREMENTS, not phlogiston, and the
revolution, when it came, came fast.

With ``reliability_nu`` on (the Student-t channel gate of ``reliability.py``,
already wired into ``step._fuse_then_observe``), the same population produces
the missing shape endogenously:

  1. DENIAL: at the regime flip the anomalous channels become surprising
     (``z^2 >> nu``), so their inferred reliability ``lambda`` collapses -- the
     community discounts the instruments before it revises the theory.
  2. CRISIS-AS-DELAY: updating on the contested channels is throttled, so the
     conversion stalls at the old paradigm long after the world has flipped.
  3. REHABILITATION: the trickle of discounted evidence still moves the
     belief; as the belief approaches the data, surprise falls, ``lambda``
     recovers, and the update gain returns -- the channels the community
     condemned are rehabilitated and conversion completes.

Shape, stated honestly: the channel gate THROTTLES, it does not break. The
gated conversion is a stretched, decelerating slide (max slope at the flip; no
interior slope maximum at nu = 2, 1, or 0.5; harsher gates just stretch it
further and leave it incomplete) -- so this experiment shows denial and delay,
not slow-then-sudden. The SUDDEN half of the Kuhnian shape appears one level
up, in the social trust gate (``lockin_inferred_trust``): there the feedback
is collective (any leak shrinks the inter-camp gap, which reopens the gate,
which accelerates the leak), and the laggards' merge has a genuine interior
slope maximum late in the run.

The lambda trace reported here is the per-step EXPECTED reliability read off the
logged beliefs at the noise-free observation ``o = H phi_true(t)`` -- a
deterministic diagnostic of the same quantity the dynamics compute on sampled
``o`` (it slightly overstates lambda since real draws add noise). Honest scope:
with the instantaneous (memoryless) gate, precision never falls, so this shows
denial and delay, not the variance-inflation form of crisis -- that needs the
retrospective accumulator of ``abc_conflict``.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register


def _traces(cfg):
    """Run one population and return per-step (oxy_index, mean lambda on the
    disagreement channels, mean lambda on the agreement channels)."""
    import jax
    import jax.numpy as jnp
    from src.structural import phlogiston as ph
    from src.structural import reliability as rel
    from src.structural import step

    state = step.init_state(cfg, jax.random.PRNGKey(cfg.seed))
    Pi_t, h_t = step.run_trace_net(cfg, state)             # (T,N,d,d), (T,N,d)
    T, N, d, _ = Pi_t.shape

    H = ph.observation_operator(cfg)
    dm = np.asarray(ph.disagreement_row_mask(cfg)).astype(bool)
    mass_idx = jnp.asarray([cfg.node_names.index(n) for n in ph.DISAGREEMENT_NODES])
    nu = cfg.reliability_nu if cfg.reliability_nu is not None else 4.0

    phis = jnp.stack([ph.phi_true_at(cfg, t) for t in range(cfg.n_steps)])

    def one(Pi, h, phi):
        mu = jnp.linalg.solve(Pi, h)
        oxy = jnp.clip((mu[mass_idx].mean() - cfg.mu_phlog_mass)
                       / (cfg.mu_oxy_mass - cfg.mu_phlog_mass), 0.0, 1.0)
        lam = rel.channel_reliability(Pi, h, H, H @ phi, cfg.sigma_o, nu)
        return oxy, lam

    over_agents = jax.vmap(one, in_axes=(0, 0, None))     # (N,...) at one step
    oxy_tn, lam_tnm = jax.vmap(over_agents)(Pi_t, h_t, phis)   # (T,N), (T,N,m)
    oxy_t = np.asarray(oxy_tn).mean(axis=1)                          # (T,)
    lam_tm = np.asarray(lam_tnm).mean(axis=1)                        # (T, m)
    return oxy_t, lam_tm[:, dm].mean(axis=1), lam_tm[:, ~dm].mean(axis=1)


def _t_half(oxy_t):
    hit = np.nonzero(oxy_t > 0.5)[0]
    return int(hit[0]) if hit.size else len(oxy_t)


def fig_trace(t_shift, base, robust, path) -> dict:
    oxy_b, _, _ = base
    oxy_r, lam_dis, lam_agr = robust
    T = len(oxy_b)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.0, 0.8]))
    a0.plot(range(T), oxy_b, lw=2.2, color="#7f8c8d", ls="--",
            label="fixed precision: the smooth slide")
    a0.plot(range(T), oxy_r, lw=2.4, color="#1a5276",
            label="inferred reliability: denial, then a throttled conversion")
    a0.axvline(t_shift, color="k", lw=0.8)
    a0.text(t_shift + 1, 0.05, "the world flips", fontsize=8)
    a0.set_ylabel("oxygen index (population)"); a0.set_ylim(-0.04, 1.04)
    a0.legend(fontsize=8)
    a0.set_title("the Kuhn cycle with inferred channel reliability: the community doubts the "
                 "instruments before it doubts the theory", fontsize=9.5)
    a1.plot(range(T), lam_dis, lw=2.2, color="#b03a2e",
            label=r"$\bar\lambda$ on the anomalous (mass-law) channels")
    a1.plot(range(T), lam_agr, lw=1.8, color="#1e8449", ls=":",
            label=r"$\bar\lambda$ on the agreement channels")
    a1.axvline(t_shift, color="k", lw=0.8)
    a1.set_ylabel("inferred channel reliability"); a1.set_xlabel("step")
    a1.set_ylim(0, 1.35); a1.legend(fontsize=8)
    a1.set_title("denial is the gate: the anomalous channels are inferred unreliable at the "
                 "flip, and rehabilitated as the belief catches up", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return {}


def run(out_dir, params: dict) -> None:
    import dataclasses
    from src.structural.phlogiston import StructuralConfig

    cfg0 = StructuralConfig(n_agents=params["N_AGENTS"], n_steps=params["N_STEPS"],
                            t_shift=params["T_SHIFT"], sigma_o=params["SIGMA_O"],
                            seed=params["SEED"])
    base = _traces(cfg0)
    robust = _traces(dataclasses.replace(cfg0, reliability_nu=params["NU"]))

    oxy_b, oxy_r = base[0], robust[0]
    lam_dis, lam_agr = robust[1], robust[2]
    t_shift = params["T_SHIFT"]
    th_b, th_r = _t_half(oxy_b), _t_half(oxy_r)

    # (1) DENIAL: the anomalous channels are inferred unreliable right after the flip...
    assert lam_dis[:t_shift].min() > 0.85, "pre-flip, every channel must be trusted"
    assert lam_dis[t_shift:t_shift + 25].min() < 0.45, \
        f"the anomalous channels must be discounted at the flip (got {lam_dis[t_shift:t_shift+25].min():.2f})"
    assert lam_agr.min() > 0.85, "the agreement channels must stay trusted throughout"
    # (2) ...which DELAYS the revolution...
    assert th_b < th_r, "inferred reliability must delay the conversion"
    assert th_r - t_shift >= 2 * (th_b - t_shift), \
        f"the denial phase should >=2x the conversion lag ({th_r - t_shift} vs {th_b - t_shift})"
    # (3) ...and REHABILITATION completes the cycle: trust restored, conversion completed.
    assert lam_dis[-1] > 0.8, f"the channels must be rehabilitated (final {lam_dis[-1]:.2f})"
    assert oxy_r[-1] > 0.85, f"the revolution must still complete (final {oxy_r[-1]:.2f})"

    fig_trace(t_shift, base, robust, out_dir / "fig_crisis_trace.png")
    np.savez_compressed(out_dir / "simulation_arrays.npz",
                        oxy_base=oxy_b, oxy_robust=oxy_r,
                        lam_disagree=lam_dis, lam_agree=lam_agr)
    summary = {"config": params,
               "t_half_baseline": th_b, "t_half_robust": th_r,
               "min_lambda_disagree": float(lam_dis.min()),
               "final_lambda_disagree": float(lam_dis[-1]),
               "denial_lag_ratio": float((th_r - t_shift) / max(th_b - t_shift, 1))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: with inferred channel reliability the Kuhn cycle gets denial and delay "
          f"-- at the regime flip the community infers the anomalous channels unreliable "
          f"(lambda falls to {lam_dis.min():.2f}: denial, selective -- agreement channels stay "
          f"trusted), conversion is throttled to {summary['denial_lag_ratio']:.1f}x the "
          f"fixed-precision lag, and the channels are rehabilitated (final lambda "
          f"{lam_dis[-1]:.2f}) as the belief catches up: doubt the instruments first, the "
          f"theory later. The gate stretches the conversion, it does not break it -- the "
          f"sudden half of the cycle lives in the social gate (lockin_inferred_trust).")


register(ExperimentSpec(
    model="phlogiston",
    name="crisis_trace",
    description="The Kuhn cycle with inferred channel reliability (reliability_nu) on the "
                "step.py population: at the regime flip the anomalous channels are inferred "
                "unreliable (denial, selective), conversion is throttled ~2.6x "
                "(crisis-as-delay), and the gate rehabilitates as the belief catches up. The "
                "channel gate stretches the conversion rather than breaking it; the "
                "slow-then-sudden merge lives in the social gate (lockin_inferred_trust).",
    run=run,
    out_dir="crisis_trace",
    params=dict(N_AGENTS=20, N_STEPS=800, T_SHIFT=40, SIGMA_O=0.5, NU=2.0, SEED=0),
    seeds=(0,),
    canonical=False,
    consumes=dict(figures=["fig_crisis_trace"]),
))
