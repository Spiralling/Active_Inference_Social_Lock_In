"""The proposal rate as an updatable belief about one's own model adequacy.

The second refinement the limitations section names: the constant Poisson rate is the
maximally agnostic model of conceiving an unconceived alternative, but the rate admits a dual
reading -- one EXPLORES at the rate one believes one's model is incomplete -- and that belief
is updatable. ``rate_mode="adaptive"`` drives each agent's base rate from an EMA of its own
residual floor (the very telemetry the discovery check records):

    s_i <- (1-rho) s_i + rho * floor_i,    r0_i(t) = clip(r0 + kappa max(s_i - ref, 0), 0, rmax)

so a persistent unexplained residual raises the agent's own arrival rate, while a quiet model
lets it decay back to the agnostic base.

What it must show (and asserts):
* during normal science the rate IS the base rate (the floor sits below ``floor_ref``);
* after crisis the rate rises with the residual the dead paradigm leaves behind, and the
  crisis-to-discovery delay falls well below the fixed-rate control at the same base rate;
* a fixed-rate control matched to the adaptive run's REALIZED mean rate gets a comparable
  delay -- the adaptive gain is rate-when-needed, not just more rate (the dissolved
  waiting-time law is reported as an effective rate, as the appendix anticipates);
* once the wake resolves the residual, the rate decays back toward the base.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register
from src.structural.models.kuhn_phlogiston import single_run


def _delays(r) -> np.ndarray:
    cs, es = r["crisis_step"], r["expand_step"]
    ok = (cs >= 0) & (es >= 0)
    return (es[ok] - cs[ok]).astype(float)


def _realized_rate(r) -> float:
    """Mean per-agent rate over the ELIGIBLE steps (post-crisis, pre-expansion) -- the
    effective rate the adaptive run actually ran at while it mattered."""
    rt, cs, es = r["rate_tn"], r["crisis_step"], r["expand_step"]
    T, N = rt.shape
    t_idx = np.arange(T)[:, None]
    eligible = (cs[None, :] >= 0) & (t_idx > cs[None, :]) \
        & ((es[None, :] < 0) | (t_idx <= es[None, :]))
    return float(rt[eligible].mean())


def fig_adaptive(r_ad, delays, path, r0: float, floor_ref: float) -> dict:
    comm = r_ad["community"]
    o = comm == 0
    rt = r_ad["rate_tn"][:, o]
    fl = r_ad["floor_tn"][:, o]
    cs, es = r_ad["crisis_step"][o], r_ad["expand_step"][o]
    tt = np.arange(rt.shape[0])
    t_crisis = int(np.median(cs[cs >= 0])) if (cs >= 0).any() else -1
    t_disc = int(np.median(es[es >= 0])) if (es >= 0).any() else -1

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.6, 4.2))
    a0.plot(tt, rt.mean(axis=1), lw=2.2, color="#1f618d", label="proposal rate $r_i(t)$ (mean)")
    a0.fill_between(tt, np.percentile(rt, 10, axis=1), np.percentile(rt, 90, axis=1),
                    color="#1f618d", alpha=0.18, label="10-90% band")
    a0.axhline(r0, color="k", ls=":", lw=1.0, label=f"base rate $r_0$={r0}")
    a0b = a0.twinx()
    a0b.plot(tt, fl.mean(axis=1), lw=1.6, color="#b9770e", alpha=0.8,
             label="residual floor (mean)")
    a0b.axhline(floor_ref, color="#b9770e", ls=":", lw=1.0)
    a0b.set_ylabel("residual floor", color="#b9770e")
    for x, lab in ((r_ad["t_shift"], "world flips"), (t_crisis, "median crisis"),
                   (t_disc, "median discovery")):
        if x >= 0:
            a0.axvline(x, color="k", lw=0.8, alpha=0.6)
            a0.text(x, a0.get_ylim()[1], f" {lab}", fontsize=7, va="top", rotation=90)
    a0.set_xlabel("step"); a0.set_ylabel("proposal rate")
    h0, l0 = a0.get_legend_handles_labels(); h1, l1 = a0b.get_legend_handles_labels()
    a0.legend(h0 + h1, l0 + l1, fontsize=7, loc="center right")
    a0.set_title("the rate is a belief about one's own inadequacy:\n"
                 "flat in normal science, raised by the post-crisis residual,\n"
                 "decaying once the wake explains it")

    labels = list(delays.keys())
    means = [np.mean(delays[k]) if len(delays[k]) else np.nan for k in labels]
    errs = [1.96 * np.std(delays[k]) / max(np.sqrt(len(delays[k])), 1) for k in labels]
    a1.bar(range(len(labels)), means, yerr=errs, capsize=4,
           color=["#1f618d", "#922b21", "#6c7a89"])
    a1.set_xticks(range(len(labels))); a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylabel("crisis $\\to$ discovery delay (steps)")
    a1.set_title("rate-when-needed: the adaptive agent discovers like a\n"
                 "matched-mean control, while idling at $r_0$ when quiet")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(t_crisis=t_crisis, t_disc=t_disc,
                delays={k: float(np.mean(v)) for k, v in delays.items()})


def run(out_dir, params: dict) -> None:
    base = dict(N=params["N_AGENTS"], inter=0.0, omega=params["OMEGA"],
                sigma_o=params["SIGMA_O"], gate_strength=params["GATE_STRENGTH"],
                s_dogma=params["S_DOGMA"], t_shift=params["T_SHIFT"],
                n_steps=params["N_STEPS"], snapshot_every=20)
    r0, kappa = params["R0"], params["KAPPA"]

    d_ad, d_fix, rates_real = [], [], []
    r_show = None
    for s in params["SEEDS"]:
        r_ad = single_run(proposal_rate=r0, rate_mode="adaptive", rate_kappa=kappa,
                          rate_floor_ema=params["FLOOR_EMA"],
                          rate_floor_ref=params["FLOOR_REF"], rate_max=params["RATE_MAX"],
                          seed=s, **base)
        r_fx = single_run(proposal_rate=r0, seed=s, **base)
        d_ad.append(_delays(r_ad)); d_fix.append(_delays(r_fx))
        rates_real.append(_realized_rate(r_ad))
        if r_show is None:
            r_show = r_ad

        # normal science is quiet: the pre-shift rate sits at the base rate
        pre = r_ad["rate_tn"][: params["T_SHIFT"]]
        assert abs(pre.mean() - r0) < 0.25 * r0, \
            f"pre-shift rate must sit at r0 (got {pre.mean():.4f} vs {r0})"
        # the wake resolves the residual: by the end the rate has decayed off its peak
        rt_open = r_ad["rate_tn"][:, r_ad["community"] == 0].mean(axis=1)
        assert rt_open[-1] < 0.5 * rt_open.max(), \
            "the rate must decay once the residual is explained"

    # matched-mean fixed-rate control: same EFFECTIVE rate, none of the timing
    r_match = float(np.clip(np.mean(rates_real), 1e-4, 0.99))
    d_match = []
    for s in params["SEEDS"]:
        r_mm = single_run(proposal_rate=r_match, seed=s, **base)
        d_match.append(_delays(r_mm))

    delays = {f"adaptive ($r_0$={r0})": np.concatenate(d_ad),
              f"fixed $r$={r0}": np.concatenate(d_fix),
              f"fixed $r$={r_match:.3f} (matched mean)": np.concatenate(d_match)}
    info = fig_adaptive(r_show, delays, out_dir / "fig_adaptive_rate.png",
                        r0=r0, floor_ref=params["FLOOR_REF"])

    m_ad = float(np.mean(delays[f"adaptive ($r_0$={r0})"]))
    m_fx = float(np.mean(delays[f"fixed $r$={r0}"]))
    m_mm = float(np.mean(delays[f"fixed $r$={r_match:.3f} (matched mean)"]))
    assert m_ad < 0.6 * m_fx, \
        f"adaptive must beat the fixed-rate control at the same base ({m_ad:.1f} vs {m_fx:.1f})"
    assert m_ad < 2.0 * m_mm, \
        f"adaptive should be in the matched-mean control's league ({m_ad:.1f} vs {m_mm:.1f})"

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        rate_tn=r_show["rate_tn"], floor_tn=r_show["floor_tn"],
        crisis_step=r_show["crisis_step"], expand_step=r_show["expand_step"],
        community=r_show["community"],
        delay_adaptive=delays[f"adaptive ($r_0$={r0})"],
        delay_fixed=delays[f"fixed $r$={r0}"],
        delay_matched=delays[f"fixed $r$={r_match:.3f} (matched mean)"],
        matched_rate=np.array([r_match]),
    )
    summary = {"config": params, "fig": info,
               "delay_adaptive": m_ad, "delay_fixed_r0": m_fx,
               "delay_matched": m_mm, "matched_rate": r_match,
               "realized_rates": [float(x) for x in rates_real]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: the exploration rate as an updatable inadequacy belief -- flat at "
          f"r0={r0} through normal science, raised by the post-crisis residual to an "
          f"effective {r_match:.3f}, cutting the crisis-to-discovery delay from {m_fx:.0f} "
          f"to {m_ad:.0f} steps (matched-mean control: {m_mm:.0f}), then decaying once the "
          f"woken hub explains the residual. The dial became a posterior.")


register(ExperimentSpec(
    model="phlogiston",
    name="adaptive_rate",
    description="The active-inference closure of the proposal rate (limitations section, "
                "implemented): the arrival rate is an updatable hyperprior on the agent's "
                "own model adequacy, driven by the residual floor the discovery check "
                "records -- persistent residual raises the rate, a quiet model decays it.",
    run=run,
    out_dir="adaptive_rate",
    params=dict(
        N_AGENTS=80, T_SHIFT=40, N_STEPS=180, OMEGA=0.97, SIGMA_O=0.5,
        GATE_STRENGTH=1.0, S_DOGMA=3.0,
        R0=0.01, KAPPA=0.2, FLOOR_EMA=0.1, FLOOR_REF=0.3, RATE_MAX=0.8,
        SEEDS=(0, 1, 2),
    ),
    seeds=(0, 1, 2),
    canonical=False,
    consumes=dict(figures=["fig_adaptive_rate"]),
))
