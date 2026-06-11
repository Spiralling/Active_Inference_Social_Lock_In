"""All the limitation-closing mechanisms, composed: one run with everything on.

Each refinement of the limitations section ships with its own isolating experiment
(hawkes_rescue, adaptive_rate, wake_then_prune, multi_candidate, lakatos_conviction,
fusion_survival, precrisis_check). This experiment is the composition check the paper needs
before any of them can be claimed together: the SAME population run with every mechanism
enabled at once --

* the proposal rate is a Hawkes-excited, adequacy-adaptive belief (``hawkes_adaptive``);
* wakes are entertained on expiring epistemic credit and audited every round
  (``wake_credit``);
* fusion is dimension-aware (``posterior_masked``: pinned peers abstain from the slot);
* conviction accretes onto entrenched commitments (``conviction_eps``, Lakatos);
* the pre-crisis discovery check is measured at every step (``precrisis_check``).

What it must show (and asserts), per seed:
* the full Kuhn cycle still completes -- anomaly, crisis, reduction, discovery, new normal;
* the rate trace is flat at the base through normal science, surges through the crisis
  cascade (adaptive inadequacy x social excitation), and decays back once the wake explains
  the residual;
* the audit never prunes the genuine hub (no false discoveries to control), and the concept
  SURVIVES fusion population-wide (Hawkes near-simultaneity + masked pooling compose);
* the pre-crisis ledger still rejects essentially every pre-crisis wake;
* the Lakatos-gated dogmatic community still converts under weak coupling (accretion
  strengthens the gate but does not beat connection -- the paper's two-lever result holds
  under the full stack);
* the null world (no regime flip) fires nothing: no crisis, no wake, rate pinned at base.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register
from src.structural.models.kuhn_phlogiston import single_run


def _med(x):
    x = x[x >= 0]
    return int(np.median(x)) if len(x) else -1


def fig_combined(r, path) -> dict:
    comm = r["community"]
    o = comm == 0
    cs, es = r["crisis_step"], r["expand_step"]
    t_shift = r["t_shift"]
    t_crisis, t_disc = _med(cs[o]), _med(es[o])
    snap_t, oxy = r["snap_t"], r["oxy_index_sc"]
    T = int(snap_t[-1]) + 1
    tt = np.arange(r["rate_tn"].shape[0])

    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9.6, 10.2), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1.1, 1.0, 1.0]))
    # --- panel 1: the cycle, under the full stack ---
    a0.plot(snap_t, oxy[:, 0], lw=2.4, color="#1e8449", label="open community")
    a0.plot(snap_t, oxy[:, 1], lw=2.4, color="#922b21",
            label="dogmatic community (Lakatos-gated, weakly coupled)")
    for x, lab in ((t_shift, "world flips"), (t_crisis, "median crisis"),
                   (t_disc, "median discovery")):
        if x and x > 0:
            a0.axvline(x, color="k", lw=0.8, alpha=0.6)
            a0.text(x, 1.04, f" {lab}", fontsize=7.5, rotation=90, va="bottom")
    a0.set_ylabel("oxygen index")
    a0.set_ylim(-0.04, 1.3)
    a0.legend(fontsize=8, loc="center left")
    a0.set_title("the full cycle with every refinement enabled at once: Hawkes+adaptive "
                 "rate, expiring-credit audit,\ndimension-aware fusion, Lakatos conviction, "
                 "pre-crisis measurement")

    # --- panel 2: the rate as a composed belief ---
    rt = r["rate_tn"][:, o]
    a1.plot(tt, rt.mean(axis=1), lw=2.2, color="#1f618d", label="proposal rate (open, mean)")
    a1.fill_between(tt, np.percentile(rt, 10, axis=1), np.percentile(rt, 90, axis=1),
                    color="#1f618d", alpha=0.18)
    a1b = a1.twinx()
    with np.errstate(all="ignore"):
        fl = np.nanmean(r["floor_tn"][:, o], axis=1)
    a1b.plot(tt, fl, lw=1.6, color="#b9770e", alpha=0.85, label="residual floor (mean)")
    a1b.set_ylabel("residual floor", color="#b9770e")
    a1.set_ylabel("proposal rate $r_i(t)$")
    h0, l0 = a1.get_legend_handles_labels(); h1, l1 = a1b.get_legend_handles_labels()
    a1.legend(h0 + h1, l0 + l1, fontsize=7.5)
    a1.set_title("the rate: flat through normal science, surged by inadequacy (adaptive) "
                 "and neighbours' discoveries (Hawkes), decaying once the wake explains "
                 "the residual", fontsize=9)

    # --- panel 3: discovery, audit, survival ---
    frac = [(es[o][(es[o] >= 0) & (es[o] <= t)].size) / o.sum() for t in tt]
    a2.plot(tt, frac, lw=2.2, color="#1e8449", label="cumulative wakes (open)")
    surv = (r["oxy_coupling_tn"][:, o] > 0.02).mean(axis=1)
    a2.plot(tt, surv, lw=2.2, color="#6c3483",
            label="fraction holding an EFFECTIVE oxygen coupling")
    with np.errstate(all="ignore"):
        aud = np.nanmean(r["hub_dF_prune_tn"][:, o], axis=1)
    a2b = a2.twinx()
    a2b.plot(tt, aud, lw=1.4, color="#7b241c", alpha=0.8,
             label="audit verdict (dF + expiring credit)")
    a2b.axhline(0.0, color="#7b241c", ls=":", lw=1.0)
    a2b.set_ylabel("audit verdict", color="#7b241c")
    a2.set_xlabel("step"); a2.set_ylabel("fraction of open community")
    a2.set_ylim(-0.04, 1.04)
    h0, l0 = a2.get_legend_handles_labels(); h1, l1 = a2b.get_legend_handles_labels()
    a2.legend(h0 + h1, l0 + l1, fontsize=7.5, loc="center left")
    a2.set_title("the audited discovery survives: masked fusion protects the staggered "
                 "wakes the excitation compresses, and the audit never fires on the "
                 "genuine hub", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(t_crisis=t_crisis, t_discovery=t_disc,
                rate_peak=float(rt.mean(axis=1).max()),
                survive_open=float(surv[-1]))


def run(out_dir, params: dict) -> None:
    base = dict(N=params["N_AGENTS"], inter=params["INTER"], omega=params["OMEGA"],
                sigma_o=params["SIGMA_O"], gate_strength=params["GATE_STRENGTH"],
                s_dogma=params["S_DOGMA"], n_steps=params["N_STEPS"],
                proposal_rate=params["R0"],
                rate_mode="hawkes_adaptive", hawkes_beta=params["HAWKES_BETA"],
                hawkes_tau=params["HAWKES_TAU"], rate_kappa=params["KAPPA"],
                rate_floor_ema=0.1, rate_floor_ref=params["FLOOR_REF"],
                rate_max=params["RATE_MAX"],
                wake_credit=True, prune_grace=params["PRUNE_GRACE"],
                prune_patience=params["PRUNE_PATIENCE"],
                credit_decay=params["CREDIT_DECAY"],
                fuse_mode="posterior_masked",
                conviction_eps=params["CONVICTION_EPS"], conviction_decay=0.01,
                precrisis_check=True, snapshot_every=20)

    rows, r_show = [], None
    for s in params["SEEDS"]:
        r = single_run(t_shift=params["T_SHIFT"], seed=s, **base)
        if r_show is None:
            r_show = r
        comm = r["community"]; o, d = comm == 0, comm == 1
        cs, es, ps = r["crisis_step"], r["expand_step"], r["prune_back_step"]
        pre = r["precrisis_dF_tn"]
        rt = r["rate_tn"]
        surv_open = float((r["oxy_coupling_tn"][-1][o] > 0.02).mean())
        row = dict(seed=s,
                   open_crisis_frac=float((cs[o] >= 0).mean()),
                   open_wake_frac=float((es[o] >= 0).mean()),
                   open_pruneback_frac=float((ps[o] >= 0).mean()),
                   open_survive_frac=surv_open,
                   dogma_oxy_end=float(r["oxy_index_sc"][-1, 1]),
                   rate_preshift=float(rt[: params["T_SHIFT"]].mean()),
                   rate_peak=float(rt[:, o].mean(axis=1).max()),
                   rate_end=float(rt[-1].mean()),
                   precrisis_frac_pos=float((pre > 0).sum() / max(np.isfinite(pre).sum(), 1)),
                   finite=bool(np.isfinite(r["oxy_coupling_tn"]).all()))
        rows.append(row)
        print(f"  seed {s}: crisis {row['open_crisis_frac']:.2f} wake "
              f"{row['open_wake_frac']:.2f} survive {row['open_survive_frac']:.2f} "
              f"pruneback {row['open_pruneback_frac']:.2f} dogma_oxy "
              f"{row['dogma_oxy_end']:.2f} rate {row['rate_preshift']:.3f}->"
              f"{row['rate_peak']:.2f}->{row['rate_end']:.3f}", flush=True)

        assert row["finite"], "the composed stack must stay numerically healthy"
        assert row["open_crisis_frac"] == 1.0 and row["open_wake_frac"] == 1.0, \
            "the full cycle must complete under the composed stack"
        assert row["open_pruneback_frac"] == 0.0, \
            "the audit must never prune the genuine hub"
        assert row["open_survive_frac"] > 0.9, \
            "excitation + masked fusion must carry the concept population-wide"
        assert abs(row["rate_preshift"] - params["R0"]) < 0.3 * params["R0"], \
            "normal science must keep the base rate"
        assert row["rate_end"] < 0.5 * row["rate_peak"], \
            "the rate must decay once the residual is explained"
        assert row["precrisis_frac_pos"] < 1e-3, \
            "the pre-crisis ledger must keep rejecting under the full stack"
        assert row["dogma_oxy_end"] > 0.5, \
            "weak coupling must still convert the Lakatos-gated community (two levers)"

    # ---- the null control: no flip => nothing fires anywhere in the stack ----
    r_null = single_run(t_shift=10 * params["N_STEPS"], seed=params["SEEDS"][0], **base)
    assert (r_null["crisis_step"] < 0).all() and (r_null["expand_step"] < 0).all(), \
        "null world: no crisis, no discovery under the full stack"
    assert abs(r_null["rate_tn"].mean() - params["R0"]) < 0.3 * params["R0"], \
        "null world: the rate must stay at base"

    info = fig_combined(r_show, out_dir / "fig_combined_mechanisms.png")

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        snap_t=r_show["snap_t"], oxy_index_sc=r_show["oxy_index_sc"],
        rate_tn=r_show["rate_tn"], floor_tn=r_show["floor_tn"],
        oxy_coupling_tn=r_show["oxy_coupling_tn"],
        hub_dF_prune_tn=r_show["hub_dF_prune_tn"],
        precrisis_dF_tn=r_show["precrisis_dF_tn"],
        crisis_step=r_show["crisis_step"], expand_step=r_show["expand_step"],
        prune_back_step=r_show["prune_back_step"], community=r_show["community"],
        null_rate_tn=r_null["rate_tn"],
    )
    summary = {"config": params, "fig": info, "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sv = np.mean([r["open_survive_frac"] for r in rows])
    print(f"HEADLINE: the refinements COMPOSE -- with the Hawkes+adaptive rate, the "
          f"expiring-credit audit, dimension-aware fusion, Lakatos conviction and the "
          f"pre-crisis measurement all enabled, the full Kuhn cycle still completes "
          f"(crisis t~{info['t_crisis']}, discovery t~{info['t_discovery']}), the rate "
          f"surges {params['R0']} -> {info['rate_peak']:.2f} -> base, the audit never "
          f"fires on the genuine hub, the concept survives population-wide "
          f"({sv:.0%}), the pre-crisis ledger keeps rejecting, and the null world is "
          f"silent. No mechanism breaks another.")


register(ExperimentSpec(
    model="phlogiston",
    name="combined_mechanisms",
    description="The composition check: every limitation-closing mechanism (Hawkes+adaptive "
                "rate, wake-then-prune audit, dimension-aware fusion, Lakatos conviction, "
                "pre-crisis measurement) enabled in ONE run -- the full Kuhn cycle "
                "completes, nothing breaks, the null world is silent.",
    run=run,
    out_dir="combined_mechanisms",
    params=dict(
        N_AGENTS=80, INTER=0.01, T_SHIFT=40, N_STEPS=320, OMEGA=0.97, SIGMA_O=0.5,
        GATE_STRENGTH=1.0, S_DOGMA=3.0,
        R0=0.02, HAWKES_BETA=0.5, HAWKES_TAU=10.0, KAPPA=0.2, FLOOR_REF=0.3, RATE_MAX=0.8,
        PRUNE_GRACE=15, PRUNE_PATIENCE=5, CREDIT_DECAY=0.85,
        CONVICTION_EPS=0.3,
        SEEDS=(0, 1, 2),
    ),
    seeds=(0, 1, 2),
    canonical=False,
    consumes=dict(figures=["fig_combined_mechanisms"]),
))
