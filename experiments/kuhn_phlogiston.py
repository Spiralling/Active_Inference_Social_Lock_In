"""The chemical revolution, endogenously: population + BMR + BME in one loop.

The integration experiment: N=80 agents in two communities (open vs dogmatic) live through the
deterministic regime flip of the phlogiston world (the precision balance arrives at t_shift:
calx is heavier). Crisis is the model's OWN prune evidence (Savage-Dickey Delta F crossing the
conviction protection lam_i * v_e); revolution is Bayesian model reduction APPLIED (the agent
adopts the CPD-reduced prior over the edges its own evidence flags, hub released); discovery is
Bayesian model expansion (a genuinely unconceived OXYGEN slot, bordered on a residual-triggered,
optionally Poisson-arriving proposal scored by the model log Bayes factor); and the population
fuses precision over an SBM graph, so awakened oxygen structure spreads socially.

Figures:
* ``kuhn_phlogiston_timeline.png`` -- the headline (disconnected communities, deterministic
  proposals): oxygen index with the Kuhn phases; belt Delta F vs each community's protection
  line (the anomaly curve IS the prune evidence; crisis = the crossing); the discovery wave
  (residual floor, cumulative oxygen adoption, per-agent crisis/expansion raster).
* ``kuhn_phlogiston_phase.png`` -- who converts, who locks in: social coupling x dogmatic gate
  strength. The two-lever result with the full machinery: conviction gating starves an isolated
  community's own evidence forever, but ANY social coupling imports the neighbours' evidence
  through fusion -- gating cannot defeat connection; only isolation protects the paradigm.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register
from src.structural.models.kuhn_phlogiston import SUSTAIN, single_run


def _med(x):
    x = x[x >= 0]
    return int(np.median(x)) if len(x) else -1


# ----------------------------------------------------------------------
# Fig A: the timeline
# ----------------------------------------------------------------------

def fig_timeline(r, path) -> dict:
    comm = r["community"]
    t_shift = r["t_shift"]
    cs, es = r["crisis_step"], r["expand_step"]
    t_crisis = _med(cs[comm == 0])
    t_disc = _med(es[comm == 0])
    snap_t = r["snap_t"]
    oxy = r["oxy_index_sc"]
    arrived = np.nonzero((snap_t > t_disc) & (oxy[:, 0] > 0.9))[0]
    t_new = int(snap_t[arrived[0]]) if len(arrived) else None
    T = int(snap_t[-1]) + 1

    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9.4, 9.6), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1.12, 1.0, 1.0]))
    # --- panel 1: oxygen index + phases ---
    a0.plot(snap_t, oxy[:, 0], lw=2.4, color="#1e8449", label="open community")
    a0.plot(snap_t, oxy[:, 1], lw=2.4, color="#922b21", label="dogmatic community (gated)")
    spans = [(0, t_shift, "#eaf2f8", "normal science\n(phlogiston fits)"),
             (t_shift, t_crisis, "#fdebd0", "the gravimetric\nanomaly"),
             (t_crisis, t_disc, "#fadbd8", "crisis:\nreduction"),
             (t_disc, t_new, "#e8daef", "discovery:\noxygen"),
             (t_new, T, "#eafaf1", "new normal")]
    for x0, x1, color, label in spans:
        if x0 is None or x1 is None or x1 <= x0:
            continue
        a0.axvspan(x0, x1, color=color, zorder=0)
        a0.text((x0 + x1) / 2, 1.06, label, ha="center", va="bottom", fontsize=7.5)
    a0.set_ylabel("oxygen index\n(0 = phlogiston, 1 = oxygen)")
    a0.set_ylim(-0.04, 1.24)
    a0.legend(fontsize=8, loc="center left")
    a0.set_title("The chemical revolution, endogenously: population + model reduction + expansion")

    # --- panel 2: the prune evidence (the anomaly curve IS Delta F) ---
    tt = np.arange(r["dF_belt_tn"].shape[0])
    ve = float(r["v_e_belt"].min())
    for label, c, color in (("open", 0, "#1e8449"), ("dogmatic", 1, "#922b21")):
        with np.errstate(all="ignore"):                 # expanded agents are NaN-masked
            m = np.nanmean(r["dF_belt_tn"][:, comm == c], axis=1)
        a1.plot(tt, m, lw=2.0, color=color, label=f"{label}: belt $\\Delta F$ (own evidence)")
    lam_o = float(r["lam"][comm == 0][0]); lam_d = float(r["lam"][comm == 1][0])
    a1.axhline(lam_o * ve, color="#1e8449", ls="--", lw=1.0,
               label=r"open protection $\lambda_o v_e$")
    a1.axhline(lam_d * ve, color="#922b21", ls="--", lw=1.0,
               label=r"dogmatic protection $\lambda_d v_e$")
    a1.axvline(t_shift, color="k", lw=0.8)
    a1.scatter(cs[cs >= 0], np.full((cs >= 0).sum(), lam_o * ve), marker="v", s=22,
               color="#c0392b", zorder=5, label="per-agent crisis (BMR applied)")
    a1.set_ylabel("Savage-Dickey prune\nevidence on the belt")
    a1.legend(fontsize=7, ncol=2)
    a1.set_title("crisis is the model's own evidence crossing its conviction protection -- "
                 "the gated community starves its own detector")

    # --- panel 3: the discovery wave ---
    fl = r["floor_tn"][:, comm == 0]
    a2.plot(tt, fl.mean(axis=1), lw=1.8, color="#b9770e",
            label="residual floor after reduction (open)")
    a2.axhline(1.5, color="k", ls=":", lw=1.0, label="expansion trigger")
    a2.set_ylabel("residual floor\n(hub proposal strength)")
    a2b = a2.twinx()
    for label, c, color in (("open", 0, "#1e8449"), ("dogmatic", 1, "#922b21")):
        esc = es[comm == c]
        frac = [(esc[(esc >= 0) & (esc <= t)].size) / esc.size for t in tt]
        a2b.plot(tt, frac, lw=2.2, color=color, ls="-",
                 label=f"{label}: fraction that discovered oxygen")
    a2b.set_ylabel("cumulative oxygen adoption"); a2b.set_ylim(-0.04, 1.04)
    a2.axvline(t_shift, color="k", lw=0.8)
    a2.set_xlabel("step")
    h0, l0 = a2.get_legend_handles_labels(); h1, l1 = a2b.get_legend_handles_labels()
    a2.legend(h0 + h1, l0 + l1, fontsize=7, loc="center left")
    a2.set_title("after reduction the residual the old paradigm absorbed becomes visible -- "
                 "and the unconceived oxygen node is bordered (model expansion)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(t_shift=t_shift, t_crisis=t_crisis, t_discovery=t_disc, t_new=t_new)


# ----------------------------------------------------------------------
# Fig B: the two-lever phase panel
# ----------------------------------------------------------------------

def fig_phase(rows, inters, s_dogmas, path) -> dict:
    conv = np.zeros((len(s_dogmas), len(inters)))
    crisis = np.zeros((len(s_dogmas), len(inters)))
    adopt = np.zeros((len(s_dogmas), len(inters)))
    for si, s in enumerate(s_dogmas):
        for ii, it in enumerate(inters):
            rs = [r for r in rows if r["s_dogma"] == s and r["inter"] == it]
            conv[si, ii] = np.mean([r["dogma_oxy_end"] for r in rs])
            crisis[si, ii] = np.mean([r["dogma_crisis_frac"] for r in rs])
            adopt[si, ii] = np.mean([r["dogma_struct_frac"] for r in rs])
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 3.8))
    panels = [(conv, "final oxygen index (dogmatic)", "inferno"),
              (crisis, "fraction reaching crisis (dogmatic)", "inferno"),
              (adopt, "fraction holding oxygen structure (dogmatic)", "inferno")]
    for ax, (M, title, cmap) in zip(axs, panels):
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="lower")
        ax.set_xticks(range(len(inters))); ax.set_xticklabels(inters)
        ax.set_yticks(range(len(s_dogmas))); ax.set_yticklabels(s_dogmas)
        ax.set_xlabel("social coupling (inter)")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
    axs[0].set_ylabel("dogmatic gate scaling $s$")
    fig.suptitle("lock-in needs BOTH levers: gating starves an isolated community's own evidence, "
                 "but any coupling imports the neighbours' evidence through fusion", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(conv=conv.tolist(), crisis=crisis.tolist(), adopt=adopt.tolist())


# ----------------------------------------------------------------------
# the experiment
# ----------------------------------------------------------------------

def run(out_dir, params: dict) -> None:
    base = dict(N=params["N_AGENTS"], gate_strength=params["GATE_STRENGTH"],
                s_open=params["S_OPEN"], lam_open=params["LAM_OPEN"],
                lam_dogma=params["LAM_DOGMA"], omega=params["OMEGA"],
                sigma_o=params["SIGMA_O"], t_shift=params["T_SHIFT"],
                n_steps=params["N_STEPS"], trigger=params["TRIGGER"])

    # ---- headline: deterministic world + deterministic proposals, isolated communities ----
    r = single_run(inter=0.0, s_dogma=params["S_DOGMA"], proposal_rate=None, seed=0, **base)
    info_a = fig_timeline(r, out_dir / "kuhn_phlogiston_timeline.png")

    comm, cs, es = r["community"], r["crisis_step"], r["expand_step"]
    t_shift = params["T_SHIFT"]
    o, d = comm == 0, comm == 1
    assert r["dF_belt_tn"][:t_shift].max() < 0, "normal science must be quiet (dF < 0 pre-shift)"
    assert (cs[o] >= 0).all() and cs[cs >= 0].min() > t_shift, \
        "the open community must reach crisis, and only after the anomaly"
    assert (es[o] >= 0).mean() > 0.8, "the open community should discover oxygen"
    assert ((es[es >= 0] > cs[es >= 0]).all()), "expansion must follow reduction"
    assert (cs[d] < 0).all() and (es[d] < 0).all(), \
        "the isolated dogmatic community must never crisis nor discover"
    assert r["oxy_index_sc"][-1, 0] > 0.9 > r["oxy_index_sc"][-1, 1], \
        "open converts, isolated dogmatic does not (fully)"

    # ---- controls ----
    r_null = single_run(inter=0.0, s_dogma=params["S_DOGMA"], proposal_rate=None, seed=0,
                        **{**base, "t_shift": params["N_STEPS"] + 10})
    assert (r_null["crisis_step"] < 0).all() and (r_null["expand_step"] < 0).all(), \
        "null world (truth never flips): no crisis, no discovery"
    r_noexp = single_run(inter=0.0, s_dogma=params["S_DOGMA"], proposal_rate=None, seed=0,
                         enable_expansion=False, **base)
    assert (r_noexp["expand_step"] < 0).all() and (r_noexp["crisis_step"][o] >= 0).all(), \
        "expansion-off ablation: crisis still fires, oxygen never bordered"

    # ---- Poisson proposals: discovery as a waiting time (3 seeds) ----
    delays = []
    for s in params["SEEDS"]:
        rp = single_run(inter=0.0, s_dogma=params["S_DOGMA"],
                        proposal_rate=params["PROPOSAL_RATE"], seed=s, **base)
        csp, esp = rp["crisis_step"], rp["expand_step"]
        ok = (csp >= 0) & (esp >= 0)
        assert ok.any(), f"Poisson seed {s}: nobody completed crisis->discovery"
        delays.append(float(np.mean(esp[ok] - csp[ok])))
    mean_delay = float(np.mean(delays))
    lo = SUSTAIN
    hi = lo + 3.0 / params["PROPOSAL_RATE"]
    assert lo <= mean_delay <= hi, \
        f"Poisson discovery delay should be a waiting time (~sustain + 1/rate): {mean_delay:.1f}"

    # ---- Fig B sweep ----
    inters = list(params["INTERS"]); s_dogmas = list(params["S_DOGMAS"])
    rows = []
    for s_d in s_dogmas:
        for it in inters:
            for s in params["SEEDS"]:
                rr = single_run(inter=float(it), s_dogma=float(s_d),
                                proposal_rate=params["PROPOSAL_RATE"], seed=s, **base)
                dd = rr["community"] == 1
                rows.append(dict(
                    s_dogma=s_d, inter=it, seed=s,
                    dogma_oxy_end=float(rr["oxy_index_sc"][-1, 1]),
                    dogma_crisis_frac=float((rr["crisis_step"][dd] >= 0).mean()),
                    dogma_struct_frac=float((rr["oxy_coupling_tn"][-1][dd] > 0.02).mean()),
                ))
                print(f"  s_dogma={s_d} inter={it} seed={s}: oxy {rows[-1]['dogma_oxy_end']:.2f} "
                      f"crisis {rows[-1]['dogma_crisis_frac']:.2f}", flush=True)
    info_b = fig_phase(rows, inters, s_dogmas, out_dir / "kuhn_phlogiston_phase.png")

    conv = np.asarray(info_b["conv"])
    assert conv[-1, 0] < 0.75, "isolated + heavily gated must stay phlogiston"
    assert conv[-1, -1] > 0.9, "strong coupling must convert even the heavily gated"

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        snap_t=r["snap_t"], oxy_index_sc=r["oxy_index_sc"],
        dF_belt_tn=r["dF_belt_tn"], floor_tn=r["floor_tn"],
        oxy_coupling_tn=r["oxy_coupling_tn"],
        crisis_step=r["crisis_step"], expand_step=r["expand_step"],
        n_pruned=r["n_pruned"], community=r["community"],
        v_e_belt=r["v_e_belt"], lam=r["lam"],
        phase_inters=np.array(inters, dtype=np.float64),
        phase_s_dogmas=np.array(s_dogmas, dtype=np.float64),
        phase_conv=np.asarray(info_b["conv"]),
        phase_crisis=np.asarray(info_b["crisis"]),
        phase_adopt=np.asarray(info_b["adopt"]),
        poisson_delays=np.array(delays),
    )
    summary = {"config": params, "timeline": info_a, "phase": info_b,
               "poisson_mean_delay": mean_delay,
               "open_crisis_med": _med(cs[o]), "open_discovery_med": _med(es[o]),
               "n_pruned_med": int(np.median(r["n_pruned"][o]))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: the full Kuhn cycle with the model's own machinery -- anomaly at "
          f"t={t_shift}, crisis (Delta F crosses lambda v_e) at t={info_a['t_crisis']}, "
          f"reduction prunes ~{int(np.median(r['n_pruned'][o]))} edges, oxygen bordered at "
          f"t={info_a['t_discovery']} (Poisson delay ~{mean_delay:.0f} steps when stochastic); "
          f"the gated community never sees its own crisis but ANY social coupling converts it: "
          f"lock-in = gating + isolation.")


register(ExperimentSpec(
    model="phlogiston",
    name="kuhn_phlogiston",
    description="The chemical revolution, endogenously: population + Bayesian model reduction "
                "+ expansion in one loop. Crisis = the agent's own Savage-Dickey evidence "
                "crossing its conviction protection; revolution = the reduction applied; "
                "discovery = the unconceived oxygen node bordered on a residual-triggered, "
                "Poisson-arriving proposal; fusion spreads the new structure socially. Lock-in "
                "requires gating AND isolation.",
    run=run,
    out_dir="kuhn_phlogiston",
    params=dict(
        N_AGENTS=80, T_SHIFT=40, N_STEPS=180, OMEGA=0.97, SIGMA_O=0.5,
        LAM_OPEN=0.10, LAM_DOGMA=0.35, GATE_STRENGTH=1.0, S_OPEN=0.3, S_DOGMA=3.0,
        TRIGGER=1.5, PROPOSAL_RATE=0.08,
        INTERS=(0.0, 0.01, 0.05, 0.2), S_DOGMAS=(1.0, 3.0, 6.0),
        SEEDS=(0, 1, 2),
    ),
    seeds=(0, 1, 2),
    canonical=False,
    consumes=dict(figures=["kuhn_phlogiston_timeline", "kuhn_phlogiston_phase"]),
))
