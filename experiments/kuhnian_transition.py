"""The Kuhnian transition: communities on real topology + the Lakatos ladder.

Two questions on top of ``rival_kuhn`` (the cycle on the represented-rival engine):

DEMO A -- does "Kuhnian PHASE TRANSITION" earn its name? A vanguard community and a
conservative community (zero own evidence source post-shift) joined by bridges of
density beta: the order parameter (conservatives' final q(oxygen)) vs beta. The name
is EARNED BY MEASUREMENT: the transition width (decades of beta between the 10% and
90% crossings) is recorded in summary.json with a verdict (sharp <= 0.5 decades /
moderate <= 1 / smooth). Honest mechanics: with log-linear pooling and a persistent
vanguard source, any connected beta converts EVENTUALLY -- the fixed-horizon
transition is a conversion-time threshold crossing the horizon, the same epistemic
status as the paper's lock-in boundary.

DISCOVERED EN ROUTE (probe, seed 0; the second curve in fig 1): for a conservative
community that retains a TRICKLE of its own disconfirming evidence (gamma = 0.99),
contact is NON-MONOTONE -- isolated, it converts by itself (t = 432); weakly bridged,
it NEVER converts. Mechanism: hypothesis-aligned NET fusion across the few bridges
imports the vanguard's ACCOMMODATED incumbent parameters (its phlogiston candidate
has already absorbed the anomaly), which explain away the conservatives' own
anomalies and kill their source. WEAK CONTACT IMMUNIZES; STRONG CONTACT CONVERTS.
(The same mechanism explains why rival_kuhn's no-pooling conservatives stayed locked
despite their trickle.)

DEMO B -- core vs periphery, revived at the HYPOTHESIS level: a K = 3 candidate
ladder. M_full = the incumbent (hub wiring + phlogiston mass means); M_artic = the
ARTICULATED incumbent -- the same hub + belt wiring with the mass-law means patched
to accommodate the anomaly (negative-weight phlogiston, the historically real
protective-belt move); M_oxy = oxygen (no hub). After the world turns, the
population retreats BELT-FIRST (fast flip to M_artic on mean evidence -- the anomaly
accommodated, the core preserved) and abandons the CORE LAST (slow flip to M_oxy on
structure evidence -- the hub's correlations are unsupported by the world). The
belt/core staircase as a sequence of MODEL retreats: Lakatos's protective belt,
measured. Timescale separation calibrated: fast leg ~77 steps, plateau ~376 steps,
core abandoned at t ~ 533 (omega = 0.5 keeps the structural signal visible -- the
within-frame Quine-Duhem masking measured in represented_rivals Demo A).
"""
from __future__ import annotations

import json

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from experiments.registry import ExperimentSpec, register

RED, AMBER, GREEN, GREY = "#922b21", "#b9770e", "#1e8449", "#7f8c8d"


# ----------------------------------------------------------------------
# Candidates + world.
# ----------------------------------------------------------------------

def articulated_bn(cfg, conviction: float = 2.0):
    """Negative-weight phlogiston: phlogiston_bn's hub + belt WIRING with the
    disagreement-node means patched to ``mu_oxy_mass`` -- the anomaly accommodated,
    the hard core preserved. PD by the CPD compile."""
    from src.structural.bayesnet import LinearGaussianBN
    from src.structural.phlogiston import (DISAGREEMENT_NODES, HUB,
                                           HUB_NEIGHBOURS, _idx)
    names = cfg.node_names
    d = len(names)
    idx = _idx(cfg)
    B = jnp.zeros((d, d))
    for n in HUB_NEIGHBOURS:
        B = B.at[idx[n], idx[HUB]].set(cfg.hub_coupling)
    anom = idx["calx_heavier_than_metal"]
    for n in ("mass_change_sign", "gas_consumed"):
        B = B.at[anom, idx[n]].set(cfg.mass_coupling)
    target = jnp.full((d,), cfg.mu_agree).at[idx[HUB]].set(0.0)
    for n in DISAGREEMENT_NODES:
        target = target.at[idx[n]].set(cfg.mu_oxy_mass)
    b = (jnp.eye(d) - B) @ target
    s = jnp.full((d,), 1.0 / conviction)
    s = s.at[idx[HUB]].set(1.0 / (conviction * cfg.hub_self_prec))
    return LinearGaussianBN(B=B, b=b, s=s, names=names)


def _build(p, n_steps, sigma):
    from experiments.rival_kuhn import oxygen_bn
    from src.structural.phlogiston import (
        StructuralConfig, phlogiston_bn, gravimetric_H, gravimetric_rows,
        phi_true_at, DISAGREEMENT_NODES)
    cfg = StructuralConfig(regime_schedule="step", t_shift=p["T_SHIFT"],
                           n_steps=n_steps, sigma_o=sigma)
    nets = (phlogiston_bn(cfg, conviction=2.0).to_info(),
            articulated_bn(cfg).to_info(),
            oxygen_bn(cfg).to_info())
    for k, n in enumerate(nets):
        e = float(np.linalg.eigvalsh(np.asarray(n.Pi)).min())
        assert e > 0, f"PD contract violated for candidate {k}: {e}"
    H = gravimetric_H(cfg)
    rows = gravimetric_rows(cfg)
    disc = [i for i, r in enumerate(rows)
            if ("mass_balance" in r) or (r in DISAGREEMENT_NODES)]
    phis = jnp.stack([phi_true_at(cfg, t) for t in range(n_steps)])
    return nets, H, disc, phis


def _flip(trace, thresh):
    hit = trace > thresh
    return int(np.argmax(hit)) if hit.any() else -1


# ----------------------------------------------------------------------
# Demo A: the transition.
# ----------------------------------------------------------------------

def _run_transition(p):
    from src.structural import graphs
    from src.structural import rival as rv
    T, ts = p["T_TRANS"], p["T_SHIFT"]
    nets, H, disc, phis = _build(p, T, p["SIGMA_TRANS"])
    K2 = (nets[0], nets[2])                                 # full vs oxygen
    m = H.shape[0]
    nv, nc = p["N_VAN"], p["N_CONS"]
    N = nv + nc
    cfgr = rv.RivalConfig(Pi0=jnp.stack([n.Pi for n in K2]),
                          h0=jnp.stack([n.h for n in K2]), H=H,
                          sigma_o=p["SIGMA_TRANS"], omega=p["OMEGA_TRANS"],
                          alpha_m=p["ALPHA_M"])
    out = {}
    for gcons in (1.0, p["GAMMA_TRICKLE"]):
        w2 = np.ones((N, m))
        w2[nv:, disc] = 1.0 - gcons
        rowsg = []
        for beta in p["BETAS"]:
            g = graphs.community([nv, nc], intra=1.0, inter=float(beta), seed=0)
            Av = np.asarray(g.A)
            W = g.trust_W()
            A_self = jnp.asarray(Av + np.eye(N))
            q_end, flips = [], []
            for seed in p["SEEDS"]:
                r1 = rv.run_rival(cfgr, W, A_self, jnp.ones((N, m)), phis[:ts],
                                  seed=seed)
                r2 = rv.run_rival(cfgr, W, A_self, jnp.asarray(w2), phis[ts:],
                                  seed=seed + 1000, Pi_init=r1["Pi"],
                                  h_init=r1["h"], L_init=r1["L"])
                qc = np.concatenate([r1["q_t"], r2["q_t"]])[:, nv:, 1].mean(axis=1)
                q_end.append(float(qc[-1]))
                flips.append(_flip(qc, p["Q_LOCK"]))
            rowsg.append(dict(beta=float(beta),
                              bridges=int(Av[:nv, nv:].sum()),
                              lambda2=graphs.algebraic_connectivity(g),
                              q_end=np.asarray(q_end), flips=flips))
        out[gcons] = rowsg
    return out


def _width_decades(rows):
    """Log10-beta width between the 10% and 90% crossings of the seed-mean order
    parameter (beta > 0 points only; linear interpolation in log space)."""
    b = np.array([r["beta"] for r in rows if r["beta"] > 0])
    q = np.array([r["q_end"].mean() for r in rows if r["beta"] > 0])
    lx = np.log10(b)

    def crossing(level):
        for i in range(len(q) - 1):
            if (q[i] - level) * (q[i + 1] - level) <= 0 and q[i] != q[i + 1]:
                f = (level - q[i]) / (q[i + 1] - q[i])
                return lx[i] + f * (lx[i + 1] - lx[i])
        return None

    lo, hi = crossing(0.1), crossing(0.9)
    return (hi - lo) if (lo is not None and hi is not None) else None


# ----------------------------------------------------------------------
# Demo B: the ladder.
# ----------------------------------------------------------------------

def _run_ladder(p):
    from src.structural import rival as rv
    T = p["T_LADDER"]
    nets, H, disc, phis = _build(p, T, p["SIGMA_LADDER"])
    N = p["N_LADDER"]
    cfgr = rv.RivalConfig(Pi0=jnp.stack([n.Pi for n in nets]),
                          h0=jnp.stack([n.h for n in nets]), H=H,
                          sigma_o=p["SIGMA_LADDER"], omega=p["OMEGA_LADDER"],
                          alpha_m=p["ALPHA_M"])
    W = jnp.ones((N, N)) / N
    A = jnp.ones((N, N))
    qs = np.stack([rv.run_rival(cfgr, W, A, jnp.ones((N, H.shape[0])), phis,
                                seed=s)["q_t"].mean(axis=1)
                   for s in p["SEEDS"]])                    # (S, T, 3)
    return qs


# ----------------------------------------------------------------------
# The experiment.
# ----------------------------------------------------------------------

def run(out_dir, params: dict) -> None:
    p = dict(params)
    ts = p["T_SHIFT"]
    print(f"kuhnian transition: communities on real topology + the K=3 Lakatos "
          f"ladder; seeds={tuple(p['SEEDS'])}")

    # ---------------- Demo A ----------------
    trans = _run_transition(p)
    clean, trickle = trans[1.0], trans[p["GAMMA_TRICKLE"]]
    print("\nDemo A (order parameter = conservatives' final q(oxy), seed mean):")
    for tag, rows in (("gamma_cons=1.0 (clean)", clean),
                      (f"gamma_cons={p['GAMMA_TRICKLE']} (trickle)", trickle)):
        print(f"  {tag}: " + "  ".join(
            f"b={r['beta']}:{r['q_end'].mean():.3f}" for r in rows))
    width = _width_decades(clean)
    verdict = ("sharp" if width is not None and width <= 0.5 else
               "moderate" if width is not None and width <= 1.0 else "smooth")
    print(f"  transition width = "
          f"{'unresolved' if width is None else f'{width:.2f} decades'} "
          f"-> verdict: {verdict} (criterion: sharp <= 0.5 decades)")
    qe = {r["beta"]: r["q_end"].mean() for r in clean}
    qt = {r["beta"]: r["q_end"].mean() for r in trickle}
    b_lo, b_hi = min(p["BETAS"]), max(p["BETAS"])
    print(f"  immunization (trickle curve): isolated q={qt[b_lo]:.3f} vs weakly "
          f"bridged q={qt[0.01]:.3f} vs strongly bridged q={qt[b_hi]:.3f}")

    # ---------------- Demo B ----------------
    qs = _run_ladder(p)
    qm = qs.mean(axis=0)                                    # (T, 3)
    dom = np.argmax(qm, axis=1)
    T = p["T_LADDER"]
    pre_full = float(qm[ts // 2:ts, 0].min())
    t_art = int(np.argmax(dom == 1)) if (dom == 1).any() else -1
    oxy_dom = (dom == 2) & (np.arange(T) > ts)
    t_oxy = int(np.argmax(oxy_dom)) if oxy_dom.any() else -1
    plateau = int((dom == 1).sum())
    print(f"\nDemo B (Lakatos ladder): q_full pre-shift min = {pre_full:.3f}; "
          f"belt sacrificed (artic dominant) t={t_art}; protective-belt plateau = "
          f"{plateau} steps; core abandoned (oxy dominant) t={t_oxy}; "
          f"final q = {np.round(qm[-1], 3).tolist()}")

    # ---------------- assertions (print-first; probe-calibrated) ----------------
    assert qe[b_lo] < 0.1, "t1: zero-source isolated community must never convert"
    assert qe[b_hi] > 0.9, "t2: strong bridging must convert fully"
    qvals = [qe[b] for b in p["BETAS"]]
    assert all(qvals[i + 1] >= qvals[i] - 0.05 for i in range(len(qvals) - 1)), \
        f"t3: the clean order parameter must be monotone in beta ({np.round(qvals, 3)})"
    # the discovered immunization effect (trickle community): contact is non-monotone
    assert qt[b_lo] > 0.9 and qt[0.01] < 0.1, \
        "t5: weak contact must IMMUNIZE the trickle community (the discovered effect)"
    assert pre_full > 0.9, "l1: normal science must form on the full incumbent"
    assert t_art > ts and plateau >= p["PLATEAU_MIN"], \
        f"l2: the protective-belt era must exist ({t_art}, {plateau})"
    assert t_oxy > t_art > 0, "l3: belt-first, core-last ordering"
    assert qm[-1, 2] > 0.9, f"l4: the core must fall by T ({qm[-1, 2]:.3f})"
    assert np.isfinite(qs).all(), "health"

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for rows, color, ls, lab in (
            (clean, GREEN, "-",
             "zero-source conservatives (clean order parameter)"),
            (trickle, AMBER, "--",
             f"trickle conservatives (gamma={p['GAMMA_TRICKLE']}): weak contact "
             f"IMMUNIZES")):
        b = [r["beta"] for r in rows]
        qmean = [r["q_end"].mean() for r in rows]
        qstd = [r["q_end"].std() for r in rows]
        ax.errorbar(b, qmean, yerr=qstd, marker="o", lw=2.2, capsize=3,
                    color=color, ls=ls, label=lab)
    ax.set_xscale("symlog", linthresh=0.003)
    ax.set_xlabel("bridge density beta between the communities (symlog)")
    ax.set_ylabel("conservatives' final q(oxygen)")
    ax.set_ylim(-0.04, 1.04)
    wtxt = "unresolved" if width is None else f"{width:.2f} decades"
    ax.set_title(f"the Kuhnian transition: conversion vs contact "
                 f"(width {wtxt} -> {verdict});\nand the discovered immunization: "
                 f"weak contact spreads the ACCOMMODATED incumbent", fontsize=10)
    ax.legend(fontsize=8, loc="center left")
    plt.tight_layout(); plt.savefig(out_dir / "fig_kuhnian_transition.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    t = np.arange(T)
    for k, (color, lab) in enumerate((
            (GREY, "M_full: hub + phlogiston means (the incumbent)"),
            (AMBER, "M_artic: hub kept, anomaly accommodated\n"
                    "(negative-weight phlogiston)"),
            (GREEN, "M_oxy: no hub (the rival core)"))):
        ax.plot(t, qm[:, k], color=color, lw=2.2, label=lab)
        ax.fill_between(t, qs[:, :, k].min(axis=0), qs[:, :, k].max(axis=0),
                        color=color, alpha=0.15)
    ax.axvline(ts, color="gray", ls=":", lw=1.5)
    for x, lab in ((ts * 0.45, "NORMAL\nSCIENCE"),
                   ((t_art + t_oxy) / 2, "PROTECTIVE BELT\n(belt sacrificed,\n"
                                         "core kept)"),
                   ((t_oxy + T) / 2, "CORE\nABANDONED")):
        ax.annotate(lab, (x, 0.55), ha="center", fontsize=8, color="#444444")
    ax.set_xlabel("step"); ax.set_ylabel("population q(m)"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("belt-first, core-last -- as a sequence of model retreats:\n"
                 "Lakatos's protective belt, measured", fontsize=10)
    ax.legend(fontsize=7.5, loc="center right")
    plt.tight_layout(); plt.savefig(out_dir / "fig_lakatos_ladder.png", dpi=130)
    plt.close(fig)

    # ---------------- save ----------------
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        betas=np.asarray(p["BETAS"], dtype=float),
        q_end_clean=np.stack([r["q_end"] for r in clean]),
        q_end_trickle=np.stack([r["q_end"] for r in trickle]),
        lambda2=np.asarray([r["lambda2"] for r in clean]),
        bridges=np.asarray([r["bridges"] for r in clean]),
        q_ladder=qs,
    )
    summary = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
        "transition": {
            "order_parameter_clean": {str(r["beta"]): float(r["q_end"].mean())
                                      for r in clean},
            "width_decades": width, "verdict": verdict,
            "criterion": "sharp <= 0.5 decades (10%-90% crossings in log10 beta)",
        },
        "immunization": {
            "order_parameter_trickle": {str(r["beta"]): float(r["q_end"].mean())
                                        for r in trickle},
            "finding": "non-monotone contact for a trickle-source community: "
                       "isolated converts by itself; weakly bridged NEVER converts "
                       "-- hypothesis-aligned net fusion imports the vanguard's "
                       "accommodated incumbent parameters, which explain away the "
                       "community's own anomalies (weak contact immunizes, strong "
                       "contact converts)",
        },
        "ladder": {"pre_full_min": pre_full, "t_belt_sacrificed": t_art,
                   "plateau_steps": plateau, "t_core_abandoned": t_oxy,
                   "final_q": qm[-1].tolist(),
                   "finding": "the belt/core staircase replicated as a sequence of "
                              "MODEL retreats: fast mean-evidence flip to the "
                              "articulated incumbent (anomaly accommodated, core "
                              "kept), slow structure-evidence flip to the rival "
                              "core -- Lakatos's protective belt, measured"},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"\nsaved arrays / 2 figures / summary to {out_dir}")
    print(f"HEADLINE: the Kuhnian transition measured on real topology -- a "
          f"zero-source conservative community converts only past a bridge-density "
          f"threshold (width {wtxt}: {verdict}); a community retaining its own "
          f"trickle of evidence shows the DISCOVERED immunization (weak contact "
          f"spreads the accommodated incumbent and kills conversion; strong contact "
          f"converts). And core-vs-belt returns at the hypothesis level: the "
          f"population sacrifices the belt fast (articulated incumbent dominant at "
          f"t={t_art}, a {plateau}-step protective-belt era) and abandons the core "
          f"only on slow structural evidence (t={t_oxy}) -- Lakatos's ladder, "
          f"measured.")


register(ExperimentSpec(
    model="phlogiston",
    name="kuhnian_transition",
    description="The Kuhnian transition on real community topology (conversion vs "
                "bridge density, with the measured transition width deciding "
                "whether 'phase transition' is earned) plus the discovered "
                "immunization effect (weak contact spreads the accommodated "
                "incumbent), and the K=3 Lakatos ladder: belt-first core-last as a "
                "sequence of model retreats.",
    run=run,
    out_dir="kuhnian_transition",
    params=dict(
        T_SHIFT=80,
        # CALIB (probe, seed 0): transition demo at sigma=1.0, omega=0.9 (the
        # rival_kuhn settings); gamma_cons=1.0 gives the clean monotone order
        # parameter, gamma=0.99 the non-monotone immunization curve.
        T_TRANS=480, SIGMA_TRANS=1.0, OMEGA_TRANS=0.9,
        N_VAN=8, N_CONS=16, GAMMA_TRICKLE=0.99,
        BETAS=(0.0, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3),
        # CALIB (probe, seed 0): ladder at sigma=0.5, omega=0.5 -- pre-shift
        # q_full=0.997, artic dominant t=157, plateau 376 steps, oxy dominant
        # t=533, final q_oxy=0.998 (omega larger masks the structural signal:
        # the within-frame Quine-Duhem effect).
        T_LADDER=800, SIGMA_LADDER=0.5, OMEGA_LADDER=0.5, N_LADDER=16,
        PLATEAU_MIN=100,
        ALPHA_M=0.02, Q_LOCK=0.9, SEEDS=(0, 1, 2),
    ),
    seeds=(0, 1, 2),
    consumes=dict(figures=["fig_kuhnian_transition", "fig_lakatos_ladder"]),
))
