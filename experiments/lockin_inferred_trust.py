"""Inferred trust at the lock-in boundary: distrust buys time, not a wall.

The v3 paper's headline lock-in result: with FIXED, content-blind trust, gating
cannot defeat connection -- at inter = 0.005 (trust toward outsiders ~200x weaker
than within) the dogmatic community converts almost fully, and only literal
isolation protects phlogiston. This experiment reruns that boundary with the
content-gated trust of ``reliability.social_gamma`` switched on in the full
Kuhn-cycle machinery (``single_run(social_nu=...)``): each round, trust toward a
neighbour is the Student-t weight of the belief disagreement on the mass-law
nodes, so a community that has diverged stops averaging with the other.

What actually happens (and what the figures show honestly):

* The gate WORKS: as the open community converts, cross-community gamma
  collapses to ~1% of within-community trust -- the dogmatists infer the
  outsiders are not worth listening to, with no isolation imposed by hand.
* But the memoryless gate buys METASTABILITY, not a fixed point: gamma > 0
  leaks a trickle of cross-evidence, the dogmatic community drifts, and as it
  drifts closer the gate REOPENS (rehabilitation) and the merge cascades. The
  paper's theorem survives -- any contact converts eventually -- and the price
  is quantified: the conversion timescale roughly DOUBLES at every coupling,
  and at the paper's boundary (inter = 0.005) conversion is left incomplete at
  the 180-step horizon (~0.54-0.73 vs 0.92 under fixed trust).
* The harsher the gate (smaller nu), the longer the lock holds -- and the less
  of the rehabilitation arc fits the horizon (nu = 0.2 holds ~0.4 conversion
  at the boundary but leaves trust unrehabilitated at the end). Lock strength
  and a completed merge trade off, because both are the same clock. The
  wall-like regime is real but needs CONFIDENCE, not coupling: see
  ``schism_threshold``, where camps above a prior-precision threshold sever
  trust outright -- mid-cycle dogmatists have not yet accumulated it.

Distrust is self-organized isolation, and it decays: the model now contains the
distrust -> plateau -> rehabilitation -> late-merge arc that fixed-trust fusion
provably cannot produce.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register


def _time_to_half(snap_t, oxy_c):
    """First snapshot time at which a community's oxygen index crosses 0.5."""
    hit = np.nonzero(oxy_c > 0.5)[0]
    return int(snap_t[hit[0]]) if hit.size else int(snap_t[-1]) + 1


def _gamma_trace(r, nu):
    """Post-hoc gamma read-out per snapshot: mean within- vs cross-community trust."""
    import jax.numpy as jnp
    from src.structural import reliability as rel
    from src.structural.phlogiston import DISAGREEMENT_NODES
    comm, names = r["community"], r["names"]
    ix = jnp.asarray([names.index(n) for n in DISAGREEMENT_NODES])
    cross = np.outer(comm == 0, comm == 1) | np.outer(comm == 1, comm == 0)
    within = ~cross & ~np.eye(comm.size, dtype=bool)
    gw, gc = [], []
    for s in range(len(r["snap_t"])):
        g = np.asarray(rel.social_gamma(jnp.asarray(r["snap_Pi"][s]),
                                        jnp.asarray(r["snap_h"][s]), ix, nu))
        gw.append(g[within].mean()); gc.append(g[cross].mean())
    return np.asarray(gw), np.asarray(gc)


def fig_boundary(rows, inters, path) -> dict:
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.2, 4.0))
    out = {}
    for nu, color, label in ((None, "#922b21", "fixed trust (the paper's regime)"),
                             (0.5, "#1a5276", "inferred trust (social_nu = 0.5)")):
        conv = [np.mean([r["oxy_end"] for r in rows if r["nu"] == nu and r["inter"] == it])
                for it in inters]
        t50 = [np.mean([r["t_half"] for r in rows if r["nu"] == nu and r["inter"] == it])
               for it in inters]
        a0.plot(inters, conv, "o-", lw=2.2, color=color, label=label)
        a1.plot(inters, t50, "o-", lw=2.2, color=color, label=label)
        out[f"conv_{nu}"] = conv; out[f"t_half_{nu}"] = t50
    a0.set_xscale("log"); a1.set_xscale("log")
    a0.set_xlabel("social coupling (inter)"); a1.set_xlabel("social coupling (inter)")
    a0.set_ylabel("final oxygen index (dogmatic)"); a0.set_ylim(-0.05, 1.05)
    a1.set_ylabel("time to half-conversion (dogmatic)")
    a0.axhline(0.5, color="k", ls=":", lw=0.8)
    a0.legend(fontsize=8); a1.legend(fontsize=8)
    a0.set_title("the lock-in boundary moves ~an order of magnitude", fontsize=9)
    a1.set_title("inferred distrust multiplies the conversion timescale", fontsize=9)
    fig.suptitle("content-gated trust at the lock-in boundary: distrust is self-organized "
                 "isolation -- and it buys time, not a wall", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(path, dpi=130); plt.close(fig)
    return out


def fig_rehabilitation(r, gw, gc, path) -> dict:
    comm, snap_t, oxy = r["community"], r["snap_t"], r["oxy_index_sc"]
    fig, a0 = plt.subplots(figsize=(9.0, 4.2))
    a0.plot(snap_t, oxy[:, 0], lw=2.2, color="#1e8449", label="open community (oxygen index)")
    a0.plot(snap_t, oxy[:, 1], lw=2.2, color="#922b21", label="dogmatic community (oxygen index)")
    a0.set_ylabel("oxygen index"); a0.set_xlabel("step"); a0.set_ylim(-0.04, 1.04)
    a1 = a0.twinx()
    ratio = gc / (gw + 1e-12)
    a1.plot(snap_t, ratio, lw=1.8, color="#7d3c98", ls="--",
            label="cross/within trust ratio (inferred)")
    a1.set_ylabel("cross-community trust (fraction of within)"); a1.set_ylim(-0.04, 1.3)
    h0, l0 = a0.get_legend_handles_labels(); h1, l1 = a1.get_legend_handles_labels()
    a0.legend(h0 + h1, l0 + l1, fontsize=8, loc="center left")
    a0.set_title("distrust -> plateau -> rehabilitation -> merge: the gate severs trust while "
                 "the camps disagree, and reopens as the laggards drift in", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(min_trust_ratio=float(ratio.min()), final_trust_ratio=float(ratio[-1]))


def run(out_dir, params: dict) -> None:
    from src.structural.models.kuhn_phlogiston import single_run

    base = dict(N=params["N_AGENTS"], s_dogma=params["S_DOGMA"],
                proposal_rate=params["PROPOSAL_RATE"], n_steps=params["N_STEPS"])
    inters, seeds = list(params["INTERS"]), list(params["SEEDS"])

    rows, showcase = [], None
    for nu in (None, params["SOCIAL_NU"]):
        for it in inters:
            for s in seeds:
                r = single_run(inter=float(it), seed=s, social_nu=nu, **base)
                oxy_end = float(r["oxy_index_sc"][-1, 1])
                t_half = _time_to_half(r["snap_t"], r["oxy_index_sc"][:, 1])
                rows.append(dict(nu=nu, inter=it, seed=s, oxy_end=oxy_end, t_half=t_half))
                print(f"  nu={nu} inter={it} seed={s}: dogma oxy {oxy_end:.2f} "
                      f"t_half {t_half}", flush=True)
                if nu is not None and it == params["SHOWCASE_INTER"] and s == seeds[0]:
                    showcase = r

    info_a = fig_boundary(rows, inters, out_dir / "fig_lockin_inferred_trust.png")
    gw, gc = _gamma_trace(showcase, params["SOCIAL_NU"])
    info_b = fig_rehabilitation(showcase, gw, gc, out_dir / "fig_rehabilitation.png")

    conv_fix = {it: np.mean([r["oxy_end"] for r in rows if r["nu"] is None and r["inter"] == it])
                for it in inters}
    conv_gat = {it: np.mean([r["oxy_end"] for r in rows
                             if r["nu"] is not None and r["inter"] == it]) for it in inters}
    th_fix = {it: np.mean([r["t_half"] for r in rows if r["nu"] is None and r["inter"] == it])
              for it in inters}
    th_gat = {it: np.mean([r["t_half"] for r in rows
                           if r["nu"] is not None and r["inter"] == it]) for it in inters}

    # (a) the paper's regime: a trace of contact converts the heavily gated community.
    assert conv_fix[inters[0]] > 0.8, \
        f"fixed trust must convert even at inter={inters[0]} (got {conv_fix[inters[0]]:.2f})"
    # (b) inferred trust leaves conversion visibly incomplete at the same trace
    # contact at this horizon (the memoryless gate delays, it does not prevent;
    # nu=0.5 keeps the rehabilitation arc inside the horizon -- see docstring).
    assert conv_gat[inters[0]] < conv_fix[inters[0]] - 0.15, \
        f"inferred trust should visibly stall conversion at inter={inters[0]} " \
        f"({conv_gat[inters[0]]:.2f} vs {conv_fix[inters[0]]:.2f})"
    # (c) where conversion does happen, the timescale is roughly doubled.
    it_s = params["SHOWCASE_INTER"]
    assert th_gat[it_s] > 1.8 * th_fix[it_s], \
        f"inferred trust should ~2x the conversion time at inter={it_s} " \
        f"({th_gat[it_s]:.0f} vs {th_fix[it_s]:.0f})"
    # (d) the rehabilitation arc: trust severed during divergence, restored by the merge.
    assert info_b["min_trust_ratio"] < 0.05, "cross trust must collapse below 5% of within"
    assert info_b["final_trust_ratio"] > 0.5, "trust must be rehabilitated once the camps merge"

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        inters=np.array(inters, dtype=float),
        conv_fixed=np.array([conv_fix[it] for it in inters]),
        conv_gated=np.array([conv_gat[it] for it in inters]),
        t_half_fixed=np.array([th_fix[it] for it in inters]),
        t_half_gated=np.array([th_gat[it] for it in inters]),
        showcase_snap_t=showcase["snap_t"], showcase_oxy=showcase["oxy_index_sc"],
        showcase_gamma_within=gw, showcase_gamma_cross=gc)
    summary = {"config": params, "boundary": {str(k): dict(fixed=conv_fix[k], gated=conv_gat[k])
                                              for k in inters},
               "t_half": {str(k): dict(fixed=th_fix[k], gated=th_gat[k]) for k in inters},
               "rehabilitation": info_b}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: content-gated trust turns the lock-in boundary into a timescale -- at "
          f"inter={inters[0]} conversion is left incomplete at the horizon "
          f"({conv_gat[inters[0]]:.2f} vs {conv_fix[inters[0]]:.2f} under fixed trust) with no "
          f"isolation imposed by hand; where conversion completes it takes "
          f"{th_gat[it_s]/max(th_fix[it_s],1):.1f}x longer, through a distrust -> plateau -> "
          f"rehabilitation -> merge arc (cross trust falls to "
          f"{info_b['min_trust_ratio']*100:.0f}% of within, then recovers). Distrust is "
          f"self-organized isolation -- it buys time, not a wall.")


register(ExperimentSpec(
    model="phlogiston",
    name="lockin_inferred_trust",
    description="The lock-in boundary rerun with content-gated trust (social_nu) in the full "
                "Kuhn-cycle machinery: cross-community trust is inferred from belief "
                "disagreement, collapses to ~1% while the camps diverge (self-organized "
                "isolation, conversion stalled where fixed trust converts), and -- being "
                "memoryless -- reopens as the laggards drift in: the conversion timescale "
                "roughly doubles and the contraction theorem survives (metastability, not a "
                "new fixed point).",
    run=run,
    out_dir="lockin_inferred_trust",
    params=dict(
        N_AGENTS=80, N_STEPS=180, S_DOGMA=6.0, PROPOSAL_RATE=0.08,
        INTERS=(0.005, 0.02, 0.05, 0.2), SOCIAL_NU=0.5, SHOWCASE_INTER=0.05,
        SEEDS=(0, 1),
    ),
    seeds=(0, 1),
    canonical=False,
    consumes=dict(figures=["fig_lockin_inferred_trust", "fig_rehabilitation"]),
))
