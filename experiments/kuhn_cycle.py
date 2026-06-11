"""The Kuhn cycle experiment: belt-first/core-last staircase + endogenous crisis + lock-in phase.

Fixes the paper's Sec. 4.4 honest negative by putting the conservatism geometry in
OBSERVABILITY (core commitments have no direct observation row; evidence reaches them only by
propagation through the relational balance rows) and adding the REWRITE_PLAN's endogenous
crisis (anomaly accumulates at the unobserved core; crossing theta releases the core prior).

Two headline figures:

* **Fig A (`kuhn_cycle_timeline.png`)** -- one population through one epoch switch, three
  panels: (1) the STAIRCASE: belt realigns right after the world changes, the core only after
  the crisis -- with the full-observability control (the paper's old null) moving in lockstep
  as the dashed counterfactual; (2) the anomaly accumulator crossing theta and the per-agent
  crisis raster: normal science / anomaly accumulation / crisis / revolution / new normal,
  none of it scripted; (3) the conviction-locked twin: same world, same theta, gate strength
  g=2 -- the RAW anomaly is even larger but the agent's own (gamma-gated) accumulator never
  reaches theta: permanent degenerate normal science.
* **Fig B (`kuhn_cycle_phase.png`)** -- the phase diagram: conviction gate strength g x crisis
  threshold theta -> fraction of runs that revolve, the sharp transition slice, and the
  bimodal end-state distribution (revolution OR lock-in, nothing in between; Sarle's
  bimodality coefficient).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register
from src.structural.shells import conservatism_split
from src.structural.models.kuhn_cycle import single_run


# ----------------------------------------------------------------------
# Fig A: the timeline
# ----------------------------------------------------------------------

def _phase_spans(r, theta):
    """Phase boundaries from the dynamics: normal / accumulation / revolution / new normal."""
    t1 = r["t1"]
    cs = r["crisis_step"]
    t_crisis = int(np.median(cs[cs >= 0])) if (cs >= 0).any() else None
    st, cp = r["snap_t"], r["core_prog_t"]
    arrived = np.nonzero((st > (t_crisis or t1)) & (cp > 0.85))[0]
    t_new = int(st[arrived[0]]) if len(arrived) else None
    return t1, t_crisis, t_new


def fig_timeline(r_rev, r_ctrl, r_lock, theta, path) -> dict:
    t1, t_crisis, t_new = _phase_spans(r_rev, theta)
    T = int(r_rev["snap_t"][-1]) + 1
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9.2, 9.4), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1.15, 1.0, 1.0]))

    # --- panel 1: the staircase ---
    st = r_rev["snap_t"]
    a0.plot(st, r_rev["belt_prog_t"], lw=2.4, color="#1e8449", label="belt (observed anomalies)")
    a0.plot(st, r_rev["core_prog_t"], lw=2.4, color="#6c3483", label="core (theory commitments)")
    a0.plot(r_ctrl["snap_t"], r_ctrl["core_prog_t"], lw=1.4, ls="--", color="#999999",
            label="core, fully-observed control (the old null: no staircase)")
    spans = [(0, t1, "#eaf2f8", "normal\nscience"),
             (t1, t_crisis, "#fdebd0", "anomaly\naccumulation"),
             (t_crisis, t_new, "#fadbd8", "crisis &\nrevolution"),
             (t_new, T, "#eafaf1", "new normal")]
    for x0, x1, color, label in spans:
        if x0 is None or x1 is None:
            continue
        a0.axvspan(x0, x1, color=color, zorder=0)
        a0.text((x0 + x1) / 2, 1.06, label, ha="center", va="bottom", fontsize=8)
    a0.set_ylabel("revision progress\n(0 = old theory, 1 = new)")
    a0.set_ylim(-0.04, 1.22); a0.legend(fontsize=8, loc="center left")
    a0.set_title("The Kuhn cycle, endogenously: belt revises first, the core only after the crisis")

    # --- panel 2: the anomaly and the crisis ---
    tt = np.arange(r_rev["A_tn"].shape[0])
    A = r_rev["A_tn"]
    a1.plot(tt, A.mean(axis=1), lw=2.0, color="#b9770e", label="anomaly accumulator $A_t$ (mean)")
    a1.fill_between(tt, np.percentile(A, 25, axis=1), np.percentile(A, 75, axis=1),
                    color="#b9770e", alpha=0.25)
    a1.axhline(theta, color="k", ls="--", lw=1.2, label=r"crisis threshold $\theta$")
    a1.axvline(t1, color="k", lw=0.8)
    cs = r_rev["crisis_step"]
    a1.scatter(cs[cs >= 0], np.full((cs >= 0).sum(), theta), marker="v", s=28, color="#c0392b",
               zorder=5, label="per-agent crisis")
    a1.set_ylabel("accumulated anomaly")
    a1.legend(fontsize=8)
    a1.set_title("anomaly accrues at the unobserved core after the world changes (nothing is scripted)")

    # --- panel 3: the conviction-locked twin ---
    Aw, Ar = r_lock["A_tn"].mean(axis=1), r_lock["A_raw_tn"].mean(axis=1)
    a2.plot(tt, Ar, lw=1.8, ls="--", color="#7b7d7d", label="anomaly IN THE WORLD (unweighted)")
    a2.plot(tt, Aw, lw=2.0, color="#1f618d", label="anomaly THE AGENT SEES ($\\gamma$-gated)")
    a2.axhline(theta, color="k", ls="--", lw=1.2)
    a2.axvline(t1, color="k", lw=0.8)
    a2b = a2.twinx()
    a2b.plot(r_lock["snap_t"], r_lock["core_prog_t"], lw=2.0, color="#6c3483")
    a2b.set_ylabel("core progress", color="#6c3483"); a2b.set_ylim(-0.04, 1.04)
    a2.set_ylabel("accumulated anomaly")
    a2.set_xlabel("step")
    a2.legend(fontsize=8, loc="center left")
    a2.set_title("the locked twin (conviction gate g=2): the anomaly is louder than ever, "
                 "but the agent never sees it -- no crisis, no revolution")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(t1=t1, t_crisis=t_crisis, t_new=t_new,
                t_belt=r_rev["t_belt"], t_core=r_rev["t_core"])


# ----------------------------------------------------------------------
# Fig B: the phase diagram
# ----------------------------------------------------------------------

def fig_phase(rows, gates, thetas, path) -> dict:
    rev = np.zeros((len(gates), len(thetas)))
    for gi, g in enumerate(gates):
        for ti, th in enumerate(thetas):
            vals = [r["revolved"] for r in rows if r["g"] == g and r["theta"] == th]
            rev[gi, ti] = float(np.mean(vals))
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(12.6, 3.9), width_ratios=[1.15, 1.0, 1.0])

    im = a0.imshow(rev, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
    a0.set_xticks(range(len(thetas)))
    a0.set_xticklabels([("$\\infty$" if not np.isfinite(t) else f"{t:.0f}") for t in thetas])
    a0.set_yticks(range(len(gates))); a0.set_yticklabels(gates)
    a0.set_xlabel(r"crisis threshold $\theta$"); a0.set_ylabel("conviction gate strength $g$")
    a0.set_title("fraction that revolves")
    fig.colorbar(im, ax=a0, shrink=0.85)

    th_mid = thetas[len(thetas) // 2 - 1]
    for ti, th in enumerate(thetas[:-1]):
        y = rev[:, ti]
        a1.plot(gates, y, "o-", lw=1.6, alpha=0.9, label=f"$\\theta$={th:.0f}")
    a1.set_xlabel("conviction gate strength $g$"); a1.set_ylabel("fraction that revolves")
    a1.set_title("the transition: a little more conviction,\nand the crisis never arrives")
    a1.set_ylim(-0.04, 1.04); a1.legend(fontsize=8)

    finals = np.array([r["core_final"] for r in rows if np.isfinite(r["theta"])])
    _tau, bc = conservatism_split(finals)
    a2.hist(finals, bins=24, color="#6c3483", alpha=0.85)
    a2.set_xlabel("final core progress (0 = old paradigm, 1 = new)")
    a2.set_ylabel("# runs")
    a2.set_title(f"end states are bimodal: revolution or lock-in\n(Sarle BC = {bc:.2f}, "
                 f"bimodal > 0.555)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(rev_frac=rev.tolist(), bimodality=float(bc), theta_mid=float(th_mid))


# ----------------------------------------------------------------------
# the experiment
# ----------------------------------------------------------------------

def run(out_dir, params: dict) -> None:
    theta = params["THETA_STAR"]
    stiff, release = params["CORE_STIFFNESS"], params["RELEASE"]
    base = dict(core_stiffness=stiff, release=release, omega=params["OMEGA"],
                n_steps=params["N_STEPS"], t1=params["T1"], N=params["N_AGENTS"])

    # ---- Fig A runs ----
    r_rev = single_run(gate_strength=0.0, theta=theta, seed=0, snapshot_every=2, **base)
    r_ctrl = single_run(core_observed=True, seed=0, snapshot_every=2,
                        omega=params["OMEGA"], n_steps=params["N_STEPS"],
                        t1=params["T1"], N=params["N_AGENTS"])
    r_lock = single_run(gate_strength=2.0, theta=theta, seed=0, snapshot_every=2, **base)
    info_a = fig_timeline(r_rev, r_ctrl, r_lock, theta, out_dir / "kuhn_cycle_timeline.png")

    # ---- self-checking headline: the cycle ----
    A = r_rev["A_tn"].mean(axis=1)
    t1 = params["T1"]
    assert A[:t1].max() < theta, f"epoch-0 must be quiet (A pre-change max {A[:t1].max():.1f} >= {theta})"
    cs = r_rev["crisis_step"]
    assert (cs >= 0).any() and cs[cs >= 0].min() > t1, "crisis must fire, and only after the world changes"
    lag = r_rev["t_core"] - r_rev["t_belt"]
    assert r_rev["t_belt"] > 0 and r_rev["t_core"] > 0 and lag >= 10, \
        f"staircase: core should lag belt by >=10 steps (lag {lag})"
    ctrl_lag = abs(r_ctrl["t_core"] - r_ctrl["t_belt"])
    assert r_ctrl["core_prog_t"][-1] > 0.5 and ctrl_lag < 10, \
        f"full observability should show NO staircase (the Sec 4.4 null; lag {ctrl_lag})"
    assert r_lock["core_prog_t"][-1] < 0.3, \
        f"the locked twin should never revolve (core {r_lock['core_prog_t'][-1]:.2f})"
    assert r_lock["A_raw_tn"].mean(axis=1)[-1] > theta, \
        "the locked twin's anomaly must exist in the world (raw accumulator above theta)"

    # ---- Fig B sweep ----
    gates, thetas, seeds = list(params["GATES"]), list(params["THETAS"]), list(params["SEEDS"])
    rows = []
    for g in gates:
        for th in thetas:
            for s in seeds:
                r = single_run(gate_strength=float(g), theta=float(th), seed=int(s),
                               snapshot_every=8, **base)
                rows.append(dict(
                    g=g, theta=th, seed=s,
                    core_final=float(r["core_prog_t"][-1]),
                    revolved=bool(r["core_prog_t"][-1] > 0.5),
                    crisis_med=(int(np.median(r["crisis_step"][r["crisis_step"] >= 0]))
                                if (r["crisis_step"] >= 0).any() else -1),
                    gamma_final=float(r["gamma_t"][-1]),
                ))
                print(f"  g={g} theta={th} seed={s}: core {rows[-1]['core_final']:.2f} "
                      f"crisis {rows[-1]['crisis_med']}", flush=True)
    info_b = fig_phase(rows, gates, thetas, out_dir / "kuhn_cycle_phase.png")

    rev = np.asarray(info_b["rev_frac"])
    ti_star = thetas.index(theta)
    assert rev[0, ti_star] > 0.9, f"g=0 should always revolve (got {rev[0, ti_star]:.2f})"
    assert rev[-1, ti_star] < 0.1, f"g={gates[-1]} should never revolve (got {rev[-1, ti_star]:.2f})"
    assert rev[:, -1].max() < 0.1, "theta=inf (no-crisis control) must never revolve"
    assert info_b["bimodality"] > 0.555, \
        f"end states should be bimodal (Sarle BC {info_b['bimodality']:.2f})"

    # ---- arrays + summary ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        snap_t=r_rev["snap_t"],
        belt_prog_t=r_rev["belt_prog_t"], core_prog_t=r_rev["core_prog_t"],
        ctrl_belt_prog_t=r_ctrl["belt_prog_t"], ctrl_core_prog_t=r_ctrl["core_prog_t"],
        lock_core_prog_t=r_lock["core_prog_t"],
        A_tn=r_rev["A_tn"], A_raw_tn=r_rev["A_raw_tn"], crisis_step=r_rev["crisis_step"],
        lock_A_tn=r_lock["A_tn"], lock_A_raw_tn=r_lock["A_raw_tn"],
        gamma_t_lock=r_lock["gamma_t"],
        sweep_gates=np.array(gates), sweep_thetas=np.array(thetas),
        rev_frac=np.asarray(info_b["rev_frac"]),
        sweep_core_final=np.array([r["core_final"] for r in rows]),
        sweep_g=np.array([r["g"] for r in rows]),
        sweep_theta=np.array([r["theta"] for r in rows]),
    )
    summary = {"config": params, "timeline": info_a, "phase": info_b,
               "staircase_lag": lag, "control_lag": ctrl_lag,
               "locked_core_final": float(r_lock["core_prog_t"][-1])}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: the Kuhn cycle emerges -- belt realigns at t={r_rev['t_belt']}, crisis at "
          f"t={info_a['t_crisis']}, core only at t={r_rev['t_core']} (staircase lag {lag}); "
          f"full observability shows no staircase (the old null, now the control); conviction "
          f"gating g makes the transition: revolve-fraction {rev[0, ti_star]:.2f} -> "
          f"{rev[-1, ti_star]:.2f}, end states bimodal (BC {info_b['bimodality']:.2f}).")


register(ExperimentSpec(
    model="cosmology",
    name="kuhn_cycle",
    description="The Kuhn cycle, endogenously: core observability (commitments have no direct "
                "observation row) makes the belt-first/core-last staircase REAL (fixing the "
                "Sec 4.4 null, which becomes the control), an anomaly accumulator at the "
                "unobserved core triggers crisis and revolution, and conviction gating turns "
                "the crisis off -- a sharp, bimodal revolution/lock-in phase transition.",
    run=run,
    out_dir="kuhn_cycle",
    params=dict(
        N_AGENTS=24, N_STEPS=160, T1=60, OMEGA=0.9,
        CORE_STIFFNESS=200.0, RELEASE=0.005, THETA_STAR=100.0,
        GATES=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
        THETAS=(40.0, 70.0, 100.0, 140.0, float("inf")),
        SEEDS=(0, 1, 2),
    ),
    seeds=(0, 1, 2),
    canonical=False,
    consumes=dict(figures=["kuhn_cycle_timeline", "kuhn_cycle_phase"]),
))
