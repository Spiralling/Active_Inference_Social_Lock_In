"""The Kuhn cycle experiment: belt-first/core-last staircase + endogenous crisis + lock-in phase.

Fixes the paper's Sec. 4.4 honest negative by putting the conservatism geometry in
OBSERVABILITY (core commitments have no direct observation row; evidence reaches them only by
propagation through the relational balance rows). Crisis is the unified move ledger: each step
the RELEASE of the core block is scored ``dF + lam dU`` (Savage-Dickey evidence for the
core-released anchor + lam times the closed-form conviction-value change); ``score > 0``
releases the core prior. No hand-made accumulator threshold -- the accumulator survives only
as telemetry.

Two headline figures:

* **Fig A (`kuhn_cycle_timeline.png`)** -- one population through one epoch switch, three
  panels: (1) the STAIRCASE: belt realigns right after the world changes, the core only after
  the crisis -- with the full-observability control (the paper's old null) moving in lockstep
  as the dashed counterfactual; (2) the release ledger crossing zero and the per-agent crisis
  raster: normal science / anomaly accumulation / crisis / revolution / new normal, none of it
  scripted; (3) the conviction-locked twin (gate strength g=2): the ledger still catches the
  misfit -- but the gate has silenced exactly the relational channels the released core would
  need to re-align, so the revolution STALLS: crisis without revolution, the lock-in.
* **Fig B (`kuhn_cycle_phase.png`)** -- the transition: conviction gate strength g -> fraction
  of runs that revolve (with the no-crisis control as baseline), the median crisis step, and
  the bimodal end-state distribution (revolution OR lock-in, nothing in between; Sarle's
  bimodality coefficient). The conviction temperature lam is NOT a useful axis on the release
  ledger -- the release barely moves the mean until the anomalous evidence arrives, and once it
  has, dF dwarfs lam*dU -- the same relocation of protection from the threshold to the gate
  that the phlogiston results show.
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

def _phase_spans(r):
    """Phase boundaries from the dynamics: normal / accumulation / revolution / new normal."""
    t1 = r["t1"]
    cs = r["crisis_step"]
    t_crisis = int(np.median(cs[cs >= 0])) if (cs >= 0).any() else None
    st, cp = r["snap_t"], r["core_prog_t"]
    arrived = np.nonzero((st > (t_crisis or t1)) & (cp > 0.85))[0]
    t_new = int(st[arrived[0]]) if len(arrived) else None
    return t1, t_crisis, t_new


def fig_timeline(r_rev, r_ctrl, r_lock, path) -> dict:
    t1, t_crisis, t_new = _phase_spans(r_rev)
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

    # --- panel 2: the release ledger and the crisis ---
    S = r_rev["score_tn"]
    tt = np.arange(S.shape[0])
    with np.errstate(all="ignore"):
        a1.plot(tt, np.nanmean(S, axis=1), lw=2.0, color="#b9770e",
                label=r"release ledger $\Delta F + \lambda\,\Delta U$ (mean)")
        a1.fill_between(tt, np.nanpercentile(S, 25, axis=1), np.nanpercentile(S, 75, axis=1),
                        color="#b9770e", alpha=0.25)
    a1.axhline(0.0, color="k", ls="--", lw=1.2, label="crisis threshold (score $> 0$)")
    a1.axvline(t1, color="k", lw=0.8)
    cs = r_rev["crisis_step"]
    a1.scatter(cs[cs >= 0], np.zeros((cs >= 0).sum()), marker="v", s=28, color="#c0392b",
               zorder=5, label="per-agent crisis (release applied)")
    a1.set_ylabel("core-release\nledger score")
    a1.set_yscale("symlog")
    a1.legend(fontsize=8)
    a1.set_title("crisis is the release ledger crossing zero: the model's own evidence vs the "
                 "closed-form conviction value of the core (nothing is scripted)")

    # --- panel 3: the conviction-locked twin ---
    Aw, Ar = r_lock["A_tn"].mean(axis=1), r_lock["A_raw_tn"].mean(axis=1)
    a2.plot(tt, Ar, lw=1.8, ls="--", color="#7b7d7d", label="anomaly IN THE WORLD (unweighted)")
    a2.plot(tt, Aw, lw=2.0, color="#1f618d", label="anomaly THE AGENT SEES ($\\gamma$-gated)")
    a2.axvline(t1, color="k", lw=0.8)
    a2b = a2.twinx()
    a2b.plot(r_lock["snap_t"], r_lock["core_prog_t"], lw=2.0, color="#6c3483")
    a2b.set_ylabel("core progress", color="#6c3483"); a2b.set_ylim(-0.04, 1.04)
    a2.set_ylabel("accumulated anomaly\n(telemetry)")
    a2.set_xlabel("step")
    a2.legend(fontsize=8, loc="center left")
    a2.set_title("the locked twin (conviction gate g=2): the ledger still catches the misfit, "
                 "but the gate silenced the channels the released core needs -- "
                 "crisis without revolution")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(t1=t1, t_crisis=t_crisis, t_new=t_new,
                t_belt=r_rev["t_belt"], t_core=r_rev["t_core"])


# ----------------------------------------------------------------------
# Fig B: the phase diagram
# ----------------------------------------------------------------------

def fig_phase(rows, gates, path) -> dict:
    """Gate sweep at lam*: the conviction gate is the lever that stalls the revolution.
    (The release ledger leaves the conviction temperature lam without leverage here: the
    release barely moves the mean until the anomalous evidence has arrived, and once it has,
    dF dwarfs lam*dU -- the same relocation of protection from the threshold to the gate the
    phlogiston results show.)"""
    rev_on = np.array([np.mean([r["core_final"] > 0.5 for r in rows
                                if r["g"] == g and r["crisis_on"]]) for g in gates])
    rev_off = np.array([np.mean([r["core_final"] > 0.5 for r in rows
                                 if r["g"] == g and not r["crisis_on"]]) for g in gates])
    crisis_med = []
    for g in gates:
        med = [r["crisis_med"] for r in rows if r["g"] == g and r["crisis_on"]
               and r["crisis_med"] >= 0]
        crisis_med.append(int(np.median(med)) if med else -1)
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(12.6, 3.9))

    a0.plot(gates, rev_on, "o-", lw=2.0, color="#1e8449", label="release ledger active")
    a0.plot(gates, rev_off, "s--", lw=1.6, color="#7b7d7d", label="no-crisis control")
    a0.set_xlabel("conviction gate strength $g$"); a0.set_ylabel("fraction that revolves")
    a0.set_title("the transition: a little more conviction,\nand the revolution stalls")
    a0.set_ylim(-0.04, 1.04); a0.legend(fontsize=8)

    a1.plot(gates, [c if c >= 0 else np.nan for c in crisis_med], "o-", lw=1.8, color="#b9770e")
    a1.set_xlabel("conviction gate strength $g$"); a1.set_ylabel("median crisis step")
    a1.set_title("the gate starves the ledger's evidence:\ncrisis arrives later, "
                 "and re-alignment dies")

    finals = np.array([r["core_final"] for r in rows if r["crisis_on"]])
    _tau, bc = conservatism_split(finals)
    a2.hist(finals, bins=24, color="#6c3483", alpha=0.85)
    a2.set_xlabel("final core progress (0 = old paradigm, 1 = new)")
    a2.set_ylabel("# runs")
    a2.set_title(f"end states are bimodal: revolution or lock-in\n(Sarle BC = {bc:.2f}, "
                 f"bimodal > 0.555)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(rev_on=rev_on.tolist(), rev_off=rev_off.tolist(),
                crisis_med=crisis_med, bimodality=float(bc))


# ----------------------------------------------------------------------
# the experiment
# ----------------------------------------------------------------------

def run(out_dir, params: dict) -> None:
    lam_star = params["LAM_STAR"]
    stiff, release = params["CORE_STIFFNESS"], params["RELEASE"]
    base = dict(core_stiffness=stiff, release=release, omega=params["OMEGA"],
                n_steps=params["N_STEPS"], t1=params["T1"], N=params["N_AGENTS"])

    # ---- Fig A runs ----
    r_rev = single_run(gate_strength=0.0, lam=lam_star, seed=0, snapshot_every=2, **base)
    r_ctrl = single_run(core_observed=True, seed=0, snapshot_every=2,
                        omega=params["OMEGA"], n_steps=params["N_STEPS"],
                        t1=params["T1"], N=params["N_AGENTS"])
    r_lock = single_run(gate_strength=2.0, lam=lam_star, seed=0, snapshot_every=2, **base)
    info_a = fig_timeline(r_rev, r_ctrl, r_lock, out_dir / "kuhn_cycle_timeline.png")

    # ---- self-checking headline: the cycle ----
    t1 = params["T1"]
    warmup = 15                                   # matches the model's initialization guard
    with np.errstate(all="ignore"):
        pre = np.nanmax(r_rev["score_tn"][warmup:t1])
    assert pre < 0, f"epoch-0 must be quiet (release ledger pre-change max {pre:.2f} >= 0)"
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
    A_lock, A_raw_lock = r_lock["A_tn"].mean(axis=1)[-1], r_lock["A_raw_tn"].mean(axis=1)[-1]
    assert A_raw_lock > 2.0 * A_lock, \
        "the locked twin's anomaly must exist in the world but be gated away from the agent " \
        f"(raw {A_raw_lock:.1f} vs gated {A_lock:.1f})"

    # ---- Fig B sweep: the gate is the lever; the no-crisis control is the baseline ----
    gates, seeds = list(params["GATES"]), list(params["SEEDS"])
    rows = []
    for g in gates:
        for crisis_on in (True, False):
            for s in seeds:
                r = single_run(gate_strength=float(g), lam=lam_star, crisis=crisis_on,
                               seed=int(s), snapshot_every=8, **base)
                rows.append(dict(
                    g=g, crisis_on=crisis_on, seed=s,
                    core_final=float(r["core_prog_t"][-1]),
                    crisis_med=(int(np.median(r["crisis_step"][r["crisis_step"] >= 0]))
                                if (r["crisis_step"] >= 0).any() else -1),
                    gamma_final=float(r["gamma_t"][-1]),
                ))
                print(f"  g={g} crisis={'on' if crisis_on else 'off'} seed={s}: "
                      f"core {rows[-1]['core_final']:.2f} crisis {rows[-1]['crisis_med']}",
                      flush=True)
    info_b = fig_phase(rows, gates, out_dir / "kuhn_cycle_phase.png")

    rev_on = np.asarray(info_b["rev_on"])
    rev_off = np.asarray(info_b["rev_off"])
    assert rev_on[0] > 0.9, f"g=0 should always revolve (got {rev_on[0]:.2f})"
    assert rev_on[-1] < 0.1, f"g={gates[-1]} should never revolve (got {rev_on[-1]:.2f})"
    assert rev_off.max() < 0.1, "the no-crisis control must never revolve"
    assert info_b["bimodality"] > 0.555, \
        f"end states should be bimodal (Sarle BC {info_b['bimodality']:.2f})"

    # ---- arrays + summary ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        snap_t=r_rev["snap_t"],
        belt_prog_t=r_rev["belt_prog_t"], core_prog_t=r_rev["core_prog_t"],
        ctrl_belt_prog_t=r_ctrl["belt_prog_t"], ctrl_core_prog_t=r_ctrl["core_prog_t"],
        lock_core_prog_t=r_lock["core_prog_t"],
        A_tn=r_rev["A_tn"], A_raw_tn=r_rev["A_raw_tn"], score_tn=r_rev["score_tn"],
        crisis_step=r_rev["crisis_step"],
        lock_A_tn=r_lock["A_tn"], lock_A_raw_tn=r_lock["A_raw_tn"],
        lock_score_tn=r_lock["score_tn"],
        gamma_t_lock=r_lock["gamma_t"],
        sweep_gates=np.array(gates),
        rev_on=np.asarray(info_b["rev_on"]), rev_off=np.asarray(info_b["rev_off"]),
        crisis_med=np.asarray(info_b["crisis_med"]),
        sweep_core_final=np.array([r["core_final"] for r in rows]),
        sweep_g=np.array([r["g"] for r in rows]),
        sweep_crisis_on=np.array([r["crisis_on"] for r in rows]),
    )
    summary = {"config": params, "timeline": info_a, "phase": info_b,
               "staircase_lag": lag, "control_lag": ctrl_lag,
               "locked_core_final": float(r_lock["core_prog_t"][-1])}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: the Kuhn cycle emerges -- belt realigns at t={r_rev['t_belt']}, crisis "
          f"(release ledger dF + lam dU crosses 0) at t={info_a['t_crisis']}, core only at "
          f"t={r_rev['t_core']} (staircase lag {lag}); full observability shows no staircase "
          f"(the old null, now the control); conviction gating g makes the transition: "
          f"revolve-fraction {rev_on[0]:.2f} -> {rev_on[-1]:.2f}, end states "
          f"bimodal (BC {info_b['bimodality']:.2f}).")


register(ExperimentSpec(
    model="cosmology",
    name="kuhn_cycle",
    description="The Kuhn cycle, endogenously: core observability (commitments have no direct "
                "observation row) makes the belt-first/core-last staircase REAL (fixing the "
                "Sec 4.4 null, which becomes the control), crisis is the core-release move's "
                "own ledger (Savage-Dickey dF + lam dU crossing zero -- no accumulator "
                "threshold), and conviction gating stalls the revolution by silencing the "
                "channels the released core needs -- a bimodal revolution/lock-in phase "
                "transition.",
    run=run,
    out_dir="kuhn_cycle",
    params=dict(
        N_AGENTS=24, N_STEPS=160, T1=60, OMEGA=0.9,
        CORE_STIFFNESS=200.0, RELEASE=0.005, LAM_STAR=0.2,
        GATES=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
        SEEDS=(0, 1, 2),
    ),
    seeds=(0, 1, 2),
    canonical=False,
    consumes=dict(figures=["kuhn_cycle_timeline", "kuhn_cycle_phase"]),
))
