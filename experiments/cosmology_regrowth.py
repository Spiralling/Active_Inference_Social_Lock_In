"""Lens A -- genuine node-wake structure re-growth on the changing cosmology world.

Lens B (``cosmology_tracking``) is honestly means-tracking: the contested edges accumulate uniformly
(Fisher ``HᵀH`` is operator-set). This companion shows the ONE place genuine data-driven structure
re-growth happens: a hidden node the agent's menu does not contain. In epoch 2 (``t >= T2``) an
unconceived ``dark_energy`` node switches on; a host-loop agent reads the residual floor from its
recent prediction errors and wakes a NEW node AFTER T2 -- and the null (coupling 0) never wakes.

The reusable science (constants + the ``single_run`` host loop) lives in
``src.structural.models.cosmology_regrowth``; this module owns the mechanisms-in-a-row report, the
figure, the coupling sweep, and the registry spec.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

from src.structural.models.cosmology_regrowth import (
    single_run, T1, T2, N_STEPS, SIGMA_O, WINDOW, WARMUP, HUB_NAME, DRIVES)
from experiments.registry import ExperimentSpec, register


def mechanisms_in_a_row(tele: dict, coupling: float):
    f = tele["floor_t"]
    ws = tele["wake_step"]
    print("\n=== mechanisms in a row (representative: a strong unconceived dark_energy node) ===")
    print(f"  [world]  3 cosmology epochs; in epoch 2 (t>={T2}) an UNCONCEIVED '{HUB_NAME}' node "
          f"switches on, coupling {DRIVES} at {coupling}.")
    print(f"  [init]   the agent's 6-node menu holds those commitments INDEPENDENT -- no slot for "
          f"the cause.")
    print(f"  [t<T1]   epoch 0; residual floor ~ {f[WARMUP:T1].max():.2f} (noise level: errors "
          f"vs the menu's prediction carry no coherent shift)")
    print(f"  [T1..T2] epoch 1 (theory flips, but errors are vs the CURRENT epoch mean -> still "
          f"just noise); floor max {f[T1:T2].max():.2f}")
    if ws >= 0:
        print(f"  [t>=T2]  the unconceived shift accumulates; the floor climbs to {f[T2:].max():.2f} "
              f"and the wake fires at t={ws} (= {ws - T2} steps after the node appears)")
        print(f"  [wake]   '{HUB_NAME}' is wired in -- the paradigm grows a NODE (not just an edge).")
        print(f"  [recover] drive-marginal edge-F1: {tele['f1_unwoken']:.2f} (no node) -> "
              f"{tele['f1_woken']:.2f} (woken)  structure_recovered={tele['structure_recovered']}")
    else:
        print("  [no wake] the model Bayes factor never accepted (the null / no-cause control).")


def make_figure(tele: dict, path):
    f, dF = tele["floor_t"], tele["delta_F_t"]
    na, nw = tele["n_active_t"], tele["n_woken_t"]
    ws = tele["wake_step"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    axs[0].plot(f, lw=2)
    axs[0].set_title("residual floor (telemetry)"); axs[0].set_ylabel(r"$\lambda_{max}(R)$")
    axs[1].plot(dF, lw=2); axs[1].axhline(0, color="grey", lw=1)
    axs[1].set_title("expansion model Bayes factor (the accept test)")
    axs[1].set_ylabel(r"$\Delta F$")
    axs[2].step(range(len(na)), na, where="post", lw=2, label="true active causes")
    axs[2].step(range(len(nw)), nw, where="post", lw=2, ls="--", label="nodes woken (agent)")
    axs[2].set_title("does the agent grow a node when the world does?")
    axs[2].set_yticks([0, 1]); axs[2].legend(fontsize=8)
    for ax in axs:
        ax.axvline(T2, color="purple", ls="-", lw=1, alpha=0.5)
        ax.text(T2 + 1, ax.get_ylim()[1] * 0.05, "T2 (dark_energy on)", color="purple", fontsize=7)
        if ws >= 0:
            ax.axvline(ws, color="seagreen", ls="--", lw=1.5)
        ax.set_xlabel("step")
    if ws >= 0:
        axs[0].text(ws, axs[0].get_ylim()[1] * 0.9, f" wake t={ws}", color="seagreen", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def run(out_dir, params: dict) -> None:
    rep_coupling = params["REP_COUPLING"]
    couplings = list(params["COUPLINGS"])
    seeds = list(params["SEEDS"])

    # ---- representative run + mechanisms in a row ----
    rep = single_run(coupling=rep_coupling, seed=0)
    mechanisms_in_a_row(rep, coupling=rep_coupling)
    make_figure(rep, out_dir / "cosmology_regrowth.png")

    # ---- coupling sweep: discovery threshold + the null ----
    print("\n=== coupling sweep (does dark_energy get discovered, and WHEN?) ===")
    wake_frac, mean_wake, after_T2_frac = [], [], []
    for cpl in couplings:
        woke, steps, after = [], [], []
        for s in seeds:
            te = single_run(coupling=cpl, seed=s)
            woke.append(te["wake_step"] >= 0)
            if te["wake_step"] >= 0:
                steps.append(te["wake_step"])
                after.append(te["woke_after_T2"])
        wf = float(np.mean(woke))
        wake_frac.append(wf)
        mean_wake.append(float(np.mean(steps)) if steps else float("nan"))
        after_T2_frac.append(float(np.mean(after)) if after else 1.0)
        mw = f"{np.mean(steps):.0f}" if steps else "--"
        print(f"  coupling={cpl}: wake_fraction={wf:.2f}  mean_wake_step={mw}  "
              f"all_after_T2={after_T2_frac[-1]:.2f}")

    # ---- controls ----
    null = single_run(coupling=0.0, seed=0)
    pos = single_run(coupling=rep_coupling, seed=0)
    print("\n=== null / sanity controls ===")
    print(f"  null (coupling=0):      wake_step={null['wake_step']}  (expect -1, never)")
    print(f"  positive (coupling={rep_coupling}): wake_step={pos['wake_step']}  woke_after_T2={pos['woke_after_T2']}")
    print(f"  pre-T2 Bayes factor max (pos): {pos['delta_F_t'][WARMUP:T2].max():.3f}  (expect <= 0)")

    assert null["wake_step"] == -1, \
        f"null world (no cause) spuriously woke at t={null['wake_step']}"
    assert pos["wake_step"] >= 0 and pos["woke_after_T2"], \
        f"positive case did not wake after T2 (wake={pos['wake_step']})"
    assert pos["delta_F_t"][WARMUP:T2].max() <= 0, \
        "pre-T2 Bayes factor went positive (would mis-fire before the cause exists)"
    assert wake_frac[0] == 0.0, f"coupling=0 had nonzero wake fraction {wake_frac[0]}"
    assert wake_frac[-1] > 0.9, f"strong cause coupling failed to discover (frac {wake_frac[-1]})"

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        couplings=np.array(couplings), seeds=np.array(seeds),
        wake_fraction=np.array(wake_frac), mean_wake_step=np.array(mean_wake),
        after_T2_fraction=np.array(after_T2_frac),
        rep_floor_t=rep["floor_t"], rep_delta_F_t=rep["delta_F_t"],
        rep_n_active_t=rep["n_active_t"], rep_n_woken_t=rep["n_woken_t"],
        rep_wake_step=np.array(rep["wake_step"]), T1=np.array(T1), T2=np.array(T2),
    )
    summary = {
        "config": {"T1": T1, "T2": T2, "n_steps": N_STEPS, "sigma_o": SIGMA_O,
                   "window": WINDOW, "drives": list(DRIVES), "hub_name": HUB_NAME},
        "representative": {"coupling": rep_coupling, "wake_step": rep["wake_step"],
                           "woke_after_T2": rep["woke_after_T2"],
                           "f1_unwoken": rep["f1_unwoken"], "f1_woken": rep["f1_woken"],
                           "structure_recovered": rep["structure_recovered"]},
        "sweep": {"couplings": couplings, "n_seeds": len(seeds),
                  "wake_fraction": wake_frac, "mean_wake_step": mean_wake,
                  "after_T2_fraction": after_T2_frac},
        "controls": {"null_wake_step": null["wake_step"],
                     "positive_wake_step": pos["wake_step"],
                     "positive_woke_after_T2": pos["woke_after_T2"],
                     "pre_T2_delta_F_max": float(pos["delta_F_t"][WARMUP:T2].max())},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figure / summary to {out_dir}")
    print(f"HEADLINE: the agent grows a NODE for the unconceived dark_energy AFTER it appears "
          f"(wake t={rep['wake_step']} >= T2={T2}); the null never wakes -- genuine node re-growth.")


register(ExperimentSpec(
    model="cosmology",
    name="cosmology_regrowth",
    description="Lens A: a host-loop agent grows a NODE for an unconceived dark_energy that switches "
                "on after T2 (wake fires only post-T2; null coupling never wakes) -- genuine "
                "data-driven structure re-growth.",
    run=run,
    out_dir="cosmology_regrowth",
    params=dict(REP_COUPLING=1.6, COUPLINGS=(0.0, 0.4, 0.8, 1.2, 1.6), SEEDS=(0, 1, 2)),
    seeds=(0, 1, 2),
    consumes=dict(notebook="nb44_cosmology_tracking_levers", figures=["cosmology_regrowth"]),
))
