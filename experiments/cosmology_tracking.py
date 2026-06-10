"""Cosmology canonical experiment (Lens B): a changing cosmology world -- do communities keep
re-tracking each successive theory, or collapse and get stuck? And which *diversity lever*
keeps them able to re-track -- network **disconnection** or **conviction**?

The truth cycles through three epochs (dark matter -> modified gravity -> scale-variant laws);
three heterogeneous communities (conservatism = prior precision, conviction = value tilt to a
home theory) sit on a trust graph and fuse precision. We sweep the two levers head-to-head as
a 2x2 (neither / disconnection / conviction / both), multi-seed, and read out which theory each
community holds vs the truth, the structural diversity, and an honest caveat (relational edges
accumulate uniformly -- theory identity rides the MEANS, not the learned couplings; data-driven
structure re-growth is Lens A, ``cosmology_regrowth``).

NO FORGETTING (per the spine refactor): agents accumulate as in nb43, so absolute tracking LAGS
~one epoch by construction -- read the lever comparison as *relative* diversity preservation.
The reusable science lives in ``src.structural.models.cosmology``; this module owns the figures,
the saved arrays, and the registry spec.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

from src.structural.simulation import run_simulation
from src.structural.models.cosmology import (
    sc, EPOCH_NAMES, COMM_LABELS, HOMES, CELLS,
    cosmology_population, cosmology_spec, cosmology_graph,
    held_by_community, run_cell)
from experiments.registry import ExperimentSpec, register


def mechanisms_in_a_row(scn, cell_both, cid, t1, n_steps):
    r = cell_both["r"]
    theory_mu = sc.cosmology_theory_means()
    held = held_by_community(r, cid, theory_mu)
    snap_t = r["snap_t"]
    true_per_snap = np.asarray(scn.epoch_t)[snap_t]
    print("\n=== mechanisms in a row (representative: BOTH levers, disconnected + conviction) ===")
    print(f"  [world]  3 epochs: {EPOCH_NAMES[0]} (t<{t1}) -> {EPOCH_NAMES[1]} -> "
          f"{EPOCH_NAMES[2]}; n_steps={n_steps}")
    print(f"  [init]   {len(cid)} agents in 3 communities {COMM_LABELS}; all start on the "
          f"incumbent ({EPOCH_NAMES[0]}) theory; homes={HOMES}")
    print(f"  [t=0]    every community holds theory 0 ({EPOCH_NAMES[0]})")
    for k in range(held.shape[0]):
        traj = "".join(str(int(x)) for x in held[k])
        print(f"  [comm{k}] {COMM_LABELS[k]:>12} (home {HOMES[k][:4]}): held-theory trace = {traj}  "
              f"-> ends on {EPOCH_NAMES[held[k][-1]]}")
    print(f"  [truth]  true theory per snapshot                       = "
          f"{''.join(str(int(x)) for x in true_per_snap)}")
    print(f"  [field]  residual structural disagreement {r['disagreement_t'][0]:.2f} -> "
          f"{r['disagreement_t'][-1]:.2f} (stays > 0 = communities never reconcile)")


def fig_tracking(rep_cells, scn, cid, cells, path, t1, t2):
    """2x2: held-theory trajectory per community per cell, with the true-epoch staircase."""
    theory_mu = sc.cosmology_theory_means()
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
    cols = ["crimson", "darkorange", "seagreen"]
    for ax, (label, disc, conv) in zip(axs.ravel(), cells):
        r = rep_cells[label]["r"]
        snap_t = r["snap_t"]
        true_per_snap = np.asarray(scn.epoch_t)[snap_t]
        held = held_by_community(r, cid, theory_mu)
        ax.step(snap_t, true_per_snap, where="post", color="0.4", lw=4, alpha=0.35,
                label="true theory")
        for k in range(held.shape[0]):
            ax.plot(snap_t, held[k] + 0.04 * (k - 1), "o-", color=cols[k], lw=1.8, ms=4,
                    label=f"{COMM_LABELS[k]} (home {HOMES[k][:4]})")
        for tt in (t1, t2):
            ax.axvline(tt, color="k", ls=":", lw=0.8)
        ax.set_title(f"{label}  (λ₂={r['lambda2']:.2f}, disconnect={disc}, conviction={conv})",
                     fontsize=10)
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels([n[:9] for n in EPOCH_NAMES], fontsize=8)
        ax.set_ylabel("held theory")
    axs[1, 0].set_xlabel("step"); axs[1, 1].set_xlabel("step")
    axs[0, 0].legend(fontsize=7, loc="center left")
    fig.suptitle("Lens B: which theory each community holds over time (tracking vs stuck)",
                 fontsize=13)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def fig_diversity(rep_cells, cells, path, t1, t2):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cols = {"neither": "0.5", "disconnection": "steelblue",
            "conviction": "darkorange", "both": "crimson"}
    for label, _, _ in cells:
        r = rep_cells[label]["r"]
        ax.plot(r["snap_t"], r["disagreement_t"], "o-", lw=2, color=cols[label], label=label)
    for tt in (t1, t2):
        ax.axvline(tt, color="k", ls=":", lw=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("residual structural disagreement")
    ax.set_title("structural diversity over time (0 = consensus / diversity lost)")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def run_controls(n_steps, t1, t2, sigma_o) -> dict:
    out = {}
    theory_mu = sc.cosmology_theory_means()
    pop = cosmology_population(seed=0)

    # 1. no-shift -> no re-track: a stationary incumbent world (epochs never start).
    scn_static = sc.cosmology_scenario(n_steps=n_steps, t1=n_steps + 10, t2=n_steps + 20,
                                       sigma_o=sigma_o)
    g = cosmology_graph(pop["sizes"], connected=True)
    r = run_simulation(scn_static, g, cosmology_spec(scn_static, pop, conviction=0.0),
                       snapshot_every=20, seed=0)
    held = held_by_community(r, pop["cluster_id"], theory_mu)
    out["no_shift_max_held"] = int(held.max())            # expect 0 (never leaves incumbent)

    # 2. single-epoch (only dark matter): hold the incumbent.
    scn_one = sc.cosmology_scenario(n_steps=t1, t1=t1, t2=t1 + 1, sigma_o=sigma_o)
    r1 = run_simulation(scn_one, g, cosmology_spec(scn_one, pop, conviction=0.0),
                        snapshot_every=20, seed=0)
    held1 = held_by_community(r1, pop["cluster_id"], theory_mu)
    out["single_epoch_max_held"] = int(held1.max())       # expect 0

    # 3. conviction -> very large, all committed to the incumbent + connected: FROZEN on 0.
    scn = sc.cosmology_scenario(n_steps=n_steps, t1=t1, t2=t2, sigma_o=sigma_o)
    spec_inf = cosmology_spec(scn, pop, conviction=50.0, homes=("dark_matter",) * 3)
    r_inf = run_simulation(scn, g, spec_inf, snapshot_every=20, seed=0)
    held_inf = held_by_community(r_inf, pop["cluster_id"], theory_mu)
    out["conviction_inf_final_held"] = [int(x) for x in held_inf[:, -1]]   # expect all 0

    # 4. inter=0 vs connected: disconnection preserves more structural disagreement.
    r_disc = run_simulation(scn, cosmology_graph(pop["sizes"], connected=False),
                            cosmology_spec(scn, pop, conviction=0.0), snapshot_every=20, seed=0)
    r_conn = run_simulation(scn, cosmology_graph(pop["sizes"], connected=True),
                            cosmology_spec(scn, pop, conviction=0.0), snapshot_every=20, seed=0)
    out["disagreement_disconnected"] = float(r_disc["disagreement_t"][-1])
    out["disagreement_connected"] = float(r_conn["disagreement_t"][-1])

    # 5. finite dF/PD across epochs.
    out["dF_finite"] = bool(np.isfinite(r_conn["final_dF"]).all()
                            and np.isfinite(r_inf["final_dF"]).all())
    return out


def run(out_dir, params: dict) -> None:
    n_steps, t1, t2 = params["N_STEPS"], params["T1"], params["T2"]
    sigma_o = params["SIGMA_O"]
    seeds = tuple(params["SEEDS"])
    conv_level = params["CONVICTION_LEVEL"]
    cells = CELLS

    scn = sc.cosmology_scenario(n_steps=n_steps, t1=t1, t2=t2, sigma_o=sigma_o)
    theory_mu = sc.cosmology_theory_means()

    print(f"cosmology world: epochs {EPOCH_NAMES}; T1={t1} T2={t2} n_steps={n_steps}")
    print("theory means (commitment cols dark/scale/modi):")
    for e, nm in enumerate(EPOCH_NAMES):
        print(f"  {nm:>18}: {np.round(theory_mu[e], 2)}")

    # ---- the 2x2 lever sweep, multi-seed ----
    print(f"\n=== 2x2 lever sweep (multi-seed, seeds={seeds}) ===")
    print(f"{'cell':>14} | {'λ₂':>6} | end-theories(rep) | n_distinct | comm0_stuck | "
          f"on_truth | final_disagree")
    rep_cells = {}                       # seed-0 raw runs for the figures / mechanisms
    cell_summaries = {}
    for label, disc, conv in cells:
        mets = []
        for s in seeds:
            cell, met = run_cell(scn, disconnect=disc, conviction=conv, seed=s)
            mets.append(met)
            if s == 0:
                rep_cells[label] = cell
        agg = {
            "lambda2": float(np.mean([m["lambda2"] for m in mets])),
            "n_distinct_end_mean": float(np.mean([m["n_distinct_end"] for m in mets])),
            "comm0_stuck_frac": float(np.mean([m["comm0_stuck_on_incumbent"] for m in mets])),
            "reached_final_frac_mean": float(np.mean([m["reached_final_frac"] for m in mets])),
            "on_truth_fraction_mean": float(np.mean([m["on_truth_fraction"] for m in mets])),
            "final_disagreement_mean": float(np.mean([m["final_disagreement"] for m in mets])),
            "end_theories_rep": mets[0]["end_theories"],
            "dF_finite": bool(all(m["dF_finite"] for m in mets)),
        }
        cell_summaries[label] = agg
        print(f"{label:>14} | {agg['lambda2']:6.2f} | {str(agg['end_theories_rep']):>17} | "
              f"{agg['n_distinct_end_mean']:10.2f} | {agg['comm0_stuck_frac']:11.2f} | "
              f"{agg['on_truth_fraction_mean']:8.2f} | {agg['final_disagreement_mean']:.2f}")

    # ---- wall probe + figures ----
    cid0 = rep_cells["both"]["pop"]["cluster_id"]
    mechanisms_in_a_row(scn, rep_cells["both"], cid0, t1, n_steps)
    fig_tracking(rep_cells, scn, cid0, cells, out_dir / "diag_tracking_2x2.png", t1, t2)
    fig_diversity(rep_cells, cells, out_dir / "diag_diversity.png", t1, t2)

    # ---- honest structural caveat: do the contested edges discriminate the theory? ----
    r_both = rep_cells["both"]["r"]
    names = scn.names
    idx = {n: i for i, n in enumerate(names)}
    learned_off = np.array([np.abs(r_both["snap_Pi"][-1][:, idx[a], idx[c]]).mean()
                            for (a, c) in scn.edges])
    true_off = np.abs(sc.cosmology_true_couplings())          # (E, n_edges)
    print("\n=== honest structural caveat (risk #3: do relational edges discriminate?) ===")
    print(f"  learned |Pi[a,c]| on the 6 contested edges (final): {np.round(learned_off, 1)}")
    print(f"  -> they accumulate ~uniformly (operator-set HᵀH dominates the ±{true_off.max():.2f} "
          f"prior coupling differences).")
    print("  -> THEORY IDENTITY IS CARRIED BY THE MEANS, not the learned precision couplings;")
    print("     genuine data-driven structure re-growth is Lens A (cosmology_regrowth).")

    # ---- controls ----
    print("\n=== null / sanity controls ===")
    ctrl = run_controls(n_steps, t1, t2, sigma_o)
    print(f"  no-shift max held theory:        {ctrl['no_shift_max_held']}  (expect 0)")
    print(f"  single-epoch max held theory:    {ctrl['single_epoch_max_held']}  (expect 0)")
    print(f"  conviction→∞ final held:         {ctrl['conviction_inf_final_held']}  (expect all 0)")
    print(f"  disagreement disconnected/conn:  {ctrl['disagreement_disconnected']:.2f} / "
          f"{ctrl['disagreement_connected']:.2f}  (expect disconnected > connected)")
    print(f"  ΔF finite/PD:                    {ctrl['dF_finite']}")

    assert ctrl["no_shift_max_held"] == 0, \
        f"stationary world re-tracked (held {ctrl['no_shift_max_held']}, expected 0)"
    assert ctrl["single_epoch_max_held"] == 0, \
        "single-epoch world left the incumbent (expected to hold it)"
    assert all(x == 0 for x in ctrl["conviction_inf_final_held"]), \
        f"conviction→∞ did not freeze on the incumbent: {ctrl['conviction_inf_final_held']}"
    assert ctrl["disagreement_disconnected"] > ctrl["disagreement_connected"] + 1e-6, \
        "disconnection did not preserve more structural disagreement than connection"
    assert ctrl["dF_finite"], "non-finite ΔF (PD violation)"

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        epoch_names=np.array(EPOCH_NAMES),
        theory_means=theory_mu,
        snap_t=rep_cells["neither"]["r"]["snap_t"],
        true_per_snap=np.asarray(scn.epoch_t)[rep_cells["neither"]["r"]["snap_t"]],
        cluster_id=cid0,
        **{f"held_{lab}": held_by_community(rep_cells[lab]["r"], cid0, theory_mu)
           for lab, _, _ in cells},
        **{f"disagree_{lab}": rep_cells[lab]["r"]["disagreement_t"]
           for lab, _, _ in cells},
        learned_offdiag=learned_off, true_offdiag=true_off,
    )
    summary = {
        "config": {"N": params["N_AGENTS"], "n_steps": n_steps, "T1": t1, "T2": t2,
                   "sigma_o": sigma_o, "seeds": list(seeds),
                   "conviction_level": conv_level,
                   "inter_connected": params["INTER_CONNECTED"], "homes": list(HOMES),
                   "communities": list(COMM_LABELS)},
        "epochs": list(EPOCH_NAMES),
        "cells": cell_summaries,
        "controls": ctrl,
        "structural_caveat": {
            "learned_offdiag_final": learned_off.tolist(),
            "true_offdiag_max": float(true_off.max()),
            "note": "relational edges accumulate uniformly (operator-set Fisher); theory "
                    "identity is in the means, not the learned couplings (see Lens A)."},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figures / summary to {out_dir}")
    print("HEADLINE (do we need both levers?):")
    for lab, _, _ in cells:
        a = cell_summaries[lab]
        print(f"  {lab:>14}: {a['n_distinct_end_mean']:.1f} distinct end-theories, "
              f"comm0 stuck {a['comm0_stuck_frac']:.0%}, disagreement {a['final_disagreement_mean']:.1f}")
    both = cell_summaries["both"]
    conv = cell_summaries["conviction"]
    if both["n_distinct_end_mean"] > conv["n_distinct_end_mean"] + 0.5:
        print("  => YES: persistent theory-divergence needs BOTH levers -- conviction alone "
              "washes out under fusion (no-pool degeneracy); disconnection protects it.")
    else:
        print("  => the levers did not separate as expected; read the cell table honestly.")


register(ExperimentSpec(
    model="cosmology",
    name="cosmology_tracking",
    canonical=True,
    description="Lens B: do communities re-track each cosmology epoch or get stuck? "
                "2x2 disconnection x conviction, multi-seed.",
    run=run,
    out_dir="cosmology_tracking",
    params=dict(N_AGENTS=150, N_STEPS=180, T1=60, T2=120, SIGMA_O=0.5, SEEDS=(0, 1, 2),
                CONVICTION_LEVEL=4.0, INTER_CONNECTED=0.08, INTRA=0.5),
    seeds=(0, 1, 2),
    consumes=dict(notebook="nb44_cosmology_tracking_levers",
                  figures=["diag_tracking_2x2", "diag_diversity"]),
))
