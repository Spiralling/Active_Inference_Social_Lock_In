"""Phlogiston canonical experiment: does network topology govern a structural revolution
(the community prunes the contested phlogiston mass-law structure) versus an evidential
lock-in (it stalls)?

``N`` agents sit on a trust graph and learn from each other by pooling precision; a scarce
low-conviction vanguard (``gamma=0``) is the only source running the refuting gravimetric
experiment, so its disconfirming precision must *flow through the trust graph* to reach the
rest. The graph's connectivity (Fiedler ``lambda_2``) and the vanguard's placement decide
revolution vs lock-in. Honest caveat: this repo has repeatedly found naive posterior fusion
*washes out* heterogeneity (the "no-pool degeneracy"); whether topology bites is reported as
an empirical result, and the ``fuse_mode`` variants isolate the effect.

The reusable science lives in ``src.structural.models.phlogiston``; this module owns the
figures, the saved arrays, and the registry spec.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from src.structural import graphs
from src.structural.models.phlogiston import (
    CALIB, BELT, StructuralConfig, build_substrate, run_world,
    topology_panel, bridge_sweep, placement_contrast, run_controls)
from experiments.registry import ExperimentSpec, register


# ----------------------------------------------------------------------
# Figures (diagnostic; the polished F1-F5 panel is built in nb43).
# ----------------------------------------------------------------------

def fig_topology(panel: list[dict], path):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8))
    lam2 = [r["lambda2"] for r in panel]
    rev = [r["final_revolted_fraction"] for r in panel]
    kept = [r["kept_t"][-1].mean() for r in panel]
    labels = [r["panel_label"] for r in panel]
    a0.plot(lam2, rev, "o-", color="crimson", lw=2)
    for x, y, l, r in zip(lam2, rev, labels, panel):
        a0.annotate(f"{l}\n<k>={r['mean_degree']:.1f}", (x, y), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    a0.set_xlabel(r"algebraic connectivity $\lambda_2$")
    a0.set_ylabel("final revolted fraction")
    a0.set_title("F1: topology governs revolution vs lock-in")
    a1.plot(lam2, kept, "s-", color="navy", lw=2)
    a1.set_xlabel(r"$\lambda_2$"); a1.set_ylabel("final mean edges kept")
    a1.set_title("structure retained vs connectivity")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def fig_bridge(sweep: list[dict], path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    lam2 = [r["lambda2"] for r in sweep]
    rev = [r["final_revolted_fraction"] for r in sweep]
    ax.plot(lam2, rev, "o-", color="seagreen", lw=2)
    for r in sweep:
        ax.annotate(f"inter={r['inter']:.3f}", (r["lambda2"], r["final_revolted_fraction"]),
                    fontsize=7, textcoords="offset points", xytext=(4, -8))
    ax.set_xlabel(r"$\lambda_2$ (set by bridge density, intra fixed)")
    ax.set_ylabel("final revolted fraction")
    ax.set_title("F2: the clean knob -- bridge density -> connectivity -> revolution")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def fig_dynamics(panel: list[dict], path):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8))
    cmap = cm.viridis
    n = len(panel)
    for i, r in enumerate(panel):
        col = cmap(i / max(n - 1, 1))
        a0.plot(r["snap_t"], r["kept_t"].mean(axis=1), lw=2, color=col,
                label=f"{r['panel_label']} (λ₂={r['lambda2']:.2f})")
        a1.plot(r["snap_t"], r["disagreement_t"], lw=2, color=col)
    a0.set_xlabel("step"); a0.set_ylabel("mean edges kept")
    a0.set_title("F4: structure(t) by topology"); a0.legend(fontsize=7)
    a1.set_xlabel("step"); a1.set_ylabel("residual structural disagreement")
    a1.set_title("persistent disagreement = fragmented field (lock-in)")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def mechanisms_in_a_row(rep: dict, sub):
    """Print the ordered story of one representative run (a moderately connected graph)."""
    print("\n=== mechanisms in a row (representative: watts-strogatz, central vanguard) ===")
    print(f"  [init]   {rep['n_edges']} edges; all agents hold the SAME over-wired phlogiston prior.")
    print(f"           contested belt (highest v_e): {[sub.edges[i] for i in sub.belt_ix]}; "
          f"balanced λ* = {rep['lstar']:.2f}")
    print(f"  [agents] {rep['vanguard_idx'].size} vanguard (γ=0, λ={rep['lam'][rep['vanguard_idx'][0]]:.2f}) "
          f"vs {rep['lam'].size - rep['vanguard_idx'].size} self-censoring rest "
          f"(γ high, λ={rep['lam'][~rep['is_vanguard']][0]:.2f}); λ₂={rep['lambda2']:.2f}")
    rv = rep["revolted_t"]
    van = rep["vanguard_idx"]
    rest = ~rep["is_vanguard"]
    print(f"  [t=0]    revolted: vanguard {rv[0][van].mean():.2f}, rest {rv[0][rest].mean():.2f} "
          f"(nobody has the disconfirming evidence yet)")
    mid = len(rep["snap_t"]) // 2
    print(f"  [shift]  the world flips at t_shift; the vanguard runs the refuting experiment and "
          f"its precision begins to FLOW over the trust graph")
    print(f"  [mid]    revolted: vanguard {rv[mid][van].mean():.2f}, rest {rv[mid][rest].mean():.2f} "
          f"(evidence diffusing through fusion)")
    print(f"  [final]  revolted: vanguard {rv[-1][van].mean():.2f}, rest {rv[-1][rest].mean():.2f}; "
          f"population fraction {rep['final_revolted_fraction']:.2f}")
    print(f"  [field]  residual structural disagreement {rep['disagreement_t'][0]:.2f} -> "
          f"{rep['disagreement_t'][-1]:.2f} (stays > 0 = fragmented field / partial lock-in)")


# ----------------------------------------------------------------------
# The experiment: run the sweeps, assert controls, save arrays + figures + summary.
# ----------------------------------------------------------------------

def run(out_dir, params: dict) -> None:
    N = params["N"]
    cfg = StructuralConfig(t_shift=params["t_shift"], n_steps=params["n_steps"],
                           sigma_o=params["sigma_o"])
    sub = build_substrate(cfg)
    kw = {k: params[k] for k in ("conviction_rest", "lambda_vanguard", "n_vanguard",
                                 "gamma_rest", "sigma_o", "t_shift", "n_steps",
                                 "snapshot_every", "seed")}

    print(f"substrate: {sub.lstar=:.3f}; v_e(belt)={[round(float(sub.v_e[i]),2) for i in sub.belt_ix]}; "
          f"v_e range [{sub.v_e.min():.2f}, {sub.v_e.max():.2f}]")

    # ---- representative run + mechanisms in a row ----
    rep_graph = graphs.watts_strogatz(N, mean_degree=4, rewiring_p=0.1, seed=0)
    rep = run_world(rep_graph, sub=sub, cfg=cfg, vanguard_placement="central", **kw)
    mechanisms_in_a_row(rep, sub)

    # ---- F1: the topology panel ----
    print("\n=== F1: topology sweep (final revolted fraction vs λ₂) ===")
    panel = topology_panel(N, sub=sub, cfg=cfg, vanguard_placement="central", **kw)
    for r in panel:
        print(f"  {r['panel_label']:>16}: λ₂={r['lambda2']:6.3f}  <k>={r['mean_degree']:4.1f}  "
              f"revolted={r['final_revolted_fraction']:.3f}  kept={r['kept_t'][-1].mean():.2f}")
    fig_topology(panel, out_dir / "diag_topology_sweep.png")
    fig_dynamics(panel, out_dir / "diag_dynamics.png")

    # ---- F2: controlled bridge sweep ----
    print("\n=== F2: bridge sweep (intra fixed, vary inter) ===")
    sweep = bridge_sweep(N, sub=sub, cfg=cfg, **kw)
    for r in sweep:
        print(f"  inter={r['inter']:.3f}: λ₂={r['lambda2']:6.3f}  revolted={r['final_revolted_fraction']:.3f}")
    fig_bridge(sweep, out_dir / "diag_bridge_sweep.png")

    # ---- F3: vanguard placement ----
    print("\n=== F3: vanguard placement (scale-free graph) ===")
    place = placement_contrast(N, sub=sub, cfg=cfg, **kw)
    for p, r in place.items():
        print(f"  {p:>10} vanguard: revolted={r['final_revolted_fraction']:.3f}")

    # ---- F5: fusion-mode contrast on one graph ----
    print("\n=== F5: fusion-mode contrast (community+bridge graph) ===")
    gmid = graphs.community([N // 3] * 3, intra=0.45, inter=0.0, seed=0).with_bridge(0.03)
    modes = {}
    for mode in ("posterior", "deposit_pool", "deposit_keep"):
        r = run_world(gmid, sub=sub, cfg=cfg, fuse_mode=mode, vanguard_placement="block0", **kw)
        modes[mode] = r["final_revolted_fraction"]
        print(f"  {mode:>13}: revolted={r['final_revolted_fraction']:.3f}")

    # ---- controls ----
    print("\n=== null / sanity controls ===")
    ctrl = run_controls(sub, cfg, N, **kw)
    print(f"  isolated (λ₂=0):        revolted={ctrl['isolated_revolted']:.3f}  "
          f"(expect ≈ n_vanguard/N = {ctrl['isolated_expected']:.3f})")
    print(f"  complete:               revolted={ctrl['complete_revolted']:.3f}  (expect ≥ isolated)")
    print(f"  conviction_rest=0:      revolted={ctrl['conviction0_revolted']:.3f}  (expect ≈ 1.00)")
    print(f"  conviction_rest=50:     revolted={ctrl['convictionInf_revolted']:.3f}  (expect ≈ 0.00)")
    print(f"  gamma_rest=0 isolated:  revolted={ctrl['gamma0_isolated']:.3f}")
    print(f"  gamma_rest=0 complete:  revolted={ctrl['gamma0_complete']:.3f}  "
          f"(expect ≈ each other: topology stops mattering)")
    print(f"  null world (no shift):  revolted={ctrl['null_revolted']:.3f}  (expect 0.00)")
    print(f"  ΔF finite/PD:           {ctrl['dF_finite']}")

    # robust assertions (structurally guaranteed); topology-dependent ones reported above.
    assert ctrl["dF_finite"], "BMR readout produced non-finite ΔF (PD violation)"
    assert ctrl["null_revolted"] == 0.0, \
        f"null world (no refutation) revolted {ctrl['null_revolted']} (expected 0)"
    assert ctrl["convictionInf_revolted"] == 0.0, \
        f"echo chamber (λ→∞) revolted {ctrl['convictionInf_revolted']} (expected 0)"
    assert ctrl["conviction0_revolted"] > 0.95, \
        f"λ=0 should prune everywhere, got {ctrl['conviction0_revolted']}"
    assert ctrl["complete_revolted"] >= ctrl["isolated_revolted"] - 1e-9, \
        "complete graph spread less than isolated (mixing should not reduce revolt)"
    assert abs(graphs.algebraic_connectivity(graphs.complete(N)) - N) < 1e-3, \
        "algebraic_connectivity(complete) should be N"

    # ---- save arrays + summary ----
    def pack(runs, key):
        return np.array([r[key] for r in runs])

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        # F1
        panel_labels=np.array([r["panel_label"] for r in panel]),
        panel_lambda2=pack(panel, "lambda2"),
        panel_mean_degree=pack(panel, "mean_degree"),
        panel_revolted=pack(panel, "final_revolted_fraction"),
        panel_kept_final=np.array([r["kept_t"][-1].mean() for r in panel]),
        panel_snap_t=panel[0]["snap_t"],
        panel_kept_t=np.stack([r["kept_t"].mean(axis=1) for r in panel]),
        panel_disagreement_t=np.stack([r["disagreement_t"] for r in panel]),
        panel_m_t=np.stack([r["m_t"] for r in panel]),
        # F2
        bridge_inter=pack(sweep, "inter"),
        bridge_lambda2=pack(sweep, "lambda2"),
        bridge_revolted=pack(sweep, "final_revolted_fraction"),
        # F3
        place_central=place["central"]["final_revolted_fraction"],
        place_peripheral=place["peripheral"]["final_revolted_fraction"],
        place_central_revolted_t=place["central"]["revolted_t"].mean(axis=1),
        place_peripheral_revolted_t=place["peripheral"]["revolted_t"].mean(axis=1),
        # F5
        mode_labels=np.array(list(modes.keys())),
        mode_revolted=np.array(list(modes.values())),
        # representative
        rep_kept_t=rep["kept_t"], rep_revolted_t=rep["revolted_t"],
        rep_snap_t=rep["snap_t"], rep_lam=rep["lam"], rep_is_vanguard=rep["is_vanguard"],
        rep_v_e=sub.v_e, rep_edges=np.array(rep["edges"]),
    )

    summary = {
        "config": dict(N=N, **kw),
        "substrate": {"n_edges": len(sub.edges),
                      "belt_edges": [list(e) for e in BELT],
                      "v_e_belt": [float(sub.v_e[i]) for i in sub.belt_ix],
                      "lstar": float(sub.lstar)},
        "F1_topology": [{"graph": r["panel_label"], "lambda2": r["lambda2"],
                         "mean_degree": r["mean_degree"],
                         "revolted": r["final_revolted_fraction"]} for r in panel],
        "F2_bridge": [{"inter": r["inter"], "lambda2": r["lambda2"],
                       "revolted": r["final_revolted_fraction"]} for r in sweep],
        "F3_placement": {p: r["final_revolted_fraction"] for p, r in place.items()},
        "F5_fuse_mode": modes,
        "controls": ctrl,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figures / summary to {out_dir}")
    print(f"HEADLINE: revolted fraction over the topology panel = "
          f"{[round(r['final_revolted_fraction'], 2) for r in panel]}")
    spread = panel[-1]["final_revolted_fraction"] - panel[0]["final_revolted_fraction"]
    if spread > 0.1:
        print(f"  => topology BITES: revolt rises {spread:+.2f} from isolated to complete.")
    else:
        print(f"  => topology effect is small ({spread:+.2f}); the honest finding is wash-out "
              f"(see F5 for the deposit-mode contrast).")


register(ExperimentSpec(
    model="phlogiston",
    name="multiagent_topology",
    canonical=True,
    description="Population precision-fusion on a phlogiston substrate: network λ₂ governs "
                "structural revolution vs lock-in.",
    run=run,
    out_dir="structural_multiagent_topology",
    params=dict(N=60, **CALIB),
    seeds=(CALIB["seed"],),
    consumes=dict(notebook="nb43_multiagent_topology",
                  figures=["diag_topology_sweep", "diag_dynamics", "diag_bridge_sweep"]),
))
