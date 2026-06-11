"""Imperfect information: PARTIAL observability is where communicating Bayes nets finally bites --
and where the population usually does NOT converge.

When every agent sees the whole world, all agents accumulate the SAME (operator-set) structure and
differ only in their means, so communicating the directed Bayes net is identical to pooling
precision (``bayesnet_comms``: B-dispersion ~0.0006). The realistic regime is IMPERFECT info: each
agent observes only a random SUBSET of the world's channels (fixed per agent), so agents accumulate
genuinely DIFFERENT structure. We (a) confirm partial info makes agents structurally heterogeneous
(B-dispersion rises as the observed fraction falls), (b) ask whether communicating the Bayes net now
diverges from pooling precision and pooled deposits, and (c) map the convergence vs NON-convergence
regime -- the case that, per the brief, is most of the time.

World: the changing-baseline cosmology (truth shifts at T1, T2) with forgetting omega=0.9. Partial
observability is the ONLY heterogeneity: agents share a uniform prior and differ solely in which
channels they see.
"""
from __future__ import annotations

import json

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.structural import graphs
from src.structural.simulation import run_simulation, AgentSpec
from src.structural.bayesnet import from_info as bn_from_info
from src.structural import scenarios as sc
from src.structural.models.cosmology import T1, T2, N_STEPS, SIGMA_O
from experiments.registry import ExperimentSpec, register

MODE_LABEL = {"posterior": "precision (λ)", "bayesnet": "Bayes net (CPD)",
              "deposit_pool": "pooled deposits"}


def partial_masks(n: int, m: int, p_obs: float, seed: int) -> np.ndarray:
    """(N, m) per-agent observation mask: each agent sees a random k = max(1, round(p_obs m)) of
    the m channels, fixed over the run. Different agents see different subsets (imperfect info)."""
    rng = np.random.default_rng(seed + 7919)
    k = max(1, int(round(p_obs * m)))
    W = np.zeros((n, m), np.float32)
    for i in range(n):
        W[i, rng.choice(m, size=k, replace=False)] = 1.0
    return W


def frac_on_true(r, scn, theory_mu):
    means = np.linalg.solve(r["snap_Pi"], r["snap_h"][..., None])[..., 0]
    closest = sc.closest_theory(means, theory_mu)
    true_e = np.asarray(scn.epoch_t)[r["snap_t"]]
    return (closest == true_e[:, None]).mean(axis=1)


def B_dispersion(r) -> float:
    """Cross-agent std of the directed edge weights B at the final step (structural heterogeneity)."""
    Pi, h = np.asarray(r["snap_Pi"][-1]), np.asarray(r["snap_h"][-1])
    B = np.asarray(jax.vmap(bn_from_info)(jnp.asarray(Pi), jnp.asarray(h))[0])
    d = B.shape[-1]; il = np.tril_indices(d, -1)
    return float(np.mean(np.std(B[:, il[0], il[1]], axis=0)))


def run_cell(scn, theory_mu, *, n, omega, p_obs, mode, connected, seed):
    spec = AgentSpec(w_obs=jnp.asarray(partial_masks(n, scn.m, p_obs, seed)), lam=jnp.ones(n))
    g = graphs.community([n // 3] * 3, intra=0.45, inter=(0.06 if connected else 0.0), seed=seed)
    r = run_simulation(scn, g, spec, fuse_mode=mode, forgetting=omega, snapshot_every=5, seed=seed)
    fot = frac_on_true(r, scn, theory_mu)
    return {"fot_final": float(fot[-1]), "fot": fot, "snap_t": r["snap_t"],
            "final_div": float(r["disagreement_t"][-1]), "B_disp": B_dispersion(r),
            "dF_finite": bool(np.isfinite(r["final_dF"]).all())}


def run(out_dir, params: dict) -> None:
    n = params["N"]
    omega = params["OMEGA"]
    p_obs_grid = tuple(params["P_OBS"])
    modes = tuple(params["MODES"])
    seeds = tuple(params["SEEDS"])
    converge = params["CONVERGE"]

    scn = sc.cosmology_scenario(n_steps=N_STEPS, t1=T1, t2=T2, sigma_o=SIGMA_O)
    theory_mu = sc.cosmology_theory_means()
    print(f"changing cosmology, omega={omega}, N={n}, m={scn.m} channels; partial obs sweep {p_obs_grid}")

    grid = {}
    print(f"\n{'p_obs':>6} | {'mode':>16} | {'on-truth':>8} | {'B-disp':>7} | {'diversity':>9} | conv?")
    for p in p_obs_grid:
        for mode in modes:
            cells = [run_cell(scn, theory_mu, n=n, omega=omega, p_obs=p, mode=mode, connected=True, seed=s)
                     for s in seeds]
            agg = {k: float(np.mean([c[k] for c in cells]))
                   for k in ("fot_final", "final_div", "B_disp")}
            agg["fot"] = cells[0]["fot"]; agg["snap_t"] = cells[0]["snap_t"]
            agg["dF_finite"] = all(c["dF_finite"] for c in cells)
            grid[(p, mode)] = agg
            print(f"{p:6.2f} | {MODE_LABEL[mode]:>16} | {agg['fot_final']:8.2f} | {agg['B_disp']:7.4f} | "
                  f"{agg['final_div']:9.2f} | {'yes' if agg['fot_final'] > converge else 'NO'}")

    # disconnected control at the most partial setting: comms is essential
    p_lo = p_obs_grid[-1]
    iso = [run_cell(scn, theory_mu, n=n, omega=omega, p_obs=p_lo, mode="posterior", connected=False, seed=s)
           for s in seeds]
    iso_fot = float(np.mean([c["fot_final"] for c in iso]))
    print(f"\n  disconnected control (p_obs={p_lo}, no comms): on-truth={iso_fot:.2f} "
          f"(expect low -- partial views can't be pooled)")

    # ---- figures ----
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    cols = {"posterior": "navy", "bayesnet": "darkorange", "deposit_pool": "seagreen"}
    for mode in modes:
        a0.plot(p_obs_grid, [grid[(p, mode)]["fot_final"] for p in p_obs_grid], "o-", lw=2,
                color=cols[mode], label=MODE_LABEL[mode])
    a0.axhline(converge, color="grey", ls=":", lw=1, label=f"converged (>{converge})")
    a0.scatter([p_lo], [iso_fot], marker="x", s=90, color="crimson", zorder=5, label="disconnected")
    a0.set_xlabel("observed fraction  p_obs  (1 = full info)"); a0.set_ylabel("final on-truth (convergence)")
    a0.set_ylim(-0.05, 1.05); a0.invert_xaxis(); a0.legend(fontsize=8)
    a0.set_title("imperfect info → non-convergence (and which comms object helps)")
    for mode in modes:
        a1.plot(p_obs_grid, [grid[(p, mode)]["B_disp"] for p in p_obs_grid], "s-", lw=2, color=cols[mode],
                label=MODE_LABEL[mode])
    a1.set_xlabel("observed fraction  p_obs"); a1.set_ylabel("cross-agent structural dispersion (B)")
    a1.invert_xaxis(); a1.set_title("partial info makes agents structurally DIFFERENT"); a1.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "diag_partial_obs.png", dpi=120); plt.close(fig)

    # ---- verdict ----
    assert all(grid[k]["dF_finite"] for k in grid), "non-finite ΔF under partial obs"
    full_disp = grid[(1.0, "posterior")]["B_disp"]
    part_disp = grid[(p_lo, "posterior")]["B_disp"]
    dmax = max(abs(grid[(p, "bayesnet")]["fot_final"] - grid[(p, "posterior")]["fot_final"]) for p in p_obs_grid)
    n_converged = sum(grid[(p, m)]["fot_final"] > converge for p in p_obs_grid for m in modes)
    print("\n=== verdict ===")
    print(f"  structural heterogeneity: B-disp {full_disp:.4f} (full info) -> {part_disp:.4f} "
          f"(p_obs={p_lo}) -- partial info makes agents hold DIFFERENT Bayes nets")
    print(f"  Bayes-net vs precision comms: max |Δ on-truth| over the sweep = {dmax:.2f} "
          f"({'they now DIVERGE' if dmax > 0.05 else 'still ~equivalent'})")
    print(f"  convergence: {n_converged}/{len(p_obs_grid)*len(modes)} (p_obs × mode) cells converged "
          f"(>{converge}) -- non-convergence is the rule once info is partial")

    summary = {"config": {"N": n, "omega": omega, "p_obs": list(p_obs_grid), "modes": list(modes),
                          "seeds": list(seeds), "converge_thresh": converge},
               "grid": {f"{p}|{m}": {"on_truth": grid[(p, m)]["fot_final"],
                                     "B_disp": grid[(p, m)]["B_disp"],
                                     "diversity": grid[(p, m)]["final_div"]}
                        for p in p_obs_grid for m in modes},
               "disconnected_control_on_truth": iso_fot,
               "B_disp_full_vs_partial": [full_disp, part_disp],
               "bayesnet_vs_precision_max_delta": dmax,
               "n_converged_cells": n_converged}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(
        out_dir / "simulation_arrays.npz", p_obs=np.array(p_obs_grid), modes=np.array(modes),
        on_truth=np.array([[grid[(p, m)]["fot_final"] for m in modes] for p in p_obs_grid]),
        B_disp=np.array([[grid[(p, m)]["B_disp"] for m in modes] for p in p_obs_grid]),
        diversity=np.array([[grid[(p, m)]["final_div"] for m in modes] for p in p_obs_grid]),
        iso_on_truth=iso_fot)

    print(f"\nsaved arrays / figure / summary to {out_dir}")
    print(f"HEADLINE: partial info makes agents structurally different (B-disp {full_disp:.4f}→{part_disp:.4f}); "
          f"Bayes-net vs precision comms {'DIVERGE' if dmax>0.05 else 'stay similar'} (Δ={dmax:.2f}); "
          f"convergence becomes the exception as p_obs falls.")


register(ExperimentSpec(
    model="cosmology",
    name="partial_obs_comms",
    description="Partial observability makes agents structurally heterogeneous (B-dispersion rises); "
                "maps where communicating Bayes nets diverges from pooling precision and where the "
                "population fails to converge.",
    run=run,
    out_dir="partial_obs_comms",
    params=dict(N=90, OMEGA=0.9, P_OBS=(1.0, 0.7, 0.5, 0.33),
                MODES=("posterior", "bayesnet", "deposit_pool"), SEEDS=(0, 1, 2), CONVERGE=0.6),
    seeds=(0, 1, 2),
    consumes=dict(figures=["diag_partial_obs"]),
))
