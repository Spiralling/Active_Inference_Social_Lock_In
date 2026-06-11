"""The endogenous-frontier experiment suite: the landscape paper's claims meet the big sweep's axes.

Three experiments on ``src.structural.models.landscape_frontier`` (the landscape environment WITH
channel gating -- the world legible only through earned channels):

* ``frontier_fixed_points`` -- Prop. 1 + the gate ablation. Static deep world with buried pockets;
  communities seeded with different coarse cuts reach DIFFERENT seed-determined fixed points (each
  resolves only the pockets its descent path approaches); with the gate off (full observability)
  the same seeds collapse to ONE fixed point. Pluralism is the frontier, not noise.
* ``frontier_connectivity`` -- the lambda2 lever. Same setup, sweeping the social inter-density:
  shared data-supported proposals (testimony) merge frontiers and collapse the plural fixed
  points; disconnection protects them. The landscape analogue of the big sweep's map A.
* ``frontier_lockin`` -- the changing-world lock-in map. After a regime change (the band of new
  structure jumps to the least-valued region), agile communities (low omega, no conviction)
  re-allocate resolution and re-track; long memory OR conviction tilt locks the budget into the
  old region. The landscape analogue of the big sweep's map C (omega x tilt under changing
  epochs), with both levers acting through one rank competition.
"""
from __future__ import annotations

import json
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register
from src.structural.models.landscape_frontier import (
    FrontierConfig,
    FrontierEnvironment,
)


def _cfg(params: dict, **over) -> FrontierConfig:
    base = dict(
        n_communities=params["N_COMMUNITIES"], init_mode="subtree_biased",
        init_bias_depth=params["INIT_BIAS_DEPTH"], budget=params["BUDGET"],
        omega=params["OMEGA"], n_steps=params["N_STEPS"], seed=params["SEED"],
    )
    base.update(over)
    return FrontierConfig(**base)


def _pocket_block_sizes(env: FrontierEnvironment, assignment: np.ndarray) -> list[float]:
    """Mean held-block size over each pocket's members (small = resolved, large = unconceived)."""
    blocks, counts = np.unique(assignment, return_counts=True)
    size_of = dict(zip(blocks, counts))
    return [float(np.mean([size_of[assignment[m]] for m in env.dendro.node(p).members]))
            for p in env.pocket_nodes]


def _pairwise_pdist_matrix(assign_cf: np.ndarray) -> np.ndarray:
    """Rand-style partition distance between every pair of final community cuts."""
    n_c, n_f = assign_cf.shape
    iu = np.triu_indices(n_f, 1)
    D = np.zeros((n_c, n_c))
    for i in range(n_c):
        for j in range(i + 1, n_c):
            a, b = assign_cf[i], assign_cf[j]
            D[i, j] = D[j, i] = float(np.mean(
                (a[iu[0]] == a[iu[1]]) != (b[iu[0]] == b[iu[1]])))
    return D


def _n_distinct(D: np.ndarray, eps: float = 0.05) -> int:
    """Number of distinct fixed points: greedy clustering of the distance matrix at ``eps``."""
    n = D.shape[0]
    rep: list[int] = []
    for i in range(n):
        if not any(D[i, r] < eps for r in rep):
            rep.append(i)
    return len(rep)


def _resolution_strip(assignment: np.ndarray, order: np.ndarray) -> np.ndarray:
    blocks, counts = np.unique(assignment, return_counts=True)
    size_of = dict(zip(blocks, counts))
    return np.array([size_of[assignment[i]] for i in order], dtype=np.float64)


# ======================================================================
# 1. frontier_fixed_points
# ======================================================================

def run_fixed_points(out_dir, params: dict) -> None:
    seed = params["SEED"]
    env_on = FrontierEnvironment(_cfg(params, gate="on", world_mode="static"))
    env_off = FrontierEnvironment(_cfg(params, gate="off", world_mode="static"))
    r_on = env_on.run(seed=seed)
    r_on2 = env_on.run(seed=seed)                      # replay: Prop. 1 determinism
    r_off = env_off.run(seed=seed)
    n_c = params["N_COMMUNITIES"]

    # ---- self-checking headline ----
    assert np.array_equal(r_on["final_assignment_cf"], r_on2["final_assignment_cf"]), \
        "the fixed point must be a function of the seed alone (same seed, same closure)"
    tail = r_on["mean_pdist_t"][-10:]
    assert float(tail.max() - tail.min()) < 0.05, \
        f"gate-on cuts should be macroscopically frozen (tail pdist range {tail.max()-tail.min():.3f})"
    D_on = _pairwise_pdist_matrix(r_on["final_assignment_cf"])
    D_off = _pairwise_pdist_matrix(r_off["final_assignment_cf"])
    k_on, k_off = _n_distinct(D_on), _n_distinct(D_off)
    pd_on, pd_off = float(r_on["final_pdist"]), float(r_off["final_pdist"])
    assert k_on >= 2, f"gate-on should hold MULTIPLE fixed points (got {k_on})"
    assert pd_on > 0.15, f"gate-on fixed points should differ structurally (pdist {pd_on:.3f})"
    assert pd_off < 0.5 * pd_on, \
        f"full observability should collapse the pluralism ({pd_on:.3f} -> {pd_off:.3f})"
    # each gate-on community resolves SOME pocket and leaves SOME pocket unconceived;
    # every gate-off community resolves every pocket.
    pk_on = np.array([_pocket_block_sizes(env_on, r_on["final_assignment_cf"][ci])
                      for ci in range(n_c)])
    pk_off = np.array([_pocket_block_sizes(env_off, r_off["final_assignment_cf"][ci])
                       for ci in range(n_c)])
    assert (pk_on.min(axis=1) < 8).all() and (pk_on.max(axis=1) > 30).all(), \
        "gate-on: each community should resolve a near pocket and miss a far one"
    assert (pk_off < 8).all(), "gate-off: every community should resolve every pocket"

    # ---- figures ----
    _fig_strips(env_on, r_on, r_off, out_dir / "diag_fixed_point_cuts.png")
    _fig_pdist_matrices(D_on, D_off, out_dir / "diag_pdist_matrices.png")
    _fig_pockets(pk_on, pk_off, out_dir / "diag_pocket_resolution.png")
    _fig_unlock(r_on, r_off, out_dir / "diag_unlocked.png")

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        on_final_assignment_cf=r_on["final_assignment_cf"],
        off_final_assignment_cf=r_off["final_assignment_cf"],
        on_mean_pdist_t=r_on["mean_pdist_t"], off_mean_pdist_t=r_off["mean_pdist_t"],
        on_cut_size_tc=r_on["cut_size_tc"], off_cut_size_tc=r_off["cut_size_tc"],
        on_unlocked_frac_tc=r_on["unlocked_frac_tc"], off_unlocked_frac_tc=r_off["unlocked_frac_tc"],
        on_recon_error_tc=r_on["recon_error_tc"], off_recon_error_tc=r_off["recon_error_tc"],
        D_on=D_on, D_off=D_off, pocket_sizes_on=pk_on, pocket_sizes_off=pk_off,
        pocket_nodes=np.array(env_on.pocket_nodes), leaf_order=r_on["leaf_order"],
        n_fine=r_on["n_fine"],
    )
    summary = {
        "config": params,
        "gate_on": {"final_pdist": pd_on, "n_distinct_fixed_points": k_on,
                    "pocket_block_sizes": pk_on.tolist()},
        "gate_off": {"final_pdist": pd_off, "n_distinct_fixed_points": k_off,
                     "pocket_block_sizes": pk_off.tolist()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: {n_c} seeds, gate ON -> {k_on} distinct fixed points (pdist {pd_on:.2f}), "
          f"each blind to far pockets; gate OFF -> {k_off} (pdist {pd_off:.2f}), all pockets found. "
          f"Pluralism IS the frontier.")


def _fig_strips(env, r_on, r_off, path) -> None:
    order = r_on["leaf_order"]
    n_c = r_on["final_assignment_cf"].shape[0]
    rows = [(f"gate on  C{ci}", r_on["final_assignment_cf"][ci]) for ci in range(n_c)]
    rows += [(f"gate off C{ci}", r_off["final_assignment_cf"][ci]) for ci in range(n_c)]
    pos = np.empty(len(order), dtype=np.int64); pos[order] = np.arange(len(order))
    fig, axs = plt.subplots(len(rows), 1, figsize=(9, 0.62 * len(rows) + 1.4), sharex=True)
    for ax, (title, assign) in zip(axs, rows):
        strip = _resolution_strip(assign, order)[None, :]
        ax.imshow(strip, aspect="auto", cmap="viridis", vmin=1, vmax=40)
        for p in env.pocket_nodes:
            mem = pos[list(env.dendro.node(p).members)]
            ax.axvspan(mem.min() - 0.5, mem.max() + 0.5, color="crimson", alpha=0.35)
        ax.set_yticks([]); ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=8)
    axs[-1].set_xlabel("fine node (dendrogram order); red = buried pockets")
    axs[0].set_title("final block size per node (dark = finely resolved): the gate makes fixed points seed-local")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def _fig_pdist_matrices(D_on, D_off, path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, D, title in ((axs[0], D_on, "gate on"), (axs[1], D_off, "gate off")):
        im = ax.imshow(D, cmap="magma", vmin=0, vmax=max(D_on.max(), 0.4))
        ax.set_title(f"{title}: pairwise partition distance")
        ax.set_xlabel("community"); ax.set_ylabel("community")
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def _fig_pockets(pk_on, pk_off, path) -> None:
    n_c, n_p = pk_on.shape
    x = np.arange(n_p)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    w = 0.8 / n_c
    for ax, pk, title in ((axs[0], pk_on, "gate on"), (axs[1], pk_off, "gate off")):
        for ci in range(n_c):
            ax.bar(x + ci * w, pk[ci], width=w, label=f"C{ci}")
        ax.axhline(8, color="k", ls="--", lw=0.8)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels([f"pocket {p}" for p in range(n_p)])
        ax.set_title(title)
    axs[0].set_ylabel("block size over pocket (small = resolved)")
    axs[0].legend(fontsize=8)
    fig.suptitle("which buried pockets each community ever conceives", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def _fig_unlock(r_on, r_off, path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    t = np.arange(r_on["unlocked_frac_tc"].shape[0])
    for ci in range(r_on["unlocked_frac_tc"].shape[1]):
        ax.plot(t, r_on["unlocked_frac_tc"][:, ci], lw=1.6, label=f"gate on C{ci}")
    ax.plot(t, r_on["union_unlocked_frac_t"], lw=2.2, color="k", ls="--", label="union")
    ax.set_xlabel("step"); ax.set_ylabel("fraction of channels unlocked")
    ax.set_title("the frontier climbs: structure unlocks channels unlocks structure")
    ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


# ======================================================================
# 2. frontier_connectivity
# ======================================================================

def run_connectivity(out_dir, params: dict) -> None:
    inters = list(params["INTERS"])
    seeds = list(params["RUN_SEEDS"])
    pdist = np.zeros((len(inters), len(seeds)))
    lam2 = np.zeros(len(inters))
    union = np.zeros((len(inters), len(seeds)))
    base_traj = {}
    for ii, inter in enumerate(inters):
        env = FrontierEnvironment(_cfg(params, gate="on", world_mode="static", inter=float(inter)))
        lam2[ii] = env.lambda2
        for si, s in enumerate(seeds):
            r = env.run(seed=s)
            pdist[ii, si] = float(r["mean_pdist"])
            union[ii, si] = float(r["union_unlocked_frac_t"][-1])
            if si == 0:
                base_traj[inter] = r["mean_pdist_t"]

    m = pdist.mean(axis=1)
    assert m[0] > 0.15, f"disconnected communities should stay plural (pdist {m[0]:.3f})"
    assert m[-1] < 0.5 * m[0], \
        f"full connection should collapse the pluralism ({m[0]:.3f} -> {m[-1]:.3f})"

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.6, 3.8))
    for si in range(len(seeds)):
        a0.plot(lam2, pdist[:, si], "o-", alpha=0.4, lw=1, color="#6c3483")
    a0.plot(lam2, m, "o-", lw=2.4, color="#6c3483", label="mean over seeds")
    a0.set_xlabel(r"social connectivity $\lambda_2$"); a0.set_ylabel("mean partition distance")
    a0.set_title("connection collapses plural fixed points"); a0.legend(fontsize=8)
    for inter, traj in base_traj.items():
        a1.plot(np.arange(len(traj)), traj, lw=1.8, label=f"inter={inter}")
    a1.set_xlabel("step"); a1.set_ylabel("mean pairwise pdist")
    a1.set_title("divergence over time by connectivity"); a1.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "diag_connectivity.png", dpi=120); plt.close(fig)

    np.savez_compressed(out_dir / "simulation_arrays.npz",
                        inters=np.array(inters, dtype=np.float64), lambda2=lam2,
                        mean_pdist=pdist, union_unlocked=union,
                        run_seeds=np.array(seeds))
    summary = {"config": params,
               "lambda2": lam2.tolist(), "mean_pdist": pdist.mean(axis=1).tolist(),
               "union_unlocked": union.mean(axis=1).tolist()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: raising lambda2 {lam2[0]:.1f}->{lam2[-1]:.1f} collapses mean partition "
          f"distance {m[0]:.2f}->{m[-1]:.2f}: testimony merges frontiers.")


# ======================================================================
# 3. frontier_lockin
# ======================================================================

def run_lockin(out_dir, params: dict) -> None:
    lams = list(params["VALUE_LAMBDAS"])
    omegas = list(params["OMEGAS"])
    seeds = list(params["RUN_SEEDS"])
    n_steps, t_change = params["N_STEPS"], params["T_CHANGE"]
    cold = np.zeros((len(lams), len(omegas), len(seeds)))
    traj_corner = {}
    for li, lam in enumerate(lams):
        for oi, om in enumerate(omegas):
            cfg = _cfg(params, gate="on", world_mode="regime_change",
                       value_beta=params["VALUE_BETA"], value_lambda=float(lam),
                       omega=float(om), band_amp=params["BAND_AMP"], t_change=t_change)
            env = FrontierEnvironment(cfg)
            for si, s in enumerate(seeds):
                r = env.run(seed=s)
                cold[li, oi, si] = float(np.mean(r["cold_region_block_size_c"]))
                if si == 0 and (li in (0, len(lams) - 1)) and (oi in (0, len(omegas) - 1)):
                    traj_corner[(lam, om)] = r["cold_block_size_tc"].mean(axis=1)

    # normalize to a lock-in fraction: 0 = resolved to pocket scale, 1 = never touched
    size0 = cold.max()
    lockin = np.clip((cold.mean(axis=2) - 8.0) / max(size0 - 8.0, 1e-9), 0.0, 1.0)
    agile, locked = lockin[0, 0], lockin[-1, -1]
    assert agile < 0.5, f"agile corner (lam={lams[0]}, omega={omegas[0]}) should re-track (lock-in {agile:.2f})"
    assert locked > 0.8, f"conviction+memory corner should lock in (lock-in {locked:.2f})"
    assert lockin[-1, 0] > agile + 0.3 and lockin[0, -1] > agile + 0.3, \
        "conviction and memory should EACH produce lock-in on their own"

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.8, 3.9))
    im = a0.imshow(lockin, cmap="inferno_r", vmin=0, vmax=1, aspect="auto")
    a0.set_xticks(range(len(omegas))); a0.set_xticklabels(omegas)
    a0.set_yticks(range(len(lams))); a0.set_yticklabels(lams)
    a0.set_xlabel(r"forgetting $\omega$ (memory)"); a0.set_ylabel(r"conviction tilt $\lambda$")
    a0.set_title("lock-in after the regime change")
    fig.colorbar(im, ax=a0, shrink=0.85, label="lock-in fraction")
    for (lam, om), tr in traj_corner.items():
        a1.plot(np.arange(len(tr)), tr, lw=1.8, label=f"$\\lambda$={lam}, $\\omega$={om}")
    a1.axvline(t_change, color="k", ls="--", lw=1)
    a1.set_xlabel("step"); a1.set_ylabel("cold-region block size")
    a1.set_title("re-tracking the new structure (corners)"); a1.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out_dir / "diag_lockin.png", dpi=120); plt.close(fig)

    np.savez_compressed(out_dir / "simulation_arrays.npz",
                        value_lambdas=np.array(lams, dtype=np.float64),
                        omegas=np.array(omegas, dtype=np.float64),
                        cold_block_size=cold, lockin=lockin,
                        run_seeds=np.array(seeds))
    summary = {"config": params, "lockin": lockin.tolist(),
               "agile_corner": float(agile), "locked_corner": float(locked)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: lock-in map over conviction x memory: agile corner {agile:.2f}, "
          f"locked corner {locked:.2f}; each lever locks in alone.")


# ======================================================================
# registry
# ======================================================================

_COMMON = dict(N_COMMUNITIES=4, INIT_BIAS_DEPTH=3, BUDGET=16, OMEGA=0.9, N_STEPS=80, SEED=0)

register(ExperimentSpec(
    model="landscape",
    name="frontier_fixed_points",
    description="Prop. 1 + gate ablation: seed-determined fixed points on a static deep world with "
                "buried pockets; the SAME seeds collapse to one fixed point under full observability "
                "-- structural pluralism is the endogenous observation frontier, not noise.",
    run=run_fixed_points,
    out_dir="frontier_fixed_points",
    params=dict(_COMMON),
    seeds=(0,),
    canonical=True,
    consumes=dict(figures=["diag_fixed_point_cuts", "diag_pdist_matrices",
                           "diag_pocket_resolution", "diag_unlocked"]),
))

register(ExperimentSpec(
    model="landscape",
    name="frontier_connectivity",
    description="The lambda2 lever on the frontier: sharing data-supported proposals (testimony) "
                "merges frontiers and collapses plural fixed points; disconnection protects them "
                "(landscape analogue of big-sweep map A).",
    run=run_connectivity,
    out_dir="frontier_connectivity",
    params=dict(_COMMON, INTERS=(0.0, 0.1, 0.25, 0.5, 1.0), RUN_SEEDS=(0, 1, 2)),
    seeds=(0, 1, 2),
    consumes=dict(figures=["diag_connectivity"]),
))

register(ExperimentSpec(
    model="landscape",
    name="frontier_lockin",
    description="Lock-in map after a regime change: conviction tilt x forgetting decide whether a "
                "community re-allocates its resolution budget to the new structure or stays locked "
                "into the old region (landscape analogue of big-sweep map C).",
    run=run_lockin,
    out_dir="frontier_lockin",
    params=dict(_COMMON, N_COMMUNITIES=2, N_STEPS=140, T_CHANGE=40, BAND_AMP=3.0,
                VALUE_BETA=5.0, VALUE_LAMBDAS=(0.0, 2.0, 8.0), OMEGAS=(0.8, 0.9, 0.97),
                RUN_SEEDS=(0, 1, 2)),
    seeds=(0, 1, 2),
    consumes=dict(figures=["diag_lockin"]),
))
