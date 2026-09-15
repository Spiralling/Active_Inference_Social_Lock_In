"""Provenance rerun of the Figure 9 trajectories (fig_kuhn_trajectories_by_topology).

This is scripts/run_traj_preview.py (the one-off that produced the paper figure on
2026-06-12) with one addition: it SAVES what that run threw away -- the per-seed,
per-agent trajectories, the exact adjacency of each contact graph, the vanguard
node ids, the seed list and the environment -- and then checks the rendered PNG
and the flip statistics against the rescued originals in results/network_epistemology/.

Jobs are seeded identically to the canonical experiment (kind="traj", gamma=GAMMA_EASY,
seed in range(R_TRAJ)), so the traces are the ones the paper plots.

Output: results/network_epistemology_traj/
    trajectories.npz      q_<topo>: (R_TRAJ, T, N) float32 conservative+vanguard q(oxygen)
                          A_<topo>: (N, N) int adjacency; lambda2_<topo>; seeds; vanguard_nodes
    graphs.json           adjacency lists + membership per topology, human-readable
    traj_stats.json       flip medians / stagger / final mean, same recipe as the original
    provenance.json       params, git sha/branch/dirty, environment, hash comparison
    fig_kuhn_trajectories_by_topology.png
"""
import hashlib
import json
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import _bootstrap  # noqa: F401  (repo root on sys.path + utf-8 stdout)


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None


def main():
    from src.repro.headless import headless
    headless()
    import numpy as np
    import jax, scipy, equinox, networkx, matplotlib
    from experiments.network_epistemology import (
        _one_job, _fig_trajectories, _graph, TOPOS)
    from experiments.registry import load_specs
    from src.repro.paths import ROOT
    from src.structural import graphs

    p = dict(load_specs()["network_epistemology"].params)
    out = ROOT / "results" / "network_epistemology_traj"
    out.mkdir(parents=True, exist_ok=True)
    ref = ROOT / "results" / "network_epistemology"
    n, nv, T = p["N"], p["N_VAN"], p["T"]
    seeds = list(range(p["R_TRAJ"]))

    jobs = [dict(p, kind="traj", topo=topo, gamma=float(p["GAMMA_EASY"]), seed=s)
            for topo in TOPOS for s in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=p["MAX_WORKERS"]) as pool:
        for r in pool.map(_one_job, jobs, chunksize=2):
            rows.append(r)
            print(f"  traj {r['topo']} seed {r['seed']} done "
                  f"({len(rows)}/{len(jobs)})", flush=True)

    # --- arrays + graphs (the part the original run did not keep) ---
    arrays = {"seeds": np.asarray(seeds), "vanguard_nodes": np.arange(nv)}
    gjson = {}
    stats = {}
    for name in TOPOS:
        g = _graph(name, n, nv, p["Z_BETA"])
        A = np.asarray(g.A, dtype=np.int8)
        q = np.stack([r["q"] for r in sorted(
            (r for r in rows if r["topo"] == name), key=lambda r: r["seed"])])
        arrays[f"q_{name}"] = q
        arrays[f"A_{name}"] = A
        arrays[f"lambda2_{name}"] = np.float64(graphs.algebraic_connectivity(g))
        gjson[name] = {
            "kind": g.kind, "n": n, "vanguard_nodes": list(range(nv)),
            "membership": None if g.membership is None else g.membership.tolist(),
            "lambda2": float(graphs.algebraic_connectivity(g)),
            "n_edges": int(A.sum() // 2),
            "adjacency_list": {str(i): np.flatnonzero(A[i]).tolist() for i in range(n)},
        }
        hit = q[:, :, nv:] > p["Q_CONS"]
        t_f = np.where(hit.any(axis=1), hit.argmax(axis=1), T)
        stats[name] = {
            "flip_median": float(np.median(t_f)),
            "flip_stagger_iqr": float(np.median(np.percentile(t_f, 75, axis=1)
                                                - np.percentile(t_f, 25, axis=1))),
            "final_cons_mean": float(q[:, -1, nv:].mean()),
        }
        print(f"{name:>10}: median flip {stats[name]['flip_median']:.0f}, "
              f"stagger IQR {stats[name]['flip_stagger_iqr']:.0f}, "
              f"final {stats[name]['final_cons_mean']:.3f}")

    np.savez(out / "trajectories.npz", **arrays)
    (out / "graphs.json").write_text(json.dumps(gjson, indent=1), encoding="utf-8")
    (out / "traj_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    png = out / "fig_kuhn_trajectories_by_topology.png"
    _fig_trajectories(rows, p, png)

    # --- compare against the rescued originals ---
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    ref_png = ref / "fig_kuhn_trajectories_by_topology.png"
    ref_stats = json.loads((ref / "traj_stats.json").read_text(encoding="utf-8"))
    cmp = {
        "png_sha256_new": sha(png),
        "png_sha256_original": sha(ref_png) if ref_png.exists() else None,
        "png_identical": ref_png.exists() and sha(png) == sha(ref_png),
        "traj_stats_identical": stats == ref_stats,
        "traj_stats_original": ref_stats,
    }
    prov = {
        "script": "scripts/run_traj_provenance.py",
        "params": p,
        "git": {"sha": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
                "dirty": bool(_git("status", "--porcelain"))},
        "env": {"python": sys.version.split()[0], "platform": sys.platform,
                "machine": platform.machine(), "jax": jax.__version__,
                "numpy": np.__version__, "scipy": scipy.__version__,
                "equinox": equinox.__version__, "networkx": networkx.__version__,
                "matplotlib": matplotlib.__version__},
        "comparison_to_original": cmp,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str),
                                         encoding="utf-8")
    print(f"\nPNG identical to original: {cmp['png_identical']}")
    print(f"traj_stats identical to original: {cmp['traj_stats_identical']}")
    print(f"outputs -> {out}")


if __name__ == "__main__":
    main()
