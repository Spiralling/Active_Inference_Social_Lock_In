"""One-off preview: gather ONLY the 80 trajectory jobs of network_epistemology into
a side dir and render fig_kuhn_trajectories_by_topology, so the figure and its paper
caption land NOW and the full canonical refresh can run later. Seeded identically to
the canonical jobs, so the canonical run reproduces these traces exactly. Delete
after the canonical run settles."""
import json
from concurrent.futures import ProcessPoolExecutor

import _bootstrap  # noqa: F401  (repo root on sys.path + utf-8 stdout)


def main():
    from src.repro.headless import headless
    headless()
    import numpy as np
    from experiments.network_epistemology import (
        _one_job, _fig_trajectories, TOPOS)
    from experiments.registry import load_specs
    from src.repro.paths import ROOT

    p = dict(load_specs()["network_epistemology"].params)
    out = ROOT / "results" / "_traj_preview"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(p, kind="traj", topo=topo, gamma=float(p["GAMMA_EASY"]), seed=s)
            for topo in TOPOS for s in range(p["R_TRAJ"])]
    rows = []
    with ProcessPoolExecutor(max_workers=p["MAX_WORKERS"]) as pool:
        for r in pool.map(_one_job, jobs, chunksize=2):
            rows.append(r)
            print(f"  traj {r['topo']} seed {r['seed']} done "
                  f"({len(rows)}/{len(jobs)})", flush=True)
    nv, T = p["N_VAN"], p["T"]
    stats = {}
    for name in TOPOS:
        q = np.stack([r["q"] for r in rows if r["topo"] == name])   # (S, T, N)
        hit = q[:, :, nv:] > p["Q_CONS"]
        t_f = np.where(hit.any(axis=1), hit.argmax(axis=1), T)      # (S, Nc)
        stats[name] = {
            "flip_median": float(np.median(t_f)),
            "flip_stagger_iqr": float(np.median(np.percentile(t_f, 75, axis=1)
                                                - np.percentile(t_f, 25, axis=1))),
            "final_cons_mean": float(q[:, -1, nv:].mean()),
        }
        print(f"{name:>10}: median flip {stats[name]['flip_median']:.0f}, "
              f"stagger IQR {stats[name]['flip_stagger_iqr']:.0f}, "
              f"final {stats[name]['final_cons_mean']:.3f}")
    _fig_trajectories(rows, p, out / "fig_kuhn_trajectories_by_topology.png")
    (out / "traj_stats.json").write_text(json.dumps(stats, indent=2),
                                         encoding="utf-8")
    print(f"\npreview figure + stats -> {out}")


if __name__ == "__main__":
    main()
