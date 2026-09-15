"""Collect the published paper's figures from results/ into paper/figures/.

THE one place that records which experiment produces each figure in the published
paper (IWAI 2026, "Paradigm Shifts: Motivated Inference on Webs of Belief"),
including the one rename (the paper calls rival_kuhn's fig_kuhn_cycle
"fig_rival_kuhn_cycle"). Run the experiments first (each writes its provenance
manifest), then:

    python scripts/collect_paper_figures.py           # copy what exists, report gaps
    python scripts/collect_paper_figures.py --check   # report only, copy nothing
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# paper/figures/<name> -> results/<experiment out_dir>/<source figure>
PAPER_FIGURES = {
    "fig_walls_vs_time.png": "represented_rivals/fig_walls_vs_time.png",
    "fig_rival_kuhn_cycle.png": "rival_kuhn/fig_kuhn_cycle.png",
    "fig_kuhn_lockin_boundary.png": "kuhn_sweep/figB_lockin_boundary.png",
    "fig_kuhn_fusion.png": "kuhn_sweep/figF_fusion.png",
    "fig_abc_variance.png": "abc_conflict/fig_abc_variance.png",
    "fig_schur_generic.png": "schur_generic/fig_schur_generic.png",
    "fig_kuhn_trajectories_by_topology.png":
        "network_epistemology/fig_kuhn_trajectories_by_topology.png",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, copy nothing")
    args = ap.parse_args()

    dest_dir = ROOT / "paper" / "figures"
    dest_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, rel in PAPER_FIGURES.items():
        src = ROOT / "results" / rel
        if not src.exists():
            missing.append((name, rel))
            print(f"  MISSING  {name}  <-  results/{rel}")
            continue
        if args.check:
            print(f"  ok       {name}  <-  results/{rel}")
        else:
            shutil.copy2(src, dest_dir / name)
            print(f"  copied   {name}  <-  results/{rel}")
    if missing:
        print(f"\n{len(missing)} source figure(s) missing -- run the generating "
              f"experiment(s) first (see EXPERIMENTS.md), then re-run this script.")
        return 1
    print(f"\nall {len(PAPER_FIGURES)} paper figures accounted for -> {dest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
