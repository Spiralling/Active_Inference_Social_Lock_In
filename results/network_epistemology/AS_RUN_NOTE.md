# As-run note

`fig_kuhn_trajectories_by_topology.png` and `traj_stats.json` were rescued (2026-08-14)
from `results/_traj_preview/`, the output of the deleted one-off
`scripts/run_traj_preview.py` (run 2026-06-12). That script ran ONLY the 80 trajectory
jobs of this experiment, seeded identically to the canonical jobs, so a fresh
`python scripts/run.py run network_epistemology` reproduces these traces exactly --
and also regenerates `manifest.json`, which here still predates the R_TRAJ /
SWEEP_BETAS / CONVICTION params (it records the older, smaller job set).

The paper figure (paper/figures/fig_kuhn_trajectories_by_topology.png) is a copy of
the rescued png; its flip medians 220/232/453/488 match the paper text.
