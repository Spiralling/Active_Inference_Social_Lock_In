"""The config space and tiered job generator for the big sweep.

Pure data: no simulation here. A *job* is a flat dict with every axis set (swept axes vary, the
rest sit at DEFAULTS) plus bookkeeping keys ``tier``, ``job`` (which phase map), ``seed``. The
DEFAULT cell is the validated structural-pluralism anchor (n=2, disconnected, full blend,
beta=10, omega=0.9) -- so Tier-1 maps pass through a known-answer point.

Axes that the multi-agent engine cannot express are deliberately omitted (per-community *initial
priors*: run_simulation broadcasts one scenario prior to all agents -- a frontier item, not a
sweep axis).
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# Axis grids
# ----------------------------------------------------------------------
OMEGA = [0.70, 0.80, 0.85, 0.90, 0.95, 1.0]          # forgetting
INTER = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0]  # inter-block density -> lambda2
NPOP = [30, 60, 120, 240]                             # population size
NCOMM = [1, 2, 3]                                     # # value-communities (block i values theory i)
BETA = [1, 2, 5, 10, 100]                             # value->attention selectivity (W_HI/W_LO)
TILT = [0.0, 0.5, 1.0, 2.0, 4.0]                      # conviction tilt (motivated update strength)
WORLD = ["stable", "underdetermined_blend", "changing_epochs"]
BLEND = [0.0, 0.5, 1.0]                               # underdetermination (1.0 = de-risk point)
SIGMA = [0.25, 0.5, 1.0]                              # observation noise
FUSE = ["posterior", "deposit_pool", "deposit_keep"]

# The DEFAULT cell == the validated anchor (reproduces struct-dist ~1.25 disconnected).
DEFAULTS = dict(
    N=60, inter=0.0, omega=0.90, n_communities=2, attention_beta=10,
    conviction_tilt=0.0, world_mode="underdetermined_blend", blend_ratio=1.0,
    sigma_o=0.5, fuse_mode="posterior", n_steps=120,
)

TIER1_SEEDS = (0, 1, 2, 3, 4)

# Which two axes each Tier-1 job sweeps (used by analyze.py to build the phase grids).
JOB_AXES = {
    "A_omega_inter": ("omega", "inter"),
    "B_inter_ncomm": ("inter", "n_communities"),
    "C_omega_tilt": ("omega", "conviction_tilt"),
    "D_N_inter": ("N", "inter"),
    "E_blend_beta": ("blend_ratio", "attention_beta"),
    "F_sigma_omega": ("sigma_o", "omega"),
    "G_fuse_inter": ("fuse_mode", "inter"),
}


def _job(tier, job, seed, **over):
    cfg = dict(DEFAULTS)
    cfg.update(over)
    cfg["tier"], cfg["job"], cfg["seed"] = tier, job, seed
    return cfg


def tier1_jobs():
    """Dense 2-D phase maps; non-swept axes at DEFAULTS; x5 seeds."""
    jobs = []
    for s in TIER1_SEEDS:
        for om in OMEGA:
            for it in INTER:
                jobs.append(_job(1, "A_omega_inter", s, omega=om, inter=it))
        for it in INTER:
            for nc in NCOMM:
                jobs.append(_job(1, "B_inter_ncomm", s, inter=it, n_communities=nc))
        # C is the LOCK-IN map -> changing-epochs world (conviction vs forgetting)
        for om in OMEGA:
            for tl in TILT:
                jobs.append(_job(1, "C_omega_tilt", s, omega=om, conviction_tilt=tl,
                                 world_mode="changing_epochs"))
        for n in NPOP:
            for it in INTER:
                jobs.append(_job(1, "D_N_inter", s, N=n, inter=it))
        for bl in BLEND:
            for be in BETA:
                jobs.append(_job(1, "E_blend_beta", s, blend_ratio=bl, attention_beta=be))
        for sg in SIGMA:
            for om in OMEGA:
                jobs.append(_job(1, "F_sigma_omega", s, sigma_o=sg, omega=om))
        for fm in FUSE:
            for it in INTER:
                jobs.append(_job(1, "G_fuse_inter", s, fuse_mode=fm, inter=it))
    return jobs


def tier2_jobs(n_samples=1500, seed=12345):
    """Broad random sample over the full joint space (x1 seed each) -- catches interactions the
    2-D slices miss; feeds the per-axis importance read-out in analyze.py."""
    rng = np.random.default_rng(seed)
    def pick(grid):
        return grid[int(rng.integers(len(grid)))]
    jobs = []
    for i in range(n_samples):
        world = pick(WORLD)
        jobs.append(_job(
            2, "random", int(rng.integers(10_000)),
            N=pick(NPOP), inter=pick(INTER), omega=pick(OMEGA),
            n_communities=pick(NCOMM), attention_beta=pick(BETA),
            conviction_tilt=pick(TILT), world_mode=world, blend_ratio=pick(BLEND),
            sigma_o=pick(SIGMA), fuse_mode=pick(FUSE),
        ))
    return jobs


def all_jobs(tier="all", tier2_n=1500):
    if tier in ("1", 1):
        return tier1_jobs()
    if tier in ("2", 2):
        return tier2_jobs(tier2_n)
    return tier1_jobs() + tier2_jobs(tier2_n)
