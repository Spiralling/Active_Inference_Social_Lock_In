"""Phlogiston model -- the reusable builders for the multi-agent structure-learning world.

The phlogiston scenario is the project's foundational paradigm-shift world: a population
holds an over-wired phlogiston (mass-law) Bayes net, a scarce low-conviction *vanguard*
runs the refuting gravimetric experiment, and the question is whether the disconfirming
precision *diffuses through the trust graph* fast enough to drive a structural revolution
(the contested belt edges get pruned) versus an evidential lock-in (it stalls).

This module is the **pure compute library** for that model -- ``run_world`` (a thin wrapper
over ``simulation.run_simulation``) plus the topology / bridge / placement sweep builders and
the null controls. No matplotlib, no file I/O, no ``main``: the figures, saving, and printing
live in the experiment module (``experiments/multiagent_topology.py``). The phlogiston
``Scenario`` substrate (``build_substrate`` etc.) is re-exported here so this is the single
import surface for the model.

The three jit kernels live (verbatim from nb43) in ``simulation.run_simulation``, so the
numbers are byte-stable.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from src.structural import graphs
from src.structural.phlogiston import StructuralConfig
from src.structural.simulation import run_simulation, AgentSpec
from src.structural.scenarios import (
    build_substrate, phlogiston_scenario, Substrate, BELT, DEFAULT_SPURIOUS)

__all__ = [
    "vanguard_indices", "run_world", "topology_panel", "bridge_sweep",
    "placement_contrast", "run_controls", "CALIB",
    # re-exported substrate (single import surface for the model)
    "build_substrate", "phlogiston_scenario", "Substrate", "BELT", "DEFAULT_SPURIOUS",
    "StructuralConfig", "AgentSpec",
]

# Calibrated operating point: the sweet spot where the vanguard's evidence is SCARCE and
# DIFFUSING, so network topology is the load-bearing variable.
#   sigma_o = 0.5          -- structural signal visible above observation noise (nb42 used 0.1).
#   gamma_rest = 1.0       -- the rest are PURE RECEIVERS: belt evidence reaches them ONLY via
#                             fusion from the vanguard. (At gamma_rest=0.95 the 5% residual
#                             channel self-saturates and topology washes out -- near-total
#                             self-censorship is what makes topology load-bearing.)
#   conviction_rest = 0.20 -- poised just below the well-mixed belt evidence (~0.24 of v_e):
#                             an isolated rest agent never revolts; a well-connected one does.
#   n_vanguard = 6         -- a small persistent source (10% of the population).
CALIB = dict(conviction_rest=0.20, lambda_vanguard=0.05, n_vanguard=6,
             gamma_rest=1.0, sigma_o=0.5, t_shift=40, n_steps=120,
             snapshot_every=5, seed=0)


def vanguard_indices(graph: graphs.Graph, n_vanguard: int,
                     placement: str, seed: int) -> np.ndarray:
    """Which agents are the vanguard. ``central`` = highest-degree (well-trusted hubs),
    ``peripheral`` = lowest-degree, ``random`` = a seeded subset, ``block0`` = the first
    community block (needs a community graph). Ties broken by index."""
    A = np.asarray(graph.A)
    deg = A.sum(axis=1)
    n = graph.n
    if placement == "central":
        return np.argsort(-deg, kind="stable")[:n_vanguard]
    if placement == "peripheral":
        return np.argsort(deg, kind="stable")[:n_vanguard]
    if placement == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=n_vanguard, replace=False))
    if placement == "block0":
        if graph.membership is None:
            raise ValueError("placement='block0' needs a community graph (membership)")
        return np.where(np.asarray(graph.membership) == 0)[0][:n_vanguard]
    raise ValueError(f"unknown placement {placement!r}")


def run_world(graph: graphs.Graph, *, cfg: StructuralConfig | None = None,
              sub: Substrate | None = None,
              conviction_rest: float = 0.20, lambda_vanguard: float = 0.05,
              n_vanguard: int = 6, vanguard_placement: str = "central",
              gamma_rest: float = 1.0, fuse_mode: str = "posterior",
              eta: float = 0.5, n_steps: int = 120, t_shift: int = 40,
              sigma_o: float = 0.5, snapshot_every: int = 5, seed: int = 0) -> dict:
    """Simulate ``N = graph.n`` agents pooling precision over ``graph`` and read out the
    conviction-gated belt prune per agent.

    Every agent holds the SAME over-wired phlogiston prior. The heterogeneity:
      * a ``n_vanguard``-agent vanguard with ``gamma=0`` (runs the refuting gravimetric
        experiment) and low conviction ``lambda_vanguard``;
      * the rest self-censor (``gamma_rest`` -> down-weight the disconfirming channel) and
        hold moderate conviction ``conviction_rest`` (poised just above the revolution
        boundary).
    The only inter-agent influence is the genuine precision fusion ``step.fuse`` (``fuse_mode``).
    The prune is a READOUT of where each fused agent lands (it never mutates the net).
    """
    cfg = (StructuralConfig(t_shift=t_shift, n_steps=n_steps, sigma_o=sigma_o)
           if cfg is None else cfg)
    if sub is None:
        sub = build_substrate(cfg)
    scenario = phlogiston_scenario(cfg, sub=sub)
    N, m = graph.n, scenario.m

    van = vanguard_indices(graph, n_vanguard, vanguard_placement, seed)
    is_van = np.zeros(N, bool)
    is_van[van] = True

    # per-agent conviction threshold and self-censorship gamma
    lam = np.where(is_van, lambda_vanguard, conviction_rest)               # (N,)
    gamma = np.where(is_van, 0.0, gamma_rest)                              # (N,)

    # per-agent observation weights over the gravimetric rows: down-weight the
    # disconfirming channel by (1 - gamma_i) (vanguard keeps it at full weight).
    w_obs = np.ones((N, m))
    disc = np.asarray(sub.disc_rows, dtype=int)
    w_obs[:, disc] = (1.0 - gamma)[:, None]

    spec = AgentSpec(w_obs=jnp.asarray(w_obs), lam=lam)
    r = run_simulation(scenario, graph, spec, fuse_mode=fuse_mode, eta=eta,
                       snapshot_every=snapshot_every, seed=seed)
    r["vanguard_idx"] = np.asarray(van)
    r["is_vanguard"] = is_van
    r["vanguard_placement"] = vanguard_placement
    return r


def topology_panel(N: int = 60, *, sub=None, cfg=None, **kw) -> list[dict]:
    """The F1 panel: one representative graph per family, ordered by connectivity.
    ``isolated`` (lambda2=0) ... ``complete`` (max mixing). community is built then lightly
    bridged so it is connected (lambda2 > 0)."""
    base = graphs.community([N // 3] * 3, intra=0.45, inter=0.0, seed=0)
    panel = [
        ("isolated", base.isolated()),
        ("ring", graphs.ring(N, mean_degree=2)),
        ("erdos_renyi", graphs.erdos_renyi(N, mean_degree=4, seed=0)),
        ("watts_strogatz", graphs.watts_strogatz(N, mean_degree=4, rewiring_p=0.1, seed=0)),
        ("community+bridge", base.with_bridge(inter=0.03)),
        ("complete", graphs.complete(N)),
    ]
    out = []
    for label, g in panel:
        r = run_world(g, sub=sub, cfg=cfg, **kw)
        r["panel_label"] = label
        out.append(r)
    return out


def bridge_sweep(N: int = 60, inters=(0.0, 0.005, 0.01, 0.02, 0.04, 0.08),
                 *, sub=None, cfg=None, **kw) -> list[dict]:
    """F2: the clean knob. Fix the within-block structure (``intra``), vary only the
    cross-block bridge density ``inter`` -> sweeps ``lambda2`` while holding degree roughly
    fixed, de-confounding connectivity from density. Vanguard sits in block 0."""
    base = graphs.community([N // 3] * 3, intra=0.45, inter=0.0, seed=0)
    out = []
    for inter in inters:
        g = base.with_bridge(inter=float(inter))
        r = run_world(g, sub=sub, cfg=cfg, vanguard_placement="block0", **kw)
        r["inter"] = float(inter)
        out.append(r)
    return out


def placement_contrast(N: int = 60, *, sub=None, cfg=None, **kw) -> dict:
    """F3: does WHERE the vanguard sits matter? Central (hub) vs peripheral vanguard on a
    fixed scale-free graph (degree heterogeneity makes the contrast meaningful)."""
    g = graphs.scale_free(N, mean_degree=4, seed=0)
    return {p: run_world(g, sub=sub, cfg=cfg, vanguard_placement=p, **kw)
            for p in ("central", "peripheral")}


def run_controls(sub: Substrate, cfg: StructuralConfig, N: int = 60, **kw) -> dict:
    """The null/sanity controls. Returns a dict of measured numbers; the experiment asserts
    the robust ones and reports the topology-dependent ones honestly."""
    iso = graphs.complete(N).isolated()
    comp = graphs.complete(N)
    nv = kw.get("n_vanguard", 4)

    out = {}
    # 1. isolated (lambda2 = 0): only the vanguard ever gathers disconfirming evidence.
    out["isolated_revolted"] = run_world(iso, sub=sub, cfg=cfg, **kw)["final_revolted_fraction"]
    out["isolated_expected"] = nv / N
    # 2. complete: maximal spread.
    out["complete_revolted"] = run_world(comp, sub=sub, cfg=cfg, **kw)["final_revolted_fraction"]
    # 3a. conviction_rest -> 0: everyone prunes (no protection).
    kw0 = {**kw, "conviction_rest": 0.0, "lambda_vanguard": 0.0}
    out["conviction0_revolted"] = run_world(comp, sub=sub, cfg=cfg, **kw0)["final_revolted_fraction"]
    # 3b. conviction -> inf for EVERYONE (vanguard too): the echo chamber, none prune.
    kwInf = {**kw, "conviction_rest": 50.0, "lambda_vanguard": 50.0}
    out["convictionInf_revolted"] = run_world(comp, sub=sub, cfg=cfg, **kwInf)["final_revolted_fraction"]
    # 4. gamma_rest = 0 (everyone runs the experiment): topology stops mattering.
    kwg = {**kw, "gamma_rest": 0.0}
    out["gamma0_isolated"] = run_world(iso, sub=sub, cfg=cfg, **kwg)["final_revolted_fraction"]
    out["gamma0_complete"] = run_world(comp, sub=sub, cfg=cfg, **kwg)["final_revolted_fraction"]
    # 5. no refutation (t_shift > n_steps): null world, no revolt anywhere.
    cfg_null = StructuralConfig(t_shift=cfg.n_steps + 10, n_steps=cfg.n_steps, sigma_o=cfg.sigma_o)
    sub_null = build_substrate(cfg_null)
    out["null_revolted"] = run_world(comp, sub=sub_null, cfg=cfg_null, **kw)["final_revolted_fraction"]
    # finiteness / PD of the BMR readout
    out["dF_finite"] = bool(np.isfinite(
        run_world(comp, sub=sub, cfg=cfg, **kw)["final_dF"]).all())
    return out
