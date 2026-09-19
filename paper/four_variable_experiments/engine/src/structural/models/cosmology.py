"""Cosmology model -- the reusable builders for the changing-cosmology-world experiments.

The world is three genuinely different cosmology theories (the ``landscape_presets`` Bayes-net
presets) that the truth cycles through in three epochs (dark matter -> modified gravity ->
scale-variant laws). ``N`` heterogeneous agents in three communities (differing in
*conservatism* = prior precision and *conviction* = a value tilt toward the community's home
theory) sit on a trust graph, each step observe the current world and fuse precision -- the
same ``simulation.run_simulation`` engine as the phlogiston model, with ``cosmology_scenario``
plugged in.

This module is the **pure compute library** shared across the cosmology experiment family
(``cosmology_tracking`` and the dependents ``bayesnet_comms`` / ``cosmology_sweeps`` /
``cosmology_forgetting`` / ``partial_obs_comms`` / ``cosmology_coarse_world``). It holds the
model's default config constants and the population / spec / graph builders and read-outs --
no matplotlib, no file I/O, no ``main``. The scenario helpers are re-exported here so this is
the single import surface for the model.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from src.structural import graphs
from src.structural.simulation import run_simulation, AgentSpec
from src.structural import scenarios as sc
from src.structural.landscape_presets import ClusterProfile, sample_cluster_scalings

__all__ = [
    # config constants (the cosmology family's defaults; dependents import these)
    "N_AGENTS", "N_STEPS", "T1", "T2", "SIGMA_O", "SEEDS",
    "INTER_CONNECTED", "INTER_DISCONNECTED", "INTRA", "CONVICTION_LEVEL",
    "PROFILES", "HOMES", "COMM_LABELS", "EPOCH_NAMES", "CELLS",
    # builders + read-outs
    "cosmology_population", "cosmology_spec", "cosmology_graph",
    "held_by_community", "alignment_by_community", "cell_metrics", "run_cell",
    # re-exported scenario helpers (single import surface)
    "sc", "AgentSpec",
]

# Configuration -- the world, the population, the levers.
N_AGENTS = 150
N_STEPS, T1, T2 = 180, 60, 120
SIGMA_O = 0.5
SEEDS = (0, 1, 2)

INTER_CONNECTED = 0.08        # bridged communities  (lambda_2 > 0)
INTER_DISCONNECTED = 0.0      # echo chambers         (lambda_2 = 0)
INTRA = 0.5
CONVICTION_LEVEL = 4.0        # the tilt strength when the conviction lever is ON

# Three communities, by conservatism (prior precision) and conviction (utility scale).
PROFILES: tuple[ClusterProfile, ...] = (
    ClusterProfile("conservative", 1 / 3, precision_mean=0.9, precision_sd=0.15,
                   utility_scale_mean=1.2, utility_scale_sd=0.2),
    ClusterProfile("moderate", 1 / 3, precision_mean=0.2, precision_sd=0.15,
                   utility_scale_mean=0.6, utility_scale_sd=0.2),
    ClusterProfile("frontier", 1 / 3, precision_mean=-0.4, precision_sd=0.15,
                   utility_scale_mean=0.2, utility_scale_sd=0.2),
)
# Each community is committed (when conviction is on) to a DIFFERENT theory: the conservative
# establishment to the OLD incumbent (dark matter), the frontier to the eventually-true theory.
HOMES = ("dark_matter", "modified_gravity", "scale_variant_laws")
COMM_LABELS = tuple(p.label for p in PROFILES)
EPOCH_NAMES = sc.COSMOLOGY_EPOCHS         # ("dark_matter","modified_gravity","scale_variant_laws")

# The lever 2x2 used by cosmology_tracking (cell label, disconnect, conviction tilt).
CELLS = (("neither", False, 0.0), ("disconnection", True, 0.0),
         ("conviction", False, CONVICTION_LEVEL), ("both", True, CONVICTION_LEVEL))


def cosmology_population(n_agents: int = N_AGENTS,
                         profiles: tuple[ClusterProfile, ...] = PROFILES,
                         seed: int = 0) -> dict:
    """Sample per-agent conservatism (``precision_scale``) and conviction strength
    (``utility_scale``), block-ordered so community ``k`` is contiguous (membership == cluster
    id). Reuses ``landscape_presets.sample_cluster_scalings`` then sorts by cluster so the graph
    blocks line up with the belief communities."""
    samp = sample_cluster_scalings(n_agents, profiles, seed=seed)
    order = np.argsort(np.asarray(samp.cluster_id), kind="stable")
    ps = np.asarray(samp.precision_scale)[order]
    us = np.asarray(samp.utility_scale)[order]
    cid = np.asarray(samp.cluster_id)[order]
    sizes = [int((cid == k).sum()) for k in range(len(profiles))]
    return {"precision_scale": ps, "utility_scale": us, "cluster_id": cid, "sizes": sizes}


def cosmology_spec(scn: sc.Scenario, pop: dict, *, conviction: float = 0.0,
                   homes: tuple[str, ...] = HOMES) -> AgentSpec:
    """Build the per-agent ``AgentSpec``. Everyone observes the world (``w_obs`` all ones --
    no vanguard self-censorship here; the cosmology levers are disconnection and conviction,
    not evidence scarcity). ``precision_scale`` = community conservatism. When ``conviction >
    0`` the conviction lever is ON: a per-agent tilt ``conviction * max(utility_scale, 0)``
    toward the agent's community home theory."""
    ps = pop["precision_scale"]
    us = pop["utility_scale"]
    cid = pop["cluster_id"]
    N = len(cid)
    tilt = None
    u_agent = None
    if conviction > 0.0:
        u_rows = np.stack([np.asarray(sc.cosmology_utility_toward(homes[int(cid[i])]))
                           for i in range(N)])
        u_agent = jnp.asarray(u_rows)
        tilt = jnp.asarray(conviction * np.clip(us, 0.0, None))
    return AgentSpec(w_obs=jnp.ones((N, scn.m)), lam=np.ones(N),
                     precision_scale=jnp.asarray(ps), tilt=tilt, u_agent=u_agent)


def cosmology_graph(sizes: list[int], *, connected: bool, seed: int = 0) -> graphs.Graph:
    inter = INTER_CONNECTED if connected else INTER_DISCONNECTED
    return graphs.community(sizes, intra=INTRA, inter=inter, seed=seed)


def held_by_community(r: dict, cid: np.ndarray, theory_mu: np.ndarray) -> np.ndarray:
    """(n_comm, S) the nearest candidate-theory index of each community's *mean* belief at each
    snapshot -- the headline tracking read-out."""
    h = r["snap_h"]                                    # (S, N, d)
    n_comm = int(cid.max()) + 1
    return np.stack([sc.closest_theory(h[:, cid == k, :].mean(axis=1), theory_mu)
                     for k in range(n_comm)])          # (n_comm, S)


def alignment_by_community(r: dict, cid: np.ndarray, theory_mu: np.ndarray) -> np.ndarray:
    """(n_comm, S, E) the soft per-theory alignment of each community's mean belief -- the
    graded version of ``held_by_community`` (so partial drift between theories is visible)."""
    h = r["snap_h"]
    n_comm = int(cid.max()) + 1
    return np.stack([sc.theory_alignment(h[:, cid == k, :].mean(axis=1), theory_mu)
                     for k in range(n_comm)])          # (n_comm, S, E)


def cell_metrics(r: dict, cid: np.ndarray, theory_mu: np.ndarray,
                 true_per_snap: np.ndarray) -> dict:
    """Scalar metrics for one (disconnect, conviction, seed) cell."""
    held = held_by_community(r, cid, theory_mu)        # (n_comm, S)
    end = held[:, -1]                                  # final theory per community
    n_comm = held.shape[0]
    # fraction of (community, snapshot) pairs holding the *current* true theory (warm-up
    # dropped -- the first snapshot is t=0, always the incumbent).
    on_truth = (held[:, 1:] == true_per_snap[None, 1:]).mean()
    return {
        "end_theories": end.tolist(),
        "n_distinct_end": int(len(np.unique(end))),
        "reached_final_frac": float((end == theory_mu.shape[0] - 1).mean()),
        "comm0_stuck_on_incumbent": bool(end[0] == 0),
        "on_truth_fraction": float(on_truth),
        "final_disagreement": float(r["disagreement_t"][-1]),
        "max_disagreement": float(r["disagreement_t"].max()),
        "dF_finite": bool(np.isfinite(r["final_dF"]).all()),
    }


def run_cell(scn: sc.Scenario, *, disconnect: bool, conviction: float, seed: int,
             snapshot_every: int = 10) -> tuple[dict, dict]:
    """One cell of the lever 2x2 at one seed. Returns ``(raw_run, metrics)``."""
    pop = cosmology_population(seed=seed)
    graph = cosmology_graph(pop["sizes"], connected=not disconnect, seed=seed)
    spec = cosmology_spec(scn, pop, conviction=conviction)
    r = run_simulation(scn, graph, spec, snapshot_every=snapshot_every, seed=seed)
    theory_mu = sc.cosmology_theory_means()
    true_per_snap = np.asarray(scn.epoch_t)[r["snap_t"]]
    met = cell_metrics(r, pop["cluster_id"], theory_mu, true_per_snap)
    met["lambda2"] = float(r["lambda2"])
    return {"r": r, "pop": pop}, met
