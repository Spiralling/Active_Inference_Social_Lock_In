"""Build one run from a flat config dict, execute it, and return its metric battery.

Generalises the validated per-run logic of scripts/run_structural_pluralism.py to:
  * n value-communities (graph blocks = value communities; block c values theory VALUED[n][c]),
  * value -> attention DERIVED from u (weight balance(theory) high iff you value that theory;
    selectivity W_HI/W_LO set by attention_beta),
  * three world modes (stable / underdetermined blend / changing epochs),
  * conviction tilt, forgetting, fuse_mode, N, connectivity (inter-block density).
"""

from __future__ import annotations

import dataclasses
import numpy as np
import jax.numpy as jnp

from src.structural import graphs
from src.structural.simulation import run_simulation, AgentSpec
from src.structural.scenarios import (
    cosmology_scenario, cosmology_theory_means, cosmology_true_couplings,
    cosmology_utility_toward, COSMOLOGY_EDGES,
)
from scripts.sweep.metrics import compute_metrics

THEORIES = ("dark_matter", "modified_gravity", "scale_variant_laws")  # == theory_mus / epoch order
VALUED = {1: (0,), 2: (0, 2), 3: (0, 1, 2)}                            # which theory each community values
INCUMBENT = 0
INTRA_BLOCK = 1.0


def _block_sizes(N, n):
    base = N // n
    sizes = [base] * n
    for i in range(N - base * n):
        sizes[i] += 1
    return sizes


def _build(cfg):
    """Return (scn, graph, spec, ctx) for one config."""
    N, n = int(cfg["N"]), int(cfg["n_communities"])
    n_steps = int(cfg["n_steps"])
    world = cfg["world_mode"]
    theory_mu = np.asarray(cosmology_theory_means())              # (E,d)

    # ---- scenario + world truth ----
    if world == "changing_epochs":
        scn = cosmology_scenario(n_steps=n_steps, t1=40, t2=80, sigma_o=float(cfg["sigma_o"]))
    else:
        scn = cosmology_scenario(n_steps=n_steps, t1=10_000, t2=20_000, sigma_o=float(cfg["sigma_o"]))
        valued_idx = VALUED[n]
        valued_mean = theory_mu[list(valued_idx)].mean(0)        # maximally-ambiguous point
        if world == "stable":
            phi = theory_mu[INCUMBENT]                           # determined incumbent world
        else:                                                    # underdetermined_blend
            r = float(cfg["blend_ratio"])
            phi = (1.0 - r) * theory_mu[INCUMBENT] + r * valued_mean
        scn = dataclasses.replace(scn, phis=jnp.asarray(np.tile(phi, (n_steps, 1))))

    names = scn.names
    idx = {nm: i for i, nm in enumerate(names)}
    m = scn.m
    disc = list(scn.disc_rows)                                   # balance(dm),balance(mg),balance(sv)
    edges_ij = [(idx[a], idx[c]) for (a, c) in COSMOLOGY_EDGES]

    # ---- communities, values, value->attention ----
    sizes = _block_sizes(N, n)
    bounds = np.cumsum([0] + sizes)
    community_idx = [np.arange(bounds[c], bounds[c + 1]) for c in range(n)]
    valued_idx = VALUED[n]

    W_HI, W_LO = 1.0, 1.0 / float(cfg["attention_beta"])
    w = np.ones((N, m))
    u_agent = np.zeros((N, len(names)), dtype="float32")
    for c, ci in enumerate(community_idx):
        ti = valued_idx[c]
        u_agent[ci] = np.asarray(cosmology_utility_toward(THEORIES[ti]))
        for k in range(len(disc)):
            w[np.ix_(ci, [disc[k]])] = W_HI if k == ti else W_LO

    tilt = (jnp.full((N,), float(cfg["conviction_tilt"]))
            if float(cfg["conviction_tilt"]) != 0.0 else None)
    spec = AgentSpec(w_obs=jnp.asarray(w), lam=jnp.full((N,), 0.2),
                     tilt=tilt, u_agent=jnp.asarray(u_agent))

    # ---- social graph ----
    if n == 1:
        graph = graphs.complete(N)
    else:
        graph = graphs.community(sizes, intra=INTRA_BLOCK, inter=float(cfg["inter"]),
                                 seed=int(cfg["seed"]))

    ctx = dict(community_idx=community_idx, edges_ij=edges_ij, theory_mus=theory_mu,
               true_couplings=np.asarray(cosmology_true_couplings()),
               n_communities=n, world_mode=world)
    return scn, graph, spec, ctx


def run_one(cfg):
    """Execute one config and return (scalars, traj). Pure function of cfg (seed inside)."""
    scn, graph, spec, ctx = _build(cfg)
    r = run_simulation(scn, graph, spec, forgetting=float(cfg["omega"]),
                       fuse_mode=cfg["fuse_mode"], snapshot_every=5, seed=int(cfg["seed"]))
    return compute_metrics(r, ctx)
