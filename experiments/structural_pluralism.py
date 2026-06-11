"""Structural pluralism: communities that VALUE different theories learn different WIRING.

The structure-level sibling of the means-level lock-in results -- the one thing the paper always
*claimed* (paradigms are different network structures) but never *demonstrated* (every prior result
diverged in the posterior MEANS on a frozen graph). Made concrete on the cosmology substrate (6
contested anomaly->commitment couplings; phlogiston has only ONE relational edge, too thin):

  * VALUE -> ATTENTION -> STRUCTURE. Each agent weights the relational ``balance(theory)`` row in
    proportion to how much it values that theory (one rule, derived from ``u_agent``).
  * UNDERDETERMINED world. ``phi`` is a blend of two theories, so the data genuinely support more
    than one wiring; value resolves *which structure* each community builds.
  * DISCONNECTION is the lock-in lever at the level of structure. Disconnected communities learn
    genuinely different couplings ``Pi[anomaly, commitment]``; CONNECTING them collapses the
    difference (precision-pooling merges the wirings) -- the structural reading of section 7.2.

Honest mechanism note: the fully-emergent endogenous-gamma path does NOT produce value-aligned
structural divergence here (both communities collapse to the incumbent); the mechanism that carries
structural pluralism is selective attention. Self-contained (cosmology substrate, no model lib);
classed under the phlogiston model as a structure-level companion to the lock-in family.
"""
from __future__ import annotations

import json
import dataclasses

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.structural import graphs
from src.structural.simulation import run_simulation, AgentSpec
from src.structural.scenarios import (
    cosmology_scenario, cosmology_theory_means, cosmology_utility_toward, COSMOLOGY_EDGES)
from experiments.registry import ExperimentSpec, register


def build(*, n, n_steps, commits, value_a, value_b, w_hi, w_lo):
    """The underdetermined scenario + the value-derived attention weights (shared across seeds)."""
    base = cosmology_scenario(n_steps=n_steps, t1=10_000, t2=20_000, sigma_o=0.5)  # single epoch
    names = base.names
    idx = {nm: i for i, nm in enumerate(names)}
    disc = list(base.disc_rows)                       # balance(dm), balance(mg), balance(sv)
    edges_ij = [(idx[a], idx[c]) for (a, c) in COSMOLOGY_EDGES]

    # UNDERDETERMINED world: blend the two communities' theories so the data support both wirings.
    mu = np.asarray(cosmology_theory_means())                       # (E, d)
    blend = 0.5 * (mu[commits.index(value_a)] + mu[commits.index(value_b)])
    scn = dataclasses.replace(base, phis=jnp.asarray(np.tile(blend, (n_steps, 1))))

    # per-agent VALUE u (A values value_a, B values value_b)
    uA = np.asarray(cosmology_utility_toward(value_a))
    uB = np.asarray(cosmology_utility_toward(value_b))
    u_agent = np.where((np.arange(n) < n // 2)[:, None], uA[None, :], uB[None, :]).astype("float32")

    # VALUE -> ATTENTION (one rule): weight balance(c) high iff this agent values commitment c.
    m = base.m
    w = np.ones((n, m))
    for k, c in zip(disc, commits):
        val = u_agent[:, idx[f"{c}_commitment"]]                    # +1 home, -1 rival
        w[:, k] = np.where(val > 0, w_hi, w_lo)
    return scn, edges_ij, jnp.asarray(w)


def structure(P, edges_ij):
    """The 6 learned contested couplings Pi[anomaly, commitment] of a (mean) net."""
    return np.array([P[i, j] for (i, j) in edges_ij])


def by_commitment(s):
    """|coupling| summed over the two anomalies, per rival commitment -> (3,)."""
    return np.array([abs(s[0]) + abs(s[1]), abs(s[2]) + abs(s[3]), abs(s[4]) + abs(s[5])])


def reldist(sA, sB):
    return float(np.linalg.norm(sA - sB) / (0.5 * (np.linalg.norm(sA) + np.linalg.norm(sB)) + 1e-9))


def run_condition(scn, w, edges_ij, graph, seed, *, n, omega):
    spec = AgentSpec(w_obs=w, lam=jnp.full((n,), 0.2))
    r = run_simulation(scn, graph, spec, forgetting=omega, snapshot_every=5, seed=seed)
    P = r["snap_Pi"][-1]                                            # (N, d, d) final
    sA = structure(P[: n // 2].mean(0), edges_ij)
    sB = structure(P[n // 2:].mean(0), edges_ij)
    return sA, sB, float(np.abs(r["snap_Pi"]).max())


def run(out_dir, params: dict) -> None:
    n = params["N"]
    seeds = tuple(params["SEEDS"])
    omega = params["OMEGA"]
    n_steps = params["N_STEPS"]
    w_hi, w_lo = params["W_HI"], params["W_LO"]
    value_a, value_b = params["VALUE_A"], params["VALUE_B"]
    commits = tuple(params["COMMITS"])

    scn, edges_ij, w = build(n=n, n_steps=n_steps, commits=commits, value_a=value_a,
                             value_b=value_b, w_hi=w_hi, w_lo=w_lo)

    conditions = {
        "disconnected": graphs.community([n // 2, n // 2], intra=1.0, inter=0.0, seed=0),
        "connected": graphs.complete(n),
    }
    agg = {}
    print(f"structural pluralism: A values {value_a}, B values {value_b}; "
          f"underdetermined world; omega={omega}; seeds={seeds}")
    for cond, graph in conditions.items():
        dists, cAs, cBs, mxs = [], [], [], []
        for s in seeds:
            sA, sB, mx = run_condition(scn, w, edges_ij, graph, s, n=n, omega=omega)
            dists.append(reldist(sA, sB)); cAs.append(by_commitment(sA))
            cBs.append(by_commitment(sB)); mxs.append(mx)
        agg[cond] = {
            "dist_mean": float(np.mean(dists)), "dist_std": float(np.std(dists)),
            "cA": np.mean(cAs, axis=0), "cB": np.mean(cBs, axis=0),
            "max_Pi": float(np.max(mxs)),
        }
        a = agg[cond]
        print(f"  {cond:>12}: struct-dist = {a['dist_mean']:.3f} +/- {a['dist_std']:.3f} | "
              f"A favors {commits[a['cA'].argmax()]}, B favors {commits[a['cB'].argmax()]} | "
              f"max|Pi|={a['max_Pi']:.1f}")

    dc, cc = agg["disconnected"], agg["connected"]
    # the claim: disconnected communities wire their OWN value's theory and stay apart; connecting
    # them collapses the structural difference (precision-pooling merges the wirings).
    assert dc["cA"].argmax() == commits.index(value_a), "A should wire its valued theory when disconnected"
    assert dc["cB"].argmax() == commits.index(value_b), "B should wire its valued theory when disconnected"
    assert dc["dist_mean"] > 3.0 * cc["dist_mean"] + 1e-6, \
        f"connection should collapse the divergence ({dc['dist_mean']:.2f} -> {cc['dist_mean']:.2f})"
    assert dc["max_Pi"] < 500 and cc["max_Pi"] < 500, "forgetting should bound the precision"

    # ---- figure: the two wirings (disconnected vs connected) + the collapse ----
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(3); width = 0.38
    for ax, cond, title in ((a0, "disconnected", "Disconnected: different wirings"),
                            (a1, "connected", "Connected: fusion merges them")):
        a = agg[cond]
        ax.bar(x - width / 2, a["cA"], width, color="#2c6fbb", label=f"A (values {value_a})")
        ax.bar(x + width / 2, a["cB"], width, color="#c0392b", label=f"B (values {value_b})")
        ax.set_xticks(x); ax.set_xticklabels(["dark\nmatter", "mod.\ngravity", "scale\nvariant"])
        ax.set_ylabel("learned coupling  |Pi[anomaly, commitment]|")
        ax.set_title(title); ax.legend(fontsize=8)
    conds = ["disconnected", "connected"]
    a2.bar(conds, [agg[c]["dist_mean"] for c in conds],
           yerr=[agg[c]["dist_std"] for c in conds], color=["#6c3483", "#aab7b8"], capsize=5)
    a2.set_ylabel("structural distance  A vs B  (relative)")
    a2.set_title("Pluralism survives only under disconnection")
    fig.suptitle("Structural pluralism: value -> selective attention -> different network structure",
                 fontsize=12)
    plt.tight_layout(); plt.savefig(out_dir / "diag_structural_pluralism.png", dpi=120); plt.close(fig)

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        commits=np.array(commits),
        disc_cA=dc["cA"], disc_cB=dc["cB"], conn_cA=cc["cA"], conn_cB=cc["cB"],
        disc_dist=dc["dist_mean"], conn_dist=cc["dist_mean"],
        disc_dist_std=dc["dist_std"], conn_dist_std=cc["dist_std"],
    )
    summary = {
        "config": {"N": n, "seeds": list(seeds), "omega": omega, "n_steps": n_steps,
                   "value_A": value_a, "value_B": value_b, "world": "underdetermined blend"},
        "disconnected": {"dist": dc["dist_mean"], "dist_std": dc["dist_std"],
                         "A_wiring": dc["cA"].tolist(), "B_wiring": dc["cB"].tolist()},
        "connected": {"dist": cc["dist_mean"], "dist_std": cc["dist_std"],
                      "A_wiring": cc["cA"].tolist(), "B_wiring": cc["cB"].tolist()},
        "mechanism": "value -> selective attention -> structure (endogenous-gamma does NOT align)",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figure / summary to {out_dir}")
    print(f"HEADLINE: disconnected communities wire their valued theory (dist {dc['dist_mean']:.2f}); "
          f"connecting them collapses the structural difference to {cc['dist_mean']:.2f} "
          f"(precision-pooling merges the wirings) -- structural pluralism needs disconnection.")


register(ExperimentSpec(
    model="phlogiston",
    name="structural_pluralism",
    description="Communities that VALUE different theories learn different WIRING (value -> selective "
                "attention -> structure); disconnection preserves the structural divergence, "
                "connection collapses it -- the structure-level sibling of the lock-in results.",
    run=run,
    out_dir="structural_pluralism",
    params=dict(N=60, SEEDS=(0, 1, 2, 3, 4), OMEGA=0.9, N_STEPS=120, W_HI=1.0, W_LO=0.1,
                VALUE_A="dark_matter", VALUE_B="scale_variant_laws",
                COMMITS=("dark_matter", "modified_gravity", "scale_variant_laws")),
    seeds=(0, 1, 2, 3, 4),
    consumes=dict(figures=["diag_structural_pluralism"]),
))
