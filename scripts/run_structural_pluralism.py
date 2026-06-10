"""Structural pluralism: communities that VALUE different theories learn different WIRING.

This is the structure-level sibling of the means-level lock-in results: it shows the one
thing the paper has always *claimed* (paradigms are different network structures) but never
*demonstrated* (every prior result diverged in the posterior MEANS on a frozen graph).

The claim, made concrete on the cosmology substrate (6 contested anomaly->commitment couplings;
PHLOGISTON has only ONE relational edge, too thin to carry this):

  * VALUE -> ATTENTION -> STRUCTURE.  Each agent weights the relational ``balance(theory)`` row
    in proportion to how much it values that theory (one rule, derived from ``u_agent`` -- NOT
    a hand-set per-community mask).  Motivated/selective attention to confirming evidence.
  * UNDERDETERMINED world.  ``phi`` is a blend of two theories, so the data genuinely support
    more than one wiring; value resolves *which structure* each community builds, and neither
    community is simply wrong.
  * DISCONNECTION is the lock-in lever -- now at the level of structure.  Disconnected
    communities learn genuinely different couplings ``Pi[anomaly, commitment]``; CONNECTING them
    collapses the difference, because precision-pooling MERGES the wirings (each community
    inherits the other's edge).  That is "how you share evidence across different structures" on
    shared nodes: the structural union.  It is also why pluralism survives only under
    disconnection -- the structural reading of the paper's section 7.2.

Mechanism note (honest): the fully-emergent endogenous-gamma path does NOT produce value-aligned
structural divergence here (both communities collapse to the incumbent) -- gamma-silencing is the
E2 phlogiston mechanism, a different claim.  The mechanism that carries structural pluralism is
selective attention.

Readout ISOLATES structure: the learned off-diagonal couplings ``Pi[anomaly, commitment]`` (the 6
``COSMOLOGY_EDGES``), never the means -- the means are exactly what the operator-set-Fisher wall
lets move trivially.

Run it::

    python scripts/run_structural_pluralism.py
"""

from __future__ import annotations

import json
import sys
import dataclasses
from pathlib import Path

import numpy as np
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.structural import graphs
from src.structural.simulation import run_simulation, AgentSpec
from src.structural.scenarios import (
    cosmology_scenario, cosmology_theory_means, cosmology_utility_toward,
    COSMOLOGY_EDGES,
)

N = 60                                   # two communities of 30
SEEDS = (0, 1, 2, 3, 4)
OMEGA = 0.9                              # forgetting: bounded accumulation (NOT 1.0)
N_STEPS = 120
W_HI, W_LO = 1.0, 0.1                    # attention on a valued vs unvalued balance row
VALUE_A, VALUE_B = "dark_matter", "scale_variant_laws"   # the two communities' values
COMMITS = ("dark_matter", "modified_gravity", "scale_variant_laws")


def build():
    """The underdetermined scenario + the value-derived attention weights (shared across seeds)."""
    base = cosmology_scenario(n_steps=N_STEPS, t1=10_000, t2=20_000, sigma_o=0.5)  # single epoch
    names = base.names
    idx = {n: i for i, n in enumerate(names)}
    disc = list(base.disc_rows)                       # balance(dm), balance(mg), balance(sv)
    edges_ij = [(idx[a], idx[c]) for (a, c) in COSMOLOGY_EDGES]

    # UNDERDETERMINED world: blend the two communities' theories so the data support both wirings.
    mu = np.asarray(cosmology_theory_means())                       # (E, d)
    blend = 0.5 * (mu[COMMITS.index(VALUE_A)] + mu[COMMITS.index(VALUE_B)])
    scn = dataclasses.replace(base, phis=jnp.asarray(np.tile(blend, (N_STEPS, 1))))

    # per-agent VALUE u (A values VALUE_A, B values VALUE_B)
    uA = np.asarray(cosmology_utility_toward(VALUE_A))
    uB = np.asarray(cosmology_utility_toward(VALUE_B))
    u_agent = np.where((np.arange(N) < N // 2)[:, None], uA[None, :], uB[None, :]).astype("float32")

    # VALUE -> ATTENTION (one rule): weight balance(c) high iff this agent values commitment c.
    m = base.m
    w = np.ones((N, m))
    for k, c in zip(disc, COMMITS):
        val = u_agent[:, idx[f"{c}_commitment"]]                    # +1 home, -1 rival
        w[:, k] = np.where(val > 0, W_HI, W_LO)
    return scn, edges_ij, jnp.asarray(w)


def structure(P, edges_ij):
    """The 6 learned contested couplings Pi[anomaly, commitment] of a (mean) net."""
    return np.array([P[i, j] for (i, j) in edges_ij])


def by_commitment(s):
    """|coupling| summed over the two anomalies, per rival commitment -> (3,)."""
    return np.array([abs(s[0]) + abs(s[1]), abs(s[2]) + abs(s[3]), abs(s[4]) + abs(s[5])])


def reldist(sA, sB):
    return float(np.linalg.norm(sA - sB) / (0.5 * (np.linalg.norm(sA) + np.linalg.norm(sB)) + 1e-9))


def run_condition(scn, w, edges_ij, graph, seed):
    spec = AgentSpec(w_obs=w, lam=jnp.full((N,), 0.2))
    r = run_simulation(scn, graph, spec, forgetting=OMEGA, snapshot_every=5, seed=seed)
    P = r["snap_Pi"][-1]                                            # (N, d, d) final
    sA = structure(P[: N // 2].mean(0), edges_ij)
    sB = structure(P[N // 2:].mean(0), edges_ij)
    return sA, sB, float(np.abs(r["snap_Pi"]).max())


def main() -> int:
    out_dir = ROOT / "results" / "structural_pluralism"
    out_dir.mkdir(parents=True, exist_ok=True)
    scn, edges_ij, w = build()

    conditions = {
        "disconnected": graphs.community([N // 2, N // 2], intra=1.0, inter=0.0, seed=0),
        "connected": graphs.complete(N),
    }
    agg = {}
    print(f"structural pluralism: A values {VALUE_A}, B values {VALUE_B}; "
          f"underdetermined world; omega={OMEGA}; seeds={SEEDS}")
    for cond, graph in conditions.items():
        dists, cAs, cBs, mxs = [], [], [], []
        for s in SEEDS:
            sA, sB, mx = run_condition(scn, w, edges_ij, graph, s)
            dists.append(reldist(sA, sB)); cAs.append(by_commitment(sA))
            cBs.append(by_commitment(sB)); mxs.append(mx)
        agg[cond] = {
            "dist_mean": float(np.mean(dists)), "dist_std": float(np.std(dists)),
            "cA": np.mean(cAs, axis=0), "cB": np.mean(cBs, axis=0),
            "max_Pi": float(np.max(mxs)),
        }
        a = agg[cond]
        print(f"  {cond:>12}: struct-dist = {a['dist_mean']:.3f} +/- {a['dist_std']:.3f} | "
              f"A favors {COMMITS[a['cA'].argmax()]}, B favors {COMMITS[a['cB'].argmax()]} | "
              f"max|Pi|={a['max_Pi']:.1f}")

    dc, cc = agg["disconnected"], agg["connected"]
    # the claim: disconnected communities wire their OWN value's theory and stay apart;
    # connecting them collapses the structural difference (precision-pooling merges the wirings).
    assert dc["cA"].argmax() == COMMITS.index(VALUE_A), "A should wire its valued theory when disconnected"
    assert dc["cB"].argmax() == COMMITS.index(VALUE_B), "B should wire its valued theory when disconnected"
    assert dc["dist_mean"] > 3.0 * cc["dist_mean"] + 1e-6, \
        f"connection should collapse the divergence ({dc['dist_mean']:.2f} -> {cc['dist_mean']:.2f})"
    assert dc["max_Pi"] < 500 and cc["max_Pi"] < 500, "forgetting should bound the precision"

    # ---- figure: the two wirings (disconnected vs connected) + the collapse ----
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(3); width = 0.38
    for ax, cond, title in ((a0, "disconnected", "Disconnected: different wirings"),
                            (a1, "connected", "Connected: fusion merges them")):
        a = agg[cond]
        ax.bar(x - width / 2, a["cA"], width, color="#2c6fbb", label=f"A (values {VALUE_A})")
        ax.bar(x + width / 2, a["cB"], width, color="#c0392b", label=f"B (values {VALUE_B})")
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
        commits=np.array(COMMITS),
        disc_cA=dc["cA"], disc_cB=dc["cB"], conn_cA=cc["cA"], conn_cB=cc["cB"],
        disc_dist=dc["dist_mean"], conn_dist=cc["dist_mean"],
        disc_dist_std=dc["dist_std"], conn_dist_std=cc["dist_std"],
    )
    summary = {
        "config": {"N": N, "seeds": list(SEEDS), "omega": OMEGA, "n_steps": N_STEPS,
                   "value_A": VALUE_A, "value_B": VALUE_B, "world": "underdetermined blend"},
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
