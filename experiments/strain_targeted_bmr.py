"""Strain-targeted BMR: the conflict read-off finds the misspecified edge.

The repo's BMR machinery (``bmr.prune_edge_prior`` + ``action.reduction_score``)
can price the removal of ANY prior edge -- but nothing localizes WHERE the model
is misspecified, so an agent must scan every edge. The per-edge strain
(``observables.edge_strain``: the node-splitting conflict z^2 between the full
posterior's endpoint marginals and the marginals under the SAME deposit with
that one prior edge zeroed -- exactly the edit BMR would price) is a cheap O(1)
read-off that RANKS the edges by how hard they fight the data, so the expensive
Savage-Dickey verdict only needs to be evaluated at the top.

Setup: one agent, a 3-node chain A--B--C, both prior edges expressing positive
coupling. The world agrees with A--B (phi_A = phi_B) and VIOLATES B--C (phi_C
strongly opposite to phi_B). Node-wise H => the deposit's Fisher is diagonal,
so the edge precisions never move from data alone; only BMR can cut the bad
edge. What the run must show: strain ranks (B,C) above (A,B) essentially
always, the final Savage-Dickey verdict accepts pruning (B,C) (Delta F > 0)
while holding (A,B) (Delta F < 0), and strain identifies the target no later
than the Delta F sign flip.

Calibration note (verified before freezing): the zero-edge strain reads the
calibrated displacement of the endpoint marginals, which scales with the
coupled potential magnitude -- a conflict of the same magnitude as the
agreement scores symmetrically. Discrimination therefore needs the violation
to be strong (PHI's C entry at -3.0, the plan's documented |phi_C - phi_B|
lever); with it, strain(B,C) > strain(A,B) at every step and both Delta F
signs are correct from step 1. Noise-free evidence => deterministic assertions.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from experiments.registry import ExperimentSpec, register
from src.structural import action, linalg
from src.structural import observables as obs
from src.structural.belief import GaussianBeliefNet

NAMES = ("A", "B", "C")
EDGES = (("A", "B"), ("B", "C"))
EDGE_IDX = ((0, 1), (1, 2))


def _chain_prior(params: dict) -> GaussianBeliefNet:
    Pi0 = params["DIAG"] * jnp.eye(3)
    c = params["COUPLING"]
    Pi0 = Pi0.at[0, 1].set(c).at[1, 0].set(c).at[1, 2].set(c).at[2, 1].set(c)
    return GaussianBeliefNet(Pi=Pi0, h=jnp.zeros(3), names=NAMES)


def _rollout(params: dict) -> dict:
    prior = _chain_prior(params)
    H = jnp.eye(3)
    phi = jnp.asarray(params["PHI"])
    J, j = linalg.fisher_deposit(H, H @ phi, params["SIGMA_O"])  # noise-free o

    strain = np.zeros((params["T"], 2))
    dF = np.zeros((params["T"], 2))
    for t in range(params["T"]):
        Pi_p = prior.Pi + (t + 1) * J
        h_p = prior.h + (t + 1) * j
        post = GaussianBeliefNet(Pi=Pi_p, h=h_p, names=NAMES)
        for e, (u, v) in enumerate(EDGE_IDX):
            strain[t, e] = float(obs.edge_strain(Pi_p, h_p, prior.Pi, prior.h, u, v))
            dF[t, e] = action.reduction_score(post, prior, EDGES[e]).delta_F
    final_scores = [action.reduction_score(
        GaussianBeliefNet(Pi=prior.Pi + params["T"] * J,
                          h=prior.h + params["T"] * j, names=NAMES),
        prior, e) for e in EDGES]
    return {"strain": strain, "dF": dF, "final": final_scores}


def _first_true(mask: np.ndarray, horizon: int) -> int:
    """First index at which ``mask`` holds; ``horizon`` if it never does."""
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else horizon


def _figures(tr: dict, params: dict, out_dir) -> None:
    T = params["T"]
    ts = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(ts, tr["strain"][:, 1], lw=2.0, color="#922b21",
            label="strain(B,C) -- the misspecified edge")
    ax.plot(ts, tr["strain"][:, 0], lw=2.0, color="#1e8449",
            label="strain(A,B) -- the true edge")
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("edge strain (z^2, log)")
    ax.set_title("strain ranks the misspecified edge first at every step")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "fig_edge_strain.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(ts, tr["dF"][:, 1], lw=2.0, color="#922b21",
            label="Delta F prune (B,C): > 0 => cut the edge")
    ax.plot(ts, tr["dF"][:, 0], lw=2.0, color="#1e8449",
            label="Delta F prune (A,B): < 0 => the data hold it")
    ax.axhline(0.0, color="k", lw=0.8)
    t_acc = _first_true(tr["dF"][:, 1] > 0, T)
    if t_acc < T:
        ax.plot([t_acc + 1], [tr["dF"][t_acc, 1]], "o", ms=8, color="#922b21",
                label=f"BMR accepts the cut (step {t_acc + 1})")
    ax.set_xlabel("step"); ax.set_ylabel("Savage-Dickey Delta F")
    ax.set_title("the Savage-Dickey verdict confirms what strain targeted")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "fig_deltaF.png", dpi=130)
    plt.close(fig)


def run(out_dir, params: dict) -> None:
    T, burn = params["T"], params["BURN_IN"]
    tr = _rollout(params)
    strain, dF = tr["strain"], tr["dF"]

    # (a) strain ranks the misspecified edge first after burn-in.
    frac = float((strain[burn:, 1] > strain[burn:, 0]).mean())
    assert frac > 0.8, f"strain must rank (B,C) first in >80% of steps (got {frac:.0%})"

    # (b) the final Savage-Dickey verdict agrees with the strain ranking.
    sc_ab, sc_bc = tr["final"]
    assert strain[-1, 1] > strain[-1, 0], "final top-strain edge must be (B,C)"
    assert sc_bc.accept and sc_bc.delta_F > 0, \
        f"BMR must accept cutting (B,C) (Delta F={sc_bc.delta_F:.3f})"
    assert sc_ab.delta_F < 0, \
        f"BMR must hold the true edge (A,B) (Delta F={sc_ab.delta_F:.3f})"

    # (c) strain identifies the target no later than the Delta F sign flip.
    t_strain = _first_true(strain[:, 1] > strain[:, 0], T)
    t_dF = _first_true(dF[:, 1] > 0, T)
    assert t_dF < T, "Delta F(B,C) must flip positive within the horizon"
    assert t_strain <= t_dF, \
        f"strain must lead the Delta F flip (t_strain={t_strain}, t_dF={t_dF})"

    _figures(tr, params, out_dir)
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        strain_AB=strain[:, 0], strain_BC=strain[:, 1],
        dF_AB=dF[:, 0], dF_BC=dF[:, 1],
    )
    summary = {
        "config": {k: v for k, v in params.items()},
        "frac_BC_top_strain": frac,
        "final_dF_BC": float(sc_bc.delta_F),
        "final_dF_AB": float(sc_ab.delta_F),
        "t_strain": t_strain, "t_dF": t_dF,
        "lead_time": t_dF - t_strain,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"HEADLINE: per-edge strain localizes the misspecification without "
          f"scanning: the violated edge (B,C) carries the top strain in "
          f"{frac:.0%} of post-burn-in steps (identified at step {t_strain + 1}, "
          f"{t_dF - t_strain} steps before the Savage-Dickey sign flip at step "
          f"{t_dF + 1}), and the BMR verdict it targets is the right one -- "
          f"Delta F(B,C)={sc_bc.delta_F:.2f} > 0 (cut) while "
          f"Delta F(A,B)={sc_ab.delta_F:.2f} < 0 (the true edge is held). Strain "
          f"tells BMR WHERE to look; Savage-Dickey says WHETHER to cut.")


register(ExperimentSpec(
    model="phlogiston",
    name="strain_targeted_bmr",
    description="Per-edge strain (node-splitting conflict z^2, the zero-edge "
                "read-off) ranks a planted misspecified prior edge first at "
                "every step and targets the Savage-Dickey pruning verdict to "
                "it -- BMR guided by WHERE the model hurts instead of an "
                "all-edges scan.",
    run=run,
    out_dir="strain_targeted_bmr",
    params=dict(
        T=80, BURN_IN=10, SIGMA_O=1.0, DIAG=1.5, COUPLING=-0.7,
        # phi_C at -3.0 (not -1.5): the documented |phi_C - phi_B| lever --
        # verified to keep Delta F(A,B) < 0 at every step while strain and
        # Delta F(B,C) both discriminate from step 1.
        PHI=(1.5, 1.5, -3.0),
    ),
    seeds=(),
    canonical=False,
    consumes=dict(figures=["fig_edge_strain", "fig_deltaF"]),
))
