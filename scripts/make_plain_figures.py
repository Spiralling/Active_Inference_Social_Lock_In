"""Plain-view figures: every headline result as 'how many scientists hold which theory,
over time'.

The per-experiment figures plot dose-response curves (mechanism dial -> outcome), which is
what the claims need but presumes the narrative. This script renders the same runs in the
most literal possible coordinates -- x = time (steps), y = number of scientists (out of the
community) -- one line per condition:

  A. the revolution itself, with everything on (combined mechanisms): believers per theory;
  B. lone discoveries die without social excitement: who HOLDS the oxygen concept;
  C. the pooling rule decides survival: same lines under the old vs new fusion;
  D. values that accrete entrench: believers in the dogmatic community, static vs accreting.

Writes results/plain_view/fig_plain_view.png + summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.structural.models.kuhn_phlogiston import single_run
from src.structural.phlogiston import DISAGREEMENT_NODES

OUT = ROOT / "results" / "plain_view"
EFF = 0.02      # effective-coupling threshold: "holds the oxygen concept in their model"


def believers(r) -> np.ndarray:
    """(S, N) bool: does each agent's belief sit on the oxygen side at each snapshot?
    Per-agent oxygen index from the posterior means on the disagreement (mass-law) nodes:
    -1 = the phlogiston reading, +1 = the oxygen reading; believer iff index > 0.5."""
    names = list(r["names"])
    ix = [names.index(n) for n in DISAGREEMENT_NODES]
    S = r["snap_Pi"].shape[0]
    out = np.zeros((S, r["snap_Pi"].shape[1]), dtype=bool)
    for s in range(S):
        mu = np.linalg.solve(r["snap_Pi"][s], r["snap_h"][s][..., None])[..., 0]
        idx = np.clip((mu[:, ix].mean(axis=1) + 1.0) / 2.0, 0.0, 1.0)
        out[s] = idx > 0.5
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    info = {}

    fig, axs = plt.subplots(2, 2, figsize=(12.6, 8.6))
    (aA, aB), (aC, aD) = axs

    # ---- A: the revolution, everything on -------------------------------------------
    rA = single_run(N=80, inter=0.01, t_shift=40, n_steps=320, proposal_rate=0.02,
                    rate_mode="hawkes_adaptive", hawkes_beta=0.5, rate_kappa=0.2,
                    rate_floor_ref=0.3, rate_max=0.8, wake_credit=True, prune_grace=15,
                    prune_patience=5, fuse_mode="posterior_masked", conviction_eps=0.3,
                    seed=0, snapshot_every=4)
    bel = believers(rA)
    comm = rA["community"]
    t = rA["snap_t"]
    for c, label, color in ((0, "open community (40 scientists)", "#1e8449"),
                            (1, "dogmatic community (40, gated + weakly coupled)",
                             "#922b21")):
        aA.plot(t, bel[:, comm == c].sum(axis=1), lw=2.4, color=color, label=label)
    aA.axvline(40, color="k", lw=0.9, ls="--")
    aA.text(41, 37, "the world changes\n(calx IS heavier)", fontsize=8)
    aA.set_title("A. The scientific revolution, with every new mechanism on:\n"
                 "how many scientists believe the OXYGEN theory", fontsize=10)
    aA.set_ylabel("scientists believing oxygen (of 40)")
    aA.legend(fontsize=8, loc="center right")
    info["A_final_open"] = int(bel[-1, comm == 0].sum())
    info["A_final_dogma"] = int(bel[-1, comm == 1].sum())

    # ---- B: lone discoveries die without social excitement ---------------------------
    for beta, label, color in ((0.0, "no social excitement ($\\beta$=0): "
                                "discoveries stay lonely", "#922b21"),
                               (1.0, "discoveries excite trusted colleagues "
                                "($\\beta$=1)", "#1e8449")):
        rB = single_run(N=80, inter=0.0, t_shift=40, n_steps=320, proposal_rate=0.01,
                        rate_mode=("hawkes" if beta > 0 else "poisson"),
                        hawkes_beta=beta, seed=1, snapshot_every=4)
        o = rB["community"] == 0
        holders = (rB["oxy_coupling_tn"][:, o] > EFF).sum(axis=1)
        aB.plot(np.arange(len(holders)), holders, lw=2.4, color=color, label=label)
        info[f"B_final_beta{beta}"] = int(holders[-1])
    aB.axvline(40, color="k", lw=0.9, ls="--")
    aB.set_title("B. A new CONCEPT needs the community to get excited:\n"
                 "how many scientists carry the oxygen concept in their model", fontsize=10)
    aB.set_ylabel("scientists holding the concept (of 40)")
    aB.legend(fontsize=8)

    # ---- C: the pooling rule decides survival ----------------------------------------
    for mode, label, color in (("posterior", 'old pooling: peers\' "no such thing" '
                                "crushes the concept", "#922b21"),
                               ("posterior_masked", "new pooling: peers who never "
                                "conceived it ABSTAIN", "#1e8449")):
        rC = single_run(N=80, inter=0.0, t_shift=40, n_steps=320, proposal_rate=0.01,
                        fuse_mode=mode, seed=1, snapshot_every=4)
        o = rC["community"] == 0
        holders = (rC["oxy_coupling_tn"][:, o] > EFF).sum(axis=1)
        aC.plot(np.arange(len(holders)), holders, lw=2.4, color=color, label=label)
        info[f"C_final_{mode}"] = int(holders[-1])
    aC.axvline(40, color="k", lw=0.9, ls="--")
    aC.set_title("C. ...or the community to change how it pools opinions:\n"
                 "same lonely discoveries, two pooling rules", fontsize=10)
    aC.set_xlabel("time (steps)")
    aC.set_ylabel("scientists holding the concept (of 40)")
    aC.legend(fontsize=8)

    # ---- D: values that accrete entrench (visible as a conversion DELAY) --------------
    halves = {}
    for eps, label, color in ((0.0, "values fixed (the paper's model)", "#1e8449"),
                              (1.0, "values accrete onto entrenched beliefs "
                               "($\\epsilon$=1, Lakatos)", "#922b21")):
        rD = single_run(N=80, inter=0.002, t_shift=40, n_steps=320, proposal_rate=0.08,
                        conviction_eps=eps, conviction_decay=0.01,
                        seed=0, snapshot_every=4)
        belD = believers(rD)
        dg = rD["community"] == 1
        nD = belD[:, dg].sum(axis=1)
        aD.plot(rD["snap_t"], nD, lw=2.4, color=color, label=label)
        half = int(rD["snap_t"][np.argmax(nD >= 20)]) if (nD >= 20).any() else -1
        halves[eps] = half
        if half > 0:
            aD.axvline(half, color=color, lw=1.0, ls=":")
            aD.text(half + 2, 21, f"half converted\nt={half}", fontsize=7.5, color=color)
        info[f"D_final_eps{eps}"] = int(nD[-1])
        info[f"D_half_eps{eps}"] = half
    aD.axvline(40, color="k", lw=0.9, ls="--")
    aD.set_title("D. A community whose values grow onto its own programme converts "
                 f"~{halves.get(1.0, 0) / max(halves.get(0.0, 1), 1):.1f}x later\n"
                 "(connection still wins eventually): oxygen believers in the DOGMATIC "
                 "community", fontsize=10)
    aD.set_xlabel("time (steps)")
    aD.set_ylabel("scientists believing oxygen (of 40)")
    aD.legend(fontsize=8)

    for ax in axs.flat:
        ax.set_ylim(-1.5, 43)
    fig.suptitle("The new mechanisms in plain coordinates: scientists per theory, over time",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT / "fig_plain_view.png"
    plt.savefig(path, dpi=130)
    plt.close(fig)
    (OUT / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
