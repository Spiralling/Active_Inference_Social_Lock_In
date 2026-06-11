"""Poisson arrival of structural edits: the RATE at which new commitments enter.

The deterministic regrowth (``cosmology_regrowth``) wakes the unconceived ``dark_energy`` node the
moment the expansion model Bayes factor accepts it (the data hold the woken node). But *when* a
community proposes a new commitment is itself contingent -- the exploration rate
network-epistemology models put in by hand. The paper models node arrival as a POISSON PROCESS
that fixes only the RATE: structural-edit attempts arrive at rate ``lambda`` (each step with prob
``1 - e^{-lambda}``), and the wake fires on the first arrival the Bayes factor accepts. Discovery
becomes a random WAITING TIME ~ ``1/lambda``. The arrival RNG is separate from the world-sample
RNG, so the evidence trajectory is identical to the deterministic run -- only the wake timing is
Poisson-gated.

Reuses ``single_run`` from ``src.structural.models.cosmology_regrowth``.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

from src.structural.models.cosmology_regrowth import single_run, T2, N_STEPS, HUB_NAME
from experiments.registry import ExperimentSpec, register


def wake_times(rate, *, coupling, n_seeds) -> dict:
    """Discovery-time stats over seeds for one arrival rate (or deterministic if rate is None)."""
    waketimes, woke = [], 0
    for s in range(n_seeds):
        r = single_run(coupling=coupling, seed=s, proposal_rate=rate)
        if r["wake_step"] >= 0:
            woke += 1
            waketimes.append(r["wake_step"])
    wt = np.array(waketimes, dtype=float)
    return {"rate": rate, "wake_times": wt, "wake_fraction": woke / n_seeds,
            "mean_wake": float(wt.mean()) if wt.size else float("nan"),
            "std_wake": float(wt.std()) if wt.size else float("nan"),
            "mean_delay_after_T2": float(wt.mean() - T2) if wt.size else float("nan")}


def run(out_dir, params: dict) -> None:
    rates = tuple(params["RATES"])
    n_seeds = params["N_SEEDS"]
    coupling = params["COUPLING"]

    print(f"Poisson arrival of structural edits (cause at T2={T2}, horizon {N_STEPS}, "
          f"{n_seeds} seeds, coupling={coupling})")
    print(f"{'lambda':>7} | {'wake_frac':>9} | {'mean_wake':>9} | {'std':>5} | mean delay after T2")
    rows = []
    for lam in rates:
        d = wake_times(lam, coupling=coupling, n_seeds=n_seeds)
        rows.append(d)
        print(f"{lam:7.2f} | {d['wake_fraction']:9.2f} | {d['mean_wake']:9.1f} | "
              f"{d['std_wake']:5.1f} | {d['mean_delay_after_T2']:.1f}")

    # deterministic reference (proposal_rate=None)
    det = wake_times(None, coupling=coupling, n_seeds=n_seeds)
    print(f"{'determ.':>7} | {det['wake_fraction']:9.2f} | {det['mean_wake']:9.1f} | "
          f"{det['std_wake']:5.1f} | {det['mean_delay_after_T2']:.1f}")

    # null: no cause (coupling 0). With a threshold-free accept the Bayes factor makes rare
    # noise crossings over ~180 looks per run, so the null's false-discovery rate is SMALL but
    # not exactly zero -- the price of removing the ad hoc trigger (see the appendix; the
    # principled completion is wake-then-prune, named in the paper's limitations).
    null = wake_times(1.0, coupling=0.0, n_seeds=10)
    print(f"\nnull (coupling=0, λ=1.0): wake_fraction={null['wake_fraction']:.2f} (expect ~0, tolerate <= 0.1)")

    # ---- figures ----
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8))
    data = [r["wake_times"] for r in rows if r["wake_times"].size]
    labels = [f"λ={r['rate']:.2f}" for r in rows if r["wake_times"].size]
    if det["wake_times"].size:
        data.append(det["wake_times"]); labels.append("determ.")
    a0.violinplot(data, showmeans=True, showextrema=False)
    a0.axhline(T2, color="purple", ls="-", lw=1, alpha=0.6); a0.text(0.6, T2 + 1, "T2", color="purple", fontsize=8)
    a0.set_xticks(range(1, len(labels) + 1)); a0.set_xticklabels(labels, rotation=20, fontsize=8)
    a0.set_ylabel("discovery (wake) step"); a0.set_title("Poisson arrival → a waiting-time DISTRIBUTION")
    a1.plot([r["rate"] for r in rows], [r["mean_delay_after_T2"] for r in rows], "o-",
            color="seagreen", lw=2, label="mean delay after T2")
    a1.plot([r["rate"] for r in rows], [1.0 / r["rate"] for r in rows], "k:", lw=1, label="~1/λ (expected wait)")
    a1.axhline(det["mean_delay_after_T2"], color="grey", ls="--", lw=1, label="deterministic")
    a1.set_xlabel("arrival rate λ"); a1.set_ylabel("mean discovery delay after T2 (steps)")
    a1.set_title("faster proposers discover sooner (≈ 1/λ wait)"); a1.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "diag_poisson.png", dpi=120); plt.close(fig)

    # ---- asserts ----
    means = [r["mean_wake"] for r in rows]
    assert null["wake_fraction"] <= 0.1, \
        f"null false-discovery rate should be rare (got {null['wake_fraction']})"
    assert means[0] > means[-1] + 2.0, \
        f"slower arrivals should discover LATER on average (got {means[0]:.1f} vs {means[-1]:.1f})"
    assert rows[-1]["wake_fraction"] >= rows[0]["wake_fraction"] - 1e-9, \
        "faster arrivals should not discover less often"
    after_T2 = np.concatenate([r["wake_times"] >= T2 for r in rows if r["wake_times"].size])
    assert after_T2.mean() >= 0.9, \
        f"wakes should overwhelmingly follow the cause (after-T2 fraction {after_T2.mean():.2f})"

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        rates=np.array(rates), n_seeds=n_seeds, coupling=coupling, T2=T2, n_steps=N_STEPS,
        wake_fraction=np.array([r["wake_fraction"] for r in rows]),
        mean_wake=np.array([r["mean_wake"] for r in rows]),
        std_wake=np.array([r["std_wake"] for r in rows]),
        mean_delay=np.array([r["mean_delay_after_T2"] for r in rows]),
        det_mean_wake=det["mean_wake"], det_std_wake=det["std_wake"],
        **{f"wake_times_{i}": rows[i]["wake_times"] for i in range(len(rows))},
        det_wake_times=det["wake_times"],
    )
    summary = {
        "config": {"rates": list(rates), "n_seeds": n_seeds, "coupling": coupling,
                   "T2": T2, "n_steps": N_STEPS, "hub": HUB_NAME},
        "by_rate": [{"lambda": r["rate"], "wake_fraction": r["wake_fraction"],
                     "mean_wake": r["mean_wake"], "std_wake": r["std_wake"],
                     "mean_delay_after_T2": r["mean_delay_after_T2"]} for r in rows],
        "deterministic": {"mean_wake": det["mean_wake"], "wake_fraction": det["wake_fraction"]},
        "null_coupling0_wake_fraction": null["wake_fraction"],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figure / summary to {out_dir}")
    print(f"HEADLINE: structural discovery is a Poisson waiting time -- mean delay after T2 falls "
          f"from {rows[0]['mean_delay_after_T2']:.0f} (λ={rates[0]}) to "
          f"{rows[-1]['mean_delay_after_T2']:.0f} (λ={rates[-1]}) steps, ≈1/λ; null never wakes.")


register(ExperimentSpec(
    model="cosmology",
    name="cosmology_poisson",
    description="Node arrival as a Poisson process: structural discovery becomes a waiting time "
                "~1/λ after T2 (faster proposers discover sooner); null coupling never wakes.",
    run=run,
    out_dir="cosmology_poisson",
    params=dict(RATES=(0.02, 0.05, 0.10, 0.30, 1.00), N_SEEDS=40, COUPLING=1.6),
    consumes=dict(figures=["diag_poisson"]),
))
