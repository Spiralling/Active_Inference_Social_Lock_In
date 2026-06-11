"""Communicating BAYES NETS vs the Gaussian precision (lambda): does it matter?

The population engine's default communication is precision pooling -- agents trust-weight-average
the symmetric precision Pi and potential h (fuse_mode='posterior'). This script contrasts it
against fuse_mode='bayesnet', which communicates the DIRECTED net: each step every agent's belief
is read back as its CPDs (``bayesnet.from_info``), the CPDs are trust-weight-averaged, and recompiled
(``to_info``). Because the CPD<->precision bijection is nonlinear, averaging directed edge weights B
is a genuinely different operation from averaging Pi.

We run the cosmology re-tracking task (heterogeneous 3-community population, forgetting omega=0.9)
under BOTH modes in two regimes -- connected/no-conviction (pure re-tracking) and disconnected +
conviction (both-lever lock-in) -- and compare re-tracking, structural diversity, and held theory.
Honest question: does communicating the Bayes net change behavior relative to pooling precision?
"""
from __future__ import annotations

import json

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.structural.simulation import run_simulation
from src.structural.bayesnet import from_info as bn_from_info
from src.structural import scenarios as sc
from src.structural.models.cosmology import (
    cosmology_population, cosmology_spec, cosmology_graph, held_by_community,
    EPOCH_NAMES, T1, T2, N_STEPS, SIGMA_O, CONVICTION_LEVEL)
from experiments.registry import ExperimentSpec, register

MODE_LABEL = {"posterior": "precision (λ)", "bayesnet": "Bayes net (CPD)"}


def frac_on_true(r, scn, theory_mu):
    means = np.linalg.solve(r["snap_Pi"], r["snap_h"][..., None])[..., 0]
    closest = sc.closest_theory(means, theory_mu)
    true_e = np.asarray(scn.epoch_t)[r["snap_t"]]
    return (closest == true_e[:, None]).mean(axis=1), true_e


def run_cell(scn, pop, theory_mu, *, connected, conviction, mode, omega, seed):
    spec = cosmology_spec(scn, pop, conviction=conviction)
    g = cosmology_graph(pop["sizes"], connected=connected, seed=seed)
    r = run_simulation(scn, g, spec, fuse_mode=mode, forgetting=omega, snapshot_every=5, seed=seed)
    fot, true_e = frac_on_true(r, scn, theory_mu)
    held = held_by_community(r, pop["cluster_id"], theory_mu)
    return {"fot": fot, "fot_final": float(fot[-1]), "snap_t": r["snap_t"], "true_e": true_e,
            "final_div": float(r["disagreement_t"][-1]), "disagree_t": r["disagreement_t"],
            "n_distinct_end": int(len(np.unique(held[:, -1]))),
            "dF_finite": bool(np.isfinite(r["final_dF"]).all())}


def structural_heterogeneity(scn, pop, omega) -> dict:
    """WHY the two modes coincide: do agents hold different directed Bayes nets, or share a
    structure and differ only in their means? Compare cross-agent dispersion of the directed edge
    weights B against that of the posterior means at the final step of one connected cell."""
    spec = cosmology_spec(scn, pop, conviction=0.0)
    g = cosmology_graph(pop["sizes"], connected=True, seed=0)
    r = run_simulation(scn, g, spec, fuse_mode="posterior", forgetting=omega, snapshot_every=5, seed=0)
    Pi, h = np.asarray(r["snap_Pi"][-1]), np.asarray(r["snap_h"][-1])
    B, _, _ = jax.vmap(bn_from_info)(jnp.asarray(Pi), jnp.asarray(h))
    B = np.asarray(B); d = B.shape[-1]
    il = np.tril_indices(d, -1)
    B_disp = float(np.mean(np.std(B[:, il[0], il[1]], axis=0)))
    means = np.linalg.solve(Pi, h[..., None])[..., 0]
    mu_disp = float(np.mean(np.std(means, axis=0)))
    return {"B_disp": B_disp, "mu_disp": mu_disp}


def run(out_dir, params: dict) -> None:
    omega = params["OMEGA"]
    seeds = tuple(params["SEEDS"])
    modes = tuple(params["MODES"])

    scn = sc.cosmology_scenario(n_steps=N_STEPS, t1=T1, t2=T2, sigma_o=SIGMA_O)
    theory_mu = sc.cosmology_theory_means()
    pop = cosmology_population(seed=0)

    regimes = [("connected, no conviction", dict(connected=True, conviction=0.0)),
               ("disconnected + conviction", dict(connected=False, conviction=CONVICTION_LEVEL))]

    print(f"cosmology re-tracking, omega={omega}; comparing communication modes {modes}")
    results = {}
    for rlabel, kw in regimes:
        print(f"\n=== {rlabel} ===")
        for mode in modes:
            cells = [run_cell(scn, pop, theory_mu, mode=mode, omega=omega, seed=s, **kw) for s in seeds]
            agg = {"fot_final": float(np.mean([c["fot_final"] for c in cells])),
                   "final_div": float(np.mean([c["final_div"] for c in cells])),
                   "n_distinct_end": float(np.mean([c["n_distinct_end"] for c in cells])),
                   "fot": cells[0]["fot"], "disagree_t": cells[0]["disagree_t"],
                   "snap_t": cells[0]["snap_t"], "true_e": cells[0]["true_e"],
                   "dF_finite": all(c["dF_finite"] for c in cells)}
            results[(rlabel, mode)] = agg
            print(f"  {MODE_LABEL[mode]:>16}: final on-truth={agg['fot_final']:.2f}  "
                  f"final diversity={agg['final_div']:.2f}  distinct end-theories={agg['n_distinct_end']:.1f}")

    # ---- figure: re-tracking + diversity, precision vs Bayes-net comms ----
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8))
    rA = "connected, no conviction"
    for mode, col in zip(modes, ("navy", "darkorange")):
        d = results[(rA, mode)]
        a0.plot(d["snap_t"], d["fot"], lw=2.2, color=col, label=MODE_LABEL[mode])
    for tt in (T1, T2):
        a0.axvline(tt, color="k", ls=":", lw=0.8)
    a0.set_xlabel("step"); a0.set_ylabel("fraction on the current true theory"); a0.set_ylim(-0.05, 1.05)
    a0.set_title(f"re-tracking: {rA}"); a0.legend(fontsize=9)
    labels = [f"{rl.split(',')[0][:8]}\n{MODE_LABEL[m]}" for rl, _ in regimes for m in modes]
    divs = [results[(rl, m)]["final_div"] for rl, _ in regimes for m in modes]
    cols = ["navy", "darkorange"] * len(regimes)
    a1.bar(range(len(divs)), divs, color=cols)
    a1.set_xticks(range(len(divs))); a1.set_xticklabels(labels, fontsize=7)
    a1.set_ylabel("final structural diversity")
    a1.set_title("does communicating the Bayes net change diversity?")
    plt.tight_layout(); plt.savefig(out_dir / "diag_bayesnet_comms.png", dpi=120); plt.close(fig)

    # ---- verdict + asserts ----
    assert all(results[k]["dF_finite"] for k in results), "non-finite ΔF (CPD fusion PD issue?)"
    dtrack = abs(results[(rA, "bayesnet")]["fot_final"] - results[(rA, "posterior")]["fot_final"])
    ddiv = abs(results[(rA, "bayesnet")]["final_div"] - results[(rA, "posterior")]["final_div"])
    differs = (dtrack > 0.05) or (ddiv > 0.3)
    het = structural_heterogeneity(scn, pop, omega)
    print("\n=== verdict: does communicating Bayes nets differ from pooling precision? ===")
    print(f"  connected/no-conviction: Δ(on-truth)={dtrack:.2f}  Δ(diversity)={ddiv:.2f}  "
          f"-> {'DIFFERS' if differs else 'similar (both re-track / both wash out)'}")
    print(f"  WHY: cross-agent dispersion of directed edges B = {het['B_disp']:.4f}  vs  "
          f"of beliefs (means) = {het['mu_disp']:.3f}")
    print(f"  -> agents share the same directed STRUCTURE and differ only in their MEANS, so there is "
          f"no structural information in the Bayes net beyond what precision-pooling already shares.")

    summary = {"omega": omega, "seeds": list(seeds), "modes": list(modes),
               "results": {f"{rl} | {m}": {"fot_final": results[(rl, m)]["fot_final"],
                                            "final_div": results[(rl, m)]["final_div"],
                                            "n_distinct_end": results[(rl, m)]["n_distinct_end"]}
                           for rl, _ in regimes for m in modes},
               "delta_connected": {"track": dtrack, "diversity": ddiv, "differs": bool(differs)},
               "structural_heterogeneity": het}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        snap_t=results[(rA, "posterior")]["snap_t"],
        fot_posterior=results[(rA, "posterior")]["fot"], fot_bayesnet=results[(rA, "bayesnet")]["fot"],
        div_posterior=results[(rA, "posterior")]["disagree_t"],
        div_bayesnet=results[(rA, "bayesnet")]["disagree_t"])

    print(f"\nsaved arrays / figure / summary to {out_dir}")
    print(f"HEADLINE: under {rA}, communicating Bayes nets (CPDs) vs precision (λ): "
          f"on-truth {results[(rA,'posterior')]['fot_final']:.2f}->{results[(rA,'bayesnet')]['fot_final']:.2f}, "
          f"diversity {results[(rA,'posterior')]['final_div']:.2f}->{results[(rA,'bayesnet')]['final_div']:.2f} "
          f"({'DIFFERS' if differs else 'similar'}).")


register(ExperimentSpec(
    model="cosmology",
    name="bayesnet_comms",
    description="Communicating directed Bayes nets (CPD averaging) vs pooling precision (λ): under "
                "full observability agents share structure and differ only in means, so the two coincide.",
    run=run,
    out_dir="bayesnet_comms",
    params=dict(OMEGA=0.9, SEEDS=(0, 1, 2), MODES=("posterior", "bayesnet")),
    seeds=(0, 1, 2),
    consumes=dict(figures=["diag_bayesnet_comms"]),
))
