"""The modeling lens made concrete on the dark-energy paradigm (nb45 / nb44 Part 0).

One directed Bayes net wires the 1998 dark-energy turn, and three faces of a single fact fall out as
linear algebra: (1) the directed SEM is primary and the symmetric coherence precision its shadow;
(2) one propagation operator gives two independent fields -- conservatism ``kappa = T 1`` (GR carries
everything) and conviction ``U = T^T u`` (Lambda is cheap-yet-cherished); (3) conditioning on the
shared supernova node manufactures the Omega_m/Lambda Schur degeneracy (the banana).

The DAG geometry + the ``draw_dag`` / ``_short`` helpers live in
``src.structural.models.cosmology_twofield``; this module owns the experiment's figures, the three
asserted claims, and the saved arrays. nb45 imports the builders directly and draws polished figures.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

from src.structural.models.cosmology_twofield import (
    NODES, build_dark_energy_dag, conservatism, conviction, utility_vector, schur_banana,
    draw_dag, _short)
from experiments.registry import ExperimentSpec, register


def fig_two_fields(net, path):
    kap = conservatism(net)
    U = conviction(net)
    names = list(net.names)
    order = np.argsort(-kap)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 5.2))
    y = np.arange(len(names))
    a0.barh(y, kap[order], color="steelblue")
    a0.set_yticks(y); a0.set_yticklabels([_short(names[i]) for i in order], fontsize=8)
    a0.invert_yaxis(); a0.set_xlabel("conservatism  κ = T·1  (carry-over cost)")
    a0.set_title("κ: GR carries everything; Λ disturbs only μ(z)")
    cols = ["seagreen" if v > 0 else ("crimson" if v < 0 else "0.6") for v in U[order]]
    a1.barh(y, U[order], color=cols)
    a1.set_yticks(y); a1.set_yticklabels([_short(names[i]) for i in order], fontsize=8)
    a1.invert_yaxis(); a1.axvline(0, color="k", lw=0.6)
    a1.set_xlabel("conviction  U = T·u  (propagated value)")
    a1.set_title("U: Λ is cheap yet cherished; GR central yet value-neutral")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def fig_decoupling(net, path):
    kap = conservatism(net); U = conviction(net); names = list(net.names)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(kap, U, s=40, color="0.5", zorder=2)
    for i, n in enumerate(names):
        if n in ("general_relativity", "dark_energy", "matter_density", "light_bending",
                 "friedmann_expansion", "supernova_obs", "mond_prediction"):
            ax.annotate(_short(n), (kap[i], U[i]), fontsize=9, fontweight="bold",
                        textcoords="offset points", xytext=(5, 4))
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("conservatism  κ = T·1")
    ax.set_ylabel("conviction  U = T·u")
    r = np.corrcoef(kap, U)[0, 1]
    ax.set_title(f"Two fields, one operator: κ and U are independent (r = {r:.2f})")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def _ellipse(ax, cov, mu, color, label):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, np.argmax(vals)], vecs[0, np.argmax(vals)]))
    from matplotlib.patches import Ellipse
    for k in (1.0, 2.0):
        w, h = 2 * k * np.sqrt(np.maximum(vals, 1e-12))
        ax.add_patch(Ellipse(mu, w, h, angle=ang, fill=(k == 1.0), alpha=0.18 if k == 1 else 1.0,
                             edgecolor=color, facecolor=color, lw=1.6))
    ax.plot([], [], color=color, lw=2, label=label)


def fig_banana(net, path):
    s = schur_banana(net)
    mu = (s["mu_a"], s["mu_b"])
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    _ellipse(ax, s["prior_cov"], mu, "0.55", f"prior (marginal): r = {s['prior_corr']:.2f}")
    _ellipse(ax, s["post_cov"], mu, "crimson", f"given SN Ia: r = {s['post_corr']:.2f}")
    ax.set_xlabel("Ω_m  (matter density)"); ax.set_ylabel("Λ  (dark energy)")
    ax.set_title("Schur degeneracy: clamping the supernova node\ncouples the two dials (the banana)")
    ax.legend(loc="best", fontsize=9); ax.set_aspect("equal", "datalim")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def fig_dag_fields(net, path):
    """The two fields on the actual network: κ (cost, OrRd) and U (value, RdYlGn)."""
    kap = conservatism(net); U = conviction(net)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(16, 6.2))
    draw_dag(net, kap, a0, cmap="OrRd",
             title="κ = T·1  (conservatism / carry-over cost): GR hot, leaves cold")
    draw_dag(net, U, a1, cmap="RdYlGn", diverging=True,
             title="U = T·u  (conviction): Λ & SN green, MOND red, machinery neutral")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def run(out_dir, params: dict) -> None:
    net = build_dark_energy_dag()
    names = list(net.names)
    kap = conservatism(net)
    U = conviction(net)
    s = schur_banana(net)

    i_gr, i_lam = names.index("general_relativity"), names.index("dark_energy")

    print("=== dark-energy two-field readout ===")
    print(f"{'node':>22} {'kappa':>8} {'U':>8}")
    for i in np.argsort(-kap):
        print(f"{names[i]:>22} {kap[i]:8.3f} {U[i]:8.3f}")

    print("\n--- the three claims ---")
    ratio = kap[i_gr] / kap[i_lam]
    print(f"1. kappa(GR)={kap[i_gr]:.2f}  kappa(Lambda)={kap[i_lam]:.2f}  ratio={ratio:.2f}"
          f"  (GR is the costliest commitment, Lambda among the cheapest dials)")
    print(f"2. U(Lambda)={U[i_lam]:.2f} > U(GR)={U[i_gr]:.2f}  while  kappa(Lambda) < kappa(GR)"
          f"  => the value order INVERTS the cost order: the fields are decoupled")
    print(f"   global corr(kappa, U) = {np.corrcoef(kap, U)[0,1]:.2f}")
    print(f"3. corr(Omega_m, Lambda): prior={s['prior_corr']:.3f} -> given SN={s['post_corr']:.3f}"
          f"  (conditioning manufactured the off-diagonal)")

    fig_two_fields(net, out_dir / "twofield_bars.png")
    fig_decoupling(net, out_dir / "twofield_decoupling.png")
    fig_banana(net, out_dir / "schur_banana.png")
    fig_dag_fields(net, out_dir / "twofield_dag.png")

    assert kap[i_gr] == kap.max(), "GR should be the costliest node (max kappa)"
    assert ratio > 3.0, f"kappa(GR) should dwarf kappa(Lambda); ratio {ratio:.2f}"
    assert U[i_lam] > U[i_gr], "Lambda should be more cherished than GR (decoupling)"
    assert kap[i_lam] < kap[i_gr], "Lambda should be cheaper than GR (decoupling)"
    assert abs(s["prior_corr"]) < 0.05, f"prior Omega_m/Lambda should be ~uncorrelated, got {s['prior_corr']:.3f}"
    assert abs(s["post_corr"]) > 0.2, f"conditioning on SN should couple the dials, got {s['post_corr']:.3f}"

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        names=np.array(names), kappa=kap, U=U, utility=utility_vector(),
        B=np.asarray(net.B), Pi=np.asarray(net.to_info().Pi),
        prior_cov=s["prior_cov"], post_cov=s["post_cov"],
    )
    summary = {
        "kappa": {n: float(kap[i]) for i, n in enumerate(names)},
        "U": {n: float(U[i]) for i, n in enumerate(names)},
        "kappa_GR_over_Lambda": float(ratio),
        "corr_kappa_U": float(np.corrcoef(kap, U)[0, 1]),
        "schur": {"prior_corr": s["prior_corr"], "post_corr": s["post_corr"]},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figures / summary to {out_dir}")
    print("ALL THREE CLAIMS HOLD.")


register(ExperimentSpec(
    model="cosmology",
    name="cosmology_twofield",
    description="The dark-energy paradigm as one directed Bayes net: conservatism κ=T·1 (GR carries "
                "everything) and conviction U=Tᵀu (Λ cheap-yet-cherished) are decoupled, and "
                "conditioning on the supernova node manufactures the Schur degeneracy (the banana).",
    run=run,
    out_dir="cosmology_twofield",
    params=dict(),
    consumes=dict(notebook="nb45_two_fields_darkenergy",
                  figures=["twofield_bars", "twofield_decoupling", "schur_banana", "twofield_dag"]),
))
