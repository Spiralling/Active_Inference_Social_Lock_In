"""E3-E5 -- three near-free recoveries of prior-work claims on the cosmology re-tracking task.

All three reuse the existing engine + population builders (no new mechanism), run WITH forgetting
(omega = 0.9, so the population actually re-tracks the moving world -- otherwise the no-forgetting
wall masks everything), and are delivered as ONE three-panel figure for the paper.

  E3  CONNECTIVITY (Zollman). Sweep the cross-community bridge density -> algebraic connectivity
      lambda_2. Denser graphs reach consensus faster but shed the transient structural diversity
      that a sparser community preserves -- the Zollman speed-vs-diversity trade-off.
  E4  CONSERVATISM. Sweep a uniform prior stiffness (precision_scale). A stiffer prior re-tracks
      LATER -- "centrality/stiffness sets how slowly the core moves."
  E5  SHARING CONCLUSIONS vs RAW EVIDENCE. fuse_mode = posterior (share whole belief nets =
      conclusions) vs deposit_pool / deposit_keep (share raw Fisher = evidence). Sharing conclusions
      herds the population (lower diversity) -- "sharing conclusions propagates bias along with signal."

Run it::

    python scripts/run_cosmology_sweeps.py
"""

from __future__ import annotations

import json
import sys
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
from src.structural import scenarios as sc
from scripts.run_cosmology_tracking import (
    cosmology_population, cosmology_spec, cosmology_graph, INTRA, T1, T2, N_STEPS, SIGMA_O)

OMEGA = 0.9
SEEDS = (0, 1)


def frac_on_true(r, scn, theory_mu):
    means = np.linalg.solve(r["snap_Pi"], r["snap_h"][..., None])[..., 0]
    closest = sc.closest_theory(means, theory_mu)
    true_e = np.asarray(scn.epoch_t)[r["snap_t"]]
    return (closest == true_e[:, None]).mean(axis=1), true_e


def t_half_final(fot, snap_t, true_e):
    """Steps after T2 to re-track the final epoch (frac-on-true >= 0.5); censored at horizon."""
    mask = true_e == true_e[-1]
    idx = np.where(mask)[0]
    seg = fot[idx]
    hit = seg >= 0.5
    return float(snap_t[idx[np.argmax(hit)]] - T2) if hit.any() else float(N_STEPS - T2)


def main() -> int:
    out_dir = ROOT / "results" / "cosmology_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    scn = sc.cosmology_scenario(n_steps=N_STEPS, t1=T1, t2=T2, sigma_o=SIGMA_O)
    theory_mu = sc.cosmology_theory_means()
    pop = cosmology_population(seed=0)

    def run(spec, graph, *, fuse_mode="posterior", seed=0):
        return run_simulation(scn, graph, spec, fuse_mode=fuse_mode, forgetting=OMEGA,
                              snapshot_every=5, seed=seed)

    # ---- E3: connectivity (Zollman) ----
    print("=== E3: connectivity lambda_2 sweep (Zollman speed vs diversity) ===")
    inters = (0.0, 0.01, 0.03, 0.08, 0.2)
    e3 = []
    for it in inters:
        l2, fr, div, th = [], [], [], []
        for s in SEEDS:
            g = graphs.community(pop["sizes"], intra=INTRA, inter=float(it), seed=s)
            r = run(cosmology_spec(scn, pop, conviction=0.0), g, seed=s)
            fot, true_e = frac_on_true(r, scn, theory_mu)
            l2.append(r["lambda2"]); fr.append(float(fot[-1]))
            div.append(float(r["disagreement_t"][-1])); th.append(t_half_final(fot, r["snap_t"], true_e))
        e3.append({"inter": it, "lambda2": float(np.mean(l2)), "final_track": float(np.mean(fr)),
                   "final_div": float(np.mean(div)), "t_half": float(np.mean(th))})
        print(f"  inter={it:.3f}: lambda2={e3[-1]['lambda2']:6.2f}  retrack_delay={e3[-1]['t_half']:5.1f}"
              f"  final_diversity={e3[-1]['final_div']:.2f}")

    # ---- E4: conservatism (prior stiffness) ----
    print("\n=== E4: conservatism (prior stiffness) sweep ===")
    precs = (0.5, 1.0, 2.0, 4.0, 8.0)
    g_fixed = cosmology_graph(pop["sizes"], connected=True, seed=0)
    e4 = []
    for ps in precs:
        th = []
        for s in SEEDS:
            spec = AgentSpec(w_obs=jnp.ones((len(pop["cluster_id"]), scn.m)),
                             lam=jnp.ones(len(pop["cluster_id"])),
                             precision_scale=jnp.full((len(pop["cluster_id"]),), float(ps)))
            r = run(spec, g_fixed, seed=s)
            fot, true_e = frac_on_true(r, scn, theory_mu)
            th.append(t_half_final(fot, r["snap_t"], true_e))
        e4.append({"prec": ps, "t_half": float(np.mean(th))})
        print(f"  precision_scale={ps:.1f}: retrack_delay={e4[-1]['t_half']:5.1f} steps")

    # ---- E5: sharing conclusions vs raw evidence ----
    print("\n=== E5: fuse_mode (share conclusions vs raw evidence) ===")
    modes = ("posterior", "deposit_pool", "deposit_keep")
    e5 = []
    for mode in modes:
        fr, div = [], []
        for s in SEEDS:
            r = run(cosmology_spec(scn, pop, conviction=0.0), g_fixed, fuse_mode=mode, seed=s)
            fot, _ = frac_on_true(r, scn, theory_mu)
            fr.append(float(fot[-1])); div.append(float(r["disagreement_t"][-1]))
        e5.append({"mode": mode, "final_track": float(np.mean(fr)), "final_div": float(np.mean(div))})
        print(f"  {mode:>13}: final_track={e5[-1]['final_track']:.2f}  final_diversity={e5[-1]['final_div']:.2f}")

    # ---- figure (one 3-panel) ----
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4.4))
    l2 = [e["lambda2"] for e in e3]
    a0.plot(l2, [e["t_half"] for e in e3], "o-", color="seagreen", lw=2, label="re-track delay")
    a0.set_xlabel(r"connectivity $\lambda_2$"); a0.set_ylabel("re-track delay (steps)", color="seagreen")
    a0.tick_params(axis="y", labelcolor="seagreen")
    a0b = a0.twinx()
    a0b.plot(l2, [e["final_div"] for e in e3], "s--", color="indianred", lw=2)
    a0b.set_ylabel("final structural diversity", color="indianred"); a0b.tick_params(axis="y", labelcolor="indianred")
    a0.set_title("E3: Zollman — denser = faster but less diverse")
    a1.plot([e["prec"] for e in e4], [e["t_half"] for e in e4], "o-", color="purple", lw=2)
    a1.set_xlabel("prior stiffness (precision scale)"); a1.set_ylabel("re-track delay (steps)")
    a1.set_title("E4: a stiffer prior re-tracks later")
    x = np.arange(len(modes))
    a2.bar(x - 0.2, [e["final_track"] for e in e5], 0.4, color="steelblue", label="final on-truth")
    a2.bar(x + 0.2, [e["final_div"] / max(e["final_div"] for e in e5) for e in e5], 0.4,
           color="0.6", label="final diversity (norm.)")
    a2.set_xticks(x); a2.set_xticklabels(["posterior\n(conclusions)", "deposit_pool\n(evidence)",
                                          "deposit_keep\n(evidence)"], fontsize=7)
    a2.set_title("E5: sharing conclusions herds more"); a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "diag_sweeps.png", dpi=120); plt.close(fig)

    # ---- gentle assertions (report-honestly trends) ----
    assert e4[-1]["t_half"] >= e4[0]["t_half"] - 1e-6, \
        f"stiffer prior should not re-track FASTER ({e4[-1]['t_half']} vs {e4[0]['t_half']})"
    # connected (high lambda2) should not keep MORE diversity than disconnected
    assert e3[-1]["final_div"] <= e3[0]["final_div"] + 1e-6, \
        f"denser graph should not preserve more diversity ({e3[-1]['final_div']} vs {e3[0]['final_div']})"

    summary = {"omega": OMEGA, "seeds": list(SEEDS),
               "E3_connectivity": e3, "E4_conservatism": e4, "E5_fuse_mode": e5}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        e3_lambda2=np.array(l2), e3_delay=np.array([e["t_half"] for e in e3]),
        e3_div=np.array([e["final_div"] for e in e3]),
        e4_prec=np.array([e["prec"] for e in e4]), e4_delay=np.array([e["t_half"] for e in e4]),
        e5_modes=np.array(modes), e5_track=np.array([e["final_track"] for e in e5]),
        e5_div=np.array([e["final_div"] for e in e5]))

    print(f"\nsaved arrays / figure / summary to {out_dir}")
    print(f"HEADLINE: lambda_2 up -> diversity {e3[0]['final_div']:.1f}->{e3[-1]['final_div']:.1f} "
          f"(Zollman); stiffer prior delay {e4[0]['t_half']:.0f}->{e4[-1]['t_half']:.0f} steps; "
          f"posterior-sharing diversity {e5[0]['final_div']:.1f} vs deposit {e5[2]['final_div']:.1f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
