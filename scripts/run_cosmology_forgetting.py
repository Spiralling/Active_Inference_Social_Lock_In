"""Forgetting (precision relaxation) breaks the no-forgetting wall -- and turns conviction into
*motivated persistence*.

nb44 found the wall: with pure accumulation (omega = 1) a connected, lever-free population locks
onto epoch 0 and never catches the moving truth -- early evidence is simply too heavy by the later
epochs. Here we add the paper's forgetting factor omega in (0, 1] (Sec. 3.3), now wired into
``simulation.run_simulation``: each step the accumulated evidence relaxes back toward the prior,

    Pi <- Pi_prior + omega (Pi - Pi_prior),   h <- h_prior + omega (h - h_prior),

so a deposit from k steps ago carries weight omega^k -- an exponential memory of ~1/(1-omega)
steps. Two consequences, both reported here:

  1. THE WALL BREAKS. With omega < 1 the lever-free population RE-TRACKS each successive theory:
     the modal belief follows the true-theory staircase instead of freezing on dark matter. Too
     much forgetting (omega small) trades tracking for noise -- a sweet spot, swept below.

  2. CONVICTION BECOMES MOTIVATED PERSISTENCE. The value tilt h <- h + lambda U is re-applied every
     step, so it accumulates to a steady level even as *evidence* forgets. A committed, disconnected
     bloc therefore stays on its home theory even though the evidence that once supported it has
     leaked away -- the paper's "a belief wanted-true after its reasons have leaked away", now a
     consequence of omega_pi < 1 <= omega_u rather than a posited bias.

Run it::

    python scripts/run_cosmology_forgetting.py

Reuses the cosmology world + population builders of ``run_cosmology_tracking``; sweeps omega, asserts
the wall-break and the persistence, writes arrays/figures/summary to ``results/cosmology_forgetting/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.structural.simulation import run_simulation
from src.structural import scenarios as sc
from scripts.run_cosmology_tracking import (
    cosmology_population, cosmology_spec, cosmology_graph, held_by_community,
    COMM_LABELS, HOMES, EPOCH_NAMES, T1, T2, N_STEPS, SIGMA_O, CONVICTION_LEVEL)

OMEGAS = (1.0, 0.97, 0.93, 0.88, 0.80, 0.70)
SNAP = 5


def _agent_means(r: dict) -> np.ndarray:
    """(S, N, d) posterior means (NumPy 2.x: vectors need a trailing axis)."""
    return np.linalg.solve(r["snap_Pi"], r["snap_h"][..., None])[..., 0]


def frac_on_true(r: dict, scn, theory_mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(frac_on_true (S,), true_epoch (S,)): fraction of the whole population whose mean belief
    sits closest to the CURRENT true theory at each snapshot."""
    means = _agent_means(r)                                   # (S, N, d)
    closest = sc.closest_theory(means, theory_mu)             # (S, N)
    true_e = np.asarray(scn.epoch_t)[r["snap_t"]]             # (S,)
    return (closest == true_e[:, None]).mean(axis=1), true_e


def run_omega(scn, pop, *, omega: float, conviction: float, connected: bool, seed: int = 0) -> dict:
    spec = cosmology_spec(scn, pop, conviction=conviction)
    g = cosmology_graph(pop["sizes"], connected=connected, seed=seed)
    r = run_simulation(scn, g, spec, forgetting=omega, snapshot_every=SNAP, seed=seed)
    theory_mu = sc.cosmology_theory_means()
    fot, true_e = frac_on_true(r, scn, theory_mu)
    held = held_by_community(r, pop["cluster_id"], theory_mu)
    return {"omega": omega, "snap_t": r["snap_t"], "frac_on_true": fot, "true_epoch": true_e,
            "held": held, "disagreement": r["disagreement_t"],
            # steady-state precision magnitude (the "is it still confident?" axis of the tradeoff)
            "mean_prec": float(np.trace(r["snap_Pi"][-1], axis1=1, axis2=2).mean()),
            "final_dF_finite": bool(np.isfinite(r["final_dF"]).all())}


def main() -> int:
    out_dir = ROOT / "results" / "cosmology_forgetting"
    out_dir.mkdir(parents=True, exist_ok=True)
    scn = sc.cosmology_scenario(n_steps=N_STEPS, t1=T1, t2=T2, sigma_o=SIGMA_O)
    pop = cosmology_population(seed=0)
    theory_mu = sc.cosmology_theory_means()
    print(f"cosmology world: epochs {EPOCH_NAMES}; T1={T1} T2={T2} n_steps={N_STEPS}")

    # ---- (1) omega sweep on the LEVER-FREE cell (connected, no conviction): the wall ----
    print("\n=== (1) does forgetting break the wall? (connected, no conviction) ===")
    print(f"{'omega':>6} | by-epoch frac-on-true [ep0, ep1, ep2] | end modal | mean precision")
    sweep = []
    for om in OMEGAS:
        d = run_omega(scn, pop, omega=om, conviction=0.0, connected=True)
        by_ep = [float(d["frac_on_true"][d["true_epoch"] == e].mean()) for e in range(3)]
        end_modal = int(np.bincount(d["held"][:, -1]).argmax())
        sweep.append({**d, "by_epoch": by_ep, "end_modal": end_modal})
        print(f"{om:6.2f} | [{by_ep[0]:.2f}, {by_ep[1]:.2f}, {by_ep[2]:.2f}]"
              f"                  | {EPOCH_NAMES[end_modal][:12]:>12} | {d['mean_prec']:.1f}")

    # ---- (2) motivated persistence: under forgetting, lock-in must be EARNED ----
    # Without forgetting the value tilt accumulates ~ t, so ANY positive conviction eventually
    # dominates -> lock-in is automatic (the degenerate limit). With forgetting the tilt saturates
    # at lambda*U/(1-omega) and only holds if it beats the (also-saturated) per-step evidence -- so
    # lock-in becomes a THRESHOLD on conviction. We sweep conviction at omega=0.90 (disconnected)
    # and find where the conservative bloc (home = dark matter) stops re-tracking and stays put:
    # motivated persistence -- a belief held after its evidence has leaked away.
    print("\n=== (2) motivated persistence: lock-in is now EARNED (ω=0.90, disconnected) ===")
    conv_grid = (0.0, 6.0, 15.0, 30.0, 60.0)
    persist = []
    for cv in conv_grid:
        d = run_omega(scn, pop, omega=0.90, conviction=cv, connected=False)
        persist.append({"conviction": cv, "cons_end": int(d["held"][0][-1]), "held": d["held"]})
        print(f"  conviction={cv:5.1f}: conservative bloc ends on "
              f"{EPOCH_NAMES[persist[-1]['cons_end']]:>18}  (0=dark=persisted)")
    locked = [p["conviction"] for p in persist if p["cons_end"] == 0 and p["conviction"] > 0]
    thresh = min(locked) if locked else None
    print(f"  => threshold: conviction ≥ {thresh} holds the bloc on dark matter despite forgetting "
          f"(without forgetting ANY conviction would eventually do so -- lock-in is now EARNED).")

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    true_e = sweep[0]["true_epoch"]; snap_t = sweep[0]["snap_t"]
    ax.step(snap_t, true_e / 2.0, where="post", color="0.4", lw=5, alpha=0.25,
            label="true theory (0→1→2, scaled)")
    for i, d in enumerate(sweep):
        ax.plot(snap_t, d["frac_on_true"], lw=2, color=cm.viridis(i / (len(sweep) - 1)),
                label=f"ω={d['omega']:.2f}")
    for tt in (T1, T2):
        ax.axvline(tt, color="k", ls=":", lw=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("fraction of population on the CURRENT true theory")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8, ncol=2)
    ax.set_title("(1) forgetting breaks the wall: ω<1 re-tracks each epoch (ω=1 stays stuck)")
    plt.tight_layout(); plt.savefig(out_dir / "diag_wall_break.png", dpi=120); plt.close(fig)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8))
    oms = [d["omega"] for d in sweep]
    track12 = [float(np.mean(d["by_epoch"][1:])) for d in sweep]   # re-tracking (epochs 1&2)
    a0.plot(oms, track12, "o-", color="seagreen", lw=2)
    a0.set_xlabel("forgetting ω (1 = no forgetting)"); a0.set_ylabel("re-tracking (mean frac-on-true, ep 1&2)")
    a0.set_title("more forgetting → better re-tracking"); a0.invert_xaxis()
    a1.plot(oms, [d["mean_prec"] for d in sweep], "s-", color="indianred", lw=2)
    a1.set_xlabel("forgetting ω"); a1.set_ylabel("steady-state precision (trace Π)")
    a1.set_title("…but less forgetting keeps more confidence (the tradeoff)"); a1.invert_xaxis()
    plt.tight_layout(); plt.savefig(out_dir / "diag_tradeoff.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    cvs = [p["conviction"] for p in persist]
    ax.plot(cvs, [p["cons_end"] for p in persist], "o-", color="purple", lw=2)
    if thresh is not None:
        ax.axvline(thresh, color="0.5", ls=":", lw=1)
    ax.set_xlabel("conviction λ  (ω=0.90, disconnected)")
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels([n.replace('_', ' ') for n in EPOCH_NAMES])
    ax.set_ylabel("conservative bloc's FINAL theory")
    ax.set_title("(2) motivated persistence: above a conviction threshold the bloc stays on dark matter")
    plt.tight_layout(); plt.savefig(out_dir / "diag_motivated_persistence.png", dpi=120); plt.close(fig)

    # ---- asserts ----
    wall = sweep[0]                                # omega = 1.0
    broke = next(d for d in sweep if d["omega"] <= 0.88)
    print("\n=== controls ===")
    print(f"  ω=1.00 epoch-2 frac-on-true = {wall['by_epoch'][2]:.2f} (expect low: the wall)")
    print(f"  ω={broke['omega']:.2f} epoch-2 frac-on-true = {broke['by_epoch'][2]:.2f} (expect high: re-tracks)")
    print(f"  no-conviction bloc (ω=0.90) ends on {EPOCH_NAMES[persist[0]['cons_end']]} "
          f"(expect re-tracks, not dark_matter)")
    print(f"  high-conviction bloc (ω=0.90, λ={conv_grid[-1]}) ends on "
          f"{EPOCH_NAMES[persist[-1]['cons_end']]} (expect dark_matter: motivated persistence)")
    print(f"  ΔF finite: {all(d['final_dF_finite'] for d in sweep)}")

    assert wall["by_epoch"][2] < 0.34, \
        f"ω=1 should stay stuck (low epoch-2 tracking), got {wall['by_epoch'][2]:.2f}"
    assert broke["by_epoch"][2] > wall["by_epoch"][2] + 0.25, \
        f"forgetting should improve epoch-2 tracking ({broke['by_epoch'][2]:.2f} vs {wall['by_epoch'][2]:.2f})"
    assert persist[0]["cons_end"] != 0, \
        "with no conviction + forgetting the bloc should RE-TRACK (not persist on dark_matter)"
    assert persist[-1]["cons_end"] == 0, \
        "high conviction + forgetting should hold the bloc on dark_matter (motivated persistence)"
    assert all(d["final_dF_finite"] for d in sweep), "non-finite ΔF under forgetting"

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        omegas=np.array(OMEGAS), snap_t=snap_t, true_epoch=true_e,
        frac_on_true=np.stack([d["frac_on_true"] for d in sweep]),
        by_epoch=np.array([d["by_epoch"] for d in sweep]),
        mean_prec=np.array([d["mean_prec"] for d in sweep]),
        persist_conv=np.array([p["conviction"] for p in persist]),
        persist_cons_end=np.array([p["cons_end"] for p in persist]),
        persist_held=np.stack([p["held"] for p in persist]),       # (n_conv, n_comm, S)
        epoch_names=np.array(EPOCH_NAMES), comm_labels=np.array(COMM_LABELS),
        T1=T1, T2=T2,
    )
    summary = {
        "omegas": list(OMEGAS),
        "wall_break": {str(d["omega"]): d["by_epoch"] for d in sweep},
        "mean_precision": {str(d["omega"]): d["mean_prec"] for d in sweep},
        "motivated_persistence": {
            "conviction_grid": list(conv_grid),
            "conservative_bloc_end": [p["cons_end"] for p in persist],
            "threshold": thresh},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figures / summary to {out_dir}")
    print(f"HEADLINE: ω=1 stuck on epoch 0 (ep-2 track {wall['by_epoch'][2]:.2f}); "
          f"ω={broke['omega']:.2f} RE-TRACKS (ep-2 track {broke['by_epoch'][2]:.2f}); "
          f"conviction+forgetting = motivated persistence (committed bloc stays stuck).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
