"""E2 -- ENDOGENOUS gamma: conviction silences its own disconfirming channels, deriving lock-in.

The paper (eq:gamma) derives evidential lock-in from gamma = the fraction of a paradigm's
disconfirming sensory precision the *conviction field* has driven toward zero. nb43 produced the
lock-in with an EXOGENOUS, hand-set gamma (a self-censorship knob). Here we make gamma ENDOGENOUS:
each step the observation weight on the disconfirming rows is a saturating function of the
conviction field U=T u projected onto those rows,

    w_disc = exp(-gate_strength * |U . H_disc|),     gamma = mean(1 - w_disc),

so the more a community values the incumbent, the more it silences the very channels that would
refute it -- and the lock-in nb43 imposed by hand falls out of conviction.

Substrate: PHLOGISTON (single regime shift at t_shift). Its disconfirming rows (the gravimetric
mass-balance + disagreement-node reads) genuinely carry the refuting signal, so silencing them
prevents the structural revolution -- exactly the channel the paper's gamma gates. (Cosmology's
balance rows are structural; silencing them would not stop mean-tracking, so it is the wrong
substrate for this mechanism.) NO conviction TILT is applied: conviction acts ONLY through the
gating, so the result is a clean isolation of gamma -- the make-or-break control is "deaf but
honest": the SAME conviction with the gate OFF leaves the channels open and the community revolts.

Run it::

    python scripts/run_endogenous_gamma.py

Sweeps gate_strength, shows gamma(g) rising while the revolted fraction collapses (the gamma-crossover
phase boundary), asserts the controls, and saves to results/structural_endogenous_gamma/.
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

from src.structural import graphs, shells
from src.structural.phlogiston import StructuralConfig
from src.structural.simulation import run_simulation, AgentSpec
from src.structural.scenarios import build_substrate, phlogiston_scenario, BELT

N = 60
LAM = 0.20                       # prune threshold; at gamma=0 the refutation lands -> revolt
GATES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
SEEDS = (0, 1, 2)
CFG = dict(t_shift=40, n_steps=120, sigma_o=0.5)


def run_g(scn, *, gate_strength: float, endogenous: bool, seed: int) -> dict:
    spec = AgentSpec(w_obs=jnp.ones((N, scn.m)), lam=jnp.full((N,), LAM))
    g = graphs.complete(N)
    r = run_simulation(scn, g, spec, forgetting=1.0, endogenous_gamma=endogenous,
                       gate_strength=gate_strength, snapshot_every=5, seed=seed)
    return {"revolt": r["final_revolted_fraction"], "gamma_final": float(r["gamma_t"][-1]),
            "gamma_t": r["gamma_t"], "m_t": r["m_t"], "snap_t": r["snap_t"],
            "dF_finite": bool(np.isfinite(r["final_dF"]).all())}


def main() -> int:
    out_dir = ROOT / "results" / "structural_endogenous_gamma"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = StructuralConfig(**CFG)
    sub = build_substrate(cfg)
    scn = phlogiston_scenario(cfg, sub=sub)
    print(f"phlogiston: contested belt {list(BELT)}; disc rows = {len(scn.disc_rows)}; "
          f"shift at t={CFG['t_shift']}/{CFG['n_steps']}")

    # ---- sweep gate_strength (endogenous gamma ON) ----
    print("\n=== endogenous gamma sweep (conviction silences its own disc channels) ===")
    print(f"{'gate g':>7} | {'gamma_final':>11} | revolted fraction")
    sweep = []
    for g in GATES:
        rs = [run_g(scn, gate_strength=g, endogenous=True, seed=s) for s in SEEDS]
        rev = float(np.mean([r["revolt"] for r in rs]))
        gam = float(np.mean([r["gamma_final"] for r in rs]))
        sweep.append({"g": g, "gamma": gam, "revolt": rev,
                      "m_t": rs[0]["m_t"], "snap_t": rs[0]["snap_t"]})
        print(f"{g:7.2f} | {gam:11.2f} | {rev:.3f}")

    # ---- deaf-but-honest control: SAME high conviction, gate OFF (channels open) ----
    g_hi = GATES[-1]
    deaf = [run_g(scn, gate_strength=g_hi, endogenous=False, seed=s) for s in SEEDS]
    deaf_rev = float(np.mean([r["revolt"] for r in deaf]))
    deaf_gam = float(np.mean([r["gamma_final"] for r in deaf]))

    g0, ghi = sweep[0], sweep[-1]
    stalled, m_final, m_peak = shells.core_stall(ghi["m_t"], level=0.5)
    print("\n=== controls ===")
    print(f"  g=0 (gamma~0):            revolted={g0['revolt']:.3f} gamma={g0['gamma']:.2f} "
          f"(expect revolt high: channels open)")
    print(f"  g={g_hi} endogenous:        revolted={ghi['revolt']:.3f} gamma={ghi['gamma']:.2f} "
          f"(expect revolt low: gamma-silenced lock-in)")
    print(f"  g={g_hi} DEAF-BUT-HONEST:   revolted={deaf_rev:.3f} gamma={deaf_gam:.2f} "
          f"(gate OFF, same conviction -> expect revolt high: the stall is gamma, not absent evidence)")
    print(f"  ΔF finite: {all(r['dF_finite'] for r in deaf)}")

    assert g0["revolt"] > 0.6, f"g=0 should revolt (channels open), got {g0['revolt']:.2f}"
    assert ghi["revolt"] < g0["revolt"] - 0.3, \
        f"endogenous gamma should suppress the revolt ({ghi['revolt']:.2f} vs {g0['revolt']:.2f})"
    assert deaf_rev > 0.6, \
        f"deaf-but-honest (gate off, same conviction) should revolt, got {deaf_rev:.2f} -- " \
        f"if it stalls the lock-in is not the gating"
    gammas = [s["gamma"] for s in sweep]
    assert all(np.diff(gammas) >= -1e-6), f"gamma should rise monotonically with g, got {gammas}"
    assert all(r["dF_finite"] for r in deaf), "non-finite ΔF"

    # ---- figure: the gamma-crossover phase boundary ----
    gs = [s["g"] for s in sweep]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(gs, [s["revolt"] for s in sweep], "o-", color="crimson", lw=2.2,
            label="revolted fraction (endogenous γ)")
    ax.scatter([g_hi], [deaf_rev], marker="*", s=220, color="seagreen", zorder=5,
               label="deaf-but-honest (gate off, same conviction)")
    ax.set_xlabel("gate strength  (conviction → censorship coupling)")
    ax.set_ylabel("revolted fraction", color="crimson"); ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="y", labelcolor="crimson")
    axg = ax.twinx()
    axg.plot(gs, gammas, "s--", color="navy", lw=2, label="endogenous γ")
    axg.set_ylabel("endogenous γ (disc precision silenced)", color="navy")
    axg.set_ylim(-0.02, 1.02); axg.tick_params(axis="y", labelcolor="navy")
    ax.set_title("E2: conviction-driven γ silences the refuting channel → lock-in\n"
                 "(γ↑ as the revolt collapses; deaf-but-honest control revolts)")
    lines = ax.get_legend_handles_labels()[0] + axg.get_legend_handles_labels()[0]
    labs = ax.get_legend_handles_labels()[1] + axg.get_legend_handles_labels()[1]
    ax.legend(lines, labs, fontsize=8, loc="center right")
    plt.tight_layout(); plt.savefig(out_dir / "diag_endogenous_gamma.png", dpi=120); plt.close(fig)

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        gates=np.array(GATES), gamma=np.array(gammas),
        revolt=np.array([s["revolt"] for s in sweep]),
        deaf_revolt=deaf_rev, deaf_gamma=deaf_gam,
        m_t_g0=g0["m_t"], m_t_ghi=ghi["m_t"], snap_t=g0["snap_t"],
    )
    summary = {
        "config": {"N": N, "lam": LAM, "gates": list(GATES), "seeds": list(SEEDS), **CFG},
        "sweep": [{"g": s["g"], "gamma": s["gamma"], "revolt": s["revolt"]} for s in sweep],
        "deaf_but_honest": {"g": g_hi, "revolt": deaf_rev, "gamma": deaf_gam},
        "core_stall_high_g": {"stalled": bool(stalled), "final": m_final, "peak": m_peak},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figure / summary to {out_dir}")
    print(f"HEADLINE: conviction-driven γ rises 0→{ghi['gamma']:.2f} and the revolt collapses "
          f"{g0['revolt']:.2f}→{ghi['revolt']:.2f}; the deaf-but-honest control (gate off, same "
          f"conviction) still revolts ({deaf_rev:.2f}) -- lock-in is DERIVED from γ, not imposed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
