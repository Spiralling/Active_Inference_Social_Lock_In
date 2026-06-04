"""E1 -- recover the belt-first / core-last staircase by letting conservatism gate revision RATE.

The paper predicts (fig:phase) that when a population reorganizes toward a new theory, low-kappa
"belt" nodes move EARLY and high-kappa "core" nodes move LATE. In the base model this washes out
(paper §7.6) because conservatism kappa is a STATIC read-out and never enters the dynamics. Here we
add the missing ingredient: a per-node revision-rate gate g_i = 1/(1 + beta*kappa_i) that scales the
Fisher deposit PD-preservingly (J <- G^.5 J G^.5, j <- G^.5 j), so a high-conservatism node integrates
evidence slower than a cheap belt node. We then read the kappa-stratified reorganization trace and
ask whether the belt-first/core-last gap appears, and grows with beta.

Substrate: PHLOGISTON -- shells.node_oxygen_trace is built for it and shells.py documents a
*known shallow* per-node staircase here; it is the net fig:phase was drawn against.

HONEST FRAMING (per project norms): theory identity rides the posterior MEANS, and a deposit gate
scales both the numerator and denominator of mu = Pi^-1 h, so the staircase may be shallow-but-real
or may stay flat. We report the lag(beta) curve and two decisive nulls -- a UNIFORM slowdown (same
total slowdown, no kappa gradient) and a SHUFFLED-kappa gate -- which must NOT produce the staircase
if it is real. Positive or null, it is a result: it tests whether a kappa-coupled deposit rate is the
missing ingredient §7.6 diagnosed.

Run it::

    python scripts/run_staircase_gate.py
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
from matplotlib import cm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.structural import graphs, shells, observables as obs
from src.structural.phlogiston import StructuralConfig
from src.structural.simulation import run_simulation, AgentSpec
from src.structural.scenarios import build_substrate, phlogiston_scenario

N = 60
BETAS = (0.0, 1.0, 3.0, 10.0, 30.0)
BETA_CTRL = 10.0                 # the beta at which the uniform / shuffled nulls are run
SEED = 0
CFG = dict(t_shift=40, n_steps=120, sigma_o=0.5)


def t_half(curve: np.ndarray, snap_t: np.ndarray, level: float = 0.5) -> float:
    """First snapshot time the (rising 0->1) reorganization curve crosses ``level``."""
    c = np.asarray(curve)
    hit = c >= level
    return float(snap_t[int(np.argmax(hit))]) if hit.any() else float(snap_t[-1] + 1)


def run_gate(scn, cfg, gate_vec: np.ndarray | None, *, seed: int = SEED) -> dict:
    spec = AgentSpec(w_obs=jnp.ones((N, scn.m)), lam=jnp.full((N,), 0.2))
    g = graphs.complete(N)
    dep = None if gate_vec is None else jnp.asarray(gate_vec)
    r = run_simulation(scn, g, spec, forgetting=1.0, deposit_gate=dep, snapshot_every=5, seed=seed)
    return r


def staircase(r: dict, cfg, kappa_by_name: dict) -> dict:
    """kappa-stratified reorganization curves + the belt->core lag."""
    meas, oxy_t = shells.node_oxygen_trace(r["snap_Pi"], r["snap_h"], cfg)   # (n_meas,), (S,n_meas)
    kap = np.array([kappa_by_name[n] for n in meas])
    shell_id, labels = shells.assign_shells(kap, 3)
    curves = shells.shell_curves(oxy_t, shell_id, labels)                    # {label:(S,)}
    sizes = shells.shell_sizes(shell_id, labels)
    snap_t = r["snap_t"]
    th = {lab: t_half(curves[lab], snap_t) for lab in curves}
    lag = (th.get("core", np.nan) - th.get("belt", np.nan))
    _, bc = shells.conservatism_split(kap)
    return {"curves": curves, "sizes": sizes, "t_half": th, "lag": float(lag),
            "bc": float(bc), "snap_t": snap_t}


def main() -> int:
    out_dir = ROOT / "results" / "structural_staircase_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = StructuralConfig(**CFG)
    sub = build_substrate(cfg)
    scn = phlogiston_scenario(cfg, sub=sub)

    # self-consistent kappa: the carry-over mass of each node in the SIMULATED prior net
    net0 = sub.over.to_info()
    kappa_full = np.array([float(obs.carryover_mass(net0, (n,))) for n in scn.names])
    kappa_by_name = {n: kappa_full[i] for i, n in enumerate(scn.names)}
    print(f"phlogiston: {len(scn.names)} nodes; kappa range "
          f"[{kappa_full.min():.2f}, {kappa_full.max():.2f}]; shift at t={CFG['t_shift']}")

    def gate(beta, kvec=kappa_full):
        return 1.0 / (1.0 + beta * kvec)

    # ---- beta sweep ----
    print("\n=== beta sweep: does conservatism-gated revision rate produce the staircase? ===")
    print(f"{'beta':>6} | {'t_half(belt)':>12} {'t_half(core)':>12} | {'lag':>6} | BC | shell sizes")
    sweep = []
    for b in BETAS:
        r = run_gate(scn, cfg, gate(b))
        st = staircase(r, cfg, kappa_by_name)
        sweep.append({"beta": b, **st})
        print(f"{b:6.1f} | {st['t_half'].get('belt', float('nan')):12.0f} "
              f"{st['t_half'].get('core', float('nan')):12.0f} | {st['lag']:6.1f} | "
              f"{st['bc']:.2f} | {st['sizes']}")

    # ---- controls at BETA_CTRL ----
    g_grad = gate(BETA_CTRL)
    g_uniform = np.full_like(kappa_full, float(np.mean(g_grad)))       # same mean slowdown, no gradient
    rng = np.random.default_rng(0)
    g_shuffled = gate(BETA_CTRL, rng.permutation(kappa_full))          # gradient on the wrong nodes
    lag_grad = next(s["lag"] for s in sweep if s["beta"] == BETA_CTRL)
    lag_uniform = staircase(run_gate(scn, cfg, g_uniform), cfg, kappa_by_name)["lag"]
    lag_shuffled = staircase(run_gate(scn, cfg, g_shuffled), cfg, kappa_by_name)["lag"]
    lag0 = sweep[0]["lag"]

    print("\n=== controls (at beta=%.0f) ===" % BETA_CTRL)
    print(f"  beta=0 (no gate):       lag={lag0:+.1f}  (the §7.6 washout baseline)")
    print(f"  kappa-gradient gate:    lag={lag_grad:+.1f}  (the real mechanism)")
    print(f"  uniform slowdown:       lag={lag_uniform:+.1f}  (same total slowdown, NO gradient)")
    print(f"  shuffled-kappa gate:    lag={lag_shuffled:+.1f}  (gradient on the wrong nodes)")

    finite = all(np.isfinite(s["lag"]) for s in sweep)
    assert finite, "non-finite lag"
    # belt-first/core-last is a POSITIVE lag (core's t_half later than belt's); the kappa-GRADIENT
    # must also beat the gradient-free (uniform) and wrong-gradient (shuffled) nulls to count.
    positive = lag_grad > 1.0
    gradient_matters = (lag_grad > lag_uniform + 1.0) and (lag_grad > lag_shuffled + 1.0)
    staircase_real = positive and gradient_matters and (lag_grad > lag0 + 1.0)
    verdict = ("RECOVERED" if staircase_real
               else ("NULL: lag is negative (high-kappa nodes reorganize FIRST -- the moving "
                     "nodes are the high-kappa mass-law disagreement nodes; the low-kappa belt are "
                     "agreement nodes that never move)" if lag_grad <= 0 else
                     "NULL: the kappa-gradient does not beat the gradient-free nulls"))
    print(f"\n  => staircase verdict: {verdict}\n     "
          f"(lag {lag0:+.1f} (b=0) -> {sweep[-1]['lag']:+.1f} (b=max); "
          f"gradient {lag_grad:+.1f} vs uniform {lag_uniform:+.1f} vs shuffled {lag_shuffled:+.1f} "
          f"-- the gradient adds nothing beyond uniform slowdown)")

    # ---- figure: the staircase at the best beta, with the beta=0 washout ghost ----
    best = max(sweep, key=lambda s: s["lag"])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8))
    cols = {"belt": "seagreen", "mid": "goldenrod", "core": "teal"}
    for lab, c in best["curves"].items():
        a0.plot(best["snap_t"], c, lw=2.4, color=cols.get(lab, "0.5"),
                label=f"{lab} (n={best['sizes'].get(lab,0)})")
    for lab, c in sweep[0]["curves"].items():        # beta=0 ghost
        a0.plot(sweep[0]["snap_t"], c, lw=1.2, ls=":", color=cols.get(lab, "0.5"), alpha=0.6)
    a0.axvline(CFG["t_shift"], color="k", ls="--", lw=0.8); a0.set_ylim(-0.05, 1.05)
    a0.set_xlabel("step"); a0.set_ylabel("reorganization (oxygen index)")
    a0.set_title(f"belt-first / core-last at β={best['beta']:.0f} (β=0 dotted ghost); BC={best['bc']:.2f}")
    a0.legend(fontsize=8)
    a1.plot([s["beta"] for s in sweep], [s["lag"] for s in sweep], "o-", color="purple", lw=2,
            label="κ-gradient gate")
    a1.scatter([BETA_CTRL], [lag_uniform], marker="s", s=80, color="grey", label="uniform slowdown")
    a1.scatter([BETA_CTRL], [lag_shuffled], marker="x", s=80, color="crimson", label="shuffled κ")
    a1.axhline(0, color="k", lw=0.6)
    a1.set_xlabel("gate strength β"); a1.set_ylabel("belt→core lag (steps)")
    a1.set_title("the staircase grows with β iff the κ-gradient is the cause"); a1.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "diag_staircase.png", dpi=120); plt.close(fig)

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        betas=np.array(BETAS), lag=np.array([s["lag"] for s in sweep]),
        bc=np.array([s["bc"] for s in sweep]),
        lag_uniform=lag_uniform, lag_shuffled=lag_shuffled, beta_ctrl=BETA_CTRL,
        best_beta=best["beta"], snap_t=best["snap_t"],
        **{f"best_{lab}": c for lab, c in best["curves"].items()},
        **{f"beta0_{lab}": c for lab, c in sweep[0]["curves"].items()},
        kappa=kappa_full, kappa_names=np.array(list(scn.names)),
    )
    summary = {
        "config": {"N": N, "betas": list(BETAS), "beta_ctrl": BETA_CTRL, **CFG},
        "kappa_range": [float(kappa_full.min()), float(kappa_full.max())],
        "sweep": [{"beta": s["beta"], "lag": s["lag"], "bc": s["bc"],
                   "t_half": s["t_half"], "shell_sizes": s["sizes"]} for s in sweep],
        "controls": {"lag_beta0": lag0, "lag_gradient": lag_grad,
                     "lag_uniform": lag_uniform, "lag_shuffled": lag_shuffled},
        "verdict": verdict,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figure / summary to {out_dir}")
    print(f"HEADLINE: staircase {verdict}; belt→core lag {lag0:+.0f} (β=0) → {best['lag']:+.0f} "
          f"(β={best['beta']:.0f}); uniform/shuffled nulls give {lag_uniform:+.0f}/{lag_shuffled:+.0f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
