"""A complex world a COARSE Bayes net resolves only slowly (and, under forgetting, never fully).

The cosmology menu the population carries is *coarse*: a handful of commitments. But the world it
samples can be far richer -- a high-dimensional latent state with no clean low-dimensional answer.
This experiment isolates that mismatch. The true world is a complex Gaussian: ``M`` fine
parameters with a structured mean ``mu*`` and a dense, low-rank-plus-noise covariance ``Sigma*``
(``K`` latent factors -> the directions are correlated and individually ambiguous). A coarse agent
senses only ``res`` random aggregate combinations of the parameters per step (its limited
bandwidth) and accumulates them in information form, reconstructing its estimate ``mu_t`` of the
world state.

Two findings, both swept:

  1. COARSER => SLOWER. With sensing resolution ``res`` the agent covers the ``M`` complex
     directions only ``res`` at a time, so the time to resolve the world (mean error < eps) grows
     as ``res`` shrinks -- a fine sensor (``res = M``) nails it quickly, a coarse one (``res = 1``)
     grinds, and below some resolution it does not finish within the horizon. *The complex answer
     takes a while precisely because the net is coarse.*

  2. FORGETTING => AN IRREDUCIBLE FLOOR. Add the same forgetting omega < 1 used in
     ``run_cosmology_forgetting``: a forgetful coarse net can only hold a windowed estimate, so in a
     complex world it never fully resolves -- the mean error plateaus at a floor that rises as omega
     falls. *A complex world has no clear answer for a net that both under-senses and forgets.*

(Honest note: in this linear-Gaussian model the precision deposit ``H^T H`` is operator-set, so what
the agent *resolves* here is the world STATE -- the mean -- not the off-diagonal covariance
structure, which would need a structure-matched operator. That is the same caveat Lens B reports.)

Run it::

    python scripts/run_cosmology_coarse_world.py

Saves arrays/figures/summary to ``results/cosmology_coarse_world/`` and asserts the two orderings.
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

# Complex world + agent config
M = 12               # fine world dimension (the complex P)
K = 4                # latent factors (low-rank structure -> directions correlated/ambiguous)
N_STEPS = 260
SIGMA_O = 0.3
PREC0 = 1.0          # coarse isotropic prior (knows the scale, none of the structure)
EPS = 0.15           # "resolved" threshold on relative mean error
RES_GRID = (1, 2, 3, 6, 12)
OMEGA_GRID = (1.0, 0.97, 0.93, 0.88)
RES_FOR_OMEGA = 3    # a coarse sensor for the forgetting sweep


def make_world(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A complex Gaussian world: structured mean mu*, dense low-rank+noise covariance Sigma*."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(M, K))
    Sigma = A @ A.T / K + 0.35 * np.eye(M)            # low-rank structure + ridge (PD)
    d = np.sqrt(np.diag(Sigma))
    Sigma = Sigma / np.outer(d, d)                    # correlation matrix
    mu = rng.normal(size=M) * 1.5                     # the complex state to reconstruct
    L = np.linalg.cholesky(Sigma)
    return mu, Sigma, L


def run_resolve(res: int, *, omega: float = 1.0, seed: int = 0,
                n_steps: int = N_STEPS) -> dict:
    """Accumulate ``res`` random aggregate reads/step of the complex world; track mean error."""
    mu, Sigma, L = make_world(seed)
    rng = np.random.default_rng(seed + 1)
    Pi0 = PREC0 * np.eye(M)
    h0 = np.zeros(M)
    Pi, h = Pi0.copy(), h0.copy()
    err = np.empty(n_steps)
    for t in range(n_steps):
        if omega < 1.0:
            Pi = Pi0 + omega * (Pi - Pi0)
            h = h0 + omega * (h - h0)
        x = mu + L @ rng.normal(size=M)                       # a draw of the complex world
        Q = np.linalg.qr(rng.normal(size=(M, M)))[0]
        H = Q[:res]                                           # res random aggregate reads
        o = H @ x + SIGMA_O * rng.normal(size=res)
        Pi = Pi + (H.T @ H) / SIGMA_O ** 2
        h = h + (H.T @ o) / SIGMA_O ** 2
        mu_t = np.linalg.solve(Pi, h)
        err[t] = np.linalg.norm(mu_t - mu) / np.linalg.norm(mu)
    t_resolve = int(np.argmax(err < EPS)) if (err < EPS).any() else -1
    return {"res": res, "omega": omega, "err": err, "t_resolve": t_resolve,
            "final_err": float(err[-1])}


def main() -> int:
    out_dir = ROOT / "results" / "cosmology_coarse_world"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"complex world: M={M} fine params, K={K} latent factors; coarse sensor sees `res` "
          f"aggregate reads/step; horizon {N_STEPS}")

    # ---- (1) coarser => slower (no forgetting) ----
    print("\n=== (1) coarser net => slower to resolve the complex world (ω=1) ===")
    print(f"{'res':>4} | {'t_resolve':>9} | final mean-err")
    res_runs = []
    for res in RES_GRID:
        d = run_resolve(res, omega=1.0)
        res_runs.append(d)
        tr = d["t_resolve"] if d["t_resolve"] >= 0 else f">{N_STEPS}"
        print(f"{res:>4} | {str(tr):>9} | {d['final_err']:.3f}")

    # ---- (2) forgetting => irreducible floor (fixed coarse sensor) ----
    print(f"\n=== (2) forgetting => the complex world is never fully resolved (res={RES_FOR_OMEGA}) ===")
    print(f"{'omega':>6} | final mean-err | t_resolve")
    om_runs = []
    for om in OMEGA_GRID:
        d = run_resolve(RES_FOR_OMEGA, omega=om)
        om_runs.append(d)
        tr = d["t_resolve"] if d["t_resolve"] >= 0 else f">{N_STEPS}"
        print(f"{om:6.2f} | {d['final_err']:14.3f} | {tr}")

    # ---- figures ----
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for i, d in enumerate(res_runs):
        a0.plot(d["err"], lw=2, color=cm.viridis(i / (len(res_runs) - 1)), label=f"res={d['res']}")
    a0.axhline(EPS, color="grey", ls=":", lw=1, label=f"resolved (ε={EPS})")
    a0.set_xlabel("observation step"); a0.set_ylabel("relative mean error ‖μ̂−μ*‖/‖μ*‖")
    a0.set_yscale("log"); a0.set_title("(1) a coarse sensor resolves a complex world slowly")
    a0.legend(fontsize=8)
    res_vals = [d["res"] for d in res_runs]
    tr_vals = [d["t_resolve"] if d["t_resolve"] >= 0 else N_STEPS for d in res_runs]
    a1.plot(res_vals, tr_vals, "o-", color="seagreen", lw=2)
    a1.set_xlabel("sensing resolution res (12 = full)"); a1.set_ylabel("steps to resolve (ε)")
    a1.set_title("time-to-resolve grows as the net gets coarser")
    plt.tight_layout(); plt.savefig(out_dir / "diag_coarse_resolve.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, d in enumerate(om_runs):
        ax.plot(d["err"], lw=2, color=cm.plasma(i / (len(om_runs) - 1)), label=f"ω={d['omega']:.2f}")
    ax.axhline(EPS, color="grey", ls=":", lw=1)
    ax.set_xlabel("observation step"); ax.set_ylabel("relative mean error")
    ax.set_yscale("log"); ax.set_title(f"(2) forgetting floors the resolution of a complex world (res={RES_FOR_OMEGA})")
    ax.legend(fontsize=9); plt.tight_layout(); plt.savefig(out_dir / "diag_forgetting_floor.png", dpi=120)
    plt.close(fig)

    # ---- asserts ----
    # (1) finer resolution resolves at least as fast (monotone, allowing ties at the fast end)
    eff = [(d["t_resolve"] if d["t_resolve"] >= 0 else N_STEPS) for d in res_runs]
    assert eff[0] > eff[-1] + 5, \
        f"coarsest should be much slower than finest ({eff[0]} vs {eff[-1]})"
    assert eff == sorted(eff, reverse=True), \
        f"time-to-resolve should fall monotonically with resolution, got {eff}"
    # (2) more forgetting => higher final error floor
    floors = [d["final_err"] for d in om_runs]
    assert floors[-1] > floors[0] + 0.02, \
        f"forgetting should raise the residual floor ({floors[-1]:.3f} vs {floors[0]:.3f})"
    assert np.all(np.diff(floors) > -1e-3), \
        f"final error should rise (not fall) as ω decreases, got {floors}"

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        M=M, K=K, n_steps=N_STEPS, eps=EPS, res_grid=np.array(RES_GRID),
        omega_grid=np.array(OMEGA_GRID), res_for_omega=RES_FOR_OMEGA,
        err_by_res=np.stack([d["err"] for d in res_runs]),
        t_resolve=np.array(eff),
        err_by_omega=np.stack([d["err"] for d in om_runs]),
        floors=np.array(floors),
    )
    summary = {
        "config": {"M": M, "K": K, "n_steps": N_STEPS, "eps": EPS, "sigma_o": SIGMA_O,
                   "res_grid": list(RES_GRID), "omega_grid": list(OMEGA_GRID),
                   "res_for_omega": RES_FOR_OMEGA},
        "resolution_sweep": [{"res": d["res"], "t_resolve": d["t_resolve"],
                              "final_err": d["final_err"]} for d in res_runs],
        "forgetting_floor": [{"omega": d["omega"], "final_err": d["final_err"],
                              "t_resolve": d["t_resolve"]} for d in om_runs],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figures / summary to {out_dir}")
    print(f"HEADLINE: time-to-resolve a complex M={M} world grows from {eff[-1]} steps (full res) "
          f"to {eff[0]} (res=1); with forgetting the coarse net plateaus at a floor "
          f"({floors[0]:.2f}→{floors[-1]:.2f} as ω falls) -- a complex world has no clear answer "
          f"for an under-sensing, forgetting net.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
