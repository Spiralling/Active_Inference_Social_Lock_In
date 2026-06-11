"""A complex world a COARSE Bayes net resolves only slowly (and, under forgetting, never fully).

The cosmology menu the population carries is *coarse*: a handful of commitments. But the world it
samples can be far richer -- a high-dimensional latent state with no clean low-dimensional answer.
This experiment isolates that mismatch. The true world is a complex Gaussian: ``M`` fine parameters
with a structured mean ``mu*`` and a dense low-rank+noise covariance ``Sigma*`` (``K`` latent factors
-> correlated, individually ambiguous directions). A coarse agent senses only ``res`` random
aggregate combinations of the parameters per step and accumulates them in information form.

Two findings, both swept:

  1. COARSER => SLOWER. Time to resolve the world (mean error < eps) grows as ``res`` shrinks.
  2. FORGETTING => AN IRREDUCIBLE FLOOR. With omega < 1 a forgetful coarse net never fully resolves
     a complex world -- the mean error plateaus at a floor that rises as omega falls.

(Honest note: in this linear-Gaussian model the precision deposit H^T H is operator-set, so what the
agent resolves is the world STATE -- the mean -- not the off-diagonal covariance structure. Same
caveat Lens B reports.)

Self-contained: the complex world + coarse sensor live entirely in this module.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from experiments.registry import ExperimentSpec, register


def make_world(seed: int, M: int, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A complex Gaussian world: structured mean mu*, dense low-rank+noise covariance Sigma*."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(M, K))
    Sigma = A @ A.T / K + 0.35 * np.eye(M)            # low-rank structure + ridge (PD)
    d = np.sqrt(np.diag(Sigma))
    Sigma = Sigma / np.outer(d, d)                    # correlation matrix
    mu = rng.normal(size=M) * 1.5                     # the complex state to reconstruct
    L = np.linalg.cholesky(Sigma)
    return mu, Sigma, L


def run_resolve(res: int, *, M: int, K: int, prec0: float, sigma_o: float, eps: float,
                omega: float, seed: int, n_steps: int) -> dict:
    """Accumulate ``res`` random aggregate reads/step of the complex world; track mean error."""
    mu, Sigma, L = make_world(seed, M, K)
    rng = np.random.default_rng(seed + 1)
    Pi0 = prec0 * np.eye(M)
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
        o = H @ x + sigma_o * rng.normal(size=res)
        Pi = Pi + (H.T @ H) / sigma_o ** 2
        h = h + (H.T @ o) / sigma_o ** 2
        mu_t = np.linalg.solve(Pi, h)
        err[t] = np.linalg.norm(mu_t - mu) / np.linalg.norm(mu)
    t_resolve = int(np.argmax(err < eps)) if (err < eps).any() else -1
    return {"res": res, "omega": omega, "err": err, "t_resolve": t_resolve,
            "final_err": float(err[-1])}


def run(out_dir, params: dict) -> None:
    M = params["M"]; K = params["K"]; n_steps = params["N_STEPS"]
    sigma_o = params["SIGMA_O"]; prec0 = params["PREC0"]; eps = params["EPS"]
    res_grid = tuple(params["RES_GRID"]); omega_grid = tuple(params["OMEGA_GRID"])
    res_for_omega = params["RES_FOR_OMEGA"]

    print(f"complex world: M={M} fine params, K={K} latent factors; coarse sensor sees `res` "
          f"aggregate reads/step; horizon {n_steps}")

    # ---- (1) coarser => slower (no forgetting) ----
    print("\n=== (1) coarser net => slower to resolve the complex world (ω=1) ===")
    print(f"{'res':>4} | {'t_resolve':>9} | final mean-err")
    res_runs = []
    for res in res_grid:
        d = run_resolve(res, M=M, K=K, prec0=prec0, sigma_o=sigma_o, eps=eps, omega=1.0,
                        seed=0, n_steps=n_steps)
        res_runs.append(d)
        tr = d["t_resolve"] if d["t_resolve"] >= 0 else f">{n_steps}"
        print(f"{res:>4} | {str(tr):>9} | {d['final_err']:.3f}")

    # ---- (2) forgetting => irreducible floor (fixed coarse sensor) ----
    print(f"\n=== (2) forgetting => the complex world is never fully resolved (res={res_for_omega}) ===")
    print(f"{'omega':>6} | final mean-err | t_resolve")
    om_runs = []
    for om in omega_grid:
        d = run_resolve(res_for_omega, M=M, K=K, prec0=prec0, sigma_o=sigma_o, eps=eps, omega=om,
                        seed=0, n_steps=n_steps)
        om_runs.append(d)
        tr = d["t_resolve"] if d["t_resolve"] >= 0 else f">{n_steps}"
        print(f"{om:6.2f} | {d['final_err']:14.3f} | {tr}")

    # ---- figures ----
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for i, d in enumerate(res_runs):
        a0.plot(d["err"], lw=2, color=cm.viridis(i / (len(res_runs) - 1)), label=f"res={d['res']}")
    a0.axhline(eps, color="grey", ls=":", lw=1, label=f"resolved (ε={eps})")
    a0.set_xlabel("observation step"); a0.set_ylabel("relative mean error ‖μ̂−μ*‖/‖μ*‖")
    a0.set_yscale("log"); a0.set_title("(1) a coarse sensor resolves a complex world slowly")
    a0.legend(fontsize=8)
    res_vals = [d["res"] for d in res_runs]
    tr_vals = [d["t_resolve"] if d["t_resolve"] >= 0 else n_steps for d in res_runs]
    a1.plot(res_vals, tr_vals, "o-", color="seagreen", lw=2)
    a1.set_xlabel("sensing resolution res (12 = full)"); a1.set_ylabel("steps to resolve (ε)")
    a1.set_title("time-to-resolve grows as the net gets coarser")
    plt.tight_layout(); plt.savefig(out_dir / "diag_coarse_resolve.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, d in enumerate(om_runs):
        ax.plot(d["err"], lw=2, color=cm.plasma(i / (len(om_runs) - 1)), label=f"ω={d['omega']:.2f}")
    ax.axhline(eps, color="grey", ls=":", lw=1)
    ax.set_xlabel("observation step"); ax.set_ylabel("relative mean error")
    ax.set_yscale("log"); ax.set_title(f"(2) forgetting floors the resolution of a complex world (res={res_for_omega})")
    ax.legend(fontsize=9); plt.tight_layout(); plt.savefig(out_dir / "diag_forgetting_floor.png", dpi=120)
    plt.close(fig)

    # ---- asserts ----
    eff = [(d["t_resolve"] if d["t_resolve"] >= 0 else n_steps) for d in res_runs]
    assert eff[0] > eff[-1] + 5, \
        f"coarsest should be much slower than finest ({eff[0]} vs {eff[-1]})"
    assert eff == sorted(eff, reverse=True), \
        f"time-to-resolve should fall monotonically with resolution, got {eff}"
    floors = [d["final_err"] for d in om_runs]
    assert floors[-1] > floors[0] + 0.02, \
        f"forgetting should raise the residual floor ({floors[-1]:.3f} vs {floors[0]:.3f})"
    assert np.all(np.diff(floors) > -1e-3), \
        f"final error should rise (not fall) as ω decreases, got {floors}"

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        M=M, K=K, n_steps=n_steps, eps=eps, res_grid=np.array(res_grid),
        omega_grid=np.array(omega_grid), res_for_omega=res_for_omega,
        err_by_res=np.stack([d["err"] for d in res_runs]),
        t_resolve=np.array(eff),
        err_by_omega=np.stack([d["err"] for d in om_runs]),
        floors=np.array(floors),
    )
    summary = {
        "config": {"M": M, "K": K, "n_steps": n_steps, "eps": eps, "sigma_o": sigma_o,
                   "res_grid": list(res_grid), "omega_grid": list(omega_grid),
                   "res_for_omega": res_for_omega},
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


register(ExperimentSpec(
    model="cosmology",
    name="cosmology_coarse_world",
    description="A coarse Bayes net resolves a complex high-dim world only slowly (time-to-resolve "
                "grows as sensing resolution falls); forgetting imposes an irreducible error floor.",
    run=run,
    out_dir="cosmology_coarse_world",
    params=dict(M=12, K=4, N_STEPS=260, SIGMA_O=0.3, PREC0=1.0, EPS=0.15,
                RES_GRID=(1, 2, 3, 6, 12), OMEGA_GRID=(1.0, 0.97, 0.93, 0.88), RES_FOR_OMEGA=3),
    consumes=dict(figures=["diag_coarse_resolve", "diag_forgetting_floor"]),
))
