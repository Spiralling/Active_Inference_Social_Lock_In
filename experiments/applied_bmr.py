"""Applied BMR: the crisis check executed -- structure that actually follows evidence.

The paper's Section 5 claims "the agent applies the reduction"; until now the engine only
ever SCORED it (the Savage-Dickey ledger ``dF + lambda dU`` was a snapshot read-out --
``kept_t`` flags -- and the net was never touched). This experiment runs the new
``run_simulation(bmr_every=...)`` path, where every ``BMR_EVERY`` steps each agent's
flagged edges are removed by swapping in the exact Savage-Dickey reduced posterior (joint
reduced reference prior + the agent's untouched deposit, forgetting anchor included), on
the phlogiston substrate with the world stepping from phlogiston-supporting to
oxygen-supporting data at ``T_SHIFT``.

What it shows, in three single-message pictures:
  1. the stale mass-law coupling actually LEAVES the net (read-out keeps it forever);
  2. removal is evidence-TIMED: the never-supported spurious over-wiring goes almost
     immediately (Occam -- the data were always silent on it), the protective mass-law
     belt goes only AFTER the world turns;
  3. conviction lambda is the lock-in dial: evidence prunes below a sharp threshold,
     value protection stops the prune above it -- the same ledger, now with consequences.

Open-endedness placement: fusion destroys disagreement (the structure-gate results),
attention fabricates structure (the divergent-wiring three-camps refutation) -- and
neither process consulted the evidence. Applied BMR is the missing adjudication step:
structure the data abandon now actually dies, structure the data support persists, and a
cleared flag RESTORES the edge (reversible -- regrowth on supporting evidence is allowed;
the probe indeed saw one spurious edge partially return post-shift, reported honestly).

Honest notes (probe, seed 1): at low lambda the ledger also Occam-prunes the agreement
edges the gravimetric data are merely silent on (dF ~ +0.07 > 0 -- indifference reads as
"remove the prior cost"); reported in summary, figures are belt/spurious-scoped. The
residual coupling after a prune is the KEPT deposit (BMR removes prior structure, never
the data), so the fig-1 contrast needs the deposit window small vs the prior: GAMMA=0.98,
OMEGA=0.92 (window ~ 1.0 vs prior 1.8).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from experiments.registry import ExperimentSpec, register

RED, GREEN, GREY = "#922b21", "#1e8449", "#7f8c8d"


# ----------------------------------------------------------------------
# Worker side.
# ----------------------------------------------------------------------

def _edge_classes(scn):
    """(belt, spurious, core) edge index arrays of the phlogiston substrate."""
    from src.structural.scenarios import DEFAULT_SPURIOUS
    belt = np.asarray(scn.belt_ix)
    spur_pairs = {(p, c) for (p, c, _) in DEFAULT_SPURIOUS}
    spur = np.asarray([k for k, e in enumerate(scn.edges) if e in spur_pairs])
    core = np.asarray([k for k in range(scn.n_edges)
                       if k not in set(belt.tolist()) | set(spur.tolist())])
    return belt, spur, core


def _one_job(cfg: dict) -> dict:
    import jax.numpy as jnp
    from src.structural import graphs
    from src.structural.phlogiston import StructuralConfig
    from src.structural.scenarios import phlogiston_scenario
    from src.structural.simulation import run_simulation, AgentSpec

    p = cfg
    scn = phlogiston_scenario(StructuralConfig(
        regime_schedule="step", t_shift=p["T_SHIFT"], n_steps=p["N_STEPS"],
        sigma_o=p["SIGMA_O"]))
    n = p["N"]
    w = np.ones((n, scn.m))
    w[:, np.asarray(scn.disc_rows)] = 1.0 - p["GAMMA"]
    spec = AgentSpec(w_obs=jnp.asarray(w), lam=jnp.full((n,), float(p["lam"])))
    kw = {} if p["condition"] == "readout" else dict(bmr_every=p["BMR_EVERY"])
    r = run_simulation(scn, graphs.complete(n), spec, forgetting=p["OMEGA"],
                       snapshot_every=p["SNAPSHOT_EVERY"], seed=int(p["seed"]), **kw)

    idx = {nm: i for i, nm in enumerate(scn.names)}
    m_i, g_i, c_i = (idx["mass_change_sign"], idx["gas_consumed"],
                     idx["calx_heavier_than_metal"])
    belt, spur, core = _edge_classes(scn)
    belt_cpl_t = np.abs(r["snap_Pi"][:, :, [m_i, g_i], c_i]).mean(axis=(1, 2))  # (S,)
    out = dict(condition=p["condition"], lam=float(p["lam"]), seed=int(p["seed"]),
               snap_t=np.asarray(r["snap_t"]),
               belt_cpl_t=belt_cpl_t,
               belt_cpl_final=float(belt_cpl_t[-1]),
               kept_mean_t=r["kept_t"].mean(axis=1),
               max_Pi=float(np.abs(r["snap_Pi"]).max()),
               min_eig=float(min(np.linalg.eigvalsh(P).min()
                                 for P in r["snap_Pi"][-1])),
               finite=bool(np.isfinite(r["snap_Pi"]).all()
                           and np.isfinite(r["snap_h"]).all()))
    if p["condition"] == "applied":
        ap = r["applied_pruned_t"]                                   # (S, N, E)
        out.update(rem_belt_t=ap[:, :, belt].mean(axis=(1, 2)),
                   rem_spur_t=ap[:, :, spur].mean(axis=(1, 2)),
                   rem_core_t=ap[:, :, core].mean(axis=(1, 2)),
                   rem_belt_final=float(ap[-1][:, belt].mean()),
                   rem_belt_per_agent=ap[-1][:, belt].mean(axis=1))   # (N,)
    return out


# ----------------------------------------------------------------------
# Host side.
# ----------------------------------------------------------------------

def _rows(rows, **match):
    out = [r for r in rows if all(r[k] == v for k, v in match.items())]
    assert out, f"no rows for {match}"
    return out


def _mean(rows, key, **match):
    return np.mean([r[key] for r in _rows(rows, **match)], axis=0)


def _std(rows, key, **match):
    return np.std([r[key] for r in _rows(rows, **match)], axis=0)


def _band(ax, t, m, s, color, label, ls="-"):
    ax.plot(t, m, ls, color=color, lw=2.2, label=label)
    ax.fill_between(t, m - s, m + s, color=color, alpha=0.15)


def run(out_dir, params: dict) -> None:
    p = dict(params)
    seeds, lambdas = tuple(p["SEEDS"]), tuple(p["LAMBDAS"])
    lam_lo, lam_hi = min(lambdas), max(lambdas)

    jobs = ([dict(p, condition="readout", lam=p["LAM_LOW"], seed=s) for s in seeds]
            + [dict(p, condition="applied", lam=l, seed=s)
               for l in lambdas for s in seeds])
    rows: list[dict] = []
    print(f"applied BMR: phlogiston world steps to oxygen at t={p['T_SHIFT']}; "
          f"gamma={p['GAMMA']}, omega={p['OMEGA']}, bmr_every={p['BMR_EVERY']}; "
          f"{len(jobs)} jobs on {p['MAX_WORKERS']} workers")
    with ProcessPoolExecutor(max_workers=p["MAX_WORKERS"]) as pool:
        for row in pool.map(_one_job, jobs, chunksize=1):
            rows.append(row)
            extra = (f"  removed belt={row['rem_belt_final']:.2f}"
                     if row["condition"] == "applied" else "")
            print(f"  {row['condition']:>8} lam={row['lam']:<5} seed={row['seed']}: "
                  f"belt|cpl|={row['belt_cpl_final']:.3f}{extra}", flush=True)

    t = _rows(rows, condition="readout")[0]["snap_t"]
    t_shift = p["T_SHIFT"]
    read_cpl = float(_mean(rows, "belt_cpl_final", condition="readout"))
    app_cpl = float(_mean(rows, "belt_cpl_final", condition="applied", lam=p["LAM_LOW"]))
    app_cpl_hi = float(_mean(rows, "belt_cpl_final", condition="applied", lam=lam_hi))
    rem_by_lam = {l: float(_mean(rows, "rem_belt_final", condition="applied", lam=l))
                  for l in lambdas}
    rb = _mean(rows, "rem_belt_t", condition="applied", lam=p["LAM_LOW"])
    rs = _mean(rows, "rem_spur_t", condition="applied", lam=p["LAM_LOW"])
    rc = _mean(rows, "rem_core_t", condition="applied", lam=p["LAM_LOW"])
    onset = {k: (int(t[np.argmax(tr > 0.5)]) if (tr > 0.5).any() else -1)
             for k, tr in (("belt", rb), ("spurious", rs))}

    # ---- print every measured value BEFORE asserting ----
    print(f"\nfinal belt |coupling| (seed mean): readout={read_cpl:.3f}  "
          f"applied(lam={p['LAM_LOW']})={app_cpl:.3f} ({app_cpl / read_cpl:.0%})  "
          f"applied(lam={lam_hi})={app_cpl_hi:.3f}")
    print(f"removal onset (>0.5): spurious t={onset['spurious']}  belt t={onset['belt']}"
          f"  (world turns at t={t_shift})")
    print("lambda sweep, final belt removal: "
          + "  ".join(f"{l}:{rem_by_lam[l]:.2f}" for l in lambdas))
    print(f"core (agreement) edges Occam-removed at lam={p['LAM_LOW']}: {rc[-1]:.2f} "
          f"(data silent on them -- reported, figures are belt/spurious-scoped)")
    print(f"health: max|Pi|={max(r['max_Pi'] for r in rows):.0f}  "
          f"min final eigenvalue={min(r['min_eig'] for r in rows):.3f}")

    # ---- assertions (probe-calibrated; honest-findings rule: printed first) ----
    for r in _rows(rows, condition="applied", lam=p["LAM_LOW"]):
        assert r["rem_belt_final"] > 0.9, \
            f"a1: belt should be removed at lam={p['LAM_LOW']} (seed {r['seed']})"
    assert app_cpl < 0.5 * read_cpl, \
        f"a2: applied belt coupling should be < 0.5x readout ({app_cpl:.2f} vs {read_cpl:.2f})"
    assert rem_by_lam[lam_hi] < 0.1 and abs(app_cpl_hi - read_cpl) < 0.15 * read_cpl, \
        "a3: high conviction should protect the belt (no removal, coupling ~ readout)"
    rseq = [rem_by_lam[l] for l in sorted(lambdas)]
    assert all(rseq[i + 1] <= rseq[i] + 0.1 for i in range(len(rseq) - 1)), \
        f"a4: removal should be non-increasing in lambda ({np.round(rseq, 2)})"
    assert all(r["finite"] for r in rows), "a5: numerics must stay finite"
    assert all(r["min_eig"] > 0.0 for r in rows if r["condition"] == "applied"), \
        "a5: applied nets must stay positive-definite"
    assert onset["spurious"] < t_shift, \
        "a6: never-supported over-wiring should be pruned BEFORE the shift (Occam)"
    assert onset["belt"] > t_shift, \
        "a6: the supported belt should be pruned only AFTER the world turns"

    # ---- the 3 figures ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _band(ax, t, _mean(rows, "belt_cpl_t", condition="readout"),
          _std(rows, "belt_cpl_t", condition="readout"), RED,
          "read-out (flags never applied)")
    _band(ax, t, _mean(rows, "belt_cpl_t", condition="applied", lam=p["LAM_LOW"]),
          _std(rows, "belt_cpl_t", condition="applied", lam=p["LAM_LOW"]), GREEN,
          f"applied BMR (lam={p['LAM_LOW']})")
    ax.axvline(t_shift, color="gray", ls=":", lw=1.5)
    ax.annotate("world turns", (t_shift + 2, ax.get_ylim()[1] * 0.95), fontsize=8,
                color="gray")
    ax.set_xlabel("step"); ax.set_ylabel("mean belt coupling  |Pi[mass-law edge]|")
    ax.set_title("with applied BMR the stale mass-law structure actually leaves the net")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "fig_belt_coupling.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(t, rs, "--", color=GREY, lw=2.2, label="spurious over-wiring (never supported)")
    ax.plot(t, rb, "-", color=GREEN, lw=2.2, label="mass-law belt (supported until the shift)")
    ax.axvline(t_shift, color="gray", ls=":", lw=1.5)
    ax.annotate("world turns", (t_shift + 2, 0.5), fontsize=8, color="gray")
    ax.set_xlabel("step"); ax.set_ylabel("fraction of edges removed from the net")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("structure follows evidence: the unsupported wiring goes immediately,\n"
                 "the supported belt only when the world turns")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "fig_removal_timing.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(lambdas, [rem_by_lam[l] for l in lambdas],
                yerr=[_std(rows, "rem_belt_final", condition="applied", lam=l)
                      for l in lambdas],
                marker="o", lw=2.2, capsize=3, color=GREEN)
    ax.set_xscale("log")
    ax.set_xlabel("conviction lambda (value protection on the ledger dF + lam dU)")
    ax.set_ylabel("final belt removal fraction")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("conviction is the lock-in dial:\n"
                 "evidence prunes below the threshold, value protects above it")
    plt.tight_layout(); plt.savefig(out_dir / "fig_lambda_sweep.png", dpi=130)
    plt.close(fig)

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        snap_t=t,
        belt_cpl_readout=_mean(rows, "belt_cpl_t", condition="readout"),
        belt_cpl_applied=_mean(rows, "belt_cpl_t", condition="applied", lam=p["LAM_LOW"]),
        rem_belt_t=rb, rem_spur_t=rs, rem_core_t=rc,
        lambdas=np.asarray(lambdas),
        rem_belt_by_lambda=np.asarray([rem_by_lam[l] for l in lambdas]),
    )
    summary = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
        "belt_coupling_final": {"readout": read_cpl, "applied_lam_low": app_cpl,
                                "applied_lam_high": app_cpl_hi},
        "removal_onset": dict(onset, t_shift=t_shift),
        "lambda_sweep_belt_removal": {str(l): rem_by_lam[l] for l in lambdas},
        "occam_core_removal_at_lam_low": float(rc[-1]),
        "spurious_partial_regrowth": float(rs[-1]),
        "mechanism": "the crisis check executed: every BMR_EVERY steps the ledger-"
                     "flagged edges are removed by the exact Savage-Dickey reduced "
                     "posterior (joint reduced prior + untouched deposit, anchor "
                     "included); reversible -- a cleared flag restores the edge",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"\nsaved arrays / 3 figures / summary to {out_dir}")
    print(f"HEADLINE: the crisis check now has consequences. With applied BMR the stale "
          f"mass-law coupling actually leaves the net ({read_cpl:.1f} -> {app_cpl:.1f}; "
          f"read-out flags alone leave it untouched forever); removal is evidence-timed "
          f"(never-supported over-wiring at t={onset['spurious']}, the supported belt "
          f"only at t={onset['belt']} after the world turns at t={t_shift}); and "
          f"conviction is the lock-in dial -- removal falls from "
          f"{rem_by_lam[lam_lo]:.0%} to {rem_by_lam[lam_hi]:.0%} as lambda rises. "
          f"Structure is now answerable to evidence, reversibly.")


register(ExperimentSpec(
    model="phlogiston",
    name="applied_bmr",
    description="The crisis check executed: in-loop Bayesian Model Reduction that "
                "actually mutates the net (exact Savage-Dickey reduced posterior, "
                "anchor included). Stale structure leaves the net when the world "
                "turns, never-supported over-wiring goes immediately, and conviction "
                "lambda is the lock-in dial that stops the prune.",
    run=run,
    out_dir="applied_bmr",
    params=dict(
        N=48, T_SHIFT=40, N_STEPS=160, SIGMA_O=0.5,
        # CALIB (probe, seed 1): GAMMA=0.98 keeps the deposit window (~1.0) below the
        # belt prior (1.8) so the fig-1 contrast is real (readout 2.8 vs applied 1.0);
        # the residual after a prune is the KEPT deposit, by Savage-Dickey semantics.
        GAMMA=0.98, OMEGA=0.92,
        # CALIB: removal onset is cadence-insensitive over 5..20; 10 keeps cost trivial.
        BMR_EVERY=10,
        # CALIB: the protection threshold sits between lam=0.05 (full removal) and
        # lam=1.0 (zero removal); the sweep brackets it.
        LAM_LOW=0.05, LAMBDAS=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0),
        SNAPSHOT_EVERY=5, SEEDS=(1, 2, 3), MAX_WORKERS=6,
    ),
    seeds=(1, 2, 3),
    consumes=dict(figures=["fig_belt_coupling", "fig_removal_timing",
                           "fig_lambda_sweep"]),
))
