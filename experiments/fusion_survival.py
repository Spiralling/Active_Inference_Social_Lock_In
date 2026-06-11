"""Fusion decides what survives -- and the pooling rule is now a dial, not a fate.

The paper's limitations section owns that the staggered-discovery death (a lone discoverer's
oxygen coupling crushed by fusion with still-pinned peers) RIDES ON the posterior-fusion
modelling choice: the unconceived slot's pin is pooled as if it were a firmly held "no such
thing". It reads naturally as incommensurability, but "fusion schemes that exclude
unrepresented dimensions would soften the effect, and the survival threshold for a new
concept would move with them. Which pooling rule real communities implement is an empirical
question this model can only pose." This experiment implements the named alternative and
poses it quantitatively:

* ``posterior``        -- the paper's mode: the pin is a vote against the concept;
* ``posterior_masked`` -- dimension-aware fusion (``simulation._fuse_masked``): an agent
  pools each entry only over the neighbours that REPRESENT it, so a pinned peer simply has
  no opinion about the oxygen slot (and a pinned receiver keeps its pin -- a concept still
  cannot arrive through averaging, only through one's own discovery check).

What it must show (and asserts): under ``posterior`` staggered survival is ~0 at every
Poisson rate (the paper's result); under ``posterior_masked`` survival jumps to track
DISCOVERY itself -- the survival threshold moves from "critical mass in time" to "conceive
it at all". Incommensurability, on this model, is not a fact about concepts; it is a fact
about how communities pool them.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register

SURVIVE_EFF = 0.02


def _one_job(cfg: dict) -> dict:
    from src.structural.models.kuhn_phlogiston import single_run

    r = single_run(N=cfg["N"], inter=0.0, omega=cfg["OMEGA"], sigma_o=cfg["SIGMA_O"],
                   gate_strength=cfg["GATE_STRENGTH"], s_dogma=cfg["S_DOGMA"],
                   t_shift=cfg["T_SHIFT"], n_steps=cfg["N_STEPS"],
                   proposal_rate=cfg["rate"], fuse_mode=cfg["fuse_mode"],
                   seed=cfg["seed"], snapshot_every=40)
    o = r["community"] == 0
    es = r["expand_step"][o]
    return dict(
        fuse_mode=cfg["fuse_mode"], rate=cfg["rate"], seed=cfg["seed"],
        discover_frac=float((es >= 0).mean()),
        survive_frac=float((r["oxy_coupling_tn"][-1][o] > SURVIVE_EFF).mean()),
        finite=bool(np.isfinite(r["oxy_coupling_tn"]).all()),
    )


def _mci(vals):
    a = np.asarray(vals, dtype=np.float64)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else np.nan), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def fig_survival(rows, rates, path) -> dict:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styles = {"posterior": ("#922b21", "the paper's pooling: a pin is a vote against"),
              "posterior_masked": ("#1e8449", "dimension-aware: pinned peers abstain")}
    out = {}
    for mode, (color, lab) in styles.items():
        disc = [_mci([r["discover_frac"] for r in rows
                      if r["fuse_mode"] == mode and r["rate"] == rt]) for rt in rates]
        surv = [_mci([r["survive_frac"] for r in rows
                      if r["fuse_mode"] == mode and r["rate"] == rt]) for rt in rates]
        ax.errorbar(rates, [m for m, _ in surv], yerr=[c for _, c in surv], marker="s",
                    lw=2.2, capsize=3, color=color, label=f"SURVIVES | {lab}")
        ax.errorbar(rates, [m for m, _ in disc], yerr=[c for _, c in disc], marker="o",
                    lw=1.2, ls="--", capsize=3, color=color, alpha=0.6,
                    label=f"discovers | {mode}")
        out[mode] = dict(rates=list(rates), survive=[m for m, _ in surv],
                         discover=[m for m, _ in disc])
    ax.set_xscale("log")
    ax.set_xlabel(r"Poisson proposal rate $\lambda$ (staggered discovery)")
    ax.set_ylabel("fraction of the open community")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("the survival threshold moves with the pooling rule:\n"
                 "under dimension-aware fusion a staggered discovery survives")
    ax.legend(fontsize=7.5)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return out


def run(out_dir, params: dict) -> None:
    rates, seeds = params["RATES"], params["SEEDS"]
    jobs = [dict(params, fuse_mode=fm, rate=float(rt), seed=int(s))
            for fm in ("posterior", "posterior_masked") for rt in rates for s in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=params["WORKERS"]) as pool:
        for row in pool.map(_one_job, jobs, chunksize=1):
            rows.append(row)
            print(f"  {row['fuse_mode']:17s} rate={row['rate']} seed={row['seed']}: "
                  f"discover {row['discover_frac']:.2f} survive {row['survive_frac']:.2f}",
                  flush=True)
    assert all(r["finite"] for r in rows), "masked fusion must stay numerically healthy"

    info = fig_survival(rows, list(rates), out_dir / "fig_fusion_survival.png")

    surv_plain = np.mean([r["survive_frac"] for r in rows if r["fuse_mode"] == "posterior"])
    masked = [r for r in rows if r["fuse_mode"] == "posterior_masked"]
    surv_masked = np.mean([r["survive_frac"] for r in masked])
    disc_masked = np.mean([r["discover_frac"] for r in masked])
    assert surv_plain < 0.05, \
        f"posterior fusion must reproduce the staggered death (got {surv_plain:.2f})"
    assert surv_masked > 0.5, \
        f"masked fusion must let staggered discoveries survive (got {surv_masked:.2f})"
    assert surv_masked > 0.8 * disc_masked, \
        "under masked fusion, survival should track discovery itself"

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        rates=np.array(rates, dtype=np.float64),
        survive_posterior=np.array(info["posterior"]["survive"]),
        survive_masked=np.array(info["posterior_masked"]["survive"]),
        discover_posterior=np.array(info["posterior"]["discover"]),
        discover_masked=np.array(info["posterior_masked"]["discover"]),
    )
    summary = {"config": params, "fig": info,
               "survive_posterior_mean": float(surv_plain),
               "survive_masked_mean": float(surv_masked)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"HEADLINE: the staggered-discovery death is a property of the POOLING RULE, "
          f"not of discovery: posterior fusion kills every staggered concept "
          f"({surv_plain:.0%} survival) while dimension-aware fusion -- pinned peers "
          f"abstain from dimensions they do not represent -- lets survival track discovery "
          f"({surv_masked:.0%} vs {disc_masked:.0%} discovering). Which rule a real "
          f"community implements is the empirical question the paper poses; the model now "
          f"prices both sides of it.")


register(ExperimentSpec(
    model="phlogiston",
    name="fusion_survival",
    description="The pooling rule as a dial (limitations section, implemented): "
                "dimension-aware fusion that excludes unrepresented dimensions moves the "
                "concept-survival threshold from 'critical mass in time' to 'conceive it "
                "at all' -- incommensurability located in how communities pool, not in "
                "the concepts.",
    run=run,
    out_dir="fusion_survival",
    params=dict(
        N=80, T_SHIFT=40, N_STEPS=320, OMEGA=0.97, SIGMA_O=0.5,
        GATE_STRENGTH=1.0, S_DOGMA=3.0,
        # rates low enough that an unexcited posterior-fusion community never completes
        # the wake within the horizon (the staggered regime -- see hawkes_rescue)
        RATES=(0.005, 0.01, 0.02), SEEDS=(1, 2, 3, 4, 5, 6), WORKERS=6,
    ),
    seeds=(1, 2, 3, 4, 5, 6),
    canonical=False,
    consumes=dict(figures=["fig_fusion_survival"]),
))
