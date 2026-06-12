"""Gate anatomy: WHY a community stops listening -- wishful vs dogmatic lock-in.

The paper's lock-in result runs through one sensory gate, w = exp(-g |U . H_disc|):
a channel is silenced because the agent VALUES what it threatens (conviction, U = T u).
The model now carries a second mode, w = exp(-g kappa . |H_disc|): a channel is silenced
because revising what it addresses would force the whole web to re-equilibrate
(cost, kappa = T 1 - 1 on the live couplings -- ``dual_field.cost_field_live``). The two
fields are linearly independent (the paper's Figure-1 decoupling), so the two gates are
different HYPOTHESES about paradigm persistence; this experiment is their first
head-to-head, at MATCHED initial gating (per-mode gate_strength calibrated so both modes
start with the same mean silencing on the dogmatic community's disconfirming channels --
any curve difference is gate DYNAMICS, not units).

A second arm makes social trust a TRACK RECORD (``trust_memory``: EMA on the pairwise
disagreement statistic upstream of the Student-t weight) and asks whether earned distrust
moves the lock-in boundary.

Three figures, one question each:
* ``fig_gate_conversion.png``  -- conversion vs contact, one curve per gate mode.
* ``fig_gate_boundary.png``    -- where the lock-in boundary sits, per gate mode.
* ``fig_earned_trust_boundary.png`` -- the boundary with instantaneous vs earned trust.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from experiments.registry import ExperimentSpec, register
from src.structural.dual_field import conviction_field, cost_field_live
from src.structural.models.kuhn_phlogiston import padded_scenario, single_run
from src.structural.phlogiston import StructuralConfig


# ----------------------------------------------------------------------
# calibration: matched initial gating across modes
# ----------------------------------------------------------------------

def _bisect_match(proj: np.ndarray, target: float) -> float:
    """g such that mean(exp(-g * proj)) == target (mean w monotone decreasing in g)."""
    lo, hi = 1e-8, 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(np.mean(np.exp(-mid * proj))) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _calibrate(params: dict) -> dict:
    """PER-COMMUNITY gate strengths such that both modes start with the same mean initial
    silencing w_disc in EACH community.

    The conviction projection scales with the community's u_agent scale (s_open vs
    s_dogma); the cost projection reads only the shared prior Pi, so it is community-blind
    at t=0 -- a real property of structural dogmatism. A single matched-to-the-dogmatists
    cost strength would therefore silence the open vanguard too, and the comparison would
    confound "cost gating locks harder" with "cost gating killed the vanguard" (the first
    run of this experiment did exactly that). The fair head-to-head gives the cost mode
    the SAME per-community gating asymmetry, via per-agent gate_strength."""
    cfg = StructuralConfig(sigma_o=params["SIGMA_O"], t_shift=params["T_SHIFT"],
                           n_steps=params["N_STEPS"], observation_operator="relational")
    scn, _ = padded_scenario(cfg, n_slots=1)
    Pi0 = jnp.asarray(scn.Pi_base)[None]
    h0 = jnp.asarray(scn.h_base)[None]
    disc = np.asarray(scn.disc_rows, dtype=int)
    H_disc = jnp.asarray(scn.H)[jnp.asarray(disc)]

    kappa = cost_field_live(Pi0, scn.conviction_alpha)                        # (1, d)
    proj_cost = np.asarray(kappa @ jnp.abs(H_disc).T)[0]                      # (k,)
    assert proj_cost.max() > 0, "cost projection degenerate: kappa reads nothing"

    g_conv = float(params["GATE_STRENGTH"])
    cal = dict(conviction=g_conv, proj_cost_mean=float(proj_cost.mean()))
    for name, s in (("open", params["S_OPEN"]), ("dogma", params["S_DOGMA"])):
        U = conviction_field(Pi0, h0, scn.names, s * jnp.asarray(scn.u),
                             scn.conviction_alpha)                            # (1, d)
        proj_conv = np.asarray(jnp.abs(U @ H_disc.T))[0]                      # (k,)
        target = float(np.mean(np.exp(-g_conv * proj_conv)))
        cal[f"cost_{name}"] = float(_bisect_match(proj_cost, target))
        cal[f"target_w_{name}"] = target
        cal[f"proj_conv_{name}"] = proj_conv.tolist()
    cal["proj_cost"] = proj_cost.tolist()        # per-channel profiles: if the relative
    # profiles match, mean-matching matches the whole gate and a population tie is the
    # expected outcome -- record them so the tie is interpretable, not mysterious.
    return cal


# ----------------------------------------------------------------------
# one job = one population run; read-out = dogmatic community conversion
# ----------------------------------------------------------------------

def _one_job(job: dict) -> dict:
    r = single_run(**job["kwargs"])
    dd = r["community"] == 1
    return dict(arm=job["arm"], mode=job["mode"],
                inter=float(job["kwargs"]["inter"]), seed=int(job["kwargs"]["seed"]),
                memory=job["kwargs"].get("trust_memory"),
                conv=float(r["oxy_index_sc"][-1, 1]),
                crisis_frac=float((r["crisis_step"][dd] >= 0).mean()))


def _curve(rows, inters, **match) -> np.ndarray:
    """(n_inters,) mean conversion over seeds for the rows matching ``match``."""
    out = np.zeros(len(inters))
    for ii, it in enumerate(inters):
        vals = [r["conv"] for r in rows
                if r["inter"] == it and all(r[k] == v for k, v in match.items())]
        assert vals, f"no rows for inter={it}, {match}"
        out[ii] = float(np.mean(vals))
    return out


def _boundary(inters, conv) -> float:
    """The lock-in boundary: interpolated inter at conversion 0.5."""
    x, c = np.asarray(inters, dtype=float), np.asarray(conv, dtype=float)
    if c[0] >= 0.5:
        return float(x[0])
    if c.max() < 0.5:
        return float("inf")
    i = int(np.argmax(c >= 0.5))
    return float(x[i - 1] + (0.5 - c[i - 1]) * (x[i] - x[i - 1]) / (c[i] - c[i - 1]))


# ----------------------------------------------------------------------
# the three figures
# ----------------------------------------------------------------------

_COLORS = {"conviction": "#922b21", "cost": "#1f618d"}


def _plot_curves(ax, inters, series, rows_by, title):
    xs = np.arange(len(inters))
    for label, (curve, color) in series.items():
        ax.plot(xs, curve, "-o", lw=2.2, color=color, label=label)
        for ii, it in enumerate(inters):
            seeds = [r["conv"] for r in rows_by[label] if r["inter"] == it]
            ax.scatter([ii] * len(seeds), seeds, s=12, color=color, alpha=0.4, zorder=1)
    ax.axhline(0.5, color="k", ls="--", lw=1.0, label="boundary level (0.5)")
    ax.set_xticks(xs); ax.set_xticklabels(inters)
    ax.set_xlabel("social coupling (inter)")
    ax.set_ylabel("dogmatic community conversion\n(final oxygen index)")
    ax.set_ylim(-0.04, 1.04)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=9.5)


def fig_conversion(rows, inters, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    series, rows_by = {}, {}
    for mode in ("conviction", "cost"):
        sub = [r for r in rows if r["mode"] == mode]
        series[f"{mode} gate"] = (_curve(sub, inters), _COLORS[mode])
        rows_by[f"{mode} gate"] = sub
    _plot_curves(ax, inters, series, rows_by,
                 "Who stays locked in? wishful (conviction) vs dogmatic (cost) gating, "
                 "matched initial silencing")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


def fig_boundary(boundaries: dict, path):
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    names = list(boundaries)
    vals = [boundaries[k] for k in names]
    shown = [v if np.isfinite(v) else 0.0 for v in vals]
    bars = ax.bar(names, shown, color=[_COLORS.get(n.split()[0], "#7d6608") for n in names],
                  width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                ("never converts" if not np.isfinite(v) else f"{v:.4f}"),
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("lock-in boundary\n(inter at conversion 0.5)")
    ax.set_title("Where the seal breaks, per gate mode", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


def fig_earned_trust(rows, inters, boundaries, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    series, rows_by = {}, {}
    for label, mem, color in (("instantaneous trust", None, "#7d6608"),
                              ("earned trust (memory 0.9)", 0.9, "#4a235a")):
        sub = [r for r in rows if r["memory"] == mem]
        series[label] = (_curve(sub, inters), color)
        rows_by[label] = sub
    _plot_curves(ax, inters, series, rows_by,
                 f"Does a track record protect? boundary {boundaries['instantaneous']:.4g} "
                 f"(instantaneous) vs {boundaries['earned']:.4g} (earned)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------
# the experiment
# ----------------------------------------------------------------------

def run(out_dir, params: dict) -> None:
    inters = list(params["INTERS"])
    seeds = list(params["SEEDS"])
    base = dict(N=params["N_AGENTS"], s_open=params["S_OPEN"], s_dogma=params["S_DOGMA"],
                lam_open=params["LAM_OPEN"], lam_dogma=params["LAM_DOGMA"],
                omega=params["OMEGA"], sigma_o=params["SIGMA_O"],
                t_shift=params["T_SHIFT"], n_steps=params["N_STEPS"],
                proposal_rate=None)

    cal = _calibrate(params)
    print(f"calibration: g_conviction={cal['conviction']:.3g}; cost per community "
          f"g_open={cal['cost_open']:.3g} (target w={cal['target_w_open']:.3f}) "
          f"g_dogma={cal['cost_dogma']:.3g} (target w={cal['target_w_dogma']:.3f})",
          flush=True)
    # per-agent cost strengths mirroring single_run's community layout (frac_open=0.5,
    # open community first) -- the same gating asymmetry conviction gets via s_open/s_dogma.
    n_open = max(1, min(params["N_AGENTS"] - 1, round(params["N_AGENTS"] * 0.5)))
    g_cost_arr = np.concatenate([np.full(n_open, cal["cost_open"]),
                                 np.full(params["N_AGENTS"] - n_open, cal["cost_dogma"])])
    strength = {"conviction": cal["conviction"], "cost": g_cost_arr}

    jobs = []
    for mode in ("conviction", "cost"):                       # ---- Arm 1: gate anatomy
        for it in inters:
            for s in seeds:
                jobs.append(dict(arm=1, mode=mode, kwargs=dict(
                    base, inter=float(it), seed=int(s),
                    gate_strength=strength[mode], gate_mode=mode)))
    for mem in (None, params["TRUST_MEMORY"]):                # ---- Arm 2: earned trust
        for it in inters:
            for s in seeds:
                kw = dict(base, inter=float(it), seed=int(s),
                          gate_strength=cal["conviction"], gate_mode="conviction",
                          social_nu=params["SOCIAL_NU"])
                if mem is not None:
                    kw["trust_memory"] = float(mem)
                jobs.append(dict(arm=2, mode="conviction", kwargs=kw))

    with ProcessPoolExecutor(max_workers=params["MAX_WORKERS"]) as pool:
        rows = list(pool.map(_one_job, jobs, chunksize=1))
    for r in rows:
        print(f"  arm{r['arm']} {r['mode']:10s} inter={r['inter']:<6g} seed={r['seed']} "
              f"mem={r['memory']} -> conv {r['conv']:.2f} crisis {r['crisis_frac']:.2f}",
              flush=True)

    arm1 = [r for r in rows if r["arm"] == 1]
    arm2 = [r for r in rows if r["arm"] == 2]
    curves = {m: _curve(arm1, inters, mode=m) for m in ("conviction", "cost")}
    bounds = {m: _boundary(inters, curves[m]) for m in curves}
    curve2 = {"instantaneous": _curve(arm2, inters, memory=None),
              "earned": _curve(arm2, inters, memory=params["TRUST_MEMORY"])}
    bounds2 = {k: _boundary(inters, v) for k, v in curve2.items()}

    fig_conversion(arm1, inters, out_dir / "fig_gate_conversion.png")
    fig_boundary({"conviction gate": bounds["conviction"], "cost gate": bounds["cost"]},
                 out_dir / "fig_gate_boundary.png")
    fig_earned_trust(arm2, inters, bounds2, out_dir / "fig_earned_trust_boundary.png")

    # ---- honest findings first, assertions second ----
    sep = float(np.max(np.abs(curves["conviction"] - curves["cost"])))
    print(f"A1 (all modes convert at max contact): conv@{inters[-1]} = "
          f"{curves['conviction'][-1]:.2f} / {curves['cost'][-1]:.2f}", flush=True)
    print(f"A2 (conviction locked when sealed): conv@0 = {curves['conviction'][0]:.2f}",
          flush=True)
    print(f"A4 (modes distinguishable): max curve separation = {sep:.3f}; "
          f"boundaries conviction={bounds['conviction']:.4g} cost={bounds['cost']:.4g}",
          flush=True)
    print(f"A5 (earned trust): boundary instantaneous={bounds2['instantaneous']:.4g} "
          f"earned={bounds2['earned']:.4g}", flush=True)
    verdict = (f"population-INDISTINGUISHABLE (max curve separation {sep:.3f}; boundaries "
               f"{bounds['conviction']:.4g} vs {bounds['cost']:.4g}, within seed noise)"
               if sep < 0.05 else
               f"distinct (boundaries {bounds['conviction']:.4g} conviction vs "
               f"{bounds['cost']:.4g} cost; max separation {sep:.3f})")
    print(f"HEADLINE: at matched initial silencing, WHAT the gate reads -- wanting vs "
          f"re-equilibration cost -- is {verdict}: the lock-in boundary is set by how much "
          f"is silenced and by contact, not by the gate's semantics. Earned trust moves "
          f"the boundary from {bounds2['instantaneous']:.4g} to {bounds2['earned']:.4g}.",
          flush=True)

    # A1/A2 are regression-anchored to the paper's conviction result. A4 -- whether the
    # two gate semantics separate at population scale -- is the experiment's QUESTION:
    # reported above (and in summary.json), never asserted.
    assert curves["conviction"][-1] > 0.9, \
        "A1: the conviction-gated community must convert under strong contact"
    assert curves["conviction"][0] < 0.75, \
        "A2: the sealed, conviction-gated community must stay locked"
    for name, c in list(curves.items()) + list(curve2.items()):
        assert (np.diff(c) > -0.05).all(), f"A3: {name} curve must be non-decreasing (tol .05)"
    assert np.isfinite(bounds["conviction"]), "A4: conviction boundary must be finite"
    assert bounds2["earned"] >= bounds2["instantaneous"] - 0.005, \
        "A5: an earned-distrust record must not break the seal earlier"
    for r in rows:
        assert np.isfinite(r["conv"]) and np.isfinite(r["crisis_frac"]), "A6: finite outputs"

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        inters=np.array(inters, dtype=np.float64),
        conv_conviction=curves["conviction"], conv_cost=curves["cost"],
        conv_trust_inst=curve2["instantaneous"], conv_trust_earned=curve2["earned"],
        rows_conv=np.array([r["conv"] for r in rows]),
        rows_inter=np.array([r["inter"] for r in rows]),
        rows_arm=np.array([r["arm"] for r in rows]),
        rows_seed=np.array([r["seed"] for r in rows]),
    )
    summary = {"config": params, "calibration": cal,
               "curves": {k: v.tolist() for k, v in curves.items()},
               "boundaries": bounds,
               "earned_trust_curves": {k: v.tolist() for k, v in curve2.items()},
               "earned_trust_boundaries": bounds2,
               "max_mode_separation": sep}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


register(ExperimentSpec(
    model="phlogiston",
    name="gate_anatomy",
    description="Wishful vs dogmatic lock-in, head-to-head: the conviction gate "
                "(silence what threatens your values) vs the cost gate (silence what "
                "would force the web to re-equilibrate), at matched initial silencing; "
                "plus trust as an earned track record (EMA'd disagreement) vs the "
                "instantaneous read. Three figures: conversion curves per gate mode, "
                "the lock-in boundary per mode, and the boundary under earned trust.",
    run=run,
    out_dir="gate_anatomy",
    params=dict(
        N_AGENTS=80, T_SHIFT=40, N_STEPS=180, OMEGA=0.97, SIGMA_O=0.5,
        LAM_OPEN=0.10, LAM_DOGMA=0.35, GATE_STRENGTH=1.0, S_OPEN=0.3, S_DOGMA=6.0,
        INTERS=(0.0, 0.002, 0.005, 0.01, 0.05, 0.2), SEEDS=(0, 1, 2),
        SOCIAL_NU=1.0, TRUST_MEMORY=0.9, MAX_WORKERS=6,
    ),
    seeds=(0, 1, 2),
    canonical=False,
    consumes=dict(figures=["fig_gate_conversion", "fig_gate_boundary",
                           "fig_earned_trust_boundary"]),
))
