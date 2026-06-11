"""Beta calibration diagnostic for the landscape two-stage simulation.

For each stage (utility-dominant fixed-truth, subtle-drift high-noise) estimate the ``beta_u``
scaling at which the fit and utility terms of the hypothesis score have comparable influence -- the
calibration that sets the two-stage ``beta_u`` choices. A pure diagnostic: it runs the same
simulation ingredients but only reads off fit/utility spread statistics (no trajectories saved),
recording the six derived beta values per stage.

Uses ``estimate_beta_calibration`` from ``src.structural.models.landscape`` (the same library that
dissolves the two_stage<->beta_calibration circular import).
"""
from __future__ import annotations

import json

import numpy as np

from src.structural.models.landscape import (
    SharedConfig, StageConfig, estimate_beta_calibration,
    _true_mean_fixed, _true_mean_subtle_drift)
from experiments.registry import ExperimentSpec, register

TRUE_MEAN_FNS = {"fixed": _true_mean_fixed, "subtle_drift": _true_mean_subtle_drift}
FIELDS = ("beta_equal_t0", "beta_equal_tfinal", "beta_equal_mean",
          "beta_utility_dominant", "beta_balanced", "beta_fit_dominant")


def run(out_dir, params: dict) -> None:
    shared = SharedConfig(n_agents=params["N_AGENTS"], n_steps=params["N_STEPS"],
                          seed=params["SEED"], structure_pull=params["STRUCTURE_PULL"])

    cals = {}
    summary = {}
    for sp in params["STAGES"]:
        stage = StageConfig(stage_tag=sp["tag"], sigma_obs=sp["sigma_obs"], beta_u=sp["beta_u"],
                            true_mean_fn=TRUE_MEAN_FNS[sp["true_mean"]])
        cal = estimate_beta_calibration(shared, stage)
        cals[sp["key"]] = np.array([getattr(cal, f) for f in FIELDS], dtype=np.float64)
        summary[sp["key"]] = {f: float(getattr(cal, f)) for f in FIELDS}
        print(f"{sp['key']} beta calibration: "
              + ", ".join(f"{f}={getattr(cal, f):.6f}" for f in FIELDS))

    np.savez_compressed(out_dir / "simulation_arrays.npz",
                        fields=np.array(FIELDS), stageA=cals["stageA"], stageB=cals["stageB"])
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved arrays / summary to {out_dir}")


register(ExperimentSpec(
    model="landscape",
    name="beta_calibration",
    description="Diagnostic: the fit-vs-utility beta scaling per landscape stage (the six derived "
                "beta values) that calibrates the two-stage beta_u choices.",
    run=run,
    out_dir="beta_calibration",
    params=dict(N_AGENTS=240, N_STEPS=80, SEED=0, STRUCTURE_PULL=0.18,
                STAGES=[dict(key="stageA", tag="landscape_stageA_utility_dominant",
                             sigma_obs=0.45, beta_u=4.0, true_mean="fixed"),
                        dict(key="stageB", tag="landscape_stageB_subtle_high_noise",
                             sigma_obs=2.0, beta_u=1.0, true_mean="subtle_drift")]),
    seeds=(0,),
    consumes=dict(),
))
