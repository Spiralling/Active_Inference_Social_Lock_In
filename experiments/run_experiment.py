"""Experiment runner: loads YAML config, expands sweep grid, runs experiments.

Supports two model types (set via model_type field in YAML):
  "simple"  — SimpleConfig / run_simple (hierarchical context model)
  "cont"    — ContConfig / run_cont (continuous lambda model)

Usage:
    python -m experiments.run_experiment experiments/configs/E1_stationary.yaml
    python -m experiments.run_experiment experiments/configs/E4_cont_lambda.yaml --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import yaml

from src.config import WorldConfig
from src.pomdp.gen_model import PomdpConfig
from src.pomdp.simple_step import SimpleConfig, run_simple
from src.pomdp.cont_step import ContConfig, run_cont


def _build_world_config(d: dict) -> WorldConfig:
    return WorldConfig(**{k: v for k, v in d.items()
                         if k in WorldConfig.__dataclass_fields__})


def _build_pomdp_config(d: dict) -> PomdpConfig:
    kw = {}
    for k, v in d.items():
        if k == "world":
            kw["world"] = _build_world_config(v)
        elif k == "x_grid":
            kw["x_grid"] = tuple(v)
        elif k == "theta_vals":
            kw["theta_vals"] = tuple(v)
        elif k == "belief_utility":
            kw["belief_utility"] = tuple(v)
        elif k == "D0" and v is not None:
            kw["D0"] = tuple(v)
        elif k in PomdpConfig.__dataclass_fields__:
            kw[k] = v
    return PomdpConfig(**kw)


def _build_simple_config(base: dict, overrides: dict, seed: int) -> SimpleConfig:
    pomdp_d = dict(base.get("pomdp", {}))
    simple_kw = {}

    for k, v in base.items():
        if k == "pomdp":
            continue
        if k in SimpleConfig.__dataclass_fields__:
            simple_kw[k] = v

    for k, v in overrides.items():
        if k in PomdpConfig.__dataclass_fields__ or k == "world":
            pomdp_d[k] = v
        elif k in SimpleConfig.__dataclass_fields__:
            simple_kw[k] = v

    simple_kw["pomdp"] = _build_pomdp_config(pomdp_d)
    simple_kw["seed"] = seed
    return SimpleConfig(**simple_kw)


def _build_cont_config(base: dict, overrides: dict, seed: int) -> ContConfig:
    pomdp_d = dict(base.get("pomdp", {}))
    cont_kw = {}

    for k, v in base.items():
        if k == "pomdp":
            continue
        if k in ContConfig.__dataclass_fields__:
            cont_kw[k] = v

    for k, v in overrides.items():
        if k in PomdpConfig.__dataclass_fields__ or k == "world":
            pomdp_d[k] = v
        elif k in ContConfig.__dataclass_fields__:
            cont_kw[k] = v

    cont_kw["pomdp"] = _build_pomdp_config(pomdp_d)
    cont_kw["seed"] = seed
    return ContConfig(**cont_kw)


def expand_sweep(sweep: dict) -> list[dict]:
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    values = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]]
              for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _build_social_mask(cfg: SimpleConfig, exp: dict) -> np.ndarray | None:
    """Build per-agent social mask from maverick config."""
    mav = exp.get("base", {}).get("maverick_fraction", 0.0)
    if mav <= 0.0:
        return None
    mav_strength = exp.get("base", {}).get("maverick_social_mask", 0.2)
    rng = np.random.RandomState(cfg.seed)
    return np.where(rng.rand(cfg.n_agents) < mav, mav_strength, 1.0)


def _build_social_mask_generic(cfg, exp: dict) -> np.ndarray | None:
    mav = exp.get("base", {}).get("maverick_fraction", 0.0)
    if mav <= 0.0:
        return None
    mav_strength = exp.get("base", {}).get("maverick_social_mask", 0.2)
    rng = np.random.RandomState(cfg.seed)
    return np.where(rng.rand(cfg.n_agents) < mav, mav_strength, 1.0)


def run_single_simple(cfg: SimpleConfig, social_mask: np.ndarray | None = None) -> dict:
    out = run_simple(cfg, social_mask_per_agent=social_mask)
    return {
        "final_mean_qB": float(out["mean_qB"][-1]),
        "final_occ_B": float(out["occ_B"][-1]),
        "mean_qB_trajectory": out["mean_qB"].tolist(),
        "theta_star_trace": out["theta_star_trace"].tolist(),
    }


def run_single_cont(cfg: ContConfig, social_mask: np.ndarray | None = None) -> dict:
    out = run_cont(cfg, social_mask_per_agent=social_mask)
    result = {
        "final_mean_qB": float(out["mean_qB"][-1]),
        "final_occ_B": float(out["occ_B"][-1]),
        "final_mean_lambda": float(out["mean_lambda"][-1]),
        "mean_qB_trajectory": out["mean_qB"].tolist(),
        "theta_star_trace": out["theta_star_trace"].tolist(),
    }
    if "mean_r" in out:
        result["final_mean_r"] = float(out["mean_r"][-1])
    return result


def main():
    parser = argparse.ArgumentParser(description="Run theory-ladenness experiments")
    parser.add_argument("config", type=str, help="Path to YAML experiment config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sweep grid without running")
    args = parser.parse_args()

    with open(args.config) as f:
        exp = yaml.safe_load(f)

    sweep_grid = expand_sweep(exp.get("sweep", {}))
    seeds = exp.get("seeds", [0])
    name = exp.get("experiment", {}).get("name", "unnamed")
    model_type = exp.get("model_type", "simple")

    print(f"Experiment: {name} (model: {model_type})")
    print(f"Sweep points: {len(sweep_grid)}, seeds: {len(seeds)}, "
          f"total runs: {len(sweep_grid) * len(seeds)}")

    if args.dry_run:
        for i, sp in enumerate(sweep_grid):
            print(f"  [{i}] {sp}")
        return

    results = []
    total = len(sweep_grid) * len(seeds)
    done = 0
    t0 = time.time()

    for sp in sweep_grid:
        for seed in seeds:
            if model_type == "cont":
                cfg = _build_cont_config(exp["base"], sp, seed)
                social_mask = _build_social_mask_generic(cfg, exp)
                result = run_single_cont(cfg, social_mask)
            else:
                cfg = _build_simple_config(exp["base"], sp, seed)
                social_mask = _build_social_mask(cfg, exp)
                result = run_single_simple(cfg, social_mask)
            result["sweep"] = sp
            result["seed"] = seed
            results.append(result)
            done += 1
            elapsed = time.time() - t0
            rate = elapsed / done
            print(f"  [{done}/{total}] sweep={sp} seed={seed} "
                  f"mean_qB={result['final_mean_qB']:.3f} "
                  f"({rate:.1f}s/run)")

    out_dir = exp.get("outputs", {}).get("dir", f"experiments/results/{name}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
