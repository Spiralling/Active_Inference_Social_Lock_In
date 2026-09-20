#!/usr/bin/env python3
"""Whole-seed analysis of the frozen communication study, without simulator imports.

Time indices in NPZ histories are zero-based rounds; public tables use completed
states (round + 1). Every summary uses all raw rounds in its declared window.
Bootstrap draws resample whole seeds and are shared across paired conditions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import xlogy
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
GROUP_NAMES = {"network": ("fully_observing", "restricted"), "camp": ("gas", "mass")}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def bootstrap_indices(n, replicates=2000, seed=20260920):
    return np.random.default_rng(seed).integers(0, n, size=(replicates, n))


def estimate(values, indices):
    x = np.asarray(values, dtype=float)
    if x.shape[0] != indices.shape[1] or not np.isfinite(x).all():
        raise ValueError("Incomplete/nonfinite seed values in bootstrap")
    boots = x[indices].mean(axis=1)
    lo, hi = np.quantile(boots, [.025, .975], axis=0)
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": len(x)}


def wilson(k, n):
    z = norm.ppf(.975)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    r = z * np.sqrt(p * (1-p) / n + z*z/(4*n*n)) / (1 + z*z/n)
    return {"passed": int(k), "n": int(n), "proportion": p,
            "lo": float(max(0., c-r)), "hi": float(min(1., c+r)), "interval": "Wilson 95%"}


def first_event(mask, start_state=1, dwell=1):
    """First fully observed run: start and detection are distinct, never imputed."""
    a = np.asarray(mask, dtype=bool)
    segment = a[start_state-1:]
    starts = np.flatnonzero(np.convolve(segment.astype(int), np.ones(dwell, int), "valid") == dwell) if len(segment) >= dwell else []
    start = int(start_state + starts[0]) if len(starts) else None
    detection = None if start is None else start + dwell - 1
    return {"event_observed": start is not None, "event_state": start,
            "event_round": None if start is None else start-1,
            "detection_state": detection,
            "detection_round": None if detection is None else detection-1,
            "censor_state": len(a), "search_start_state": start_state,
            "dwell": dwell}


def intervals(mask, first_state=1):
    """Observed true runs with right-censoring; boundary clipping is explicit."""
    a = np.asarray(mask, bool)
    padded = np.r_[False, a, False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return [{"start_state": int(s+first_state), "end_state": int(e+first_state-1),
             "length": int(e-s), "left_boundary": bool(s == 0),
             "right_censored": bool(e == len(a))} for s, e in zip(starts, ends)]


def load_condition(raw, condition, records):
    meta = json.loads((raw/condition/"setup.json").read_text())
    seeds = meta["seeds"]
    lookup = {s: i for i, s in enumerate(seeds)}
    S, T, N = len(seeds), meta["T"], meta["N"]
    family = "network" if condition.startswith("N") else "camp"
    group_names = GROUP_NAMES[family]
    groups = [np.asarray(meta["group_indices"][g], int) for g in group_names]
    values = {k: np.full((S, T), np.nan) for k in (
        "q_A", "q_B", "mse_A", "mse_B", "nll_A", "nll_B", "score_A", "score_B",
        "logodds_A", "logodds_B", "fraction_A_low", "fraction_B_high",
        "H_individual", "H_population", "disagreement", "mixture_distance", "map_distance")}
    values.update({k: np.full((S, T, 3), np.nan) for k in (
        "mixture_mean_A", "mixture_mean_B", "map_mean_A", "map_mean_B", "conditional_mean_A_0",
        "conditional_mean_A_1", "conditional_mean_B_0", "conditional_mean_B_1")})
    q_individual = np.full((S, T, N), np.nan)
    coverage = np.zeros((S, T), np.uint8)
    for row in sorted(records, key=lambda r: (r["start_step"], r["seeds"])):
        si = np.array([lookup[s] for s in row["seeds"]])
        start, steps = row["start_step"], row["steps"]
        sl = slice(start, start+steps)
        with np.load(raw/row["file"], allow_pickle=False) as a:
            q = a["q"]
            if q.shape != (len(si), steps, N, 2):
                raise ValueError(f"Unexpected q shape for {condition}: {q.shape}")
            if not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 1) or not np.allclose(q.sum(-1), 1, atol=2e-14, rtol=0):
                raise ValueError(f"Invalid probability in {condition}")
            q_individual[si, sl] = q[..., 1]
            for letter, group in zip(("A", "B"), groups):
                values[f"q_{letter}"][si, sl] = q[:, :, group, 1].mean(2)
                for source, metric in (("mse_full", "mse"), ("nll_full", "nll")):
                    values[f"{metric}_{letter}"][si, sl] = a[source][:, :, group].mean(2)
                for source, metric in (("L", "logodds"), ("ell", "score")):
                    v = a[source][:, :, group]
                    values[f"{metric}_{letter}"][si, sl] = (v[..., 1]-v[..., 0]).mean(2)
                for source, metric in (("mu_mixture", "mixture_mean"), ("mu_map", "map_mean")):
                    values[f"{metric}_{letter}"][si, sl] = a[source][:, :, group].mean(2)
                for candidate in range(2):
                    values[f"conditional_mean_{letter}_{candidate}"][si, sl] = a["mu_conditional"][:, :, group, candidate].mean(2)
            values["fraction_A_low"][si, sl] = (q[:, :, groups[0], 1] < .1).mean(2)
            values["fraction_B_high"][si, sl] = (q[:, :, groups[1], 1] > .9).mean(2)
            hi = -xlogy(q, q).sum(-1).mean(-1)
            qp = q.mean(2)
            hp = -xlogy(qp, qp).sum(-1)
            values["H_individual"][si, sl] = hi
            values["H_population"][si, sl] = hp
            values["disagreement"][si, sl] = hp-hi
            for source, metric in (("mu_mixture", "mixture_distance"), ("mu_map", "map_distance")):
                mu = a[source]
                values[metric][si, sl] = np.linalg.norm(mu[:, :, groups[0]].mean(2)-mu[:, :, groups[1]].mean(2), axis=-1)
        coverage[si, sl] += 1
    if not np.all(coverage == 1):
        raise ValueError(f"Missing/overlapping rounds in {condition}: {np.unique(coverage, return_counts=True)}")
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError(f"Nonfinite derived metric in {condition}")
    values["primary_opposition"] = (values["q_A"] < .1) & (values["q_B"] > .9)
    values["secondary_opposition"] = (values["fraction_A_low"] >= .9) & (values["fraction_B_high"] >= .9)
    values["community_logodds"] = (len(groups[0])*values["logodds_A"]+len(groups[1])*values["logodds_B"])/N
    values["logodds_gap"] = values["logodds_B"]-values["logodds_A"]
    return meta, values, q_individual, group_names, groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, default=HERE/"analysis")
    a = p.parse_args()
    raw, out = a.raw.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE/"protocol.json").read_text())
    manifest = json.loads((raw/"manifest.json").read_text())
    condition_names = [c["name"] for c in protocol["conditions"]] + protocol["branch_conditions"]
    if set(r["condition"] for r in manifest) != set(condition_names):
        raise ValueError("Manifest does not contain exactly all frozen conditions")
    idx = bootstrap_indices(len(protocol["camp_seeds"]), protocol["analysis"]["bootstrap_replicates"], protocol["analysis"]["seed"])
    seed_rows, events, episodes, forecasts = [], [], {}, []
    summary, plot = {}, {"bootstrap_indices": idx}
    seed_metrics = {}
    for name in condition_names:
        meta, v, qi, group_names, groups = load_condition(raw, name, [r for r in manifest if r["condition"] == name])
        seeds, T = meta["seeds"], meta["T"]
        if len(seeds) != idx.shape[1]:
            raise ValueError("Seed count differs from frozen paired bootstrap")
        network = name.startswith("N")
        expected_seeds = protocol["network_seeds"] if network else protocol["camp_seeds"]
        if seeds != expected_seeds:
            raise ValueError(f"Frozen ordered seed list changed for {name}")
        windows = {600: (450, 600)} if network else {1500: (1125, 1500), 5000: (3750, 5000)}
        cs = {"seeds": seeds, "group_names": group_names, "opposition_applicable": not network, "windows": {}, "settings": {
            k: meta[k] for k in ("N", "T", "omega", "alpha_m", "sigma_o", "fuse_nets", "beta", "reference_variant")}}
        if name.startswith("S_"):
            cs["opposing_at_branch_state_1500"] = int(v["primary_opposition"][:, 1499].sum())
            cs["individual_opposing_at_branch_state_1500"] = int(v["secondary_opposition"][:, 1499].sum())
        seed_metrics[name] = {}
        for horizon, (start, end) in windows.items():
            rows = []
            for s, seed in enumerate(seeds):
                row = {"condition": name, "seed": seed, "horizon": horizon,
                       "window_first_state": start+1, "window_last_state": end,
                       "window_n_states": end-start}
                for metric, x in v.items():
                    if x.ndim == 2:
                        row[metric+"_tail"] = float(x[s, start:end].mean())
                        row[metric+"_endpoint"] = float(x[s, end-1])
                    else:
                        for d, coord in enumerate(("C", "G", "M")):
                            row[f"{metric}_{coord}_tail"] = float(x[s, start:end, d].mean())
                            row[f"{metric}_{coord}_endpoint"] = float(x[s, end-1, d])
                for criterion in ("primary", "secondary"):
                    mask = v[criterion+"_opposition"][s, start:end]
                    iv = intervals(mask, start+1)
                    row[criterion+"_pass"] = bool(mask.mean() >= .9)
                    row[criterion+"_longest_interval"] = max((z["length"] for z in iv), default=0)
                    row[criterion+"_losses"] = int(np.count_nonzero(mask[:-1] & ~mask[1:]))
                    row[criterion+"_returns"] = int(np.count_nonzero(~mask[:-1] & mask[1:]))
                for kind in ("mixture", "map"):
                    diff = v[f"{kind}_mean_A"][s, start:end].mean(0)-v[f"{kind}_mean_B"][s, start:end].mean(0)
                    row[kind+"_distance_of_time_means_tail"] = float(np.linalg.norm(diff))
                    for d, coord in enumerate(("C", "G", "M")):
                        row[f"{kind}_signed_difference_{coord}_tail"] = float(diff[d])
                rows.append(row)
            seed_rows.extend(rows)
            seed_metrics[name][horizon] = rows
            scalar_keys = [k for k in rows[0] if k not in ("condition", "seed", "horizon", "window_first_state", "window_last_state", "window_n_states")]
            ws = {"window_first_state": start+1, "window_last_state": end,
                  "metrics": {k: estimate([r[k] for r in rows], idx) for k in scalar_keys}}
            ws["primary_pass"] = wilson(sum(r["primary_pass"] for r in rows), len(rows))
            ws["secondary_pass"] = wilson(sum(r["secondary_pass"] for r in rows), len(rows))
            cs["windows"][str(horizon)] = ws
            for s, seed in enumerate(seeds):
                search_start = 81 if network else 1
                for letter, gname, group in zip(("A", "B"), group_names, groups):
                    for dwell in (1, 20):
                        events.append(dict(condition=name, seed=seed, horizon=horizon, group=gname, agent="group_mean",
                                           event="oxygen_adoption", **first_event(v[f"q_{letter}"][s, :horizon] > .9, search_start, dwell)))
                        for agent in group:
                            events.append(dict(condition=name, seed=seed, horizon=horizon, group=gname, agent=int(agent),
                                               event="oxygen_adoption", **first_event(qi[s, :horizon, agent] > .9, search_start, dwell)))
                if name.startswith("S_") and horizon == 5000:
                    ev = first_event(~v["primary_opposition"][s, :horizon], 1501, 20)
                    events.append(dict(condition=name, seed=seed, horizon=horizon, group="both", agent="group_mean",
                                       event="first_sustained_nonopposition_after_branch", opposing_at_branch=bool(v["primary_opposition"][s, 1499]), **ev))
                    both_oxygen = (v["q_A"][s, :horizon] > .9) & (v["q_B"][s, :horizon] > .9)
                    for dwell in (1, 20):
                        events.append(dict(condition=name, seed=seed, horizon=horizon, group="both", agent="group_mean",
                                           event="both_groups_oxygen_after_branch", opposing_at_branch=bool(v["primary_opposition"][s, 1499]),
                                           **first_event(both_oxygen, 1501, dwell)))
            if name in protocol["stationary_forecasts"]:
                for j, (letter, gname) in enumerate(zip(("A", "B"), group_names)):
                    es = ws["metrics"][f"score_{letter}_tail"]
                    forecasts.append(dict(condition=name, horizon=horizon, group=gname,
                                          frozen_stationary_forecast=protocol["stationary_forecasts"][name][j], **es))
            plot[f"{name}::individual_q_{horizon}"] = qi[:, horizon-1]
            for metric in ("q_A", "q_B", "mse_A", "mse_B", "nll_A", "nll_B", "score_A", "score_B", "primary_opposition", "secondary_opposition"):
                for endpoint in ("tail", "endpoint"):
                    plot[f"{name}::{metric}_{horizon}_{endpoint}"] = np.array([r[metric+"_"+endpoint] for r in rows])
        for s, seed in enumerate(seeds):
            episodes[f"{name}/{seed}"] = {"primary": intervals(v["primary_opposition"][s]),
                                           "secondary": intervals(v["secondary_opposition"][s])}
        # q curves retain every completed state, all seeds. Auxiliary curves use
        # a fixed display grid only; all tables above use every raw observation.
        display = np.unique(np.r_[np.arange(0, T, 10), np.array([0, 79, 80, 1499, 1500, 1599, 3750, T-1])])
        display = display[(display >= 0) & (display < T)]
        plot[f"{name}::states"] = np.arange(1, T+1)
        plot[f"{name}::seeds"] = np.array(seeds)
        plot[f"{name}::aux_states"] = display+1
        for metric in ("q_A", "q_B", "primary_opposition", "secondary_opposition"):
            plot[f"{name}::{metric}"] = v[metric]
        for metric, x in v.items():
            if metric not in ("q_A", "q_B", "primary_opposition", "secondary_opposition"):
                plot[f"{name}::{metric}"] = x[:, display]
        fixed = 0 if network else seeds.index(protocol["figure_rule"]["illustrative_seed"])
        plot[f"{name}::illustrative_seed"] = np.array(seeds[fixed])
        plot[f"{name}::illustrative_individual_q"] = qi[fixed]
        summary[name] = cs
        print(f"Analyzed {name}: {len(seeds)} seeds, all {T} states", flush=True)

    contrasts = []
    equations = {"01-00": {"N01": 1, "N00": -1}, "11-10": {"N11": 1, "N10": -1},
                 "10-00": {"N10": 1, "N00": -1}, "11-01": {"N11": 1, "N01": -1},
                 "(11-10)-(01-00)": {"N11": 1, "N10": -1, "N01": -1, "N00": 1}}
    for contrast, terms in equations.items():
        for letter, group in zip(("A", "B"), GROUP_NAMES["network"]):
            for metric in ("q", "mse", "nll"):
                for period in ("tail", "endpoint"):
                    key = f"{metric}_{letter}_{period}"
                    d = sum(coef * np.array([r[key] for r in seed_metrics[name][600]]) for name, coef in terms.items())
                    contrasts.append(dict(contrast=contrast, group=group, metric=metric, period=period, horizon=600, **estimate(d, idx)))
                    plot[f"factorial::{contrast}::{metric}_{letter}_{period}"] = d
    branches = []
    for letter, group in zip(("A", "B"), GROUP_NAMES["camp"]):
        for metric in ("q", "mse", "nll"):
            for period in ("tail", "endpoint"):
                key = f"{metric}_{letter}_{period}"
                d = np.array([r[key] for r in seed_metrics["S_enable_weights"][5000]])-np.array([r[key] for r in seed_metrics["S_keep_private"][5000]])
                branches.append(dict(contrast="enable_minus_keep", group=group, metric=metric, period=period, horizon=5000, **estimate(d, idx)))
                plot[f"branch::{metric}_{letter}_{period}"] = d
    grouped = {}
    for e in events:
        if e["agent"] != "group_mean":
            continue
        key = (e["condition"], e["horizon"], e["group"], e["event"], e["dwell"])
        grouped.setdefault(key, []).append(e)
    event_summary = []
    for (condition, horizon, group, event, dwell), es in grouped.items():
        observed = [e for e in es if e["event_observed"]]
        event_summary.append(dict(condition=condition, horizon=horizon, group=group, event=event, dwell=dwell,
                                  event_count=len(observed), censored_count=len(es)-len(observed),
                                  median_start_state_among_observed=float(np.median([e["event_state"] for e in observed])) if observed else None,
                                  median_detection_state_among_observed=float(np.median([e["detection_state"] for e in observed])) if observed else None,
                                  median_start_round_among_observed=float(np.median([e["event_round"] for e in observed])) if observed else None))
    write_csv(out/"all_seed_metrics.csv", seed_rows)
    write_csv(out/"network_factorial_contrasts.csv", contrasts)
    write_csv(out/"branch_paired_contrasts.csv", branches)
    write_csv(out/"stationary_forecast_comparison.csv", forecasts)
    write_csv(out/"events.csv", events)
    write_csv(out/"event_summary.csv", event_summary)
    dump(out/"opposition_episodes.json", episodes)
    dump(out/"summary.json", summary)
    np.savez_compressed(out/"plot_data.npz", **plot)
    dump(out/"analysis_provenance.json", {
        "raw_root": str(raw), "manifest_sha256": sha(raw/"manifest.json"),
        "protocol_sha256": sha(HERE/"protocol.json"), "analysis_script_sha256": sha(__file__),
        "conditions": condition_names, "seed_condition_count": sum(len(s["seeds"]) for s in summary.values()),
        "seed_window_rows": len(seed_rows), "raw_rounds": "All raw rounds used in every statistical window",
        "states": "Completed state n equals raw round t+1; windows are inclusive in public tables",
        "bootstrap": {"replicates": 2000, "seed": 20260920, "unit": "whole independent seed",
                      "pairing": "Same resample indices across every arm; paired differences formed before bootstrap",
                      "bands": "pointwise 95% percentile intervals of means, not simultaneous or individual dispersion"},
        "pass_interval": "Wilson 95%", "endpoint_vs_window": "Reported separately; no endpoint substituted for occupancy",
        "display": "Every-round group q retained; auxiliary curves fixed stride10 plus boundaries; no smoothing",
        "illustrative_seed": 10000, "network_illustrative_seed": 0,
        "event_censoring": "Non-events null; horizon is censoring time, not event; sustained start and detection separately recorded",
        "branch_loss": "First 20 nonopposing states after1500 across all50seeds; does not imply a transition from opposition for a seed already nonopposing",
        "secondary_distances": "Mixture/MAP groupmean differences use observable C,G,M only, never compare candidate hidden P/O",
        "forecasts": "Frozen stationary score predictions; not exact finite-trajectory/adoption targets",
        "selection": "All frozen conditions and seeds retained; no outcome-based figure seed selection",
        "outputs": {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name != "analysis_provenance.json"}})
    print(f"Completed {len(summary)} conditions; {len(seed_rows)} seed-window records", flush=True)


if __name__ == "__main__":
    main()
