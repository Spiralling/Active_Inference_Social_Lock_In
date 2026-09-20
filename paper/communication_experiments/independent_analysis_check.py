#!/usr/bin/env python3
"""Independent raw-data statistical reconstruction; imports no study helpers.

Recomputes group probabilities, primary/secondary occupancy, predictive losses,
first and 20-state sustained events, and network paired factorial contrasts.
This auditor reads the production manifest and actual raw arrays directly.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
WINDOWS = {"network_tail": (450, 600), "camp_1500_tail": (1125, 1500), "camp_5000_tail": (3750, 5000)}


def events(mask, start_index=0, duration=1):
    """First completely observed qualifying run; states are one-based."""
    mask = np.asarray(mask, dtype=bool)
    run = 0
    for index in range(start_index, len(mask)):
        run = run + 1 if mask[index] else 0
        if run >= duration:
            first_state = index - duration + 2
            return dict(event=True, start_state=first_state, detection_state=index + 1,
                        round_index=first_state - 1, right_censored=False, horizon_state=len(mask))
    return dict(event=False, start_state=None, detection_state=None, round_index=None,
                right_censored=True, horizon_state=len(mask))


def longest(mask):
    longest_run = current = 0
    for hit in mask:
        current = current + 1 if hit else 0
        longest_run = max(longest_run, current)
    return longest_run


def intervals(mask):
    padded = np.r_[False, np.asarray(mask, bool), False]
    changes = np.diff(padded.astype(np.int8))
    return [(int(a + 1), int(b)) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


def mean_ci(values, resamples):
    values = np.asarray(values, float)
    limits = np.quantile(values[resamples].mean(axis=1), [.025, .975])
    return dict(mean=float(values.mean()), low=float(limits[0]), high=float(limits[1]))


def wilson(passed, total):
    z = norm.ppf(.975)
    point = passed / total
    denom = 1 + z * z / total
    centre = (point + z * z / (2 * total)) / denom
    half = z / denom * np.sqrt(point * (1 - point) / total + z * z / (4 * total * total))
    return dict(numerator=int(passed), denominator=int(total), fraction=float(point),
                low=float(centre - half), high=float(centre + half))


def reconstruct(out):
    manifest = json.loads((out / "manifest.json").read_text())
    grouped = defaultdict(list)
    for record in manifest:
        grouped[record["condition"]].append(record)
    result = {}
    for name, records in sorted(grouped.items()):
        setup = json.loads((out / name / "setup.json").read_text())
        seeds = setup["seeds"]
        lookup = {seed: index for index, seed in enumerate(seeds)}
        T, N = setup["T"], setup["N"]
        q = np.empty((len(seeds), T, N), dtype=np.float64)
        mse, nll, source = (np.empty_like(q) for _ in range(3))
        coverage = np.zeros((len(seeds), T), dtype=np.int8)
        for record in records:
            start, stop = record["start_step"], record["start_step"] + record["steps"]
            indices = [lookup[seed] for seed in record["seeds"]]
            with np.load(out / record["file"]) as arrays:
                q[indices, start:stop] = arrays["q"][..., 1]
                mse[indices, start:stop] = arrays["mse_full"]
                nll[indices, start:stop] = arrays["nll_full"]
                source[indices, start:stop] = arrays["ell"][..., 1] - arrays["ell"][..., 0]
            coverage[indices, start:stop] += 1
        if not np.all(coverage == 1):
            raise AssertionError(f"Missing or repeated logical coverage in {name}")
        groups = setup["group_indices"]
        group_q = {group: q[:, :, indices].mean(axis=-1) for group, indices in groups.items()}
        family = setup["family"]
        network = family == "network"
        opposition = None if network else (group_q["gas"] < .1) & (group_q["mass"] > .9)
        strong = None if network else ((q[:, :, groups["gas"]] < .1).mean(-1) >= .9) & ((q[:, :, groups["mass"]] > .9).mean(-1) >= .9)
        windows = ["network_tail"] if network else ["camp_1500_tail", "camp_5000_tail"]
        rows = []
        for index, seed in enumerate(seeds):
            row = dict(seed=seed, windows={}, endpoint={}, events={}, individual_adoption={}, events_by_horizon={})
            for group, ids in groups.items():
                row["endpoint"][group] = dict(probability=float(group_q[group][index, -1]),
                    mse=float(mse[index, -1, ids].mean()), nll=float(nll[index, -1, ids].mean()))
                above = group_q[group][index] > .9
                event_start = 80 if network else 0
                row["events"][group] = dict(first=events(above, event_start), sustained20=events(above, event_start, 20))
                row["individual_adoption"][group] = [dict(agent=int(agent), first=events(q[index, :, agent] > .9, event_start),
                                                            sustained20=events(q[index, :, agent] > .9, event_start, 20)) for agent in ids]
            for window in windows:
                start, stop = WINDOWS[window]
                values = {}
                for group, ids in groups.items():
                    values[group] = dict(probability=float(q[index, start:stop][:, ids].mean()),
                                         mse=float(mse[index, start:stop][:, ids].mean()),
                                         nll=float(nll[index, start:stop][:, ids].mean()),
                                         score_difference=float(source[index, start:stop][:, ids].mean()),
                                         endpoint_probability=float(q[index, stop-1, ids].mean()),
                                         endpoint_mse=float(mse[index, stop-1, ids].mean()),
                                         endpoint_nll=float(nll[index, stop-1, ids].mean()),
                                         endpoint_score_difference=float(source[index, stop-1, ids].mean()))
                horizon_events = {}
                for group, ids in groups.items():
                    horizon_events[group] = {
                        "group_mean": {str(dwell): events(group_q[group][index, :stop] > .9, 80 if network else 0, dwell) for dwell in (1,20)},
                        "individual": {str(agent): {str(dwell): events(q[index, :stop, agent] > .9, 80 if network else 0, dwell) for dwell in (1,20)} for agent in ids}}
                row["events_by_horizon"][str(stop)] = horizon_events
                if not network:
                    active = opposition[index, start:stop]
                    individual = strong[index, start:stop]
                    values["opposition"] = dict(occupancy=float(active.mean()), passed=bool(active.mean() >= .9),
                                                longest=int(longest(active)), losses=int(np.sum(active[:-1] & ~active[1:])),
                                                returns=int(np.sum(~active[:-1] & active[1:])),
                                                endpoint=float(opposition[index, stop-1]),
                                                secondary_occupancy=float(individual.mean()), secondary_passed=bool(individual.mean() >= .9),
                                                secondary_longest=int(longest(individual)), secondary_losses=int(np.sum(individual[:-1] & ~individual[1:])),
                                                secondary_returns=int(np.sum(~individual[:-1] & individual[1:])), secondary_endpoint=float(strong[index, stop-1]))
                row["windows"][window] = values
            if not network:
                row["opposition_at1500"] = bool(opposition[index, 1499])
                row["opposition_intervals"] = intervals(opposition[index])
                row["loss_after1500"] = events(~opposition[index], 1500, 20)
                both_oxygen = (group_q["gas"][index] > .9) & (group_q["mass"][index] > .9)
                row["both_oxygen_after1500"] = {str(dwell): events(both_oxygen, 1500, dwell) for dwell in (1,20)}
            rows.append(row)
        result[name] = dict(family=family, seeds=seeds, rows=rows)
        print(f"RECOMPUTED {name}: {len(seeds)} seeds, {T} states", flush=True)
    return result


def aggregate(data):
    seed_count = len(next(iter(data.values()))["seeds"])
    resamples = np.random.default_rng(20260920).integers(0, seed_count, (2000, seed_count))
    summaries = {}
    for condition, result in data.items():
        summaries[condition] = {}
        for window in result["rows"][0]["windows"]:
            metrics = {}
            values = result["rows"][0]["windows"][window]
            for group in values:
                if group == "opposition":
                    metrics[group] = dict(occupancy=mean_ci([r["windows"][window][group]["occupancy"] for r in result["rows"]], resamples),
                        pass_fraction=wilson(sum(r["windows"][window][group]["passed"] for r in result["rows"]), len(result["rows"])),
                        secondary_occupancy=mean_ci([r["windows"][window][group]["secondary_occupancy"] for r in result["rows"]], resamples),
                        secondary_pass_fraction=wilson(sum(r["windows"][window][group]["secondary_passed"] for r in result["rows"]), len(result["rows"])))
                else:
                    metrics[group] = {key: mean_ci([r["windows"][window][group][key] for r in result["rows"]], resamples)
                                      for key in values[group]}
            summaries[condition][window] = metrics
    contrasts = {}
    patterns = {"01-00": {"N01": 1, "N00": -1}, "11-10": {"N11": 1, "N10": -1},
                "10-00": {"N10": 1, "N00": -1}, "11-01": {"N11": 1, "N01": -1},
                "(11-10)-(01-00)": {"N11": 1, "N10": -1, "N01": -1, "N00": 1}}
    for group in ("fully_observing", "restricted"):
        for metric in ("probability", "mse", "nll"):
            for period in ("tail", "endpoint"):
                key = metric if period == "tail" else "endpoint_" + metric
                for label, coefficients in patterns.items():
                    difference = np.zeros(seed_count)
                    for condition, coefficient in coefficients.items():
                        difference += coefficient * np.array([row["windows"]["network_tail"][group][key] for row in data[condition]["rows"]])
                    contrasts[f"{group}/{metric}/{period}/{label}"] = dict(mean_ci(difference, resamples), per_seed=difference.tolist())
    branches = {}
    for group in ("gas", "mass"):
        for metric in ("probability", "mse", "nll"):
            for period in ("tail", "endpoint"):
                key = metric if period == "tail" else "endpoint_" + metric
                enable = np.array([r["windows"]["camp_5000_tail"][group][key] for r in data["S_enable_weights"]["rows"]])
                keep = np.array([r["windows"]["camp_5000_tail"][group][key] for r in data["S_keep_private"]["rows"]])
                difference = enable - keep
                branches[f"{group}/{metric}/{period}"] = dict(mean_ci(difference, resamples), per_seed=difference.tolist())
    return dict(conditions=summaries, network_paired_contrasts=contrasts, branch_paired_contrasts=branches,
                bootstrap_seed=20260920, bootstrap_draws=2000, replication_unit="seed")


def compare_published(data, aggregates, analysis):
    """Compare independent raw reconstruction with the writing handoff tables."""
    conditions = {name: {row["seed"]: row for row in value["rows"]} for name, value in data.items()}
    counts = defaultdict(int)
    maximum = defaultdict(float)
    def number(a, b, category, identity):
        delta = abs(float(a) - float(b))
        maximum[category] = max(maximum[category], delta)
        counts[category] += 1
        if not np.isclose(float(a), float(b), atol=3e-12, rtol=3e-12):
            raise AssertionError(f"{category} {identity}: independently {a}, published {b}")
    def categorical(a, b, category, identity):
        counts[category] += 1
        if a != b:
            raise AssertionError(f"{category} {identity}: independently {a!r}, published {b!r}")
    with (analysis / "all_seed_metrics.csv").open() as stream:
        for published in csv.DictReader(stream):
            name, seed, horizon = published["condition"], int(published["seed"]), int(published["horizon"])
            row = conditions[name][seed]
            window = "network_tail" if horizon == 600 else f"camp_{horizon}_tail"
            expected = row["windows"][window]
            lo, hi = WINDOWS[window]
            for field, value in (("window_first_state", lo+1), ("window_last_state", hi), ("window_n_states", hi-lo)):
                categorical(value, int(published[field]), "windows", f"{name}/{seed}/{field}")
            group_names = ("fully_observing", "restricted") if horizon == 600 else ("gas", "mass")
            for letter, group in zip(("A", "B"), group_names):
                for their, ours in (("q", "probability"), ("mse", "mse"), ("nll", "nll"), ("score", "score_difference")):
                    number(expected[group][ours], published[f"{their}_{letter}_tail"], "seed_metrics", f"{name}/{seed}/{horizon}/{group}/{their}/tail")
                    number(expected[group]["endpoint_"+ours], published[f"{their}_{letter}_endpoint"], "seed_metrics", f"{name}/{seed}/{horizon}/{group}/{their}/endpoint")
            if horizon != 600:
                for prefix, keyprefix in (("primary", ""), ("secondary", "secondary_")):
                    value = expected["opposition"]
                    for field, key in (("opposition_tail", "occupancy"), ("opposition_endpoint", "endpoint"),
                                       ("longest_interval", "longest"), ("losses", "losses"), ("returns", "returns")):
                        number(value[keyprefix+key], published[f"{prefix}_{field}"], "opposition", f"{name}/{seed}/{horizon}/{prefix}/{field}")
                    categorical(value[keyprefix+"passed"], published[f"{prefix}_pass"] == "True", "opposition_pass", f"{name}/{seed}/{horizon}/{prefix}")
    for filename, branch in (("network_factorial_contrasts.csv", False), ("branch_paired_contrasts.csv", True)):
        with (analysis / filename).open() as stream:
            for published in csv.DictReader(stream):
                metric = "probability" if published["metric"] == "q" else published["metric"]
                key = f"{published['group']}/{metric}/{published['period']}"
                expected = aggregates["branch_paired_contrasts"][key] if branch else aggregates["network_paired_contrasts"][key + "/" + published["contrast"]]
                for ours, theirs in (("mean", "mean"), ("low", "lo"), ("high", "hi")):
                    number(expected[ours], published[theirs], "paired_bootstrap", f"{filename}/{key}/{ours}")
    summary = json.loads((analysis / "summary.json").read_text())
    for condition, condition_summary in aggregates["conditions"].items():
        network = condition.startswith("N")
        for window, window_summary in condition_summary.items():
            horizon = str(WINDOWS[window][1])
            published = summary[condition]["windows"][horizon]
            groups = ("fully_observing", "restricted") if network else ("gas", "mass")
            for letter, group in zip(("A", "B"), groups):
                for theirs, ours in (("q", "probability"), ("mse", "mse"), ("nll", "nll"), ("score", "score_difference")):
                    for period in ("tail", "endpoint"):
                        value = window_summary[group][ours if period == "tail" else "endpoint_" + ours]
                        field = f"{theirs}_{letter}_{period}"
                        for independent_key, published_key in (("mean", "mean"), ("low", "lo"), ("high", "hi")):
                            number(value[independent_key], published["metrics"][field][published_key], "summary_bootstrap", f"{condition}/{horizon}/{field}/{independent_key}")
            if not network:
                for prefix, keyprefix in (("primary", ""), ("secondary", "secondary_")):
                    value = window_summary["opposition"][keyprefix+"occupancy"]
                    for independent_key, published_key in (("mean", "mean"), ("low", "lo"), ("high", "hi")):
                        number(value[independent_key], published["metrics"][prefix+"_opposition_tail"][published_key], "occupancy_bootstrap", f"{condition}/{horizon}/{prefix}/{independent_key}")
                    value = window_summary["opposition"][keyprefix+"pass_fraction"]
                    for independent_key, published_key in (("numerator", "passed"), ("denominator", "n"), ("fraction", "proportion"), ("low", "lo"), ("high", "hi")):
                        number(value[independent_key], published[prefix+"_pass"][published_key], "wilson_interval", f"{condition}/{horizon}/{prefix}/{independent_key}")
    event_groups = defaultdict(list)
    with (analysis / "events.csv").open() as stream:
        for published in csv.DictReader(stream):
            name, seed, horizon = published["condition"], int(published["seed"]), int(published["horizon"])
            row = conditions[name][seed]
            if published["event"] == "first_sustained_nonopposition_after_branch":
                expected = row["loss_after1500"]
                categorical(row["opposition_at1500"], published["opposing_at_branch"] == "True", "branch_status", f"{name}/{seed}")
            elif published["event"] == "both_groups_oxygen_after_branch":
                expected = row["both_oxygen_after1500"][published["dwell"]]
                categorical(row["opposition_at1500"], published["opposing_at_branch"] == "True", "branch_status", f"{name}/{seed}")
            else:
                group = row["events_by_horizon"][str(horizon)][published["group"]]
                expected = group["group_mean"][published["dwell"]] if published["agent"] == "group_mean" else group["individual"][published["agent"]][published["dwell"]]
            categorical(expected["event"], published["event_observed"] == "True", "event_flags", f"{name}/{seed}/{horizon}")
            for ours, theirs in (("start_state", "event_state"), ("detection_state", "detection_state"), ("round_index", "event_round"), ("horizon_state", "censor_state")):
                actual = int(published[theirs]) if published[theirs] else None
                categorical(expected[ours], actual, "event_times", f"{name}/{seed}/{horizon}/{theirs}")
            if published["agent"] == "group_mean":
                event_groups[name, horizon, published["group"], published["event"], int(published["dwell"])].append(expected)
    with (analysis / "event_summary.csv").open() as stream:
        for published in csv.DictReader(stream):
            key = (published["condition"], int(published["horizon"]), published["group"], published["event"], int(published["dwell"]))
            expected = event_groups[key]
            observed = [event for event in expected if event["event"]]
            categorical(len(observed), int(published["event_count"]), "event_summary_counts", str(key))
            categorical(len(expected)-len(observed), int(published["censored_count"]), "event_summary_counts", str(key))
            for ours, theirs in (("start_state", "median_start_state_among_observed"), ("detection_state", "median_detection_state_among_observed"), ("round_index", "median_start_round_among_observed")):
                value = float(np.median([event[ours] for event in observed])) if observed else None
                actual = float(published[theirs]) if published[theirs] else None
                categorical(value, actual, "event_summary_medians", str(key))
    return dict(status="passed", comparisons=dict(counts), maximum_absolute_errors=dict(maximum),
                interpretation="Independent raw reconstruction agrees on every specified seed metric/window, occupancy, crossing/censoring event, and network/branch paired contrast interval.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "INDEPENDENT_ANALYSIS_RECOMPUTATION.json")
    parser.add_argument("--analysis", type=Path, default=HERE / "analysis")
    args = parser.parse_args()
    result = reconstruct(args.run)
    aggregates = aggregate(result)
    document = dict(status="independently_recomputed", manifest_sha256=hashlib.sha256((args.run / "manifest.json").read_bytes()).hexdigest(),
                    implementation="NumPy/SciPy, no imports from production or analysis modules", seed_results=result, aggregates=aggregates)
    if (args.analysis / "all_seed_metrics.csv").exists():
        comparison = compare_published(result, aggregates, args.analysis)
        document["comparison"] = comparison
        (HERE / "ANALYSIS_CHECK.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
