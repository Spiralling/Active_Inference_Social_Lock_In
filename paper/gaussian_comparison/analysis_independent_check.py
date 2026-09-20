#!/usr/bin/env python3
"""Independent raw reductions and statistical verification for the new figure.

Uses NumPy/stdlib only. It imports neither simulator nor analysis/plot helpers.
All seed/window statistics and paired intervals are recomputed from raw chunks.
All portable trajectory inputs and plotted curve means are checked; curve
intervals are independently recomputed at fixed preselected boundary states.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
NEW = ("G_plain_info", "G_gated_info", "G_plain_moment", "G_gated_moment")
OLD = ("C10", "C11", "C10_gasvar_matched")
SEEDS = list(range(10000, 10050))
WINDOWS = {"1500": (1126, 1500), "5000": (3751, 5000)}
CURVE_CHECK_STATES = (1, 2, 3, 10, 100, 250, 251, 1125, 1126, 1500, 3750, 3751, 5000)
CONTRASTS = {
    "gated_minus_plain_info": ("G_gated_info", "G_plain_info"),
    "gated_minus_plain_moment": ("G_gated_moment", "G_plain_moment"),
    "moment_minus_info_plain": ("G_plain_moment", "G_plain_info"),
    "moment_minus_info_gated": ("G_gated_moment", "G_gated_info"),
    "shared_minus_private_candidates": ("C11", "C10"),
    "gasvar_matched_minus_original_candidates": ("C10_gasvar_matched", "C10"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Check:
    def __init__(self):
        self.counts, self.maximum = {}, {}

    def same(self, actual, expected, category):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            raise AssertionError(f"{category}: unequal shapes {actual.shape} and {expected.shape}")
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError(f"{category}: nonfinite value")
        error = float(np.max(abs(actual - expected), initial=0.))
        self.maximum[category] = max(error, self.maximum.get(category, 0.))
        self.counts[category] = self.counts.get(category, 0) + actual.size
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=5e-12, err_msg=category)


def interval(x, samples):
    # Explicit indexed resampling, independent of figure's count-weight matrix.
    distribution = np.sum(x[samples], axis=1) / 50
    ends = np.percentile(distribution, [2.5, 97.5])
    return (float(np.sum(x) / 50), float(ends[0]), float(ends[1]))


def load_raw(root, condition, manifest):
    setup = json.loads((root / condition / "setup.json").read_text())
    assert setup["seeds"] == SEEDS
    assert (setup["N"], setup["T"], setup["omega"], setup["sigma_o"], setup["beta"]) == (16, 5000, .3, .5, .05)
    assert setup["group_indices"] == {"gas": list(range(8)), "mass": list(range(8, 16))}
    assert setup["fuse_nets"] is True
    assert setup["alpha_m"] == (.02 if condition == "C11" else 0.)
    assert setup["K"] == (1 if condition in NEW else 2)
    if condition in NEW:
        assert setup["gate_nu"] == (.01 if "_gated_" in condition else None)
    result = {k: np.zeros((50, 5000, 3)) for k in ("mixture_difference", "map_difference")}
    scalar = ["mixture_distance", "map_distance", "mse_A", "mse_B", "nll_A", "nll_B", "cross_A", "cross_B"]
    if condition in OLD:
        scalar += ["q_A", "q_B"]
    result.update({key: np.zeros((50, 5000)) for key in scalar})
    coverage = np.zeros((50, 5000), dtype=np.int16)
    file_count = 0
    for record in manifest:
        if record["condition"] != condition:
            continue
        path = root / record["file"]
        assert digest(path) == record["sha256"]
        first, count = record["start_step"], record["steps"]
        sl = slice(first, first + count)
        with np.load(path, allow_pickle=False) as a:
            for local, seed in enumerate(record["seeds"]):
                s = SEEDS.index(seed)
                coverage[s, sl] += 1
                for kind, source in (("mixture", "mu_mixture"), ("map", "mu_map")):
                    # Sum explicitly over the two fixed groups, not analysis helpers.
                    means = a[source][local]
                    difference = (np.sum(means[:, :8], axis=1) - np.sum(means[:, 8:], axis=1)) / 8
                    result[kind + "_difference"][s, sl] = difference
                    result[kind + "_distance"][s, sl] = np.sqrt(np.sum(difference * difference, axis=1))
                for letter, group in (("A", slice(0, 8)), ("B", slice(8, 16))):
                    for short, key in (("mse", "mse_full"), ("nll", "nll_full")):
                        result[short + "_" + letter][s, sl] = np.sum(a[key][local, :, group], axis=1) / 8
                    if condition in OLD:
                        result["q_" + letter][s, sl] = np.sum(a["q"][local, :, group, 1], axis=1) / 8
                        result["cross_" + letter][s, sl] = .05
                    else:
                        result["cross_" + letter][s, sl] = np.sum(a["gate_crossweight"][local, :, group], axis=1) / 8
        file_count += 1
    assert np.all(coverage == 1), condition + " incomplete or overlapping coverage"
    assert all(np.isfinite(a).all() for a in result.values())
    return result, file_count


def reduce_window(values, start, end):
    metrics, vectors = {}, {}
    for key, a in values.items():
        tail = a[:, start-1:end]
        avg = np.sum(tail, axis=1) / (end - start + 1)
        if a.ndim == 2:
            metrics[key + "_tail"] = avg
            metrics[key + "_endpoint"] = a[:, end-1]
        else:
            vectors[key.replace("_difference", "")] = avg
            for j, coordinate in enumerate(("C", "G", "M")):
                metrics[key + "_" + coordinate + "_tail"] = avg[:, j]
                metrics[key + "_" + coordinate + "_endpoint"] = a[:, end-1, j]
            metrics[key.replace("_difference", "") + "_distance_of_time_means_tail"] = np.sqrt(np.sum(avg * avg, axis=1))
    return metrics, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--candidate-raw", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, default=HERE / "analysis")
    parser.add_argument("--curves", type=Path, default=HERE / "figures/curve_statistics.csv")
    parser.add_argument("--output", type=Path, default=HERE / "INDEPENDENT_ANALYSIS_CHECK.json")
    parser.add_argument("--portable-only", action="store_true",
                        help="Verify reductions/figures from portable raw-reduced inputs; companion audit.py checks all raw chunks independently")
    args = parser.parse_args()
    started = time.monotonic()
    check = Check()
    samples = np.random.Generator(np.random.PCG64(20260920)).integers(0, 50, size=(2000, 50))
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (args.raw, args.candidate_raw)]
    summary = json.loads((args.analysis / "summary.json").read_text())
    contrasts = json.loads((args.analysis / "paired_contrasts.json").read_text())
    portable = np.load(args.analysis / "plot_data.npz", allow_pickle=False)
    check.same(portable["bootstrap_indices"], samples, "bootstrap_seed_indices")
    check.same(portable["states"], np.arange(1, 5001), "public_state_axis")
    seed_rows = {(r["condition"], r["horizon"], int(r["seed"])): r for r in csv.DictReader((args.analysis / "seed_metrics.csv").open())}
    diff_rows = {(r["contrast"], r["horizon"], int(r["seed"])): r for r in csv.DictReader((args.analysis / "paired_seed_differences.csv").open())}
    curve_rows = {}
    if args.curves.exists():
        for r in csv.DictReader(args.curves.open()):
            curve_rows.setdefault((r["condition"], r["metric"]), []).append(r)
    stored_metrics, stationary = {}, {}
    forecasts = json.loads((HERE / "STATIONARY_PREDICTIONS.json").read_text())["conditions"]
    total_files = 0
    for name in NEW + OLD:
        root, manifest = (args.raw, manifests[0]) if name in NEW else (args.candidate_raw, manifests[1])
        if args.portable_only:
            # Deliberately separate scope: the companion raw audit independently
            # verifies these upstream reductions. Here verify the portable
            # scientific inputs, summaries and actual plotted statistics.
            values = {key.split("::")[1]: portable[key] for key in portable.files
                      if key.startswith(name + "::") and portable[key].ndim >= 2}
            count = 0
            for kind in ("mixture", "map"):
                vector = values[kind + "_difference"]
                check.same(values[kind + "_distance"], np.sqrt(np.sum(vector * vector, axis=-1)),
                           "portable_distance_from_observable_difference")
        else:
            values, count = load_raw(root, name, manifest)
        total_files += count
        if not args.portable_only:
            for key, value in values.items():
                portable_key = name + "::" + key
                if portable_key in portable:
                    check.same(portable[portable_key], value, "complete_portable_trajectories")
        stored_metrics[name] = {}
        for horizon, (start, end) in WINDOWS.items():
            metrics, vectors = reduce_window(values, start, end)
            if args.portable_only:
                # Full MSE/NLL time series are intentionally absent from the
                # portable plotting inputs. Their per-seed windows are supplied
                # and independently checked against raw data by audit.py.
                for measure in ("mse", "nll"):
                    for group in ("A", "B"):
                        for which in ("tail", "endpoint"):
                            key = measure + "_" + group + "_" + which
                            metrics[key] = portable[name + "::" + key + "_" + horizon]
            stored_metrics[name][horizon] = metrics
            reported = summary[name]["windows"][horizon]
            assert (reported["first_state"], reported["last_state"], reported["n_states"]) == (start, end, end-start+1)
            for key, x in metrics.items():
                listed = [float(seed_rows[name, horizon, seed][key]) for seed in SEEDS]
                check.same(listed, x, "all_seed_window_metrics")
                check.same(portable[name + "::" + key + "_" + horizon], x, "portable_seed_window_metrics")
                stat = reported["metrics"][key]
                assert stat["n"] == 50
                check.same([stat[k] for k in ("mean", "lo", "hi")], interval(x, samples), "all_summary_intervals")
            for kind, vector in vectors.items():
                point = np.sqrt(np.sum((np.sum(vector, axis=0) / 50) ** 2))
                stat = reported["metrics"][kind + "_norm_of_across_seed_time_mean_difference"]
                check.same(stat["mean"], point, "descriptive_ensemble_norm_order")
                # The norm is nonregular at zero. The finalized analysis reports
                # no inferential percentile interval for this descriptive point.
                assert stat["lo"] is None and stat["hi"] is None
        for (condition, metric), rows in curve_rows.items():
            if condition != name:
                continue
            states = [int(r["completed_state"]) for r in rows]
            assert states == list(range(1, 5001))
            x = values[metric]
            check.same([float(r["mean"]) for r in rows], np.sum(x, axis=0) / 50, "all_plotted_curve_means")
            for state in CURVE_CHECK_STATES:
                check.same([float(rows[state-1][k]) for k in ("mean", "lo", "hi")],
                           interval(x[:, state-1], samples), "fixed_state_curve_intervals")
        if name in ("G_plain_info", "G_plain_moment"):
            forecast = forecasts["information" if name.endswith("info") else "moment"]
            signed = np.sum(values["mixture_difference"][:, 3750:5000], axis=1) / 1250
            moment2 = np.sum(values["mixture_distance"][:, 3750:5000] ** 2, axis=1) / 1250
            stationary[name] = {
                "gas_minus_mass_expected_vector": forecast["expected_mean_gap"],
                "measured_mean_vector": signed.mean(0).tolist(),
                "measured_coordinate_intervals": [list(interval(signed[:, i], samples)) for i in range(3)],
                "analytic_norm_of_expected_vector": forecast["norm_expected_mean_gap"],
                "measured_norm_of_mean_vector": float(np.linalg.norm(signed.mean(0))),
                "analytic_expected_squared_centroid_gap": forecast["expected_squared_centroid_gap"],
                "measured_squared_centroid_gap_interval": list(interval(moment2, samples)),
                "analytic_gas_mse": forecast["gas_expected_mse"],
                "measured_gas_mse_interval": list(interval(stored_metrics[name]["5000"]["mse_A_tail"], samples)),
                "analytic_mass_mse": forecast["mass_expected_mse"],
                "measured_mass_mse_interval": list(interval(stored_metrics[name]["5000"]["mse_B_tail"], samples)),
                "interpretation": "Stationary analytic mean vector, squared norm and MSE are compared to their matching finite late-window estimands. RMS is not compared to mean instantaneous norm. Intervals are pointwise; coverage is not a deterministic acceptance test.",
            }
        print("Independently verified", name, "portable inputs" if args.portable_only else str(count) + " raw chunks", flush=True)
    for label, (plus, minus) in CONTRASTS.items():
        assert (contrasts[label]["plus"], contrasts[label]["minus"]) == (plus, minus)
        for horizon in WINDOWS:
            pm, mm = stored_metrics[plus][horizon], stored_metrics[minus][horizon]
            for key in set(pm) & set(mm):
                difference = pm[key] - mm[key]
                listed = [float(diff_rows[label, horizon, seed][key]) for seed in SEEDS]
                check.same(listed, difference, "all_paired_seed_differences")
                stat = contrasts[label]["windows"][horizon][key]
                check.same([stat[k] for k in ("mean", "lo", "hi")], interval(difference, samples), "all_paired_contrast_intervals")
    for name in NEW:
        assert not any(key.startswith(name + "::q_") for key in portable.files)
    report = dict(status="passed", elapsed_seconds=time.monotonic()-started,
                  raw_files_verified=total_files, compared_scalar_counts=check.counts,
                  maximum_absolute_discrepancy=check.maximum,
                  conditions=list(NEW + OLD), seeds=SEEDS, windows=WINDOWS,
                  independence="NumPy/stdlib reductions without simulator, analysis or plotting imports. Explicit indexed whole-seed resampling independently checks figure count-weight intervals.",
                  input_mode="portable inputs previously checked against raw data by separate audit.py" if args.portable_only else "independent complete raw reductions",
                  curves_present=bool(curve_rows), curve_series_checked=len(curve_rows),
                  curve_interval_states=list(CURVE_CHECK_STATES),
                  scope=("Portable vector-to-distance consistency, trajectory-to-window reductions, all summary/contrast intervals and plotted curve means. Upstream raw-to-portable reductions and per-seed MSE/NLL inputs are verified by the separate raw auditor, not repeated here. " if args.portable_only else "All raw-derived seed/window scalars, portable trajectory inputs, summary/contrast intervals, plotted curve means. ") + "Curve confidence bounds checked at preselected boundary states, not every state.",
                  stationary_estimand_comparisons=stationary,
                  source_hashes={"script": digest(__file__), "new_manifest": digest(args.raw / "manifest.json"),
                      "candidate_manifest": digest(args.candidate_raw / "manifest.json"),
                      "analysis_manifest": digest(args.analysis / "analysis_manifest.json"),
                      "curve_csv": digest(args.curves) if args.curves.exists() else None})
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("PASS", json.dumps(check.counts), flush=True)


if __name__ == "__main__":
    main()
