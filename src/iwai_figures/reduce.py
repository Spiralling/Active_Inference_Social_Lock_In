"""Raw runs -> minimal per-seed inputs for the sixteen supplied figure assets.

No seeds are selected by outcome. Completed state n is raw index n-1.
The original plot reductions and bootstrap draws are preserved exactly.
"""

from pathlib import Path
import hashlib
import json
import numpy as np

CAMERA_CONDITIONS = [
    "cycle",
    "wall_plain_original",
    "wall_gated_original",
    "wall_rivals_original",
    "network_cycle",
    "network_wheel",
    "network_community",
    "network_complete",
    "adaptation_social",
    "adaptation_isolated",
    "adaptation_reference",
    "structural_check",
]
COMMUNICATION_CONDITIONS = [
    "N00",
    "N10",
    "N01",
    "N11",
    "C00",
    "C10",
    "C01",
    "C11",
    "C10_beta50",
    "C10_omega90",
    "C10_gasvar_matched",
    "S_keep_private",
    "S_enable_weights",
]
GAUSSIAN_CONDITIONS = [
    "G_plain_info",
    "G_gated_info",
    "G_plain_moment",
    "G_gated_moment",
]
CANDIDATE_CONDITIONS = ["C10", "C11", "C10_gasvar_matched"]
CAMERA_PLOT_PROTOCOL_KEYS = (
    "shift",
    "network_seeds",
    "informed_nodes",
    "network_graphs",
    "threshold",
    "network_N",
    "network_T",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def bootstrap_indices():
    return np.random.default_rng(20260920).integers(0, 50, size=(2000, 50))


def estimate(x, idx):
    x = np.asarray(x, float)
    if x.shape != (50,) or not np.isfinite(x).all():
        raise ValueError("Expected 50 finite independent seed summaries")
    lo, hi = np.quantile(x[idx].mean(1), [0.025, 0.975])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": 50}


def save_inputs(out, groups, summary=None, provenance=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, arrays in groups.items():
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Nonfinite plot input: " + name)
        np.savez_compressed(out / (name + ".npz"), **arrays)
    if summary is not None:
        dump(out / "summary.json", summary)
    dump(
        out / "manifest.json",
        {
            "schema": 1,
            "files": {
                p.name: sha(p)
                for p in sorted(out.iterdir())
                if p.suffix in {".npz", ".json"} and p.name != "manifest.json"
            },
            "provenance": provenance or {},
            "note": "Unsmoothed per-seed inputs. No outcome-based seed selection; see figure-specific methods.",
        },
    )


def reduce_camera(raw, out):
    """Keep exactly the arrays consumed by the nine historical camera figures."""
    raw, out = Path(raw), Path(out)
    groups = {}
    sources = {}
    for name in CAMERA_CONDITIONS:
        path = raw / (name + ".npz")
        with np.load(path, allow_pickle=False) as a:
            d = {}
            if name.startswith("wall_"):
                if a["mu_map"].shape != (20, 1500, 16, 4):
                    raise ValueError("Unexpected historical camp dimensions")
                mu = a["mu_map"][..., 1:]
                half = mu.shape[2] // 2
                d["distance"] = np.linalg.norm(
                    mu[:, :, :half].mean(2) - mu[:, :, half:].mean(2), axis=-1
                )
                if name == "wall_rivals_original":
                    d["oxygen"] = a["q"][..., 1]
            elif name == "structural_check":
                if a["q"].shape != (4, 2, 200, 500, 2):
                    raise ValueError("Unexpected structural-discrimination dimensions")
                for k in [
                    "sample_counts",
                    "sigmas",
                    "observable_covariance",
                    "accuracy",
                    "product_marginals_accuracy",
                ]:
                    d[k] = a[k]
                d["q_at_sample_counts"] = a["q"][..., a["sample_counts"] - 1, :]
            else:
                expected = (
                    (50, 240, 16, 2)
                    if name == "cycle"
                    else (
                        (50, 600, 24, 2)
                        if name.startswith("network_")
                        else (20, 1500, 16, 2)
                    )
                )
                if a["q"].shape != expected:
                    raise ValueError("Unexpected camera experiment dimensions: " + name)
                d["oxygen"] = a["q"][..., 1]
                if name.startswith("adaptation_"):
                    # These metrics are displayed only over the last 375 updates.
                    d["mse_obs_tail"] = a["mse_obs"][:, -375:].mean(axis=(1, 2))
                    d["heldout_nll_tail"] = a["heldout_nll"][:, -375:].mean(axis=(1, 2))
            groups[name] = d
        sources[name] = sha(path)
    for name in ["world", "graphs"]:
        with np.load(raw / (name + ".npz"), allow_pickle=False) as a:
            groups[name] = {k: a[k] for k in a.files}
        sources[name] = sha(raw / (name + ".npz"))
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((raw / "protocol.json").read_text())
    dump(
        out / "protocol.json", {key: protocol[key] for key in CAMERA_PLOT_PROTOCOL_KEYS}
    )
    save_inputs(
        out,
        groups,
        provenance={
            "raw_sha256": sources,
            "raw_protocol_sha256": sha(raw / "protocol.json"),
            "reduction": "Only figure-used arrays; wall distances and measured structural prefixes reduced without rounding.",
        },
    )


def read_chunks(raw, name, records, keys):
    """Yield hash-verified chunks with exact coverage and seed-order validation."""
    raw = Path(raw)
    meta = json.loads((raw / name / "setup.json").read_text())
    seeds = meta["seeds"]
    S, T = len(seeds), meta["T"]
    lookup = {s: i for i, s in enumerate(seeds)}
    expected = list(range(50)) if name.startswith("N") else list(range(10000, 10050))
    if seeds != expected:
        raise ValueError("Unexpected ordered seed list: " + name)
    coverage = np.zeros((S, T), np.uint8)
    for r in sorted(records, key=lambda r: (r["start_step"], r["seeds"])):
        if r["condition"] != name:
            continue
        path = raw / r["file"]
        if sha(path) != r["sha256"]:
            raise ValueError("Raw chunk checksum mismatch: " + str(path))
        si = np.array([lookup[s] for s in r["seeds"]])
        sl = slice(r["start_step"], r["start_step"] + r["steps"])
        with np.load(path, allow_pickle=False) as a:
            values = {k: a[k] for k in keys}
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Nonfinite raw data")
        coverage[si, sl] += 1
        yield meta, si, sl, values
    if not np.all(coverage == 1):
        raise ValueError("Missing or duplicate raw seed/update coverage: " + name)


def reduce_communication(raw, out):
    """Four communication figures: probabilities, endpoint strips, errors, scores."""
    raw = Path(raw)
    records = json.loads((raw / "manifest.json").read_text())
    groups = {"common": {"bootstrap_indices": bootstrap_indices()}}
    summary = {}
    for name in COMMUNICATION_CONDITIONS:
        meta = json.loads((raw / name / "setup.json").read_text())
        T = meta["T"]
        N = meta["N"]
        if (N, T) != ((24, 600) if name.startswith("N") else (16, 5000)):
            raise ValueError("Unexpected communication dimensions")
        ids = [
            np.array(meta["group_indices"][g], int)
            for g in (
                ["fully_observing", "restricted"]
                if name.startswith("N")
                else ["gas", "mass"]
            )
        ]
        q = np.full((50, T, N), np.nan)
        v = {
            k: np.full((50, T), np.nan)
            for k in ["mse_A", "mse_B", "nll_A", "nll_B", "score_A"]
        }
        for _, si, sl, a in read_chunks(
            raw, name, records, ["q", "mse_full", "nll_full", "ell"]
        ):
            if (
                a["q"].shape[-1] != 2
                or np.any(a["q"] < 0)
                or np.any(a["q"] > 1)
                or not np.allclose(a["q"].sum(-1), 1, atol=2e-14, rtol=0)
            ):
                raise ValueError("Invalid candidate probabilities")
            q[si, sl] = a["q"][..., 1]
            for letter, group in zip("AB", ids):
                for source, key in [("mse_full", "mse"), ("nll_full", "nll")]:
                    v[key + "_" + letter][si, sl] = a[source][:, :, group].mean(2)
            scores = a["ell"][:, :, ids[0]]
            v["score_A"][si, sl] = (scores[..., 1] - scores[..., 0]).mean(2)
        d = {
            "states": np.arange(1, T + 1),
            "q_A": q[:, :, ids[0]].mean(2),
            "q_B": q[:, :, ids[1]].mean(2),
        }
        # Only quantities actually drawn are retained, with every probability update.
        if name.startswith("N"):
            for key in ["mse_A", "mse_B", "nll_A", "nll_B"]:
                d[key + "_600_tail"] = v[key][:, 450:600].mean(1)
        if name in ["C00", "C10", "C01", "C11"]:
            for h in [1500, 5000]:
                d["individual_q_" + str(h)] = q[:, h - 1]
        if name in ["C10", "C10_beta50", "C10_omega90", "C10_gasvar_matched"]:
            d["score_A_5000_tail"] = v["score_A"][:, 3750:5000].mean(1)
            for h in [1500, 5000]:
                d["q_A_" + str(h) + "_endpoint"] = d["q_A"][:, h - 1]
        groups[name] = d
        summary[name] = {
            "seeds": meta["seeds"],
            "settings": meta,
            "result": {
                "q_A_endpoint": float(d["q_A"][:, -1].mean()),
                "q_B_endpoint": float(d["q_B"][:, -1].mean()),
            },
        }
        print("Reduced communication", name, flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    protocol = json.loads((raw / "protocol.json").read_text())
    dump(
        Path(out) / "protocol.json",
        {"stationary_forecasts": protocol["stationary_forecasts"]},
    )
    save_inputs(
        out,
        groups,
        summary,
        {
            "raw_manifest_sha256": sha(raw / "manifest.json"),
            "bootstrap_seed": 20260920,
            "bootstrap_replicates": 2000,
        },
    )


def gaussian_values(raw, name, records):
    meta = json.loads((Path(raw) / name / "setup.json").read_text())
    if (meta["N"], meta["T"]) != (16, 5000):
        raise ValueError("Unexpected Gaussian-comparison dimensions")
    ids = [np.array(meta["group_indices"][g], int) for g in ["gas", "mass"]]
    v = {
        k: np.full((50, 5000), np.nan)
        for k in [
            "mixture_distance",
            "map_distance",
            "mse_A",
            "mse_B",
            "nll_A",
            "nll_B",
            "cross_A",
            "cross_B",
        ]
    }
    v.update(
        {
            k: np.full((50, 5000, 3), np.nan)
            for k in ["mixture_difference", "map_difference"]
        }
    )
    candidate = name in CANDIDATE_CONDITIONS
    if candidate:
        v.update({k: np.full((50, 5000), np.nan) for k in ["q_A", "q_B"]})
    keys = ["mu_mixture", "mu_map", "mse_full", "nll_full"] + (
        ["q"] if candidate else ["gate_crossweight"]
    )
    for _, si, sl, a in read_chunks(raw, name, records, keys):
        if candidate and (
            a["q"].shape[-1] != 2
            or np.any(a["q"] < 0)
            or np.any(a["q"] > 1)
            or not np.allclose(a["q"].sum(-1), 1, atol=2e-14, rtol=0)
        ):
            raise ValueError("Invalid candidate probabilities")
        for source, kind in [("mu_mixture", "mixture"), ("mu_map", "map")]:
            means = a[source]
            if means.shape[-2:] != (16, 3):
                raise ValueError("Expected only common observable C,G,M means")
            diff = means[:, :, ids[0]].mean(2) - means[:, :, ids[1]].mean(2)
            v[kind + "_difference"][si, sl] = diff
            v[kind + "_distance"][si, sl] = np.linalg.norm(diff, axis=-1)
        for letter, group, other in [("A", ids[0], ids[1]), ("B", ids[1], ids[0])]:
            for source, kind in [("mse_full", "mse"), ("nll_full", "nll")]:
                v[kind + "_" + letter][si, sl] = a[source][:, :, group].mean(2)
            if candidate:
                v["q_" + letter][si, sl] = a["q"][:, :, group, 1].mean(2)
                v["cross_" + letter][si, sl] = (
                    np.asarray(meta["W"])[np.ix_(group, other)].sum(1).mean()
                )
            else:
                v["cross_" + letter][si, sl] = a["gate_crossweight"][:, :, group].mean(
                    2
                )
    return meta, v


def reduce_gaussian(raw, candidate_raw, out):
    """Matched plots: all curve states and three correctly ordered distance norms."""
    raw, candidate_raw = Path(raw), Path(candidate_raw)
    idx = bootstrap_indices()
    groups = {"common": {"bootstrap_indices": idx, "states": np.arange(1, 5001)}}
    summary = {}
    for name in GAUSSIAN_CONDITIONS + CANDIDATE_CONDITIONS:
        root = raw if name in GAUSSIAN_CONDITIONS else candidate_raw
        records = json.loads((root / "manifest.json").read_text())
        meta, v = gaussian_values(root, name, records)
        keys = ["mixture_distance"]
        if name in ["C10", "C11"]:
            keys += ["map_distance", "q_A", "q_B"]
        if name.startswith("G_gated"):
            keys += ["cross_A", "cross_B"]
        if name == "C10_gasvar_matched":
            keys = []
        d = {k: v[k] for k in keys}
        metrics = {}
        for key in ["mixture_distance", "mse_A", "mse_B"]:
            seed_values = v[key][:, 3750:5000].mean(1)
            d[key + "_tail"] = seed_values
            metrics[key + "_tail"] = estimate(seed_values, idx)
        signed = v["mixture_difference"][:, 3750:5000].mean(1)
        seed_values = np.linalg.norm(signed, axis=-1)
        d["mixture_signed_time_mean"] = signed
        d["mixture_distance_of_time_means_tail"] = seed_values
        metrics["mixture_distance_of_time_means_tail"] = estimate(seed_values, idx)
        metrics["mixture_norm_of_across_seed_time_mean_difference"] = {
            "mean": float(np.linalg.norm(signed.mean(0))),
            "lo": None,
            "hi": None,
            "n": 50,
        }
        groups[name] = d
        summary[name] = {"windows": {"5000": {"metrics": metrics}}, "settings": meta}
        print("Reduced matched comparison", name, flush=True)
    save_inputs(
        out,
        groups,
        summary,
        {
            "gaussian_manifest_sha256": sha(raw / "manifest.json"),
            "candidate_manifest_sha256": sha(candidate_raw / "manifest.json"),
            "bootstrap_seed": 20260920,
            "bootstrap_replicates": 2000,
        },
    )
