#!/usr/bin/env python3
"""Optionally compare fresh scientific arrays with the archived producing runs.

This audit requires the original large raw archives, which are deliberately not
part of the clean figure branch. It is not needed to simulate or plot. All
comparisons include array dtype, shape and C-order bytes. Numeric tolerances are
reported separately and never promoted to exact reproduction. Exit status is 0
only for exact reproduction, 2 for roundoff-only differences, and 1 for material
errors or an incomplete run. File paths in the JSON report are portable labels.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

RTOL = 1e-12
ATOL = 5e-12
CAMERA_CONDITIONS = (
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
)
COMMUNICATION_CONDITIONS = {
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
}
GAUSSIAN_CONDITIONS = {
    "G_plain_info",
    "G_gated_info",
    "G_plain_moment",
    "G_gated_moment",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def record_key(record):
    return (
        record["condition"],
        tuple(record["seeds"]),
        record["start_step"],
        record["steps"],
    )


def unique_index(records):
    result = {}
    for record in records:
        key = record_key(record)
        if key in result:
            raise ValueError(f"Duplicate scientific manifest key: {key}")
        result[key] = record
    return result


def require_complete(root):
    complete = root / "completion.json"
    if not complete.exists() or load_json(complete).get("status") != "complete":
        raise ValueError(f"{root.name}: completed production run required")


def compare_file(job):
    new_root, old_root, relative, old_relative, expected_new, expected_old, subset = job
    new_path, old_path = new_root / relative, old_root / old_relative
    result = dict(
        file=relative,
        archived_file=old_relative,
        arrays=0,
        elements=0,
        byte_exact_arrays=0,
        numerically_exact_arrays=0,
        roundoff_only_arrays=0,
        maximum_absolute_error=0.0,
        failures=[],
        differences=[],
        ignored_archived_arrays=[],
    )
    if not new_path.exists() or not old_path.exists():
        result["failures"].append("Required scientific NPZ file missing")
        return result
    # The producer's recorded file checksums are verified before accepting bytes.
    for label, path, expected in (
        ("new", new_path, expected_new),
        ("archive", old_path, expected_old),
    ):
        if expected is not None and sha(path) != expected:
            result["failures"].append(
                f"{label} NPZ checksum disagrees with producer manifest"
            )
    with (
        np.load(new_path, allow_pickle=False) as current,
        np.load(old_path, allow_pickle=False) as archived,
    ):
        new_keys, old_keys = set(current.files), set(archived.files)
        if new_keys - old_keys or (not subset and old_keys - new_keys):
            result["failures"].append(
                dict(
                    array_key_mismatch=True,
                    new_only=sorted(new_keys - old_keys),
                    archive_only=sorted(old_keys - new_keys),
                )
            )
        if subset:
            result["ignored_archived_arrays"] = sorted(old_keys - new_keys)
        for key in sorted(new_keys & old_keys):
            new, old = current[key], archived[key]
            result["arrays"] += 1
            result["elements"] += int(new.size)
            if new.shape != old.shape or new.dtype != old.dtype:
                result["failures"].append(
                    dict(
                        array=key,
                        new_shape=list(new.shape),
                        archive_shape=list(old.shape),
                        new_dtype=str(new.dtype),
                        archive_dtype=str(old.dtype),
                    )
                )
                continue
            numeric = np.issubdtype(new.dtype, np.number)
            floating = np.issubdtype(new.dtype, np.inexact)
            if numeric and (not np.isfinite(new).all() or not np.isfinite(old).all()):
                result["failures"].append(dict(array=key, nonfinite=True))
                continue
            exact_numeric = bool(np.array_equal(new, old))
            exact_bytes = array_sha(new) == array_sha(old)
            result["numerically_exact_arrays"] += int(exact_numeric)
            result["byte_exact_arrays"] += int(exact_bytes)
            if exact_bytes:
                continue
            error = (
                float(np.max(np.abs(new - old)))
                if floating and new.size
                else 0.0
                if exact_numeric
                else None
            )
            if error is not None:
                result["maximum_absolute_error"] = max(
                    result["maximum_absolute_error"], error
                )
            roundoff = bool(
                floating
                and np.allclose(new, old, rtol=RTOL, atol=ATOL, equal_nan=False)
            )
            detail = dict(
                array=key,
                maximum_absolute_error=error,
                numerically_exact=exact_numeric,
                within_separately_reported_tolerance=roundoff,
            )
            result["differences"].append(detail)
            if roundoff:
                result["roundoff_only_arrays"] += 1
            else:
                result["failures"].append(detail)
    return result


def collect_camera(new, old):
    require_complete(new)
    current_manifest = load_json(new / "results_manifest.json")
    archive_manifest = load_json(old / "results_manifest.json")
    if set(current_manifest) != set(CAMERA_CONDITIONS):
        raise ValueError(
            "Camera condition set differs from the eleven depicted adaptive conditions"
        )
    expected_graph_arrays = {
        prefix + graph
        for graph in ("cycle", "wheel", "community", "complete")
        for prefix in ("A_", "W_", "informed_")
    }
    with np.load(new / "graphs.npz", allow_pickle=False) as graphs:
        if set(graphs.files) != expected_graph_arrays:
            raise ValueError(
                "Camera topology archive must contain all twelve required graph arrays"
            )
    jobs = []
    for condition in CAMERA_CONDITIONS:
        a, b = current_manifest[condition], archive_manifest[condition]
        jobs.append((new, old, a["file"], b["file"], a["sha256"], b["sha256"], False))
    for name in ("world.npz", "graphs.npz", "structural_check.npz"):
        jobs.append((new, old, name, name, None, None, name == "graphs.npz"))
    return jobs, dict(
        logical_conditions=11,
        separate_static_check=True,
        ignored_archive_conditions=sorted(
            set(archive_manifest) - set(current_manifest)
        ),
        ignored_archive_graph_arrays="Recorded explicitly per graphs.npz; these belong to undisplayed controls",
    )


def collect_chunked(new, old, family):
    require_complete(new)
    current_manifest = load_json(new / "manifest.json")
    archive_manifest = load_json(old / "manifest.json")
    current, archived = unique_index(current_manifest), unique_index(archive_manifest)
    expected = (
        COMMUNICATION_CONDITIONS if family == "communication" else GAUSSIAN_CONDITIONS
    )
    if {r["condition"] for r in current_manifest} != expected:
        raise ValueError(f"{family}: required condition set differs")
    if set(current) != set(archived):
        raise ValueError(f"{family}: scientific manifest keys differ")
    pairs, hash_comparisons = {}, 0
    for key, record in current.items():
        original = archived[key]
        for field in ("observation_sha256", "evaluation_observation_sha256"):
            if record[field] != original[field]:
                raise ValueError(f"{family}: {field} differs for {key}")
            hash_comparisons += 1
        for field, checksum in (
            ("file", "sha256"),
            ("precision_file", "precision_sha256"),
        ):
            if field not in record and field not in original:
                continue
            if field not in record or field not in original:
                raise ValueError(f"{family}: precision storage schema differs")
            pair = (record[field], original[field])
            job = (
                new,
                old,
                record[field],
                original[field],
                record[checksum],
                original[checksum],
                False,
            )
            if pair in pairs and pairs[pair] != job:
                raise ValueError("Conflicting references to one raw NPZ file")
            pairs[pair] = job
    return list(pairs.values()), dict(
        logical_conditions=len(expected),
        logical_chunks=len(current),
        paired_observation_hash_checks=hash_comparisons,
        duplicate_logical_references_deduplicated=True,
        new_manifest_sha256=sha(new / "manifest.json"),
        archived_manifest_sha256=sha(old / "manifest.json"),
    )


def compare_family(family, new, old, workers):
    start = time.perf_counter()
    jobs, details = (
        collect_camera(new, old)
        if family == "camera"
        else collect_chunked(new, old, family)
    )
    results = []
    print(f"{family}: comparing {len(jobs)} distinct scientific NPZ files", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(compare_file, jobs), 1):
            results.append(result)
            if index % 100 == 0 or index == len(jobs):
                print(f"{family}: checked {index}/{len(jobs)} files", flush=True)
    failures = [r for r in results if r["failures"]]
    different = [r for r in results if r["differences"]]
    arrays = sum(r["arrays"] for r in results)
    exact = sum(r["byte_exact_arrays"] for r in results)
    status = (
        "different" if failures else "exact" if arrays == exact else "roundoff_only"
    )
    report = dict(
        status=status,
        original_directory_label=old.name,
        new_directory_label=new.name,
        files_compared=len(results),
        arrays_compared=arrays,
        elements_compared=sum(r["elements"] for r in results),
        byte_exact_arrays=exact,
        numerically_exact_arrays=sum(r["numerically_exact_arrays"] for r in results),
        roundoff_only_arrays=sum(r["roundoff_only_arrays"] for r in results),
        maximum_absolute_error=max(
            (r["maximum_absolute_error"] for r in results), default=0.0
        ),
        differing_files=different,
        failed_files=failures,
        omitted_archived_graph_arrays={
            r["file"]: r["ignored_archived_arrays"]
            for r in results
            if r["ignored_archived_arrays"]
        },
        elapsed_seconds=time.perf_counter() - start,
        **details,
    )
    print(f"{family}: {status}; {exact}/{arrays} arrays byte-identical", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-root",
        type=Path,
        required=True,
        help="Directory containing camera/, communication/ and gaussian/ fresh runs",
    )
    parser.add_argument("--camera-original", type=Path)
    parser.add_argument("--communication-original", type=Path)
    parser.add_argument("--gaussian-original", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    selected = {
        name: getattr(args, name + "_original")
        for name in ("camera", "communication", "gaussian")
    }
    selected = {name: path for name, path in selected.items() if path is not None}
    if not selected:
        parser.error("At least one original archive must be specified")
    report = dict(
        schema=1,
        comparator_sha256=sha(__file__),
        families={},
        scope="Every emitted scientific NPZ array in the selected fresh figure families; no selected seeds or time slices",
        exact_definition="Identical dtype, shape and C-order array bytes; all numeric arrays finite",
        tolerance_fallback=dict(
            rtol=RTOL,
            atol=ATOL,
            floating_arrays_only=True,
            status="reported separately, never counted as exact; exit 2",
        ),
        excluded_metadata=[
            "JSON provenance, source filenames/hashes, runtime paths and elapsed time",
            "Protocol descriptions/default output/source paths changed by the clean extraction",
            "Original camera conditions and graph arrays not displayed in the supplied figures",
        ],
        file_checksum_validation="Both producers' manifest NPZ checksums are verified where present",
        independence="Standalone NumPy raw-array comparison; does not call simulator, reducer or renderer",
    )
    for family, archive in selected.items():
        try:
            report["families"][family] = compare_family(
                family, args.new_root / family, archive, args.workers
            )
        except Exception as exc:
            # Keep machine-specific paths out of the committed report.
            message = (
                str(exc)
                .replace(str(args.new_root), "<new-root>")
                .replace(str(archive), f"<archived-{family}>")
            )
            report["families"][family] = dict(
                status="error", error_type=type(exc).__name__, message=message
            )
            print(f"{family}: ERROR {type(exc).__name__}: {message}", flush=True)
    statuses = [r["status"] for r in report["families"].values()]
    report["status"] = (
        "exact"
        if all(s == "exact" for s in statuses)
        else "roundoff_only"
        if all(s in ("exact", "roundoff_only") for s in statuses)
        else "failed"
    )
    report["all_three_families_compared"] = set(selected) == {
        "camera",
        "communication",
        "gaussian",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Overall: {report['status']}", flush=True)
    return (
        0
        if report["status"] == "exact"
        else 2
        if report["status"] == "roundoff_only"
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
