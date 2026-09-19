"""Provenance manifest -- *what produced a result*.

Written next to each experiment's outputs as ``manifest.json``: git SHA (+ dirty
flag), UTC timestamp, seeds, the fully-resolved params, key package versions, and
per-array hashes of the outputs. Tiny and git-tracked (the ``.npz`` arrays stay
gitignored), so any figure can be traced back to the exact code, environment, and
configuration that made it -- the gap that made month-old runs irreproducible.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from src.repro.hashing import hash_file, hash_npz
from src.repro.paths import ROOT

RUNNER_VERSION = 1
_PKGS = ("jax", "jaxlib", "numpy", "equinox", "networkx", "scipy", "matplotlib")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def git_info() -> dict:
    return {"sha": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain"))}


def env_info() -> dict:
    out = {"python": platform.python_version(), "platform": __import__("sys").platform}
    for pkg in _PKGS:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _hash_outputs(out_dir: Path) -> dict:
    out: dict = {}
    npz = out_dir / "simulation_arrays.npz"
    if npz.exists():
        out["simulation_arrays.npz"] = {"arrays": hash_npz(npz)}
    summary = out_dir / "summary.json"
    if summary.exists():
        out["summary.json"] = hash_file(summary)
    return out


def write_manifest(out_dir, *, experiment: str, params: dict,
                   seeds=None, extra: dict | None = None) -> Path:
    """Write ``<out_dir>/manifest.json`` AFTER the experiment's outputs exist.

    Hashes whatever ``simulation_arrays.npz`` / ``summary.json`` are present, so
    call this once the run has saved its files.
    """
    out_dir = Path(out_dir)
    manifest = {
        "experiment": experiment,
        "git": git_info(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seeds": list(seeds) if seeds is not None else None,
        "params": _jsonable(params),
        "env": env_info(),
        "outputs": _hash_outputs(out_dir),
        "runner_version": RUNNER_VERSION,
    }
    if extra:
        manifest.update(_jsonable(extra))
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
