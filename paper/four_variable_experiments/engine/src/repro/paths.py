"""Canonical repo paths -- one place that knows the directory layout."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def results_dir(name: str) -> Path:
    """``results/<name>/``, created if absent. The per-experiment output folder."""
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d
