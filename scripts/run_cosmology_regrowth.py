"""DEPRECATED entry point. Use:  python scripts/run.py run cosmology_regrowth

The experiment now lives in the registry (``experiments/cosmology_regrowth.py``) on the regrowth
model library (``src.structural.models.cosmology_regrowth``); ``cosmology_poisson`` and notebooks
44/45 import the reusable builders directly from there. This file is kept as a thin forwarder.
"""
from __future__ import annotations

try:                              # run case (scripts/ on path): ROOT on sys.path + utf-8 stdout
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:       # imported as scripts.run_cosmology_regrowth (ROOT already on path)
    pass

from experiments.runner import run_named


def main() -> int:
    run_named("cosmology_regrowth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
