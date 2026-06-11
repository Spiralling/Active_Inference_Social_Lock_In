"""DEPRECATED entry point. Use:  python scripts/run.py run cosmology_coarse_world

The experiment now lives in the registry (``experiments/cosmology_coarse_world.py``) -- a
self-contained complex-world / coarse-sensor experiment. This file is kept as a thin forwarder.
"""
from __future__ import annotations

try:                              # run case (scripts/ on path): ROOT on sys.path + utf-8 stdout
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:       # imported as scripts.run_cosmology_coarse_world (ROOT already on path)
    pass

from experiments.runner import run_named


def main() -> int:
    run_named("cosmology_coarse_world")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
