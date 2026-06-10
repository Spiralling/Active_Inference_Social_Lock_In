"""DEPRECATED entry point. Use:  python scripts/run.py cosmology_tracking

The experiment now lives in the registry (``experiments/cosmology_tracking.py``) on the
cosmology model library (``src.structural.models.cosmology``). This file is kept as a thin
forwarder plus a re-export shim, so existing imports such as

    from scripts.run_cosmology_tracking import cosmology_population, cosmology_spec,
        cosmology_graph, held_by_community, run_cell, CELLS, EPOCH_NAMES, T1, T2, N_STEPS, ...

(used by the cosmology dependent scripts and notebooks 44/45) still resolve until they are
repointed to ``src.structural.models.cosmology``.
"""
from __future__ import annotations

try:                              # run case (scripts/ on path): ROOT on sys.path + utf-8 stdout
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:       # imported as scripts.run_cosmology_tracking (ROOT already on path)
    pass

from src.structural.models.cosmology import *  # noqa: F401,F403  (dependent/notebook import shim)
from experiments.runner import run_named


def main() -> int:
    run_named("cosmology_tracking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
