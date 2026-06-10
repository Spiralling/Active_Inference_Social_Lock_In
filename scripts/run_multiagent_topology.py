"""DEPRECATED entry point. Use:  python scripts/run.py multiagent_topology

The experiment now lives in the registry (``experiments/multiagent_topology.py``) on the
phlogiston model library (``src.structural.models.phlogiston``). This file is kept as a thin
forwarder plus a re-export shim, so existing notebook imports such as

    from scripts.run_multiagent_topology import build_substrate, BELT, run_world, ...

still resolve until the notebooks are repointed to ``src.structural.models.phlogiston``.
"""
from __future__ import annotations

try:                              # run case (scripts/ on path): ROOT on sys.path + utf-8 stdout
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:       # imported as scripts.run_multiagent_topology (ROOT already on path)
    pass

from src.structural.models.phlogiston import *  # noqa: F401,F403  (notebook import shim)
from experiments.runner import run_named


def main() -> int:
    run_named("multiagent_topology")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
