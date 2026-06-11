"""DEPRECATED entry point. Use:  python scripts/run.py run partial_obs_comms

The experiment now lives in the registry (``experiments/partial_obs_comms.py``) on the cosmology
model library (``src.structural.models.cosmology``). This file is kept as a thin forwarder.
"""
from __future__ import annotations

try:                              # run case (scripts/ on path): ROOT on sys.path + utf-8 stdout
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:       # imported as scripts.run_partial_obs_comms (ROOT already on path)
    pass

from experiments.runner import run_named


def main() -> int:
    run_named("partial_obs_comms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
