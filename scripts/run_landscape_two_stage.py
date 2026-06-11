"""DEPRECATED entry point. Use:  python scripts/run.py run landscape_two_stage

The experiment now lives in the registry (``experiments/landscape_two_stage.py``) on the landscape
model library (``src.structural.models.landscape``), which also dissolves the former
two_stage<->beta_calibration circular import. This file is kept as a thin forwarder.
"""
from __future__ import annotations

try:                              # run case (scripts/ on path): ROOT on sys.path + utf-8 stdout
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:       # imported as scripts.run_landscape_two_stage (ROOT already on path)
    pass

from experiments.runner import run_named


def main() -> int:
    run_named("landscape_two_stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
