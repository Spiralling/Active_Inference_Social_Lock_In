"""DEPRECATED entry point. Use:  python scripts/run.py run beta_calibration

The diagnostic now lives in the registry (``experiments/beta_calibration.py``); its reusable
``estimate_beta_calibration`` + configs live in ``src.structural.models.landscape`` (which dissolves
the former two_stage<->beta_calibration circular import). This file is kept as a thin forwarder.
"""
from __future__ import annotations

try:                              # run case (scripts/ on path): ROOT on sys.path + utf-8 stdout
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:       # imported as scripts.beta_calibration (ROOT already on path)
    pass

from experiments.runner import run_named


def main() -> int:
    run_named("beta_calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
