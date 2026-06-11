"""DEPRECATED entry point. Use:  python scripts/run.py run cosmology_twofield

The experiment now lives in the registry (``experiments/cosmology_twofield.py``) on the two-field
model library (``src.structural.models.cosmology_twofield``); notebook 45 imports the builders
directly from there. This file is kept as a thin forwarder.
"""
from __future__ import annotations

try:                              # run case (scripts/ on path): ROOT on sys.path + utf-8 stdout
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:       # imported as scripts.run_cosmology_twofield (ROOT already on path)
    pass

from experiments.runner import run_named


def main() -> int:
    run_named("cosmology_twofield")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
