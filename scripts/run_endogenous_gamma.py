"""DEPRECATED entry point. Use:  python scripts/run.py endogenous_gamma

The experiment now lives in the registry (``experiments/endogenous_gamma.py``).
"""
from __future__ import annotations

try:                              # run case (scripts/ on path): ROOT on sys.path + utf-8 stdout
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:       # imported as a package (ROOT already on path)
    pass

from experiments.runner import run_named


def main() -> int:
    run_named("endogenous_gamma")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
