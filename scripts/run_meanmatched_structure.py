"""DEPRECATED entry point. Use:  python scripts/run.py meanmatched_structure

The experiment now lives in the registry (``experiments/meanmatched_structure.py``).
"""
from __future__ import annotations

try:                              # run case (scripts/ on path): ROOT on sys.path + utf-8 stdout
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:       # imported as a package (ROOT already on path)
    pass

from experiments.runner import run_named


def main() -> int:
    run_named("meanmatched_structure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
