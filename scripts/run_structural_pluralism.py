"""DEPRECATED entry point. Use:  python scripts/run.py run structural_pluralism

The experiment now lives in the registry (``experiments/structural_pluralism.py``) -- a
self-contained structure-level companion under the phlogiston model. This file is a thin forwarder.
"""
from __future__ import annotations

try:                              # run case (scripts/ on path): ROOT on sys.path + utf-8 stdout
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:       # imported as scripts.run_structural_pluralism (ROOT already on path)
    pass

from experiments.runner import run_named


def main() -> int:
    run_named("structural_pluralism")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
