"""Headless plotting helper -- call ONLY from the runner, never at module import.

The old footgun was scripts doing ``matplotlib.use("Agg")`` at import time: a
notebook that imported such a script silently lost its interactive backend and
``%matplotlib inline`` stopped rendering. Here the backend switch is a function
the runner calls once in its ``main`` -- library and experiment modules never call
it, so importing them from a notebook is always safe.
"""
from __future__ import annotations


def headless() -> None:
    """Switch matplotlib to the Agg (file-only) backend for batch script runs."""
    import matplotlib

    matplotlib.use("Agg")
