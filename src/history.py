"""Tiny rollout-history accumulator for notebook ergonomics.

The library is otherwise headless; this is the one piece that makes the
``for t in range(T): pop, out = pop.step(env); h.append(out)`` loop one line.

`snap(pop, every=K)` is opt-in periodic capture of (C, gamma, q_post) so the
network-evolution visualisation can replay a rollout without re-running it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class History:
    """Append per-step env_out dicts; collate to arrays at the end.

    >>> h = History()
    >>> for t in range(T):
    ...     pop, out = pop.step(env); h.append(out); h.snap(pop, every=5)
    >>> arrays = h.as_arrays()
    >>> snaps  = h.as_snapshots()
    """

    def __init__(self) -> None:
        self._records: dict[str, list] = defaultdict(list)
        self._snapshots: list[dict[str, Any]] = []
        self._snap_step: int = 0

    def append(self, env_out: dict) -> None:
        for k, v in env_out.items():
            self._records[k].append(v)

    def snap(self, pop, every: int = 1) -> None:
        """Capture (mu, tau, gamma, alpha, beta) for the current pop state on
        every K-th call.

        ``every=1`` snaps every step; ``every=5`` every fifth, etc. Step counter
        increments on every call so the cadence is wall-clock-of-rollout, not
        wall-clock-of-process.
        """
        if every < 1 or self._snap_step % every == 0:
            snap = {
                "step": self._snap_step,
                "mu": np.asarray(pop.mu),       # (N,) posterior mean over theta
                "tau": np.asarray(pop.tau),     # (N,) posterior precision
                "gamma": np.asarray(pop.gamma), # (N, N) trust matrix
                "alpha": np.asarray(pop.alpha), # (N, N) Gamma shape
                "beta": np.asarray(pop.beta),   # (N, N) Gamma rate
            }
            r = getattr(pop, "r", None)
            if r is not None:
                snap["r"] = np.asarray(r)       # (N,) endogenous resource
            self._snapshots.append(snap)
        self._snap_step += 1

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self._records.items()}

    def as_snapshots(self) -> list[dict[str, Any]]:
        return list(self._snapshots)

    def __len__(self) -> int:
        return len(next(iter(self._records.values()), []))
