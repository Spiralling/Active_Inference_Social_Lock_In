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
        """Capture (C, gamma, q_post) for the current pop state on every K-th call.

        ``every=1`` snaps every step; ``every=5`` every fifth, etc. Step counter
        increments on every call so the cadence is wall-clock-of-rollout, not
        wall-clock-of-process.
        """
        if every < 1 or self._snap_step % every == 0:
            self._snapshots.append({
                "step": self._snap_step,
                "C": np.asarray(pop.C),         # multi-feature salience prior, (N, R)
                "gamma": np.asarray(pop.gamma),
                "q_post": np.asarray(pop.q_post),
            })
        self._snap_step += 1

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self._records.items()}

    def as_snapshots(self) -> list[dict[str, Any]]:
        return list(self._snapshots)

    def __len__(self) -> int:
        return len(next(iter(self._records.values()), []))
