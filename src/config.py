"""Minimal configuration surface for structural production code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkConfig:
    """Graph settings used by structural graph builders."""

    kind: str = "watts_strogatz"
    mean_degree: int = 4
    rewiring_p: float = 0.1
    intra_prob: float = 0.04
    inter_prob: float = 0.003
    p_edge: float | None = None
