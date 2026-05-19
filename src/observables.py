"""Derived measurements consumed by gating notebook + axis sweeps.

Each observable reads a History (per-step env_out arrays + snapshots) plus,
where needed, a community membership vector. All are numpy-only — these
run post-hoc, not inside the JIT'd step.

History contract:
  arrays = h.as_arrays()          # dict with 'theta_star', 'x_chosen', 'o_obs'
                                  # each shape (T,) or (T, N)
  snaps  = h.as_snapshots()       # list of dicts; each contains 'step', 'mu',
                                  # 'tau', 'gamma', 'alpha', 'beta', 'r'
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


# ----------------------------------------------------------------------
# Belief-trajectory observables
# ----------------------------------------------------------------------


def _stack_mu(snaps: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, mu) where steps: (S,), mu: (S, N)."""
    steps = np.array([s["step"] for s in snaps])
    mu = np.stack([s["mu"] for s in snaps])
    return steps, mu


def _stack_r(snaps: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, r) where r: (S, N). Empty if r not recorded."""
    if not snaps or "r" not in snaps[0]:
        return np.array([]), np.array([])
    steps = np.array([s["step"] for s in snaps])
    r = np.stack([s["r"] for s in snaps])
    return steps, r


def time_to_shift(snaps: list[dict], community_mask: np.ndarray,
                  threshold: float = 0.5, t_start: int = 0) -> int | None:
    """First snapshot step at which the community-mean μ crosses threshold.

    Returns None if it never crosses. ``t_start`` filters out earlier crossings
    (useful when measuring shift after a paradigm-event onset).
    """
    steps, mu = _stack_mu(snaps)
    if steps.size == 0:
        return None
    mu_community = mu[:, community_mask].mean(axis=1)        # (S,)
    valid = steps >= t_start
    crossed = (mu_community >= threshold) & valid
    if not crossed.any():
        return None
    return int(steps[np.argmax(crossed)])


def hysteresis_loop_area(snaps: list[dict],
                          theta_star_trace: np.ndarray,
                          community_mask: np.ndarray | None = None) -> float:
    """Signed area of the loop traced by (theta*(t), mean_mu(t)) in time-order.

    Uses the shoelace formula. theta_star_trace must have one entry per
    snapshot in ``snaps``.
    """
    steps, mu = _stack_mu(snaps)
    if steps.size == 0:
        return 0.0
    if community_mask is None:
        y = mu.mean(axis=1)
    else:
        y = mu[:, community_mask].mean(axis=1)
    x = np.asarray(theta_star_trace)
    if x.shape[0] != y.shape[0]:
        # Subsample theta_star_trace to snapshot steps.
        x = x[steps]
    n = x.shape[0]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j] - x[j] * y[i]
    return abs(area) / 2.0


def resource_weighted_belief_dist(snaps: list[dict],
                                   target: float = 1.0) -> np.ndarray:
    """Time series of sum_i (r_i/Σr) * |mu_i - target|."""
    steps_r, r = _stack_r(snaps)
    _, mu = _stack_mu(snaps)
    if r.size == 0:
        return np.zeros(mu.shape[0])
    r_share = r / (r.sum(axis=1, keepdims=True) + 1e-12)     # (S, N)
    dist = np.abs(mu - target)                                # (S, N)
    return (r_share * dist).sum(axis=1)                       # (S,)


# ----------------------------------------------------------------------
# Trust / gatekeeper observables
# ----------------------------------------------------------------------


def aggregate_trust(gamma: np.ndarray) -> np.ndarray:
    """eta_i ∝ sum_j gamma[j, i]. Returned normalised to sum 1."""
    col = gamma.sum(axis=0)
    total = col.sum() + 1e-12
    return col / total


def top_k_centrality_by_r(snap: dict, k: int = 5) -> np.ndarray:
    """Indices of the k agents with highest resource at a snapshot."""
    r = np.asarray(snap["r"])
    return np.argsort(-r)[:k]


def aggregate_trust_top_k(snap: dict, k: int = 5) -> np.ndarray:
    """Indices of the k highest-eta agents at a snapshot."""
    eta = aggregate_trust(np.asarray(snap["gamma"]))
    return np.argsort(-eta)[:k]


def endogenous_gini_trajectory(snaps: list[dict]) -> np.ndarray:
    """Per-snapshot Gini of r. Returns (S,)."""
    _, r = _stack_r(snaps)
    if r.size == 0:
        return np.array([])
    out = np.zeros(r.shape[0])
    for s, row in enumerate(r):
        abs_diff = np.abs(row[:, None] - row[None, :]).sum()
        denom = 2.0 * row.shape[0] * row.sum() + 1e-12
        out[s] = abs_diff / denom
    return out


# ----------------------------------------------------------------------
# Coupling observables
# ----------------------------------------------------------------------


def correlation_r_with_paradigm_alignment(snap: dict,
                                          target_mu: float = 1.0) -> float:
    """Pearson correlation between r_i and -|mu_i - target_mu| at one snap."""
    r = np.asarray(snap["r"])
    mu = np.asarray(snap["mu"])
    alignment = -np.abs(mu - target_mu)
    if r.std() < 1e-12 or alignment.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(r, alignment)[0, 1])


def instrumental_info_gain(snaps: list[dict],
                            world_cfg: Any) -> np.ndarray:
    """Per-snapshot mean of log(tau) — proxy for cumulative info gain.

    The Bayesian update grows tau monotonically (modulo forgetting), so
    log(tau) reads as accumulated Fisher information weighted by the agent's
    chosen experiments. Trends in this series reveal whether information
    gain is happening as a byproduct of the resource-gain policy.
    """
    steps, mu = _stack_mu(snaps)
    if steps.size == 0:
        return np.array([])
    tau = np.stack([s["tau"] for s in snaps])
    return np.log(tau + 1e-12).mean(axis=1)
