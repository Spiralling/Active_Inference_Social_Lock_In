"""Shell decomposition: the conservatism gradient ``lambda`` and per-shell order params.

The paper's central empirical signature (Fig. 6) is that reorganisation is *graded*:
cheap **belt** commitments move early, the densely-connected **core** moves last, so
the order parameter ``m(t)`` is a staircase rather than a single cascade. To *see* that
we have to split the population's progress by structural position. This module supplies
the two ingredients, on host numpy (post-hoc read-offs, never part of the dynamics):

  1. A *conservatism* value per unit (node or agent), the thing that orders the shells.
       * ``node_conservatism``  : per-NODE ``lambda_v`` -- the carry-over fill-in mass
         of node ``v`` (Eq. 5, ``observables.carryover_mass``). This is the paper's
         literal gradient (Fig. 3): high in the core hub, low on the belt.
       * ``agent_conservatism`` : per-AGENT structural stiffness -- a stiff/entrenched
         or graph-central agent reorganises last. Two readings are offered (prior
         precision mass; trust-graph centrality), because the "core vs belt" of the
         *lock-in & flip* story (nb27) is an agent-level gradient.
  2. ``assign_shells`` buckets those values into belt/mid/core, and ``shell_curves``
     averages a per-unit trajectory within each shell -> the decomposed staircase.

Honest note (see ``notes/unified_architecture.md`` Sec. 4): the 10-node phlogiston
scenario only *flips* on the 3-node mass-law block, which is a single tight cluster, so
the per-NODE staircase is shallow by construction -- whether it appears is a finding to
report, not to force. The per-AGENT decomposition (entrenched bloc vs periphery) is the
richer staircase for the lock-in experiments.
"""

from __future__ import annotations

import numpy as np

from src.structural import observables as obs
from src.structural import phlogiston as ph
from src.structural.phlogiston import StructuralConfig

DEFAULT_LABELS = ("belt", "mid", "core")


# ----------------------------------------------------------------------
# 1a. per-NODE conservatism lambda_v (Eq. 5)
# ----------------------------------------------------------------------

def node_conservatism(cfg: StructuralConfig,
                      paradigm: str = "phlogiston") -> tuple[tuple[str, ...], np.ndarray]:
    """Per-node conservatism ``lambda_v = ||carryover(net, (v,))||`` for every measured
    node, read off the incumbent paradigm's prior net (Eq. 5; the Schur fill-in mass a
    node leaves on the survivors when removed -- large for a core hub, small for a belt
    node). Returns ``(measured_node_names, lambda_values)`` aligned to the rows of
    ``H_observable``. The hidden hub is excluded (it is not measured)."""
    net = (ph.phlogiston_prior(cfg) if paradigm == "phlogiston"
           else ph.oxygen_prior(cfg))
    nodes = ph.measured_nodes(cfg)
    lam = np.array([obs.carryover_mass(net, (v,)) for v in nodes])
    return nodes, lam


# ----------------------------------------------------------------------
# 1b. per-AGENT conservatism (the lock-in & flip gradient)
# ----------------------------------------------------------------------

def agent_conservatism(Pi0: np.ndarray, W: np.ndarray | None = None,
                       kind: str = "precision") -> np.ndarray:
    """Per-agent structural stiffness (N,) -- a high value reorganises last.

    ``Pi0`` : (N, d, d) the agents' *initial* precision matrices (entrenchment is a
              prior property; read it at ``t=0``). ``W`` : (N, N) trust/fusion weights
              (only needed for ``kind='centrality'``).

      * ``kind='precision'`` (default): total prior precision mass ``trace(Pi0_i)`` --
        a stiff/confident prior (an entrenched agent) resists revision. This is the
        agent-level analogue of node conservatism.
      * ``kind='centrality'``: eigenvector centrality of agent ``i`` on the (symmetrised)
        trust graph -- a densely-trusted hub agent moves with the crowd and so flips
        late once the crowd is committed. Falls back to weighted degree if the power
        iteration does not separate.
    """
    Pi0 = np.asarray(Pi0)
    if kind == "precision":
        return np.trace(Pi0, axis1=1, axis2=2)
    if kind == "centrality":
        if W is None:
            raise ValueError("kind='centrality' needs the trust matrix W")
        A = 0.5 * (np.asarray(W) + np.asarray(W).T)
        v = np.ones(A.shape[0]) / A.shape[0]
        for _ in range(100):                       # power iteration
            v_new = A @ v
            n = np.linalg.norm(v_new)
            if n < 1e-12:
                return A.sum(axis=1)               # degenerate -> weighted degree
            v_new = v_new / n
            if np.linalg.norm(v_new - v) < 1e-9:
                break
            v = v_new
        return np.abs(v)
    raise ValueError(f"kind must be 'precision' or 'centrality', got {kind!r}")


# ----------------------------------------------------------------------
# 2. bucketing + per-shell curves
# ----------------------------------------------------------------------

def assign_shells(values: np.ndarray, n_shells: int = 3,
                  labels: tuple[str, ...] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """Bucket ``values`` (N,) into ``n_shells`` ordered shells by quantile, low->high.

    Returns ``(shell_id, labels)`` where ``shell_id`` (N,) ints in ``[0, n_shells)`` and
    label ``0`` is the lowest-conservatism shell (the belt). Ties on a quantile edge are
    resolved toward the lower shell. Quantile bucketing (not equal-width) keeps the shells
    populated even when ``lambda`` is heavily skewed (the hub dwarfs the belt)."""
    values = np.asarray(values, dtype=float)
    if labels is None:
        labels = DEFAULT_LABELS if n_shells == 3 else tuple(
            f"shell{i}" for i in range(n_shells))
    if len(labels) != n_shells:
        raise ValueError("labels length must equal n_shells")
    qs = np.quantile(values, np.linspace(0, 1, n_shells + 1))
    qs[-1] = np.inf                                 # include the max in the top shell
    shell_id = np.clip(np.digitize(values, qs[1:-1], right=False), 0, n_shells - 1)
    return shell_id.astype(int), labels


def shell_curves(idx_t: np.ndarray, shell_id: np.ndarray,
                 labels: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Mean trajectory within each shell. ``idx_t`` (T, N) is a per-unit progress trace
    (e.g. ``RolloutTrace.idx_t`` over agents, or a per-node oxygen trace); ``shell_id``
    (N,) the shell of each unit. Returns ``{label: (T,) mean curve}``; empty shells are
    omitted. The belt curve leads and the core curve lags iff the model produces the
    graded staircase."""
    idx_t = np.asarray(idx_t)
    out: dict[str, np.ndarray] = {}
    for s, label in enumerate(labels):
        members = np.where(shell_id == s)[0]
        if members.size:
            out[label] = idx_t[:, members].mean(axis=1)
    return out


def shell_sizes(shell_id: np.ndarray, labels: tuple[str, ...]) -> dict[str, int]:
    """How many units landed in each shell (report alongside the curves so a thin shell
    is never read as a clean signal)."""
    return {label: int(np.sum(shell_id == s)) for s, label in enumerate(labels)}


# ----------------------------------------------------------------------
# 3. per-NODE oxygen trace from a full belief-net trajectory
# ----------------------------------------------------------------------

def node_oxygen_trace(Pi_t: np.ndarray, h_t: np.ndarray,
                      cfg: StructuralConfig) -> tuple[tuple[str, ...], np.ndarray]:
    """Per-node reorganisation trace from a full ``(Pi_t, h_t)`` trajectory (the output
    of ``run_trace_net``). For each measured node ``v`` and step ``t``, the population-mean
    oxygen index *of that node*: ``(mu_v - mu_phlog) / (mu_oxy - mu_phlog)`` clipped to
    [0,1]. Only the disagreement (mass-law) nodes have ``mu_oxy != mu_phlog``; agreement
    nodes return ~0 (the two paradigms posit the same mean, so there is nothing to
    reorganise) -- which is exactly why the per-node staircase is shallow in this net.

    Returns ``(measured_node_names, node_oxy_t)`` with ``node_oxy_t`` of shape
    ``(T, n_measured)``. Feed a node-level conservatism bucketing + ``shell_curves`` (with
    the trace transposed to (T, n_measured)) to get the per-shell node staircase."""
    Pi_t, h_t = np.asarray(Pi_t), np.asarray(h_t)
    T, N, d, _ = Pi_t.shape
    names = cfg.node_names
    meas = ph.measured_nodes(cfg)
    meas_idx = np.array([names.index(n) for n in meas])
    # posterior means per step per agent: solve Pi mu = h  (T, N, d)
    mu = np.linalg.solve(Pi_t.reshape(T * N, d, d),
                         h_t.reshape(T * N, d, 1)).reshape(T, N, d)
    mu_meas = mu[:, :, meas_idx].mean(axis=1)        # (T, n_measured) population mean
    # per-node phlog/oxy reference means (disagreement nodes differ, others coincide).
    phlog = ph.phlogiston_prior(cfg).mean()
    oxy = ph.oxygen_prior(cfg).mean()
    mu_phlog = np.asarray(phlog)[meas_idx]
    mu_oxy = np.asarray(oxy)[meas_idx]
    denom = mu_oxy - mu_phlog
    safe = np.abs(denom) > 1e-9
    node_oxy = np.zeros_like(mu_meas)
    node_oxy[:, safe] = np.clip(
        (mu_meas[:, safe] - mu_phlog[safe]) / denom[safe], 0.0, 1.0)
    return meas, node_oxy
