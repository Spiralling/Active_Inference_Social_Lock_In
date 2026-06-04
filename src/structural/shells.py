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


# ----------------------------------------------------------------------
# 4. edge (coupling) trajectories -- did the data actually MOVE the structure?
# ----------------------------------------------------------------------

def edge_trace(Pi_t: np.ndarray, cfg: StructuralConfig,
               pairs: list[tuple[str, str]]) -> dict[tuple[str, str], np.ndarray]:
    """Population-mean off-diagonal precision ``Pi[a, b]`` over time for each node pair.

    ``Pi_t`` (T, N, d, d) is the full belief-net trajectory from ``step.run_trace_net``;
    ``pairs`` a list of ``(node_a, node_b)`` name tuples. Returns ``{(a, b): (T,) curve}``,
    the population mean of the precision coupling between the two nodes at each step.

    This is the raw structural-learning signal and the headline read-out of the substrate
    migration (P1): the *relational* (gravimetric) observation operator deposits
    OFF-DIAGONAL Fisher, so a belt coupling such as
    ``("calx_heavier_than_metal", "mass_change_sign")`` GROWS over the rollout; the *node*
    operator deposits only diagonal Fisher, so the same entry stays pinned at its prior
    value -- the edges are frozen. Read straight off ``Pi`` (no inversion), so it is robust
    to the improper hub prior. The normalized CPD weight ``B[child, parent]`` is the same
    signal rescaled -- recover it with ``bayesnet.LinearGaussianBN.from_info`` on a PD
    posterior if a CPD reading is wanted.
    """
    Pi_t = np.asarray(Pi_t)
    names = cfg.node_names
    out: dict[tuple[str, str], np.ndarray] = {}
    for a, b in pairs:
        ia, ib = names.index(a), names.index(b)
        out[(a, b)] = Pi_t[:, :, ia, ib].mean(axis=1)        # (T,)
    return out


# ----------------------------------------------------------------------
# 5. is there really a belt AND a core?  +  residual structural disagreement
# ----------------------------------------------------------------------

def conservatism_split(values: np.ndarray) -> tuple[float, float]:
    """The belt/core threshold ``tau`` and an honesty number for the split.

    ``tau`` is the median conservatism (the quantile boundary ``assign_shells`` uses for
    ``n_shells=2``). The second return is **Sarle's bimodality coefficient**
    ``BC = (skew^2 + 1) / kurtosis`` (non-excess kurtosis): for a unimodal Gaussian
    ``BC ~ 0.33``, for a uniform ``~ 0.56``, and ``-> 1`` for a clean two-spike
    distribution. The classic cutoff is ``5/9 ~ 0.555``: ``BC > 0.555`` suggests the
    population really does have two conservatism modes (a belt and a core), while a low
    ``BC`` means the belt/core split is a quantile *convenience* on a single mode, not a
    structural fact -- which is exactly the caveat to PRINT next to any staircase figure
    (per the honest-findings rule: a quantile split always *produces* two curves; this
    number says whether the split is real)."""
    v = np.asarray(values, dtype=float)
    tau = float(np.median(v))
    s = v.std()
    if s < 1e-12:
        return tau, 0.0
    z = (v - v.mean()) / s
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))                # non-excess (Gaussian -> 3)
    return tau, (skew ** 2 + 1.0) / kurt


def residual_disagreement(Pi_t: np.ndarray) -> np.ndarray:
    """Population dispersion of *structure* over time -- the paper's "residual structural
    disagreement" measured quantity. At each step, the mean Frobenius distance of each
    agent's precision matrix to the population-mean precision:

        d(t) = (1/N) sum_i || Pi_i(t) - mean_j Pi_j(t) ||_F .

    ``Pi_t`` (T, N, d, d) from ``step.run_trace_net``. Returns ``(T,)``.

    This falls toward 0 as the population reaches structural *consensus* and stays high
    when an entrenched bloc refuses to move (the lock-in signature: the core blocs go on
    disagreeing forever). Read on the precision (not the mean), so it tracks the NET
    converging; robust to the overall precision growth because every agent sharpens
    together and the population mean tracks it (contrast a distance to a *fixed* prior,
    which would be swamped by confidence growth). Pairs with ``observables.settling_time``
    on this curve to read time-to-consensus."""
    Pi_t = np.asarray(Pi_t)
    Pi_bar = Pi_t.mean(axis=1, keepdims=True)            # (T,1,d,d)
    fro = np.sqrt(((Pi_t - Pi_bar) ** 2).sum(axis=(2, 3)))  # (T,N)
    return fro.mean(axis=1)                               # (T,)


def edge_count_trace(Pi_t: np.ndarray, threshold: float = 0.15
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Population-mean structural edge count over time and its signed step-to-step change,
    for ``plot.plot_edge_edit_timeline``. An edge ``(i,j)`` is counted present when the
    population-mean ``|Pi[i,j]|`` exceeds ``threshold`` (upper triangle only).

    ``Pi_t`` (T, N, d, d) from ``run_trace_net`` (or (T, d, d)). Returns
    ``(edge_count_t (T,), edge_delta_t (T,))``. On the relational substrate the count RISES
    as the mass-balance data deposit new belt couplings (data-driven structure *expansion*);
    Bayesian Model Reduction prunes the prior couplings the data stop supporting (structure
    *reduction*). The two moves are the paper's expansion/reduction, read off the trajectory."""
    Pi_t = np.asarray(Pi_t)
    Pim = np.abs(Pi_t.mean(axis=1)) if Pi_t.ndim == 4 else np.abs(Pi_t)   # (T, d, d)
    T, d, _ = Pim.shape
    iu = np.triu_indices(d, k=1)
    counts = np.array([(Pim[t][iu] > threshold).sum() for t in range(T)], dtype=float)
    delta = np.concatenate([[0.0], np.diff(counts)])
    return counts, delta


def core_stall(curve: np.ndarray, level: float = 0.5) -> tuple[bool, float, float]:
    """Lock-in read-out for an order-parameter trajectory ``curve`` (T,) -- a population
    m(t), or a single shell's m_S(t). Returns ``(stalled, final, peak)``:

      * ``stalled`` -- ``True`` if the curve never reaches ``level`` (it stayed near its
        starting pole: evidential lock-in -- the paradigm shift never happened);
      * ``final``   -- the last value (the realised attractor over the horizon);
      * ``peak``    -- the max reached (so a curve that rose then was dragged back is
        distinguishable from one that never moved).

    A converged run has ``final`` near the post-shift truth (> ``level``, not stalled); a
    locked-in run stalls near its origin (< ``level``, stalled). The boolean is the cell
    classifier for the gamma x conservatism lock-in phase diagram."""
    c = np.asarray(curve, dtype=float)
    return bool(c.max() < level), float(c[-1]), float(c.max())
