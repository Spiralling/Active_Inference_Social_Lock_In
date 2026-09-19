"""Structure-level measurement: distances between learned WIRINGS, and the collective's net.

The structural-pluralism results live at the level of *which edges an agent's belief net
holds* -- the off-diagonal of the information-form precision ``Pi`` (``belief.py``: the
precision pattern IS the undirected conditional-independence structure). This module is the
measurement layer for that level, on host numpy (post-hoc read-offs of ``snap_Pi`` stacks,
never part of the dynamics -- the ``shells.py`` discipline):

  1. **Coupling read-offs** -- :func:`coupling_vector` (the contested entries as a vector),
     :func:`edge_set` (the thresholded precision-pattern edge set), and
     :func:`spectral_signature` (a permutation-invariant shape descriptor).
  2. **Pairwise structure distances** between agents -- :func:`pairwise_distance` in three
     readings (Frobenius on couplings / Jaccard on edge sets / spectral), and their
     within- vs cross-community decomposition over time, :func:`dispersion_trace`.
     ``cross >> within`` is structural pluralism; ``cross -> within`` is collapse.
  3. **The collective vs its members** -- :func:`consensus_union_report`. The *consensus*
     net is the plain mean of the individual ``Pi`` -- deliberately, because that is exactly
     what uniform complete-graph precision pooling (``step.fuse``) computes, so the report
     compares the model's own fusion operator against the distribution it pools. The *union*
     is the set of edges held by at least one member.

Theorem-let (verified empirically by the ``synergy_edges`` key, provable in one line): for
any entry ``e``, ``|mean_i Pi_i[e]| <= max_i |Pi_i[e]|`` -- **averaging can never CREATE an
edge absent from every individual.** So the "collective knows more than any member"
phenomenon can only live in the UNION of the individual nets; the average can only blur a
coupling held by a subset (``dilution_ratio``) or destroy it outright (``edges_lost``).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------
# 1. coupling read-offs of a single net (or a stack -- all are batch-safe).
# ----------------------------------------------------------------------

def coupling_vector(Pi: np.ndarray, pairs) -> np.ndarray:
    """The contested couplings of a net as a vector: ``Pi[..., i_k, j_k]`` for each pair.

    ``Pi`` is ``(..., d, d)`` -- one net ``(d, d)``, an agent stack ``(N, d, d)``, or a
    trajectory ``(T, N, d, d)``; the leading axes pass through unchanged. ``pairs`` is a
    length-``k`` sequence of ``(i, j)`` index tuples. Returns ``(..., k)``."""
    Pi = np.asarray(Pi)
    p = np.asarray(pairs, dtype=int)                       # (k, 2)
    return Pi[..., p[:, 0], p[:, 1]]


def edge_set(Pi: np.ndarray, threshold: float, pairs=None) -> frozenset:
    """The PRECISION-PATTERN edge set of one net: ``{(i, j), i < j : |Pi[i, j]| > threshold}``.

    Restricted to the candidate ``pairs`` when given (the contested couplings), otherwise
    over the full upper triangle. Pairs are canonicalized to ``i < j``.

    Decision note: the precision pattern is the PRIMARY structure object here -- the
    contested structure IS ``Pi``'s off-diagonal (an entry ``Pi[a, c] != 0`` is exactly a
    conditional dependence between ``a`` and ``c``). The alternative -- directed-CPD edges
    via ``bayesnet.from_info`` -- depends on the node ORDER and a nonlinear UDU'
    factorization, so small precision differences can reshuffle the directed weights; it is
    noisy as a *distance* and is deliberately not used."""
    Pi = np.asarray(Pi)
    if pairs is None:
        d = Pi.shape[-1]
        iu = np.triu_indices(d, k=1)
        pairs = list(zip(iu[0].tolist(), iu[1].tolist()))
    out = []
    for (i, j) in pairs:
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        if abs(float(Pi[a, b])) > threshold:
            out.append((a, b))
    return frozenset(out)


def spectral_signature(Pi: np.ndarray) -> np.ndarray:
    """Descending real eigenvalues ``(d,)`` of the row-stochastic structure operator
    ``W = rownorm(|Pi| * (1 - I))`` -- the numpy mirror of
    ``dual_field.PrecisionUtilityNet.utility_operator`` (a unit test pins exact agreement),
    so this descriptor reads the SAME operator the conviction field propagates along.

    ``W`` is similar to a symmetric matrix (``D^{-1} S ~ D^{-1/2} S D^{-1/2}``), so the
    spectrum is real; isolated nodes (zero off-diagonal mass) get a self-loop, exactly as
    the operator does. The signature is PERMUTATION-INVARIANT: it compares wiring SHAPE
    independent of node identity -- two mirror-image wirings (same shape on different
    nodes) read as distance 0 (documented behavior; the unit fixture demonstrates it)."""
    Pi = np.asarray(Pi, dtype=float)
    d = Pi.shape[-1]
    off = np.abs(Pi) * (1.0 - np.eye(d))
    row_sum = off.sum(axis=1, keepdims=True)
    W = np.where(row_sum > 0.0, off / np.where(row_sum > 0.0, row_sum, 1.0), 0.0)
    W = W + np.diag((row_sum[:, 0] <= 0.0).astype(float))
    ev = np.linalg.eigvals(W).real
    return np.sort(ev)[::-1]


# ----------------------------------------------------------------------
# 2. pairwise structure distances + the within/cross decomposition.
# ----------------------------------------------------------------------

def pairwise_distance(Pi_stack: np.ndarray, kind: str = "fro", pairs=None,
                      threshold: float = 0.15) -> np.ndarray:
    """Symmetric ``(N, N)`` structure-distance matrix (zero diagonal) over an agent stack.

    ``kind='fro'``      : ``||coupling_vector_i - coupling_vector_j||_2`` on ``pairs``
                          (full Frobenius distance on ``Pi`` when ``pairs=None``).
    ``kind='jaccard'``  : ``1 - |E_i n E_j| / |E_i u E_j|`` on :func:`edge_set`
                          (``threshold``, ``pairs``); ``0.0`` when both sets are empty
                          (two agents both *lacking* every edge agree).
    ``kind='spectral'`` : ``||spectral_signature_i - spectral_signature_j||_2``
                          (permutation-invariant -- see :func:`spectral_signature`)."""
    Pi_stack = np.asarray(Pi_stack)
    N = Pi_stack.shape[0]
    if kind == "fro":
        vec = (coupling_vector(Pi_stack, pairs) if pairs is not None
               else Pi_stack.reshape(N, -1))
        diff = vec[:, None, :] - vec[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))
    if kind == "jaccard":
        sets = [edge_set(Pi_stack[i], threshold, pairs) for i in range(N)]
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                union = sets[i] | sets[j]
                D[i, j] = D[j, i] = (0.0 if not union
                                     else 1.0 - len(sets[i] & sets[j]) / len(union))
        return D
    if kind == "spectral":
        sig = np.stack([spectral_signature(Pi_stack[i]) for i in range(N)])
        diff = sig[:, None, :] - sig[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))
    raise ValueError(f"kind must be 'fro'|'jaccard'|'spectral', got {kind!r}")


def dispersion_trace(Pi_t: np.ndarray, membership: np.ndarray, kind: str = "fro",
                     pairs=None, threshold: float = 0.15) -> dict:
    """Within- vs cross-community mean structure distance over a trajectory.

    ``Pi_t`` ``(T, N, d, d)``, ``membership`` ``(N,)`` int block ids. At each snapshot the
    full :func:`pairwise_distance` matrix is decomposed over unordered agent pairs:

      ``within``  : ``(T, B)`` mean distance over same-block pairs, per block;
      ``cross``   : ``(T,)`` mean distance over different-block pairs;
      ``overall`` : ``(T,)`` mean distance over all pairs.

    ``cross >> within`` is structural pluralism (each community internally coherent,
    communities apart); ``cross -> within`` is the collapse signature."""
    Pi_t = np.asarray(Pi_t)
    membership = np.asarray(membership, dtype=int)
    T, N = Pi_t.shape[0], Pi_t.shape[1]
    B = int(membership.max()) + 1
    iu = np.triu_indices(N, k=1)
    same = membership[iu[0]] == membership[iu[1]]
    within_masks = [(same & (membership[iu[0]] == b)) for b in range(B)]
    within = np.zeros((T, B))
    cross = np.zeros(T)
    overall = np.zeros(T)
    for t in range(T):
        D = pairwise_distance(Pi_t[t], kind=kind, pairs=pairs, threshold=threshold)
        flat = D[iu]
        for b in range(B):
            within[t, b] = flat[within_masks[b]].mean() if within_masks[b].any() else 0.0
        cross[t] = flat[~same].mean() if (~same).any() else 0.0
        overall[t] = flat.mean()
    return {"within": within, "cross": cross, "overall": overall}


# ----------------------------------------------------------------------
# 3. the collective vs its members: consensus (the engine's own pooling) vs union.
# ----------------------------------------------------------------------

def consensus_union_report(Pi_stack: np.ndarray, h_stack: np.ndarray | None = None,
                           pairs=None, threshold: float = 0.15) -> dict:
    """Compare the CONSENSUS net against the UNION of the individual nets.

    The consensus is the plain mean ``Pi`` -- exactly what uniform complete-graph precision
    pooling (``step.fuse`` with ``W = 1/N``) computes at its fixed point, so this is the
    model's OWN fusion operator read against the distribution it pools. Keys:

      ``consensus_edges``        : :func:`edge_set` of the mean net;
      ``consensus_couplings``    : :func:`coupling_vector` of the mean net (on ``pairs``);
      ``individual_edges``       : per-agent edge sets (list of frozensets);
      ``per_agent_counts``       : ``(N,)`` per-agent edge counts;
      ``union_edges`` / ``union_count`` : edges held by >= 1 member;
      ``best_individual_count``  : ``max_i |E_i|``;
      ``union_coverage_ratio``   : ``|union| / max_i |E_i|`` -- > 1 iff the collective's
                                   union holds structure no single member holds;
      ``dilution_ratio``         : ``mean_e |Pi_cons[e]| / max_i |Pi_i[e]|`` over the union
                                   edges -- how much averaging blurs the couplings;
      ``edges_lost``             : held by >= 1 individual, absent from the consensus;
      ``synergy_edges``          : in the consensus, in NO individual -- provably EMPTY
                                   (``|mean| <= max |.|``; reported to verify);
      ``consensus_mean``         : posterior mean of the consensus net
                                   ``solve(mean Pi, mean h)`` (only if ``h_stack`` given)."""
    Pi_stack = np.asarray(Pi_stack)
    N = Pi_stack.shape[0]
    Pi_cons = Pi_stack.mean(axis=0)
    consensus_edges = edge_set(Pi_cons, threshold, pairs)
    individual_edges = [edge_set(Pi_stack[i], threshold, pairs) for i in range(N)]
    per_agent_counts = np.array([len(e) for e in individual_edges])
    union_edges = frozenset().union(*individual_edges) if N else frozenset()
    best = int(per_agent_counts.max()) if N else 0
    dil = [abs(float(Pi_cons[i, j])) /
           max(abs(coupling_vector(Pi_stack, [(i, j)])).max(), EPS)
           for (i, j) in sorted(union_edges)]
    report = {
        "consensus_edges": consensus_edges,
        "consensus_couplings": (coupling_vector(Pi_cons, pairs)
                                if pairs is not None else None),
        "individual_edges": individual_edges,
        "per_agent_counts": per_agent_counts,
        "union_edges": union_edges,
        "union_count": len(union_edges),
        "best_individual_count": best,
        "union_coverage_ratio": len(union_edges) / max(best, 1),
        "dilution_ratio": float(np.mean(dil)) if dil else float("nan"),
        "edges_lost": union_edges - consensus_edges,
        "synergy_edges": consensus_edges - union_edges,
    }
    if h_stack is not None:
        h_cons = np.asarray(h_stack).mean(axis=0)
        report["consensus_mean"] = np.linalg.solve(Pi_cons, h_cons)
    return report
