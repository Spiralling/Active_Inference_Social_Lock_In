"""First-class social-graph objects for the structural coupling layer.

The legacy path (``src/network.build_adjacency``) builds a (N, N) adjacency,
row-normalises it to the trust/fusion matrix ``W``, and *throws the adjacency
away*. Every "change the topology" operation downstream then has to either
reverse-engineer from ``W`` (``Network.isolated`` overwrites it with the
identity) or rebuild the adjacency from scratch (``Network._coupled_W`` re-runs
``build_adjacency`` and re-derives membership from the belief ``groups``). That
is the entanglement this module removes:

  * **The graph is a value.** ``Graph`` is a thin frozen dataclass carrying the
    adjacency ``A``, the optional block ``membership`` (for community graphs),
    and just enough provenance (``kind`` + the SBM ``intra``/``seed``) to *re-emit
    itself at a different cross-density*. ``Network`` stores a ``Graph`` and asks
    it for ``W``; ``isolated`` / ``with_bridge`` are now graph methods, not
    ``W``-surgery.

  * **Topology is decoupled from beliefs.** A community graph is specified by
    ``community(sizes, intra, inter)`` -- block *sizes*, not the belief
    ``groups``. You can put an entrenched bloc and a vanguard on the *same*
    Erdos-Renyi graph, or two same-belief blocs on an SBM, independently.

Named constructors (the menu): :func:`erdos_renyi`, :func:`barabasi_albert`
(alias :func:`scale_free`), :func:`watts_strogatz`, :func:`community` (the SBM /
planted-block graph), :func:`ring`, :func:`lattice`, :func:`complete`.

The three constructors that already exist in ``build_adjacency``
(``scale_free`` / ``watts_strogatz`` / ``planted_sbm``) *delegate to it*, so a
``Graph`` built here is byte-identical to the legacy adjacency -- the structural
dynamics and every equivalence test are untouched. The new families
(Erdos-Renyi, ring, lattice, complete) are built directly with networkx.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from src.network import build_adjacency
from src.structural.step import trust_weights


def _from_nx(g: nx.Graph, n: int) -> np.ndarray:
    """networkx graph -> symmetric (n, n) 0/1 adjacency with zero diagonal,
    in canonical node order ``0..n-1`` (the convention of ``build_adjacency``)."""
    A = nx.to_numpy_array(g, nodelist=list(range(n)), dtype=np.float32)
    np.fill_diagonal(A, 0.0)
    return A


@dataclass(frozen=True, eq=False)
class Graph:
    """A social graph as a value: the adjacency plus its provenance.

    ``A``           : (N, N) symmetric 0/1 adjacency, zero diagonal. ``A[i, j] = 1``
                      iff ``j`` is a neighbour of ``i``; self-contribution lives in
                      the ``+I`` of :meth:`trust_W`, never in ``A``.
    ``membership``  : (N,) int block id per agent, or ``None`` for a non-block
                      graph. Set by :func:`community`; read by :meth:`with_bridge`
                      and by per-community read-outs in notebooks.
    ``kind``        : the constructor that produced this graph (provenance).
    ``intra``/``seed`` : SBM provenance, so :meth:`with_bridge` can re-emit the
                      *same* within-block structure at a new cross-density.
    """

    A: np.ndarray
    membership: np.ndarray | None = None
    kind: str = "custom"
    intra: float = 0.0
    seed: int = 0

    # -- hashing: metadata only ----------------------------------------
    # A Graph is carried as an ``eqx.field(static=True)`` on ``Network`` (it is
    # host-side metadata, never read inside a traced rollout), so it lands in the
    # JIT treedef and must be HASHABLE -- but ``A`` / ``membership`` are numpy
    # arrays (unhashable, and irrelevant to the compiled function, which only sees
    # ``W``). Hash/compare on the structural metadata only: same ``(kind, n,
    # n_blocks)`` => identical compiled rollout, stable cache key, no spurious
    # recompiles across seeds.
    def __hash__(self) -> int:
        return hash((self.kind, self.n, self.n_blocks))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Graph):
            return NotImplemented
        return (self.kind, self.n, self.n_blocks) == \
               (other.kind, other.n, other.n_blocks)

    # -- shape ----------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_blocks(self) -> int:
        return 0 if self.membership is None else int(self.membership.max()) + 1

    # -- the one thing the dynamics needs: the fusion matrix ------------
    def trust_W(self, gamma: jax.Array | None = None) -> jax.Array:
        """Row-stochastic fusion weights over the *closed* neighbourhood
        (self + neighbours), exactly ``step.trust_weights(A + I, gamma)``. This is
        the single bridge from "a graph" to "what the rollout scans"."""
        A_self = jnp.asarray(self.A) + jnp.eye(self.n)
        return trust_weights(A_self, gamma)

    # -- topology operations (were W-surgery / groups-rebuilds) ---------
    def isolated(self) -> "Graph":
        """Communication off: every edge removed, so :meth:`trust_W` is the
        identity and each agent only ever sees its own data (the no-fusion
        baseline). Membership is preserved for read-outs."""
        return Graph(A=np.zeros_like(self.A), membership=self.membership,
                     kind="isolated", intra=self.intra, seed=self.seed)

    def with_bridge(self, inter: float, intra: float | None = None,
                    seed: int | None = None) -> "Graph":
        """Re-emit this community graph with the blocks bridged at cross-density
        ``inter`` (the within-block ``intra`` and ``seed`` default to this graph's
        own provenance, so only the cross-block edges change). Requires a block
        ``membership``. This replaces the old ``Network._coupled_W(inter, groups)``
        -- the membership now travels *with the graph*, not the belief groups."""
        if self.membership is None:
            raise ValueError("with_bridge needs a block membership; build the "
                             "graph with community(...) / kind='community'")
        sizes = [int((self.membership == b).sum()) for b in range(self.n_blocks)]
        return community(sizes,
                         intra=self.intra if intra is None else intra,
                         inter=inter,
                         seed=self.seed if seed is None else seed)


# ----------------------------------------------------------------------
# The constructor menu. Each returns a Graph.
# ----------------------------------------------------------------------

def erdos_renyi(n: int, p: float | None = None, *, seed: int = 0,
                mean_degree: int | None = None) -> Graph:
    """Erdos-Renyi random graph ``G(n, p)``: every distinct pair is connected
    independently with probability ``p``. Give ``p`` directly, or a target
    ``mean_degree`` (then ``p = mean_degree / (n - 1)``). The maximally
    unstructured baseline -- a Poisson degree distribution, no hubs, no blocks."""
    if p is None:
        if mean_degree is None:
            raise ValueError("erdos_renyi needs either p or mean_degree")
        p = mean_degree / max(n - 1, 1)
    g = nx.gnp_random_graph(n=n, p=float(p), seed=seed)
    return Graph(A=_from_nx(g, n), kind="erdos_renyi", seed=seed)


def barabasi_albert(n: int, *, mean_degree: int = 4, seed: int = 0) -> Graph:
    """Barabasi-Albert scale-free graph: preferential attachment with
    ``m = max(mean_degree // 2, 1)`` edges per arriving node. Power-law degree
    tail and visible hub clusters. (Delegates to ``build_adjacency`` -- identical
    to the legacy ``kind='scale_free'`` adjacency.)"""
    A = build_adjacency(n, mean_degree=mean_degree, rewiring_p=0.0, seed=seed,
                        kind="scale_free")
    return Graph(A=A, kind="scale_free", seed=seed)


#: Alias -- "the scale-free thing".
scale_free = barabasi_albert


def watts_strogatz(n: int, *, mean_degree: int = 4, rewiring_p: float = 0.1,
                   seed: int = 0) -> Graph:
    """Watts-Strogatz small-world graph: a ``mean_degree``-regular ring with each
    edge rewired with probability ``rewiring_p`` (short path length, high
    clustering). (Delegates to ``build_adjacency``; ``rewiring_p=0`` is a pure
    ring lattice -- see :func:`ring`.)"""
    A = build_adjacency(n, mean_degree=mean_degree, rewiring_p=rewiring_p,
                        seed=seed, kind="watts_strogatz")
    return Graph(A=A, kind="watts_strogatz", seed=seed)


def community(sizes: list[int], *, intra: float, inter: float,
              seed: int = 0) -> Graph:
    """Planted stochastic-block-model graph: ``len(sizes)`` communities whose
    agents are wired densely WITHIN their block (edge prob ``intra``) and sparsely
    BETWEEN blocks (edge prob ``inter``). ``inter=0`` gives genuinely disconnected
    communities (echo chambers); dial it up to couple them. Blocks are contiguous
    in agent order (block 0 first, then block 1, ...), and the returned graph
    carries the ``membership`` so it can be re-bridged later (:meth:`Graph.with_bridge`).

    Decoupled from beliefs: ``sizes`` are *topology* block sizes, independent of
    any belief ``groups`` (delegates to ``build_adjacency(kind='planted_sbm')``
    with contiguous membership, so it matches the legacy SBM adjacency)."""
    n = int(sum(sizes))
    membership = np.concatenate(
        [np.full(s, b, dtype=np.int64) for b, s in enumerate(sizes)])
    A = build_adjacency(n, mean_degree=0, rewiring_p=0.0, seed=seed,
                        kind="planted_sbm", society_membership=membership,
                        intra_prob=intra, inter_prob=inter)
    return Graph(A=A, membership=membership, kind="community",
                 intra=intra, seed=seed)


#: Alias -- the model-theoretic name.
sbm = community


def ring(n: int, *, mean_degree: int = 2) -> Graph:
    """Deterministic ring lattice: each node joined to its ``mean_degree`` nearest
    neighbours on a cycle (a Watts-Strogatz graph with zero rewiring). A
    controlled, fully-homogeneous baseline -- every agent structurally identical."""
    g = nx.watts_strogatz_graph(n=n, k=mean_degree, p=0.0, seed=0)
    return Graph(A=_from_nx(g, n), kind="ring")


def lattice(rows: int, cols: int) -> Graph:
    """Deterministic 2-D grid lattice (``rows x cols`` agents, 4-neighbour). The
    canonical "local interactions only, no long-range edges" baseline. Agent index
    is row-major: agent ``r * cols + c`` sits at grid cell ``(r, c)``."""
    g = nx.grid_2d_graph(rows, cols)
    g = nx.convert_node_labels_to_integers(g, ordering="sorted")
    return Graph(A=_from_nx(g, rows * cols), kind="lattice")


def complete(n: int) -> Graph:
    """Fully-connected graph: every agent talks to every other (mean-field /
    full-mixing baseline). :meth:`Graph.trust_W` is then the uniform ``1/n``
    matrix."""
    g = nx.complete_graph(n)
    return Graph(A=_from_nx(g, n), kind="complete")


# ----------------------------------------------------------------------
# Spectral read-outs: the connectivity axis of the Zollman transient.
# ----------------------------------------------------------------------

def laplacian(graph: Graph) -> np.ndarray:
    """The (combinatorial) graph Laplacian ``L = D - A`` of the *bare* adjacency
    (degree diagonal minus adjacency). ``L`` is symmetric PSD with a zero eigenvalue
    per connected component; its spectrum is the connectivity read-out below."""
    A = np.asarray(graph.A, dtype=float)
    return np.diag(A.sum(axis=1)) - A


def algebraic_connectivity(graph: Graph) -> float:
    """The **Fiedler value** ``lambda_2`` -- the second-smallest Laplacian eigenvalue.
    Zero iff the graph is disconnected (``isolated()`` or fully separate communities);
    larger means a better-knit graph that mixes information faster. This is the axis the
    Zollman speed/accuracy transient lives on: raising ``lambda_2`` (denser bridges, more
    rewiring) speeds consensus but spends the transient diversity that, near a lock-in
    boundary, protects a community from committing early to the worse paradigm.

    Computed with ``numpy.linalg.eigvalsh`` (the Laplacian is symmetric), so the
    eigenvalues come back sorted ascending and ``lambda_2`` is ``eigvals[1]``."""
    w = np.linalg.eigvalsh(laplacian(graph))
    return float(w[1]) if w.shape[0] > 1 else 0.0


# ----------------------------------------------------------------------
# Config -> Graph (the one place that reads a NetworkConfig).
# ----------------------------------------------------------------------

def graph_from_config(nc, n: int, seed: int,
                      membership: np.ndarray | list[int] | None = None) -> Graph:
    """Build the :class:`Graph` a ``NetworkConfig`` describes. This is the single
    bridge from the declarative config to a graph value; ``Network.init`` uses it
    for the default (no explicit graph) path, and it reproduces *exactly* the
    adjacency ``step.init_state`` builds, so the default rollout is unchanged.

    ``membership`` is only consulted for ``kind='planted_sbm'`` (the legacy path
    where community membership is derived from the belief ``groups``); the new
    decoupled path passes an explicit ``community(...)`` graph instead.
    """
    kind = nc.kind
    if kind == "planted_sbm":
        if membership is None:
            raise ValueError(
                "planted_sbm via config needs a membership (derived from groups); "
                "for topology decoupled from beliefs use graphs.community(sizes,...)")
        membership = np.asarray(membership, dtype=np.int64)
        A = build_adjacency(n, mean_degree=nc.mean_degree, rewiring_p=nc.rewiring_p,
                            seed=seed, kind="planted_sbm",
                            society_membership=membership,
                            intra_prob=nc.intra_prob, inter_prob=nc.inter_prob)
        return Graph(A=A, membership=membership, kind="community",
                     intra=nc.intra_prob, seed=seed)
    if kind == "erdos_renyi":
        p = getattr(nc, "p_edge", None)
        return erdos_renyi(n, p, seed=seed, mean_degree=nc.mean_degree)
    if kind == "ring":
        return ring(n, mean_degree=nc.mean_degree)
    if kind == "lattice":
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(f"lattice kind needs a square n_agents; got n={n}")
        return lattice(side, side)
    if kind == "complete":
        return complete(n)
    # scale_free / watts_strogatz: delegate to build_adjacency (legacy-identical).
    A = build_adjacency(n, mean_degree=nc.mean_degree, rewiring_p=nc.rewiring_p,
                        seed=seed, kind=kind)
    return Graph(A=A, kind=kind, seed=seed)
