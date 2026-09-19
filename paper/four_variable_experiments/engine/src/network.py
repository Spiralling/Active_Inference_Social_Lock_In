"""Network construction for the multi-agent coupling layer.

Three graph families:
  - Watts–Strogatz small-world (the paper's stated topology)
  - Barabási–Albert scale-free (default for headline-agnostic runs;
    visible hub clusters under spring layout)
  - Planted-3-block stochastic block model (society = community by
    construction; used by the three-society demo)

Once built, the adjacency is just static data — no class needed.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def build_adjacency(
    n_agents: int,
    mean_degree: int,
    rewiring_p: float,
    seed: int,
    kind: str = "scale_free",
    society_membership: np.ndarray | None = None,
    intra_prob: float = 0.04,
    inter_prob: float = 0.003,
) -> np.ndarray:
    """Symmetric (N, N) 0/1 adjacency matrix; zero diagonal.

    Convention: ``A[i, j] = 1`` iff j is a neighbour of i. Self-contribution
    lives in the (1 - d_i) weight of the posterior mixture, not in the
    adjacency.

    ``kind``:
      - ``"scale_free"``    : Barabási–Albert with m = max(mean_degree // 2, 1).
                              Power-law degree distribution, hub clusters.
      - ``"watts_strogatz"``: small-world with k=mean_degree, p=rewiring_p.
      - ``"planted_sbm"``   : stochastic block model with society blocks.
                              Requires ``society_membership``. Within-block
                              edge prob = ``intra_prob``; between-block =
                              ``inter_prob``. Society = community by design.
    """
    if kind == "scale_free":
        m = max(mean_degree // 2, 1)
        g = nx.barabasi_albert_graph(n=n_agents, m=m, seed=seed)
    elif kind == "watts_strogatz":
        g = nx.watts_strogatz_graph(n=n_agents, k=mean_degree, p=rewiring_p, seed=seed)
    elif kind == "planted_sbm":
        if society_membership is None:
            raise ValueError("planted_sbm requires society_membership")
        membership = np.asarray(society_membership, dtype=np.int64)
        if membership.shape != (n_agents,):
            raise ValueError(
                f"society_membership shape {membership.shape} != ({n_agents},)"
            )
        # Group agent indices by society
        unique_blocks = sorted(set(int(b) for b in membership))
        sizes = [int((membership == b).sum()) for b in unique_blocks]
        n_blocks = len(sizes)
        # SBM probability matrix: p[i][j] = intra if i==j else inter
        p_matrix = [
            [intra_prob if i == j else inter_prob for j in range(n_blocks)]
            for i in range(n_blocks)
        ]
        # nx.stochastic_block_model assigns nodes 0..n-1 in block-order, so
        # we build the graph then permute node labels to match the given
        # membership array.
        g_sbm = nx.stochastic_block_model(sizes, p_matrix, seed=seed)
        # Build permutation: agents grouped by society in input order.
        ordered_indices: list[int] = []
        for b in unique_blocks:
            ordered_indices.extend(int(i) for i in np.where(membership == b)[0])
        # ordered_indices maps SBM-position → agent_index
        mapping = {sbm_pos: agent_idx for sbm_pos, agent_idx in enumerate(ordered_indices)}
        g = nx.relabel_nodes(g_sbm, mapping)
    else:
        raise ValueError(f"Unknown network kind: {kind!r}")

    A = nx.to_numpy_array(g, nodelist=list(range(n_agents)), dtype=np.float32)
    np.fill_diagonal(A, 0.0)
    return A
