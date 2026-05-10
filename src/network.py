"""Network construction for the multi-agent coupling layer.

Two graph families:
  - Watts–Strogatz small-world (the paper's stated topology)
  - Barabási–Albert scale-free (default; produces visible hub clusters
    under spring layout, which is what the population-scale visualisations
    in `src/viz.py` rely on at large N)

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
) -> np.ndarray:
    """Symmetric (N, N) 0/1 adjacency matrix; zero diagonal.

    Convention: ``A[i, j] = 1`` iff j is a neighbour of i. Self-contribution
    lives in the (1 - d_i) weight of the posterior mixture, not in the
    adjacency.

    ``kind``:
      - ``"scale_free"``: Barabási–Albert with m = max(mean_degree // 2, 1).
        Produces a power-law degree distribution with hub structure that
        spring layouts render as visible clusters.
      - ``"watts_strogatz"``: small-world with k=mean_degree, p=rewiring_p.
    """
    if kind == "scale_free":
        m = max(mean_degree // 2, 1)
        g = nx.barabasi_albert_graph(n=n_agents, m=m, seed=seed)
    elif kind == "watts_strogatz":
        g = nx.watts_strogatz_graph(n=n_agents, k=mean_degree, p=rewiring_p, seed=seed)
    else:
        raise ValueError(f"Unknown network kind: {kind!r}")

    A = nx.to_numpy_array(g, dtype=np.float32)
    np.fill_diagonal(A, 0.0)
    return A
