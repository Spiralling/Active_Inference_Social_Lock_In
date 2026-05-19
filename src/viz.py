"""Network-evolution visualisation helpers.

Pure-display, numpy input. Used by `notebooks/02_phase_space.ipynb` and
suitable for direct paper-figure rendering. No JAX in the public API.

Primitives:
- ``fixed_layout``        — one ``spring_layout`` reused across snapshots
- ``cluster_layout``      — community-detect + per-community sub-layout;
                            produces visibly separated clusters at large N
- ``detect_communities``  — returns membership array + community list
- ``draw_population``     — one snapshot of the trust graph
- ``animate_population``  — matplotlib FuncAnimation over snapshot list
"""

from __future__ import annotations

from typing import Any

import matplotlib.animation as manim
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def fixed_layout(adjacency: np.ndarray, seed: int = 0) -> dict:
    """One spring_layout over the graph's edges. Reuse across snapshots so node
    positions stay stable while node colour and edge thickness evolve."""
    g = nx.from_numpy_array(adjacency)
    return nx.spring_layout(g, seed=seed)


def detect_communities(
    adjacency: np.ndarray,
    method: str = "greedy_modularity",
    seed: int = 0,
) -> tuple[np.ndarray, list[list[int]]]:
    """Cluster nodes into communities. Returns ``(membership, communities)``.

    ``membership[i]`` is the community index of node i (0-indexed).
    ``communities[c]`` is the sorted list of node indices in community c.

    ``method``:
      - ``"greedy_modularity"``: networkx.community.greedy_modularity_communities
        — deterministic, no parameters, good default for scale-free graphs.
      - ``"louvain"``: networkx.community.louvain_communities (stochastic; uses ``seed``).
    """
    g = nx.from_numpy_array(adjacency)
    if method == "greedy_modularity":
        comms = list(nx.community.greedy_modularity_communities(g))
    elif method == "louvain":
        comms = list(nx.community.louvain_communities(g, seed=seed))
    else:
        raise ValueError(f"Unknown community method: {method!r}")

    # Normalise: sort by size descending for stable colour assignment.
    comms_sorted = sorted((sorted(c) for c in comms), key=len, reverse=True)
    N = adjacency.shape[0]
    membership = np.zeros(N, dtype=np.int32)
    for ci, members in enumerate(comms_sorted):
        for node in members:
            membership[node] = ci
    return membership, [list(c) for c in comms_sorted]


def society_layout(
    adjacency: np.ndarray,
    society_membership: np.ndarray,
    seed: int = 0,
    cluster_radius: float = 1.0,
    intra_scale: float = 0.35,
) -> dict:
    """Cluster layout where the partition is *given* (by society membership),
    not detected.

    Skips the community-detection step in ``cluster_layout`` and uses the
    society_membership array directly. Used by the three-society demo to
    guarantee that the planted SBM blocks render exactly as the three
    intended societies.
    """
    g = nx.from_numpy_array(adjacency)
    membership = np.asarray(society_membership, dtype=int)
    unique = sorted(set(int(b) for b in membership))
    communities = [
        sorted(int(i) for i in np.where(membership == b)[0]) for b in unique
    ]
    return _layout_from_partition(
        g, communities, seed=seed,
        cluster_radius=cluster_radius, intra_scale=intra_scale,
    )


def _layout_from_partition(
    g: "nx.Graph",
    communities: list[list[int]],
    seed: int,
    cluster_radius: float,
    intra_scale: float,
) -> dict:
    """Inner: place each community on an outer circle, spring-lay within."""
    n_c = len(communities)
    layout: dict = {}
    for ci, members in enumerate(communities):
        if n_c == 1:
            centre = np.zeros(2)
        else:
            angle = 2 * np.pi * ci / n_c
            centre = cluster_radius * np.array([np.cos(angle), np.sin(angle)])

        sub_g = g.subgraph(members)
        if sub_g.number_of_edges() == 0:
            rng = np.random.default_rng(seed + ci)
            for node in members:
                layout[node] = centre + intra_scale * 0.3 * rng.standard_normal(2)
        else:
            sub_layout = nx.spring_layout(sub_g, seed=seed + ci)
            for node, pos in sub_layout.items():
                layout[node] = centre + intra_scale * np.asarray(pos)
    return layout


def cluster_layout(
    adjacency: np.ndarray,
    seed: int = 0,
    method: str = "greedy_modularity",
    cluster_radius: float = 1.0,
    intra_scale: float = 0.35,
) -> dict:
    """Community-aware layout: each detected community gets its own
    ``spring_layout``, then communities are placed on an outer circle.

    Produces visibly separated cluster blobs at large N — much clearer than
    a single global spring_layout, which at N≈300 turns into a hairball.

    ``cluster_radius``: radius of the circle on which community centres lie.
    ``intra_scale``:    scale of the per-community sub-layout (in same units).

    Returns ``{node_index: np.ndarray(x, y)}`` like ``fixed_layout``.
    """
    g = nx.from_numpy_array(adjacency)
    _, communities = detect_communities(adjacency, method=method, seed=seed)
    return _layout_from_partition(
        g, communities, seed=seed,
        cluster_radius=cluster_radius, intra_scale=intra_scale,
    )


def _node_metric_from_snapshot(
    snap: dict,
    kind: str,
    lambda_per_agent: np.ndarray | None = None,
    feature_idx: int = 0,
    community_membership: np.ndarray | None = None,
) -> np.ndarray:
    """Per-node scalar metric for one snapshot under the v2 continuous-θ model.

    Snapshot dict keys expected: ``mu`` (N,), ``tau`` (N,), ``gamma`` (N, N).

    ``kind`` options:
      - ``"theta_mean"``     : μ_i — agent's posterior mean over θ.
      - ``"theta_precision"``: τ_i — agent's posterior precision.
      - ``"community"``      : community membership index (community_membership required).

    ``feature_idx`` and ``lambda_per_agent`` are retained for signature
    compatibility with v1-era callers but unused in v2.
    """
    if kind == "theta_mean":
        return np.asarray(snap["mu"], dtype=float)
    if kind == "theta_precision":
        return np.asarray(snap["tau"], dtype=float)
    if kind == "community":
        if community_membership is None:
            raise ValueError("node_metric_key='community' requires community_membership kwarg")
        return np.asarray(community_membership, dtype=float)
    raise ValueError(f"Unknown node_metric_key: {kind!r}")


def _vrange_for(
    kind: str,
    snapshots: list[dict] | None,
    community_membership: np.ndarray | None = None,
) -> tuple[float, float]:
    if kind == "theta_mean":
        # Auto-range across the snapshot list so the colour scale is stable.
        if snapshots:
            vals = np.concatenate([np.asarray(s["mu"]) for s in snapshots])
            m = float(np.max(np.abs(vals)) + 1e-6)
            return (-m, m)
        return (-1.0, 1.0)
    if kind == "theta_precision":
        if snapshots:
            vals = np.concatenate([np.asarray(s["tau"]) for s in snapshots])
            return (float(vals.min()), float(vals.max()) + 1e-6)
        return (0.0, 1.0)
    if kind == "community" and community_membership is not None:
        m = float(community_membership.max())
        return (-0.5, m + 0.5)
    return (0.0, 1.0)


def draw_population(
    ax,
    adjacency: np.ndarray,
    gamma: np.ndarray,
    node_metric: np.ndarray,
    layout: dict,
    *,
    node_cmap: str = "RdBu_r",
    edge_cmap: str = "viridis",
    node_vrange: tuple[float, float] | None = None,
    edge_width_scale: float = 3.0,
    edge_min_width: float = 0.3,
    gamma_ref: float | None = None,
    node_size: float = 220.0,
    title: str | None = None,
) -> None:
    """Draw one snapshot of the population on the given matplotlib axes.

    Edge thickness ∝ γ_ij / gamma_ref; edge colour mapped from γ via
    ``edge_cmap`` with the same fixed scale. Passing a ``gamma_ref`` (computed
    once across all snapshots) keeps widths/colours comparable across panels —
    edges shrink visibly as γ decays but never vanish below ``edge_min_width``.
    Node colour from ``node_metric`` via ``node_cmap``.
    """
    ax.clear()
    ax.set_aspect("equal")
    ax.set_axis_off()

    g = nx.from_numpy_array(adjacency)
    edges = list(g.edges())
    if edges:
        gamma_vals = np.array([gamma[u, v] for u, v in edges])
        ref = float(gamma_ref) if gamma_ref is not None else float(gamma_vals.max() + 1e-9)
        widths = np.maximum(edge_width_scale * (gamma_vals / ref), edge_min_width)
        nx.draw_networkx_edges(
            g, layout, edgelist=edges, ax=ax,
            edge_color=gamma_vals, edge_cmap=plt.get_cmap(edge_cmap),
            edge_vmin=0.0, edge_vmax=ref,
            width=widths, alpha=0.85,
        )

    if node_vrange is None:
        m = float(np.max(np.abs(node_metric)) + 1e-6)
        node_vrange = (-m, m)
    nx.draw_networkx_nodes(
        g, layout, ax=ax,
        node_color=node_metric, cmap=plt.get_cmap(node_cmap),
        vmin=node_vrange[0], vmax=node_vrange[1],
        node_size=node_size, edgecolors="black", linewidths=0.6,
    )
    if title is not None:
        ax.set_title(title, fontsize=11)


def animate_population(
    snapshots: list[dict],
    adjacency: np.ndarray,
    layout: dict,
    *,
    node_metric_key: str = "theta_mean",
    feature_idx: int = 0,
    lambda_per_agent: np.ndarray | None = None,
    community_membership: np.ndarray | None = None,
    interval_ms: int = 200,
    figsize: tuple[float, float] = (6.0, 6.0),
    node_cmap: str = "RdBu_r",
    edge_cmap: str = "viridis",
    node_size: float = 60.0,
) -> manim.FuncAnimation:
    """Build a FuncAnimation re-drawing draw_population per snapshot.

    Caller chooses the display: ``HTML(anim.to_jshtml())`` for inline scrubbing,
    ``anim.save("out.gif", writer=PillowWriter(fps=5))`` for a GIF.
    """
    if not snapshots:
        raise ValueError("snapshots list is empty")

    fig, ax = plt.subplots(figsize=figsize)
    plt.close(fig)  # don't show the empty figure when we render the anim

    vrange = _vrange_for(node_metric_key, snapshots, community_membership)
    # Global gamma reference so edge widths/colours are consistent across frames
    gamma_ref = max(float(s["gamma"].max()) for s in snapshots) + 1e-9

    def render(frame_idx: int) -> tuple:
        snap = snapshots[frame_idx]
        metric = _node_metric_from_snapshot(
            snap, node_metric_key, lambda_per_agent, feature_idx, community_membership,
        )
        draw_population(
            ax, adjacency, snap["gamma"], metric, layout,
            node_cmap=node_cmap, edge_cmap=edge_cmap,
            node_vrange=vrange, gamma_ref=gamma_ref,
            node_size=node_size,
            title=f"t = {snap['step']}",
        )
        return ()

    anim = manim.FuncAnimation(
        fig, render, frames=len(snapshots),
        interval=interval_ms, blit=False, repeat=True,
    )
    return anim


def static_strip(
    snapshots: list[dict],
    adjacency: np.ndarray,
    layout: dict,
    *,
    indices: list[int] | None = None,
    node_metric_key: str = "theta_mean",
    feature_idx: int = 0,
    lambda_per_agent: np.ndarray | None = None,
    community_membership: np.ndarray | None = None,
    figsize_per_panel: tuple[float, float] = (4.0, 4.0),
    node_cmap: str = "RdBu_r",
    edge_cmap: str = "viridis",
    node_size: float = 60.0,
) -> "plt.Figure":
    """Render N panels side-by-side from selected snapshot indices.

    Picks 4 evenly-spaced indices by default. Returns the figure; caller can
    save it or display it as desired.
    """
    if not snapshots:
        raise ValueError("snapshots list is empty")
    if indices is None:
        n_panels = 4
        indices = [int(i) for i in np.linspace(0, len(snapshots) - 1, n_panels)]

    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]))
    if n == 1:
        axes = [axes]

    vrange = _vrange_for(node_metric_key, snapshots, community_membership)
    gamma_ref = max(float(s["gamma"].max()) for s in snapshots) + 1e-9

    for ax, idx in zip(axes, indices):
        snap = snapshots[idx]
        metric = _node_metric_from_snapshot(
            snap, node_metric_key, lambda_per_agent, feature_idx, community_membership,
        )
        draw_population(
            ax, adjacency, snap["gamma"], metric, layout,
            node_cmap=node_cmap, edge_cmap=edge_cmap,
            node_vrange=vrange, gamma_ref=gamma_ref,
            node_size=node_size,
            title=f"t = {snap['step']}",
        )
    fig.tight_layout()
    return fig
