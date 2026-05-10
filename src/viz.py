"""Network-evolution visualisation helpers.

Pure-display, numpy input. Used by `notebooks/02_phase_space.ipynb` and
suitable for direct paper-figure rendering. No JAX in the public API.

Three primitives:
- ``fixed_layout``       — one ``spring_layout`` reused across snapshots
- ``draw_population``    — one snapshot of the trust graph
- ``animate_population`` — matplotlib FuncAnimation over a snapshot list
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


def _node_metric_from_snapshot(
    snap: dict,
    kind: str,
    lambda_per_agent: np.ndarray | None,
    feature_idx: int = 0,
) -> np.ndarray:
    """Per-node scalar metric for one snapshot.

    The model is now multi-feature: C has shape (N, R) and q_post (N, R, 2).
    ``feature_idx`` selects which feature φ_r drives the metric.
    """
    if kind == "argmax_q":
        return np.argmax(snap["q_post"][:, feature_idx, :], axis=-1).astype(float)
    if kind == "preference":
        # σ(C[:, r]) ∈ [0, 1] — agent's preference weight on feature r being 1.
        return 1.0 / (1.0 + np.exp(-snap["C"][:, feature_idx]))
    if kind == "lambda":
        if lambda_per_agent is None:
            raise ValueError("node_metric_key='lambda' requires lambda_per_agent kwarg")
        lam = np.asarray(lambda_per_agent)
        return lam[:, feature_idx] if lam.ndim == 2 else lam
    raise ValueError(f"Unknown node_metric_key: {kind!r}")


def _vrange_for(kind: str, snapshots: list[dict] | None) -> tuple[float, float]:
    if kind == "argmax_q":
        return (-0.2, 1.2)
    if kind == "preference":
        return (0.0, 1.0)  # σ(·) is always in [0, 1]
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
    node_metric_key: str = "argmax_q",
    feature_idx: int = 0,
    lambda_per_agent: np.ndarray | None = None,
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

    vrange = _vrange_for(node_metric_key, snapshots)
    # Global gamma reference so edge widths/colours are consistent across frames
    gamma_ref = max(float(s["gamma"].max()) for s in snapshots) + 1e-9

    def render(frame_idx: int) -> tuple:
        snap = snapshots[frame_idx]
        metric = _node_metric_from_snapshot(snap, node_metric_key, lambda_per_agent, feature_idx)
        draw_population(
            ax, adjacency, snap["gamma"], metric, layout,
            node_cmap=node_cmap, edge_cmap=edge_cmap,
            node_vrange=vrange, gamma_ref=gamma_ref,
            node_size=node_size,
            title=f"t = {snap['step']}  (φ_{feature_idx})",
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
    node_metric_key: str = "argmax_q",
    feature_idx: int = 0,
    lambda_per_agent: np.ndarray | None = None,
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

    vrange = _vrange_for(node_metric_key, snapshots)
    gamma_ref = max(float(s["gamma"].max()) for s in snapshots) + 1e-9

    for ax, idx in zip(axes, indices):
        snap = snapshots[idx]
        metric = _node_metric_from_snapshot(snap, node_metric_key, lambda_per_agent, feature_idx)
        draw_population(
            ax, adjacency, snap["gamma"], metric, layout,
            node_cmap=node_cmap, edge_cmap=edge_cmap,
            node_vrange=vrange, gamma_ref=gamma_ref,
            node_size=node_size,
            title=f"t = {snap['step']}  (φ_{feature_idx})",
        )
    fig.tight_layout()
    return fig
