"""Reusable structural plotting registry.

This module provides a small registry of high-value plotting functions with a
shared callable interface and explicit I/O metadata. Every plot function
accepts ``save_path`` and returns ``PlotOutput`` so callers can use the same
plot repeatedly with different data and parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

try:
    import networkx as nx  # optional dependency
except Exception:  # pragma: no cover - optional import
    nx = None


ArrayLike = np.ndarray | list[float] | list[int]


@dataclass
class PlotOutput:
    """Standard output payload for all plot functions."""

    path: Path | None
    figure: plt.Figure
    axes: Any
    metadata: dict[str, Any]


def _as_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _finalize_output(
    fig: plt.Figure,
    axes: Any,
    metadata: dict[str, Any],
    save_path: str | Path | None,
) -> PlotOutput:
    out_path: Path | None = None
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return PlotOutput(path=out_path, figure=fig, axes=axes, metadata=metadata)


def _resolve_layout(
    adjacency: np.ndarray,
    node_positions: dict[int, tuple[float, float]] | np.ndarray | None,
    seed: int,
) -> dict[int, tuple[float, float]]:
    n = int(adjacency.shape[0])
    if node_positions is not None:
        if isinstance(node_positions, dict):
            missing = [i for i in range(n) if i not in node_positions]
            if missing:
                raise ValueError(f"node_positions dict missing nodes: {missing}")
            return {i: tuple(node_positions[i]) for i in range(n)}
        pos_arr = _as_array("node_positions", node_positions)
        if pos_arr.shape != (n, 2):
            raise ValueError(
                f"node_positions array must have shape {(n, 2)}, got {pos_arr.shape}"
            )
        return {i: (float(pos_arr[i, 0]), float(pos_arr[i, 1])) for i in range(n)}

    if nx is not None:
        graph = nx.from_numpy_array(adjacency)
        pos = nx.spring_layout(graph, seed=int(seed))
        return {i: (float(pos[i][0]), float(pos[i][1])) for i in range(n)}

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return {i: (float(np.cos(t)), float(np.sin(t))) for i, t in enumerate(theta)}


def plot_community_bayesnet_snapshots(
    Pi_t: ArrayLike,
    h_t: ArrayLike,
    adjacency_t: ArrayLike | None = None,
    node_positions: dict[int, tuple[float, float]] | np.ndarray | None = None,
    layout_seed: int = 0,
    edge_threshold: float = 1e-8,
    snapshot_indices: list[int] | None = None,
    max_snapshots: int = 4,
    agent_reduce: str = "mean",
    cmap: str = "viridis",
    show_graph: bool = True,
    graph_only: bool = False,
    save_path: str | Path | None = None,
) -> PlotOutput:
    """Plot time snapshots of community Bayes-net parameters.

    Supported shapes:
    - Pi_t: (T, N, d, d) or (T, d, d)
    - h_t:  (T, N, d) or (T, d)
    - adjacency_t: optional (N, N) or (T, N, N)
    """

    Pi = _as_array("Pi_t", Pi_t)
    h = _as_array("h_t", h_t)

    if Pi.ndim not in (3, 4):
        raise ValueError(f"Pi_t must have ndim 3 or 4, got shape {Pi.shape}")
    if h.ndim not in (2, 3):
        raise ValueError(f"h_t must have ndim 2 or 3, got shape {h.shape}")

    if Pi.ndim == 4:
        if agent_reduce != "mean":
            raise ValueError("agent_reduce currently supports only 'mean'")
        Pi = Pi.mean(axis=1)
    if h.ndim == 3:
        if agent_reduce != "mean":
            raise ValueError("agent_reduce currently supports only 'mean'")
        h = h.mean(axis=1)

    if Pi.ndim != 3 or h.ndim != 2:
        raise ValueError("Pi_t and h_t could not be reduced to (T,d,d) and (T,d)")

    t_pi, d0, d1 = Pi.shape
    t_h, d_h = h.shape
    if d0 != d1:
        raise ValueError(f"Pi_t last two dims must be square, got {(d0, d1)}")
    if t_pi != t_h or d0 != d_h:
        raise ValueError(
            f"Pi_t and h_t time/latent dims must match, got Pi_t={Pi.shape}, h_t={h.shape}"
        )

    T = t_pi
    if snapshot_indices is None:
        k = min(max_snapshots, T)
        snapshot_indices = np.linspace(0, T - 1, num=k, dtype=int).tolist()
    if not snapshot_indices:
        raise ValueError("snapshot_indices must not be empty")
    if any((idx < 0 or idx >= T) for idx in snapshot_indices):
        raise ValueError(f"snapshot_indices must be in [0, {T - 1}]")

    adj_series = None
    if adjacency_t is not None:
        adj = _as_array("adjacency_t", adjacency_t).astype(float)
        if adj.ndim == 2:
            if adj.shape[0] != adj.shape[1]:
                raise ValueError(f"adjacency_t must be square, got {adj.shape}")
            adj_series = np.repeat(adj[np.newaxis, :, :], T, axis=0)
        elif adj.ndim == 3:
            if adj.shape[0] != T or adj.shape[1] != adj.shape[2]:
                raise ValueError(
                    f"adjacency_t with time axis must have shape (T,N,N) with T={T}, got {adj.shape}"
                )
            adj_series = adj
        else:
            raise ValueError("adjacency_t must be 2D or 3D")
    else:
        abs_pi = np.abs(Pi)
        adj_series = (abs_pi > float(edge_threshold)).astype(float)
        for t_idx in range(T):
            np.fill_diagonal(adj_series[t_idx], 0.0)

    n_nodes = int(adj_series.shape[1])
    if adj_series.shape[2] != n_nodes:
        raise ValueError(f"adjacency_t must be square, got {adj_series.shape}")
    positions = _resolve_layout(adj_series[snapshot_indices[0]], node_positions, seed=layout_seed)

    cols = len(snapshot_indices)
    effective_show_graph = True if graph_only else show_graph
    n_rows = 1 if graph_only else (3 if effective_show_graph else 2)
    fig, axes = plt.subplots(n_rows, cols, figsize=(4 * cols, 3.2 * n_rows), squeeze=False)
    for col, t_idx in enumerate(snapshot_indices):
        if not graph_only:
            ax0 = axes[0, col]
            im = ax0.imshow(Pi[t_idx], cmap=cmap, aspect="auto")
            ax0.set_title(f"Pi at t={t_idx}")
            ax0.set_xlabel("latent")
            ax0.set_ylabel("latent")
            fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

            ax1 = axes[1, col]
            ax1.bar(np.arange(d0), h[t_idx])
            ax1.set_title(f"h at t={t_idx}")
            ax1.set_xlabel("latent")
            ax1.set_ylabel("value")

        if effective_show_graph:
            ax2 = axes[0, col] if graph_only else axes[2, col]
            a = adj_series[t_idx]
            for i in range(n_nodes):
                xi, yi = positions[i]
                for j in range(n_nodes):
                    if i == j or abs(a[i, j]) <= 0.0:
                        continue
                    xj, yj = positions[j]
                    ax2.plot([xi, xj], [yi, yj], color="0.75", linewidth=0.8, zorder=1)
            node_vals = h[t_idx]
            if node_vals.shape[0] != n_nodes:
                node_vals = np.resize(node_vals, n_nodes)
            xy = np.array([positions[i] for i in range(n_nodes)], dtype=float)
            sc = ax2.scatter(
                xy[:, 0],
                xy[:, 1],
                c=node_vals,
                cmap="coolwarm",
                s=120,
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )
            for i in range(n_nodes):
                ax2.text(xy[i, 0], xy[i, 1], str(i), ha="center", va="center", fontsize=8, zorder=3)
            ax2.set_title(f"Graph view at t={t_idx}")
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_aspect("equal", adjustable="box")
            fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("Community Bayes-net snapshots")
    fig.tight_layout()
    return _finalize_output(
        fig,
        axes,
        {
            "plot": "plot_community_bayesnet_snapshots",
            "time_steps": T,
            "latent_dim": d0,
            "snapshot_indices": snapshot_indices,
            "agent_reduced": True,
            "layout_seed": int(layout_seed),
            "n_nodes": n_nodes,
            "graph_only": bool(graph_only),
            "show_graph": bool(effective_show_graph),
        },
        save_path,
    )


def plot_edge_edit_timeline(
    edge_count_t: ArrayLike,
    edge_edit_delta_t: ArrayLike | None = None,
    accepted_t: ArrayLike | None = None,
    save_path: str | Path | None = None,
) -> PlotOutput:
    """Plot edge-count evolution and edge-edit activity over time."""

    edge_count = _as_array("edge_count_t", edge_count_t).astype(float)
    if edge_count.ndim != 1:
        raise ValueError(f"edge_count_t must be 1D, got shape {edge_count.shape}")
    T = edge_count.shape[0]
    t_axis = np.arange(T)

    signed_delta = None
    if edge_edit_delta_t is not None:
        delta = _as_array("edge_edit_delta_t", edge_edit_delta_t).astype(float)
        if delta.ndim == 1:
            if delta.shape[0] != T:
                raise ValueError(
                    f"edge_edit_delta_t length must match T={T}, got {delta.shape[0]}"
                )
            signed_delta = delta
        elif delta.ndim == 2:
            if delta.shape[0] != T:
                raise ValueError(
                    f"edge_edit_delta_t first dim must match T={T}, got {delta.shape[0]}"
                )
            signed_delta = delta.sum(axis=1)
        else:
            raise ValueError("edge_edit_delta_t must be 1D or 2D")

    accepted = None
    if accepted_t is not None:
        accepted = _as_array("accepted_t", accepted_t).astype(float)
        if accepted.shape != (T,):
            raise ValueError(f"accepted_t must have shape ({T},), got {accepted.shape}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t_axis, edge_count, color="tab:blue", linewidth=2)
    axes[0].set_ylabel("edge count")
    axes[0].set_title("Edge count timeline")
    axes[0].grid(alpha=0.3)

    if signed_delta is not None:
        colors = np.where(signed_delta >= 0.0, "tab:green", "tab:red")
        axes[1].bar(t_axis, signed_delta, color=colors, alpha=0.8, label="signed edit delta")
    if accepted is not None:
        axes[1].plot(t_axis, accepted, color="black", linewidth=1.5, label="accepted")
    axes[1].axhline(0.0, color="gray", linewidth=1)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("edit activity")
    axes[1].set_title("Edge edits over time")
    if signed_delta is not None or accepted is not None:
        axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return _finalize_output(
        fig,
        axes,
        {
            "plot": "plot_edge_edit_timeline",
            "time_steps": T,
            "has_delta": signed_delta is not None,
            "has_accepted": accepted is not None,
        },
        save_path,
    )


def plot_log_evidence_race(
    log_evidence_tk: ArrayLike,
    hypothesis_names: list[str] | None = None,
    normalize_first_step: bool = True,
    save_path: str | Path | None = None,
) -> PlotOutput:
    """Plot per-hypothesis log-evidence trajectories over time."""

    log_evidence = _as_array("log_evidence_tk", log_evidence_tk).astype(float)
    if log_evidence.ndim != 2:
        raise ValueError(f"log_evidence_tk must be 2D (T,K), got {log_evidence.shape}")

    T, K = log_evidence.shape
    if hypothesis_names is None:
        hypothesis_names = [f"H{k}" for k in range(K)]
    if len(hypothesis_names) != K:
        raise ValueError(
            f"hypothesis_names length must match K={K}, got {len(hypothesis_names)}"
        )

    y = log_evidence.copy()
    if normalize_first_step:
        y = y - y[0:1, :]

    winner = np.argmax(log_evidence, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(T)

    for k in range(K):
        ax.plot(x, y[:, k], linewidth=2, label=hypothesis_names[k])

    ax.set_xlabel("time")
    ax.set_ylabel("log evidence" + (" (offset)" if normalize_first_step else ""))
    ax.set_title("Log-evidence race across hypotheses")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", ncol=min(4, K))

    fig.tight_layout()
    return _finalize_output(
        fig,
        ax,
        {
            "plot": "plot_log_evidence_race",
            "time_steps": T,
            "n_hypotheses": K,
            "winner_index_t": winner.tolist(),
        },
        save_path,
    )


def plot_mean_field_hypothesis_share(
    posterior_tnk: ArrayLike | None = None,
    winner_tn: ArrayLike | None = None,
    hypothesis_names: list[str] | None = None,
    save_path: str | Path | None = None,
) -> PlotOutput:
    """Plot mean-field share of each hypothesis over time.

    Provide exactly one of:
    - posterior_tnk with shape (T,N,K)
    - winner_tn with shape (T,N), containing winner indices in [0, K-1]
    """

    if (posterior_tnk is None) == (winner_tn is None):
        raise ValueError("Provide exactly one of posterior_tnk or winner_tn")

    if posterior_tnk is not None:
        post = _as_array("posterior_tnk", posterior_tnk).astype(float)
        if post.ndim != 3:
            raise ValueError(f"posterior_tnk must be 3D (T,N,K), got {post.shape}")
        if np.any(post < 0.0):
            raise ValueError("posterior_tnk must be non-negative")
        norm = post.sum(axis=2, keepdims=True)
        if np.any(norm <= 0.0):
            raise ValueError("posterior_tnk rows must have positive sum over K")
        post = post / norm
        shares = post.mean(axis=1)
        T, K = shares.shape
    else:
        winners = _as_array("winner_tn", winner_tn).astype(int)
        if winners.ndim != 2:
            raise ValueError(f"winner_tn must be 2D (T,N), got {winners.shape}")
        T, _ = winners.shape
        max_idx = int(winners.max())
        min_idx = int(winners.min())
        if min_idx < 0:
            raise ValueError("winner_tn indices must be >= 0")
        K = max_idx + 1
        shares = np.zeros((T, K), dtype=float)
        for t in range(T):
            counts = np.bincount(winners[t], minlength=K)
            shares[t] = counts / max(1, counts.sum())

    if hypothesis_names is None:
        hypothesis_names = [f"H{k}" for k in range(K)]
    if len(hypothesis_names) != K:
        raise ValueError(
            f"hypothesis_names length must match K={K}, got {len(hypothesis_names)}"
        )

    x = np.arange(T)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.stackplot(x, shares.T, labels=hypothesis_names, alpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time")
    ax.set_ylabel("mean-field share")
    ax.set_title("Mean-field hypothesis share")
    ax.legend(loc="upper left", ncol=min(4, K))
    ax.grid(alpha=0.25)

    fig.tight_layout()
    return _finalize_output(
        fig,
        ax,
        {
            "plot": "plot_mean_field_hypothesis_share",
            "time_steps": T,
            "n_hypotheses": K,
        },
        save_path,
    )


def plot_community_distance_heatmap(
    distance_cc: ArrayLike,
    community_labels: list[str] | None = None,
    time_index: int = -1,
    cmap: str = "magma",
    save_path: str | Path | None = None,
) -> PlotOutput:
    """Plot a community distance heatmap.

    Supported shapes:
    - distance_cc: (C, C)
    - distance_cc: (T, C, C), then ``time_index`` selects one matrix.
    """

    dist = _as_array("distance_cc", distance_cc).astype(float)
    if dist.ndim == 3:
        T, c0, c1 = dist.shape
        if c0 != c1:
            raise ValueError(f"distance_cc must be square in last dims, got {dist.shape}")
        if not (-T <= time_index < T):
            raise ValueError(f"time_index out of range for T={T}: {time_index}")
        dist = dist[time_index]
    elif dist.ndim != 2:
        raise ValueError(f"distance_cc must be 2D or 3D, got shape {dist.shape}")

    C0, C1 = dist.shape
    if C0 != C1:
        raise ValueError(f"distance_cc must be square, got {dist.shape}")
    C = C0

    if community_labels is None:
        community_labels = [f"C{i}" for i in range(C)]
    if len(community_labels) != C:
        raise ValueError(
            f"community_labels length must match C={C}, got {len(community_labels)}"
        )

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(dist, cmap=cmap, aspect="equal")
    ax.set_xticks(np.arange(C), labels=community_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(C), labels=community_labels)
    ax.set_title("Community distance heatmap")
    ax.set_xlabel("community")
    ax.set_ylabel("community")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    return _finalize_output(
        fig,
        ax,
        {
            "plot": "plot_community_distance_heatmap",
            "n_communities": C,
            "time_index": time_index,
        },
        save_path,
    )


def _plot_empty_graph(
    adjacency: np.ndarray,
    node_positions: dict[int, tuple[float, float]] | np.ndarray | None = None,
    seed: int = 0,
) -> dict[int, tuple[float, float]]:
    """Return deterministic node positions for a graph adjacency.

    This helper exists so external callers can reuse deterministic defaults.
    """

    adjacency = _as_array("adjacency", adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square 2D, got {adjacency.shape}")
    return _resolve_layout(adjacency, node_positions=node_positions, seed=seed)


PlotFn = Callable[..., PlotOutput]


PLOT_REGISTRY: dict[str, PlotFn] = {
    "plot_community_bayesnet_snapshots": plot_community_bayesnet_snapshots,
    "plot_edge_edit_timeline": plot_edge_edit_timeline,
    "plot_log_evidence_race": plot_log_evidence_race,
    "plot_mean_field_hypothesis_share": plot_mean_field_hypothesis_share,
    "plot_community_distance_heatmap": plot_community_distance_heatmap,
}


PLOT_IO: dict[str, dict[str, Any]] = {
    "plot_community_bayesnet_snapshots": {
        "inputs": {
            "Pi_t": "(T,N,d,d) or (T,d,d)",
            "h_t": "(T,N,d) or (T,d)",
            "snapshot_indices": "optional list[int]",
            "save_path": "str | Path | None",
        },
        "outputs": "PlotOutput(path, figure, axes, metadata)",
        "notes": "Averages over agents when N axis is present.",
    },
    "plot_edge_edit_timeline": {
        "inputs": {
            "edge_count_t": "(T,)",
            "edge_edit_delta_t": "optional (T,) or (T,E)",
            "accepted_t": "optional (T,)",
            "save_path": "str | Path | None",
        },
        "outputs": "PlotOutput(path, figure, axes, metadata)",
        "notes": "Shows edge count plus signed edit activity.",
    },
    "plot_log_evidence_race": {
        "inputs": {
            "log_evidence_tk": "(T,K)",
            "hypothesis_names": "optional list[str]",
            "save_path": "str | Path | None",
        },
        "outputs": "PlotOutput(path, figure, axes, metadata)",
        "notes": "Tracks per-hypothesis evidence trajectories and winners.",
    },
    "plot_mean_field_hypothesis_share": {
        "inputs": {
            "posterior_tnk": "optional (T,N,K)",
            "winner_tn": "optional (T,N)",
            "hypothesis_names": "optional list[str]",
            "save_path": "str | Path | None",
        },
        "outputs": "PlotOutput(path, figure, axes, metadata)",
        "notes": "Requires exactly one of posterior_tnk or winner_tn.",
    },
    "plot_community_distance_heatmap": {
        "inputs": {
            "distance_cc": "(C,C) or (T,C,C)",
            "community_labels": "optional list[str]",
            "time_index": "int, used when time axis exists",
            "save_path": "str | Path | None",
        },
        "outputs": "PlotOutput(path, figure, axes, metadata)",
        "notes": "Displays inter-community distance matrix as heatmap.",
    },
}


def render_plot(name: str, **kwargs: Any) -> PlotOutput:
    """Dispatch plot calls by name using ``PLOT_REGISTRY``."""

    try:
        fn = PLOT_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PLOT_REGISTRY))
        raise ValueError(f"Unknown plot '{name}'. Available: {available}") from exc
    return fn(**kwargs)


__all__ = [
    "PlotOutput",
    "plot_community_bayesnet_snapshots",
    "plot_edge_edit_timeline",
    "plot_log_evidence_race",
    "plot_mean_field_hypothesis_share",
    "plot_community_distance_heatmap",
    "PLOT_REGISTRY",
    "PLOT_IO",
    "render_plot",
]
