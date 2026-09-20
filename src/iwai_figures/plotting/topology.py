"""Figure 9: the original graph row and overlaid trajectories, at print size."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .orx_figstyle import PALETTE
from .style import WIDTH, GREY, typography


def graph_positions(name, n):
    """Deterministic layout only: all nodes and archived edges are retained."""
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.column_stack((np.cos(angle), np.sin(angle)))
    if name == "wheel":
        angle = np.linspace(0, 2 * np.pi, n - 1, endpoint=False)
        positions[0] = 0
        positions[1:] = np.column_stack((np.cos(angle), np.sin(angle)))
    elif name == "community":
        # Separate the actual 4/20 blocks so the seven bridges can be seen.
        for ids, centre, radius in ((np.arange(4), (-.78, 0), .23),
                                    (np.arange(4, n), (.37, 0), .66)):
            angle = np.linspace(0, 2 * np.pi, len(ids), endpoint=False)
            positions[ids] = np.array(centre) + radius * np.column_stack((np.cos(angle), np.sin(angle)))
    return positions


def figure9(data, params, confidence_interval):
    fig = plt.figure(figsize=(WIDTH, 3.65))
    grid = fig.add_gridspec(2, 4, height_ratios=(.91, 2.12), left=.125, right=.965,
                           top=.91, bottom=.135, hspace=.39, wspace=.22)
    ax = fig.add_subplot(grid[1, :])
    vanguard = np.asarray(data["vanguard_nodes"])
    seeds = np.asarray(data["seeds"])
    representative = int(np.flatnonzero(seeds == 0)[0])
    summary = {
        "seed_count": len(seeds), "seeds": seeds.tolist(), "vanguard_nodes": vanguard.tolist(),
        "layout": "four exact graph diagrams above one shared trajectory panel",
        "band": "pointwise 95% Student t interval for the arithmetic mean of 50 seed-level conservative means; df49; untruncated",
        "individual_agent_traces": "all 20 conservative agents from seed0 in each graph; thin lines are not seed-level uncertainty",
        "representative_seed": 0, "plotting_smoothing": "none",
        "seed_pairing": "same 50 seed ids; marginal intervals, not paired-difference intervals",
        "graph_drawing": "all archived nodes and edges; layouts changed for clarity; black outlines mark the four vanguard nodes",
        "typography": typography(),
        "graphs": {}, "params": params,
    }
    bounds = []
    conditions = [
        ("cycle", "Cycle", PALETTE["red"], "-"),
        ("wheel", "Wheel", PALETTE["orange"], "--"),
        ("community", "Communities", PALETTE["blue"], "-."),
        ("complete", "Complete", PALETTE["green"], ":"),
    ]
    for i, (name, label, color, linestyle) in enumerate(conditions):
        graph_ax = fig.add_subplot(grid[0, i])
        adjacency = np.asarray(data[f"A_{name}"])
        q = np.asarray(data[f"q_{name}"])
        if q.shape != (50, 600, 24) or adjacency.shape != (24, 24):
            raise ValueError("Unexpected graph/trajectory dimensions")
        positions = graph_positions(name, len(adjacency))
        edges = np.column_stack(np.where(np.triu(adjacency, 1) > 0))
        # Dense graphs keep every edge but lower their visual weight.
        segments = positions[edges]
        graph_ax.add_collection(LineCollection(segments, colors="#B0B0B0",
                                linewidths=.36 if len(edges)<60 else .25,
                                alpha=.75 if len(edges)<60 else .38, zorder=0))
        graph_ax.scatter(*positions.T, s=10, color=color, edgecolors="none", zorder=2)
        graph_ax.scatter(*positions[vanguard].T, s=15, color=color,
                         edgecolors="#202020", linewidths=.65, zorder=3)
        graph_ax.set(xlim=(-1.18, 1.18), ylim=(-1.18, 1.18), aspect="equal")
        graph_ax.axis("off")
        graph_ax.text(.5, 1.19, label, color=color, transform=graph_ax.transAxes,
                      ha="center", va="bottom", fontsize=8.5)
        graph_ax.plot([.22,.78],[1.10,1.10], color=color, ls=linestyle,
                      lw=1.4, transform=graph_ax.transAxes, clip_on=False)
        graph_ax.text(.5, -.14, rf"$\lambda_2={float(data[f'lambda2_{name}']):.2f}$",
                      transform=graph_ax.transAxes, ha="center", va="top", fontsize=8)

        conservative = np.setdiff1d(np.arange(q.shape[2]), vanguard)
        seed_means = q[:, :, conservative].mean(axis=2)
        mean, lo, hi = confidence_interval(seed_means)
        t = np.arange(q.shape[1])
        for agent in conservative:
            ax.plot(t, q[representative, :, agent], color=color, lw=.38, alpha=.18, zorder=1)
        ax.fill_between(t, lo, hi, color=color, alpha=.13, linewidth=0, zorder=2)
        ax.plot(t, mean, color=color, ls=linestyle, lw=1.55, zorder=3)
        bounds.extend((float(lo.min()), float(hi.max())))
        hits = q[:, :, conservative] > params["Q_CONS"]
        hits[:, :params["T_SHIFT"]] = False
        crossed = hits.any(axis=1)
        crossing = hits.argmax(axis=1)[crossed]
        support = crossed.size
        # A missing crossing is a right-censored observation, not an event at T.
        # With a common censoring horizon, this order statistic is the KM median.
        median_index = int(np.ceil(support / 2)) - 1
        median_crossing = (float(np.sort(crossing)[median_index])
                           if len(crossing) > median_index else None)
        summary["graphs"][name] = {
            "lambda2": float(data[f"lambda2_{name}"]), "edge_count": int(len(edges)),
            "drawing_edge_count": int(len(segments)), "positions": positions.tolist(),
            "line_color": color, "line_style": linestyle,
            "median_agent_first_crossing": median_crossing,
            "median_among_crossing_agent_seed_pairs": float(np.median(crossing)) if len(crossing) else None,
            "crossing_threshold": params["Q_CONS"],
            "noncrossing_agent_seed_pairs": int((~crossed).sum()),
            "censoring": "Crossings counted after the world shift only; noncrossers right-censored at 600. Population median null if not reached; event-only median separately labelled.",
            "final_mean": float(mean[-1]), "band_minimum": float(lo.min()), "band_maximum": float(hi.max()),
            "mean_trajectory": mean.tolist(),
        }
    ax.axvline(params["T_SHIFT"], color=GREY, ls=":", lw=.9, zorder=0)
    ax.set(xlim=(0, 600), ylim=(min(-.035, min(bounds)-.012), max(1.035,max(bounds)+.012)),
           xticks=[0,100,200,300,400,500,600], yticks=[0,.25,.5,.75,1],
           xlabel="Simulation step", ylabel=r"Conservative $q(\mathrm{oxygen})$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E1E1E1", lw=.4)
    ax.set_axisbelow(True)
    return fig, summary
