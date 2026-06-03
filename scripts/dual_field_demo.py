"""Visual demo of dual fields: revision cost vs effective utility."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.structural.dual_field import PrecisionUtilityNet


def _normalize(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _draw_panel(ax, xy, edges, values, names, title):
    for i, j in edges:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="0.75", lw=1.2, zorder=1)

    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=values,
        cmap="viridis",
        s=650,
        edgecolors="black",
        linewidths=1.0,
        zorder=2,
        vmin=0.0,
        vmax=1.0,
    )
    for i, name in enumerate(names):
        ax.text(xy[i, 0], xy[i, 1], name, ha="center", va="center", fontsize=9, color="white", zorder=3)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    return sc


def main() -> int:
    names = (
        "core_A",
        "core_B",
        "core_C",
        "core_D",
        "belt_E",
        "belt_F",
        "belt_G",
        "belt_H",
    )

    Pi = jnp.array(
        [
            [4.8, -1.0, -0.7, -0.6, -0.5, 0.0, 0.0, -0.3],
            [-1.0, 4.6, -0.8, -0.7, -0.4, -0.4, 0.0, 0.0],
            [-0.7, -0.8, 4.7, -0.9, 0.0, -0.3, -0.5, 0.0],
            [-0.6, -0.7, -0.9, 4.9, -0.2, 0.0, -0.4, -0.6],
            [-0.5, -0.4, 0.0, -0.2, 3.0, -0.3, 0.0, 0.0],
            [0.0, -0.4, -0.3, 0.0, -0.3, 2.9, -0.2, 0.0],
            [0.0, 0.0, -0.5, -0.4, 0.0, -0.2, 3.1, -0.2],
            [-0.3, 0.0, 0.0, -0.6, 0.0, 0.0, -0.2, 3.2],
        ]
    )
    h = jnp.array([0.5, 0.2, -0.1, 0.0, 0.2, -0.4, 0.1, -0.2])
    intrinsic_u = jnp.array([2.5, 1.3, -1.8, 0.4, 0.8, -0.7, 0.3, -1.2])

    net = PrecisionUtilityNet(names=names, Pi=Pi, h=h, u=intrinsic_u, alpha=0.5)
    cost = np.asarray(net.cost_field())
    eff_u = np.asarray(net.effective_utility())

    cost_n = _normalize(cost)
    eff_u_n = _normalize(eff_u)

    core_xy = np.array([[-0.6, 0.6], [0.6, 0.6], [0.6, -0.6], [-0.6, -0.6]])
    belt_xy = np.array([[-1.3, 1.2], [1.3, 1.2], [1.3, -1.2], [-1.3, -1.2]])
    xy = np.vstack([core_xy, belt_xy])

    Pi_np = np.asarray(Pi)
    edges = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names)) if abs(Pi_np[i, j]) > 1e-12]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    sc0 = _draw_panel(axes[0], xy, edges, cost_n, names, "Normalized revision-cost field")
    sc1 = _draw_panel(axes[1], xy, edges, eff_u_n, names, "Normalized effective-utility field")
    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dual_field_demo.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
