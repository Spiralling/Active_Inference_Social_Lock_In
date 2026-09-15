"""The Schur residue, generalized: the banana without the cosmology.

Renders ``fig_schur_generic.png`` -- the visual intuition for the operator appendix's
Schur paragraph, on the smallest nets that show it. Two panels, dual operations:

  (a) conditioning on a shared EFFECT (collider x1 -> y <- x2): the parents are a priori
      independent; clamping y manufactures an anti-correlation (explaining-away);
  (b) marginalizing a shared CAUSE (b -> x1, b -> x2, the matrix of Eq. eq:schur): the
      children are conditionally independent given b; integrating b out leaves them coupled.

Both fill in an off-diagonal coupling where the joint precision had a zero. No domain
labels: the panels are the generic geometry the dark-energy banana used to illustrate.

Deterministic (pure closed-form linear algebra, no RNG); registered so the paper figure
carries a provenance manifest like every simulated figure. Formerly
``scripts/make_fig_schur_generic.py``, which wrote straight into ``paper/figures/`` --
it now writes to ``results/schur_generic/`` and ``scripts/collect_paper_figures.py``
copies it over.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from experiments.registry import ExperimentSpec, register


def _corr(cov: np.ndarray) -> float:
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def _draw_ellipses(ax, cov, color, label):
    vals, vecs = np.linalg.eigh(cov)
    major = int(np.argmax(vals))
    ang = np.degrees(np.arctan2(vecs[1, major], vecs[0, major]))
    for k in (1.0, 2.0):
        w = 2 * k * np.sqrt(max(vals[major], 1e-12))      # width axis = major eigenvector
        h = 2 * k * np.sqrt(max(vals[1 - major], 1e-12))
        ax.add_patch(Ellipse((0, 0), w, h, angle=ang, fill=(k == 1.0),
                             alpha=0.18 if k == 1.0 else 1.0,
                             edgecolor=color, facecolor=color, lw=1.6))
    ax.plot([], [], color=color, lw=2, label=label)


def run(out_dir, params: dict) -> None:
    # (a) collider: x = Bx + noise with y = x1 + x2 + zeta.
    #     Joint precision (x1, x2, y) = (I-B)^T D (I-B).
    B = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]], dtype=float)
    D = np.diag(np.asarray(params["COLLIDER_NOISE_PRECISIONS"], dtype=float))
    Pi_collider = (np.eye(3) - B).T @ D @ (np.eye(3) - B)
    prior_cov = np.linalg.inv(Pi_collider)[:2, :2]          # marginal over (x1, x2)
    given_y_cov = np.linalg.inv(Pi_collider[:2, :2])        # condition on y: take the block
    assert abs(prior_cov[0, 1]) < 1e-12  # parents a priori independent (the zero)

    # (b) common cause: the exact matrix of Eq. (eq:schur) over (x1, x2, b).
    Pi_cause = np.array(params["PI_COMMON_CAUSE"], dtype=float)
    given_b_cov = np.linalg.inv(Pi_cause[:2, :2])            # condition on b
    # marginal precision = Schur complement of b, exactly the RHS of Eq. (eq:schur)
    marg_cov = np.linalg.inv(Pi_cause[:2, :2]
                             - np.outer(Pi_cause[:2, 2], Pi_cause[2, :2]) / Pi_cause[2, 2])

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.6, 5.6))

    _draw_ellipses(a0, prior_cov, "0.55", f"marginal: r = {_corr(prior_cov):.2f}")
    _draw_ellipses(a0, given_y_cov, "crimson", f"given $y$: r = {_corr(given_y_cov):.2f}")
    a0.set_title("conditioning on a shared effect\n$x_1 \\rightarrow y \\leftarrow x_2$:"
                 " clamp $y$, the parents anti-couple")
    a0.set_xlabel("$x_1$"); a0.set_ylabel("$x_2$")

    _draw_ellipses(a1, given_b_cov, "0.55", f"given $b$: r = {_corr(given_b_cov):.2f}")
    _draw_ellipses(a1, marg_cov, "steelblue", f"$b$ marginalized: r = {_corr(marg_cov):.2f}")
    a1.set_title("marginalizing a shared cause\n$x_1 \\leftarrow b \\rightarrow x_2$:"
                 " integrate $b$ out, the children couple")
    a1.set_xlabel("$x_1$"); a1.set_ylabel("$x_2$")

    for ax in (a0, a1):
        ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
        ax.set_aspect("equal", "datalim"); ax.legend(loc="upper right", fontsize=9)
        lim = 1.6
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    fig.suptitle("The Schur residue: a coupling appears where the joint precision had a zero",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_schur_generic.png", dpi=120)
    plt.close(fig)

    summary = {
        "collider": {"prior_r": _corr(prior_cov), "given_y_r": _corr(given_y_cov)},
        "common_cause": {"given_b_r": _corr(given_b_cov), "marginalized_r": _corr(marg_cov)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"collider:     prior r = {_corr(prior_cov):+.2f} -> given y      "
          f"r = {_corr(given_y_cov):+.2f}")
    print(f"common cause: given b r = {_corr(given_b_cov):+.2f} -> marginalized "
          f"r = {_corr(marg_cov):+.2f}")


register(ExperimentSpec(
    model="cosmology",
    name="schur_generic",
    description="The Schur residue on the smallest nets that show it (paper: "
                "fig_schur_generic): conditioning on a shared effect anti-couples the "
                "parents; marginalizing a shared cause couples the children -- a coupling "
                "appears where the joint precision had a zero. Deterministic closed-form "
                "linear algebra; the generic geometry behind the dark-energy banana.",
    run=run,
    out_dir="schur_generic",
    params=dict(
        COLLIDER_NOISE_PRECISIONS=(2.0, 2.0, 2.0),
        PI_COMMON_CAUSE=((2.0, 0.0, 1.0), (0.0, 2.0, 1.0), (1.0, 1.0, 2.0)),
    ),
))
