"""Two-field dark-energy model -- the directed Bayes net and its geometry.

The 1998 dark-energy turn read as the paper's structure-learning geometry. One *directed* Bayes net
``x = B x + b + zeta`` (``zeta ~ N(0, diag(s))``) wires the whole thing, and three faces of a single
fact fall out as linear algebra:

  1. DIRECTED SEM PRIMARY.  The symmetric coherence precision is its SHADOW
     ``Pi = (I - B)^T D^{-1} (I - B)`` (``bayesnet.to_info``).
  2. ONE OPERATOR, TWO FIELDS.  ``T = (I - |B|)^{-1}`` gives CONSERVATISM ``kappa = T 1`` (downstream
     mass / carry-over cost) and CONVICTION ``U = T^T u`` (value inherited from cherished
     descendants). GR carries everything (huge kappa) yet Lambda is cheap-yet-cherished -- the two
     fields are linearly independent.
  3. SCHUR DEGENERACY (Eq. 2).  Conditioning on the shared supernova node couples Omega_m and Lambda
     -- the off-diagonal banana every dark-energy contour plot draws.

This is the **pure geometry library** for the ``cosmology_twofield`` experiment and notebook 45: the
DAG definition, the two fields, the Schur banana, and the ``draw_dag`` / ``_short`` drawing helpers
nb45 imports. No ``matplotlib.use``, no file I/O, no ``main`` -- the experiment's own figures, the
asserts, and the saved arrays live in ``experiments/cosmology_twofield.py``.
"""
from __future__ import annotations

import numpy as np

from src.structural import linalg
from src.structural.bayesnet import LinearGaussianBN, from_edges

__all__ = [
    "NODES", "EDGES", "VARIANCES", "UTILITY",
    "build_dark_energy_dag", "propagation_operator", "conservatism", "conviction",
    "utility_vector", "schur_banana", "draw_dag", "_short",
]

# ---- the dark-energy DAG (directed SEM, parents-before-children order) ----
# CORE general_relativity (everything observable hangs below it); DIALS matter_density (Omega_m) +
# dark_energy (Lambda); MECH friedmann_expansion + GR's other consequences; BELT the leaf
# observations; RIVAL mond_prediction (the modified-gravity signal the Lambda-community wants false).
NODES: tuple[str, ...] = (
    "general_relativity",       # 0  core (root)
    "matter_density",           # 1  Omega_m (root dial)
    "dark_energy",              # 2  Lambda  (root dial)
    "friedmann_expansion",      # 3  <- GR, Omega_m, Lambda
    "light_bending",            # 4  <- GR
    "perihelion",               # 5  <- GR
    "grav_redshift",            # 6  <- GR
    "cmb_physics",              # 7  <- GR
    "bbn_physics",              # 8  <- GR
    "distance_modulus",         # 9  <- friedmann_expansion          (mu(z))
    "supernova_obs",            # 10 <- distance_modulus             (SN Ia: the 1998 leaf)
    "lensing_obs",              # 11 <- light_bending
    "mercury_obs",              # 12 <- perihelion
    "clock_obs",                # 13 <- grav_redshift
    "cmb_obs",                  # 14 <- cmb_physics
    "bbn_obs",                  # 15 <- bbn_physics
    "mond_prediction",          # 16 <- matter_density               (rival; wanted false)
)

# Couplings are SIGNED structural-equation weights (they build the Gaussian Pi and the Schur banana).
# Matter and dark energy enter the expansion with OPPOSITE sign -- matter decelerates, Lambda
# accelerates -- so a supernova reading the expansion constrains a *combination*, and clamping it
# leaves the physical (Omega_m, Lambda) banana. The two FIELDS (kappa, U) instead use the coupling
# MAGNITUDES |B| (the paper's positive coupling precisions pi_{uv}), so downstream mass does not
# cancel when a child opposes a parent. Lambda gets a deliberately SHORT single branch; GR parents
# eight, so its downstream mass dwarfs Lambda's.
EDGES: dict[tuple[str, str], float] = {
    ("general_relativity", "friedmann_expansion"): 0.50,
    ("matter_density", "friedmann_expansion"): 0.90,    # matter: + into the expansion read
    ("dark_energy", "friedmann_expansion"): -0.65,      # Lambda: - (opposite) -> the degeneracy
    ("general_relativity", "light_bending"): 0.85,
    ("general_relativity", "perihelion"): 0.85,
    ("general_relativity", "grav_redshift"): 0.85,
    ("general_relativity", "cmb_physics"): 0.85,
    ("general_relativity", "bbn_physics"): 0.85,
    ("friedmann_expansion", "distance_modulus"): 0.95,
    ("distance_modulus", "supernova_obs"): 0.95,
    ("light_bending", "lensing_obs"): 0.90,
    ("perihelion", "mercury_obs"): 0.90,
    ("grav_redshift", "clock_obs"): 0.90,
    ("cmb_physics", "cmb_obs"): 0.90,
    ("bbn_physics", "bbn_obs"): 0.90,
    ("matter_density", "mond_prediction"): 0.80,
}

# A TIGHT latent chain (friedmann -> mu(z) -> SN) so the supernova sharply reads the dials'
# combination (=> a strong, visible degeneracy when clamped); order-1 on the dials + GR machinery.
_TIGHT = {"friedmann_expansion": 0.12, "distance_modulus": 0.10, "supernova_obs": 0.08}
_LEAVES = ("lensing_obs", "mercury_obs", "clock_obs", "cmb_obs", "bbn_obs", "mond_prediction")
VARIANCES = {n: _TIGHT.get(n, 0.25 if n in _LEAVES else 1.0) for n in NODES}

# Intrinsic utility u (the conviction SOURCE): the dark-energy community wants the SN obs (+1),
# values the Lambda commitment (+0.6), and wants the MOND rival FALSE (-0.8). U = T u propagates
# this up to ancestors of the cherished leaves.
UTILITY = {"supernova_obs": 1.0, "dark_energy": 0.6, "mond_prediction": -0.8}


def build_dark_energy_dag() -> LinearGaussianBN:
    """The directed dark-energy Bayes net as explicit CPDs (``x = B x + b + zeta``). Only the
    COVARIANCE structure matters for the two fields and the banana; the means are illustrative.
    Its ``to_info`` is the symmetric coherence precision ``Pi = (I - B)^T D^{-1} (I - B)``."""
    intercepts = {"general_relativity": 1.0, "matter_density": 0.3, "dark_energy": 0.7}
    return from_edges(NODES, EDGES, intercepts=intercepts, variances=VARIANCES)


def propagation_operator(net: LinearGaussianBN) -> np.ndarray:
    """``M = (I - |B|)^{-1}`` -- directed propagation on coupling PRECISIONS (magnitudes). The
    Neumann series ``sum_k |B|^k`` (finite on a DAG); magnitudes so downstream mass does not cancel
    when a child opposes a parent."""
    B = np.abs(np.asarray(net.B))
    d = B.shape[0]
    return np.linalg.inv(np.eye(d) - B)


def conservatism(net: LinearGaussianBN) -> np.ndarray:
    """``kappa = T 1``: the downstream mass (carry-over cost) of every node -- column sums of M."""
    return propagation_operator(net).sum(axis=0)


def conviction(net: LinearGaussianBN, u: np.ndarray | None = None) -> np.ndarray:
    """``U = T^T u``: the same operator on the value source -- the value a commitment inherits from
    the cherished observations below it."""
    if u is None:
        u = utility_vector()
    return propagation_operator(net).T @ np.asarray(u)


def utility_vector() -> np.ndarray:
    return np.array([UTILITY.get(n, 0.0) for n in NODES], dtype=float)


def schur_banana(net: LinearGaussianBN, a: str = "matter_density", b: str = "dark_energy",
                 observed: str = "supernova_obs") -> dict:
    """The 2x2 covariance of ``(a, b)`` PRIOR (marginal -- round) versus CONDITIONED on the shared
    descendant ``observed`` (the banana). Conditioning removes ``observed``'s row/column from the
    precision; the conditional cov is ``inv(Pi[rest, rest])`` and its (a,b) block carries the Schur
    fill-in Eq. (2) manufactures. Returns both 2x2 blocks, their correlations, and the means."""
    gbn = net.to_info()
    Pi = np.asarray(gbn.Pi)
    names = list(net.names)
    ia, ib, io = names.index(a), names.index(b), names.index(observed)

    Sigma = np.linalg.inv(Pi)
    prior = Sigma[np.ix_([ia, ib], [ia, ib])]

    rest = [k for k in range(len(names)) if k != io]
    Pi_rest = Pi[np.ix_(rest, rest)]
    Sigma_cond = np.linalg.inv(Pi_rest)
    ra, rb = rest.index(ia), rest.index(ib)
    post = Sigma_cond[np.ix_([ra, rb], [ra, rb])]

    mu = np.asarray(linalg.info_mean(gbn.Pi, gbn.h))

    def corr(C):
        return float(C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]))

    return {"prior_cov": prior, "post_cov": post,
            "prior_corr": corr(prior), "post_corr": corr(post),
            "mu_a": float(mu[ia]), "mu_b": float(mu[ib]), "a": a, "b": b, "observed": observed}


def _short(n: str) -> str:
    return {"general_relativity": "GR", "matter_density": "Ω_m", "dark_energy": "Λ",
            "friedmann_expansion": "Friedmann", "distance_modulus": "μ(z)",
            "supernova_obs": "SN Ia", "mond_prediction": "MOND"}.get(n, n)


def _layered_positions(net: LinearGaussianBN):
    """Node positions by longest-path depth from a root: GR/dials on top, observations at the bottom
    -- 'the core at the top with everything hanging below it'."""
    B = np.abs(np.asarray(net.B)); d = B.shape[0]
    depth = np.zeros(d, int)
    for j in range(d):
        par = [i for i in range(d) if B[j, i] > 1e-9]
        if par:
            depth[j] = max(depth[i] for i in par) + 1
    pos = {}
    for L in range(int(depth.max()) + 1):
        nodes = [j for j in range(d) if depth[j] == L]
        for k, j in enumerate(nodes):
            pos[j] = (k - (len(nodes) - 1) / 2.0, -float(L))
    return pos, depth


def draw_dag(net: LinearGaussianBN, values: np.ndarray, ax, *, cmap: str,
             diverging: bool = False, title: str = ""):
    """Draw the directed dark-energy DAG with arrows (parent -> child), nodes coloured by ``values``
    (kappa or U). Caller supplies ``ax`` -- matplotlib is imported lazily, so importing this module
    never touches the backend (notebook-safe)."""
    import matplotlib as mpl
    B = np.asarray(net.B); names = net.names; d = B.shape[0]
    pos, _ = _layered_positions(net)
    for j in range(d):
        for i in range(d):
            if abs(B[j, i]) > 1e-9:
                ax.annotate("", xy=pos[j], xytext=pos[i],
                            arrowprops=dict(arrowstyle="-|>", color="0.65", lw=0.8,
                                            shrinkA=7, shrinkB=7))
    if diverging:
        vmax = float(np.abs(values).max()) or 1.0
        norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmp = mpl.colormaps[cmap]
    for j in range(d):
        x, y = pos[j]
        ax.scatter([x], [y], s=360, c=[cmp(norm(values[j]))], edgecolor="k", lw=0.6, zorder=3)
        ax.text(x, y - 0.26, _short(names[j]), ha="center", va="top", fontsize=6.2, zorder=4)
    ax.set_title(title, fontsize=11); ax.axis("off")
    ax.set_ylim(-float(max(p[1] for p in pos.values())) - 1.0 + min(p[1] for p in pos.values()),
                0.6)
