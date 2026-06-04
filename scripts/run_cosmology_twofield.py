"""The new modeling lens, made concrete on the dark-energy paradigm (nb44, Part 0).

This is the worked example David and Jonas were circling: the 1998 dark-energy turn read
as the paper's structure-learning geometry. One *directed* Bayes net wires the whole thing,
and three faces of a single fact fall out of it as linear algebra rather than philosophy:

  1. DIRECTED SEM IS PRIMARY.  The agent holds a generative model ``x = B x + b + zeta``
     (``zeta ~ N(0, diag(s))``) -- general relativity at the root, every predicted observable
     hanging below it.  The symmetric "coherence" precision the undirected reading uses is its
     SHADOW, recovered by the exact bridge ``Pi = (I - B)^T diag(1/s) (I - B)`` (``bayesnet.to_info``).
     We generate and propagate with the directed form; we read coherence off the symmetric one.

  2. ONE OPERATOR, TWO FIELDS.  The propagation operator is ``T = (I - B)^{-1}`` (the Neumann
     series over directed paths, finite because the DAG is nilpotent).  Apply it to the unit
     source and you get CONSERVATISM ``kappa = T 1`` -- the downstream mass of a commitment, the
     carry-over cost of revising it.  Apply the SAME operator to the value source and you get
     CONVICTION ``U = T u`` -- the value a commitment inherits from the cherished observations
     below it.  GR has everything below it -> huge ``kappa`` -> the field flips the cheap dial
     (Lambda) and not the expensive ancestor (GR): Lakatos's belt-protects-core, straight out of
     the operator.  And ``kappa`` and ``U`` are linearly independent (they are ``T 1`` and ``T u``):
     Lambda is CHEAP yet CHERISHED; GR is EXPENSIVE yet comparatively value-neutral.

  3. THE SCHUR DEGENERACY (Eq. 2).  Omega_m and Lambda are independent a priori (round cloud).
     CONDITION ON THE SHARED OBSERVATION -- clamp the supernova node -- and an off-diagonal
     coupling appears between them where the joint had a zero: the cloud collapses onto the
     tilted "banana" every dark-energy contour plot draws and calls "the degeneracy direction."
     A shared latent hub marginalized, or a shared observation conditioned on, are the two
     readings of the SAME Schur complement.

Run it::

    python scripts/run_cosmology_twofield.py

It builds the DAG, computes ``(kappa, U)`` and the Schur banana, ASSERTS the three claims
(``kappa(GR) >> kappa(Lambda)``; ``U`` inverts the ``kappa`` order so the two fields are
decoupled; conditioning manufactures the off-diagonal), saves arrays + figures to
``results/cosmology_twofield/``, and prints the readout.  nb44 imports the builders here and
draws the polished figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.structural import linalg
from src.structural.bayesnet import LinearGaussianBN, from_edges


# ======================================================================
# 1. The dark-energy DAG (directed SEM, parents-before-children order).
# ======================================================================

# Node roles:
#   CORE   general_relativity      -- the deep commitment; everything observable hangs below it.
#   DIALS  matter_density (Omega_m), dark_energy (Lambda) -- the two cosmological parameters.
#   MECH   friedmann_expansion     -- the expansion history GR+Omega_m+Lambda jointly fix.
#          light_bending / perihelion / grav_redshift / cmb / bbn -- GR's *other* consequences,
#          the branches that make GR's descendant set (hence kappa) enormous.
#   BELT   distance_modulus, and the leaf OBSERVATIONS (supernova_obs, lensing_obs, ...).
#   RIVAL  mond_prediction         -- the modified-gravity signal the Lambda-community wants FALSE.
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

# Coupling weights are POSITIVE (they are coupling *precisions* -- how tightly bound a child is
# to a parent). The physical SIGN of an effect (Lambda accelerates, matter decelerates) lives in
# the means/utilities, not in the coupling magnitude that sets kappa. Lambda gets a deliberately
# SHORT, single branch (it "only disturbs the distance-redshift relation"); GR parents eight
# branches, so its downstream mass dwarfs Lambda's.
# Couplings are SIGNED here (the structural-equation weights that build the Gaussian Pi and the
# Schur degeneracy). Crucially, matter and dark energy enter the expansion with OPPOSITE sign --
# matter decelerates, Lambda accelerates -- so a supernova that reads the expansion constrains a
# *combination* of the two, and clamping it leaves the physical positive-slope degeneracy (the
# real (Omega_m, Lambda) banana). The two FIELDS (kappa, U) instead use the coupling MAGNITUDES
# |B| (the paper's positive coupling precisions pi_{uv}), so a node's downstream mass does not
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

# Per-node residual variances: a TIGHT latent chain (friedmann -> mu(z) -> SN) so the supernova is
# a sharp read of the dials' combination (=> a strong, visible degeneracy when clamped); order-1
# on the dials and GR's other machinery.
_TIGHT = {"friedmann_expansion": 0.12, "distance_modulus": 0.10, "supernova_obs": 0.08}
_LEAVES = ("lensing_obs", "mercury_obs", "clock_obs", "cmb_obs", "bbn_obs", "mond_prediction")
VARIANCES = {n: _TIGHT.get(n, 0.25 if n in _LEAVES else 1.0) for n in NODES}

# Intrinsic utility u (the conviction SOURCE). The dark-energy community is attached to the
# accelerating-universe reading: it WANTS the supernova observation it built its case on (+1),
# values the Lambda commitment itself (+0.6), and WANTS the modified-gravity rival FALSE (-0.8).
# Everything else is value-neutral. Conviction U = T u then propagates this UP to the ancestors of
# the cherished leaves -- so GR inherits some value (it is upstream of the SN), but Lambda inherits
# more *relative to its tiny cost*.
UTILITY = {"supernova_obs": 1.0, "dark_energy": 0.6, "mond_prediction": -0.8}


def build_dark_energy_dag() -> LinearGaussianBN:
    """The directed dark-energy Bayes net as explicit CPDs (``x = B x + b + zeta``).

    Means are set so GR/observation nodes sit near 1 and the dials near their concordance-ish
    values; only the COVARIANCE structure matters for the two fields and the Schur banana, so the
    means are illustrative. Returns a :class:`LinearGaussianBN` whose ``to_info`` is the symmetric
    coherence precision ``Pi = (I - B)^T D^{-1} (I - B)``."""
    intercepts = {"general_relativity": 1.0, "matter_density": 0.3, "dark_energy": 0.7}
    return from_edges(NODES, EDGES, intercepts=intercepts, variances=VARIANCES)


# ======================================================================
# 2. The two fields: kappa = T 1 (conservatism) and U = T u (conviction).
# ======================================================================

def propagation_operator(net: LinearGaussianBN) -> np.ndarray:
    """``M = (I - |B|)^{-1}`` -- the directed propagation operator on the coupling PRECISIONS
    (magnitudes). ``M[w, i]`` is the precision-weighted coupling propagated from ancestor ``i`` to
    descendant ``w`` along all directed paths (the Neumann series ``sum_k |B|^k``, finite on a DAG).
    Magnitudes, not signed weights, so a node's downstream mass does not cancel when a child opposes
    a parent (matches the paper's positive coupling precisions ``pi_{uv}`` in ``kappa = T 1``)."""
    B = np.abs(np.asarray(net.B))
    d = B.shape[0]
    return np.linalg.inv(np.eye(d) - B)


def conservatism(net: LinearGaussianBN) -> np.ndarray:
    """``kappa = T 1``: the downstream mass of every node (its own unit source + the
    precision-weighted mass of its descendants). ``kappa_i = sum_w M[w, i]`` = the carry-over cost
    of revising node ``i`` -- large in the core, small on the belt. (Column sums of ``M``.)"""
    return propagation_operator(net).sum(axis=0)


def conviction(net: LinearGaussianBN, u: np.ndarray | None = None) -> np.ndarray:
    """``U = T u``: the same operator on the value source. ``U_i = u_i + sum_{w in desc(i)}
    pi_{i->w} u_w`` = the value a commitment inherits from the cherished observations below it.
    (``M^T u``.)"""
    if u is None:
        u = utility_vector()
    return propagation_operator(net).T @ np.asarray(u)


def utility_vector() -> np.ndarray:
    return np.array([UTILITY.get(n, 0.0) for n in NODES], dtype=float)


# ======================================================================
# 3. The Schur degeneracy: conditioning on the supernova couples the parents.
# ======================================================================

def schur_banana(net: LinearGaussianBN, a: str = "matter_density", b: str = "dark_energy",
                 observed: str = "supernova_obs") -> dict:
    """The 2x2 covariance of ``(a, b)`` PRIOR (marginal -- round) versus CONDITIONED on the shared
    descendant ``observed`` (the banana).

    Prior marginal cov = the (a,b) block of ``Sigma = Pi^{-1}``; since the two dials are
    independent roots it is diagonal. Conditioning on ``observed`` removes its row/column from the
    precision (Gaussian conditioning), and the conditional cov of the remaining nodes is
    ``inv(Pi[rest, rest])``; its (a,b) block carries the induced off-diagonal -- the Schur fill-in
    Eq. (2) manufactures. Returns both 2x2 blocks, their correlation coefficients, and the means."""
    gbn = net.to_info()
    Pi = np.asarray(gbn.Pi)
    h = np.asarray(gbn.h)
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


# ======================================================================
# 4. Figures.
# ======================================================================

def _short(n: str) -> str:
    return {"general_relativity": "GR", "matter_density": "Ω_m", "dark_energy": "Λ",
            "friedmann_expansion": "Friedmann", "distance_modulus": "μ(z)",
            "supernova_obs": "SN Ia", "mond_prediction": "MOND"}.get(n, n)


def fig_two_fields(net: LinearGaussianBN, path: Path):
    kap = conservatism(net)
    U = conviction(net)
    names = list(net.names)
    order = np.argsort(-kap)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 5.2))
    y = np.arange(len(names))
    a0.barh(y, kap[order], color="steelblue")
    a0.set_yticks(y); a0.set_yticklabels([_short(names[i]) for i in order], fontsize=8)
    a0.invert_yaxis(); a0.set_xlabel("conservatism  κ = T·1  (carry-over cost)")
    a0.set_title("κ: GR carries everything; Λ disturbs only μ(z)")
    cols = ["seagreen" if v > 0 else ("crimson" if v < 0 else "0.6") for v in U[order]]
    a1.barh(y, U[order], color=cols)
    a1.set_yticks(y); a1.set_yticklabels([_short(names[i]) for i in order], fontsize=8)
    a1.invert_yaxis(); a1.axvline(0, color="k", lw=0.6)
    a1.set_xlabel("conviction  U = T·u  (propagated value)")
    a1.set_title("U: Λ is cheap yet cherished; GR central yet value-neutral")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def fig_decoupling(net: LinearGaussianBN, path: Path):
    kap = conservatism(net); U = conviction(net); names = list(net.names)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(kap, U, s=40, color="0.5", zorder=2)
    for i, n in enumerate(names):
        if n in ("general_relativity", "dark_energy", "matter_density", "light_bending",
                 "friedmann_expansion", "supernova_obs", "mond_prediction"):
            ax.annotate(_short(n), (kap[i], U[i]), fontsize=9, fontweight="bold",
                        textcoords="offset points", xytext=(5, 4))
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("conservatism  κ = T·1")
    ax.set_ylabel("conviction  U = T·u")
    r = np.corrcoef(kap, U)[0, 1]
    ax.set_title(f"Two fields, one operator: κ and U are independent (r = {r:.2f})")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def _ellipse(ax, cov, mu, color, label):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, np.argmax(vals)], vecs[0, np.argmax(vals)]))
    from matplotlib.patches import Ellipse
    for k in (1.0, 2.0):
        w, h = 2 * k * np.sqrt(np.maximum(vals, 1e-12))
        ax.add_patch(Ellipse(mu, w, h, angle=ang, fill=(k == 1.0), alpha=0.18 if k == 1 else 1.0,
                             edgecolor=color, facecolor=color, lw=1.6))
    ax.plot([], [], color=color, lw=2, label=label)


def fig_banana(net: LinearGaussianBN, path: Path):
    s = schur_banana(net)
    mu = (s["mu_a"], s["mu_b"])
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    _ellipse(ax, s["prior_cov"], mu, "0.55", f"prior (marginal): r = {s['prior_corr']:.2f}")
    _ellipse(ax, s["post_cov"], mu, "crimson", f"given SN Ia: r = {s['post_corr']:.2f}")
    ax.set_xlabel("Ω_m  (matter density)"); ax.set_ylabel("Λ  (dark energy)")
    ax.set_title("Schur degeneracy: clamping the supernova node\ncouples the two dials (the banana)")
    ax.legend(loc="best", fontsize=9); ax.set_aspect("equal", "datalim")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def _layered_positions(net: LinearGaussianBN):
    """Node positions by longest-path depth from a root: GR/dials on top, observations at the
    bottom -- 'the core at the top with everything hanging below it' (the chat's picture)."""
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
    """Draw the directed dark-energy DAG with arrows (parent -> child), nodes coloured by
    ``values`` (kappa or U). The arrows make the generative direction visible -- GR at the
    root, every observation a descendant -- so ``kappa = T 1`` is legible at a glance."""
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


def fig_dag_fields(net: LinearGaussianBN, path: Path):
    """The two fields on the actual network: κ (cost, OrRd) and U (value, RdYlGn)."""
    kap = conservatism(net); U = conviction(net)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(16, 6.2))
    draw_dag(net, kap, a0, cmap="OrRd",
             title="κ = T·1  (conservatism / carry-over cost): GR hot, leaves cold")
    draw_dag(net, U, a1, cmap="RdYlGn", diverging=True,
             title="U = T·u  (conviction): Λ & SN green, MOND red, machinery neutral")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


# ======================================================================
# 5. main: compute, assert the three claims, save.
# ======================================================================

def main() -> int:
    out_dir = ROOT / "results" / "cosmology_twofield"
    out_dir.mkdir(parents=True, exist_ok=True)

    net = build_dark_energy_dag()
    names = list(net.names)
    kap = conservatism(net)
    U = conviction(net)
    s = schur_banana(net)

    i_gr, i_lam = names.index("general_relativity"), names.index("dark_energy")
    i_lb = names.index("light_bending")

    print("=== dark-energy two-field readout ===")
    print(f"{'node':>22} {'kappa':>8} {'U':>8}")
    for i in np.argsort(-kap):
        print(f"{names[i]:>22} {kap[i]:8.3f} {U[i]:8.3f}")

    print("\n--- the three claims ---")
    ratio = kap[i_gr] / kap[i_lam]
    print(f"1. kappa(GR)={kap[i_gr]:.2f}  kappa(Lambda)={kap[i_lam]:.2f}  ratio={ratio:.2f}"
          f"  (GR is the costliest commitment, Lambda among the cheapest dials)")
    print(f"2. U(Lambda)={U[i_lam]:.2f} > U(GR)={U[i_gr]:.2f}  while  kappa(Lambda) < kappa(GR)"
          f"  => the value order INVERTS the cost order: the fields are decoupled")
    print(f"   global corr(kappa, U) = {np.corrcoef(kap, U)[0,1]:.2f}")
    print(f"3. corr(Omega_m, Lambda): prior={s['prior_corr']:.3f} -> given SN={s['post_corr']:.3f}"
          f"  (conditioning manufactured the off-diagonal)")

    # figures
    fig_two_fields(net, out_dir / "twofield_bars.png")
    fig_decoupling(net, out_dir / "twofield_decoupling.png")
    fig_banana(net, out_dir / "schur_banana.png")
    fig_dag_fields(net, out_dir / "twofield_dag.png")

    # assertions
    assert kap[i_gr] == kap.max(), "GR should be the costliest node (max kappa)"
    assert ratio > 3.0, f"kappa(GR) should dwarf kappa(Lambda); ratio {ratio:.2f}"
    assert U[i_lam] > U[i_gr], "Lambda should be more cherished than GR (decoupling)"
    assert kap[i_lam] < kap[i_gr], "Lambda should be cheaper than GR (decoupling)"
    assert abs(s["prior_corr"]) < 0.05, f"prior Omega_m/Lambda should be ~uncorrelated, got {s['prior_corr']:.3f}"
    assert abs(s["post_corr"]) > 0.2, f"conditioning on SN should couple the dials, got {s['post_corr']:.3f}"

    # save
    np.savez_compressed(
        out_dir / "twofield_arrays.npz",
        names=np.array(names), kappa=kap, U=U, utility=utility_vector(),
        B=np.asarray(net.B), Pi=np.asarray(net.to_info().Pi),
        prior_cov=s["prior_cov"], post_cov=s["post_cov"],
    )
    summary = {
        "kappa": {n: float(kap[i]) for i, n in enumerate(names)},
        "U": {n: float(U[i]) for i, n in enumerate(names)},
        "kappa_GR_over_Lambda": float(ratio),
        "corr_kappa_U": float(np.corrcoef(kap, U)[0, 1]),
        "schur": {"prior_corr": s["prior_corr"], "post_corr": s["post_corr"]},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figures / summary to {out_dir}")
    print("ALL THREE CLAIMS HOLD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
