"""Genuine multi-agent structure learning: does network topology govern a structural
revolution (the community prunes the contested phlogiston mass-law structure) versus an
evidential lock-in (it stalls)?

This is the re-runnable, *simulated* lift of nb42. nb42 lifted nb38's conviction-gated
prune to a "population", but its multi-agent layer was thin: Part A shared ONE evidence
stream and only thresholded it by ``lambda_i`` (no interaction); Part B was a one-shot
50/50 Fisher average. So nb42 is a conviction *sweep*, not a simulation -- agents never
actually learn from each other.

Here ``N`` agents sit on a trust graph and **learn from each other by pooling precision**
(``step.fuse``: ``Pi_i <- sum_j W_ij Pi_j``), and the conviction-gated belt prune is read
out per agent. The independent variable is the network **topology**; the load-bearing
mechanism is the genuine precision fusion, not a hard-coded rule.

The mechanism that makes topology matter is **scarce, diffusing evidence**: a small,
low-conviction VANGUARD (``gamma=0``) is the only source that runs the refuting
gravimetric experiment; everyone else SELF-CENSORS (``gamma`` high -> down-weights the
disconfirming mass-balance channel) and holds moderate conviction poised just above the
revolution boundary. The vanguard's disconfirming precision must therefore *flow through
the trust graph* to reach the rest; the graph's connectivity (Fiedler ``lambda_2``) and
the vanguard's placement decide whether it arrives -> revolution vs lock-in.

**Engine refactor (nb44).** The run-loop now lives in the reusable
``src/structural/simulation.run_simulation`` engine; this script supplies the *phlogiston
environment* via ``scenarios.phlogiston_scenario`` and builds the *vanguard* ``AgentSpec``.
``build_substrate`` / ``BELT`` are re-exported here so nb43's import is unchanged, and the
engine lifts nb43's three jit kernels verbatim, so the numbers below are byte-stable.

Run it::

    python scripts/run_multiagent_topology.py

It prints a "mechanisms in a row" readout for one representative run, runs the topology
sweep (the headline F1), the controlled bridge sweep (F2), the vanguard-placement contrast
(F3), and the fusion-mode contrast (F5); writes ``simulation_arrays.npz`` + ``summary.json``
+ diagnostic figures to ``results/structural_multiagent_topology/``; and **asserts the
null/sanity controls**.

Honest-findings caveat (per project norms): this repo has repeatedly found that naive
posterior fusion **washes out** heterogeneity (the "no-pool degeneracy", nb12/nb28). So
*whether topology bites or washes out* is treated as an empirical result to report, not an
assumption -- and the script ships ``fuse_mode`` variants (``posterior`` default,
``deposit_pool``, ``deposit_keep``) that isolate the effect.
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
from matplotlib import cm

try:                                  # Windows consoles default to cp1252; keep unicode alive
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.structural import graphs
from src.structural.phlogiston import StructuralConfig
from src.structural.simulation import run_simulation, AgentSpec
# Re-exported for nb43's import (`from scripts.run_multiagent_topology import build_substrate,
# BELT, ...`) -- the phlogiston substrate now lives in scenarios.py.
from src.structural.scenarios import (
    build_substrate, phlogiston_scenario, Substrate, BELT, DEFAULT_SPURIOUS)

# Calibrated operating point (see the module docstring + scripts/run_multiagent_topology
# calibration). The sweet spot where the vanguard's evidence is SCARCE and DIFFUSING:
#   * sigma_o = 0.5      -- a clean-enough structural signal that the diffusion gradient is
#                           visible rather than buried in observation noise (nb42 used 0.1).
#   * gamma_rest = 1.0   -- the rest are PURE RECEIVERS: they never run the refuting
#                           experiment, so their belt evidence arrives ONLY via fusion from
#                           the vanguard. (HONEST FINDING: at gamma_rest = 0.95 -- nb42's
#                           bubble value -- the 5% residual channel already self-saturates the
#                           belt evidence over the horizon, so topology WASHES OUT. Near-total
#                           self-censorship is what makes topology the load-bearing variable.)
#   * conviction_rest = 0.20 -- poised just below the well-mixed belt evidence (~0.24 of v_e),
#                           so an isolated rest agent (0 evidence) never revolts but a
#                           well-connected one (gets the diffused vanguard evidence) does.
#   * n_vanguard = 6     -- a small persistent source (10% of the population).
CALIB = dict(conviction_rest=0.20, lambda_vanguard=0.05, n_vanguard=6,
             gamma_rest=1.0, sigma_o=0.5, t_shift=40, n_steps=120,
             snapshot_every=5, seed=0)


# ----------------------------------------------------------------------
# Placement of the vanguard on a graph.
# ----------------------------------------------------------------------

def vanguard_indices(graph: graphs.Graph, n_vanguard: int,
                     placement: str, seed: int) -> np.ndarray:
    """Which agents are the vanguard. ``central`` = highest-degree nodes (the well-trusted
    hubs), ``peripheral`` = lowest-degree, ``random`` = a seeded subset, ``block0`` = the
    first community block (needs a community graph). Ties broken by index."""
    A = np.asarray(graph.A)
    deg = A.sum(axis=1)
    n = graph.n
    if placement == "central":
        return np.argsort(-deg, kind="stable")[:n_vanguard]
    if placement == "peripheral":
        return np.argsort(deg, kind="stable")[:n_vanguard]
    if placement == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=n_vanguard, replace=False))
    if placement == "block0":
        if graph.membership is None:
            raise ValueError("placement='block0' needs a community graph (membership)")
        return np.where(np.asarray(graph.membership) == 0)[0][:n_vanguard]
    raise ValueError(f"unknown placement {placement!r}")


# ----------------------------------------------------------------------
# THE SIMULATION: run_world -- a thin wrapper over the reusable engine.
# ----------------------------------------------------------------------

def run_world(graph: graphs.Graph, *, cfg: StructuralConfig | None = None,
              sub: Substrate | None = None,
              conviction_rest: float = 0.20, lambda_vanguard: float = 0.05,
              n_vanguard: int = 6, vanguard_placement: str = "central",
              gamma_rest: float = 1.0, fuse_mode: str = "posterior",
              eta: float = 0.5, n_steps: int = 120, t_shift: int = 40,
              sigma_o: float = 0.5, snapshot_every: int = 5, seed: int = 0) -> dict:
    """Simulate ``N = graph.n`` agents pooling precision over ``graph`` and read out the
    conviction-gated belt prune per agent.

    Every agent holds the SAME over-wired phlogiston prior. The heterogeneity:
      * a ``n_vanguard``-agent vanguard with ``gamma=0`` (runs the refuting gravimetric
        experiment) and low conviction ``lambda_vanguard``;
      * the rest self-censor (``gamma_rest`` -> down-weight the disconfirming channel) and
        hold moderate conviction ``conviction_rest`` (poised just above the revolution
        boundary).
    The only inter-agent influence is the genuine precision fusion ``step.fuse`` (``fuse_mode``).
    The prune is a READOUT of where each fused agent lands (it never mutates the net).

    This builds the phlogiston ``Scenario`` + the vanguard ``AgentSpec`` and delegates the
    rollout to ``simulation.run_simulation``; it then tags on the vanguard-specific keys.
    The returned dict matches the original nb43 ``run_world`` contract.
    """
    cfg = (StructuralConfig(t_shift=t_shift, n_steps=n_steps, sigma_o=sigma_o)
           if cfg is None else cfg)
    if sub is None:
        sub = build_substrate(cfg)
    scenario = phlogiston_scenario(cfg, sub=sub)
    N, m = graph.n, scenario.m

    van = vanguard_indices(graph, n_vanguard, vanguard_placement, seed)
    is_van = np.zeros(N, bool)
    is_van[van] = True

    # per-agent conviction threshold and self-censorship gamma
    lam = np.where(is_van, lambda_vanguard, conviction_rest)               # (N,)
    gamma = np.where(is_van, 0.0, gamma_rest)                              # (N,)

    # per-agent observation weights over the gravimetric rows: down-weight the
    # disconfirming channel by (1 - gamma_i) (vanguard keeps it at full weight).
    w_obs = np.ones((N, m))
    disc = np.asarray(sub.disc_rows, dtype=int)
    w_obs[:, disc] = (1.0 - gamma)[:, None]

    spec = AgentSpec(w_obs=jnp.asarray(w_obs), lam=lam)
    r = run_simulation(scenario, graph, spec, fuse_mode=fuse_mode, eta=eta,
                       snapshot_every=snapshot_every, seed=seed)
    r["vanguard_idx"] = np.asarray(van)
    r["is_vanguard"] = is_van
    r["vanguard_placement"] = vanguard_placement
    return r


# ----------------------------------------------------------------------
# The topology panel and the controlled sweeps.
# ----------------------------------------------------------------------

def topology_panel(N: int = 60, *, sub=None, cfg=None, **kw) -> list[dict]:
    """The F1 panel: one representative graph per family, ordered by connectivity.
    ``isolated`` (lambda2=0) ... ``complete`` (max mixing). community is built then
    lightly bridged so it is connected (lambda2 > 0)."""
    base = graphs.community([N // 3] * 3, intra=0.45, inter=0.0, seed=0)
    panel = [
        ("isolated", base.isolated()),
        ("ring", graphs.ring(N, mean_degree=2)),
        ("erdos_renyi", graphs.erdos_renyi(N, mean_degree=4, seed=0)),
        ("watts_strogatz", graphs.watts_strogatz(N, mean_degree=4, rewiring_p=0.1, seed=0)),
        ("community+bridge", base.with_bridge(inter=0.03)),
        ("complete", graphs.complete(N)),
    ]
    out = []
    for label, g in panel:
        r = run_world(g, sub=sub, cfg=cfg, **kw)
        r["panel_label"] = label
        out.append(r)
    return out


def bridge_sweep(N: int = 60, inters=(0.0, 0.005, 0.01, 0.02, 0.04, 0.08),
                 *, sub=None, cfg=None, **kw) -> list[dict]:
    """F2: the clean knob. Fix the within-block structure (``intra``), vary only the
    cross-block bridge density ``inter`` -> sweeps ``lambda2`` while holding degree roughly
    fixed, de-confounding connectivity from density. Vanguard sits in block 0."""
    base = graphs.community([N // 3] * 3, intra=0.45, inter=0.0, seed=0)
    out = []
    for inter in inters:
        g = base.with_bridge(inter=float(inter))
        r = run_world(g, sub=sub, cfg=cfg, vanguard_placement="block0", **kw)
        r["inter"] = float(inter)
        out.append(r)
    return out


def placement_contrast(N: int = 60, *, sub=None, cfg=None, **kw) -> dict:
    """F3: does WHERE the vanguard sits matter? Central (hub) vs peripheral vanguard on a
    fixed scale-free graph (degree heterogeneity makes the contrast meaningful)."""
    g = graphs.scale_free(N, mean_degree=4, seed=0)
    return {p: run_world(g, sub=sub, cfg=cfg, vanguard_placement=p, **kw)
            for p in ("central", "peripheral")}


# ----------------------------------------------------------------------
# Figures (diagnostic; the polished F1-F5 panel is built in nb43).
# ----------------------------------------------------------------------

def fig_topology(panel: list[dict], path: Path):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8))
    lam2 = [r["lambda2"] for r in panel]
    rev = [r["final_revolted_fraction"] for r in panel]
    kept = [r["kept_t"][-1].mean() for r in panel]
    labels = [r["panel_label"] for r in panel]
    a0.plot(lam2, rev, "o-", color="crimson", lw=2)
    for x, y, l, r in zip(lam2, rev, labels, panel):
        a0.annotate(f"{l}\n<k>={r['mean_degree']:.1f}", (x, y), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    a0.set_xlabel(r"algebraic connectivity $\lambda_2$")
    a0.set_ylabel("final revolted fraction")
    a0.set_title("F1: topology governs revolution vs lock-in")
    a1.plot(lam2, kept, "s-", color="navy", lw=2)
    a1.set_xlabel(r"$\lambda_2$"); a1.set_ylabel("final mean edges kept")
    a1.set_title("structure retained vs connectivity")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def fig_bridge(sweep: list[dict], path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    lam2 = [r["lambda2"] for r in sweep]
    rev = [r["final_revolted_fraction"] for r in sweep]
    ax.plot(lam2, rev, "o-", color="seagreen", lw=2)
    for r in sweep:
        ax.annotate(f"inter={r['inter']:.3f}", (r["lambda2"], r["final_revolted_fraction"]),
                    fontsize=7, textcoords="offset points", xytext=(4, -8))
    ax.set_xlabel(r"$\lambda_2$ (set by bridge density, intra fixed)")
    ax.set_ylabel("final revolted fraction")
    ax.set_title("F2: the clean knob -- bridge density -> connectivity -> revolution")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def fig_dynamics(panel: list[dict], path: Path):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8))
    cmap = cm.viridis
    n = len(panel)
    for i, r in enumerate(panel):
        col = cmap(i / max(n - 1, 1))
        a0.plot(r["snap_t"], r["kept_t"].mean(axis=1), lw=2, color=col,
                label=f"{r['panel_label']} (λ₂={r['lambda2']:.2f})")
        a1.plot(r["snap_t"], r["disagreement_t"], lw=2, color=col)
    a0.set_xlabel("step"); a0.set_ylabel("mean edges kept")
    a0.set_title("F4: structure(t) by topology"); a0.legend(fontsize=7)
    a1.set_xlabel("step"); a1.set_ylabel("residual structural disagreement")
    a1.set_title("persistent disagreement = fragmented field (lock-in)")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------
# Controls + the representative "mechanisms in a row".
# ----------------------------------------------------------------------

def run_controls(sub: Substrate, cfg: StructuralConfig, N: int = 60,
                 **kw) -> dict:
    """The null/sanity controls. Returns a dict of measured numbers; ``main`` asserts the
    robust ones and reports the topology-dependent ones honestly."""
    iso = graphs.complete(N).isolated()
    comp = graphs.complete(N)
    nv = kw.get("n_vanguard", 4)

    out = {}
    # 1. isolated (lambda2 = 0): only the vanguard ever gathers disconfirming evidence.
    out["isolated_revolted"] = run_world(iso, sub=sub, cfg=cfg, **kw)["final_revolted_fraction"]
    out["isolated_expected"] = nv / N
    # 2. complete: maximal spread.
    out["complete_revolted"] = run_world(comp, sub=sub, cfg=cfg, **kw)["final_revolted_fraction"]
    # 3a. conviction_rest -> 0: everyone prunes (no protection).
    kw0 = {**kw, "conviction_rest": 0.0, "lambda_vanguard": 0.0}
    out["conviction0_revolted"] = run_world(comp, sub=sub, cfg=cfg, **kw0)["final_revolted_fraction"]
    # 3b. conviction -> inf for EVERYONE (vanguard too): the echo chamber, none prune.
    kwInf = {**kw, "conviction_rest": 50.0, "lambda_vanguard": 50.0}
    out["convictionInf_revolted"] = run_world(comp, sub=sub, cfg=cfg, **kwInf)["final_revolted_fraction"]
    # 4. gamma_rest = 0 (everyone runs the experiment): topology stops mattering.
    kwg = {**kw, "gamma_rest": 0.0}
    out["gamma0_isolated"] = run_world(iso, sub=sub, cfg=cfg, **kwg)["final_revolted_fraction"]
    out["gamma0_complete"] = run_world(comp, sub=sub, cfg=cfg, **kwg)["final_revolted_fraction"]
    # 5. no refutation (t_shift > n_steps): null world, no revolt anywhere.
    cfg_null = StructuralConfig(t_shift=cfg.n_steps + 10, n_steps=cfg.n_steps, sigma_o=cfg.sigma_o)
    sub_null = build_substrate(cfg_null)
    out["null_revolted"] = run_world(comp, sub=sub_null, cfg=cfg_null, **kw)["final_revolted_fraction"]
    # finiteness / PD of the BMR readout
    out["dF_finite"] = bool(np.isfinite(
        run_world(comp, sub=sub, cfg=cfg, **kw)["final_dF"]).all())
    return out


def mechanisms_in_a_row(rep: dict, sub: Substrate):
    """Print the ordered story of one representative run (a moderately connected graph)."""
    print("\n=== mechanisms in a row (representative: watts-strogatz, central vanguard) ===")
    print(f"  [init]   {rep['n_edges']} edges; all agents hold the SAME over-wired phlogiston prior.")
    print(f"           contested belt (highest v_e): {[sub.edges[i] for i in sub.belt_ix]}; "
          f"balanced λ* = {rep['lstar']:.2f}")
    print(f"  [agents] {rep['vanguard_idx'].size} vanguard (γ=0, λ={rep['lam'][rep['vanguard_idx'][0]]:.2f}) "
          f"vs {rep['lam'].size - rep['vanguard_idx'].size} self-censoring rest "
          f"(γ high, λ={rep['lam'][~rep['is_vanguard']][0]:.2f}); λ₂={rep['lambda2']:.2f}")
    rv = rep["revolted_t"]
    van = rep["vanguard_idx"]
    rest = ~rep["is_vanguard"]
    print(f"  [t=0]    revolted: vanguard {rv[0][van].mean():.2f}, rest {rv[0][rest].mean():.2f} "
          f"(nobody has the disconfirming evidence yet)")
    mid = len(rep["snap_t"]) // 2
    print(f"  [shift]  the world flips at t_shift; the vanguard runs the refuting experiment and "
          f"its precision begins to FLOW over the trust graph")
    print(f"  [mid]    revolted: vanguard {rv[mid][van].mean():.2f}, rest {rv[mid][rest].mean():.2f} "
          f"(evidence diffusing through fusion)")
    print(f"  [final]  revolted: vanguard {rv[-1][van].mean():.2f}, rest {rv[-1][rest].mean():.2f}; "
          f"population fraction {rep['final_revolted_fraction']:.2f}")
    print(f"  [field]  residual structural disagreement {rep['disagreement_t'][0]:.2f} -> "
          f"{rep['disagreement_t'][-1]:.2f} (stays > 0 = fragmented field / partial lock-in)")


# ----------------------------------------------------------------------
# main: run the sweep, assert controls, save arrays + figures + summary.
# ----------------------------------------------------------------------

def main() -> int:
    N = 60
    cfg = StructuralConfig(t_shift=CALIB["t_shift"], n_steps=CALIB["n_steps"],
                           sigma_o=CALIB["sigma_o"])
    sub = build_substrate(cfg)
    out_dir = ROOT / "results" / "structural_multiagent_topology"
    out_dir.mkdir(parents=True, exist_ok=True)

    kw = dict(CALIB)                       # the calibrated operating point

    print(f"substrate: {sub.lstar=:.3f}; v_e(belt)={[round(float(sub.v_e[i]),2) for i in sub.belt_ix]}; "
          f"v_e range [{sub.v_e.min():.2f}, {sub.v_e.max():.2f}]")

    # ---- representative run + mechanisms in a row ----
    rep_graph = graphs.watts_strogatz(N, mean_degree=4, rewiring_p=0.1, seed=0)
    rep = run_world(rep_graph, sub=sub, cfg=cfg, vanguard_placement="central", **kw)
    mechanisms_in_a_row(rep, sub)

    # ---- F1: the topology panel ----
    print("\n=== F1: topology sweep (final revolted fraction vs λ₂) ===")
    panel = topology_panel(N, sub=sub, cfg=cfg, vanguard_placement="central", **kw)
    for r in panel:
        print(f"  {r['panel_label']:>16}: λ₂={r['lambda2']:6.3f}  <k>={r['mean_degree']:4.1f}  "
              f"revolted={r['final_revolted_fraction']:.3f}  kept={r['kept_t'][-1].mean():.2f}")
    fig_topology(panel, out_dir / "diag_topology_sweep.png")
    fig_dynamics(panel, out_dir / "diag_dynamics.png")

    # ---- F2: controlled bridge sweep ----
    print("\n=== F2: bridge sweep (intra fixed, vary inter) ===")
    sweep = bridge_sweep(N, sub=sub, cfg=cfg, **kw)
    for r in sweep:
        print(f"  inter={r['inter']:.3f}: λ₂={r['lambda2']:6.3f}  revolted={r['final_revolted_fraction']:.3f}")
    fig_bridge(sweep, out_dir / "diag_bridge_sweep.png")

    # ---- F3: vanguard placement ----
    print("\n=== F3: vanguard placement (scale-free graph) ===")
    place = placement_contrast(N, sub=sub, cfg=cfg, **kw)
    for p, r in place.items():
        print(f"  {p:>10} vanguard: revolted={r['final_revolted_fraction']:.3f}")

    # ---- F5: fusion-mode contrast on one graph ----
    print("\n=== F5: fusion-mode contrast (community+bridge graph) ===")
    gmid = graphs.community([N // 3] * 3, intra=0.45, inter=0.0, seed=0).with_bridge(0.03)
    modes = {}
    for mode in ("posterior", "deposit_pool", "deposit_keep"):
        r = run_world(gmid, sub=sub, cfg=cfg, fuse_mode=mode, vanguard_placement="block0", **kw)
        modes[mode] = r["final_revolted_fraction"]
        print(f"  {mode:>13}: revolted={r['final_revolted_fraction']:.3f}")

    # ---- controls ----
    print("\n=== null / sanity controls ===")
    ctrl = run_controls(sub, cfg, N, **kw)
    print(f"  isolated (λ₂=0):        revolted={ctrl['isolated_revolted']:.3f}  "
          f"(expect ≈ n_vanguard/N = {ctrl['isolated_expected']:.3f})")
    print(f"  complete:               revolted={ctrl['complete_revolted']:.3f}  (expect ≥ isolated)")
    print(f"  conviction_rest=0:      revolted={ctrl['conviction0_revolted']:.3f}  (expect ≈ 1.00)")
    print(f"  conviction_rest=50:     revolted={ctrl['convictionInf_revolted']:.3f}  (expect ≈ 0.00)")
    print(f"  gamma_rest=0 isolated:  revolted={ctrl['gamma0_isolated']:.3f}")
    print(f"  gamma_rest=0 complete:  revolted={ctrl['gamma0_complete']:.3f}  "
          f"(expect ≈ each other: topology stops mattering)")
    print(f"  null world (no shift):  revolted={ctrl['null_revolted']:.3f}  (expect 0.00)")
    print(f"  ΔF finite/PD:           {ctrl['dF_finite']}")

    # robust assertions (structurally guaranteed); topology-dependent ones reported above.
    assert ctrl["dF_finite"], "BMR readout produced non-finite ΔF (PD violation)"
    assert ctrl["null_revolted"] == 0.0, \
        f"null world (no refutation) revolted {ctrl['null_revolted']} (expected 0)"
    assert ctrl["convictionInf_revolted"] == 0.0, \
        f"echo chamber (λ→∞) revolted {ctrl['convictionInf_revolted']} (expected 0)"
    assert ctrl["conviction0_revolted"] > 0.95, \
        f"λ=0 should prune everywhere, got {ctrl['conviction0_revolted']}"
    assert ctrl["complete_revolted"] >= ctrl["isolated_revolted"] - 1e-9, \
        "complete graph spread less than isolated (mixing should not reduce revolt)"
    # lambda2 cross-check
    assert abs(graphs.algebraic_connectivity(graphs.complete(N)) - N) < 1e-3, \
        "algebraic_connectivity(complete) should be N"

    # ---- save arrays + summary ----
    def pack(runs, key):
        return np.array([r[key] for r in runs])

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        # F1
        panel_labels=np.array([r["panel_label"] for r in panel]),
        panel_lambda2=pack(panel, "lambda2"),
        panel_mean_degree=pack(panel, "mean_degree"),
        panel_revolted=pack(panel, "final_revolted_fraction"),
        panel_kept_final=np.array([r["kept_t"][-1].mean() for r in panel]),
        panel_snap_t=panel[0]["snap_t"],
        panel_kept_t=np.stack([r["kept_t"].mean(axis=1) for r in panel]),
        panel_disagreement_t=np.stack([r["disagreement_t"] for r in panel]),
        panel_m_t=np.stack([r["m_t"] for r in panel]),
        # F2
        bridge_inter=pack(sweep, "inter"),
        bridge_lambda2=pack(sweep, "lambda2"),
        bridge_revolted=pack(sweep, "final_revolted_fraction"),
        # F3
        place_central=place["central"]["final_revolted_fraction"],
        place_peripheral=place["peripheral"]["final_revolted_fraction"],
        place_central_revolted_t=place["central"]["revolted_t"].mean(axis=1),
        place_peripheral_revolted_t=place["peripheral"]["revolted_t"].mean(axis=1),
        # F5
        mode_labels=np.array(list(modes.keys())),
        mode_revolted=np.array(list(modes.values())),
        # representative
        rep_kept_t=rep["kept_t"], rep_revolted_t=rep["revolted_t"],
        rep_snap_t=rep["snap_t"], rep_lam=rep["lam"], rep_is_vanguard=rep["is_vanguard"],
        rep_v_e=sub.v_e, rep_edges=np.array(rep["edges"]),
    )

    summary = {
        "config": dict(N=N, **kw),
        "substrate": {"n_edges": len(sub.edges),
                      "belt_edges": [list(e) for e in BELT],
                      "v_e_belt": [float(sub.v_e[i]) for i in sub.belt_ix],
                      "lstar": float(sub.lstar)},
        "F1_topology": [{"graph": r["panel_label"], "lambda2": r["lambda2"],
                         "mean_degree": r["mean_degree"],
                         "revolted": r["final_revolted_fraction"]} for r in panel],
        "F2_bridge": [{"inter": r["inter"], "lambda2": r["lambda2"],
                       "revolted": r["final_revolted_fraction"]} for r in sweep],
        "F3_placement": {p: r["final_revolted_fraction"] for p, r in place.items()},
        "F5_fuse_mode": modes,
        "controls": ctrl,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figures / summary to {out_dir}")
    print(f"HEADLINE: revolted fraction over the topology panel = "
          f"{[round(r['final_revolted_fraction'], 2) for r in panel]}")
    spread = panel[-1]["final_revolted_fraction"] - panel[0]["final_revolted_fraction"]
    if spread > 0.1:
        print(f"  => topology BITES: revolt rises {spread:+.2f} from isolated to complete.")
    else:
        print(f"  => topology effect is small ({spread:+.2f}); the honest finding is wash-out "
              f"(see F5 for the deposit-mode contrast).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
