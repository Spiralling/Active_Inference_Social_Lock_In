"""Lens B -- a changing cosmology world on the reusable engine: do communities keep
re-tracking each successive theory, or collapse and get stuck? And which *diversity lever*
keeps them able to re-track -- network **disconnection** or **conviction**?

The world is three genuinely different cosmology theories (the existing ``landscape_presets``
Bayes-net presets) that the truth cycles through in three epochs:

    [0, T1)  dark matter   ->   [T1, T2)  modified gravity   ->   [T2, n)  scale-variant laws

``N`` heterogeneous agents in three communities (differing in *conservatism* = prior
precision, and *conviction* = a per-community value tilt toward that community's favoured
theory) sit on a trust graph and each step OBSERVE the current world and FUSE precision over
the graph -- the *same* ``simulation.run_simulation`` engine the phlogiston experiment (nb43)
uses, with ``scenarios.cosmology_scenario`` plugged in. We sweep the two diversity levers
head-to-head:

    (a) DISCONNECTION  -- block transmission (community ``inter`` density -> Fiedler lambda_2)
    (b) CONVICTION     -- block revision (per-community tilt ``h <- h + lambda U`` toward home)

as a 2x2 (neither / disconnection / conviction / both), multi-seed, and read out:

  1. **which theory each community holds over time vs the true theory** (project the mean
     belief onto the three candidate theories -- the headline "tracking vs stuck" picture);
  2. **structural diversity** (``shells.residual_disagreement`` -- high = diverse, 0 = consensus);
  3. an **honest structural caveat** -- the relational edges accumulate *uniformly* (the Fisher
     deposit ``H^T H`` is operator-set, not data-set), so theory identity is carried by the
     MEANS, not the learned couplings; genuine data-driven structure re-growth is the job of
     Lens A (``scripts/run_cosmology_regrowth.py``).

Run it::

    python scripts/run_cosmology_tracking.py

It prints the wall probe ("mechanisms in a row"), the 2x2 cell table, asserts the null/sanity
controls, and writes ``simulation_arrays.npz`` + ``summary.json`` + diagnostic figures to
``results/cosmology_tracking/``.

NO FORGETTING (per design): agents accumulate as in nb43. With pure accumulation, early-epoch
evidence is heavy by later epochs, so absolute tracking LAGS (~one epoch) and late-epoch
tracking is harder *by construction* -- read the lever comparison as **relative** diversity
preservation, not as absolute correctness. (A disconnected community that ends on the
eventually-true theory is committed to it, not *right*.)
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

import matplotlib                       # backend set to "Agg" inside main() (not at import,
import matplotlib.pyplot as plt         # so importing this module in a notebook keeps %inline)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.structural import graphs
from src.structural.simulation import run_simulation, AgentSpec
from src.structural import scenarios as sc
from src.structural.landscape_presets import ClusterProfile, sample_cluster_scalings


# ----------------------------------------------------------------------
# Configuration -- the world, the population, the levers.
# ----------------------------------------------------------------------

N_AGENTS = 150
N_STEPS, T1, T2 = 180, 60, 120
SIGMA_O = 0.5
SEEDS = (0, 1, 2)

INTER_CONNECTED = 0.08        # bridged communities  (lambda_2 > 0)
INTER_DISCONNECTED = 0.0      # echo chambers         (lambda_2 = 0)
INTRA = 0.5
CONVICTION_LEVEL = 4.0        # the tilt strength when the conviction lever is ON

# Three communities, by conservatism (prior precision) and conviction (utility scale).
PROFILES: tuple[ClusterProfile, ...] = (
    ClusterProfile("conservative", 1 / 3, precision_mean=0.9, precision_sd=0.15,
                   utility_scale_mean=1.2, utility_scale_sd=0.2),
    ClusterProfile("moderate", 1 / 3, precision_mean=0.2, precision_sd=0.15,
                   utility_scale_mean=0.6, utility_scale_sd=0.2),
    ClusterProfile("frontier", 1 / 3, precision_mean=-0.4, precision_sd=0.15,
                   utility_scale_mean=0.2, utility_scale_sd=0.2),
)
# Each community is committed (when the conviction lever is on) to a DIFFERENT theory: the
# conservative establishment to the OLD incumbent (dark matter), the frontier to the
# eventually-true theory (scale-variant). The narrative blocs.
HOMES = ("dark_matter", "modified_gravity", "scale_variant_laws")
COMM_LABELS = tuple(p.label for p in PROFILES)
EPOCH_NAMES = sc.COSMOLOGY_EPOCHS         # ("dark_matter","modified_gravity","scale_variant_laws")


# ----------------------------------------------------------------------
# Population + spec builders (importable by the notebook).
# ----------------------------------------------------------------------

def cosmology_population(n_agents: int = N_AGENTS,
                         profiles: tuple[ClusterProfile, ...] = PROFILES,
                         seed: int = 0) -> dict:
    """Sample per-agent conservatism (``precision_scale``) and conviction strength
    (``utility_scale``), block-ordered so community ``k`` is contiguous (membership == cluster
    id). Reuses ``landscape_presets.sample_cluster_scalings`` then sorts by cluster so the graph
    blocks line up with the belief communities."""
    samp = sample_cluster_scalings(n_agents, profiles, seed=seed)
    order = np.argsort(np.asarray(samp.cluster_id), kind="stable")
    ps = np.asarray(samp.precision_scale)[order]
    us = np.asarray(samp.utility_scale)[order]
    cid = np.asarray(samp.cluster_id)[order]
    sizes = [int((cid == k).sum()) for k in range(len(profiles))]
    return {"precision_scale": ps, "utility_scale": us, "cluster_id": cid, "sizes": sizes}


def cosmology_spec(scn: sc.Scenario, pop: dict, *, conviction: float = 0.0,
                   homes: tuple[str, ...] = HOMES) -> AgentSpec:
    """Build the per-agent ``AgentSpec``. Everyone observes the world (``w_obs`` all ones --
    no vanguard self-censorship here; the cosmology levers are disconnection and conviction,
    not evidence scarcity). ``precision_scale`` = community conservatism. When ``conviction >
    0`` the conviction lever is ON: a per-agent tilt ``conviction * max(utility_scale, 0)``
    toward the agent's community home theory."""
    ps = pop["precision_scale"]
    us = pop["utility_scale"]
    cid = pop["cluster_id"]
    N = len(cid)
    tilt = None
    u_agent = None
    if conviction > 0.0:
        u_rows = np.stack([np.asarray(sc.cosmology_utility_toward(homes[int(cid[i])]))
                           for i in range(N)])
        u_agent = jnp.asarray(u_rows)
        tilt = jnp.asarray(conviction * np.clip(us, 0.0, None))
    return AgentSpec(w_obs=jnp.ones((N, scn.m)), lam=np.ones(N),
                     precision_scale=jnp.asarray(ps), tilt=tilt, u_agent=u_agent)


def cosmology_graph(sizes: list[int], *, connected: bool, seed: int = 0) -> graphs.Graph:
    inter = INTER_CONNECTED if connected else INTER_DISCONNECTED
    return graphs.community(sizes, intra=INTRA, inter=inter, seed=seed)


# ----------------------------------------------------------------------
# Read-outs (importable by the notebook).
# ----------------------------------------------------------------------

def held_by_community(r: dict, cid: np.ndarray, theory_mu: np.ndarray) -> np.ndarray:
    """(n_comm, S) the nearest candidate-theory index of each community's *mean* belief at each
    snapshot -- the headline tracking read-out."""
    h = r["snap_h"]                                    # (S, N, d)
    n_comm = int(cid.max()) + 1
    return np.stack([sc.closest_theory(h[:, cid == k, :].mean(axis=1), theory_mu)
                     for k in range(n_comm)])          # (n_comm, S)


def alignment_by_community(r: dict, cid: np.ndarray, theory_mu: np.ndarray) -> np.ndarray:
    """(n_comm, S, E) the soft per-theory alignment of each community's mean belief -- the
    graded version of ``held_by_community`` (so partial drift between theories is visible)."""
    h = r["snap_h"]
    n_comm = int(cid.max()) + 1
    return np.stack([sc.theory_alignment(h[:, cid == k, :].mean(axis=1), theory_mu)
                     for k in range(n_comm)])          # (n_comm, S, E)


def cell_metrics(r: dict, cid: np.ndarray, theory_mu: np.ndarray,
                 true_per_snap: np.ndarray) -> dict:
    """Scalar metrics for one (disconnect, conviction, seed) cell."""
    held = held_by_community(r, cid, theory_mu)        # (n_comm, S)
    end = held[:, -1]                                  # final theory per community
    n_comm = held.shape[0]
    # fraction of (community, snapshot) pairs holding the *current* true theory (warm-up
    # dropped -- the first snapshot is t=0, always the incumbent).
    on_truth = (held[:, 1:] == true_per_snap[None, 1:]).mean()
    return {
        "end_theories": end.tolist(),
        "n_distinct_end": int(len(np.unique(end))),
        "reached_final_frac": float((end == theory_mu.shape[0] - 1).mean()),
        "comm0_stuck_on_incumbent": bool(end[0] == 0),
        "on_truth_fraction": float(on_truth),
        "final_disagreement": float(r["disagreement_t"][-1]),
        "max_disagreement": float(r["disagreement_t"].max()),
        "dF_finite": bool(np.isfinite(r["final_dF"]).all()),
    }


def run_cell(scn: sc.Scenario, *, disconnect: bool, conviction: float, seed: int,
             snapshot_every: int = 10) -> tuple[dict, dict]:
    """One cell of the lever 2x2 at one seed. Returns ``(raw_run, metrics)``."""
    pop = cosmology_population(seed=seed)
    graph = cosmology_graph(pop["sizes"], connected=not disconnect, seed=seed)
    spec = cosmology_spec(scn, pop, conviction=conviction)
    r = run_simulation(scn, graph, spec, snapshot_every=snapshot_every, seed=seed)
    theory_mu = sc.cosmology_theory_means()
    true_per_snap = np.asarray(scn.epoch_t)[r["snap_t"]]
    met = cell_metrics(r, pop["cluster_id"], theory_mu, true_per_snap)
    met["lambda2"] = float(r["lambda2"])
    return {"r": r, "pop": pop}, met


# ----------------------------------------------------------------------
# The wall probe ("mechanisms in a row") for one representative cell.
# ----------------------------------------------------------------------

def mechanisms_in_a_row(scn, cell_both, cid):
    r = cell_both["r"]
    theory_mu = sc.cosmology_theory_means()
    held = held_by_community(r, cid, theory_mu)
    snap_t = r["snap_t"]
    true_per_snap = np.asarray(scn.epoch_t)[snap_t]
    print("\n=== mechanisms in a row (representative: BOTH levers, disconnected + conviction) ===")
    print(f"  [world]  3 epochs: {EPOCH_NAMES[0]} (t<{T1}) -> {EPOCH_NAMES[1]} (t<{T2}) -> "
          f"{EPOCH_NAMES[2]} (t>={T2}); n_steps={N_STEPS}")
    print(f"  [init]   {len(cid)} agents in 3 communities {COMM_LABELS}; all start on the "
          f"incumbent ({EPOCH_NAMES[0]}) theory; homes={HOMES}")
    print(f"  [t=0]    every community holds theory 0 ({EPOCH_NAMES[0]})")
    for k in range(held.shape[0]):
        traj = "".join(str(int(x)) for x in held[k])
        print(f"  [comm{k}] {COMM_LABELS[k]:>12} (home {HOMES[k][:4]}): held-theory trace = {traj}  "
              f"-> ends on {EPOCH_NAMES[held[k][-1]]}")
    print(f"  [truth]  true theory per snapshot                       = "
          f"{''.join(str(int(x)) for x in true_per_snap)}")
    print(f"  [field]  residual structural disagreement {r['disagreement_t'][0]:.2f} -> "
          f"{r['disagreement_t'][-1]:.2f} (stays > 0 = communities never reconcile)")


# ----------------------------------------------------------------------
# Figures (diagnostic; the polished panel is in nb44).
# ----------------------------------------------------------------------

CELLS = (("neither", False, 0.0), ("disconnection", True, 0.0),
         ("conviction", False, CONVICTION_LEVEL), ("both", True, CONVICTION_LEVEL))


def fig_tracking(rep_cells: dict, scn, cid, path: Path):
    """2x2: held-theory trajectory per community per cell, with the true-epoch staircase."""
    theory_mu = sc.cosmology_theory_means()
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
    cols = ["crimson", "darkorange", "seagreen"]
    for ax, (label, disc, conv) in zip(axs.ravel(), CELLS):
        r = rep_cells[label]["r"]
        snap_t = r["snap_t"]
        true_per_snap = np.asarray(scn.epoch_t)[snap_t]
        held = held_by_community(r, cid, theory_mu)
        ax.step(snap_t, true_per_snap, where="post", color="0.4", lw=4, alpha=0.35,
                label="true theory")
        for k in range(held.shape[0]):
            ax.plot(snap_t, held[k] + 0.04 * (k - 1), "o-", color=cols[k], lw=1.8, ms=4,
                    label=f"{COMM_LABELS[k]} (home {HOMES[k][:4]})")
        for tt in (T1, T2):
            ax.axvline(tt, color="k", ls=":", lw=0.8)
        ax.set_title(f"{label}  (λ₂={r['lambda2']:.2f}, disconnect={disc}, conviction={conv})",
                     fontsize=10)
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels([n[:9] for n in EPOCH_NAMES], fontsize=8)
        ax.set_ylabel("held theory")
    axs[1, 0].set_xlabel("step"); axs[1, 1].set_xlabel("step")
    axs[0, 0].legend(fontsize=7, loc="center left")
    fig.suptitle("Lens B: which theory each community holds over time (tracking vs stuck)",
                 fontsize=13)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def fig_diversity(rep_cells: dict, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cols = {"neither": "0.5", "disconnection": "steelblue",
            "conviction": "darkorange", "both": "crimson"}
    for label, _, _ in CELLS:
        r = rep_cells[label]["r"]
        ax.plot(r["snap_t"], r["disagreement_t"], "o-", lw=2, color=cols[label], label=label)
    for tt in (T1, T2):
        ax.axvline(tt, color="k", ls=":", lw=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("residual structural disagreement")
    ax.set_title("structural diversity over time (0 = consensus / diversity lost)")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------
# Controls (mirror nb43's run_controls).
# ----------------------------------------------------------------------

def run_controls() -> dict:
    out = {}
    theory_mu = sc.cosmology_theory_means()
    pop = cosmology_population(seed=0)

    # 1. no-shift -> no re-track: a stationary incumbent world (epochs never start).
    scn_static = sc.cosmology_scenario(n_steps=N_STEPS, t1=N_STEPS + 10, t2=N_STEPS + 20,
                                       sigma_o=SIGMA_O)
    g = cosmology_graph(pop["sizes"], connected=True)
    r = run_simulation(scn_static, g, cosmology_spec(scn_static, pop, conviction=0.0),
                       snapshot_every=20, seed=0)
    held = held_by_community(r, pop["cluster_id"], theory_mu)
    out["no_shift_max_held"] = int(held.max())            # expect 0 (never leaves incumbent)

    # 2. single-epoch (only dark matter) == nb43-like single regime: hold the incumbent.
    scn_one = sc.cosmology_scenario(n_steps=T1, t1=T1, t2=T1 + 1, sigma_o=SIGMA_O)
    r1 = run_simulation(scn_one, g, cosmology_spec(scn_one, pop, conviction=0.0),
                        snapshot_every=20, seed=0)
    held1 = held_by_community(r1, pop["cluster_id"], theory_mu)
    out["single_epoch_max_held"] = int(held1.max())       # expect 0

    # 3. conviction -> very large, all committed to the incumbent + connected: FROZEN on 0.
    scn = sc.cosmology_scenario(n_steps=N_STEPS, t1=T1, t2=T2, sigma_o=SIGMA_O)
    spec_inf = cosmology_spec(scn, pop, conviction=50.0,
                              homes=("dark_matter",) * 3)
    r_inf = run_simulation(scn, g, spec_inf, snapshot_every=20, seed=0)
    held_inf = held_by_community(r_inf, pop["cluster_id"], theory_mu)
    out["conviction_inf_final_held"] = [int(x) for x in held_inf[:, -1]]   # expect all 0

    # 4. inter=0 vs connected: disconnection preserves more structural disagreement.
    r_disc = run_simulation(scn, cosmology_graph(pop["sizes"], connected=False),
                            cosmology_spec(scn, pop, conviction=0.0), snapshot_every=20, seed=0)
    r_conn = run_simulation(scn, cosmology_graph(pop["sizes"], connected=True),
                            cosmology_spec(scn, pop, conviction=0.0), snapshot_every=20, seed=0)
    out["disagreement_disconnected"] = float(r_disc["disagreement_t"][-1])
    out["disagreement_connected"] = float(r_conn["disagreement_t"][-1])

    # 5. finite dF/PD across epochs.
    out["dF_finite"] = bool(np.isfinite(r_conn["final_dF"]).all()
                            and np.isfinite(r_inf["final_dF"]).all())
    return out


# ----------------------------------------------------------------------
# main.
# ----------------------------------------------------------------------

def main() -> int:
    matplotlib.use("Agg")                  # batch (file-only) figures when run as a script
    out_dir = ROOT / "results" / "cosmology_tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    scn = sc.cosmology_scenario(n_steps=N_STEPS, t1=T1, t2=T2, sigma_o=SIGMA_O)
    theory_mu = sc.cosmology_theory_means()

    print(f"cosmology world: epochs {EPOCH_NAMES}; T1={T1} T2={T2} n_steps={N_STEPS}")
    print("theory means (commitment cols dark/scale/modi):")
    for e, nm in enumerate(EPOCH_NAMES):
        print(f"  {nm:>18}: {np.round(theory_mu[e], 2)}")

    # ---- the 2x2 lever sweep, multi-seed ----
    print(f"\n=== 2x2 lever sweep (multi-seed, seeds={SEEDS}) ===")
    print(f"{'cell':>14} | {'λ₂':>6} | end-theories(rep) | n_distinct | comm0_stuck | "
          f"on_truth | final_disagree")
    rep_cells = {}                       # seed-0 raw runs for the figures / mechanisms
    cell_summaries = {}
    for label, disc, conv in CELLS:
        mets = []
        for s in SEEDS:
            cell, met = run_cell(scn, disconnect=disc, conviction=conv, seed=s)
            mets.append(met)
            if s == 0:
                rep_cells[label] = cell
        # aggregate over seeds
        agg = {
            "lambda2": float(np.mean([m["lambda2"] for m in mets])),
            "n_distinct_end_mean": float(np.mean([m["n_distinct_end"] for m in mets])),
            "comm0_stuck_frac": float(np.mean([m["comm0_stuck_on_incumbent"] for m in mets])),
            "reached_final_frac_mean": float(np.mean([m["reached_final_frac"] for m in mets])),
            "on_truth_fraction_mean": float(np.mean([m["on_truth_fraction"] for m in mets])),
            "final_disagreement_mean": float(np.mean([m["final_disagreement"] for m in mets])),
            "end_theories_rep": mets[0]["end_theories"],
            "dF_finite": bool(all(m["dF_finite"] for m in mets)),
        }
        cell_summaries[label] = agg
        print(f"{label:>14} | {agg['lambda2']:6.2f} | {str(agg['end_theories_rep']):>17} | "
              f"{agg['n_distinct_end_mean']:10.2f} | {agg['comm0_stuck_frac']:11.2f} | "
              f"{agg['on_truth_fraction_mean']:8.2f} | {agg['final_disagreement_mean']:.2f}")

    # ---- wall probe + figures ----
    cid0 = rep_cells["both"]["pop"]["cluster_id"]
    mechanisms_in_a_row(scn, rep_cells["both"], cid0)
    fig_tracking(rep_cells, scn, cid0, out_dir / "diag_tracking_2x2.png")
    fig_diversity(rep_cells, out_dir / "diag_diversity.png")

    # ---- honest structural caveat: do the contested edges discriminate the theory? ----
    r_both = rep_cells["both"]["r"]
    names = scn.names
    idx = {n: i for i, n in enumerate(names)}
    learned_off = np.array([np.abs(r_both["snap_Pi"][-1][:, idx[a], idx[c]]).mean()
                            for (a, c) in scn.edges])
    true_off = np.abs(sc.cosmology_true_couplings())          # (E, n_edges)
    print("\n=== honest structural caveat (risk #3: do relational edges discriminate?) ===")
    print(f"  learned |Pi[a,c]| on the 6 contested edges (final): {np.round(learned_off, 1)}")
    print(f"  -> they accumulate ~uniformly (operator-set HᵀH dominates the ±{true_off.max():.2f} "
          f"prior coupling differences).")
    print("  -> THEORY IDENTITY IS CARRIED BY THE MEANS, not the learned precision couplings;")
    print("     genuine data-driven structure re-growth is Lens A (run_cosmology_regrowth.py).")

    # ---- controls ----
    print("\n=== null / sanity controls ===")
    ctrl = run_controls()
    print(f"  no-shift max held theory:        {ctrl['no_shift_max_held']}  (expect 0)")
    print(f"  single-epoch max held theory:    {ctrl['single_epoch_max_held']}  (expect 0)")
    print(f"  conviction→∞ final held:         {ctrl['conviction_inf_final_held']}  (expect all 0)")
    print(f"  disagreement disconnected/conn:  {ctrl['disagreement_disconnected']:.2f} / "
          f"{ctrl['disagreement_connected']:.2f}  (expect disconnected > connected)")
    print(f"  ΔF finite/PD:                    {ctrl['dF_finite']}")

    assert ctrl["no_shift_max_held"] == 0, \
        f"stationary world re-tracked (held {ctrl['no_shift_max_held']}, expected 0)"
    assert ctrl["single_epoch_max_held"] == 0, \
        "single-epoch world left the incumbent (expected to hold it)"
    assert all(x == 0 for x in ctrl["conviction_inf_final_held"]), \
        f"conviction→∞ did not freeze on the incumbent: {ctrl['conviction_inf_final_held']}"
    assert ctrl["disagreement_disconnected"] > ctrl["disagreement_connected"] + 1e-6, \
        "disconnection did not preserve more structural disagreement than connection"
    assert ctrl["dF_finite"], "non-finite ΔF (PD violation)"

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        epoch_names=np.array(EPOCH_NAMES),
        theory_means=theory_mu,
        snap_t=rep_cells["neither"]["r"]["snap_t"],
        true_per_snap=np.asarray(scn.epoch_t)[rep_cells["neither"]["r"]["snap_t"]],
        cluster_id=cid0,
        **{f"held_{lab}": held_by_community(rep_cells[lab]["r"], cid0, theory_mu)
           for lab, _, _ in CELLS},
        **{f"disagree_{lab}": rep_cells[lab]["r"]["disagreement_t"]
           for lab, _, _ in CELLS},
        learned_offdiag=learned_off, true_offdiag=true_off,
    )
    summary = {
        "config": {"N": N_AGENTS, "n_steps": N_STEPS, "T1": T1, "T2": T2,
                   "sigma_o": SIGMA_O, "seeds": list(SEEDS),
                   "conviction_level": CONVICTION_LEVEL,
                   "inter_connected": INTER_CONNECTED, "homes": list(HOMES),
                   "communities": list(COMM_LABELS)},
        "epochs": list(EPOCH_NAMES),
        "cells": cell_summaries,
        "controls": ctrl,
        "structural_caveat": {
            "learned_offdiag_final": learned_off.tolist(),
            "true_offdiag_max": float(true_off.max()),
            "note": "relational edges accumulate uniformly (operator-set Fisher); theory "
                    "identity is in the means, not the learned couplings (see Lens A)."},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figures / summary to {out_dir}")
    print("HEADLINE (do we need both levers?):")
    for lab, _, _ in CELLS:
        a = cell_summaries[lab]
        print(f"  {lab:>14}: {a['n_distinct_end_mean']:.1f} distinct end-theories, "
              f"comm0 stuck {a['comm0_stuck_frac']:.0%}, disagreement {a['final_disagreement_mean']:.1f}")
    both = cell_summaries["both"]
    conv = cell_summaries["conviction"]
    if both["n_distinct_end_mean"] > conv["n_distinct_end_mean"] + 0.5:
        print("  => YES: persistent theory-divergence needs BOTH levers -- conviction alone "
              "washes out under fusion (no-pool degeneracy); disconnection protects it.")
    else:
        print("  => the levers did not separate as expected; read the cell table honestly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
