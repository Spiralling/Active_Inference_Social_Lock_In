"""Divergent wiring: trust that reads WIRING preserves structural pluralism under contact.

The priority question of the structure-level program: when a community has divergent values
in a world that genuinely supports multiple wirings (``structural_pluralism``'s 50/50
underdetermined blend), its members learn DIFFERENT NETWORK STRUCTURES -- and what does
communication do to that? Known: disconnection preserves the wiring divergence, plain
connection collapses it (precision-pooling merges the wirings). This experiment adds the
trust axis and the collective-vs-individual read-out:

  * **The means-gate cannot save it** (the probe, recorded in ``summary.json``): in this
    world the camps AGREE in their posterior means (both fit the same blended data) while
    diverging in wiring, so trust that reads opinions (``reliability.social_gamma``) is
    blind to the pluralism and collapses exactly like fixed trust (~10% retention).
  * **The structure-gate can** (``social_gate="structure"``): the same Student-t trust
    functional pointed at the contested couplings ``Pi[a, c]``
    (``reliability.pairwise_structure_z2``) -- agents extend trust by *how you wire the
    world*, not what opinions or values you hold. Under full contact it retains most of the
    disconnected divergence.
  * **The collective knows more than any member ONLY as a union**: the union of individual
    nets holds both wirings (no individual does); the consensus net -- the model's own
    complete-graph pooling operator -- holds a diluted blur of each
    (``structure_metrics.consensus_union_report``), and provably creates no edge no member
    holds (synergy = 0, the averaging theorem-let).

Mechanism note: the trust gate IS a divergence-reading mechanism -- ``gamma_ij`` is a
Student-t weight on the (calibrated, Mahalanobis-like) divergence between agents' posterior
Gaussians, and severing fusion when divergence is large is what lets genuinely different
distributions persist in one connected population. The means-gate reads the divergence of
the distributions' centers (opinions); the structure-gate reads the divergence of their
coupling pattern (wiring) -- the same gate, pointed at the part of the distribution where
this world's pluralism actually lives.

Honest calibration notes (probe, seed 0): the gate is BISTABLE in ``nu`` -- divergence must
bootstrap from identical priors, and above a critical ``nu`` (between 0.1 and 0.2 here) the
collapse is self-confirming (no divergence -> no gate signal -> no divergence), so the
working ``nu`` range is ~an order of magnitude below the plan's prior expectation of
0.4-0.5; ``NU_SWEEP`` straddles the located transition. ``SCALE_FLOOR`` is inert in this
world: the "lacking" couplings sit at the shared prior wiring scale (|Pi| ~ 4.0), far above
any reasonable floor.

Applied-BMR coda (probe, seed 0): the bmr conditions ask whether in-loop model reduction
(``bmr_every``) makes the fabricated wiring answerable to evidence. It does NOT, and the
invariance is the finding: the dm reference prior's contested couplings are ~0, the
pluralism is ~100% attention-driven deposit, and Savage-Dickey reduction removes prior
structure while keeping the data -- so the couplings are unchanged to one decimal.
Deletion adjudicates believed PRIOR structure; it cannot un-run your experiments.

Honest three-camps note (the prediction was REFUTED and is reported as found): a camp
valuing the commitment the 50/50 blend does NOT support still consolidates its wiring at
full strength, because the Gaussian Fisher deposit ``J = H^T diag(w) H / sigma^2`` never
sees the observation -- attention alone fabricates the precision-pattern edge, and the
data's verdict on an unsupported theory lives in the potential ``h`` (the means), not in
``|Pi|``. A scope condition on what "structure learning" means in this model class.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from experiments.registry import ExperimentSpec, register


# ----------------------------------------------------------------------
# Worker-side: world cache + one condition run + measurements.
# ----------------------------------------------------------------------

_WORLD: dict = {}

# Commitment of each contested edge (COSMOLOGY_EDGES is commitment-major: two anomaly
# couplings per commitment, in COMMITS order).
EDGE_COMMIT = np.array([0, 0, 1, 1, 2, 2])


def _world(p: dict):
    """The shared underdetermined world (memoized per worker process)."""
    key = (p["N"], p["N_STEPS"], p["W_HI"], p["W_LO"], tuple(p["COMMITS"]),
           p["VALUE_A"], p["VALUE_B"])
    if key not in _WORLD:
        from experiments.structural_pluralism import build
        _WORLD[key] = build(n=p["N"], n_steps=p["N_STEPS"], commits=tuple(p["COMMITS"]),
                            value_a=p["VALUE_A"], value_b=p["VALUE_B"],
                            w_hi=p["W_HI"], w_lo=p["W_LO"])
    return _WORLD[key]


def _three_camp_attention(scn, n: int, w_hi: float, w_lo: float):
    """VALUE -> ATTENTION for THREE equal camps: camp k weights ``balance(commits[k])``
    high (``scn.disc_rows`` is ordered like the commitments). Camp 1 values
    modified_gravity -- the commitment the 50/50 dm+sv blend does NOT support."""
    camp = np.minimum(np.arange(n) * 3 // n, 2)
    w = np.ones((n, scn.m))
    for k, row in enumerate(scn.disc_rows):
        w[:, row] = np.where(camp == k, w_hi, w_lo)
    return jnp.asarray(w), camp


def run_condition(scn, w, edges_ij, graph, seed: int, *, n: int, omega: float,
                  snapshot_every: int = 5, **sim_kwargs) -> dict:
    """One simulation run; returns the full ``run_simulation`` dict (``snap_Pi`` kept)."""
    from src.structural.simulation import run_simulation, AgentSpec
    spec = AgentSpec(w_obs=w, lam=jnp.full((n,), 0.2))
    return run_simulation(scn, graph, spec, forgetting=omega,
                          snapshot_every=snapshot_every, seed=seed, **sim_kwargs)


def _measure(r: dict, edges_ij, camp: np.ndarray, p: dict) -> dict:
    """All structure-level read-outs of one run (host numpy; pickle-light)."""
    from src.structural import structure_metrics as sm
    from experiments.structural_pluralism import structure, by_commitment, reldist

    thr = p["EDGE_THRESHOLD"]
    Pi_t = r["snap_Pi"]                                            # (S, N, d, d)
    S = Pi_t.shape[0]
    disp = {kind: sm.dispersion_trace(Pi_t, camp, kind=kind, pairs=edges_ij,
                                      threshold=thr)
            for kind in ("fro", "jaccard", "spectral")}
    n_camps = int(camp.max()) + 1
    c_t = np.stack([                                               # (B, S, 3) portrait
        np.stack([by_commitment(structure(Pi_t[s][camp == b].mean(0), edges_ij))
                  for s in range(S)])
        for b in range(n_camps)])
    P = Pi_t[-1]
    rep = sm.consensus_union_report(P, pairs=edges_ij, threshold=thr)
    sA = structure(P[camp == 0].mean(0), edges_ij)
    sB = structure(P[camp == n_camps - 1].mean(0), edges_ij)
    out = dict(snap_t=np.asarray(r["snap_t"]), c_t=c_t,
               reldist=float(reldist(sA, sB)),
               union_count=int(rep["union_count"]),
               best_individual_count=int(rep["best_individual_count"]),
               consensus_edge_count=len(rep["consensus_edges"]),
               union_coverage_ratio=float(rep["union_coverage_ratio"]),
               dilution_ratio=float(rep["dilution_ratio"]),
               edges_lost_count=len(rep["edges_lost"]),
               synergy_count=len(rep["synergy_edges"]),
               consensus_by_commit=by_commitment(np.asarray(rep["consensus_couplings"])),
               max_Pi=float(np.abs(Pi_t).max()))
    for kind, tag in (("fro", "fro"), ("jaccard", "jac"), ("spectral", "spec")):
        out[f"cross_{tag}"] = disp[kind]["cross"]                  # (S,)
        out[f"within_{tag}"] = disp[kind]["within"].mean(axis=1)   # (S,)
    if "applied_pruned_t" in r:                # applied-BMR telemetry (bmr conditions)
        flags = np.asarray(r["applied_pruned_t"][-1])              # (N, 6)
        out["bmr_flags_by_camp"] = np.stack([
            [float(flags[camp == b][:, EDGE_COMMIT == k].mean()) for k in range(3)]
            for b in range(n_camps)])                              # (B, 3)
        out["min_eig"] = float(min(np.linalg.eigvalsh(Q).min() for Q in P))
    return out


def _one_job(cfg: dict) -> dict:
    p = cfg
    scn, edges_ij, w = _world(p)
    n = p["N"]
    from src.structural import graphs
    if p.get("camps", 2) == 3:
        w, camp = _three_camp_attention(scn, n, p["W_HI"], p["W_LO"])
        graph = graphs.community([n - 2 * (n // 3), n // 3, n // 3][:3],
                                 intra=1.0, inter=float(p.get("inter", 0.0)), seed=0)
        camp = np.asarray(graph.membership)
    elif p["graph"] == "complete":
        graph = graphs.complete(n)
        camp = (np.arange(n) >= n // 2).astype(int)        # the VALUE community
    else:
        graph = graphs.community([n // 2, n // 2], intra=1.0,
                                 inter=float(p.get("inter", 0.0)), seed=0)
        camp = (np.arange(n) >= n // 2).astype(int)
    gate_kw = {}
    if p.get("gate") == "structure":
        gate_kw = dict(social_nu=float(p["nu"]), social_gate="structure",
                       social_pairs=tuple(edges_ij),
                       social_scale_floor=p["SCALE_FLOOR"])
    elif p.get("gate") == "means":
        gate_kw = dict(social_nu=float(p["nu"]),
                       social_idx=tuple(range(len(scn.names))))
    if p.get("fuse_mode"):
        gate_kw["fuse_mode"] = p["fuse_mode"]
    if p.get("bmr"):
        gate_kw["bmr_every"] = int(p["bmr"])
    r = run_condition(scn, w, edges_ij, graph, int(p["seed"]), n=n, omega=p["OMEGA"],
                      snapshot_every=p["SNAPSHOT_EVERY"], **gate_kw)
    row = _measure(r, edges_ij, camp, p)
    row.update(name=p["name"], seed=int(p["seed"]), nu=p.get("nu"),
               inter=p.get("inter"), lambda2=float(r["lambda2"]))
    return row


# ----------------------------------------------------------------------
# Host-side: aggregation helpers.
# ----------------------------------------------------------------------

def _rows(rows, name):
    out = [r for r in rows if r["name"] == name]
    assert out, f"no rows for condition {name!r}"
    return out


def _mean(rows, name, key):
    return np.mean([r[key] for r in _rows(rows, name)], axis=0)


def _std(rows, name, key):
    return np.std([r[key] for r in _rows(rows, name)], axis=0)


def _final_cross(rows, name) -> float:
    return float(np.mean([r["cross_fro"][-1] for r in _rows(rows, name)]))


# ----------------------------------------------------------------------
# Figures.
# ----------------------------------------------------------------------

def _fig_traces(rows, trio, out_path):
    """One axes, three colours: the three conditions' cross-community divergence
    overlaid (solid, with seed bands); within-community as thin dashed lines in
    the matching colour (all ~0). Replaces the unreadable 1x3 panel layout."""
    colors = ("#2c6fbb", "#c0392b", "#27ae60")
    labels = ("disconnected", "connected, fixed trust",
              "connected, wiring-gated trust")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for (name, _), color, lab in zip(trio, colors, labels):
        t = _rows(rows, name)[0]["snap_t"]
        m, s = _mean(rows, name, "cross_fro"), _std(rows, name, "cross_fro")
        ax.plot(t, m, "-", color=color, lw=2.4, label=lab)
        ax.fill_between(t, m - s, m + s, color=color, alpha=0.15)
        ax.plot(t, _mean(rows, name, "within_fro"), "--", color=color,
                lw=1.0, alpha=0.6)
    ax.plot([], [], "--", color="grey", lw=1.0, label="within-community (all)")
    ax.set_xlabel("step", fontsize=11)
    ax.set_ylabel("cross-community structure distance\non contested couplings (fro)",
                  fontsize=11)
    ax.set_title("Structural pluralism under contact", fontsize=12)
    ax.legend(fontsize=10, loc="center right")
    ax.tick_params(labelsize=10)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(fig)


def _fig_portrait(rows, panels, commits, value_a, value_b, out_path,
                  three_camps=None):
    n_p = len(panels) + (1 if three_camps else 0)
    fig, axes = plt.subplots(1, n_p, figsize=(4.5 * n_p, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(3); width = 0.38
    short = ["dark\nmatter", "mod.\ngravity", "scale\nvariant"]
    for ax, (name, title) in zip(axes, panels):
        c_t = _mean(rows, name, "c_t")                     # (2, S, 3)
        cons = _mean(rows, name, "consensus_by_commit")    # (3,)
        ax.bar(x - width / 2, c_t[0, -1], width, color="#2c6fbb",
               label=f"A (values {value_a})")
        ax.bar(x + width / 2, c_t[1, -1], width, color="#c0392b",
               label=f"B (values {value_b})")
        for k in range(3):
            ax.plot([k - width, k + width], [cons[k], cons[k]], color="black", lw=2,
                    label="consensus net" if k == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels(short)
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7)
    if three_camps:
        ax = axes[len(panels)]
        c_t = _mean(rows, three_camps, "c_t")              # (3, S, 3)
        w3 = 0.26
        for b, (color, lab) in enumerate((("#2c6fbb", f"A ({commits[0]})"),
                                          ("#95a5a6", f"C ({commits[1]}, unsupported)"),
                                          ("#c0392b", f"B ({commits[2]})"))):
            ax.bar(x + (b - 1) * w3, c_t[b, -1], w3, color=color, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(short)
        ax.set_title("Three camps, disconnected: attention FABRICATES the\n"
                     "coupling pattern -- the unsupported camp consolidates too\n"
                     "(the data's verdict lives in the means, not |Pi|)", fontsize=8)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("learned coupling  |Pi[anomaly, commitment]|")
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close(fig)


def _fig_collective(rows, trio, threshold, out_path):
    names = [n for n, _ in trio]
    labels = [n.replace("_", "\n") for n in names]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(names)); w = 0.26
    for off, key, color, lab in ((-w, "union_count", "#1e8449", "union of members"),
                                 (0.0, "best_individual_count", "#2c6fbb",
                                  "best individual"),
                                 (w, "consensus_edge_count", "#aab7b8",
                                  "consensus (mean) net")):
        a0.bar(x + off, [_mean(rows, n, key) for n in names], w, color=color, label=lab)
    a0.set_xticks(x); a0.set_xticklabels(labels, fontsize=8)
    a0.set_ylabel(f"contested edges held (|Pi| > {threshold})")
    a0.set_title("the collective surplus lives in the union"); a0.legend(fontsize=8)
    a1.bar(labels, [_mean(rows, n, "dilution_ratio") for n in names],
           yerr=[_std(rows, n, "dilution_ratio") for n in names],
           color="#6c3483", capsize=5)
    a1.axhline(0.6, color="gray", ls=":", lw=1)
    a1.set_ylabel("dilution ratio  mean |Pi_cons[e]| / max_i |Pi_i[e]|")
    a1.set_title("averaging blurs what it keeps")
    fig.suptitle("averaging can never create an edge no individual holds "
                 "(synergy = 0, verified); the surplus lives in the union", fontsize=10)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close(fig)


def _fig_gate_strength(rows, nus, disc, fixed_ret, nu_star, out_path):
    rets = [[r["cross_fro"][-1] / disc for r in _rows(rows, f"gated_nu={nu}")]
            for nu in nus]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(nus, [np.mean(r) for r in rets], yerr=[np.std(r) for r in rets],
                marker="o", lw=2, capsize=3, color="#1e8449",
                label="structure-gated trust")
    ax.axhline(fixed_ret, color="#922b21", ls="--", lw=1.5,
               label=f"fixed trust ({fixed_ret:.0%})")
    ax.axvline(nu_star, color="gray", ls=":", lw=1)
    ax.annotate(f"nu* = {nu_star}", (nu_star, 0.5), fontsize=9, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("gate sharpness nu (smaller = quicker to sever trust)")
    ax.set_ylabel("cross-community structure retention\n(fraction of disconnected)")
    ax.set_ylim(0, 1.05)
    ax.set_title("the structure gate is bistable in nu:\n"
                 "divergence must bootstrap before fusion erases it")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close(fig)


def _fig_bmr(rows, bmr_every, prior_by_commit, out_path):
    """Camp C (values the unsupported theory) with and without applied BMR, plus the
    reference-prior coupling tick -- the ONLY part Savage-Dickey reduction can remove."""
    c0 = _mean(rows, "three_camps", "c_t")[1, -1]                   # (3,) final
    c1 = _mean(rows, "three_camps_bmr", "c_t")[1, -1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, c0, w, color="#95a5a6", label="no BMR")
    ax.bar(x + w / 2, c1, w, color="#1e8449", hatch="//",
           label=f"applied BMR (every {bmr_every} steps)")
    for k in range(3):
        ax.plot([k - w, k + w], [prior_by_commit[k]] * 2, color="black", lw=2.5,
                label="reference-prior coupling\n(all a prune can remove)" if k == 0
                else None)
    ax.set_xticks(x)
    ax.set_xticklabels(["dark\nmatter", "mod. gravity\n(unsupported, valued)",
                        "scale\nvariant"])
    ax.set_ylabel("camp C learned coupling  |Pi[anomaly, commitment]|")
    ax.set_title("applied BMR cannot touch fabricated wiring: the coupling is\n"
                 "~100% experiment deposit, and Savage-Dickey reduction removes\n"
                 "only PRIOR structure (black tick ~ 0)", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close(fig)


def _fig_connectivity(rows, fixed_pts, gated_pts, disc, out_path):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for pts, color, lab in ((fixed_pts, "#922b21", "fixed trust"),
                            (gated_pts, "#1e8449", "structure-gated trust (nu*)")):
        lam = [np.mean([r["lambda2"] for r in _rows(rows, n)]) for n in pts]
        ret = [[r["cross_fro"][-1] / disc for r in _rows(rows, n)] for n in pts]
        order = np.argsort(lam)
        lam = np.asarray(lam)[order]
        m = np.asarray([np.mean(r) for r in ret])[order]
        s = np.asarray([np.std(r) for r in ret])[order]
        ax.errorbar(lam, m, yerr=s, marker="s", lw=2, capsize=3, color=color, label=lab)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel(r"algebraic connectivity $\lambda_2$ (network mixing)")
    ax.set_ylabel("cross-community structure retention\n(fraction of disconnected)")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("where structural pluralism dies as mixing rises --\n"
                 "and how far the wiring-reading gate moves the boundary")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------
# The experiment.
# ----------------------------------------------------------------------

def run(out_dir, params: dict) -> None:
    p = dict(params)
    seeds = tuple(p["SEEDS"])
    commits = tuple(p["COMMITS"])
    nus = tuple(p["NU_SWEEP"])
    inters = tuple(p["INTER_SWEEP"])

    def jobs_for(name, graph, **kw):
        return [dict(p, name=name, graph=graph, seed=int(s), **kw) for s in seeds]

    gate = p["SOCIAL_GATE"]
    phase1 = (
        jobs_for("disconnected", "community", inter=0.0)
        + jobs_for("connected_fixed", "complete")
        + [j for nu in nus
           for j in jobs_for(f"gated_nu={nu}", "complete", gate=gate, nu=float(nu))]
        + jobs_for("means_gate_probe", "complete", gate="means",
                   nu=float(p["PROBE_MEANS_NU"]))
        + [j for it in inters if it > 0
           for j in jobs_for(f"fixed_inter={it}", "community", inter=float(it))]
        + (jobs_for("three_camps", "community", camps=3, inter=0.0)
           if p["INCLUDE_THREE_CAMPS"] else [])
        + (jobs_for("connected_bayesnet", "complete", fuse_mode="bayesnet")
           if p["INCLUDE_BAYESNET"] else [])
        # applied-BMR conditions: does evidence-driven pruning touch the pluralism?
        + jobs_for("disconnected_bmr", "community", inter=0.0, bmr=p["BMR_EVERY"])
        + (jobs_for("three_camps_bmr", "community", camps=3, inter=0.0,
                    bmr=p["BMR_EVERY"])
           if p["INCLUDE_THREE_CAMPS"] else [])
    )

    rows: list[dict] = []
    print(f"divergent wiring: A values {p['VALUE_A']}, B values {p['VALUE_B']}; "
          f"underdetermined world; omega={p['OMEGA']}; seeds={seeds}; "
          f"gate={gate}; {len(phase1)} phase-1 jobs on {p['MAX_WORKERS']} workers")
    with ProcessPoolExecutor(max_workers=p["MAX_WORKERS"]) as pool:
        for row in pool.map(_one_job, phase1, chunksize=1):
            rows.append(row)
            print(f"  {row['name']:>22} seed={row['seed']}: "
                  f"cross={row['cross_fro'][-1]:7.2f}  within={row['within_fro'][-1]:5.2f}"
                  f"  max|Pi|={row['max_Pi']:.0f}", flush=True)

        # ---- pick NU_STAR (best-performing nu of sweep A), then phase 2 ----
        disc = _final_cross(rows, "disconnected")
        nu_star = float(max(nus, key=lambda nu: _final_cross(rows, f"gated_nu={nu}")))
        print(f"\nNU_STAR = {nu_star} (best retention of sweep A)")
        phase2 = ([j for it in inters
                   for j in jobs_for(f"gated_inter={it}", "community", gate=gate,
                                     nu=nu_star, inter=float(it))]
                  # the full open-endedness stack: structure gate + applied BMR
                  + jobs_for("gated_bmr", "complete", gate=gate, nu=nu_star,
                             bmr=p["BMR_EVERY"]))
        for row in pool.map(_one_job, phase2, chunksize=1):
            rows.append(row)
            print(f"  {row['name']:>22} seed={row['seed']}: "
                  f"cross={row['cross_fro'][-1]:7.2f}  within={row['within_fro'][-1]:5.2f}"
                  f"  max|Pi|={row['max_Pi']:.0f}", flush=True)

    gated_star = f"gated_nu={nu_star}"
    trio = [("disconnected", "Disconnected: two crisp wirings"),
            ("connected_fixed", "Connected, fixed trust: one blurry mush"),
            (gated_star, "Connected, structure-gated trust:\n"
                         "two crisp wirings under contact")]

    # ---- print every measured value BEFORE asserting (honest-findings rule) ----
    fixed = _final_cross(rows, "connected_fixed")
    gated = _final_cross(rows, gated_star)
    probe = _final_cross(rows, "means_gate_probe")
    print(f"\nfinal cross-community fro distance (seed mean): disconnected={disc:.2f}  "
          f"fixed={fixed:.2f} ({fixed / disc:.1%})  means-gate={probe:.2f} "
          f"({probe / disc:.1%})  structure-gate(nu*)={gated:.2f} ({gated / disc:.1%})")
    rel_d = float(np.mean([r["reldist"] for r in _rows(rows, "disconnected")]))
    rel_f = float(np.mean([r["reldist"] for r in _rows(rows, "connected_fixed")]))
    print(f"reldist (structural_pluralism's measure): disconnected={rel_d:.3f}  "
          f"fixed={rel_f:.3f}")
    for name, _ in trio:
        rs = _rows(rows, name)
        print(f"  {name:>22}: union={_mean(rows, name, 'union_count'):.1f}  "
              f"best={_mean(rows, name, 'best_individual_count'):.1f}  "
              f"consensus={_mean(rows, name, 'consensus_edge_count'):.1f}  "
              f"dilution={_mean(rows, name, 'dilution_ratio'):.3f}  "
              f"synergy={max(r['synergy_count'] for r in rs)}")
    ret_nu = {nu: _final_cross(rows, f"gated_nu={nu}") / disc for nu in nus}
    print("sweep A retention vs nu: "
          + "  ".join(f"{nu}:{ret_nu[nu]:.1%}" for nu in nus))
    fixed_pts = ["disconnected"] + [f"fixed_inter={it}" for it in inters if it > 0] \
        + ["connected_fixed"]
    gated_pts = [f"gated_inter={it}" for it in inters] + [gated_star]
    lam_fixed = [float(np.mean([r["lambda2"] for r in _rows(rows, n)]))
                 for n in fixed_pts]
    ret_fixed = [_final_cross(rows, n) / disc for n in fixed_pts]
    ret_gated = [_final_cross(rows, n) / disc for n in gated_pts]
    print("sweep B (lambda2 -> retention fixed | gated): "
          + "  ".join(f"{l:.3g}->{rf:.1%}|{rg:.1%}"
                      for l, rf, rg in zip(lam_fixed, ret_fixed, ret_gated)))
    if p["INCLUDE_THREE_CAMPS"]:
        c3 = _mean(rows, "three_camps", "c_t")             # (3, S, 3)
        print("three camps (final |coupling| by commitment, camps A/C/B): "
              + "  ".join(f"camp{b}={np.round(c3[b, -1], 1)}" for b in range(3)))

    # ---- applied-BMR read-outs (printed before asserting) ----
    from experiments.structural_pluralism import build, structure, by_commitment
    scn0, edges0, _ = build(n=p["N"], n_steps=p["N_STEPS"], commits=commits,
                            value_a=p["VALUE_A"], value_b=p["VALUE_B"],
                            w_hi=p["W_HI"], w_lo=p["W_LO"])
    prior_by_commit = by_commitment(structure(np.asarray(scn0.Pi0[0]), edges0))
    bmr_pairs = [("disconnected", "disconnected_bmr")]
    if p["INCLUDE_THREE_CAMPS"]:
        bmr_pairs.append(("three_camps", "three_camps_bmr"))
    bmr_delta, bmr_flags = {}, {}
    for base, bm in bmr_pairs:
        d = float(np.abs(_mean(rows, bm, "c_t")[:, -1]
                         - _mean(rows, base, "c_t")[:, -1]).max())
        bmr_delta[bm] = d
        bmr_flags[bm] = _mean(rows, bm, "bmr_flags_by_camp")
        print(f"applied BMR {bm}: max |coupling| change vs {base} = {d:.3f}  "
              f"(flags per camp, dm/mg/sv: {np.round(bmr_flags[bm], 2).tolist()})")
    gated_bmr_cross = _final_cross(rows, "gated_bmr")
    print(f"gate + applied BMR retention: {gated_bmr_cross / disc:.1%} "
          f"(gate alone: {gated / disc:.1%})")
    print(f"reference-prior couplings by commitment (all a prune can remove): "
          f"{np.round(prior_by_commit, 2).tolist()}")

    # ---- assertions ----
    # A1 replicate structural_pluralism's claim on its own measure
    assert rel_d > 3.0 * rel_f + 1e-6, \
        f"A1: disconnection should preserve >3x the divergence ({rel_d:.2f} vs {rel_f:.2f})"
    # A2 HEADLINE
    assert gated >= 0.5 * disc, \
        f"A2: structure gate should retain >=50% of disconnected ({gated / disc:.1%})"
    assert fixed < 0.2 * disc, \
        f"A2: fixed trust should collapse to <20% of disconnected ({fixed / disc:.1%})"
    # A3 wiring identity survives contact under the gate
    for name in ("disconnected", gated_star):
        c_t = _mean(rows, name, "c_t")
        assert int(c_t[0, -1].argmax()) == commits.index(p["VALUE_A"]), \
            f"A3: camp A should wire its valued theory in {name}"
        assert int(c_t[1, -1].argmax()) == commits.index(p["VALUE_B"]), \
            f"A3: camp B should wire its valued theory in {name}"
    # A4 union surplus, every seed (fixed-connected reported above, not asserted)
    for name in ("disconnected", gated_star):
        for r in _rows(rows, name):
            assert r["union_count"] > r["best_individual_count"], \
                f"A4: union should exceed best individual in {name} seed {r['seed']}"
    # A5 dilution
    dil = float(_mean(rows, "disconnected", "dilution_ratio"))
    assert dil < 0.6, f"A5: disconnected dilution should be < 0.6 (got {dil:.3f})"
    # A6 theorem-let, every condition/seed
    assert all(r["synergy_count"] == 0 for r in rows), \
        "A6: averaging created an edge no individual holds (impossible)"
    # A7 health
    assert all(r["max_Pi"] < 500 for r in rows), "A7: forgetting should bound precision"
    # A8 sweeps (loose, report-first; tolerance 0.05 on seed-mean retention)
    rs = [ret_nu[nu] for nu in sorted(nus)]
    assert all(rs[i + 1] <= rs[i] + 0.05 for i in range(len(rs) - 1)), \
        f"A8: sweep-A retention should be non-increasing in nu (got {np.round(rs, 3)})"
    order = np.argsort(lam_fixed)
    rf_sorted = np.asarray(ret_fixed)[order]
    assert all(rf_sorted[i + 1] <= rf_sorted[i] + 0.05 for i in range(len(rf_sorted) - 1)), \
        f"A8: sweep-B fixed retention should be non-increasing in lambda2 ({rf_sorted})"
    assert all(rg >= rf - 0.02 for rf, rg in zip(ret_fixed, ret_gated)), \
        "A8: gated retention should dominate fixed at every lambda2"
    # B-series: applied BMR (probe-calibrated; the INVARIANCE is the finding)
    for bm, d in bmr_delta.items():
        assert d < 0.5, \
            f"B1: BMR should leave deposit-built couplings unchanged ({bm}: {d:.3f})"
    assert max(fl.max() for fl in bmr_flags.values()) > 0.05, \
        "B2: the ledger should flag at least some zero-prior edges (Occam)"
    assert all(r["min_eig"] > 0.0 for r in rows if "min_eig" in r), \
        "B3: applied-BMR nets must stay positive-definite"
    assert gated_bmr_cross >= 0.5 * disc, \
        f"B4: gate + BMR should still retain the pluralism ({gated_bmr_cross / disc:.1%})"

    # ---- figures ----
    _fig_traces(rows, trio, out_dir / "fig_divergence_traces.png")
    _fig_portrait(rows, trio, commits, p["VALUE_A"], p["VALUE_B"],
                  out_dir / "fig_contested_portrait.png",
                  three_camps="three_camps" if p["INCLUDE_THREE_CAMPS"] else None)
    _fig_collective(rows, trio, p["EDGE_THRESHOLD"],
                    out_dir / "fig_collective_vs_individual.png")
    _fig_gate_strength(rows, nus, disc, fixed / disc, nu_star,
                       out_dir / "fig_gate_strength.png")
    _fig_connectivity(rows, fixed_pts, gated_pts, disc,
                      out_dir / "fig_connectivity_boundary.png")
    if p["INCLUDE_THREE_CAMPS"]:
        _fig_bmr(rows, p["BMR_EVERY"], prior_by_commit,
                 out_dir / "fig_bmr_three_camps.png")

    # ---- save ----
    snap_t = _rows(rows, "disconnected")[0]["snap_t"]
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        snap_t=snap_t,
        commits=np.array(commits),
        **{f"{key}_{name}": _mean(rows, name, key)
           for name in ("disconnected", "connected_fixed", gated_star)
           for key in ("cross_fro", "within_fro", "cross_jac", "within_jac",
                       "cross_spec", "within_spec", "c_t")},
        nu_sweep=np.asarray(nus),
        retention_vs_nu=np.asarray([ret_nu[nu] for nu in nus]),
        lambda2_fixed=np.asarray(lam_fixed),
        retention_fixed=np.asarray(ret_fixed),
        retention_gated=np.asarray(ret_gated),
    )
    summary = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
        "nu_star": nu_star,
        "final_cross_fro": {"disconnected": disc, "connected_fixed": fixed,
                            "connected_gated": gated},
        "retention": {"fixed": fixed / disc, "structure_gated": gated / disc,
                      "means_gated": probe / disc},
        "probe_means_gate": {
            "nu": p["PROBE_MEANS_NU"], "cross": probe, "retention": probe / disc,
            "finding": "trust that reads opinions cannot protect a pluralism that "
                       "lives in wiring: the camps agree in means (both fit the same "
                       "blended data), so the means-gate collapses like fixed trust",
        },
        "reldist": {"disconnected": rel_d, "connected_fixed": rel_f},
        "collective_vs_individual": {
            name: {"union": float(_mean(rows, name, "union_count")),
                   "best_individual": float(_mean(rows, name, "best_individual_count")),
                   "consensus": float(_mean(rows, name, "consensus_edge_count")),
                   "dilution_ratio": float(_mean(rows, name, "dilution_ratio")),
                   "synergy": 0}
            for name, _ in trio},
        "sweep_A_retention_vs_nu": {str(nu): ret_nu[nu] for nu in nus},
        "sweep_B": {"lambda2": lam_fixed, "retention_fixed": ret_fixed,
                    "retention_gated": ret_gated},
        "mechanism": "same Student-t trust gate, pointed at the coupling pattern "
                     "(wiring) instead of the posterior means (opinions); bistable "
                     "in nu -- divergence must bootstrap before fusion erases it",
    }
    if p["INCLUDE_THREE_CAMPS"]:
        c3 = _mean(rows, "three_camps", "c_t")
        summary["three_camps"] = {
            "final_by_commitment": {f"camp_{b}": c3[b, -1].tolist() for b in range(3)},
            "finding": "PREDICTION REFUTED (reported honestly): the camp valuing the "
                       "unsupported theory (modified_gravity) consolidates its wiring "
                       "exactly as strongly as the supported camps. In linear-Gaussian "
                       "Fisher learning the deposit J = H^T diag(w) H / sigma^2 never "
                       "sees the observation, so attention alone fabricates the "
                       "precision-pattern edge; the data's verdict on an unsupported "
                       "theory lives in the potential h (the means), not in |Pi|. "
                       "Structure-as-precision-pattern is attention-driven in this "
                       "model class -- a scope condition on the pluralism results; "
                       "see 'applied_bmr': in-loop model reduction does not remove "
                       "it either.",
        }
    summary["applied_bmr"] = {
        "bmr_every": p["BMR_EVERY"],
        "max_coupling_change": bmr_delta,
        "flags_by_camp": {bm: fl.tolist() for bm, fl in bmr_flags.items()},
        "gated_bmr_retention": gated_bmr_cross / disc,
        "prior_coupling_by_commitment": np.asarray(prior_by_commit).tolist(),
        "finding": "applied Savage-Dickey BMR leaves this pluralism numerically "
                   "unchanged (max coupling change "
                   f"{max(bmr_delta.values()):.3f}): the contested couplings are "
                   "~100% attention-driven experiment DEPOSIT, and the reference "
                   "prior's contested couplings are ~0, so the prune -- which by "
                   "construction removes PRIOR structure while keeping the data -- "
                   "has nothing to bite. The scope condition, made precise: "
                   "evidence-driven deletion adjudicates believed prior structure; "
                   "it cannot un-run your experiments. Evidence-accountability for "
                   "deposit-built structure needs a deposit-level mechanism "
                   "(retrospective reweighting / forgetting), not model reduction.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays / figures / summary to {out_dir}")
    print(f"HEADLINE: divergent-value communities in an underdetermined world learn "
          f"DIFFERENT NETWORK STRUCTURES (cross {disc:.1f} vs within "
          f"{float(_mean(rows, 'disconnected', 'within_fro')[-1]):.1f}); plain contact "
          f"destroys the pluralism (fixed trust collapses to {fixed / disc:.0%} of "
          f"disconnected) and trust that reads OPINIONS cannot save it (means gate: "
          f"{probe / disc:.0%}) -- but trust that reads WIRING retains {gated / disc:.0%} "
          f"under full contact (nu*={nu_star}). The collective's union holds both wirings "
          f"({float(_mean(rows, 'disconnected', 'union_count')):.0f} edges vs best "
          f"individual {float(_mean(rows, 'disconnected', 'best_individual_count')):.0f}); "
          f"its consensus holds neither crisply (dilution {dil:.2f}); averaging provably "
          f"creates no edge no member holds (synergy = 0, verified).")


register(ExperimentSpec(
    model="phlogiston",
    name="divergent_wiring",
    description="Trust that reads WIRING preserves structural pluralism under full "
                "contact where fixed trust and opinion-reading trust both collapse it; "
                "the collective's surplus lives in the union of nets, never the "
                "consensus (synergy = 0, the averaging theorem-let).",
    run=run,
    out_dir="divergent_wiring",
    params=dict(
        N=60, SEEDS=(0, 1, 2, 3, 4), OMEGA=0.9, N_STEPS=120, W_HI=1.0, W_LO=0.1,
        VALUE_A="dark_matter", VALUE_B="scale_variant_laws",
        COMMITS=("dark_matter", "modified_gravity", "scale_variant_laws"),
        # probe-located transition: collapse is self-confirming above nu ~ 0.1-0.2,
        # so the sweep straddles the boundary (plan's prior 0.4-0.5 sits post-collapse)
        NU_SWEEP=(0.02, 0.05, 0.1, 0.2, 0.5),
        INTER_SWEEP=(0.0, 0.01, 0.05, 0.2),
        SOCIAL_GATE="structure",
        SCALE_FLOOR=0.1,        # inert here: the "lacking" cluster sits at |Pi| ~ 4.0
        # calibrated seed-0 disconnected: median |Pi| valued 40.17, rival 4.00
        EDGE_THRESHOLD=12.7,    # sqrt(40.17 * 4.00)
        PROBE_MEANS_NU=0.5,
        # CALIB (probe, seed 0): the dm reference prior's contested couplings are
        # ~[-0.45, -0.35, 0, 0, 0, 0], so the fabricated wiring is ~100% deposit and
        # applied BMR is an INVARIANCE check here; lam (0.2) is irrelevant (dU ~ 0).
        BMR_EVERY=10,
        SNAPSHOT_EVERY=5, MAX_WORKERS=6,
        INCLUDE_THREE_CAMPS=True, INCLUDE_BAYESNET=False,
    ),
    seeds=(0, 1, 2, 3, 4),
    consumes=dict(figures=["fig_divergence_traces", "fig_contested_portrait",
                           "fig_collective_vs_individual", "fig_gate_strength",
                           "fig_connectivity_boundary", "fig_bmr_three_camps"]),
))
