"""Endogenous structure learning: a reproducible simulation of model expansion as an action.

This is nb37's pipeline (accumulate data -> propose a hub -> score {sample, reduce, expand}
-> wake the hub -> measure) with its ONE artificial part removed. nb37 hand-planted the
residual; here the world is a true latent generative net (``world_net.true_phlogiston_world``)
carrying a HIDDEN cause the agent's paradigm menu does not represent. The agent samples from it,
and the residual -- and the wake it triggers -- EMERGE over time, unplanted.

Run it::

    python scripts/run_structure_learning.py

It prints the "mechanisms in a row" for one representative run (initial state -> floor rises ->
trigger crosses -> wake fires -> structure recovered), saves a figure of floor(t)/ΔF(t)/recovery(t)
with the wake instant marked, then runs one larger sweep over WORLD properties
(latent_coupling x noise sigma x seeds) plus the node-vs-relational identifiability control, and
writes ``simulation_arrays.npz`` + ``summary.json`` to ``results/structural_learning_endogenous/``.

Why a script and not a notebook (and why a host loop): the wake GROWS the agent's basis, a
variable-dimension move that cannot live inside a ``jax.lax.scan`` (all ``step.run_*`` are fixed
shape). So the per-step traversal is an explicit Python loop -- the reproducible "here is the
initial state, here is how you traverse each step" the experiment wants. The identifiability gate
(``fisher_deposit`` gives ``J = HᵀH/σ²``, so a node-wise agent's residual lives in the prediction
ERRORS, not in ``Π``) is why the trigger is read via ``action.residual_from_errors``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:                                  # Windows consoles default to cp1252; keep unicode prints alive
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.structural import action, world_net, phlogiston as ph, world
from src.structural.belief import GaussianBeliefNet, add_fisher
from src.structural.phlogiston import StructuralConfig
from src.structural.world_net import LatentWorldConfig, UNCONCEIVED_DRIVES


@dataclass(frozen=True)
class RunConfig:
    n_steps: int = 80
    sigma_obs: float = 1.0
    t_shift: int = 200            # > n_steps => STATIONARY world (isolate accumulation from a flip)
    trigger: float = 2.0          # min residual-floor height to consider the wake (the gate)
    hub_self_prec: float = 2.0
    hub_name: str = "discovered_cause"
    drives: tuple[str, ...] = UNCONCEIVED_DRIVES


@dataclass(frozen=True)
class StepRecord:
    floor: float          # residual-floor height lambda_max(R) -- the TRIGGER
    rank1: float          # how localized (rank-1) the residual is
    delta_F: float        # expansion model Bayes factor
    delta_G: float        # epistemic gain
    score: float
    accepted: bool
    action: str           # sample | reduce | expand
    recovery: float       # un-woken agent's gap to the true driven-node marginal


def _relational_H(cfg: StructuralConfig, drives) -> np.ndarray:
    """``H_observable`` PLUS one relational row reading the SUM of the driven nodes -- a channel
    that measures the common-cause combination directly, so its off-diagonal Fisher enters ``Π``
    and the coupling becomes directly representable (the identifiability control: the wake should
    then be DECLINED, ΔF<0)."""
    H = np.asarray(ph.H_observable(cfg))
    row = np.zeros((1, len(cfg.node_names)))
    for n in drives:
        row[0, cfg.node_names.index(n)] = 1.0
    return np.vstack([H, row])


def single_run(cfg: StructuralConfig, lw: LatentWorldConfig, run_cfg: RunConfig,
               *, relational: bool = False, seed: int = 0,
               full_trace: bool = False) -> dict:
    """Traverse the world step by step on the fixed phlogiston menu, scoring the expansion action
    each step. Returns telemetry. ``relational=True`` adds the sum-of-drives channel (the control).

    THE INITIAL STATE: a single agent holding the phlogiston prior (the latent absent from its
    menu), observing node-wise (or +relational), in a stationary latent world.
    THE PER-STEP TRAVERSAL: sample the world -> record the prediction error vs the menu's prediction
    -> deposit (node-wise Fisher) -> read the residual floor from the accumulated errors ->
    propose a hub -> score expand vs reduce vs sample -> (first acceptance) wake the hub and
    measure recovery."""
    drives = run_cfg.drives
    meas = ph.measured_nodes(cfg)
    H_node = jnp.asarray(np.asarray(ph.H_observable(cfg)))
    H_rel = jnp.asarray(_relational_H(cfg, drives))
    dmeas = [meas.index(n) for n in drives]                 # drive positions in the obs vector
    dfull = [cfg.node_names.index(n) for n in drives]
    # the agent's belief: a model that simply LACKS the hidden cause (drives held independent), so
    # recovery is well-posed -- the wake fills exactly the missing coupling.
    prior = world_net.agnostic_prior(cfg, lw)
    prior_mean = np.asarray(prior.mean())
    prune_edge = drives[:2]                                  # a candidate edge to score reduce on

    agent = prior
    errors: list[np.ndarray] = []
    key = jax.random.PRNGKey(seed)
    recs: list[StepRecord] = []
    wake_step = -1
    recovery_unwoken_at_wake = float("nan")
    recovery_woken_at_wake = float("nan")
    f1_unwoken_at_wake = float("nan")
    f1_woken_at_wake = float("nan")

    for t in range(run_cfg.n_steps):
        wnet = world_net.true_phlogiston_world(cfg, t, lw)
        key, k1 = jax.random.split(key)
        o = np.asarray(world_net.sample_world(wnet, cfg, k1))        # (m,) measured nodes
        # prediction error vs the MENU's prediction (the prior mean -- the unmodelled shift stays)
        errors.append(o[dmeas] - prior_mean[dfull])

        if relational:
            o_rel = jnp.asarray(np.concatenate([o, [o[dmeas].sum()]]))
            J, j = world.fisher_deposit(H_rel, o_rel, run_cfg.sigma_obs)
        else:
            J, j = world.fisher_deposit(H_node, jnp.asarray(o), run_cfg.sigma_obs)
        agent = add_fisher(agent, J, j)

        # the residual: from Π when relational (coupling is deposited there), from the
        # prediction ERRORS when node-wise (Π stays diagonal -- the footprint is in the means).
        R = None if relational else action.residual_from_errors(jnp.asarray(np.array(errors)))
        out = action.select_action(agent, prior, drives, run_cfg.hub_name,
                                   prune_edges=(prune_edge,), residual=R,
                                   hub_self_prec=run_cfg.hub_self_prec,
                                   trigger=run_cfg.trigger)
        exp = out["expand"]
        rec = world_net.recovery_scores(agent, wnet, drives)
        recs.append(StepRecord(floor=exp.detail["strength"], rank1=exp.detail["rank1_ratio"],
                               delta_F=exp.delta_F, delta_G=exp.delta_G, score=exp.score,
                               accepted=exp.accept, action=out["action"],
                               recovery=rec["precision_frobenius"]))

        if out["action"] == "expand" and wake_step < 0:
            wake_step = t
            woken = action.wake_hub(agent, out["proposal"], run_cfg.hub_name,
                                    hub_self_prec=run_cfg.hub_self_prec)
            rec_w = world_net.recovery_scores(woken, wnet, drives)
            recovery_unwoken_at_wake = rec["precision_frobenius"]
            recovery_woken_at_wake = rec_w["precision_frobenius"]
            f1_unwoken_at_wake = rec["edge_f1"]
            f1_woken_at_wake = rec_w["edge_f1"]

    tele = {
        "floor_t": np.array([r.floor for r in recs]),
        "delta_F_t": np.array([r.delta_F for r in recs]),
        "delta_G_t": np.array([r.delta_G for r in recs]),
        "score_t": np.array([r.score for r in recs]),
        "recovery_t": np.array([r.recovery for r in recs]),
        "action_t": [r.action for r in recs],
        "wake_step": wake_step,
        "recovery_unwoken_at_wake": recovery_unwoken_at_wake,
        "recovery_woken_at_wake": recovery_woken_at_wake,
        "f1_unwoken_at_wake": f1_unwoken_at_wake,
        "f1_woken_at_wake": f1_woken_at_wake,
        # structure recovered = the wake wires the RIGHT edges (sign + nodes); magnitude may
        # overshoot (the wake's coupling is a PD heuristic, not fit), so we judge on edge_f1.
        "structure_recovered": bool(wake_step >= 0 and f1_woken_at_wake > f1_unwoken_at_wake),
    }
    return tele


def scheduled_run(cfg: StructuralConfig, run_cfg: RunConfig,
                  causes: tuple, *, seed: int = 0) -> dict:
    """The BASELINE: a true structure that CHANGES multiple times. Each ``CauseSpec`` switches a
    hidden cause on at its ``activate`` step (driving a disjoint node-set). The agent holds an
    agnostic prior and, each step, scores an expansion for EACH cause-set not yet discovered --
    waking a new node when that set's residual clears the trigger. We track whether the count of
    woken nodes TRACKS the count of active true causes (does structure learning make more nodes when
    the world does), per-cause discovery lag, and the recovered edge structure of every cause."""
    meas = ph.measured_nodes(cfg)
    H = jnp.asarray(np.asarray(ph.H_observable(cfg)))
    prior = world_net.agnostic_prior(cfg, LatentWorldConfig())
    prior_mean = np.asarray(prior.mean())
    pos = {c.name: [meas.index(n) for n in c.drives] for c in causes}      # obs positions per cause

    agent = prior
    errors: list[np.ndarray] = []
    key = jax.random.PRNGKey(seed)
    woken: dict[str, object] = {}          # name -> proposal (the structural decisions so far)
    discovery_step = {c.name: -1 for c in causes}
    streak = {c.name: 0 for c in causes}   # consecutive steps the floor has cleared the trigger
    n_active_t, n_woken_t = [], []
    SUSTAIN = 5                            # require a STUBBORN floor, not a one-sample spike

    for t in range(run_cfg.n_steps):
        wnet = world_net.scheduled_world(cfg, t, causes)
        key, k1 = jax.random.split(key)
        o = np.asarray(world_net.sample_world(wnet, cfg, k1))
        errors.append(o - prior_mean[[cfg.node_names.index(n) for n in meas]])
        J, j = world.fisher_deposit(H, jnp.asarray(o), run_cfg.sigma_obs)
        agent = add_fisher(agent, J, j)

        E = np.array(errors)                                               # (t+1, m)
        for c in causes:
            if c.name in woken:
                continue
            R = action.residual_from_errors(jnp.asarray(E[:, pos[c.name]]))
            prop = action.propose_hub(agent, c.drives, residual=R)
            sc = action.expansion_score(agent, prior, prop, c.name)
            # the trigger must be SUSTAINED -- a real hidden cause holds the floor up; a noise
            # fluctuation spikes once and decays (kills the small-sample false positives).
            streak[c.name] = streak[c.name] + 1 if (sc.accept and prop.strength >= run_cfg.trigger) else 0
            if streak[c.name] >= SUSTAIN:
                woken[c.name] = prop
                discovery_step[c.name] = t
        n_active_t.append(sum(1 for c in causes if c.activate <= t))
        n_woken_t.append(len(woken))

    # recovery: border every woken hub onto the final agent, score each cause's edge structure
    final = agent
    for name, prop in woken.items():
        final = action.wake_hub(final, prop, name, hub_self_prec=run_cfg.hub_self_prec)
    wfinal = world_net.scheduled_world(cfg, run_cfg.n_steps - 1, causes)
    recovery_f1 = {c.name: world_net.recovery_scores(final, wfinal, c.drives)["edge_f1"]
                   for c in causes if c.name in woken}
    return {
        "n_active_t": np.array(n_active_t), "n_woken_t": np.array(n_woken_t),
        "discovery_step": discovery_step, "recovery_f1": recovery_f1,
        "activations": {c.name: c.activate for c in causes},
        "learned_net": final, "woken_names": tuple(woken.keys()),
    }


# ----------------------------------------------------------------------
# Graph-structure visualisation: draw the Bayes net (nodes + edges), with the
# discovered hidden-cause nodes highlighted.
# ----------------------------------------------------------------------

def _short(name: str) -> str:
    """A compact node label."""
    if name in ("cause_A", "cause_B", "cause_C"):
        return name[-1]                       # A / B / C
    parts = name.split("_")
    return parts[0][:4] + (parts[1][:3] if len(parts) > 1 else "")


def _layout(net, base_names, hub_names):
    """Base commitments on an outer ring, discovered/hidden cause-hubs on an inner ring."""
    pos = {}
    nb = len(base_names)
    for i, n in enumerate(base_names):
        th = 2 * np.pi * i / nb + np.pi / 2
        pos[n] = (np.cos(th), np.sin(th))
    nh = max(len(hub_names), 1)
    for i, n in enumerate(hub_names):
        th = 2 * np.pi * i / nh + np.pi / 2
        pos[n] = (0.42 * np.cos(th), 0.42 * np.sin(th))
    return pos


def _draw_net(ax, net, base_names, hub_names, title, edge_threshold=0.15):
    pos = _layout(net, base_names, hub_names)
    names = list(net.names)
    Pi = np.asarray(net.Pi)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            w = abs(float(Pi[i, j]))
            if w > edge_threshold and names[i] in pos and names[j] in pos:
                xi, yi = pos[names[i]]; xj, yj = pos[names[j]]
                col = "crimson" if Pi[i, j] < 0 else "steelblue"
                ax.plot([xi, xj], [yi, yj], color=col, lw=min(3.0, 0.6 + w), alpha=0.7, zorder=1)
    for n in names:
        if n not in pos:
            continue
        x, y = pos[n]
        is_hub = n in hub_names
        ax.scatter([x], [y], s=520 if is_hub else 300,
                   c="orange" if is_hub else "#cfe3f7",
                   edgecolor="black", linewidth=1.2, zorder=2)
        ax.text(x, y, _short(n), ha="center", va="center", fontsize=7, zorder=3,
                fontweight="bold" if is_hub else "normal")
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)


def draw_scheduled_graphs(cfg, causes, learned_net, true_net, path):
    """Side-by-side Bayes-net graphs: the TRUE structure (observed commitments + the hidden
    cause-nodes wired to their triples) and the AGENT's LEARNED structure after discovery. The
    woken/hidden cause-hubs are the orange nodes; edges are couplings (blue +, red -)."""
    base = list(cfg.node_names)
    hubs = [c.name for c in causes]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2))
    _draw_net(axs[0], true_net, base, hubs, "TRUE world: observed commitments + 3 hidden causes")
    _draw_net(axs[1], learned_net, base, hubs, "AGENT learned: discovered cause-nodes (orange)")
    fig.suptitle("Graph structure — the paradigm grows a node per hidden cause", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def mechanisms_in_a_row(cfg, lw, run_cfg, tele):
    """Print the ordered story of the representative run."""
    f, dF = tele["floor_t"], tele["delta_F_t"]
    ws = tele["wake_step"]
    print("\n=== mechanisms in a row (representative run) ===")
    print(f"  [init]   agent's model holds the driven nodes INDEPENDENT; the latent '{lw.latent_name}' "
          f"drives {run_cfg.drives} at coupling {lw.latent_coupling} -- ABSENT from the menu.")
    print(f"  [t=0]    residual floor = {f[0]:.2f}, dF = {dF[0]:+.2f}  (nothing learned yet)")
    print(f"  [rises]  floor climbs as the coherent shift accumulates: "
          f"t=10 -> {f[min(10,len(f)-1)]:.2f}, t={len(f)//2} -> {f[len(f)//2]:.2f}")
    if ws >= 0:
        print(f"  [cross]  the wake turns favourable; first accepted EXPAND at t={ws} "
              f"(floor {f[ws]:.2f} >= trigger {run_cfg.trigger}, dF {dF[ws]:+.2f})")
        print(f"  [wake]   the hidden cause '{run_cfg.hub_name}' is wired in -- the paradigm grows a node.")
        print(f"  [recover] driven-node edge structure F1: {tele['f1_unwoken_at_wake']:.2f} "
              f"(no hub) -> {tele['f1_woken_at_wake']:.2f} (woken)  "
              f"structure_recovered={tele['structure_recovered']}")
        print(f"           (magnitude gap {tele['recovery_unwoken_at_wake']:.2f} -> "
              f"{tele['recovery_woken_at_wake']:.2f}: the wake finds the right edges + sign but the "
              f"PD-heuristic coupling can overshoot the true magnitude -- an honest caveat.)")
    else:
        print("  [no wake] the floor never cleared the trigger -- expansion not warranted (control/null).")


def make_figure(tele, run_cfg, path):
    f, dF, rec = tele["floor_t"], tele["delta_F_t"], tele["recovery_t"]
    ws = tele["wake_step"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, y, ttl, ylab in [(axs[0], f, "residual floor (the TRIGGER)", r"$\lambda_{max}(R)$"),
                             (axs[1], dF, "expansion model Bayes factor", r"$\Delta F$"),
                             (axs[2], rec, "structure recovery (gap to true)", "‖Π_learn − Π_true‖ (drives)")]:
        ax.plot(y, lw=2)
        if ttl.startswith("residual"):
            ax.axhline(run_cfg.trigger, color="grey", ls=":", lw=1, label="trigger")
            ax.legend(fontsize=8)
        if ttl.startswith("expansion"):
            ax.axhline(0, color="grey", lw=1)
        if ws >= 0:
            ax.axvline(ws, color="seagreen", ls="--", lw=1.5)
        ax.set_title(ttl); ax.set_xlabel("step"); ax.set_ylabel(ylab)
    if ws >= 0:
        axs[0].text(ws, axs[0].get_ylim()[1]*0.9, f" wake t={ws}", color="seagreen", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def main() -> int:
    cfg = StructuralConfig(n_steps=80, t_shift=200)
    run_cfg = RunConfig()
    out_dir = ROOT / "results" / "structural_learning_endogenous"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- representative run: a strong unconceived cause, node-wise ----------
    lw_rep = LatentWorldConfig(latent_coupling=1.2, latent_mean=1.5)
    rep = single_run(cfg, lw_rep, run_cfg, relational=False, seed=0)
    mechanisms_in_a_row(cfg, lw_rep, run_cfg, rep)
    make_figure(rep, run_cfg, out_dir / "endogenous_discovery.png")

    # ---- identifiability probe: same cause, RELATIONAL channel that spans the drives ----
    rel = single_run(cfg, lw_rep, run_cfg, relational=True, seed=0)
    print("\n=== identifiability probe (same latent, RELATIONAL channel spans the drives) ===")
    print(f"  node-wise:   wake at t={rep['wake_step']}, final dF={rep['delta_F_t'][-1]:+.2f}")
    print(f"  relational:  wake at t={rel['wake_step']}, final dF={rel['delta_F_t'][-1]:+.2f}")
    print("  FINDING (honest): unlike nb37's pure-precision residual (where a relational channel makes")
    print("  the coupling directly representable and dF<0 declines the hub), here the latent's COHERENT")
    print("  MEAN-SHIFT dominates dF, so the hub is favoured under BOTH operators. The expansion score")
    print("  rewards mean-alignment, not only structural necessity -- a limit to report, not hide.")

    # ---- the larger sweep: WORLD properties (coupling x noise x seeds), node-wise ----
    couplings = [0.0, 0.4, 0.8, 1.2, 1.6]
    sigmas = [0.5, 1.0, 2.0, 4.0]
    seeds = [0, 1, 2]
    nC, nS = len(couplings), len(sigmas)
    wake_frac = np.zeros((nS, nC))            # fraction of seeds that discovered (woke)
    time_to_wake = np.full((nS, nC), np.nan)  # mean wake step among those that woke
    final_dF = np.zeros((nS, nC))
    rec_improve = np.zeros((nS, nC))          # recovery gap reduction from the wake
    print("\n=== sweep: latent_coupling x sigma (node-wise, mean over seeds) ===")
    for iS, sig in enumerate(sigmas):
        for iC, cpl in enumerate(couplings):
            woke, steps, dFs, improves = [], [], [], []
            for sd in seeds:
                lw = LatentWorldConfig(latent_coupling=cpl, latent_mean=1.5)
                te = single_run(cfg, lw, RunConfig(sigma_obs=sig), relational=False, seed=sd)
                woke.append(te["wake_step"] >= 0)
                if te["wake_step"] >= 0:
                    steps.append(te["wake_step"])
                    if te["structure_recovered"]:
                        improves.append(te["f1_woken_at_wake"] - te["f1_unwoken_at_wake"])
                dFs.append(float(te["delta_F_t"][-1]))
            wake_frac[iS, iC] = float(np.mean(woke))
            time_to_wake[iS, iC] = float(np.mean(steps)) if steps else np.nan
            final_dF[iS, iC] = float(np.mean(dFs))
            rec_improve[iS, iC] = float(np.mean(improves)) if improves else 0.0
        row = "  ".join(f"{wake_frac[iS, iC]:.2f}" for iC in range(nC))
        print(f"  sigma={sig:>4}: wake-fraction over couplings {couplings} = [{row}]")

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        couplings=np.array(couplings), sigmas=np.array(sigmas), seeds=np.array(seeds),
        wake_fraction=wake_frac, time_to_wake=time_to_wake, final_delta_F=final_dF,
        recovery_improvement=rec_improve,
        rep_floor_t=rep["floor_t"], rep_delta_F_t=rep["delta_F_t"], rep_recovery_t=rep["recovery_t"],
        rep_wake_step=np.array(rep["wake_step"]),
    )
    # sweep heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, M, ttl in [(axs[0], wake_frac, "wake fraction (discovery)"),
                       (axs[1], final_dF, "final ΔF (mean)")]:
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis",
                       extent=[couplings[0], couplings[-1], 0, len(sigmas)])
        ax.set_xlabel("latent coupling"); ax.set_yticks(np.arange(len(sigmas)) + 0.5)
        ax.set_yticklabels(sigmas); ax.set_ylabel("noise sigma"); ax.set_title(ttl)
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); plt.savefig(out_dir / "sweep_phase.png", dpi=110); plt.close(fig)

    # ---- the changing-structure baseline: does #nodes TRACK #true-causes? ----
    causes = (
        world_net.CauseSpec(activate=0, name="cause_A",
                            drives=("reduction_with_charcoal", "respiration_like_combustion", "air_has_capacity")),
        world_net.CauseSpec(activate=25, name="cause_B",
                            drives=("combustion_releases", "calcination_releases", "metal_is_calx_plus_phlog")),
        world_net.CauseSpec(activate=50, name="cause_C",
                            drives=("mass_change_sign", "gas_consumed", "calx_heavier_than_metal")),
    )
    sched = scheduled_run(cfg, run_cfg, causes, seed=0)
    print("\n=== changing-structure baseline: a hidden cause is ADDED at t=0, 25, 50 ===")
    print("  discovery step per cause:", sched["discovery_step"], "(activations:", sched["activations"], ")")
    print("  recovered edge-F1 per cause:", {k: round(v, 2) for k, v in sched["recovery_f1"].items()})
    na, nw = sched["n_active_t"], sched["n_woken_t"]
    print(f"  #true-active vs #nodes-woken:  t=24 -> {na[24]}/{nw[24]}   "
          f"t=49 -> {na[49]}/{nw[49]}   t=79 -> {na[-1]}/{nw[-1]}")
    print(f"  => the agent grows {nw[-1]} node(s) as the world reveals {na[-1]} hidden cause(s).")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.step(range(len(na)), na, where="post", lw=2, label="true active causes")
    ax.step(range(len(nw)), nw, where="post", lw=2, ls="--", label="nodes woken (agent)")
    for c in causes:
        ax.axvline(c.activate, color="grey", ls=":", lw=1)
    ax.set_xlabel("step"); ax.set_ylabel("count"); ax.set_yticks([0, 1, 2, 3])
    ax.set_title("does the agent grow nodes as the true structure adds causes?")
    ax.legend(fontsize=9); plt.tight_layout()
    plt.savefig(out_dir / "scheduled_tracking.png", dpi=110); plt.close(fig)

    # the graph structure itself: true vs learned Bayes net, cause-hubs highlighted
    true_final = world_net.scheduled_world(cfg, run_cfg.n_steps - 1, causes).to_info()
    draw_scheduled_graphs(cfg, causes, sched["learned_net"], true_final,
                          out_dir / "scheduled_graph.png")
    print(f"  graph structure saved -> {out_dir / 'scheduled_graph.png'}")

    summary = {
        "run_config": asdict(run_cfg),
        "changing_structure_baseline": {
            "activations": sched["activations"], "discovery_step": sched["discovery_step"],
            "recovery_f1": {k: float(v) for k, v in sched["recovery_f1"].items()},
            "n_active_final": int(na[-1]), "n_woken_final": int(nw[-1])},
        "representative": {"latent_coupling": lw_rep.latent_coupling, "wake_step": rep["wake_step"],
                           "structure_recovered": rep["structure_recovered"],
                           "edge_f1_unwoken": rep["f1_unwoken_at_wake"],
                           "edge_f1_woken": rep["f1_woken_at_wake"],
                           "magnitude_gap_unwoken": rep["recovery_unwoken_at_wake"],
                           "magnitude_gap_woken": rep["recovery_woken_at_wake"]},
        "control": {"node_final_delta_F": float(rep["delta_F_t"][-1]),
                    "relational_final_delta_F": float(rel["delta_F_t"][-1]),
                    "node_wake_step": rep["wake_step"], "relational_wake_step": rel["wake_step"]},
        "sweep": {"couplings": couplings, "sigmas": sigmas, "n_seeds": len(seeds),
                  "wake_fraction": wake_frac.tolist(), "final_delta_F": final_dF.tolist()},
        "null_control": {"coupling_0_wake_fraction": float(wake_frac[:, 0].mean())},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nsaved arrays/figures/summary to {out_dir}")
    print(f"null control (coupling=0): wake fraction = {wake_frac[:, 0].mean():.2f} (want 0.00)")
    print(f"strong cause (coupling=1.6, low noise): wake fraction = {wake_frac[0, -1]:.2f} (want 1.00)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
