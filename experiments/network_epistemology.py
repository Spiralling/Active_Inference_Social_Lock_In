"""Network epistemology, field-standard idioms: the Zollman figures, and why the
rival architecture DISSOLVES the speed/accuracy tradeoff.

The network-epistemology literature (Zollman 2007/2010; Bala-Goyal 1998) is organized
around a speed/ACCURACY TRADEOFF: sparser communication networks reach the correct
consensus more RELIABLY (transient diversity resists premature herding) but more
SLOWLY. That tradeoff is a property of the consensus-only representation -- agents who
pool a single belief can herd onto the wrong arm before the evidence arrives.

This experiment runs the canonical figures (accuracy and speed vs algebraic
connectivity lambda_2; the speed/accuracy scatter; the four reference topologies) on
the represented-rival engine (``src/structural/rival.py``) and finds the tradeoff GONE
-- for two distinct, both-honest reasons, separated by regime:

  * DISCOVERY is topology-free. When every agent independently accumulates prequential
    model evidence (homogeneous attention), the population reaches the right paradigm
    at the same rate on EVERY topology -- there is no disagreement to herd, so network
    structure is irrelevant. (Verified in calibration; the probe's homogeneous arm was
    flat across complete/wheel/cycle.)
  * DIFFUSION helps monotonically. When the better paradigm is discovered by a VANGUARD
    (a 4/24 minority that keeps running the disconfirming experiment) and must spread to
    a self-censoring majority, connectivity helps BOTH speed and accuracy -- a contagion
    regime (Granovetter / Centola), not Zollman's discovery regime. There is no
    diversity benefit to sparsity; sparsity just slows or blocks the spread.

Two regimes make the dual result legible:
  * GAMMA_HARD = 1.0 (zero-source conservatives): an ACCURACY threshold. complete and
    wheel convert; cycle and community never reach the rival paradigm within the
    horizon. The wheel is the lesson -- it converts despite a low lambda_2 because the
    vanguard occupies its HUB (Zollman's "royal family" broadcaster): one well-placed
    agent beats raw connectivity. Centrality, not density.
  * GAMMA_EASY = 0.99 (trickle conservatives): everyone converts, so topology orders
    SPEED ONLY -- monotone faster with lambda_2 (complete < wheel < community < cycle),
    a horizontal accuracy line with no tradeoff.

Demo H (the phase diagram) maps where contact converts / immunizes / never reaches the
rival paradigm over bridge-density beta x conservative self-censorship gamma -- locating
the immunization valley discovered in ``kuhnian_transition`` (weak contact spreads the
ACCOMMODATED incumbent and blocks conversion; strong contact converts) as a 2-D regime
map.

Honest scope: R = 200 seeds for the Zollman curves (publication-grade rates); R = 40 per
heatmap cell (color is coarse). The vanguard placement (nodes 0..NV-1) puts the hub in
the vanguard for the wheel by construction -- that IS the royal-family result, stated,
not hidden.
"""
from __future__ import annotations

import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

from experiments.registry import ExperimentSpec, register

RED, AMBER, GREEN, GREY, BLUE = "#922b21", "#b9770e", "#1e8449", "#7f8c8d", "#2c6fbb"
TOPOS = ("cycle", "wheel", "community", "complete")
TOPO_COLOR = {"cycle": RED, "wheel": AMBER, "community": BLUE, "complete": GREEN}


# ----------------------------------------------------------------------
# Worker side (module-level, picklable; lazy imports for ProcessPoolExecutor).
# ----------------------------------------------------------------------

# Per-WORKER world cache: built once, reused across every job in the process. The
# candidate menu (Pi0/h0/H) and the RivalConfig are IDENTICAL across all jobs (only the
# graph, attention, and seed vary, and those are run_rival ARGUMENTS) -- so reusing one
# cfgr object lets JAX compile the scan once per phase-length instead of per job (the
# 2560x-recompile that made the first attempt take ~75 min).
_WORLD: dict = {}


def _world(p):
    key = (p["N"], p["T"], p["T_SHIFT"], p["SIGMA_O"], p["OMEGA"], p["ALPHA_M"],
           p["CONVICTION"])
    if key not in _WORLD:
        import numpy as _np
        import jax.numpy as jnp
        from src.structural import rival as rv
        from src.structural.phlogiston import (
            StructuralConfig, phlogiston_bn, gravimetric_H, gravimetric_rows,
            phi_true_at, DISAGREEMENT_NODES)
        from experiments.rival_kuhn import oxygen_bn
        cfg = StructuralConfig(regime_schedule="step", t_shift=p["T_SHIFT"],
                               n_steps=p["T"], sigma_o=p["SIGMA_O"])
        inc = phlogiston_bn(cfg, conviction=p["CONVICTION"]).to_info()
        oxy = oxygen_bn(cfg, conviction=p["CONVICTION"]).to_info()
        H = gravimetric_H(cfg)
        rows = gravimetric_rows(cfg)
        disc = [i for i, r in enumerate(rows)
                if ("mass_balance" in r) or (r in DISAGREEMENT_NODES)]
        phis = jnp.asarray(_np.stack([_np.asarray(phi_true_at(cfg, t))
                                      for t in range(p["T"])]))
        cfgr = rv.RivalConfig(Pi0=jnp.stack([inc.Pi, oxy.Pi]),
                              h0=jnp.stack([inc.h, oxy.h]), H=H,
                              sigma_o=p["SIGMA_O"], omega=p["OMEGA"],
                              alpha_m=p["ALPHA_M"])
        _WORLD[key] = dict(cfgr=cfgr, H=H, disc=disc, phis=phis, m=int(H.shape[0]))
    return _WORLD[key]


def _graph(name, n, nv, beta):
    from src.structural import graphs
    if name == "complete":
        return graphs.complete(n)
    if name == "cycle":
        return graphs.ring(n, mean_degree=2)
    if name == "wheel":
        return graphs.wheel(n)
    # community: vanguard block (size nv) + conservative block, bridged at beta
    return graphs.community([nv, n - nv], intra=1.0, inter=float(beta), seed=0)


def _two_phase(p, world, graph, w_post, n, seed, per_agent=False):
    """Shared full-attention normal science to T_SHIFT, then ``w_post`` attention.
    Returns the conservative-block (nodes >= NV) mean q(oxygen) trace, or the full
    per-agent (T, N) trace with ``per_agent``. Reuses the per-worker cached
    ``world`` (cfgr/phis) so the scan compiles once."""
    import numpy as _np
    import jax.numpy as jnp
    from src.structural import rival as rv
    cfgr, phis, m = world["cfgr"], world["phis"], world["m"]
    W = graph.trust_W()
    A_self = jnp.asarray(_np.asarray(graph.A) + _np.eye(n))
    ts = p["T_SHIFT"]
    r1 = rv.run_rival(cfgr, W, A_self, jnp.ones((n, m)), phis[:ts], seed=seed,
                      track_means=False)
    r2 = rv.run_rival(cfgr, W, A_self, jnp.asarray(w_post), phis[ts:],
                      seed=seed + 1000, Pi_init=r1["Pi"], h_init=r1["h"],
                      L_init=r1["L"], track_means=False)
    q = _np.concatenate([r1["q_t"], r2["q_t"]])[:, :, 1]               # (T, N)
    return q if per_agent else q[:, p["N_VAN"]:].mean(axis=1)


def _one_job(cfg: dict) -> dict:
    import numpy as _np
    p, kind = cfg, cfg["kind"]
    n, nv = p["N"], p["N_VAN"]
    world = _world(p)
    disc, m = world["disc"], world["m"]
    if kind == "traj":
        # per-agent trajectories of the SAME transition: how the flip SHOWS UP on
        # each wiring (synchronized / hub broadcast / travelling wave / staircase)
        from src.structural import graphs
        graph = _graph(p["topo"], n, nv, p["Z_BETA"])
        w = _np.ones((n, m))
        w[nv:, disc] = 1.0 - p["gamma"]
        q = _two_phase(p, world, graph, w, n, int(p["seed"]), per_agent=True)
        return dict(kind="traj", topo=p["topo"], gamma=float(p["gamma"]),
                    seed=int(p["seed"]), q=_np.asarray(q, dtype=_np.float32),
                    lambda2=float(graphs.algebraic_connectivity(graph)),
                    finite=bool(_np.isfinite(q).all()))
    if kind in ("zollman", "sweep"):
        from src.structural import graphs
        graph = (_graph(p["topo"], n, nv, p["Z_BETA"]) if kind == "zollman"
                 else _graph("community", n, nv, p["beta"]))
        w = _np.ones((n, m))
        w[nv:, disc] = 1.0 - p["gamma"]                    # conservatives censor
        qc = _two_phase(p, world, graph, w, n, int(p["seed"]))
        conv = bool(qc[-1] > p["Q_CONS"])
        tconv = int(_np.argmax(qc > p["Q_CONS"])) if conv else p["T"]
        out = dict(kind=kind, gamma=float(p["gamma"]), seed=int(p["seed"]),
                   converged=conv, t_conv=tconv,
                   lambda2=float(graphs.algebraic_connectivity(graph)),
                   finite=bool(_np.isfinite(qc).all()))
        out["topo" if kind == "zollman" else "beta"] = (
            p["topo"] if kind == "zollman" else float(p["beta"]))
        return out
    # heatmap cell: community graph, beta x gamma
    graph = _graph("community", n, nv, p["beta"])
    w = _np.ones((n, m))
    w[nv:, disc] = 1.0 - p["gamma"]
    qc = _two_phase(p, world, graph, w, n, int(p["seed"]))
    return dict(kind="heat", beta=float(p["beta"]), gamma=float(p["gamma"]),
                seed=int(p["seed"]), q_end=float(qc[-1]),
                finite=bool(_np.isfinite(qc).all()))


def _mci(vals):
    a = np.asarray(vals, dtype=np.float64)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else np.nan), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


# ----------------------------------------------------------------------
# Figures.
# ----------------------------------------------------------------------

def _fig_topologies(p, out_path):
    import networkx as nx
    from src.structural import graphs
    n = 12
    builders = {"cycle": lambda: graphs.ring(n, mean_degree=2),
                "wheel": lambda: graphs.wheel(n),
                "community": lambda: graphs.community([n // 2, n // 2], intra=1.0,
                                                      inter=0.15, seed=0),
                "complete": lambda: graphs.complete(n)}
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.4))
    for ax, name in zip(axes, TOPOS):
        g = builders[name]()
        G = nx.from_numpy_array(np.asarray(g.A))
        lay = nx.circular_layout(G) if name != "wheel" else nx.spring_layout(G, seed=1)
        nx.draw(G, lay, ax=ax, node_size=90, node_color=TOPO_COLOR[name],
                edge_color="#cccccc", width=1.0)
        l2 = graphs.algebraic_connectivity(g)
        ax.set_title(f"{name}\n$\\lambda_2$={l2:.2f}", fontsize=9)
    fig.suptitle("the four reference topologies (the connectivity axis)", fontsize=11)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close(fig)


def _fig_curves(rows, p, out_path):
    zh = [r for r in rows if r["kind"] == "zollman" and r["gamma"] == p["GAMMA_HARD"]]
    ze = [r for r in rows if r["kind"] == "zollman" and r["gamma"] == p["GAMMA_EASY"]]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    # accuracy at GAMMA_HARD
    for name in TOPOS:
        rs = [r for r in zh if r["topo"] == name]
        l2 = rs[0]["lambda2"]
        m, ci = _mci([r["converged"] for r in rs])
        a0.errorbar([l2], [m], yerr=[ci], marker="o", ms=10, capsize=4,
                    color=TOPO_COLOR[name], label=name)
    a0.set_xscale("symlog", linthresh=0.1)
    a0.set_xlabel(r"algebraic connectivity $\lambda_2$")
    a0.set_ylabel("P(majority reaches the better paradigm)")
    a0.set_ylim(-0.05, 1.08)
    a0.set_title(f"ACCURACY regime ($\\gamma$={p['GAMMA_HARD']}, zero source):\n"
                 "a diffusion threshold -- centrality (wheel hub) beats density",
                 fontsize=9)
    a0.legend(fontsize=8, loc="center right")
    # speed at GAMMA_EASY
    for name in TOPOS:
        rs = [r for r in ze if r["topo"] == name and r["converged"]]
        l2 = [r for r in ze if r["topo"] == name][0]["lambda2"]
        if rs:
            m, ci = _mci([r["t_conv"] for r in rs])
            a1.errorbar([l2], [m], yerr=[ci], marker="s", ms=10, capsize=4,
                        color=TOPO_COLOR[name], label=name)
    a1.set_xscale("symlog", linthresh=0.1)
    a1.set_xlabel(r"algebraic connectivity $\lambda_2$")
    a1.set_ylabel("median time to consensus (steps)")
    a1.set_title(f"SPEED regime ($\\gamma$={p['GAMMA_EASY']}, trickle):\n"
                 "all converge -- connectivity orders speed, no accuracy cost",
                 fontsize=9)
    a1.legend(fontsize=8, loc="upper right")
    fig.suptitle("the Zollman speed/accuracy TRADEOFF dissolves in the rival "
                 "architecture: discovery is topology-free, diffusion helps both",
                 fontsize=11)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close(fig)


def _fig_sweep(rows, p, out_path):
    """The CONTINUOUS-change view: accuracy and speed as smooth curves over a single
    graph family (community bridge-density beta -> lambda_2), with the named
    topologies overlaid as labelled reference points."""
    sw = [r for r in rows if r["kind"] == "sweep"]
    betas = sorted({r["beta"] for r in sw})

    def at(gamma, key, want_conv=False):
        xs, ys, es = [], [], []
        for b in betas:
            rs = [r for r in sw if r["gamma"] == gamma and r["beta"] == b]
            l2 = float(np.mean([r["lambda2"] for r in rs]))
            if want_conv:
                vals = [r["t_conv"] for r in rs if r["converged"]]
                if not vals:
                    continue
                m, e = float(np.median(vals)), 0.0
            else:
                m, e = _mci([r[key] for r in rs])
            xs.append(l2); ys.append(m); es.append(e)
        return np.array(xs), np.array(ys), np.array(es)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    # accuracy curve (continuous) at GAMMA_HARD
    x, y, e = at(p["GAMMA_HARD"], "converged")
    a0.fill_between(x, np.clip(y - e, 0, 1), np.clip(y + e, 0, 1), color=BLUE, alpha=0.2)
    a0.plot(x, y, "-o", color=BLUE, lw=2.0, ms=4, label="community sweep (continuous)")
    for name in TOPOS:                                       # overlay named topologies
        zr = [r for r in rows if r["kind"] == "zollman"
              and r["gamma"] == p["GAMMA_HARD"] and r["topo"] == name]
        l2 = zr[0]["lambda2"]; P = np.mean([r["converged"] for r in zr])
        a0.scatter([l2], [P], s=90, color=TOPO_COLOR[name], edgecolor="black",
                   linewidth=0.6, zorder=5)
        a0.annotate(name, (l2, P), fontsize=7, xytext=(3, -10),
                    textcoords="offset points")
    a0.set_xscale("symlog", linthresh=0.05)
    a0.set_xlabel(r"algebraic connectivity $\lambda_2$  (swept via bridge density)")
    a0.set_ylabel("P(majority reaches the better paradigm)")
    a0.set_ylim(-0.05, 1.08)
    a0.set_title(f"ACCURACY ($\\gamma$={p['GAMMA_HARD']}): a continuous "
                 "diffusion threshold", fontsize=9)
    a0.legend(fontsize=8, loc="lower right")
    # speed curve (continuous) at GAMMA_EASY
    x, y, _ = at(p["GAMMA_EASY"], "t_conv", want_conv=True)
    a1.plot(x, y, "-s", color=GREEN, lw=2.0, ms=4, label="community sweep (continuous)")
    for name in TOPOS:
        zr = [r for r in rows if r["kind"] == "zollman"
              and r["gamma"] == p["GAMMA_EASY"] and r["topo"] == name and r["converged"]]
        if zr:
            l2 = zr[0]["lambda2"]; t = np.median([r["t_conv"] for r in zr])
            a1.scatter([l2], [t], s=90, color=TOPO_COLOR[name], edgecolor="black",
                       linewidth=0.6, zorder=5)
            a1.annotate(name, (l2, t), fontsize=7, xytext=(3, 4),
                        textcoords="offset points")
    a1.set_xscale("symlog", linthresh=0.05)
    a1.set_xlabel(r"algebraic connectivity $\lambda_2$  (swept via bridge density)")
    a1.set_ylabel("median time to consensus (steps)")
    a1.set_title(f"SPEED ($\\gamma$={p['GAMMA_EASY']}): a continuous decay", fontsize=9)
    a1.legend(fontsize=8, loc="upper right")
    fig.suptitle("continuous connectivity sweep (one graph family): the smooth "
                 "transition the four named topologies sample", fontsize=11)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close(fig)


def _fig_scatter(rows, p, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for gamma, marker, lab in ((p["GAMMA_HARD"], "o", f"hard ($\\gamma$={p['GAMMA_HARD']})"),
                               (p["GAMMA_EASY"], "s", f"easy ($\\gamma$={p['GAMMA_EASY']})")):
        for name in TOPOS:
            rs = [r for r in rows if r["kind"] == "zollman" and r["gamma"] == gamma
                  and r["topo"] == name]
            P, Pci = _mci([r["converged"] for r in rs])
            conv = [r["t_conv"] for r in rs if r["converged"]]
            t = np.median(conv) if conv else p["T"]
            ax.scatter([t], [P], s=130, marker=marker, color=TOPO_COLOR[name],
                       edgecolor="black", linewidth=0.6, zorder=3)
            ax.annotate(name, (t, P), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
    ax.set_xlabel("time to consensus (median steps; censored = horizon)")
    ax.set_ylabel("P(reaches the better paradigm)")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("speed/accuracy scatter (Zollman Fig 5 idiom):\n"
                 "no Pareto frontier -- circles = accuracy threshold, "
                 "squares = pure-speed line at P=1", fontsize=10)
    # legend for marker meaning
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="gray",
                              label=f"hard (gamma={p['GAMMA_HARD']})"),
                       Line2D([], [], marker="s", ls="", color="gray",
                              label=f"easy (gamma={p['GAMMA_EASY']})")],
              fontsize=8, loc="center left")
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close(fig)


def _fig_heatmap(rows, p, out_path):
    betas, gammas = p["H_BETAS"], p["H_GAMMAS"]
    grid = np.full((len(gammas), len(betas)), np.nan)
    for i, g in enumerate(gammas):
        for j, b in enumerate(betas):
            rs = [r["q_end"] for r in rows if r["kind"] == "heat"
                  and abs(r["beta"] - b) < 1e-9 and abs(r["gamma"] - g) < 1e-9]
            if rs:
                grid[i, j] = np.mean(rs)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    im = ax.imshow(grid, origin="lower", cmap="inferno", vmin=0, vmax=1,
                   aspect="auto",
                   extent=[-0.5, len(betas) - 0.5, -0.5, len(gammas) - 0.5])
    ax.set_xticks(range(len(betas))); ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.set_yticks(range(len(gammas))); ax.set_yticklabels([f"{g:g}" for g in gammas])
    ax.set_xlabel(r"bridge density $\beta$")
    ax.set_ylabel(r"conservative self-censorship $\gamma$")
    if np.isfinite(grid).all() and grid.min() < 0.5 < grid.max():
        ax.contour(grid, levels=[0.5], colors="cyan", linewidths=1.6,
                   extent=[-0.5, len(betas) - 0.5, -0.5, len(gammas) - 0.5])
    fig.colorbar(im, ax=ax, label="conservatives' final q(oxygen)")
    ax.set_title("conversion phase diagram: converts / immunizes / never reaches\n"
                 "the rival paradigm (cyan = 50% boundary; valley = immunization)",
                 fontsize=10)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close(fig)


def _traj_lambda2(tr, name):
    return [r for r in tr if r["topo"] == name][0]["lambda2"]


def _fig_trajectories(rows, p, out_path):
    """The Kuhn transition as TRAJECTORIES, all four wirings on ONE canvas: bold
    conservative-block mean +- seed band per topology, thin per-agent conservative
    lines (one representative seed) in the same colour -- the same world flip shows
    up as a synchronized jump (complete), a hub broadcast (wheel), a bridge
    staircase (community), and a travelling wave (cycle). Graph glyphs above the
    canvas are the legend (vanguard nodes outlined black)."""
    import networkx as nx
    from src.structural import graphs
    tr = [r for r in rows if r["kind"] == "traj"]
    n, nv, ts = p["N"], p["N_VAN"], p["T_SHIFT"]
    fig = plt.figure(figsize=(11.4, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 3.0], hspace=0.32)
    for j, name in enumerate(TOPOS):                       # the legend IS the graphs
        ax = fig.add_subplot(gs[0, j])
        g = _graph(name, n, nv, p["Z_BETA"])
        G = nx.from_numpy_array(np.asarray(g.A))
        lay = (nx.circular_layout(G) if name != "wheel"
               else nx.spring_layout(G, seed=1))
        edge = ["black" if i < nv else "none" for i in range(n)]
        nx.draw(G, lay, ax=ax, node_size=34, node_color=TOPO_COLOR[name],
                edgecolors=edge, linewidths=1.0, edge_color="#b9b9b9", width=0.8)
        ax.set_title(f"{name}  ($\\lambda_2$={_traj_lambda2(tr, name):.2f})",
                     fontsize=9, color=TOPO_COLOR[name])
    ax = fig.add_subplot(gs[1, :])
    for name in TOPOS:
        rs = sorted((r for r in tr if r["topo"] == name), key=lambda r: r["seed"])
        q = np.stack([r["q"] for r in rs])                 # (S, T, N)
        cons = q[:, :, nv:]                                # conservatives only
        m = cons.mean(axis=(0, 2))                         # (T,)
        s = cons.mean(axis=2).std(axis=0)                  # seed band
        for a in range(cons.shape[2]):                     # one seed's agent fan
            ax.plot(cons[0, :, a], color=TOPO_COLOR[name], lw=0.6, alpha=0.16)
        ax.fill_between(np.arange(len(m)), m - s, m + s,
                        color=TOPO_COLOR[name], alpha=0.18)
        ax.plot(m, color=TOPO_COLOR[name], lw=2.4, label=name)
    ax.axvline(ts, color=GREY, ls=":", lw=1.4)
    ax.text(ts + 4, 0.55, "world\nturns", color=GREY, fontsize=8)
    ax.axhline(p["Q_CONS"], color=GREY, ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("step")
    ax.set_ylabel("conservatives' q(oxygen)")
    ax.set_ylim(-0.04, 1.06)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("the same Kuhnian transition, four wirings, four shapes: "
                 "synchronized flip / hub broadcast /\nbridge staircase / travelling "
                 f"wave  (bold: mean of {len(tr) // len(TOPOS)} seeds; thin: one "
                 f"seed's {n - nv} conservatives; $\\gamma$={p['GAMMA_EASY']})",
                 fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.09)
    plt.savefig(out_path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------
# The experiment.
# ----------------------------------------------------------------------

def _job_key(j):
    """Stable identity of a job/row for the resume checkpoint."""
    return (j["kind"], j.get("topo"), j.get("beta"), j.get("gamma"), int(j["seed"]))


def _load_checkpoint(path):
    """Rows already computed by a previous (crashed or interrupted) run: an
    append-only pickle stream; a truncated tail record is silently dropped."""
    rows = []
    if path.exists():
        with open(path, "rb") as fh:
            while True:
                try:
                    rows.append(pickle.load(fh))
                except EOFError:
                    break
                except Exception:
                    break                       # torn tail write from a crash
    return rows


def run(out_dir, params: dict) -> None:
    p = dict(params)
    R, Rh = p["R"], p["R_HEAT"]
    jobs = []
    # per-agent TRAJECTORIES first: the paper-facing jobs land in the first
    # minutes of the run, not the last
    Rt = p["R_TRAJ"]
    for topo in TOPOS:
        for s in range(Rt):
            jobs.append(dict(p, kind="traj", topo=topo,
                             gamma=float(p["GAMMA_EASY"]), seed=s))
    for gamma in (p["GAMMA_HARD"], p["GAMMA_EASY"]):
        for topo in TOPOS:
            for s in range(R):
                jobs.append(dict(p, kind="zollman", topo=topo, gamma=float(gamma),
                                 seed=s))
    # CONTINUOUS connectivity sweep: one graph FAMILY (community bridge-density beta)
    # swept finely so lambda_2 varies continuously -- the smooth-curve view of the
    # discrete named topologies. Both gamma regimes (accuracy at HARD, speed at EASY).
    Rs = p["R_SWEEP"]
    for gamma in (p["GAMMA_HARD"], p["GAMMA_EASY"]):
        for b in p["SWEEP_BETAS"]:
            for s in range(Rs):
                jobs.append(dict(p, kind="sweep", beta=float(b), gamma=float(gamma),
                                 seed=s))
    for g in p["H_GAMMAS"]:
        for b in p["H_BETAS"]:
            for s in range(Rh):
                jobs.append(dict(p, kind="heat", beta=float(b), gamma=float(g),
                                 seed=s))
    print(f"network epistemology: {len(TOPOS)} topologies x R={Rt} (trajectories) "
          f"+ {len(TOPOS)} topologies x 2 gamma x R={R} (Zollman) + "
          f"{len(p['SWEEP_BETAS'])} beta x 2 gamma x R={Rs} (sweep) + "
          f"{len(p['H_GAMMAS'])}x{len(p['H_BETAS'])} cells x R={Rh} (heatmap) = "
          f"{len(jobs)} runs on {p['MAX_WORKERS']} workers")
    # resumable: every completed job is appended to an on-disk checkpoint the
    # moment it finishes (a worker OOM previously threw away 2000 finished jobs);
    # a rerun skips everything already in the checkpoint.
    ckpt = out_dir / "_rows.ckpt.pkl"
    rows = _load_checkpoint(ckpt)
    done_keys = {_job_key(r) for r in rows}
    todo = [j for j in jobs if _job_key(j) not in done_keys]
    if rows:
        print(f"  resume: {len(rows)} rows from checkpoint, {len(todo)} to run")
    # max_tasks_per_child bounds per-worker memory creep (JAX cache growth OOM'd a
    # 6-worker run at ~2000/2720 jobs); each restart costs one world-cache recompile.
    with ProcessPoolExecutor(max_workers=p["MAX_WORKERS"],
                             max_tasks_per_child=500) as pool, \
            open(ckpt, "ab") as fh:
        futs = [pool.submit(_one_job, j) for j in todo]
        done = len(rows)
        for fut in as_completed(futs):
            r = fut.result()
            pickle.dump(r, fh); fh.flush()
            rows.append(r); done += 1
            if done % 400 == 0:
                print(f"  ... {done}/{len(jobs)}", flush=True)

    assert all(r["finite"] for r in rows), "health: all runs must be finite"

    # ---- Zollman read-outs ----
    def acc(gamma, topo):
        return _mci([r["converged"] for r in rows if r["kind"] == "zollman"
                     and r["gamma"] == gamma and r["topo"] == topo])

    def spd(gamma, topo):
        c = [r["t_conv"] for r in rows if r["kind"] == "zollman"
             and r["gamma"] == gamma and r["topo"] == topo and r["converged"]]
        return (float(np.median(c)) if c else float("nan"))

    gh, ge = p["GAMMA_HARD"], p["GAMMA_EASY"]
    print(f"\nACCURACY regime (gamma={gh}): P(converge) by topology:")
    for topo in TOPOS:
        m, ci = acc(gh, topo)
        print(f"  {topo:>10}: P = {m:.3f} +/- {ci:.3f}")
    print(f"SPEED regime (gamma={ge}): median time-to-consensus by topology:")
    for topo in TOPOS:
        print(f"  {topo:>10}: P={acc(ge, topo)[0]:.2f}  t = {spd(ge, topo):.0f}")

    acc_dense = min(acc(gh, t)[0] for t in ("complete", "wheel"))
    acc_sparse = max(acc(gh, t)[0] for t in ("cycle", "community"))
    spd_complete, spd_cycle = spd(ge, "complete"), spd(ge, "cycle")
    print(f"\ndiffusion accuracy threshold: dense(min)={acc_dense:.2f} vs "
          f"sparse(max)={acc_sparse:.2f}")
    print(f"speed ordering (easy): complete={spd_complete:.0f} < cycle={spd_cycle:.0f}")
    print(f"wheel hub-shortcut: wheel converts at gamma={gh} "
          f"(P={acc(gh, 'wheel')[0]:.2f}) despite lambda_2 below community")

    # ---- trajectory read-outs: per-conservative flip times, by topology ----
    nv, T = p["N_VAN"], p["T"]
    traj = {name: np.stack([r["q"] for r in sorted(
                (r for r in rows if r["kind"] == "traj" and r["topo"] == name),
                key=lambda r: r["seed"])]) for name in TOPOS}     # (S, T, N)
    flips = {}                                  # per-conservative first crossing
    for name, q in traj.items():
        hit = q[:, :, nv:] > p["Q_CONS"]                          # (S, T, Nc)
        t_f = np.where(hit.any(axis=1), hit.argmax(axis=1), T)    # (S, Nc)
        flips[name] = t_f
    print(f"\ntrajectory shapes (gamma={ge}, per-conservative flip times):")
    for name in TOPOS:
        t_f = flips[name]
        print(f"  {name:>10}: median t = {np.median(t_f):.0f}, "
              f"stagger (IQR across agents) = "
              f"{np.median(np.percentile(t_f, 75, axis=1) - np.percentile(t_f, 25, axis=1)):.0f} steps")
    stagger = {name: float(np.median(np.percentile(flips[name], 75, axis=1)
                                     - np.percentile(flips[name], 25, axis=1)))
               for name in TOPOS}

    # ---- assertions (print-first) ----
    assert acc_dense > 0.8 and acc_sparse < 0.5, \
        f"z1: accuracy must order by topology at gamma={gh} ({acc_dense} vs {acc_sparse})"
    assert acc(gh, "wheel")[0] > 0.8, \
        "z2: the wheel must convert (vanguard occupies the hub -- royal-family broadcast)"
    assert spd_complete < spd_cycle, \
        f"z3: at gamma={ge} dense must be faster ({spd_complete} < {spd_cycle})"
    assert all(acc(ge, t)[0] > 0.8 for t in TOPOS), \
        "z4: the trickle regime must converge on every topology (no accuracy tradeoff)"
    assert all(traj[t][:, -1, nv:].mean() > p["Q_CONS"] for t in TOPOS), \
        "t1: every topology's conservatives must finish converted in the traj runs"
    assert stagger["cycle"] > stagger["complete"], \
        (f"t2: the cycle's wave must be more staggered than the complete graph's "
         f"synchronized flip ({stagger['cycle']:.0f} vs {stagger['complete']:.0f})")
    assert np.median(flips["complete"]) < np.median(flips["cycle"]), \
        "t3: the complete graph's conservatives must flip before the cycle's"
    # heatmap regression tie-back to kuhnian_transition's known points
    heat = {(round(r["beta"], 4), r["gamma"]): [] for r in rows if r["kind"] == "heat"}
    for r in rows:
        if r["kind"] == "heat":
            heat[(round(r["beta"], 4), r["gamma"])].append(r["q_end"])
    grid_min = min(np.mean(v) for v in heat.values())
    grid_max = max(np.mean(v) for v in heat.values())
    assert grid_min < 0.3 and grid_max > 0.7, \
        f"h1: the phase diagram must span converted..blocked ({grid_min:.2f}..{grid_max:.2f})"

    # ---- figures ----
    _fig_topologies(p, out_dir / "fig_topologies.png")
    _fig_curves(rows, p, out_dir / "fig_zollman_curves.png")
    _fig_sweep(rows, p, out_dir / "fig_connectivity_sweep.png")
    _fig_scatter(rows, p, out_dir / "fig_speed_accuracy.png")
    _fig_heatmap(rows, p, out_dir / "fig_phase_diagram.png")
    _fig_trajectories(rows, p, out_dir / "fig_kuhn_trajectories_by_topology.png")

    # ---- save ----
    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        topos=np.array(TOPOS),
        lambda2=np.array([[r for r in rows if r["kind"] == "zollman"
                           and r["topo"] == t][0]["lambda2"] for t in TOPOS]),
        acc_hard=np.array([acc(gh, t)[0] for t in TOPOS]),
        acc_easy=np.array([acc(ge, t)[0] for t in TOPOS]),
        speed_easy=np.array([spd(ge, t) for t in TOPOS]),
        h_betas=np.array(p["H_BETAS"], dtype=float),
        h_gammas=np.array(p["H_GAMMAS"], dtype=float),
        heat_grid=np.array([[np.mean(heat[(round(b, 4), g)])
                             for b in p["H_BETAS"]] for g in p["H_GAMMAS"]]),
        sweep_betas=np.array(sorted({r["beta"] for r in rows
                                     if r["kind"] == "sweep"}), dtype=float),
        sweep_lambda2=np.array([np.mean([r["lambda2"] for r in rows
                                         if r["kind"] == "sweep" and r["beta"] == b])
                                for b in sorted({r["beta"] for r in rows
                                                 if r["kind"] == "sweep"})]),
        sweep_acc_hard=np.array([np.mean([r["converged"] for r in rows
                                          if r["kind"] == "sweep"
                                          and r["gamma"] == gh and r["beta"] == b])
                                 for b in sorted({r["beta"] for r in rows
                                                  if r["kind"] == "sweep"})]),
        traj_mean=np.stack([traj[t][:, :, nv:].mean(axis=(0, 2))
                            for t in TOPOS]),               # (4, T)
        traj_band=np.stack([traj[t][:, :, nv:].mean(axis=2).std(axis=0)
                            for t in TOPOS]),               # (4, T)
        traj_agents_seed0=np.stack([traj[t][0] for t in TOPOS]),   # (4, T, N)
        traj_flip_median=np.array([np.median(flips[t]) for t in TOPOS]),
        traj_flip_stagger=np.array([stagger[t] for t in TOPOS]),
    )
    summary = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
        "accuracy_hard": {t: acc(gh, t)[0] for t in TOPOS},
        "speed_easy": {t: spd(ge, t) for t in TOPOS},
        "trajectories": {t: {"flip_median": float(np.median(flips[t])),
                             "flip_stagger_iqr": stagger[t]} for t in TOPOS},
        "trajectory_finding": "the same transition SHOWS UP differently by wiring: "
                              "synchronized flip (complete), hub broadcast (wheel), "
                              "bridge staircase (community), travelling wave (cycle) "
                              "-- read from per-conservative flip-time medians and "
                              "their stagger (IQR across agents)",
        "zollman_verdict": "tradeoff DISSOLVED: discovery topology-free (homogeneous "
                           "probe flat), diffusion helps both speed and accuracy; "
                           "the wheel converts on the hub-broadcast despite low lambda_2",
        "finding": "the network-epistemology speed/accuracy tradeoff is a property of "
                   "the consensus-only representation (herding under premature pooling). "
                   "With represented rivals the tradeoff is absent: correct prequential "
                   "evidence cannot herd, so sparsity buys no accuracy; and when truth "
                   "must DIFFUSE from a vanguard, connectivity (and hub centrality) help "
                   "both axes monotonically -- a contagion regime, not a discovery one.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    ckpt.unlink(missing_ok=True)               # run complete: checkpoint spent
    print(f"\nsaved 6 figures / arrays / summary to {out_dir}")
    print(f"HEADLINE: run on the field's own Zollman figures, the represented-rival "
          f"architecture DISSOLVES the speed/accuracy tradeoff. Independent discovery "
          f"is topology-free (no herding to protect against); when a 4/24 vanguard must "
          f"spread the better paradigm, connectivity helps BOTH accuracy (dense converts "
          f"P>={acc_dense:.0%}, sparse P<={acc_sparse:.0%}) and speed (complete "
          f"{spd_complete:.0f} < cycle {spd_cycle:.0f} steps) -- a diffusion regime, not "
          f"Zollman's discovery one. The wheel makes the lesson: it converts on its hub "
          f"broadcaster despite low connectivity -- centrality beats density.")


register(ExperimentSpec(
    model="phlogiston",
    name="network_epistemology",
    description="The canonical Zollman figures (accuracy & speed vs connectivity, the "
                "speed/accuracy scatter, the reference topologies) on the rival "
                "engine -- showing the speed/accuracy TRADEOFF dissolves: discovery is "
                "topology-free, diffusion helps both axes, and hub centrality beats "
                "density. Plus the beta x gamma conversion phase diagram.",
    run=run,
    out_dir="network_epistemology",
    params=dict(
        N=24, N_VAN=4, T_SHIFT=80, T=600, SIGMA_O=1.0, OMEGA=0.9, ALPHA_M=0.02,
        # prior precision scale on BOTH candidate nets (was a hardcoded literal in
        # _world, invisible to the manifest -- promoted so the record is complete)
        CONVICTION=2.0,
        Q_CONS=0.9,
        # CALIB (probe): gamma=1.0 zero-source => accuracy threshold (complete/wheel
        # convert, cycle/community never); gamma=0.99 trickle => all convert, speed
        # orders by lambda_2. community Zollman point at beta=0.06 (lambda_2 ~ 1.8,
        # interior between wheel and complete).
        GAMMA_HARD=1.0, GAMMA_EASY=0.99, Z_BETA=0.06,
        # R=100 is publication-grade (Weisberg-Muldoon used 100); the effect is so
        # clean (P=1.00 vs 0.00) that the curves are crisp well below this. A per-
        # worker world cache (_world) compiles the scan ONCE per worker, so the run
        # is ~10 min not ~100. An R=200 backup runs in parallel to a separate dir.
        R=100, R_HEAT=30, R_SWEEP=40, R_TRAJ=50,
        # the CONTINUOUS connectivity sweep: dense in the transition zone (~beta
        # 0.03-0.12 from the probe) so the smooth threshold is well-resolved.
        SWEEP_BETAS=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12,
                     0.15, 0.20, 0.25, 0.30),
        H_BETAS=(0.0, 0.01, 0.03, 0.06, 0.12, 0.25),
        H_GAMMAS=(0.9, 0.97, 0.99, 1.0),
        # 4 workers: each holds ~2.5-3 GB (JAX caches); 6 OOM'd a 40 GB machine
        # that had ~20 GB otherwise in use.
        MAX_WORKERS=4,
    ),
    seeds=tuple(range(200)),
    consumes=dict(figures=["fig_topologies", "fig_zollman_curves",
                           "fig_connectivity_sweep", "fig_speed_accuracy",
                           "fig_phase_diagram", "fig_kuhn_trajectories_by_topology"]),
))
