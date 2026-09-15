"""The kuhn_phlogiston sweep as a registered experiment: six panels, six figures.

Formerly the standalone ``scripts/sweep_kuhn/`` package (space.py / run_sweep.py /
analyze.py), consolidated here so the two paper figures it feeds
(``figB_lockin_boundary`` -> paper ``fig_kuhn_lockin_boundary``;
``figF_fusion`` -> paper ``fig_kuhn_fusion``) run through the registry and get a
provenance ``manifest.json`` like every other experiment.

A *job* is a flat dict with every axis set (swept axes vary, the rest sit at
``DEFAULTS`` = the validated kuhn_phlogiston anchor) plus ``panel`` and ``seed``:

* **A_vanguard** -- how many Lavoisiers does it take? Vanguard fraction x social
  coupling -> whole-population conversion: the critical-mass curve.
* **B_lockin** -- the lock-in boundary, finely: the grid concentrates on VERY weak
  coupling; axes gate scaling and prune threshold ask whether any amount of
  conviction can shift the critical coupling.
* **C_waiting** -- discovery as a waiting time (the 1/lambda law at population scale).
* **D_robust** -- does the cycle survive the dials? Forgetting x noise, including the
  omega=1.0 memory wall.
* **E_hawkes** -- the Hawkes rescue at sweep scale (excitation beta x base rate).
* **F_fusion** -- the pooling rule as an axis: posterior vs dimension-aware masked
  fusion x proposal rate.

Resumable: every completed run is one line in ``results/kuhn_sweep/rows.jsonl``;
re-running skips completed config hashes, so an interrupted sweep resumes at will.
The job dicts (and therefore the hashes) are byte-identical to the original
``scripts/sweep_kuhn`` jobs, so the existing overnight cache stays valid. Panels
E/F carry their extra keys inside their own job dicts rather than in DEFAULTS for
the same reason.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register
from src.repro.paths import RESULTS

# Workers write per-run trajectory bundles here (module-level so the function is
# picklable for ProcessPoolExecutor; must equal the spec's out_dir).
OUT = RESULTS / "kuhn_sweep"

DEFAULTS = dict(
    N=80, inter=0.0, intra=0.4, frac_open=0.5,
    lam_open=0.10, lam_dogma=0.35,
    gate_strength=1.0, s_open=0.3, s_dogma=3.0,
    omega=0.97, sigma_o=0.5, t_shift=40, n_steps=180,
    proposal_rate=0.08, snapshot_every=4,
)

A_FRACS = (0.05, 0.1, 0.2, 0.35, 0.5)
A_INTERS = (0.0, 0.002, 0.005, 0.01, 0.05, 0.2)

B_INTERS = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
B_SDOGMA = (1.0, 3.0, 6.0, 10.0)
B_LAM = (0.35, 0.8)

C_RATES = (0.01, 0.02, 0.04, 0.08, 0.16, None)        # None = deterministic attempts

D_OMEGA = (0.9, 0.95, 0.97, 0.99, 1.0)
D_SIGMA = (0.25, 0.5, 1.0)

E_BETAS = (0.0, 0.25, 0.5, 1.0, 1.5)
E_R0S = (0.005, 0.01, 0.02)          # the staggered regime (no completion without excitation)
E_STEPS = 320                        # consolidation horizon (see experiments/hawkes_rescue.py)

F_MODES = ("posterior", "posterior_masked")
F_RATES = (0.005, 0.01, 0.02)


def _job(defaults, panel, seed, **over):
    cfg = dict(defaults)
    cfg.update(over)
    cfg["panel"], cfg["seed"] = panel, seed
    return cfg


def all_jobs(p: dict) -> list[dict]:
    """Every job of the sweep, built from the registered params dict (the manifest
    record is the authoritative axis definition). Job dicts are hash-compatible
    with the original scripts/sweep_kuhn cache: same keys, same literal values."""
    d = p["DEFAULTS"]
    jobs = []
    for s in range(p["SEEDS_A"]):
        for f in p["A_FRACS"]:
            for it in p["A_INTERS"]:
                jobs.append(_job(d, "A_vanguard", s, frac_open=f, inter=it))
    for s in range(p["SEEDS_B"]):
        for it in p["B_INTERS"]:
            for sd in p["B_SDOGMA"]:
                for lm in p["B_LAM"]:
                    jobs.append(_job(d, "B_lockin", s, inter=it, s_dogma=sd, lam_dogma=lm))
    for s in range(p["SEEDS_C"]):
        for rate in p["C_RATES"]:
            jobs.append(_job(d, "C_waiting", s, proposal_rate=rate))
    for s in range(p["SEEDS_D"]):
        for om in p["D_OMEGA"]:
            for sg in p["D_SIGMA"]:
                jobs.append(_job(d, "D_robust", s, omega=om, sigma_o=sg))
    for s in range(p["SEEDS_E"]):
        for beta in p["E_BETAS"]:
            for r0 in p["E_R0S"]:
                jobs.append(_job(d, "E_hawkes", s, proposal_rate=r0, n_steps=p["E_STEPS"],
                                 rate_mode=("hawkes" if beta > 0 else "poisson"),
                                 hawkes_beta=beta, hawkes_tau=10.0))
    for s in range(p["SEEDS_F"]):
        for fm in p["F_MODES"]:
            for rate in p["F_RATES"]:
                jobs.append(_job(d, "F_fusion", s, proposal_rate=rate, n_steps=p["E_STEPS"],
                                 fuse_mode=fm))
    return jobs


def cfg_hash(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def run_one_job(cfg: dict) -> dict:
    """Execute one config in a worker process and return the flat metric row."""
    import numpy as np
    from src.structural.models.kuhn_phlogiston import single_run

    drop = {"panel", "seed"}
    kw = {k: v for k, v in cfg.items() if k not in drop}
    r = single_run(seed=int(cfg["seed"]), **kw)

    comm = r["community"]
    cs, es = r["crisis_step"], r["expand_step"]
    oxy = r["oxy_index_sc"]

    def med(x):
        x = x[x >= 0]
        return int(np.median(x)) if len(x) else -1

    row = {}
    for label, c in (("open", 0), ("dogma", 1)):
        m = comm == c
        row[f"{label}_n"] = int(m.sum())
        row[f"{label}_crisis_frac"] = float((cs[m] >= 0).mean())
        row[f"{label}_crisis_med"] = med(cs[m])
        row[f"{label}_expand_frac"] = float((es[m] >= 0).mean())
        row[f"{label}_expand_med"] = med(es[m])
        row[f"{label}_oxy_end"] = float(oxy[-1, c])
        row[f"{label}_struct_frac"] = float((r["oxy_coupling_tn"][-1][m] > 0.02).mean())
    pop_oxy = (oxy[-1, 0] * row["open_n"] + oxy[-1, 1] * row["dogma_n"]) / comm.size
    row["pop_oxy_end"] = float(pop_oxy)
    row["pop_expand_frac"] = float((es >= 0).mean())
    ok = (cs >= 0) & (es >= 0)
    row["mean_expand_delay"] = float(np.mean(es[ok] - cs[ok])) if ok.any() else -1.0
    row["first_expand"] = int(es[es >= 0].min()) if (es >= 0).any() else -1
    row["gamma_final"] = float(r["gamma_t"][-1])
    row["n_pruned_med"] = int(np.median(r["n_pruned"][cs >= 0])) if (cs >= 0).any() else 0
    # NaN in dF_belt_tn is the INTENTIONAL mask on expanded agents; only inf / NaN in the
    # belief read-outs signals a genuine numerical blowup.
    row["finite"] = bool(np.isfinite(oxy).all() and not np.isinf(r["dF_belt_tn"]).any()
                         and np.isfinite(r["oxy_coupling_tn"]).all())

    # small per-run trajectory bundle for post-hoc analysis
    OUT.joinpath("traj").mkdir(parents=True, exist_ok=True)
    with np.errstate(all="ignore"):                      # expanded agents are NaN-masked
        dF_open = np.nanmean(r["dF_belt_tn"][:, comm == 0], axis=1)
        dF_dogma = np.nanmean(r["dF_belt_tn"][:, comm == 1], axis=1)
    np.savez_compressed(
        OUT / "traj" / f"{cfg_hash(cfg)}.npz",
        snap_t=r["snap_t"].astype(np.int32),
        oxy_index_sc=oxy.astype(np.float32),
        dF_open_t=dF_open.astype(np.float32),
        dF_dogma_t=dF_dogma.astype(np.float32),
        crisis_step=cs.astype(np.int32), expand_step=es.astype(np.int32),
    )
    return row


# ----------------------------------------------------------------------
# Analysis: one figure per panel (consumes rows.jsonl only).
# ----------------------------------------------------------------------

def load_rows(out_dir) -> list[dict]:
    rows = []
    with (out_dir / "rows.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("error", "") == "" and r.get("finite", True):
                    rows.append(r)
    return rows


def _sel(rows, **kv):
    return [r for r in rows if all(r.get(k) == v for k, v in kv.items())]


def _mci(vals):
    a = np.asarray(vals, dtype=np.float64)
    if len(a) < 2:
        return float(a.mean()) if len(a) else np.nan, 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def fig_a(rows, path) -> dict:
    A = _sel(rows, panel="A_vanguard")
    if not A:
        return {}
    fracs = sorted({r["frac_open"] for r in A})
    inters = sorted({r["inter"] for r in A})
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.8, 4.0))
    out = {}
    cmap = plt.cm.viridis
    for ii, it in enumerate(inters):
        x, y, e = [], [], []
        for f in fracs:
            m, ci = _mci([r["pop_oxy_end"] for r in _sel(A, frac_open=f, inter=it)])
            x.append(f); y.append(m); e.append(ci)
        a0.errorbar(x, y, yerr=e, marker="o", lw=1.8, capsize=2,
                    color=cmap(ii / max(len(inters) - 1, 1)), label=f"inter={it}")
        out[str(it)] = dict(frac=x, pop_oxy=y)
    a0.set_xlabel("vanguard fraction (how many Lavoisiers)")
    a0.set_ylabel("population oxygen index at end")
    a0.set_title("a small connected vanguard converts the population")
    a0.legend(fontsize=7); a0.set_ylim(0, 1.04)
    for ii, it in enumerate(inters):
        x, y, e = [], [], []
        for f in fracs:
            m, ci = _mci([r["dogma_crisis_frac"] for r in _sel(A, frac_open=f, inter=it)])
            x.append(f); y.append(m); e.append(ci)
        a1.errorbar(x, y, yerr=e, marker="o", lw=1.8, capsize=2,
                    color=cmap(ii / max(len(inters) - 1, 1)))
    a1.set_xlabel("vanguard fraction")
    a1.set_ylabel("majority reaching CRISIS")
    a1.set_title("...because the vanguard's evidence, fused across\nthe graph, drives the majority into its own crisis")
    a1.set_ylim(-0.04, 1.04)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return out


def fig_b(rows, path) -> dict:
    B = _sel(rows, panel="B_lockin")
    if not B:
        return {}
    inters = sorted({r["inter"] for r in B})
    sds = sorted({r["s_dogma"] for r in B})
    lams = sorted({r["lam_dogma"] for r in B})
    fig, axs = plt.subplots(1, len(lams) + 1, figsize=(4.6 * (len(lams) + 1), 3.8),
                        squeeze=False)
    axs = axs[0]
    out = {}
    for li, lm in enumerate(lams):
        M = np.zeros((len(sds), len(inters)))
        for si, sd in enumerate(sds):
            for ii, it in enumerate(inters):
                M[si, ii] = np.mean([r["dogma_oxy_end"]
                                     for r in _sel(B, inter=it, s_dogma=sd, lam_dogma=lm)])
        im = axs[li].imshow(M, cmap="inferno", vmin=0, vmax=1, aspect="auto", origin="lower")
        axs[li].set_xticks(range(len(inters)))
        axs[li].set_xticklabels(inters, rotation=45, fontsize=7)
        axs[li].set_yticks(range(len(sds))); axs[li].set_yticklabels(sds)
        axs[li].set_xlabel("social coupling (inter)"); axs[li].set_ylabel("gate scaling $s$")
        axs[li].set_title(f"dogmatic conversion, $\\lambda_d$={lm}", fontsize=9)
        fig.colorbar(im, ax=axs[li], shrink=0.85)
        out[str(lm)] = M.tolist()
    # variance (bimodality proxy) along the boundary at the default lam
    sd0 = sds[1] if len(sds) > 1 else sds[0]
    x, v = [], []
    for it in inters:
        vals = [r["dogma_oxy_end"] for r in _sel(B, inter=it, s_dogma=sd0, lam_dogma=lams[0])]
        x.append(it); v.append(np.std(vals))
    axs[-1].plot(range(len(x)), v, "o-", lw=2.0, color="#c0392b")
    axs[-1].set_xticks(range(len(x))); axs[-1].set_xticklabels(x, rotation=45, fontsize=7)
    axs[-1].set_xlabel("social coupling (inter)")
    axs[-1].set_ylabel("run-to-run std of conversion")
    axs[-1].set_title(f"susceptibility at the boundary (s={sd0})", fontsize=9)
    fig.suptitle("the lock-in boundary sits at VERY weak coupling -- and no amount of "
                 "conviction moves it far", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.savefig(path, dpi=130); plt.close(fig)
    return out


def fig_c(rows, path) -> dict:
    C = _sel(rows, panel="C_waiting")
    if not C:
        return {}
    rates = sorted({r["proposal_rate"] for r in C if r["proposal_rate"] is not None})
    if not rates:
        return {}
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.8, 4.0))
    det_disc = [r["open_expand_frac"] for r in C if r["proposal_rate"] is None]
    det_surv = [r["open_struct_frac"] for r in C if r["proposal_rate"] is None]
    x, disc, de, surv, se = [], [], [], [], []
    for rate in rates:
        rs = _sel(C, proposal_rate=rate)
        m, ci = _mci([r["open_expand_frac"] for r in rs])
        x.append(rate); disc.append(m); de.append(ci)
        m, ci = _mci([r["open_struct_frac"] for r in rs])
        surv.append(m); se.append(ci)
    a0.errorbar(x, disc, yerr=de, marker="o", lw=2.2, capsize=3, color="#1f618d",
                label="fraction that DISCOVERS oxygen")
    xs = np.linspace(min(rates), max(rates), 200)
    T_eff = 110.0          # post-crisis, post-sustain steps inside the horizon
    a0.plot(xs, 1 - (1 - xs) ** T_eff, "--", color="k", lw=1.2,
            label=r"$1-(1-\lambda)^{T}$ (per-agent waiting time)")
    a0.errorbar(x, surv, yerr=se, marker="s", lw=2.2, capsize=3, color="#922b21",
                label="fraction whose concept SURVIVES fusion")
    if det_disc:
        a0.plot([max(rates) * 1.6], [np.mean(det_disc)], marker="o", color="#1f618d")
        a0.plot([max(rates) * 1.6], [np.mean(det_surv)], marker="s", color="#922b21")
        a0.annotate("simultaneous\n(deterministic)", (max(rates) * 1.6, np.mean(det_surv)),
                    textcoords="offset points", xytext=(-10, 12), fontsize=7, ha="right")
    a0.set_xscale("log"); a0.set_xlabel(r"proposal rate $\lambda$ (Poisson serendipity)")
    a0.set_ylabel("fraction of the open community")
    a0.set_ylim(-0.04, 1.04)
    a0.set_title("discovery follows the waiting-time law -- but a\n"
                 "STAGGERED discovery never survives the communal prior")
    a0.legend(fontsize=7)
    fe = []
    for rate in rates:
        vals = [r["first_expand"] for r in _sel(C, proposal_rate=rate) if r["first_expand"] >= 0]
        fe.append(vals)
    a1.boxplot(fe)
    a1.set_xticklabels([str(r) for r in rates])
    a1.set_xlabel(r"proposal rate $\lambda$"); a1.set_ylabel("step of FIRST discovery")
    a1.set_title("the population's FIRST discovery is cheap\n(~40 agents draw in parallel)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(rates=x, discover=disc, survive=surv,
                det_discover=float(np.mean(det_disc)) if det_disc else None,
                det_survive=float(np.mean(det_surv)) if det_surv else None)


def fig_d(rows, path) -> dict:
    D = _sel(rows, panel="D_robust")
    if not D:
        return {}
    omegas = sorted({r["omega"] for r in D})
    sigmas = sorted({r["sigma_o"] for r in D})
    fig, axs = plt.subplots(1, 1, figsize=(4.8, 3.6), squeeze=False)
    ax = axs[0][0]
    M = np.zeros((len(sigmas), len(omegas)))
    for si, sg in enumerate(sigmas):
        for oi, om in enumerate(omegas):
            rs = _sel(D, omega=om, sigma_o=sg)
            M[si, oi] = np.mean([r["open_expand_frac"] for r in rs]) if rs else np.nan
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(len(omegas))); ax.set_xticklabels(omegas)
    ax.set_yticks(range(len(sigmas))); ax.set_yticklabels(sigmas)
    ax.set_xlabel(r"forgetting $\omega$"); ax.set_ylabel(r"noise $\sigma_o$")
    ax.set_title("open community discovers oxygen", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    out = {"open_expand_frac": M.tolist()}
    fig.suptitle("robustness: where the full cycle (crisis + reduction + discovery) survives",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.9]); plt.savefig(path, dpi=130); plt.close(fig)
    return out


def fig_e(rows, path) -> dict:
    """The Hawkes rescue at sweep scale: survival vs excitation beta, per base rate."""
    E = _sel(rows, panel="E_hawkes")
    if not E:
        return {}
    betas = sorted({r["hawkes_beta"] for r in E})
    r0s = sorted({r["proposal_rate"] for r in E})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cmap = plt.cm.viridis
    out = {}
    for ri, r0 in enumerate(r0s):
        surv = [_mci([r["open_struct_frac"] for r in _sel(E, hawkes_beta=b,
                                                          proposal_rate=r0)])
                for b in betas]
        ax.errorbar(betas, [m for m, _ in surv], yerr=[c for _, c in surv], marker="o",
                    lw=2.0, capsize=3, color=cmap(ri / max(len(r0s) - 1, 1)),
                    label=f"$r_0$={r0}")
        out[str(r0)] = [m for m, _ in surv]
    ax.set_xlabel(r"Hawkes excitation $\beta$")
    ax.set_ylabel("fraction whose concept SURVIVES fusion")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("social excitation rescues the staggered discovery\n"
                 "(the trust graph manufactures near-simultaneity)")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return out


def fig_f(rows, path) -> dict:
    """The pooling rule as an axis: survival under posterior vs dimension-aware fusion."""
    F = _sel(rows, panel="F_fusion")
    if not F:
        return {}
    rates = sorted({r["proposal_rate"] for r in F})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    out = {}
    for mode, color in (("posterior", "#922b21"), ("posterior_masked", "#1e8449")):
        surv = [_mci([r["open_struct_frac"] for r in _sel(F, fuse_mode=mode,
                                                          proposal_rate=rt)])
                for rt in rates]
        disc = [_mci([r["open_expand_frac"] for r in _sel(F, fuse_mode=mode,
                                                          proposal_rate=rt)])
                for rt in rates]
        ax.errorbar(rates, [m for m, _ in surv], yerr=[c for _, c in surv], marker="s",
                    lw=2.2, capsize=3, color=color, label=f"survives | {mode}")
        ax.errorbar(rates, [m for m, _ in disc], yerr=[c for _, c in disc], marker="o",
                    lw=1.2, ls="--", capsize=3, color=color, alpha=0.6,
                    label=f"discovers | {mode}")
        out[mode] = dict(rates=list(rates), survive=[m for m, _ in surv])
    ax.set_xscale("log")
    ax.set_xlabel(r"proposal rate $\lambda$")
    ax.set_ylabel("fraction of the open community")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("the survival threshold moves with the pooling rule")
    ax.legend(fontsize=7.5)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return out


# ----------------------------------------------------------------------
# The experiment.
# ----------------------------------------------------------------------

def run(out_dir, params: dict) -> None:
    assert out_dir == OUT, f"out_dir {out_dir} must be the module OUT {OUT}"
    rows_path = out_dir / "rows.jsonl"
    jobs = all_jobs(params)

    done = set()
    if rows_path.exists():
        with rows_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        row = json.loads(line)
                        if row.get("error", "") == "":   # errored rows retried on resume
                            done.add(row["hash"])
                    except Exception:
                        pass
    todo = [(cfg_hash(c), c) for c in jobs if cfg_hash(c) not in done]
    print(f"kuhn_sweep: {len(jobs)} jobs total, {len(jobs) - len(todo)} cached, "
          f"{len(todo)} to run on {params['WORKERS']} workers", flush=True)

    if todo:
        t0 = time.time()
        n_done = n_err = 0
        with rows_path.open("a", encoding="utf-8") as fh, \
                ProcessPoolExecutor(max_workers=params["WORKERS"]) as pool:
            futures = {pool.submit(run_one_job, cfg): (h, cfg) for h, cfg in todo}
            for fut in as_completed(futures):
                h, cfg = futures[fut]
                row = {"hash": h, **cfg}
                try:
                    row.update(fut.result())
                    row["error"] = ""
                except Exception as e:              # one bad config never kills the sweep
                    row["error"] = f"{type(e).__name__}: {e}"[:300]
                    n_err += 1
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n_done += 1
                if n_done % 20 == 0:
                    rate = (time.time() - t0) / n_done
                    eta = (len(futures) - n_done) * rate / 60
                    print(f"  [{n_done}/{len(futures)}] err={n_err} | {rate:.1f}s/run "
                          f"| ETA {eta:.0f} min", flush=True)
        print(f"sweep: {n_done} runs, {n_err} errors, "
              f"{(time.time() - t0) / 60:.1f} min -> {rows_path}")

    rows = load_rows(out_dir)
    print(f"{len(rows)} healthy rows")
    info = {
        "A": fig_a(rows, out_dir / "figA_vanguard.png"),
        "B": fig_b(rows, out_dir / "figB_lockin_boundary.png"),
        "C": fig_c(rows, out_dir / "figC_waiting_time.png"),
        "D": fig_d(rows, out_dir / "figD_robustness.png"),
        "E": fig_e(rows, out_dir / "figE_hawkes.png"),
        "F": fig_f(rows, out_dir / "figF_fusion.png"),
        "n_rows": len(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    report = f"""# kuhn_phlogiston sweep

{len(rows)} healthy runs (`python scripts/run.py run kuhn_sweep`; resumable via
rows.jsonl). One figure per panel:

- **figA_vanguard.png** -- the critical-mass curve: even a 5% open vanguard lifts population
  conversion once coupled, and coupling spreads the CRISIS itself, so the dogmatic majority
  ends up making its own discovery.
- **figB_lockin_boundary.png** (paper: fig_kuhn_lockin_boundary) -- the lock-in boundary
  sits at VERY weak coupling (inter ~ 0.0005-0.002, where the run-to-run susceptibility
  peaks); neither the conviction gate nor the prune threshold moves it far. Only isolation
  protects the paradigm.
- **figC_waiting_time.png** -- per-agent discovery follows the waiting-time law
  1-(1-lambda)^T (44% at rate 0.01 -> 100% at 0.16), but the population's FIRST discovery is
  not rate-limited at all: with ~40 parallel draws an attempt arrives almost immediately, and
  the median first discovery sits at t~82 at every rate INCLUDING deterministic -- its date is
  set by when the post-reduction residual earns a positive Bayes factor, not by arrival.
  Discovery waits on evidence, not on luck. Survival is what the rate governs: **a staggered
  discovery does not survive** -- zero survival at rate <= 0.04 (the lone discoverer's oxygen
  node is crushed by precision fusion with still-pinned peers), rising to ~31% at rate 0.16,
  matching the simultaneous-proposal limit (~28%). Concepts need critical mass IN TIME.
  *Caveat*: under `posterior` fusion the unconceived slot's pin is pooled as if it were a
  strongly-held zero belief -- a modeling choice. It reads naturally as incommensurability
  (the shared conceptual scheme suppresses unshared concepts), but fusion schemes that
  exclude unconceived dimensions would soften it.
- **figD_robustness.png** -- crisis and reduction survive everywhere across forgetting x
  noise, INCLUDING the omega=1 memory wall; discovery is essentially complete for omega < 1
  and thins at the wall itself (0.55/0.43/0.16 of the open community at sigma 0.25/0.5/1.0):
  with nothing forgotten, the entrenched precision denies the woken node a positive Bayes
  factor. A community that never forgets still sees its paradigm die; it loses the capacity
  to grow the successor. (No trigger constant exists to calibrate: the Bayes factor is the
  sole expansion accept test.)
- **figE_hawkes.png** -- the Hawkes rescue at sweep scale: social excitation manufactures
  the near-simultaneity that concept survival requires.
- **figF_fusion.png** (paper: fig_kuhn_fusion) -- the survival threshold moves with the
  pooling rule: dimension-aware masked fusion shifts concept survival from 'critical mass
  in time' toward 'conceive it at all'.
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"saved 6 figures / summary / REPORT to {out_dir}")


register(ExperimentSpec(
    model="phlogiston",
    name="kuhn_sweep",
    description="The kuhn_phlogiston sweep, six panels feeding six figures: critical-mass "
                "vanguard curve, the lock-in boundary at fine coupling (paper: "
                "fig_kuhn_lockin_boundary), discovery as a waiting time, robustness over "
                "forgetting x noise, the Hawkes rescue, and the pooling rule as an axis "
                "(paper: fig_kuhn_fusion). Resumable via rows.jsonl; job hashes are "
                "backward-compatible with the original scripts/sweep_kuhn cache.",
    run=run,
    out_dir="kuhn_sweep",
    params=dict(
        DEFAULTS=DEFAULTS,
        A_FRACS=A_FRACS, A_INTERS=A_INTERS,
        B_INTERS=B_INTERS, B_SDOGMA=B_SDOGMA, B_LAM=B_LAM,
        C_RATES=C_RATES,
        D_OMEGA=D_OMEGA, D_SIGMA=D_SIGMA,
        E_BETAS=E_BETAS, E_R0S=E_R0S, E_STEPS=E_STEPS,
        F_MODES=F_MODES, F_RATES=F_RATES,
        SEEDS_A=16, SEEDS_B=16, SEEDS_C=32, SEEDS_D=8, SEEDS_E=32, SEEDS_F=32,
        WORKERS=10,
    ),
    # per-panel seed counts differ; the widest panel sweeps seeds 0..31
    seeds=tuple(range(32)),
    consumes=dict(figures=["figA_vanguard", "figB_lockin_boundary", "figC_waiting_time",
                           "figD_robustness", "figE_hawkes", "figF_fusion"]),
))
