"""The frontier sweep: three claims, three figures, ~340 resumable runs.

Every run feeds exactly one of three headline figures (no orphan phase maps):

* **Fig 1 -- pluralism needs the frontier AND disconnection.** Static world; x = social
  connectivity lambda2; y = how differently communities partition the world; two curves:
  gate on vs gate off. One sentence: communities end with different theories only when
  observation is earned and communication is sparse -- under full observability there is
  nothing left to disagree about.
* **Fig 2 -- the lock-in boundary.** Regime change; heatmap of lock-in over conviction x
  memory with the re-track/lock-in boundary drawn. One sentence: after the world changes,
  either long memory or conviction alone is enough to never see the new structure.
* **Fig 3 -- the price of consensus is foregone discovery.** A new pocket of deep (coarse-
  invisible) structure appears somewhere random; x = resolution budget; y = P(discovered).
  Disconnected populations convert extra budget into broader coverage and more discoveries;
  fully connected populations homogenize where they look, so extra lenses are redundant and
  the curve stays flat. "Did anyone already watch that region" separates discovery from
  blindness almost perfectly. One sentence: new structure is found only where someone already
  looks, and diversity -- not communication -- is what turns capacity into looking-places.

Runs are cached in ``rows.jsonl`` (hash -> skip-if-done), so the sweep is crash-safe and
re-invoking the experiment resumes instead of recomputing.
"""
from __future__ import annotations

import hashlib
import json
import time

import matplotlib.pyplot as plt
import numpy as np

from experiments.registry import ExperimentSpec, register
from src.structural.models.landscape_frontier import FrontierConfig, FrontierEnvironment

DISCOVER_BLOCK = 8.0      # target region resolved to ~atom scale
WATCH_BLOCK = 16.0        # someone held the region at sub-quarter granularity pre-change


# ----------------------------------------------------------------------
# jobs: every config maps to exactly one figure ("panel")
# ----------------------------------------------------------------------

def build_jobs(params: dict) -> list[dict]:
    jobs: list[dict] = []
    # --- panel A: pluralism vs lambda2, gate on/off (static) ---
    for gate in ("on", "off"):
        for inter in params["A_INTERS"]:
            for s in range(params["A_SEEDS"]):
                jobs.append(dict(panel="A", gate=gate, inter=float(inter), run_seed=s,
                                 n_communities=4, init_bias_depth=3, value_depth=2,
                                 budget=16, omega=0.9, n_steps=80,
                                 world_mode="static"))
    # --- panel B: lock-in over conviction x memory (regime change) ---
    for lam in params["B_LAMBDAS"]:
        for om in params["B_OMEGAS"]:
            for s in range(params["B_SEEDS"]):
                jobs.append(dict(panel="B", gate="on", inter=0.0, run_seed=s,
                                 n_communities=2, init_bias_depth=3, value_depth=2,
                                 budget=16, omega=float(om), n_steps=140,
                                 world_mode="regime_change", value_beta=5.0,
                                 value_lambda=float(lam), band_amp=3.0, t_change=40))
    # --- panel C: discovery of a new pocket vs budget x connectivity ---
    for budget in params["C_BUDGETS"]:
        for inter in (0.0, 1.0):
            for s in range(params["C_SEEDS"]):
                jobs.append(dict(panel="C", gate="on", inter=float(inter), run_seed=s,
                                 n_communities=4, init_bias_depth=4, value_depth=3,
                                 budget=int(budget), omega=0.9, n_steps=140,
                                 world_mode="pocket_appear", t_change=60))
    return jobs


def _to_cfg(job: dict) -> FrontierConfig:
    drop = {"panel", "run_seed"}
    fields = {k: v for k, v in job.items() if k not in drop}
    return FrontierConfig(init_mode="subtree_biased", seed=0, **fields)


def run_one(job: dict) -> dict:
    r = FrontierEnvironment(_to_cfg(job)).run(seed=job["run_seed"])
    row = dict(
        lambda2=float(r["lambda2"]),
        final_pdist=float(r["final_pdist"]),
        mean_pdist=float(r["mean_pdist"]),
        union_unlocked=float(r["union_unlocked_frac_t"][-1]),
    )
    if "cold_block_size_tc" in r:
        cold = r["cold_block_size_tc"]                       # (T, n_c)
        t_change = job["t_change"]
        best_post = cold[t_change:].min(axis=1)
        hit = np.nonzero(best_post < DISCOVER_BLOCK)[0]
        row.update(
            cold_end_mean=float(cold[-1].mean()),
            cold_end_min=float(cold[-1].min()),
            watched=bool(cold[t_change - 1].min() < WATCH_BLOCK),
            found=bool(len(hit) > 0),
            discovery_delay=int(hit[0]) if len(hit) else int(len(best_post)),
        )
    return row


# ----------------------------------------------------------------------
# resumable runner (hash -> skip-if-done -> append one tidy JSONL row)
# ----------------------------------------------------------------------

def _hash(job: dict) -> str:
    return hashlib.md5(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]


def run_jobs(jobs: list[dict], rows_path) -> list[dict]:
    done: dict[str, dict] = {}
    if rows_path.exists():
        with rows_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    done[row["hash"]] = row
    t0, n_ran = time.time(), 0
    out = []
    with rows_path.open("a", encoding="utf-8") as fh:
        for i, job in enumerate(jobs):
            h = _hash(job)
            if h in done:
                out.append(done[h])
                continue
            row = {"hash": h, **job, **run_one(job)}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            out.append(row)
            n_ran += 1
            if n_ran % 10 == 0:
                rate = (time.time() - t0) / n_ran
                print(f"  [{i + 1}/{len(jobs)}] ran={n_ran} | {rate:.1f}s/run | "
                      f"ETA {(len(jobs) - i - 1) * rate / 60:.1f} min", flush=True)
    return out


# ----------------------------------------------------------------------
# the three figures
# ----------------------------------------------------------------------

def _sel(rows, **kv):
    return [r for r in rows if all(r.get(k) == v for k, v in kv.items())]


def _mean_ci(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def fig1_pluralism(rows, params, path) -> dict:
    A = _sel(rows, panel="A")
    inters = sorted({r["inter"] for r in A})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    curves = {}
    for gate, color, label in (("on", "#6c3483", "frontier ON (observation must be earned)"),
                               ("off", "#999999", "frontier OFF (world fully observable)")):
        rs_gate = _sel(A, gate=gate)
        lam_vals = sorted({r["lambda2"] for r in rs_gate})     # group by REALIZED connectivity
        x, y, e = [], [], []
        for lam in lam_vals:
            rs = [r for r in rs_gate if r["lambda2"] == lam]
            m, ci = _mean_ci([r["final_pdist"] for r in rs])
            x.append(lam); y.append(m); e.append(ci)
        ax.errorbar(x, y, yerr=e, marker="o", lw=2.2, capsize=3, color=color, label=label)
        curves[gate] = dict(lambda2=x, pdist=y, ci=e)
    ax.set_xlabel(r"social connectivity $\lambda_2$")
    ax.set_ylabel("how differently communities\npartition the world (0 = identical)")
    ax.set_title("Different theories survive only when observation is earned\nAND communication is sparse")
    ax.legend(fontsize=9); ax.set_ylim(bottom=-0.01)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return curves


def fig2_lockin(rows, params, path) -> dict:
    B = _sel(rows, panel="B")
    lams = sorted({r["value_lambda"] for r in B})
    omegas = sorted({r["omega"] for r in B})
    cold = np.zeros((len(lams), len(omegas)))
    for li, lam in enumerate(lams):
        for oi, om in enumerate(omegas):
            rs = _sel(B, value_lambda=lam, omega=om)
            cold[li, oi] = np.mean([r["cold_end_mean"] for r in rs])
    lockin = np.clip((cold - DISCOVER_BLOCK) / max(cold.max() - DISCOVER_BLOCK, 1e-9), 0, 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(lockin, cmap="inferno_r", vmin=0, vmax=1, aspect="auto", origin="lower")
    X, Y = np.meshgrid(range(len(omegas)), range(len(lams)))
    cs = ax.contour(X, Y, lockin, levels=[0.5], colors="cyan", linewidths=2)
    ax.clabel(cs, fmt={0.5: "lock-in boundary"}, fontsize=8)
    ax.set_xticks(range(len(omegas))); ax.set_xticklabels(omegas)
    ax.set_yticks(range(len(lams))); ax.set_yticklabels(lams)
    ax.set_xlabel(r"memory $\omega$  (right = forgets slowly)")
    ax.set_ylabel(r"conviction $\lambda$  (up = defends valued beliefs)")
    ax.set_title("After the world changes: re-track (light)\nor never see the new structure (dark)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="lock-in (1 = new structure never resolved)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(lams=lams, omegas=omegas, lockin=lockin.tolist())


def fig3_discovery(rows, params, path) -> dict:
    C = _sel(rows, panel="C")
    budgets = sorted({r["budget"] for r in C})
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.8, 4.0), width_ratios=[2.1, 1.0])
    curves = {}
    for inter, color, label in ((0.0, "#1f618d", "disconnected ($\\lambda_2=0$)"),
                                (1.0, "#c0392b", "fully connected")):
        x, y, e = [], [], []
        for b in budgets:
            rs = _sel(C, budget=b, inter=inter)
            p = np.mean([r["found"] for r in rs])
            x.append(b); y.append(p)
            e.append(1.96 * np.sqrt(p * (1 - p) / len(rs)))
        a0.errorbar(x, y, yerr=e, marker="o", lw=2.2, capsize=3, color=color, label=label)
        curves[inter] = dict(budget=x, p_found=y, ci=e)
    a0.set_xlabel("resolution budget (size of each community's frontier)")
    a0.set_ylabel("P(new deep structure ever discovered)")
    a0.set_title("Plural populations turn extra capacity into discoveries;\nconnected ones watch the same places and plateau")
    a0.set_ylim(-0.04, 1.04); a0.legend(fontsize=9)
    watched = [r for r in C if r["watched"]]
    unwatched = [r for r in C if not r["watched"]]
    pw = np.mean([r["found"] for r in watched]) if watched else 0.0
    pu = np.mean([r["found"] for r in unwatched]) if unwatched else 0.0
    a1.bar([0, 1], [pw, pu], color=["#1e8449", "#922b21"], width=0.6)
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["someone already\nwatched the region", "nobody\nwatched it"], fontsize=8)
    a1.set_ylabel("P(discovered)"); a1.set_ylim(0, 1.04)
    a1.set_title("...and only where\nsomeone already looks", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)
    return dict(curves=curves, p_watched=float(pw), p_unwatched=float(pu),
                n_watched=len(watched), n_unwatched=len(unwatched))


REPORT_TEMPLATE = """# Frontier sweep report

{n_rows} runs, three figures, one sentence each.

## Fig 1 -- `fig1_pluralism.png`
**Communities end up with different theories only when observation must be earned AND
communication is sparse.** With the frontier on, disconnected communities settle into different
fixed-point partitions of the SAME world (distance {f1_on0:.2f}); connecting them collapses the
difference to {f1_on1:.2f}. With the frontier off (fully observable world) the distance is
{f1_off0:.2f} even with zero communication: there is nothing left to disagree about. Pluralism is
a property of HOW the world is seen, not of noise or stubbornness.

## Fig 2 -- `fig2_lockin.png`
**After the world changes, either long memory or conviction alone is enough to never see the new
structure.** The light region re-allocates its resolution budget to where the world now is; cross
the cyan boundary (remember too long, or defend valued beliefs too hard) and the community keeps
explaining the old region forever -- lock-in reaches {f2_max:.2f} in the worst corner while the
agile corner sits at {f2_min:.2f}.

## Fig 3 -- `fig3_discovery.png`
**The price of consensus is foregone discovery.** A pocket of structure invisible to coarse
aggregates appears in a random region. A DISCONNECTED population converts extra resolution budget
into broader collective coverage, so discovery rises {f3_d0:.2f} -> {f3_d1:.2f}; a FULLY CONNECTED
population homogenizes where it looks, so extra lenses are redundant and discovery stays near
{f3_c1:.2f}. The mechanism is bare in the right panel: P(discovered | someone already watched that
region) = {f3_pw:.2f} vs P(discovered | nobody watched) = {f3_pu:.2f}. New structure is found only
where someone already looks -- and diversity, not communication, is what turns capacity into
looking-places. This is Stanford's "unconceived alternatives" as a measured quantity, and the
direct cost of the consensus that Fig 1 shows connection buys.
"""


# ----------------------------------------------------------------------
# the experiment
# ----------------------------------------------------------------------

def run(out_dir, params: dict) -> None:
    jobs = build_jobs(params)
    print(f"{len(jobs)} jobs (resumable; cached rows are skipped)")
    rows = run_jobs(jobs, out_dir / "rows.jsonl")

    c1 = fig1_pluralism(rows, params, out_dir / "fig1_pluralism.png")
    c2 = fig2_lockin(rows, params, out_dir / "fig2_lockin.png")
    c3 = fig3_discovery(rows, params, out_dir / "fig3_discovery.png")

    # ---- self-checking headlines ----
    on0, on1 = c1["on"]["pdist"][0], c1["on"]["pdist"][-1]
    off0 = c1["off"]["pdist"][0]
    assert on0 > 0.15, f"fig1: gate-on disconnected should be plural (pdist {on0:.3f})"
    assert on1 < 0.5 * on0, f"fig1: connection should collapse pluralism ({on0:.3f}->{on1:.3f})"
    assert off0 < 0.5 * on0, f"fig1: full observability should kill pluralism ({off0:.3f})"
    L = np.asarray(c2["lockin"])
    assert L[0, 0] < 0.5 < L[-1, -1], f"fig2: lock-in should grow along both axes ({L[0,0]:.2f}, {L[-1,-1]:.2f})"
    assert L[-1, 0] > L[0, 0] + 0.25 and L[0, -1] > L[0, 0] + 0.25, \
        "fig2: conviction and memory should EACH produce lock-in alone"
    assert c3["p_watched"] - c3["p_unwatched"] > 0.5, \
        f"fig3: watching the region should dominate discovery ({c3['p_watched']:.2f} vs {c3['p_unwatched']:.2f})"
    p0 = np.asarray(c3["curves"][0.0]["p_found"]); p1 = np.asarray(c3["curves"][1.0]["p_found"])
    assert p0[-1] - p0[0] > 0.2, \
        f"fig3: disconnected discovery should grow with budget ({p0[0]:.2f}->{p0[-1]:.2f})"
    assert p0[-1] - p1[-1] > 0.1, \
        f"fig3: at high budget, pluralism should out-discover consensus ({p0[-1]:.2f} vs {p1[-1]:.2f})"

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        fig1_lambda2_on=np.array(c1["on"]["lambda2"]), fig1_pdist_on=np.array(c1["on"]["pdist"]),
        fig1_lambda2_off=np.array(c1["off"]["lambda2"]), fig1_pdist_off=np.array(c1["off"]["pdist"]),
        fig2_lams=np.array(c2["lams"]), fig2_omegas=np.array(c2["omegas"]),
        fig2_lockin=np.array(c2["lockin"]),
        fig3_budgets=np.array(c3["curves"][0.0]["budget"]),
        fig3_pfound_disc=np.array(c3["curves"][0.0]["p_found"]),
        fig3_pfound_conn=np.array(c3["curves"][1.0]["p_found"]),
        fig3_p_watched=np.array(c3["p_watched"]), fig3_p_unwatched=np.array(c3["p_unwatched"]),
    )
    summary = {"config": params, "n_rows": len(rows), "fig1": c1, "fig2": c2, "fig3": c3}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = REPORT_TEMPLATE.format(
        n_rows=len(rows), f1_on0=on0, f1_on1=on1, f1_off0=off0,
        f2_max=float(L.max()), f2_min=float(L[0, 0]),
        f3_pw=c3["p_watched"], f3_pu=c3["p_unwatched"],
        f3_d0=float(p0[0]), f3_d1=float(p0[-1]), f3_c1=float(p1[-1]))
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


register(ExperimentSpec(
    model="landscape",
    name="frontier_sweep",
    description="The frontier sweep, story-first: ~340 resumable runs feeding exactly three "
                "headline figures -- (1) pluralism needs the frontier AND disconnection, "
                "(2) the conviction x memory lock-in boundary, (3) discovery of new deep "
                "structure is coverage, not communication.",
    run=run,
    out_dir="frontier_sweep",
    params=dict(
        A_INTERS=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0), A_SEEDS=8,
        B_LAMBDAS=(0.0, 1.0, 2.0, 4.0, 8.0), B_OMEGAS=(0.8, 0.85, 0.9, 0.95, 0.97), B_SEEDS=6,
        C_BUDGETS=(6, 8, 12, 16), C_SEEDS=40,
    ),
    seeds=tuple(range(12)),
    consumes=dict(figures=["fig1_pluralism", "fig2_lockin", "fig3_discovery"]),
))
