"""CLI entry for the overnight kuhn_phlogiston sweep (parallel + resumable).

    python scripts/sweep_kuhn/run_sweep.py --limit 12 --workers 4    # smoke test
    python scripts/sweep_kuhn/run_sweep.py --workers 10              # the full sweep
    python scripts/sweep_kuhn/run_sweep.py --workers 10 --hours 6    # hard time cap
    python scripts/sweep_kuhn/run_sweep.py --analyze                 # (re)build figures + REPORT

Resumable: every completed run is one line in ``results/kuhn_sweep/rows.jsonl``; re-invoking
skips completed hashes, so the sweep can be stopped (Ctrl-C) and resumed at will. Workers are
separate processes (Windows spawn-safe); the main process is the single JSONL writer.
"""
from __future__ import annotations

import sys
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "kuhn_sweep"
ROWS = OUT / "rows.jsonl"


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="run only the first N jobs (smoke)")
    p.add_argument("--hours", type=float, default=None, help="stop submitting after this long")
    p.add_argument("--analyze", action="store_true", help="only (re)build figures + report")
    args = p.parse_args()

    if args.analyze:
        from scripts.sweep_kuhn.analyze import main as analyze_main
        analyze_main()
        return 0

    from scripts.sweep_kuhn.space import all_jobs
    jobs = all_jobs()
    if args.limit is not None:
        jobs = jobs[: args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if ROWS.exists():
        with ROWS.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        row = json.loads(line)
                        if row.get("error", "") == "":   # errored rows are retried on resume
                            done.add(row["hash"])
                    except Exception:
                        pass
    todo = [(cfg_hash(c), c) for c in jobs if cfg_hash(c) not in done]
    print(f"{len(jobs)} jobs total, {len(jobs) - len(todo)} cached, {len(todo)} to run "
          f"on {args.workers} workers", flush=True)
    if not todo:
        print("nothing to do; use --analyze to rebuild figures")
        return 0

    t0 = time.time()
    n_done = n_err = n_cancel = 0
    with ROWS.open("a", encoding="utf-8") as fh, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one_job, cfg): (h, cfg) for h, cfg in todo}
        capped = False
        for fut in as_completed(futures):
            if (args.hours is not None and not capped
                    and (time.time() - t0) > args.hours * 3600):
                capped = True                            # enforce the cap on PENDING work
                for f in futures:
                    if f.cancel():
                        n_cancel += 1
                print(f"time cap reached: cancelled {n_cancel} pending runs (resumable)",
                      flush=True)
            if fut.cancelled():
                continue
            h, cfg = futures[fut]
            row = {"hash": h, **cfg}
            try:
                row.update(fut.result())
                row["error"] = ""
            except Exception as e:                      # one bad config never kills the night
                row["error"] = f"{type(e).__name__}: {e}"[:300]
                n_err += 1
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 20 == 0:
                rate = (time.time() - t0) / n_done
                eta = (len(futures) - n_cancel - n_done) * rate / 60
                print(f"  [{n_done}/{len(futures) - n_cancel}] err={n_err} | {rate:.1f}s/run "
                      f"(wall) | ETA {eta:.0f} min", flush=True)
    print(f"done: {n_done} runs, {n_err} errors, {(time.time() - t0) / 60:.1f} min -> {ROWS}")
    print("now: python scripts/sweep_kuhn/run_sweep.py --analyze")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
