"""The resumable runner: hash -> skip-if-done -> run -> append a tidy JSONL row + a small npz.

Crash-safe and resumable: every completed run is a line in ``results/big_sweep/rows.jsonl`` and a
``traj/<hash>.npz``. On restart we load the set of completed hashes and skip them, so the sweep
can be stopped and re-invoked at will. A run that raises is recorded as a row with ``error`` set
(so a single bad config never kills the sweep).
"""

from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "big_sweep"
ROWS = OUT / "rows.jsonl"
TRAJ = OUT / "traj"


def cfg_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def load_done():
    if not ROWS.exists():
        return set()
    done = set()
    with ROWS.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["hash"])
                except Exception:
                    pass
    return done


def _append_row(row):
    with ROWS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def run_jobs(jobs, limit=None, quiet=False):
    """Run (or resume) a list of config dicts. Returns (n_ran, n_skipped, n_errors)."""
    OUT.mkdir(parents=True, exist_ok=True)
    TRAJ.mkdir(parents=True, exist_ok=True)
    from scripts.sweep.run_one import run_one     # late import: keeps space/runner JAX-free to load

    if limit is not None:
        jobs = jobs[:limit]
    done = load_done()
    n_ran = n_skip = n_err = 0
    t0 = time.time()
    total = len(jobs)
    for i, cfg in enumerate(jobs):
        h = cfg_hash(cfg)
        if h in done:
            n_skip += 1
            continue
        row = {"hash": h, **cfg}
        try:
            scalars, traj = run_one(cfg)
            row.update(scalars)
            row["error"] = ""
            np.savez_compressed(TRAJ / f"{h}.npz", **traj)
        except Exception as e:        # never let one config kill the sweep
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            n_err += 1
        _append_row(row)
        done.add(h)
        n_ran += 1
        if not quiet and (n_ran % 10 == 0 or i == total - 1):
            el = time.time() - t0
            rate = el / max(n_ran, 1)
            remain = (total - i - 1) * rate
            print(f"  [{i + 1}/{total}] ran={n_ran} skip={n_skip} err={n_err} "
                  f"| {rate:.2f}s/run | ETA {remain / 60:.1f} min", flush=True)
    if not quiet:
        print(f"done: ran={n_ran} skipped={n_skip} errors={n_err} "
              f"in {(time.time() - t0) / 60:.1f} min -> {ROWS}", flush=True)
    return n_ran, n_skip, n_err
