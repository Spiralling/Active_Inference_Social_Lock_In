"""CLI entry for the big resumable sweep.

    python scripts/sweep/run_big_sweep.py --limit 20      # smoke test (first 20 configs)
    python scripts/sweep/run_big_sweep.py --tier 1        # the dense 2-D phase maps (~950 runs)
    python scripts/sweep/run_big_sweep.py --tier all      # phase maps + random sample (~2.5k)
    python scripts/sweep/run_big_sweep.py --analyze       # (re)build phase maps + REPORT.md

Resumable: re-invoke with the same args and completed runs are skipped. Safe to stop anytime.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="all", choices=["1", "2", "all"])
    ap.add_argument("--limit", type=int, default=None, help="run only the first N jobs (smoke test)")
    ap.add_argument("--tier2-n", type=int, default=1500, help="# random-sample configs in Tier 2")
    ap.add_argument("--analyze", action="store_true", help="build phase maps + REPORT.md and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    from scripts.sweep import space, runner

    if args.analyze:
        from scripts.sweep import analyze
        return analyze.main()

    jobs = space.all_jobs(tier=args.tier, tier2_n=args.tier2_n)
    print(f"tier={args.tier}: {len(jobs)} jobs"
          + (f" (limited to {args.limit})" if args.limit else ""), flush=True)
    runner.run_jobs(jobs, limit=args.limit, quiet=args.quiet)

    # always refresh the analysis after a run
    from scripts.sweep import analyze
    analyze.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
