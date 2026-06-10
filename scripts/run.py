"""The single entry point for running paradigm experiments.

    python scripts/run.py list                   # all experiments, grouped by model
    python scripts/run.py run <name> [<name>...]  # run one or more, with provenance
    python scripts/run.py run-all [--model M]     # run every experiment (optionally one model)
    python scripts/run.py index                   # (re)write EXPERIMENTS.md

Each run writes ``results/<out_dir>/{simulation_arrays.npz, summary.json, manifest.json,
*.png}``; the manifest records git SHA + env + seeds + params + per-array hashes -- the
provenance trail that was previously missing.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root on sys.path + utf-8 stdout)

import argparse

from src.repro.paths import ROOT
from experiments.registry import MODELS, by_model
from experiments.runner import run_all, run_named


def cmd_list(_args) -> int:
    grouped = by_model()
    any_shown = False
    for model in MODELS:
        specs = grouped.get(model, [])
        if not specs:
            continue
        any_shown = True
        print(f"\n{model.upper()}")
        for s in specs:
            star = "*" if s.canonical else " "
            print(f"  {star} {s.name:32s} {s.description}")
    if not any_shown:
        print("(no experiments registered yet)")
    return 0


def cmd_run(args) -> int:
    for name in args.names:
        run_named(name)
    return 0


def cmd_run_all(args) -> int:
    run_all(args.model)
    return 0


def cmd_index(_args) -> int:
    from experiments.index import write_index

    path = write_index()
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="run.py", description="Run paradigm experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list experiments grouped by model").set_defaults(fn=cmd_list)
    pr = sub.add_parser("run", help="run one or more experiments by name")
    pr.add_argument("names", nargs="+")
    pr.set_defaults(fn=cmd_run)
    pa = sub.add_parser("run-all", help="run every experiment")
    pa.add_argument("--model", choices=MODELS, help="restrict to one model")
    pa.set_defaults(fn=cmd_run_all)
    sub.add_parser("index", help="(re)write EXPERIMENTS.md").set_defaults(fn=cmd_index)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
