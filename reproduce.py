#!/usr/bin/env python3
"""Run the supplied figure experiments without installing this package itself."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser("simulate", help="Run the original fixed scientific settings")
    simulate.add_argument("--family", choices=["all", "camera", "communication", "gaussian"], default="all")
    simulate.add_argument("--output", type=Path, default=ROOT / "runs" / "full")
    reduce = commands.add_parser("reduce", help="Build compact plotting data from actual run outputs")
    reduce.add_argument("--raw", type=Path, default=ROOT / "runs" / "full")
    reduce.add_argument("--output", type=Path, default=ROOT / "build" / "recomputed_data")
    figures = commands.add_parser("figures", help="Render all sixteen unique supplied figure assets")
    figures.add_argument("--data", type=Path, default=ROOT / "data")
    figures.add_argument("--output", type=Path, default=ROOT / "build" / "figures")
    compare = commands.add_parser("compare", help="Compare generated PDFs with the two archive targets")
    compare.add_argument("--figures", type=Path, default=ROOT / "build" / "figures")
    compare.add_argument("--report", type=Path, default=ROOT / "build" / "figure_comparison.json")
    args = parser.parse_args()
    if args.command == "simulate":
        from iwai_figures.experiments import run_family
        families = ["camera", "communication", "gaussian"] if args.family == "all" else [args.family]
        for family in families:
            run_family(family, args.output.resolve() / family)
    elif args.command == "reduce":
        from iwai_figures.reduce import reduce_camera, reduce_communication, reduce_gaussian
        raw, output = args.raw.resolve(), args.output.resolve()
        reduce_camera(raw / "camera", output / "camera")
        reduce_communication(raw / "communication", output / "communication")
        reduce_gaussian(raw / "gaussian", raw / "communication", output / "gaussian")
    elif args.command == "figures":
        from iwai_figures.plotting import render_all
        render_all(args.data.resolve(), args.output.resolve())
    else:
        from iwai_figures.validation import compare_figures
        result = compare_figures(ROOT / "reference" / "figures", args.figures.resolve(), args.report.resolve())
        print(json.dumps({key: value for key, value in result.items() if key != "figures"}, indent=2))
        if result["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
