"""Run experiments with provenance -- the shared orchestration behind ``scripts/run.py``
and the thin per-experiment forwarder scripts.

Switches matplotlib to the headless Agg backend once (so library/experiment modules never
have to, keeping notebooks safe), runs the spec into ``results/<out_dir>/``, then writes the
provenance ``manifest.json`` next to the outputs.
"""
from __future__ import annotations

from src.repro.headless import headless
from src.repro.manifest import write_manifest
from src.repro.paths import ROOT, results_dir
from experiments.registry import MODELS, by_model, load_specs


def run_named(name: str) -> None:
    specs = load_specs()
    if name not in specs:
        raise SystemExit(f"unknown experiment: {name!r} (see `python scripts/run.py list`)")
    headless()
    spec = specs[name]
    out = results_dir(spec.out_dir)
    print(f"[run] {spec.model}/{spec.name} -> results/{spec.out_dir}/")
    spec.run(out, dict(spec.params))
    man = write_manifest(out, experiment=spec.name, params=spec.params, seeds=spec.seeds)
    print(f"[run] {spec.name}: wrote {man.relative_to(ROOT)}")


def run_all(model: str | None = None) -> None:
    grouped = by_model()
    for m in MODELS:
        if model and m != model:
            continue
        for spec in grouped.get(m, []):
            run_named(spec.name)
