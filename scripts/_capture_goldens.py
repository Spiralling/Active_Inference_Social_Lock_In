"""Capture the byte-identity regression goldens BEFORE the spine refactor.

Hashes every array in each ``results/*/simulation_arrays.npz`` and writes
``tests/golden_array_hashes.json``. The migration regression test
(``tests/test_migration_byte_identity.py``) asserts that each migrated experiment
reproduces these hashes -- proving the refactor changed no numbers.

The deliberate ``forgetting=0.9`` + heterogeneity numerical fix is a SEPARATE
later commit; through the spine refactor every experiment stays byte-identical,
so these goldens are the right anchor.

    python scripts/_capture_goldens.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root on sys.path + utf-8 stdout)

import json

from _bootstrap import ROOT
from src.repro.hashing import hash_npz


def main() -> int:
    results = ROOT / "results"
    goldens: dict[str, dict[str, str]] = {}
    for npz in sorted(results.glob("*/simulation_arrays.npz")):
        name = npz.parent.name
        goldens[name] = hash_npz(npz)
        print(f"  {name:36s} {len(goldens[name])} arrays")
    out = ROOT / "tests" / "golden_array_hashes.json"
    out.write_text(json.dumps(goldens, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {len(goldens)} experiment goldens to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
