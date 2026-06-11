"""Verify a migrated experiment reproduces its byte-identity golden.

Sibling of ``_capture_goldens.py``: hashes the arrays a (re-run) experiment wrote to
``results/<name>/simulation_arrays.npz`` and compares them, array by array, to the committed
``tests/golden_array_hashes.json``. This is the gate every spine migration must pass -- it proves
the refactor changed no numbers.

    python scripts/_verify_golden.py cosmology_sweeps [<name> ...]
    python scripts/_verify_golden.py --all          # every golden key with a result on disk

Exit code is non-zero if any array mismatches or is missing, so it doubles as a CI check.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root on sys.path + utf-8 stdout)

import json
import sys

from _bootstrap import ROOT
from src.repro.hashing import hash_npz

GOLDENS = ROOT / "tests" / "golden_array_hashes.json"


def verify(name: str, goldens: dict) -> bool:
    if name not in goldens:
        print(f"  {name:30s} NO GOLDEN (capture it first)")
        return False
    npz = ROOT / "results" / name / "simulation_arrays.npz"
    if not npz.exists():
        print(f"  {name:30s} MISSING {npz.relative_to(ROOT)} (run the experiment first)")
        return False
    got = hash_npz(npz)
    want = goldens[name]
    ok = True
    for key in sorted(want):
        if key not in got:
            print(f"  {name}/{key}: MISSING from rerun"); ok = False
        elif got[key] != want[key]:
            print(f"  {name}/{key}: MISMATCH (golden {want[key][:12]} != got {got[key][:12]})"); ok = False
    extra = sorted(set(got) - set(want))
    if extra:
        print(f"  {name}: extra arrays not in golden: {extra}"); ok = False
    print(f"  {name:30s} {'OK' if ok else 'FAIL'}  ({len(want)} arrays)")
    return ok


def main() -> int:
    goldens = json.loads(GOLDENS.read_text(encoding="utf-8"))
    args = sys.argv[1:]
    if args == ["--all"]:
        names = sorted(goldens)
    elif args:
        names = args
    else:
        print(__doc__)
        return 2
    all_ok = all(verify(n, goldens) for n in names)
    print(f"\n{'ALL MATCH' if all_ok else 'MISMATCH -- do not collapse the old script'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
