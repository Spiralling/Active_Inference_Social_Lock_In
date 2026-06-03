"""Run production branch checks and fail fast on errors."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(label: str, cmd: list[str]) -> int:
    print(f"==> {label}")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"FAILED: {label} (exit {result.returncode})")
        return result.returncode
    print(f"OK: {label}")
    return 0


def main() -> int:
    python = sys.executable
    checks: list[tuple[str, list[str]]] = [
        (
            "module map contract check",
            [python, str(ROOT / "scripts" / "module_map.py"), "--check"],
        ),
        (
            "structural smoke rollout",
            [python, str(ROOT / "scripts" / "production_smoke.py")],
        ),
        (
            "focused pytest subset",
            [
                python,
                "-m",
                "pytest",
                "tests/test_bayesnet.py",
                "tests/test_graphs.py",
                "tests/test_linalg.py",
                "tests/test_structural_kernel.py::test_run_final_matches_step[derived]",
            ],
        ),
    ]

    for label, cmd in checks:
        code = run_step(label, cmd)
        if code != 0:
            return code

    print("All production checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
