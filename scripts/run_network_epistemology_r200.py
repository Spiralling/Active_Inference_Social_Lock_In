"""Backup R=200 run of network_epistemology into a SEPARATE results dir, so it does
not overwrite the canonical R=100 run. Launched in parallel as a higher-precision
safety copy. Delete after the runs settle."""
from pathlib import Path


def main():
    import _bootstrap  # noqa: F401
    from experiments.network_epistemology import run
    from experiments.registry import load_specs
    from src.repro.paths import ROOT

    p = dict(load_specs()["network_epistemology"].params)
    p.update(R=200, R_HEAT=40)
    out = ROOT / "results" / "network_epistemology_r200"
    out.mkdir(parents=True, exist_ok=True)
    run(out, p)
    print("\nR=200 backup complete ->", out)


if __name__ == "__main__":
    main()
