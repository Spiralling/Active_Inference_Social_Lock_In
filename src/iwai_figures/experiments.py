"""The three figure-producing experiment families; no historical dispatch hooks."""

from pathlib import Path


def run_family(family, output_dir, conditions=None):
    """Run frozen conditions into a new directory (existing results are not overwritten).

    Family names: camera, communication, gaussian. The scopes reproduce exactly
    the figures in the two supplied paper versions. Partial production suites
    are deliberately not accepted, because the analyses require paired controls.
    """
    if conditions is not None:
        raise ValueError(
            "Partial suites are not supported; use a complete frozen figure family"
        )
    if family == "camera":
        from .camera import run
    elif family == "communication":
        from .communication import run
    elif family == "gaussian":
        from .gaussian import run
    else:
        raise ValueError(f"Unknown family {family!r}")
    return run(Path(output_dir))
