"""The experiment registry: every paper experiment as one declarative spec.

Model-first. **Phlogiston** and **cosmology** are co-equal main models (phlogiston's
canonical reference is the multi-agent topology experiment); ``landscape`` is a
cosmology-adjacent group. Each :class:`ExperimentSpec` lifts a former script's
hardcoded constants into a visible ``params`` dict and points at a ``run`` callable
that does the compute (via ``src.structural.models.*``) and saves arrays/figures to
``results/<out_dir>/``. The registry doubles as the paper's reproducibility appendix:
one place that answers "what experiments exist, how do I run each, and what does it
produce."

Specs are registered in :mod:`experiments.specs` (populated during migration). This
module is just the data model + lookup helpers, so it has no heavy imports and is safe
to import anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# Model display order (also the order EXPERIMENTS.md groups by).
MODELS: tuple[str, ...] = ("phlogiston", "cosmology", "landscape")


@dataclass(frozen=True)
class ExperimentSpec:
    """One runnable experiment, declaratively.

    ``model``       : which main model this belongs to (see ``MODELS``).
    ``name``        : unique id; the runner key (``run.py run <name>``).
    ``description`` : one line for the index / ``list``.
    ``run``         : ``(out_dir: Path, params: dict) -> None`` -- does the compute and
                      saves ``simulation_arrays.npz`` / ``summary.json`` / figures into
                      ``out_dir``. The runner writes ``manifest.json`` afterwards.
    ``out_dir``     : ``results/<out_dir>/`` (kept equal to the legacy folder name so
                      existing notebooks/goldens still resolve).
    ``params``      : JSON-serializable config, lifted from the old script's top-of-file
                      constants -- the declarative surface and the manifest record.
    ``seeds``       : the seeds the experiment sweeps (recorded in the manifest).
    ``canonical``   : the headline reference experiment for its model.
    ``consumes``    : ``{"notebook": ..., "figures": [...]}`` for the index.
    """

    model: str
    name: str
    description: str
    run: Callable
    out_dir: str
    params: dict = field(default_factory=dict)
    seeds: tuple = ()
    canonical: bool = False
    consumes: dict = field(default_factory=dict)


REGISTRY: dict[str, ExperimentSpec] = {}


def register(spec: ExperimentSpec) -> ExperimentSpec:
    if spec.model not in MODELS:
        raise ValueError(f"{spec.name}: unknown model {spec.model!r} (expected one of {MODELS})")
    if spec.name in REGISTRY:
        raise ValueError(f"duplicate experiment name: {spec.name!r}")
    REGISTRY[spec.name] = spec
    return spec


def load_specs() -> dict[str, ExperimentSpec]:
    """Import the spec definitions (which call ``register``) and return the registry."""
    if not REGISTRY:
        from experiments import specs  # noqa: F401  (registration side effect)
    return REGISTRY


def by_model() -> dict[str, list[ExperimentSpec]]:
    """Registry grouped by model, in ``MODELS`` order, canonical first within a model."""
    load_specs()
    grouped: dict[str, list[ExperimentSpec]] = {m: [] for m in MODELS}
    for spec in REGISTRY.values():
        grouped.setdefault(spec.model, []).append(spec)
    for specs_list in grouped.values():
        specs_list.sort(key=lambda s: (not s.canonical, s.name))
    return grouped
