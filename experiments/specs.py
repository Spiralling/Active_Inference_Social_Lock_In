"""Experiment spec definitions -- importing this module registers every experiment.

Populated during migration (phlogiston model first). Each imported experiment module calls
``register(...)`` at import time, so importing this module is what builds the registry.
"""
from __future__ import annotations

# --- phlogiston model ---
from experiments import multiagent_topology  # noqa: F401  (registration side effect)
from experiments import endogenous_gamma  # noqa: F401  (registration side effect)
from experiments import staircase_gate  # noqa: F401  (registration side effect)
from experiments import meanmatched_structure  # noqa: F401  (registration side effect)

# --- cosmology model ---
from experiments import cosmology_tracking  # noqa: F401  (registration side effect)
