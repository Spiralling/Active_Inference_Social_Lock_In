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
from experiments import structural_pluralism  # noqa: F401  (registration side effect)
from experiments import kuhn_phlogiston  # noqa: F401  (registration side effect)

# --- cosmology model ---
from experiments import cosmology_tracking  # noqa: F401  (registration side effect)
from experiments import cosmology_sweeps  # noqa: F401  (registration side effect)
from experiments import cosmology_forgetting  # noqa: F401  (registration side effect)
from experiments import bayesnet_comms  # noqa: F401  (registration side effect)
from experiments import partial_obs_comms  # noqa: F401  (registration side effect)
from experiments import cosmology_coarse_world  # noqa: F401  (registration side effect)
from experiments import cosmology_regrowth  # noqa: F401  (registration side effect)
from experiments import cosmology_poisson  # noqa: F401  (registration side effect)
from experiments import cosmology_twofield  # noqa: F401  (registration side effect)
from experiments import kuhn_cycle  # noqa: F401  (registration side effect)

# --- landscape group (cosmology-adjacent) ---
from experiments import landscape_simulation  # noqa: F401  (registration side effect)
from experiments import landscape_two_stage  # noqa: F401  (registration side effect)
from experiments import beta_calibration  # noqa: F401  (registration side effect)
from experiments import landscape_frontier  # noqa: F401  (registration side effect)
from experiments import frontier_sweep  # noqa: F401  (registration side effect)
