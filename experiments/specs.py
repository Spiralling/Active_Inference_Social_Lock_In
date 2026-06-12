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
from experiments import divergent_wiring  # noqa: F401  (registration side effect)
from experiments import kuhn_phlogiston  # noqa: F401  (registration side effect)
from experiments import hawkes_rescue  # noqa: F401  (registration side effect)
from experiments import adaptive_rate  # noqa: F401  (registration side effect)
from experiments import fusion_survival  # noqa: F401  (registration side effect)
from experiments import wake_then_prune  # noqa: F401  (registration side effect)
from experiments import precrisis_check  # noqa: F401  (registration side effect)
from experiments import multi_candidate  # noqa: F401  (registration side effect)
from experiments import lakatos_conviction  # noqa: F401  (registration side effect)
from experiments import combined_mechanisms  # noqa: F401  (registration side effect)
from experiments import abc_conflict  # noqa: F401  (registration side effect)
from experiments import gated_divergence  # noqa: F401  (registration side effect)
from experiments import strain_targeted_bmr  # noqa: F401  (registration side effect)
from experiments import lockin_inferred_trust  # noqa: F401  (registration side effect)
from experiments import crisis_trace  # noqa: F401  (registration side effect)
from experiments import schism_threshold  # noqa: F401  (registration side effect)

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
