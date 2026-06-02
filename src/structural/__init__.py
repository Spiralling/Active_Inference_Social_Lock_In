"""Structural paradigm model: each agent's belief is a Gaussian Bayes net in
information (precision) form, paradigm change is carry-over by Schur complement,
and structural inference is closed-form Bayesian Model Reduction.

The Gaussian-structural sibling of ``src/pomdp/`` (categorical) and the matrix
generalisation of ``src/inference.py`` (scalar). See
``notes/bmr_feynman_linear_algebra.tex`` for the math and
``notebooks/18_structural_phlogiston.ipynb`` for the experiment.
"""

from src.structural.belief import (
    GaussianBeliefNet, vague_prior, add_fisher, combine, border,
)
from src.structural.bmr import (
    log_evidence, schur_marginalize, condition, carryover, bmr_prune,
    prune_node_prior, prune_edge_prior,
)
from src.structural.world import fisher_deposit, sample_o
from src.structural.phlogiston import (
    StructuralConfig, NODE_NAMES, phi_true_at, candidate_priors,
    phlogiston_prior, oxygen_prior, H_observable,
)
from src.structural.agent import evidence_race
# NB: do not import the ``step`` function into the package namespace -- it would
# shadow the ``src.structural.step`` submodule. Call ``run`` or
# ``src.structural.step.step`` directly.
from src.structural.step import (
    PopulationState, init_state, run, fuse, trust_weights,
)
from src.structural import observables
from src.structural import linalg

__all__ = [
    "linalg",
    "GaussianBeliefNet", "vague_prior", "add_fisher", "combine", "border",
    "log_evidence", "schur_marginalize", "condition", "carryover", "bmr_prune",
    "prune_node_prior", "prune_edge_prior",
    "fisher_deposit", "sample_o",
    "StructuralConfig", "NODE_NAMES", "phi_true_at", "candidate_priors",
    "phlogiston_prior", "oxygen_prior", "H_observable",
    "evidence_race",
    "PopulationState", "init_state", "run", "fuse", "trust_weights",
    "observables",
]
