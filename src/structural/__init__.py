"""Structural paradigm model: agents carry Gaussian commitment networks.

The runtime dynamics use information-form belief nets ``(Pi, h)``; paradigm
change is Schur-complement carry-over; structural inference is closed-form
Bayesian Model Reduction. Interpretability lives beside that fast form:
``bayesnet.LinearGaussianBN`` re-reads a joint posterior as named CPDs, and
``graphs.Graph`` keeps the social topology explicit instead of hiding it inside
the fusion matrix ``W``.

See ``notes/architecture_interpretability.md`` for the concept-to-code map and
``notebooks/28_unified_staircase_lockin.ipynb`` / ``29_bayesnet_cpd_over_time``
for the current experiments.
"""

from src.structural.belief import (
    GaussianBeliefNet, vague_prior, add_fisher, combine, border,
)
from src.structural.bayesnet import (
    LinearGaussianBN, from_edges, relational_operator,
)
from src.structural.bmr import (
    log_evidence, schur_marginalize, condition, carryover, bmr_prune,
    prune_node_prior, prune_edge_prior,
)
from src.structural.dual_field import (
    PrecisionUtilityNet, ScoreBreakdown, score_edit_with_utility,
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
from src.structural import graphs
from src.structural.graphs import (
    Graph, erdos_renyi, barabasi_albert, scale_free, watts_strogatz,
    community, sbm, ring, lattice, complete, graph_from_config,
)

__all__ = [
    "linalg",
    "graphs", "Graph", "erdos_renyi", "barabasi_albert", "scale_free",
    "watts_strogatz", "community", "sbm", "ring", "lattice", "complete",
    "graph_from_config",
    "GaussianBeliefNet", "vague_prior", "add_fisher", "combine", "border",
    "LinearGaussianBN", "from_edges", "relational_operator",
    "log_evidence", "schur_marginalize", "condition", "carryover", "bmr_prune",
    "prune_node_prior", "prune_edge_prior",
    "PrecisionUtilityNet", "ScoreBreakdown", "score_edit_with_utility",
    "fisher_deposit", "sample_o",
    "StructuralConfig", "NODE_NAMES", "phi_true_at", "candidate_priors",
    "phlogiston_prior", "oxygen_prior", "H_observable",
    "evidence_race",
    "PopulationState", "init_state", "run", "fuse", "trust_weights",
    "observables",
]
