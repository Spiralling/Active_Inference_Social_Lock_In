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
from src.structural.action import (
    HubProposal, MoveScore, residual_block, residual_from_errors, propose_hub,
    hub_couplings, pd_safe_scale, default_coupling_scale, wake_hub, expansion_score,
    reduction_score, select_action,
)
from src.structural import world_net
from src.structural.world_net import (
    LatentWorldConfig, UNCONCEIVED_DRIVES, true_phlogiston_world, agnostic_prior,
    sample_world, recovery_scores, CauseSpec, scheduled_world,
)
from src.structural.world import fisher_deposit, sample_o
from src.structural.phlogiston import (
    StructuralConfig, NODE_NAMES, phi_true_at, candidate_priors,
    phlogiston_prior, oxygen_prior, H_observable,
    observation_operator, observation_rows, measured_nodes, disagreement_row_mask,
    phlogiston_bn, gravimetric_H, gravimetric_rows, derived_channel_precision,
    conviction_u, conviction_field, balanced_lambda,
)
from src.structural.agent import evidence_race
# NB: do not import the ``step`` function into the package namespace -- it would
# shadow the ``src.structural.step`` submodule. Call ``run`` or
# ``src.structural.step.step`` directly.
from src.structural.step import (
    PopulationState, init_state, run, fuse, trust_weights,
    run_final, run_trace, run_trace_index, run_trace_net,
    run_trace_bmr, run_trace_precision, run_trace_schedule,
    run_trace_index_schedule, run_trace_net_schedule,
)
from src.structural import observables
from src.structural import shells
from src.structural import precision
from src.structural import plot
from src.structural import dual_field as dual_field_mod
from src.structural import linalg
from src.structural import graphs
from src.structural.graphs import (
    Graph, erdos_renyi, barabasi_albert, scale_free, watts_strogatz,
    community, sbm, ring, lattice, complete, graph_from_config,
)
from src.structural.landscape_presets import (
    LandscapeBasis,
    BayesNetPreset,
    UtilityPreset,
    NetworkPreset,
    ClusterProfile,
    ClusterSample,
    cosmology_basis,
    bayesnet_preset_dark_matter,
    bayesnet_preset_scale_variant_laws,
    bayesnet_preset_modified_gravity,
    utility_preset_precision_extreme,
    utility_preset_intrinsic_extreme,
    utility_preset_neutral,
    network_preset_four_cluster_sbm,
    default_cluster_profiles,
    sample_cluster_scalings,
    apply_agent_scalings,
)
from src.structural.landscape_examples import (
    CosmologyLandscapeExample,
    build_cosmology_extreme_landscape,
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
    "HubProposal", "MoveScore", "residual_block", "residual_from_errors",
    "propose_hub", "hub_couplings", "pd_safe_scale", "default_coupling_scale",
    "wake_hub", "expansion_score", "reduction_score", "select_action",
    "world_net", "LatentWorldConfig", "UNCONCEIVED_DRIVES",
    "true_phlogiston_world", "agnostic_prior", "sample_world", "recovery_scores",
    "CauseSpec", "scheduled_world",
    "fisher_deposit", "sample_o",
    "StructuralConfig", "NODE_NAMES", "phi_true_at", "candidate_priors",
    "phlogiston_prior", "oxygen_prior", "H_observable",
    "observation_operator", "observation_rows", "measured_nodes",
    "disagreement_row_mask", "phlogiston_bn", "gravimetric_H", "gravimetric_rows",
    "derived_channel_precision", "conviction_u", "conviction_field", "balanced_lambda",
    "evidence_race",
    "PopulationState", "init_state", "run", "fuse", "trust_weights",
    "run_final", "run_trace", "run_trace_index", "run_trace_net",
    "run_trace_bmr", "run_trace_precision", "run_trace_schedule",
    "run_trace_index_schedule", "run_trace_net_schedule",
    "observables", "shells", "precision", "plot",
    "LandscapeBasis", "BayesNetPreset", "UtilityPreset", "NetworkPreset",
    "ClusterProfile", "ClusterSample",
    "cosmology_basis",
    "bayesnet_preset_dark_matter",
    "bayesnet_preset_scale_variant_laws",
    "bayesnet_preset_modified_gravity",
    "utility_preset_precision_extreme",
    "utility_preset_intrinsic_extreme",
    "utility_preset_neutral",
    "network_preset_four_cluster_sbm",
    "default_cluster_profiles",
    "sample_cluster_scalings",
    "apply_agent_scalings",
    "CosmologyLandscapeExample",
    "build_cosmology_extreme_landscape",
]
