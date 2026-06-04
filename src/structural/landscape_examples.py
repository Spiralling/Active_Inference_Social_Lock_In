"""Concrete examples assembling reusable structural landscape presets."""

from __future__ import annotations

from dataclasses import dataclass

import jax

from src.structural.landscape_presets import (
    BayesNetPreset,
    ClusterProfile,
    ClusterSample,
    LandscapeBasis,
    NetworkPreset,
    UtilityPreset,
    apply_agent_scalings,
    bayesnet_preset_dark_matter,
    bayesnet_preset_modified_gravity,
    bayesnet_preset_scale_variant_laws,
    cosmology_basis,
    default_cluster_profiles,
    network_preset_four_cluster_sbm,
    sample_cluster_scalings,
    utility_preset_intrinsic_extreme,
    utility_preset_neutral,
    utility_preset_precision_extreme,
)


@dataclass(frozen=True)
class CosmologyLandscapeExample:
    basis: LandscapeBasis
    bayesnet_presets: tuple[BayesNetPreset, ...]
    utility_presets: tuple[UtilityPreset, ...]
    network_preset: NetworkPreset
    cluster_profiles: tuple[ClusterProfile, ...]
    cluster_sample: ClusterSample
    per_agent_utility: jax.Array


def build_cosmology_extreme_landscape(
    n_agents: int = 240,
    seed: int = 0,
) -> CosmologyLandscapeExample:
    basis = cosmology_basis()

    bayesnet_presets = (
        bayesnet_preset_dark_matter(basis),
        bayesnet_preset_scale_variant_laws(basis),
        bayesnet_preset_modified_gravity(basis),
    )
    utility_presets = (
        utility_preset_precision_extreme(basis.names),
        utility_preset_intrinsic_extreme(basis.names),
        utility_preset_neutral(basis.names),
    )

    network_preset = network_preset_four_cluster_sbm(n_agents=n_agents, seed=seed)
    profiles = default_cluster_profiles()
    sample = sample_cluster_scalings(n_agents=n_agents, profiles=profiles, seed=seed)
    per_agent_utility = apply_agent_scalings(utility_presets[1].intrinsic_utility, sample)

    return CosmologyLandscapeExample(
        basis=basis,
        bayesnet_presets=bayesnet_presets,
        utility_presets=utility_presets,
        network_preset=network_preset,
        cluster_profiles=profiles,
        cluster_sample=sample,
        per_agent_utility=per_agent_utility,
    )
