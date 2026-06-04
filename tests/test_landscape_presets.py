"""Tests for reusable structural landscape presets and examples."""

from __future__ import annotations

import numpy as np

from src.structural.landscape_examples import build_cosmology_extreme_landscape
from src.structural.landscape_presets import (
    bayesnet_preset_dark_matter,
    bayesnet_preset_modified_gravity,
    bayesnet_preset_scale_variant_laws,
    cosmology_basis,
    default_cluster_profiles,
    network_preset_four_cluster_sbm,
    sample_cluster_scalings,
)


def test_cosmology_basis_has_core_and_belt_disjoint():
    basis = cosmology_basis()
    assert len(basis.names) > 0
    assert set(basis.core_nodes).isdisjoint(set(basis.belt_nodes))
    assert set(basis.core_nodes).issubset(set(basis.names))
    assert set(basis.belt_nodes).issubset(set(basis.names))


def test_bayesnet_presets_share_basis_and_pd():
    basis = cosmology_basis()
    presets = (
        bayesnet_preset_dark_matter(basis),
        bayesnet_preset_scale_variant_laws(basis),
        bayesnet_preset_modified_gravity(basis),
    )
    for preset in presets:
        assert preset.belief_net.names == basis.names
        eigvals = np.linalg.eigvalsh(np.asarray(preset.belief_net.Pi))
        assert np.all(eigvals > 0.0)


def test_network_preset_four_cluster_membership_and_graph_size():
    preset = network_preset_four_cluster_sbm(n_agents=240, seed=7)
    assert preset.graph.A.shape == (240, 240)
    assert preset.membership.shape == (240,)
    assert len(np.unique(preset.membership)) == 4


def test_sample_cluster_scalings_shapes_and_determinism():
    profiles = default_cluster_profiles()
    sample_a = sample_cluster_scalings(n_agents=240, profiles=profiles, seed=13)
    sample_b = sample_cluster_scalings(n_agents=240, profiles=profiles, seed=13)

    assert np.asarray(sample_a.precision_scale).shape == (240,)
    assert np.asarray(sample_a.utility_scale).shape == (240,)
    assert np.asarray(sample_a.cluster_id).shape == (240,)

    assert np.array_equal(np.asarray(sample_a.precision_scale), np.asarray(sample_b.precision_scale))
    assert np.array_equal(np.asarray(sample_a.utility_scale), np.asarray(sample_b.utility_scale))
    assert np.array_equal(np.asarray(sample_a.cluster_id), np.asarray(sample_b.cluster_id))


def test_build_cosmology_extreme_landscape_shapes():
    ex = build_cosmology_extreme_landscape(n_agents=240, seed=3)
    d = len(ex.basis.names)

    assert len(ex.bayesnet_presets) == 3
    assert len(ex.utility_presets) == 3
    assert ex.network_preset.graph.A.shape == (240, 240)
    assert np.asarray(ex.cluster_sample.precision_scale).shape == (240,)
    assert np.asarray(ex.per_agent_utility).shape == (240, d)
