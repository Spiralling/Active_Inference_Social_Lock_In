"""Reusable structural landscape presets for large and extreme settings."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from src.structural.belief import GaussianBeliefNet
from src.structural.graphs import Graph, community


@dataclass(frozen=True)
class LandscapeBasis:
    names: tuple[str, ...]
    core_nodes: tuple[str, ...]
    belt_nodes: tuple[str, ...]


@dataclass(frozen=True)
class BayesNetPreset:
    name: str
    belief_net: GaussianBeliefNet
    description: str


@dataclass(frozen=True)
class UtilityPreset:
    name: str
    intrinsic_utility: jax.Array
    alpha: float
    description: str


@dataclass(frozen=True)
class NetworkPreset:
    name: str
    graph: Graph
    membership: np.ndarray
    description: str


@dataclass(frozen=True)
class ClusterProfile:
    label: str
    proportion: float
    precision_mean: float
    precision_sd: float
    utility_scale_mean: float
    utility_scale_sd: float


@dataclass(frozen=True)
class ClusterSample:
    precision_scale: jax.Array
    utility_scale: jax.Array
    cluster_id: jax.Array


def cosmology_basis() -> LandscapeBasis:
    """Shared cosmology basis with coherence, anomalies, and alternatives."""
    names = (
        "law_coherence",
        "rotation_curve_anomaly",
        "large_scale_structure_anomaly",
        "dark_matter_commitment",
        "scale_variant_laws_commitment",
        "modified_gravity_commitment",
    )
    core_nodes = (
        "law_coherence",
        "rotation_curve_anomaly",
        "large_scale_structure_anomaly",
    )
    belt_nodes = (
        "dark_matter_commitment",
        "scale_variant_laws_commitment",
        "modified_gravity_commitment",
    )
    return LandscapeBasis(names=names, core_nodes=core_nodes, belt_nodes=belt_nodes)


def _diagonally_dominant_pi(
    basis: LandscapeBasis,
    base_diag: float,
    couplings: dict[tuple[str, str], float],
    margin: float = 0.2,
) -> jax.Array:
    """Build symmetric positive-definite precision by diagonal dominance."""
    d = len(basis.names)
    idx = {name: i for i, name in enumerate(basis.names)}
    pi = np.zeros((d, d), dtype=np.float32)

    for (a, b), weight in couplings.items():
        i, j = idx[a], idx[b]
        if i == j:
            continue
        w = float(weight)
        pi[i, j] += w
        pi[j, i] += w

    row_abs = np.sum(np.abs(pi), axis=1)
    np.fill_diagonal(pi, base_diag + row_abs + margin)
    return jnp.asarray(pi)


def _preset_h_from_mu(pi: jax.Array, mu: np.ndarray) -> jax.Array:
    return pi @ jnp.asarray(mu, dtype=pi.dtype)


def _vector_from_weights(names: tuple[str, ...], weights: dict[str, float]) -> jax.Array:
    return jnp.asarray([weights.get(name, 0.0) for name in names], dtype=jnp.float32)


def bayesnet_preset_dark_matter(basis: LandscapeBasis) -> BayesNetPreset:
    couplings = {
        ("law_coherence", "dark_matter_commitment"): 0.35,
        ("rotation_curve_anomaly", "dark_matter_commitment"): -0.45,
        ("large_scale_structure_anomaly", "dark_matter_commitment"): -0.35,
        ("dark_matter_commitment", "modified_gravity_commitment"): -0.22,
        ("dark_matter_commitment", "scale_variant_laws_commitment"): -0.2,
    }
    pi = _diagonally_dominant_pi(basis, base_diag=1.1, couplings=couplings)
    mu = np.asarray(
        _vector_from_weights(
            basis.names,
            {
                "law_coherence": 0.7,
                "rotation_curve_anomaly": 0.1,
                "large_scale_structure_anomaly": 0.1,
                "dark_matter_commitment": 1.6,
                "scale_variant_laws_commitment": -0.8,
                "modified_gravity_commitment": -0.9,
            },
        )
    )
    h = _preset_h_from_mu(pi, mu)
    net = GaussianBeliefNet(Pi=pi, h=h, names=basis.names)
    return BayesNetPreset(
        name="dark_matter",
        belief_net=net,
        description="Prior favoring hidden matter while preserving coherent laws.",
    )


def bayesnet_preset_scale_variant_laws(basis: LandscapeBasis) -> BayesNetPreset:
    couplings = {
        ("rotation_curve_anomaly", "scale_variant_laws_commitment"): -0.4,
        ("large_scale_structure_anomaly", "scale_variant_laws_commitment"): -0.34,
        ("law_coherence", "scale_variant_laws_commitment"): -0.36,
        ("scale_variant_laws_commitment", "dark_matter_commitment"): -0.24,
        ("scale_variant_laws_commitment", "modified_gravity_commitment"): -0.2,
    }
    pi = _diagonally_dominant_pi(basis, base_diag=1.1, couplings=couplings)
    mu = np.asarray(
        _vector_from_weights(
            basis.names,
            {
                "law_coherence": -0.9,
                "rotation_curve_anomaly": 0.6,
                "large_scale_structure_anomaly": 0.6,
                "dark_matter_commitment": -0.7,
                "scale_variant_laws_commitment": 1.5,
                "modified_gravity_commitment": -0.6,
            },
        )
    )
    h = _preset_h_from_mu(pi, mu)
    net = GaussianBeliefNet(Pi=pi, h=h, names=basis.names)
    return BayesNetPreset(
        name="scale_variant_laws",
        belief_net=net,
        description="Prior allowing non-coherent scale-dependent laws.",
    )


def bayesnet_preset_modified_gravity(basis: LandscapeBasis) -> BayesNetPreset:
    couplings = {
        ("rotation_curve_anomaly", "modified_gravity_commitment"): -0.46,
        ("law_coherence", "modified_gravity_commitment"): 0.24,
        ("large_scale_structure_anomaly", "modified_gravity_commitment"): -0.22,
        ("modified_gravity_commitment", "dark_matter_commitment"): -0.2,
        ("modified_gravity_commitment", "scale_variant_laws_commitment"): -0.18,
    }
    pi = _diagonally_dominant_pi(basis, base_diag=1.1, couplings=couplings)
    mu = np.asarray(
        _vector_from_weights(
            basis.names,
            {
                "law_coherence": 0.3,
                "rotation_curve_anomaly": 0.8,
                "large_scale_structure_anomaly": 0.2,
                "dark_matter_commitment": -0.8,
                "scale_variant_laws_commitment": -0.7,
                "modified_gravity_commitment": 1.4,
            },
        )
    )
    h = _preset_h_from_mu(pi, mu)
    net = GaussianBeliefNet(Pi=pi, h=h, names=basis.names)
    return BayesNetPreset(
        name="modified_gravity",
        belief_net=net,
        description="Prior favoring MOND-like gravity modification over dark matter.",
    )


def utility_preset_precision_extreme(names: tuple[str, ...]) -> UtilityPreset:
    u = _vector_from_weights(
        names,
        {
            "law_coherence": 1.4,
            "rotation_curve_anomaly": 0.3,
            "large_scale_structure_anomaly": 0.3,
            "dark_matter_commitment": 0.6,
            "scale_variant_laws_commitment": -0.1,
            "modified_gravity_commitment": 0.2,
        },
    )
    return UtilityPreset(
        name="precision_extreme",
        intrinsic_utility=u,
        alpha=0.2,
        description="Rewards globally coherent high-precision commitments.",
    )


def utility_preset_intrinsic_extreme(names: tuple[str, ...]) -> UtilityPreset:
    u = _vector_from_weights(
        names,
        {
            "law_coherence": 0.7,
            "rotation_curve_anomaly": 0.2,
            "large_scale_structure_anomaly": 0.2,
            "dark_matter_commitment": 1.6,
            "scale_variant_laws_commitment": -1.1,
            "modified_gravity_commitment": -0.9,
        },
    )
    return UtilityPreset(
        name="intrinsic_extreme",
        intrinsic_utility=u,
        alpha=0.65,
        description="Strongly favors one explanatory path with high propagation.",
    )


def utility_preset_neutral(names: tuple[str, ...]) -> UtilityPreset:
    u = jnp.zeros((len(names),), dtype=jnp.float32)
    return UtilityPreset(
        name="neutral",
        intrinsic_utility=u,
        alpha=0.0,
        description="No intrinsic preference across basis commitments.",
    )


def _allocate_counts(n_agents: int, proportions: np.ndarray) -> np.ndarray:
    raw = n_agents * proportions
    counts = np.floor(raw).astype(np.int64)
    remainder = int(n_agents - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts


def network_preset_four_cluster_sbm(
    n_agents: int = 240,
    intra: float = 0.25,
    inter: float = 0.02,
    seed: int = 0,
) -> NetworkPreset:
    proportions = np.asarray([0.28, 0.24, 0.22, 0.26], dtype=np.float64)
    counts = _allocate_counts(n_agents, proportions)
    graph = community(counts.tolist(), intra=intra, inter=inter, seed=seed)
    membership = np.asarray(graph.membership, dtype=np.int64)
    return NetworkPreset(
        name="four_cluster_sbm",
        graph=graph,
        membership=membership,
        description="Four-community SBM for heterogeneous large populations.",
    )


def default_cluster_profiles() -> tuple[ClusterProfile, ...]:
    return (
        ClusterProfile(
            label="conservative_precision",
            proportion=0.28,
            precision_mean=0.55,
            precision_sd=0.22,
            utility_scale_mean=0.65,
            utility_scale_sd=0.35,
        ),
        ClusterProfile(
            label="utility_driven",
            proportion=0.24,
            precision_mean=0.1,
            precision_sd=0.28,
            utility_scale_mean=1.25,
            utility_scale_sd=0.4,
        ),
        ClusterProfile(
            label="skeptical_low_gain",
            proportion=0.22,
            precision_mean=-0.2,
            precision_sd=0.35,
            utility_scale_mean=-0.35,
            utility_scale_sd=0.5,
        ),
        ClusterProfile(
            label="volatile_frontier",
            proportion=0.26,
            precision_mean=0.35,
            precision_sd=0.55,
            utility_scale_mean=0.25,
            utility_scale_sd=0.9,
        ),
    )


def sample_cluster_scalings(
    n_agents: int,
    profiles: tuple[ClusterProfile, ...],
    seed: int = 0,
) -> ClusterSample:
    rng = np.random.default_rng(seed)
    proportions = np.asarray([p.proportion for p in profiles], dtype=np.float64)
    proportions = proportions / proportions.sum()
    counts = _allocate_counts(n_agents, proportions)

    precision_list = []
    utility_list = []
    cluster_ids = []
    for i, (profile, count) in enumerate(zip(profiles, counts)):
        if count <= 0:
            continue
        precision_draw = rng.lognormal(
            mean=profile.precision_mean,
            sigma=profile.precision_sd,
            size=int(count),
        )
        utility_draw = rng.normal(
            loc=profile.utility_scale_mean,
            scale=profile.utility_scale_sd,
            size=int(count),
        )
        utility_draw = np.clip(utility_draw, -4.0, 4.0)

        precision_list.append(precision_draw)
        utility_list.append(utility_draw)
        cluster_ids.append(np.full((int(count),), i, dtype=np.int32))

    precision_scale = np.concatenate(precision_list).astype(np.float32)
    utility_scale = np.concatenate(utility_list).astype(np.float32)
    cluster_id = np.concatenate(cluster_ids).astype(np.int32)

    perm = rng.permutation(n_agents)
    return ClusterSample(
        precision_scale=jnp.asarray(precision_scale[perm]),
        utility_scale=jnp.asarray(utility_scale[perm]),
        cluster_id=jnp.asarray(cluster_id[perm]),
    )


def apply_agent_scalings(intrinsic_utility: jax.Array, sample: ClusterSample) -> jax.Array:
    """Scale one intrinsic utility vector to per-agent utility vectors."""
    u = jnp.asarray(intrinsic_utility)
    scales = jnp.asarray(sample.utility_scale)
    if u.ndim != 1:
        raise ValueError(f"intrinsic_utility must be rank-1, got shape {u.shape}")
    return scales[:, None] * u[None, :]
