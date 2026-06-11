"""Landscape model -- shared builders for the cosmology-landscape simulations.

The landscape experiments run a multi-agent simulation over the ``landscape_examples`` Bayes-net
presets (dark_matter / scale_variant_laws / modified_gravity), where each agent fuses precision on a
trust graph, observes a (possibly drifting) truth, and switches hypotheses by a fit + ``beta_u``·utility
score. This module is the **pure compute library** shared by all three landscape experiments
(``landscape_simulation``, ``landscape_two_stage``, ``beta_calibration``): the configs, the preset /
utility builders, the two ground-truth schedules, and the ``estimate_beta_calibration`` diagnostic.

Holding ``SharedConfig`` / ``StageConfig`` / ``_true_mean_*`` *and* ``BetaCalibration`` /
``estimate_beta_calibration`` here dissolves the former circular import between
``run_landscape_two_stage`` and ``beta_calibration`` (each imported the other). No matplotlib, no
file I/O, no ``main`` -- the per-stage figures, the saved arrays, and the printouts live in the
``experiments/landscape_*`` modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.structural.landscape_examples import build_cosmology_extreme_landscape
from src.structural.step import fuse
from src.structural.world import fisher_deposit

__all__ = [
    "SharedConfig", "StageConfig", "BetaCalibration",
    "estimate_beta_calibration", "_true_mean_fixed", "_true_mean_subtle_drift",
    "_prior_mean", "_edge_count_from_mean_pi", "_build_preset_arrays",
    "_community_utility_prototypes", "_build_per_agent_utility", "_score_spreads", "_safe_ratio",
]


@dataclass(frozen=True)
class SharedConfig:
    n_agents: int = 240
    n_steps: int = 80
    seed: int = 0
    structure_pull: float = 0.18
    edge_threshold: float = 0.18


@dataclass(frozen=True)
class StageConfig:
    stage_tag: str
    sigma_obs: float
    beta_u: float
    true_mean_fn: Callable[[int, np.ndarray, dict[str, np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class BetaCalibration:
    beta_equal_t0: float
    beta_equal_tfinal: float
    beta_equal_mean: float
    beta_utility_dominant: float
    beta_balanced: float
    beta_fit_dominant: float


def _prior_mean(pi: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.linalg.solve(pi, h)


def _edge_count_from_mean_pi(mean_pi: np.ndarray, threshold: float) -> int:
    d = mean_pi.shape[0]
    mask_upper_offdiag = np.triu(np.ones((d, d), dtype=bool), k=1)
    edge_mask = np.abs(mean_pi) > float(threshold)
    return int(np.sum(edge_mask & mask_upper_offdiag))


def _build_preset_arrays(ex) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    presets = ex.bayesnet_presets
    hypothesis_names = [p.name for p in presets]
    name_to_index = {name: i for i, name in enumerate(hypothesis_names)}
    preset_pi = np.stack([np.asarray(p.belief_net.Pi, dtype=np.float64) for p in presets], axis=0)
    preset_h = np.stack([np.asarray(p.belief_net.h, dtype=np.float64) for p in presets], axis=0)
    preset_mu = np.stack([_prior_mean(preset_pi[k], preset_h[k]) for k in range(len(presets))], axis=0)
    return hypothesis_names, name_to_index, preset_pi, preset_h, preset_mu


def _community_utility_prototypes(name_to_index: dict[str, int], preset_mu: np.ndarray) -> dict[int, np.ndarray]:
    dm = preset_mu[name_to_index["dark_matter"]]
    sv = preset_mu[name_to_index["scale_variant_laws"]]
    mg = preset_mu[name_to_index["modified_gravity"]]
    mixed = (dm + sv + mg) / 3.0
    return {0: 1.35 * dm, 1: 1.35 * sv, 2: 1.35 * mg, 3: 0.45 * mixed}


def _build_per_agent_utility(membership: np.ndarray, name_to_index: dict[str, int],
                             preset_mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    prototypes = _community_utility_prototypes(name_to_index=name_to_index, preset_mu=preset_mu)
    n_agents = int(membership.shape[0])
    d = int(preset_mu.shape[1])
    out = np.zeros((n_agents, d), dtype=np.float64)
    jitter_scale = 0.06
    for c in np.unique(membership):
        proto = prototypes.get(int(c), prototypes[3])
        idx = membership == c
        noise = rng.normal(loc=0.0, scale=jitter_scale, size=(int(np.sum(idx)), d))
        out[idx] = proto[None, :] + noise
    return out


def _true_mean_fixed(_: int, base_mu: np.ndarray, __: dict[str, np.ndarray]) -> np.ndarray:
    return base_mu


def _true_mean_subtle_drift(t: int, base_mu: np.ndarray, deltas: dict[str, np.ndarray]) -> np.ndarray:
    phase = (2.0 * np.pi * float(t)) / 80.0
    w_mg = 0.10 * np.sin(phase)
    w_sv = 0.09 * np.cos(phase + 0.7)
    return base_mu + w_mg * deltas["mg_minus_dm"] + w_sv * deltas["sv_minus_dm"]


def _safe_ratio(numer: float, denom: float) -> float:
    eps = 1e-12
    return float(numer / max(abs(denom), eps))


def _score_spreads(mu_obs: np.ndarray, preset_pi: np.ndarray, preset_mu: np.ndarray,
                   per_agent_u: np.ndarray) -> tuple[float, float]:
    n = int(mu_obs.shape[0])
    k_h = int(preset_mu.shape[0])
    fit_scores = np.zeros((n, k_h), dtype=np.float64)
    utility_scores = np.zeros((n, k_h), dtype=np.float64)
    for k in range(k_h):
        delta = mu_obs - preset_mu[k][None, :]
        fit_scores[:, k] = -0.5 * np.einsum("ni,ij,nj->n", delta, preset_pi[k], delta)
        utility_scores[:, k] = np.einsum("ni,i->n", per_agent_u, preset_mu[k])
    fit_spread_mean = float(np.mean(np.ptp(fit_scores, axis=1)))
    utility_spread_mean_abs = float(np.mean(np.abs(np.ptp(utility_scores, axis=1))))
    return fit_spread_mean, utility_spread_mean_abs


def estimate_beta_calibration(shared_cfg, stage_cfg) -> BetaCalibration:
    """Estimate the beta scaling where utility-vs-fit influence is comparable. Uses the same
    simulation ingredients as the two-stage run and computes fit/utility spread statistics at t=0
    and t=final."""
    n_agents = int(shared_cfg.n_agents)
    n_steps = int(shared_cfg.n_steps)
    seed = int(shared_cfg.seed)
    structure_pull = float(shared_cfg.structure_pull)
    sigma_obs = float(stage_cfg.sigma_obs)
    true_mean_fn: Callable[[int, np.ndarray, dict[str, np.ndarray]], np.ndarray] = stage_cfg.true_mean_fn

    rng = np.random.default_rng(seed)
    ex = build_cosmology_extreme_landscape(n_agents=n_agents, seed=seed)
    _names, name_to_index, preset_pi, preset_h, preset_mu = _build_preset_arrays(ex)

    true_idx = name_to_index["dark_matter"]
    sv_idx = name_to_index["scale_variant_laws"]
    mg_idx = name_to_index["modified_gravity"]

    precision_scale = np.asarray(ex.cluster_sample.precision_scale, dtype=np.float64)
    cluster_id = np.asarray(ex.cluster_sample.cluster_id, dtype=np.int64)
    membership = np.asarray(ex.network_preset.membership, dtype=np.int64)
    per_agent_u = _build_per_agent_utility(membership=membership, name_to_index=name_to_index,
                                           preset_mu=preset_mu, rng=rng)

    winner_init = cluster_id % preset_mu.shape[0]
    pi_agent = precision_scale[:, None, None] * preset_pi[winner_init]
    h_agent = precision_scale[:, None] * preset_h[winner_init]

    d = int(preset_pi.shape[-1])
    W = np.asarray(ex.network_preset.graph.trust_W(), dtype=np.float64)
    H = np.eye(d, dtype=np.float64)
    J_np, _ = fisher_deposit(H, np.zeros((d,), dtype=np.float64), sigma_obs)
    J = np.asarray(J_np, dtype=np.float64)

    mu_dm = preset_mu[true_idx]
    mu_deltas = {"mg_minus_dm": preset_mu[mg_idx] - mu_dm,
                 "sv_minus_dm": preset_mu[sv_idx] - mu_dm}

    fit_spread_t0 = utility_spread_t0 = fit_spread_tfinal = utility_spread_tfinal = 0.0

    for t in range(n_steps):
        pi_fused_jax, h_fused_jax = fuse(pi_agent, h_agent, W)
        pi_fused = np.asarray(pi_fused_jax, dtype=np.float64)
        h_fused = np.asarray(h_fused_jax, dtype=np.float64)

        mu_true_t = true_mean_fn(t, mu_dm, mu_deltas)
        o = rng.normal(loc=mu_true_t[None, :], scale=sigma_obs, size=(n_agents, d))
        j_obs = o / (sigma_obs ** 2)

        pi_obs = pi_fused + J[None, :, :]
        h_obs = h_fused + j_obs
        mu_obs = np.stack([np.linalg.solve(pi_obs[i], h_obs[i]) for i in range(n_agents)], axis=0)

        fit_spread, utility_spread = _score_spreads(mu_obs=mu_obs, preset_pi=preset_pi,
                                                    preset_mu=preset_mu, per_agent_u=per_agent_u)
        if t == 0:
            fit_spread_t0 = fit_spread
            utility_spread_t0 = utility_spread
        if t == n_steps - 1:
            fit_spread_tfinal = fit_spread
            utility_spread_tfinal = utility_spread

        # Keep the update path aligned with the two-stage scoring model.
        k_h = int(preset_mu.shape[0])
        scores = np.zeros((n_agents, k_h), dtype=np.float64)
        for k in range(k_h):
            delta = mu_obs - preset_mu[k][None, :]
            fit = -0.5 * np.einsum("ni,ij,nj->n", delta, preset_pi[k], delta)
            utility = float(stage_cfg.beta_u) * np.einsum("ni,i->n", per_agent_u, preset_mu[k])
            scores[:, k] = fit + utility

        winner = np.argmax(scores, axis=1)
        pi_target = precision_scale[:, None, None] * preset_pi[winner]
        h_target = precision_scale[:, None] * preset_h[winner]
        pi_agent = (1.0 - structure_pull) * pi_obs + structure_pull * pi_target
        h_agent = (1.0 - structure_pull) * h_obs + structure_pull * h_target

    beta_equal_t0 = _safe_ratio(fit_spread_t0, utility_spread_t0)
    beta_equal_tfinal = _safe_ratio(fit_spread_tfinal, utility_spread_tfinal)
    beta_equal_mean = 0.5 * (beta_equal_t0 + beta_equal_tfinal)
    return BetaCalibration(
        beta_equal_t0=beta_equal_t0, beta_equal_tfinal=beta_equal_tfinal,
        beta_equal_mean=beta_equal_mean, beta_utility_dominant=3.0 * beta_equal_mean,
        beta_balanced=1.0 * beta_equal_mean, beta_fit_dominant=0.3 * beta_equal_mean)
