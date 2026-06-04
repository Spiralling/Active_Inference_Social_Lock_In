"""Run two deterministic landscape stages and save artifacts."""

from __future__ import annotations

import json
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.structural.landscape_examples import build_cosmology_extreme_landscape
from src.structural.plot import render_plot
from src.structural.step import fuse
from src.structural.world import fisher_deposit
from scripts.beta_calibration import BetaCalibration, estimate_beta_calibration


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
    return {
        0: 1.35 * dm,
        1: 1.35 * sv,
        2: 1.35 * mg,
        3: 0.45 * mixed,
    }


def _build_per_agent_utility(
    membership: np.ndarray,
    name_to_index: dict[str, int],
    preset_mu: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
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


def _run_stage(shared: SharedConfig, stage: StageConfig) -> tuple[Path, Path, list[Path], list[float]]:
    rng = np.random.default_rng(shared.seed)
    ex = build_cosmology_extreme_landscape(n_agents=shared.n_agents, seed=shared.seed)
    hypothesis_names, name_to_index, preset_pi, preset_h, preset_mu = _build_preset_arrays(ex)

    true_idx = name_to_index["dark_matter"]
    sv_idx = name_to_index["scale_variant_laws"]
    mg_idx = name_to_index["modified_gravity"]

    n = shared.n_agents
    k_h = len(hypothesis_names)
    d = preset_pi.shape[-1]

    precision_scale = np.asarray(ex.cluster_sample.precision_scale, dtype=np.float64)
    cluster_id = np.asarray(ex.cluster_sample.cluster_id, dtype=np.int64)
    membership = np.asarray(ex.network_preset.membership, dtype=np.int64)

    per_agent_u = _build_per_agent_utility(
        membership=membership,
        name_to_index=name_to_index,
        preset_mu=preset_mu,
        rng=rng,
    )

    winner_init = cluster_id % k_h
    Pi = precision_scale[:, None, None] * preset_pi[winner_init]
    h = precision_scale[:, None] * preset_h[winner_init]

    W = np.asarray(ex.network_preset.graph.trust_W(), dtype=np.float64)
    H = np.eye(d, dtype=np.float64)
    J_np, _ = fisher_deposit(H, np.zeros((d,), dtype=np.float64), stage.sigma_obs)
    J = np.asarray(J_np, dtype=np.float64)

    mu_dm = preset_mu[true_idx]
    mu_deltas = {
        "mg_minus_dm": preset_mu[mg_idx] - mu_dm,
        "sv_minus_dm": preset_mu[sv_idx] - mu_dm,
    }

    winner_tn = np.zeros((shared.n_steps, n), dtype=np.int64)
    Pi_t = np.zeros((shared.n_steps, n, d, d), dtype=np.float64)
    h_t = np.zeros((shared.n_steps, n, d), dtype=np.float64)
    log_evidence_tk = np.zeros((shared.n_steps, k_h), dtype=np.float64)
    edge_count_t = np.zeros((shared.n_steps,), dtype=np.int64)
    accepted_t = np.zeros((shared.n_steps,), dtype=np.float64)

    prev_winner = winner_init.copy()
    rho = float(shared.structure_pull)

    for t in range(shared.n_steps):
        Pi_fused_jax, h_fused_jax = fuse(Pi, h, W)
        Pi_fused = np.asarray(Pi_fused_jax, dtype=np.float64)
        h_fused = np.asarray(h_fused_jax, dtype=np.float64)

        mu_true_t = stage.true_mean_fn(t, mu_dm, mu_deltas)
        o = rng.normal(loc=mu_true_t[None, :], scale=stage.sigma_obs, size=(n, d))
        j_obs = o / (stage.sigma_obs**2)

        Pi_obs = Pi_fused + J[None, :, :]
        h_obs = h_fused + j_obs
        mu_obs = np.stack([np.linalg.solve(Pi_obs[i], h_obs[i]) for i in range(n)], axis=0)

        scores = np.zeros((n, k_h), dtype=np.float64)
        for k in range(k_h):
            delta = mu_obs - preset_mu[k][None, :]
            fit = -0.5 * np.einsum("ni,ij,nj->n", delta, preset_pi[k], delta)
            utility = stage.beta_u * np.einsum("ni,i->n", per_agent_u, preset_mu[k])
            scores[:, k] = fit + utility

        winner = np.argmax(scores, axis=1)
        Pi_target = precision_scale[:, None, None] * preset_pi[winner]
        h_target = precision_scale[:, None] * preset_h[winner]

        Pi = (1.0 - rho) * Pi_obs + rho * Pi_target
        h = (1.0 - rho) * h_obs + rho * h_target

        winner_tn[t] = winner
        Pi_t[t] = Pi
        h_t[t] = h
        log_evidence_tk[t] = scores.mean(axis=0)

        mean_pi = Pi.mean(axis=0)
        edge_count_t[t] = _edge_count_from_mean_pi(mean_pi, threshold=shared.edge_threshold)
        accepted_t[t] = float(np.mean(winner != prev_winner)) if t > 0 else 0.0
        prev_winner = winner

    edge_edit_delta_t = np.diff(edge_count_t, prepend=edge_count_t[0])

    comm_ids = np.unique(membership)
    n_comm = len(comm_ids)
    final_pi = Pi_t[-1]
    comm_mean_pi = np.zeros((n_comm, d, d), dtype=np.float64)
    for i, c in enumerate(comm_ids):
        comm_mean_pi[i] = final_pi[membership == c].mean(axis=0)

    community_distance_cc = np.zeros((n_comm, n_comm), dtype=np.float64)
    for i in range(n_comm):
        for j in range(n_comm):
            community_distance_cc[i, j] = np.linalg.norm(comm_mean_pi[i] - comm_mean_pi[j], ord="fro")

    run_dir = (
        ROOT
        / "results"
        / f"{stage.stage_tag}_seed{shared.seed}_T{shared.n_steps}_N{shared.n_agents}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    npz_path = run_dir / "simulation_arrays.npz"
    np.savez_compressed(
        npz_path,
        winner_tn=winner_tn,
        Pi_t=Pi_t,
        h_t=h_t,
        log_evidence_tk=log_evidence_tk,
        edge_count_t=edge_count_t,
        edge_edit_delta_t=edge_edit_delta_t,
        accepted_t=accepted_t,
        community_distance_cc=community_distance_cc,
        cluster_membership=membership,
        precision_scale=precision_scale,
    )

    figure_paths = [
        run_dir / "community_bayesnet_snapshots.png",
        run_dir / "edge_edit_timeline.png",
        run_dir / "log_evidence_race.png",
        run_dir / "mean_field_hypothesis_share.png",
        run_dir / "community_distance_heatmap.png",
    ]

    render_plot(
        "plot_community_bayesnet_snapshots",
        Pi_t=Pi_t,
        h_t=h_t,
        snapshot_indices=[0, shared.n_steps // 3, (2 * shared.n_steps) // 3, shared.n_steps - 1],
        show_graph=True,
        graph_only=True,
        save_path=figure_paths[0],
    )
    render_plot(
        "plot_edge_edit_timeline",
        edge_count_t=edge_count_t,
        edge_edit_delta_t=edge_edit_delta_t,
        accepted_t=accepted_t,
        save_path=figure_paths[1],
    )
    render_plot(
        "plot_log_evidence_race",
        log_evidence_tk=log_evidence_tk,
        hypothesis_names=hypothesis_names,
        save_path=figure_paths[2],
    )
    render_plot(
        "plot_mean_field_hypothesis_share",
        winner_tn=winner_tn,
        hypothesis_names=hypothesis_names,
        save_path=figure_paths[3],
    )
    render_plot(
        "plot_community_distance_heatmap",
        distance_cc=community_distance_cc,
        community_labels=[f"C{int(c)}" for c in comm_ids],
        save_path=figure_paths[4],
    )

    final_counts = np.bincount(winner_tn[-1], minlength=k_h)
    final_shares = (final_counts / float(n)).tolist()
    avg_switch_rate = float(accepted_t[1:].mean()) if shared.n_steps > 1 else 0.0

    summary = {
        "config": {
            "n_agents": shared.n_agents,
            "n_steps": shared.n_steps,
            "seed": shared.seed,
            "sigma_obs": stage.sigma_obs,
            "beta_u": stage.beta_u,
            "structure_pull": shared.structure_pull,
            "edge_threshold": shared.edge_threshold,
            "stage_tag": stage.stage_tag,
        },
        "hypothesis_names": hypothesis_names,
        "final_shares": final_shares,
        "final_edge_count": int(edge_count_t[-1]),
        "average_switch_rate": avg_switch_rate,
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return run_dir, summary_path, figure_paths, final_shares


def _print_stage_outputs(
    stage_name: str,
    run_dir: Path,
    summary_path: Path,
    figure_paths: list[Path],
    final_shares: list[float],
) -> None:
    print(f"{stage_name} run_dir: {run_dir.resolve()}")
    print(f"{stage_name} summary: {summary_path.resolve()}")
    for fig in figure_paths:
        print(f"{stage_name} figure: {fig.resolve()}")
    compact = ",".join(f"{x:.4f}" for x in final_shares)
    print(f"{stage_name} final_shares: [{compact}]")


def _print_beta_calibration(stage_name: str, c: BetaCalibration) -> None:
    print(
        f"{stage_name} beta calibration: "
        f"equal_t0={c.beta_equal_t0:.6f}, "
        f"equal_tfinal={c.beta_equal_tfinal:.6f}, "
        f"equal_mean={c.beta_equal_mean:.6f}, "
        f"utility_dominant={c.beta_utility_dominant:.6f}, "
        f"balanced={c.beta_balanced:.6f}, "
        f"fit_dominant={c.beta_fit_dominant:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-stageB",
        action="store_true",
        help="Run only Stage B (skip Stage A simulation)",
    )
    args = parser.parse_args()

    shared = SharedConfig()

    stage_a = StageConfig(
        stage_tag="landscape_stageA_utility_dominant",
        sigma_obs=0.45,
        beta_u=4.0,
        true_mean_fn=_true_mean_fixed,
    )
    stage_b = StageConfig(
        stage_tag="landscape_stageB_subtle_high_noise",
        sigma_obs=2.0,
        beta_u=1.0,
        true_mean_fn=_true_mean_subtle_drift,
    )

    _print_beta_calibration("StageA", estimate_beta_calibration(shared, stage_a))
    _print_beta_calibration("StageB", estimate_beta_calibration(shared, stage_b))

    if not args.only_stageB:
        run_a, summary_a, figs_a, shares_a = _run_stage(shared, stage_a)
        _print_stage_outputs("StageA", run_a, summary_a, figs_a, shares_a)

    run_b, summary_b, figs_b, shares_b = _run_stage(shared, stage_b)
    _print_stage_outputs("StageB", run_b, summary_b, figs_b, shares_b)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
