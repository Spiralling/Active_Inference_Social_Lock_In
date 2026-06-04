"""Run a deterministic multi-agent landscape simulation and save artifacts."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.structural.landscape_examples import build_cosmology_extreme_landscape
from src.structural.plot import render_plot
from src.structural.step import fuse
from src.structural.world import fisher_deposit


@dataclass(frozen=True)
class SimulationConfig:
    n_agents: int = 240
    n_steps: int = 80
    seed: int = 0
    sigma_obs: float = 0.45
    beta_u: float = 0.65
    structure_pull: float = 0.18
    edge_threshold: float = 0.18


def _prior_mean(pi: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.linalg.solve(pi, h)


def _edge_count_from_mean_pi(mean_pi: np.ndarray, threshold: float) -> int:
    d = mean_pi.shape[0]
    mask_upper_offdiag = np.triu(np.ones((d, d), dtype=bool), k=1)
    edge_mask = np.abs(mean_pi) > float(threshold)
    return int(np.sum(edge_mask & mask_upper_offdiag))


def main() -> int:
    cfg = SimulationConfig()
    rng = np.random.default_rng(cfg.seed)

    ex = build_cosmology_extreme_landscape(n_agents=cfg.n_agents, seed=cfg.seed)
    presets = ex.bayesnet_presets
    hypothesis_names = [p.name for p in presets]

    name_to_index = {name: i for i, name in enumerate(hypothesis_names)}
    true_idx = name_to_index["dark_matter"]

    preset_pi = np.stack([np.asarray(p.belief_net.Pi, dtype=np.float64) for p in presets], axis=0)
    preset_h = np.stack([np.asarray(p.belief_net.h, dtype=np.float64) for p in presets], axis=0)
    preset_mu = np.stack([_prior_mean(preset_pi[k], preset_h[k]) for k in range(len(presets))], axis=0)

    n = cfg.n_agents
    k_h = len(presets)
    d = preset_pi.shape[-1]

    precision_scale = np.asarray(ex.cluster_sample.precision_scale, dtype=np.float64)
    cluster_id = np.asarray(ex.cluster_sample.cluster_id, dtype=np.int64)
    per_agent_u = np.asarray(ex.per_agent_utility, dtype=np.float64)

    winner_init = cluster_id % k_h
    Pi = precision_scale[:, None, None] * preset_pi[winner_init]
    h = precision_scale[:, None] * preset_h[winner_init]

    W = np.asarray(ex.network_preset.graph.trust_W(), dtype=np.float64)

    H = np.eye(d, dtype=np.float64)
    J_np, _ = fisher_deposit(H, np.zeros((d,), dtype=np.float64), cfg.sigma_obs)
    J = np.asarray(J_np, dtype=np.float64)

    mu_true = preset_mu[true_idx]

    winner_tn = np.zeros((cfg.n_steps, n), dtype=np.int64)
    Pi_t = np.zeros((cfg.n_steps, n, d, d), dtype=np.float64)
    h_t = np.zeros((cfg.n_steps, n, d), dtype=np.float64)
    log_evidence_tk = np.zeros((cfg.n_steps, k_h), dtype=np.float64)
    edge_count_t = np.zeros((cfg.n_steps,), dtype=np.int64)
    accepted_t = np.zeros((cfg.n_steps,), dtype=np.float64)

    prev_winner = winner_init.copy()
    rho = float(cfg.structure_pull)

    for t in range(cfg.n_steps):
        Pi_fused_jax, h_fused_jax = fuse(Pi, h, W)
        Pi_fused = np.asarray(Pi_fused_jax, dtype=np.float64)
        h_fused = np.asarray(h_fused_jax, dtype=np.float64)

        o = rng.normal(loc=mu_true[None, :], scale=cfg.sigma_obs, size=(n, d))
        j_obs = o / (cfg.sigma_obs ** 2)

        Pi_obs = Pi_fused + J[None, :, :]
        h_obs = h_fused + j_obs

        mu_obs = np.stack([np.linalg.solve(Pi_obs[i], h_obs[i]) for i in range(n)], axis=0)

        scores = np.zeros((n, k_h), dtype=np.float64)
        for k in range(k_h):
            delta = mu_obs - preset_mu[k][None, :]
            fit = -0.5 * np.einsum("ni,ij,nj->n", delta, preset_pi[k], delta)
            utility = cfg.beta_u * np.einsum("ni,i->n", per_agent_u, preset_mu[k])
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
        edge_count_t[t] = _edge_count_from_mean_pi(mean_pi, threshold=cfg.edge_threshold)
        accepted_t[t] = float(np.mean(winner != prev_winner)) if t > 0 else 0.0
        prev_winner = winner

    edge_edit_delta_t = np.diff(edge_count_t, prepend=edge_count_t[0])

    membership = np.asarray(ex.network_preset.membership, dtype=np.int64)
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

    run_dir = ROOT / "results" / f"landscape_run_seed{cfg.seed}_T{cfg.n_steps}_N{cfg.n_agents}"
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
        snapshot_indices=[0, cfg.n_steps // 3, (2 * cfg.n_steps) // 3, cfg.n_steps - 1],
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
    avg_switch_rate = float(accepted_t[1:].mean()) if cfg.n_steps > 1 else 0.0

    summary = {
        "config": {
            "n_agents": cfg.n_agents,
            "n_steps": cfg.n_steps,
            "seed": cfg.seed,
            "sigma_obs": cfg.sigma_obs,
            "beta_u": cfg.beta_u,
            "structure_pull": cfg.structure_pull,
            "edge_threshold": cfg.edge_threshold,
        },
        "hypothesis_names": hypothesis_names,
        "final_shares": final_shares,
        "final_edge_count": int(edge_count_t[-1]),
        "average_switch_rate": avg_switch_rate,
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(str(run_dir.resolve()))
    print(str(npz_path.resolve()))
    print(str(summary_path.resolve()))
    for fig_path in figure_paths:
        print(str(fig_path.resolve()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
