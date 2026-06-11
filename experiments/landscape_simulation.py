"""Deterministic multi-agent landscape simulation: who wins on the cosmology-extreme landscape.

N agents (heterogeneous precision, four community-utility prototypes) fuse precision on a trust
graph, each step observe a fixed dark-matter truth, and switch hypotheses by a fit + ``beta_u``·utility
score with a structure-pull toward the winning preset. Saves the winner / precision / log-evidence
trajectories, the per-community final-net distances, and five diagnostic figures.

Shared builders come from ``src.structural.models.landscape``; this module owns the single-stage run
loop, the saved arrays, and the registry spec.
"""
from __future__ import annotations

import json

import numpy as np

from src.structural.landscape_examples import build_cosmology_extreme_landscape
from src.structural.plot import render_plot
from src.structural.step import fuse
from src.structural.world import fisher_deposit
from src.structural.models.landscape import _prior_mean, _edge_count_from_mean_pi
from experiments.registry import ExperimentSpec, register


def run(out_dir, params: dict) -> None:
    n_agents = params["N_AGENTS"]; n_steps = params["N_STEPS"]; seed = params["SEED"]
    sigma_obs = params["SIGMA_OBS"]; beta_u = params["BETA_U"]
    structure_pull = params["STRUCTURE_PULL"]; edge_threshold = params["EDGE_THRESHOLD"]

    rng = np.random.default_rng(seed)
    ex = build_cosmology_extreme_landscape(n_agents=n_agents, seed=seed)
    presets = ex.bayesnet_presets
    hypothesis_names = [p.name for p in presets]
    name_to_index = {name: i for i, name in enumerate(hypothesis_names)}
    true_idx = name_to_index["dark_matter"]

    preset_pi = np.stack([np.asarray(p.belief_net.Pi, dtype=np.float64) for p in presets], axis=0)
    preset_h = np.stack([np.asarray(p.belief_net.h, dtype=np.float64) for p in presets], axis=0)
    preset_mu = np.stack([_prior_mean(preset_pi[k], preset_h[k]) for k in range(len(presets))], axis=0)

    n = n_agents
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
    J_np, _ = fisher_deposit(H, np.zeros((d,), dtype=np.float64), sigma_obs)
    J = np.asarray(J_np, dtype=np.float64)
    mu_true = preset_mu[true_idx]

    winner_tn = np.zeros((n_steps, n), dtype=np.int64)
    Pi_t = np.zeros((n_steps, n, d, d), dtype=np.float64)
    h_t = np.zeros((n_steps, n, d), dtype=np.float64)
    log_evidence_tk = np.zeros((n_steps, k_h), dtype=np.float64)
    edge_count_t = np.zeros((n_steps,), dtype=np.int64)
    accepted_t = np.zeros((n_steps,), dtype=np.float64)

    prev_winner = winner_init.copy()
    rho = float(structure_pull)

    for t in range(n_steps):
        Pi_fused_jax, h_fused_jax = fuse(Pi, h, W)
        Pi_fused = np.asarray(Pi_fused_jax, dtype=np.float64)
        h_fused = np.asarray(h_fused_jax, dtype=np.float64)

        o = rng.normal(loc=mu_true[None, :], scale=sigma_obs, size=(n, d))
        j_obs = o / (sigma_obs ** 2)

        Pi_obs = Pi_fused + J[None, :, :]
        h_obs = h_fused + j_obs
        mu_obs = np.stack([np.linalg.solve(Pi_obs[i], h_obs[i]) for i in range(n)], axis=0)

        scores = np.zeros((n, k_h), dtype=np.float64)
        for k in range(k_h):
            delta = mu_obs - preset_mu[k][None, :]
            fit = -0.5 * np.einsum("ni,ij,nj->n", delta, preset_pi[k], delta)
            utility = beta_u * np.einsum("ni,i->n", per_agent_u, preset_mu[k])
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
        edge_count_t[t] = _edge_count_from_mean_pi(mean_pi, threshold=edge_threshold)
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

    np.savez_compressed(
        out_dir / "simulation_arrays.npz",
        winner_tn=winner_tn, Pi_t=Pi_t, h_t=h_t, log_evidence_tk=log_evidence_tk,
        edge_count_t=edge_count_t, edge_edit_delta_t=edge_edit_delta_t, accepted_t=accepted_t,
        community_distance_cc=community_distance_cc, cluster_membership=membership,
        precision_scale=precision_scale,
    )

    figure_paths = [out_dir / nm for nm in (
        "community_bayesnet_snapshots.png", "edge_edit_timeline.png", "log_evidence_race.png",
        "mean_field_hypothesis_share.png", "community_distance_heatmap.png")]
    render_plot("plot_community_bayesnet_snapshots", Pi_t=Pi_t, h_t=h_t,
                snapshot_indices=[0, n_steps // 3, (2 * n_steps) // 3, n_steps - 1],
                show_graph=True, graph_only=True, save_path=figure_paths[0])
    render_plot("plot_edge_edit_timeline", edge_count_t=edge_count_t,
                edge_edit_delta_t=edge_edit_delta_t, accepted_t=accepted_t, save_path=figure_paths[1])
    render_plot("plot_log_evidence_race", log_evidence_tk=log_evidence_tk,
                hypothesis_names=hypothesis_names, save_path=figure_paths[2])
    render_plot("plot_mean_field_hypothesis_share", winner_tn=winner_tn,
                hypothesis_names=hypothesis_names, save_path=figure_paths[3])
    render_plot("plot_community_distance_heatmap", distance_cc=community_distance_cc,
                community_labels=[f"C{int(c)}" for c in comm_ids], save_path=figure_paths[4])

    final_counts = np.bincount(winner_tn[-1], minlength=k_h)
    final_shares = (final_counts / float(n)).tolist()
    avg_switch_rate = float(accepted_t[1:].mean()) if n_steps > 1 else 0.0

    summary = {
        "config": {"n_agents": n_agents, "n_steps": n_steps, "seed": seed, "sigma_obs": sigma_obs,
                   "beta_u": beta_u, "structure_pull": structure_pull, "edge_threshold": edge_threshold},
        "hypothesis_names": hypothesis_names, "final_shares": final_shares,
        "final_edge_count": int(edge_count_t[-1]), "average_switch_rate": avg_switch_rate,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"saved arrays / figures / summary to {out_dir}")
    print(f"HEADLINE: final hypothesis shares {[round(x, 3) for x in final_shares]} "
          f"({hypothesis_names}); avg switch rate {avg_switch_rate:.3f}.")


register(ExperimentSpec(
    model="landscape",
    name="landscape_simulation",
    description="Deterministic multi-agent landscape simulation on the cosmology-extreme presets: "
                "fit+utility hypothesis switching with structure-pull; winner/precision trajectories "
                "and per-community final-net distances.",
    run=run,
    out_dir="landscape_simulation",
    params=dict(N_AGENTS=240, N_STEPS=80, SEED=0, SIGMA_OBS=0.45, BETA_U=0.65,
                STRUCTURE_PULL=0.18, EDGE_THRESHOLD=0.18),
    seeds=(0,),
    consumes=dict(figures=["community_bayesnet_snapshots", "edge_edit_timeline", "log_evidence_race",
                           "mean_field_hypothesis_share", "community_distance_heatmap"]),
))
