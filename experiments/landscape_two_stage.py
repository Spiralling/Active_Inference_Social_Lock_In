"""Two deterministic landscape stages: a utility-dominant fixed-truth stage and a subtle-drift,
high-noise stage -- the same agent population under two different (sigma_obs, beta_u, truth) regimes.

Stage A (utility-dominant, fixed dark-matter truth) vs Stage B (subtle sinusoidal drift, high noise):
each runs the landscape simulation and, per the calibration diagnostic, reports the beta scaling at
which fit and utility influence are comparable. Both stages write into one ``results/landscape_two_stage/``
with ``stageA_`` / ``stageB_`` prefixed arrays and figures.

Shared builders + ``estimate_beta_calibration`` come from ``src.structural.models.landscape`` (which
also dissolves the former two_stage<->beta_calibration circular import); this module owns the stage
run loop, the combined arrays, and the registry spec.
"""
from __future__ import annotations

import json

import numpy as np

from src.structural.landscape_examples import build_cosmology_extreme_landscape
from src.structural.plot import render_plot
from src.structural.step import fuse
from src.structural.world import fisher_deposit
from src.structural.models.landscape import (
    SharedConfig, StageConfig, estimate_beta_calibration,
    _build_preset_arrays, _build_per_agent_utility, _edge_count_from_mean_pi,
    _true_mean_fixed, _true_mean_subtle_drift)
from experiments.registry import ExperimentSpec, register

TRUE_MEAN_FNS = {"fixed": _true_mean_fixed, "subtle_drift": _true_mean_subtle_drift}
ARRAY_KEYS = ("winner_tn", "Pi_t", "h_t", "log_evidence_tk", "edge_count_t",
              "edge_edit_delta_t", "accepted_t", "community_distance_cc",
              "cluster_membership", "precision_scale")


def _run_stage(shared: SharedConfig, stage: StageConfig, out_dir, prefix: str) -> tuple[dict, dict, list]:
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
    per_agent_u = _build_per_agent_utility(membership=membership, name_to_index=name_to_index,
                                           preset_mu=preset_mu, rng=rng)

    winner_init = cluster_id % k_h
    Pi = precision_scale[:, None, None] * preset_pi[winner_init]
    h = precision_scale[:, None] * preset_h[winner_init]

    W = np.asarray(ex.network_preset.graph.trust_W(), dtype=np.float64)
    H = np.eye(d, dtype=np.float64)
    J_np, _ = fisher_deposit(H, np.zeros((d,), dtype=np.float64), stage.sigma_obs)
    J = np.asarray(J_np, dtype=np.float64)

    mu_dm = preset_mu[true_idx]
    mu_deltas = {"mg_minus_dm": preset_mu[mg_idx] - mu_dm, "sv_minus_dm": preset_mu[sv_idx] - mu_dm}

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
        j_obs = o / (stage.sigma_obs ** 2)

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

    arrays = {"winner_tn": winner_tn, "Pi_t": Pi_t, "h_t": h_t, "log_evidence_tk": log_evidence_tk,
              "edge_count_t": edge_count_t, "edge_edit_delta_t": edge_edit_delta_t,
              "accepted_t": accepted_t, "community_distance_cc": community_distance_cc,
              "cluster_membership": membership, "precision_scale": precision_scale}

    figure_paths = [out_dir / f"{prefix}_{nm}" for nm in (
        "community_bayesnet_snapshots.png", "edge_edit_timeline.png", "log_evidence_race.png",
        "mean_field_hypothesis_share.png", "community_distance_heatmap.png")]
    render_plot("plot_community_bayesnet_snapshots", Pi_t=Pi_t, h_t=h_t,
                snapshot_indices=[0, shared.n_steps // 3, (2 * shared.n_steps) // 3, shared.n_steps - 1],
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
    avg_switch_rate = float(accepted_t[1:].mean()) if shared.n_steps > 1 else 0.0
    stage_summary = {
        "config": {"n_agents": shared.n_agents, "n_steps": shared.n_steps, "seed": shared.seed,
                   "sigma_obs": stage.sigma_obs, "beta_u": stage.beta_u,
                   "structure_pull": shared.structure_pull, "edge_threshold": shared.edge_threshold,
                   "stage_tag": stage.stage_tag},
        "hypothesis_names": hypothesis_names, "final_shares": final_shares,
        "final_edge_count": int(edge_count_t[-1]), "average_switch_rate": avg_switch_rate,
    }
    return arrays, stage_summary, final_shares


def run(out_dir, params: dict) -> None:
    shared = SharedConfig(n_agents=params["N_AGENTS"], n_steps=params["N_STEPS"], seed=params["SEED"],
                          structure_pull=params["STRUCTURE_PULL"], edge_threshold=params["EDGE_THRESHOLD"])

    combined = {}
    summaries = {}
    for sp in params["STAGES"]:
        stage = StageConfig(stage_tag=sp["tag"], sigma_obs=sp["sigma_obs"], beta_u=sp["beta_u"],
                            true_mean_fn=TRUE_MEAN_FNS[sp["true_mean"]])
        cal = estimate_beta_calibration(shared, stage)
        print(f"{sp['prefix']} beta calibration: equal_mean={cal.beta_equal_mean:.6f} "
              f"utility_dominant={cal.beta_utility_dominant:.6f} balanced={cal.beta_balanced:.6f} "
              f"fit_dominant={cal.beta_fit_dominant:.6f}")
        arrays, stage_summary, shares = _run_stage(shared, stage, out_dir, sp["prefix"])
        for k in ARRAY_KEYS:
            combined[f"{sp['prefix']}_{k}"] = arrays[k]
        summaries[sp["prefix"]] = {**stage_summary,
                                   "beta_calibration": {"equal_mean": cal.beta_equal_mean,
                                                        "utility_dominant": cal.beta_utility_dominant,
                                                        "balanced": cal.beta_balanced,
                                                        "fit_dominant": cal.beta_fit_dominant}}
        print(f"{sp['prefix']} final_shares: {[round(x, 4) for x in shares]}")

    np.savez_compressed(out_dir / "simulation_arrays.npz", **combined)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nsaved combined arrays / figures / summary to {out_dir}")


register(ExperimentSpec(
    model="landscape",
    name="landscape_two_stage",
    description="Two landscape stages -- utility-dominant fixed-truth vs subtle-drift high-noise -- "
                "the same population under two (sigma_obs, beta_u, truth) regimes, with the "
                "fit-vs-utility beta calibration reported per stage.",
    run=run,
    out_dir="landscape_two_stage",
    params=dict(N_AGENTS=240, N_STEPS=80, SEED=0, STRUCTURE_PULL=0.18, EDGE_THRESHOLD=0.18,
                STAGES=[dict(tag="landscape_stageA_utility_dominant", prefix="stageA",
                             sigma_obs=0.45, beta_u=4.0, true_mean="fixed"),
                        dict(tag="landscape_stageB_subtle_high_noise", prefix="stageB",
                             sigma_obs=2.0, beta_u=1.0, true_mean="subtle_drift")]),
    seeds=(0,),
    consumes=dict(figures=["stageA_*", "stageB_*"]),
))
