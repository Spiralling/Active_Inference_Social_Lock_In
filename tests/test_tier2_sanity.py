"""Tier-2 semantic sanity tests.

These five behavioural checks map to the plan's Tier-2 table. They are
non-numerical guarantees of qualitative behaviour: the model has to do the
right thing *in principle* before sweeps are meaningful.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import (
    ModelConfig, NetworkConfig, PolicyConfig, ResourceConfig, TrustConfig,
    UtilityConfig, build,
)
from src import inference, policy, resource, trust, utility


# ----------------------------------------------------------------------
# #12: W concentrates on belief-overlap neighbours.
# ----------------------------------------------------------------------


def test_tier2_12_W_concentrates_on_belief_overlap():
    """Trust learning increases the weight on neighbours whose private
    posteriors better predict an agent's observations. Across a planted
    belief asymmetry (mu split into two camps), the belief-aligned-neighbour
    average W exceeds the misaligned-neighbour average W."""
    N = 20
    cfg = ModelConfig(
        n_agents=N, seed=2,
        network=NetworkConfig(kind="watts_strogatz", mean_degree=8,
                              rewiring_p=0.3),
        policy=PolicyConfig(objective="eig_minus_cost"),
    )
    pop = build(cfg)
    # Plant two camps with sharply separated mu and high confidence.
    cluster = jnp.arange(N) >= N // 2
    mu_new = jnp.where(cluster, 2.0, -2.0)
    tau_new = jnp.full((N,), 5.0)
    pop = pop.__class__(
        cfg=pop.cfg, A_adj=pop.A_adj, A_self_adj=pop.A_self_adj,
        mu=mu_new, tau=tau_new, alpha=pop.alpha, beta=pop.beta,
        gamma=pop.gamma, r=pop.r, key=pop.key,
    )
    for t in range(50):
        pop, _ = pop.step(t)

    W = np.asarray(resource.flow_from_trust(pop.gamma))
    cluster_np = np.asarray(cluster)
    same = cluster_np[:, None] == cluster_np[None, :]
    A_mask = np.asarray(pop.A_self_adj) > 0
    intra_W = W[same & A_mask].mean()
    inter_W = W[(~same) & A_mask].mean() if ((~same) & A_mask).any() else 0.0
    assert intra_W > inter_W, (
        f"Trust did not concentrate on aligned neighbours: "
        f"intra-cluster mean W={intra_W:.4f}, inter-cluster mean W={inter_W:.4f}"
    )


# ----------------------------------------------------------------------
# #13: eta_i tracks predictive accuracy.
# ----------------------------------------------------------------------


def test_tier2_13_eta_tracks_predictive_accuracy():
    """The agent with smallest cumulative self-surprisal accumulates the
    largest aggregate-incoming-trust eta."""
    N = 12
    cfg = ModelConfig(
        n_agents=N, seed=4,
        network=NetworkConfig(kind="watts_strogatz", mean_degree=6),
        policy=PolicyConfig(objective="eig_minus_cost"),
    )
    pop = build(cfg)
    # Agent 0: pre-loaded with mu = theta_star_post = 1.0 (the "correct" one
    # after the schedule shift). High tau so it dominates predictively.
    # Agent 1: dead wrong (mu = -2) with the same tau.
    mu_new = pop.mu.at[0].set(1.0).at[1].set(-2.0)
    tau_new = pop.tau.at[0].set(50.0).at[1].set(50.0)
    pop = pop.__class__(
        cfg=pop.cfg, A_adj=pop.A_adj, A_self_adj=pop.A_self_adj,
        mu=mu_new, tau=tau_new, alpha=pop.alpha, beta=pop.beta,
        gamma=pop.gamma, r=pop.r, key=pop.key,
    )

    for t in range(80):
        pop, _ = pop.step(t)

    eta = np.asarray(resource.inflow_share(pop.gamma))
    # Agent 0 (correct) gets more aggregate-incoming-trust than Agent 1 (wrong)
    assert eta[0] > eta[1], f"eta_correct={eta[0]:.4f} not > eta_wrong={eta[1]:.4f}"


# ----------------------------------------------------------------------
# #14: Information gain is instrumental — μ-aligned agents see η rise.
# ----------------------------------------------------------------------


def test_tier2_14_info_gain_is_instrumental():
    """An agent planted with mu = theta_star_post and high tau predicts the
    post-shift observations well; its aggregate-incoming-trust eta grows
    relative to an agent planted with the wrong mu and high tau."""
    N = 10
    cfg = ModelConfig(
        n_agents=N, seed=6,
        network=NetworkConfig(kind="watts_strogatz", mean_degree=8,
                              rewiring_p=0.3),
        policy=PolicyConfig(objective="eig_minus_cost"),
    )
    pop = build(cfg)
    # Agent 0: correct posterior (mu = theta_star_post = 1.0).
    # Agent 1: dead wrong (mu = -3).
    mu_new = pop.mu.at[0].set(1.0).at[1].set(-3.0)
    tau_new = pop.tau.at[0].set(20.0).at[1].set(20.0)
    pop = pop.__class__(
        cfg=pop.cfg, A_adj=pop.A_adj, A_self_adj=pop.A_self_adj,
        mu=mu_new, tau=tau_new, alpha=pop.alpha, beta=pop.beta,
        gamma=pop.gamma, r=pop.r, key=pop.key,
    )
    # Step past the schedule shift at t=50.
    for t in range(100):
        pop, _ = pop.step(t)

    eta = np.asarray(resource.inflow_share(pop.gamma))
    assert eta[0] > eta[1], (
        f"Correct agent didn't accumulate more eta than wrong agent: "
        f"eta_correct={eta[0]:.4f}, eta_wrong={eta[1]:.4f}"
    )


# ----------------------------------------------------------------------
# #15: Resource buffer enables exploration.
# ----------------------------------------------------------------------


def test_tier2_15_resource_buffer_enables_exploration():
    """Rich agent (high r) chooses higher-h1 x more often than poor agent
    (low r), because c(x; r) barrier is far from biting."""
    N = 2
    cfg = ModelConfig(
        n_agents=N, seed=11,
        network=NetworkConfig(kind="watts_strogatz", mean_degree=1),
        policy=PolicyConfig(objective="resource_gain",
                            x_grid=(0.1, 1.0, 3.0), beta_exp=2.0),
        resource=ResourceConfig(R_in=0.0, alpha_flow=0.0, delta_decay=0.0,
                                c0=1.0, r0=1.0, r_min=0.0),
        utility=UtilityConfig(lambda_mc=0.0),
    )
    pop = build(cfg)
    # Give agent 0 high resource, agent 1 low resource.
    r_new = jnp.asarray([5.0, 0.3])
    pop = pop.__class__(
        cfg=pop.cfg, A_adj=pop.A_adj, A_self_adj=pop.A_self_adj,
        mu=pop.mu, tau=pop.tau, alpha=pop.alpha, beta=pop.beta,
        gamma=pop.gamma, r=r_new, key=pop.key,
    )

    # Look at the gain landscape for each agent.
    gain = policy.expected_resource_gain(
        pop.mu, pop.tau, pop.r, pop.alpha, pop.beta, pop.gamma, pop.A_self_adj,
        jnp.asarray(cfg.policy.x_grid), cfg.world, cfg.trust, cfg.resource,
    )  # (2, 3)
    # Rich agent's relative preference for the high-x grid point should be
    # greater than poor agent's.
    rich_gap = float(gain[0, -1] - gain[0, 0])
    poor_gap = float(gain[1, -1] - gain[1, 0])
    assert rich_gap > poor_gap, (
        f"Rich agent should favour high-x more: rich gap={rich_gap:.3f}, "
        f"poor gap={poor_gap:.3f}"
    )


# ----------------------------------------------------------------------
# #16: High λ_mc damps belief revision.
# ----------------------------------------------------------------------


def test_tier2_16_high_lambda_damps_belief_revision():
    """Sweeping lambda_mc up reduces the rate at which the agent's posterior
    moves toward truth. ``posterior_rho`` < 1 keeps tau bounded; without it,
    the additive precision blend in lambda_modified_update geometrically
    inflates tau and the run diverges (a real model property worth flagging
    in [[project_implementation_state]]).
    """
    from src import InferenceConfig
    final_mus = []
    for lam in (0.0, 1.0):
        cfg = ModelConfig(
            n_agents=8, seed=22,
            mu_0=-1.0,
            network=NetworkConfig(kind="watts_strogatz", mean_degree=4),
            policy=PolicyConfig(objective="eig_minus_cost"),
            inference=InferenceConfig(posterior_rho=0.9),  # forgetting on
            utility=UtilityConfig(lambda_mc=lam),
        )
        pop = build(cfg)
        for t in range(120):
            pop, _ = pop.step(t)
        final_mus.append(float(pop.mu.mean()))
        assert np.isfinite(final_mus[-1]), (
            f"lam={lam} produced NaN — posterior is unstable"
        )

    dist_low = abs(final_mus[0] - 1.0)
    dist_high = abs(final_mus[1] - 1.0)
    assert dist_high > dist_low, (
        f"High lambda did not damp belief revision: lam=0 dist={dist_low:.3f}, "
        f"lam=1 dist={dist_high:.3f}"
    )
