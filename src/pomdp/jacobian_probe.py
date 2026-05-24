"""Jacobian eigenvalue probe of the EXISTING substrate (plan step 0c).

This is a paper artifact, not part of the rebuild. The diagnosis note
(`notes/dynamical_systems_diagnosis.tex`) argues — via an *abstracted linear
reduction* (its eq. 1/2, with the trust matrix frozen) — that the substrate is
globally monostable. This probe upgrades that argument from "we linearize and
argue" to "we computed the full discrete-time Jacobian (mu, tau, alpha, beta,
gamma, r — including the slow trust and resource couplings) at the fixed point
and verified the spectral radius stays below 1 across the parameter box."

Why a deterministic step
------------------------
The real per-step map (`src/population.py:_step_jit`) is *stochastic*: it samples
the experiment x from a softmax policy and the observation o from N(.). An
autodiff Jacobian is only well-defined on a deterministic map, so we linearize
the **deterministic mean dynamics**:

  * x is taken as the modal (argmax) experiment — generically locally constant,
    so this is exactly the right local linearization;
  * o is replaced by its expectation  E[o | x] = h0(x) + theta*(t) h1(x).

Every other channel — the lambda.U social blend, the precision pool, the
Gamma-conjugate trust update, and the full resource recursion — is run exactly
as in the real step. This is precisely the object the note's eq. (1) abstracts.

Discrete-time reading: a fixed point is stable iff the spectral radius
rho(J) = max_k |lambda_k(J)| < 1 (the discrete analogue of the note's
"eigenvalues in the open left half-plane").
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from src import inference, resource, trust, utility, world
from src.config import ModelConfig
from src.network import build_adjacency
from src.policy import expected_info_gain, cost as policy_cost


def _choose_x_deterministic(tau, cfg):
    """Modal experiment per agent under the eig_minus_cost objective.

    logits = beta_exp * (EIG_i(x) - cost(x)); pick argmax over the grid.
    argmax has zero gradient a.e., so x is treated as locally constant in the
    Jacobian — the intended local linearization.
    """
    x_grid = jnp.asarray(cfg.policy.x_grid)
    eig = expected_info_gain(x_grid, tau, cfg.world)          # (N, G)
    c = policy_cost(x_grid, cfg.policy)                        # (G,)
    efe = eig - c[None, :]
    idx = jnp.argmax(efe, axis=1)                             # (N,)
    return x_grid[idx]


def deterministic_step(state, t, cfg, A_self):
    """One deterministic step. `state` is a tuple (mu, tau, alpha, beta, gamma, r).

    Mirrors `population._step_jit` with x = argmax and o = E[o | x].
    """
    mu, tau, alpha, beta, gamma, r = state

    # 1. deterministic policy
    x = _choose_x_deterministic(tau, cfg)

    # 2. expected observation (no noise)
    theta_star = world.theta_schedule(t, cfg.world)
    o = world.h0(x, cfg.world) + theta_star * world.h1(x, cfg.world)

    # 3. private Bayes update (with forgetting toward the paradigm prior)
    mu_data, tau_data = inference.private_update(
        mu, tau, x, o, cfg.world,
        cfg.inference.posterior_rho, cfg.mu_0, cfg.tau_0,
    )

    # 4. lambda.U social blend (resource-weighted target)
    mu_tgt, tau_tgt = utility.socially_induced_U(mu, tau, r, A_self)
    mu_priv, tau_priv = utility.lambda_modified_update(
        mu_data, tau_data, mu_tgt, tau_tgt, cfg.utility.lambda_mc,
    )

    # 5. precision pool over the closed neighbourhood
    mu_pool, tau_pool = inference.precision_pool(mu_priv, tau_priv, gamma, A_self)

    # 6. Gamma-conjugate trust update (surprisal against E[o])
    epsilon = trust.surprisal_matrix(mu_priv, tau_priv, x, o, A_self, cfg.world)
    alpha_new, beta_new, gamma_new = trust.gamma_conjugate_step(
        alpha, beta, epsilon, A_self, cfg.trust,
    )

    # 7. resource recursion (W, eta deterministic readouts of gamma_new)
    W_new = resource.flow_from_trust(gamma_new)
    eta_new = resource.inflow_share(gamma_new)
    h1x_sq = (world.h1(x, cfg.world) ** 2) / (cfg.world.sigma ** 2 + 1e-12)
    c_x = resource.cost_x(
        x, r, h1x_sq, c0=cfg.resource.c0, r_min=cfg.resource.r_min,
        barrier_eps=cfg.resource.barrier_eps,
        fisher_cost_steepness=cfg.resource.fisher_cost_steepness,
    )
    r_new = resource.flow_step(
        r, W_new, eta_new, c_x, R_in=cfg.resource.R_in,
        alpha_flow=cfg.resource.alpha_flow, delta_decay=cfg.resource.delta_decay,
    )
    return (mu_pool, tau_pool, alpha_new, beta_new, gamma_new, r_new)


# ----------------------------------------------------------------------
# flatten / unflatten the dynamic state
# ----------------------------------------------------------------------

def _flatten(state):
    return jnp.concatenate([jnp.ravel(x) for x in state])


def _make_unflatten(shapes):
    sizes = [int(np.prod(s)) for s in shapes]
    splits = np.cumsum(sizes)[:-1]
    def unflatten(vec):
        parts = jnp.split(vec, splits)
        return tuple(p.reshape(s) for p, s in zip(parts, shapes))
    return unflatten


def init_state(cfg, key):
    """Construct an initial (mu, tau, alpha, beta, gamma, r) and A_self."""
    N = cfg.n_agents
    A_np = build_adjacency(
        n_agents=N, mean_degree=cfg.network.mean_degree,
        rewiring_p=cfg.network.rewiring_p, seed=cfg.seed,
        kind=cfg.network.kind, society_membership=None,
        intra_prob=cfg.network.intra_prob, inter_prob=cfg.network.inter_prob,
    )
    A_adj = jnp.asarray(A_np)
    A_self = A_adj + jnp.eye(N, dtype=A_adj.dtype)

    mu = jnp.full((N,), cfg.mu_0)
    tau = jnp.full((N,), cfg.tau_0)
    n0 = cfg.trust.prior_n0
    beta0 = cfg.trust.prior_n0 * cfg.trust.prior_eps0
    alpha = n0 * A_self
    beta = beta0 * A_self
    gamma_raw = (alpha / (beta + 1e-12)) * A_self
    gamma = (gamma_raw / (gamma_raw.sum(axis=1, keepdims=True) + 1e-12)) * A_self
    r = jnp.full((N,), cfg.resource.r0)
    return (mu, tau, alpha, beta, gamma, r), A_self


def find_fixed_point(cfg, t_eval, n_iter=4000, tol=1e-9):
    """Iterate the deterministic step to a fixed point at fixed time t_eval.

    Returns (state_fp, A_self, residual). Requires forgetting (posterior_rho<1,
    trust.rho<1) for an interior fixed point to exist.
    """
    state, A_self = init_state(cfg, jax.random.PRNGKey(cfg.seed))
    shapes = [x.shape for x in state]
    unflatten = _make_unflatten(shapes)

    step = jax.jit(lambda s: deterministic_step(s, t_eval, cfg, A_self))
    vec = _flatten(state)
    residual = np.inf
    for _ in range(n_iter):
        new_state = step(unflatten(vec))
        new_vec = _flatten(new_state)
        residual = float(jnp.max(jnp.abs(new_vec - vec)))
        vec = new_vec
        if residual < tol:
            break
    return unflatten(vec), A_self, residual


def spectral_radius_at_fixed_point(cfg, t_eval, n_iter=4000):
    """Find the FP and return (spectral_radius, eigenvalues, residual)."""
    state_fp, A_self, residual = find_fixed_point(cfg, t_eval, n_iter=n_iter)
    shapes = [x.shape for x in state_fp]
    unflatten = _make_unflatten(shapes)

    def flat_step(vec):
        return _flatten(deterministic_step(unflatten(vec), t_eval, cfg, A_self))

    J = jax.jacrev(flat_step)(_flatten(state_fp))
    eig = jnp.linalg.eigvals(J)
    rho = float(jnp.max(jnp.abs(eig)))
    return rho, np.asarray(eig), residual


def small_config(N=12, **overrides):
    """A small, forgetting-enabled config suitable for the Jacobian probe.

    Forgetting (posterior_rho<1, trust.rho<1) is required so an interior fixed
    point exists. Evaluation time is past the schedule shift (theta* = post).
    `overrides` are applied to the relevant sub-config via dataclasses.replace.
    """
    base = ModelConfig(n_agents=N, seed=0)
    # Forgetting (posterior_rho<1, trust.rho<1) + genuine resource decay are
    # required for an interior fixed point. Without delta_decay the resource
    # state is a pure integrator (a trivial neutral mode at eigenvalue 1) and
    # with R_in>0 it grows without bound, so no fixed point exists.
    inf = dataclasses.replace(base.inference, posterior_rho=0.9)
    tr = dataclasses.replace(base.trust, rho=0.9)
    res = dataclasses.replace(
        base.resource,
        delta_decay=overrides.get("delta_decay", 0.1),
        alpha_flow=overrides.get("alpha_flow", 0.3),
        R_in=overrides.get("R_in", base.resource.R_in),
        c0=overrides.get("c0", base.resource.c0),
    )
    util = dataclasses.replace(base.utility, lambda_mc=overrides.get("lambda_mc", 0.0))
    return dataclasses.replace(base, inference=inf, trust=tr, resource=res, utility=util)
