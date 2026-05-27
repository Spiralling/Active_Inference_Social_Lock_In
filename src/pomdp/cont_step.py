"""Continuous-lambda active-inference model: tempered belief updates + resource coupling.

Direct multi-agent lift of Hyland & Albarracin (2025) Eq. 13. Each agent has a
continuous lambda_i that tempers the likelihood in the Bayes update:

    q_i*(theta) ∝ q_i_prior(theta) · p(o | theta)^(1/lambda_i)

  lambda = 1.0 : standard Bayes (fully responsive)
  lambda > 1.0 : underweight evidence (conservative, theory-laden)
  lambda → ∞   : beliefs frozen to prior

Lambda evolves based on the KL cost of the previous update:

    delta_kl_i = KL[q_i(t) || q_i(t-1)]
    lambda_i(t+1) = lambda_i(t) + eta * (delta_kl - kl_target)

Large updates → lambda increases ("that hurt, be more careful"). Small updates
→ lambda drifts down. This is the variational cost of mind-change, formalized.

Optional resource coupling: agents with low r can only afford low-discriminability
experiments, gating exploration via social structure.

Step cycle:
  1. CHOOSE EXPERIMENT — per-agent, gated by resource if enabled
  2. OBSERVE           — sample from theta*(t) at chosen experiment
  3. EMIT + ROUTE      — trust-weighted social signal
  4. INFER             — tempered categorical Bayes
  5. UPDATE LAMBDA     — KL-driven adaptation
  6. TRUST UPDATE      — optional Gamma-conjugate
  7. RESOURCE FLOW     — optional trust-derived W, eta, cost, flow
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from src.pomdp.gen_model import PomdpConfig, build_generative_model
from src.pomdp.step import build_trust, readout_trust_mixture
from src.pomdp.simple_step import categorical_surprisal, trust_update
from src.resource import flow_from_trust, inflow_share, flow_step
from src.world import theta_schedule

EPS = 1e-12


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ContConfig:
    pomdp: PomdpConfig = field(default_factory=PomdpConfig)

    # lambda (tempering)
    lambda_init: float = 1.5
    lambda_min: float = 0.5
    lambda_max: float = 5.0
    eta_lambda: float = 0.1
    kl_target: float = 0.05

    # paradigm leak: prevents absorbing prior after extended commitment
    eps_theta: float = 0.0       # 0.0=absorbing, >0=leak toward uniform

    obs_x_index: int = 4

    n_agents: int = 80
    n_steps: int = 200

    use_theta_schedule: bool = False

    graph_kind: str = "watts_strogatz"
    mean_degree: int = 4
    rewiring_p: float = 0.1

    social_mask: float = 1.0

    # trust learning
    trust_learning: bool = False
    trust_rho: float = 0.95
    trust_alpha0: float = 1.0
    trust_beta0: float = 1.0

    # resource coupling
    resource_coupling: bool = False
    r_init: float = 1.0
    R_in: float = 0.1
    alpha_flow: float = 0.3
    delta_decay: float = 0.05
    c0: float = 0.1
    r_min: float = 0.1
    budget_fraction: float = 0.5

    seed: int = 0


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ContState:
    q: jax.Array              # (N, K)
    lam: jax.Array            # (N,)
    key: jax.Array
    alpha_t: jax.Array | None = None   # (N, N) trust Gamma shape
    beta_t: jax.Array | None = None    # (N, N) trust Gamma rate
    r: jax.Array | None = None         # (N,) resource


# ------------------------------------------------------------------
# Tempered inference
# ------------------------------------------------------------------

def _safe_log(x):
    return jnp.log(x + EPS)


def _softmax(x, axis=-1):
    x = x - jnp.max(x, axis=axis, keepdims=True)
    e = jnp.exp(x)
    return e / (jnp.sum(e, axis=axis, keepdims=True) + EPS)


def tempered_infer(prior_q: jax.Array,
                   A_world: jax.Array, o_world: jax.Array, action: jax.Array,
                   A_social: jax.Array, o_social: jax.Array,
                   social_mask: jax.Array, lam: jax.Array) -> jax.Array:
    """Tempered Bayes update: q*(theta) ∝ prior(theta) · p(o|theta,a)^(1/lam).

    lam : scalar (single agent). At lam=1, standard Bayes.
    """
    A_a = A_world[:, :, action]                               # (n_o, K)
    ll_world = jnp.sum(o_world[:, None] * _safe_log(A_a), axis=0)  # (K,)
    ll_social = jnp.sum(o_social[:, None] * _safe_log(A_social), axis=0)

    log_post = _safe_log(prior_q) + (1.0 / lam) * ll_world + social_mask * ll_social
    return _softmax(log_post)


def tempered_infer_batch(prior_q, A_world, o_world, actions,
                         A_social, o_social, social_mask, lam):
    """vmap tempered_infer over N agents. lam: (N,)."""
    f = lambda pq, ow, a, osoc, m, l: tempered_infer(
        pq, A_world, ow, a, A_social, osoc, m, l)
    return jax.vmap(f)(prior_q, o_world, actions, o_social, social_mask, lam)


# ------------------------------------------------------------------
# Lambda dynamics
# ------------------------------------------------------------------

def kl_categorical(p: jax.Array, q: jax.Array) -> jax.Array:
    """KL(p || q) for single categorical distribution."""
    return jnp.sum(p * (_safe_log(p) - _safe_log(q)))


def update_lambda(lam: jnp.ndarray, q_new: jnp.ndarray, q_old: jnp.ndarray,
                  eta: float, kl_target: float,
                  lam_min: float, lam_max: float) -> jnp.ndarray:
    """Update per-agent lambda from KL cost of previous update.

    delta_kl > kl_target → lambda increases (more conservative).
    delta_kl < kl_target → lambda decreases (more responsive).
    """
    delta_kl = jax.vmap(kl_categorical)(q_new, q_old)         # (N,)
    lam_new = lam + eta * (delta_kl - kl_target)
    return jnp.clip(lam_new, lam_min, lam_max)


# ------------------------------------------------------------------
# Resource-gated experiment selection
# ------------------------------------------------------------------

def affordable_experiment(r: jnp.ndarray, x_grid: tuple, cfg: ContConfig
                          ) -> jnp.ndarray:
    """Per-agent experiment index: highest x the agent can afford.

    Cost scales with h1(x)^2 / sigma^2 (Fisher info). Agents with low r
    are restricted to low-x (uninformative) experiments.
    """
    from src.world import h1
    x = jnp.array(x_grid)
    h1_vals = h1(x, cfg.pomdp.world)
    fisher = h1_vals ** 2 / (cfg.pomdp.world.sigma ** 2)
    cost_per_x = cfg.c0 * fisher / (r[:, None] - cfg.r_min + EPS)  # (N, A)
    budget = r * cfg.budget_fraction                              # (N,)
    affordable = cost_per_x < budget[:, None]                     # (N, A) bool
    # highest affordable index (fallback to 0 if none)
    indices = jnp.where(affordable, jnp.arange(len(x))[None, :], -1)
    return jnp.maximum(indices.max(axis=1), 0)


# ------------------------------------------------------------------
# Observation sampling
# ------------------------------------------------------------------

def sample_observations(A_world: jnp.ndarray, cfg: ContConfig,
                        actions: jnp.ndarray, t: int,
                        key: jax.Array, N: int) -> tuple[jnp.ndarray, float]:
    """Sample per-agent observations from theta*(t) at per-agent experiments."""
    if cfg.use_theta_schedule:
        theta_star = float(theta_schedule(jnp.asarray(t), cfg.pomdp.world))
        theta_vals = np.asarray(cfg.pomdp.theta_vals)
        lo, hi = theta_vals[0], theta_vals[-1]
        frac = float(jnp.clip((theta_star - lo) / (hi - lo + EPS), 0.0, 1.0))
    else:
        theta_star = float(cfg.pomdp.theta_vals[cfg.pomdp.true_paradigm])
        frac = None

    def _sample_one(key_i, action_i):
        if frac is not None:
            p_o = ((1.0 - frac) * A_world[:, 0, action_i]
                   + frac * A_world[:, 1, action_i])
        else:
            p_o = A_world[:, cfg.pomdp.true_paradigm, action_i]
        return jax.random.categorical(key_i, _safe_log(p_o))

    keys = jax.random.split(key, N)
    o_idx = jax.vmap(_sample_one)(keys, actions)
    o_world = jax.nn.one_hot(o_idx, cfg.pomdp.n_o)
    return o_world, theta_star


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

def init_cont(cfg: ContConfig,
              D_per_agent: np.ndarray | None = None,
              lam_per_agent: np.ndarray | None = None) -> ContState:
    gm = build_generative_model(cfg.pomdp)
    K = cfg.pomdp.n_paradigms
    N = cfg.n_agents

    if D_per_agent is None:
        q0 = jnp.broadcast_to(gm["D"], (N, K)).copy()
    else:
        d = jnp.asarray(D_per_agent, dtype=float)
        q0 = d / (d.sum(axis=1, keepdims=True) + EPS)

    if lam_per_agent is None:
        lam0 = jnp.full((N,), cfg.lambda_init)
    else:
        lam0 = jnp.asarray(lam_per_agent, dtype=float)

    alpha0 = jnp.full((N, N), cfg.trust_alpha0) if cfg.trust_learning else None
    beta0 = jnp.full((N, N), cfg.trust_beta0) if cfg.trust_learning else None
    r0 = jnp.full((N,), cfg.r_init) if cfg.resource_coupling else None

    return ContState(q=q0, lam=lam0, key=jax.random.PRNGKey(cfg.seed),
                     alpha_t=alpha0, beta_t=beta0, r=r0)


# ------------------------------------------------------------------
# Single step
# ------------------------------------------------------------------

def cont_step(state: ContState,
              gm: dict,
              T: jnp.ndarray,
              adj: jnp.ndarray,
              cfg: ContConfig,
              social_mask: jnp.ndarray,
              t: int) -> tuple[ContState, jnp.ndarray, dict]:
    N, K = state.q.shape
    key_obs, key_next = jax.random.split(state.key)

    A_world = gm["A_world"]
    A_social = gm["A_social"]

    # 1. CHOOSE EXPERIMENT
    if cfg.resource_coupling and state.r is not None:
        actions = affordable_experiment(state.r, cfg.pomdp.x_grid, cfg)
    else:
        actions = jnp.full((N,), cfg.obs_x_index, dtype=jnp.int32)

    # 2. OBSERVE
    o_world, theta_star_t = sample_observations(A_world, cfg, actions, t, key_obs, N)

    # 3. PRIOR LEAK (prevent absorbing commitment)
    q_prior = state.q
    if cfg.eps_theta > 0:
        uniform = jnp.ones(K) / K
        q_prior = (1.0 - cfg.eps_theta) * q_prior + cfg.eps_theta * uniform

    # 4. EMIT + ROUTE
    o_social = readout_trust_mixture(q_prior, T)

    # 5. INFER (tempered)
    q_new = tempered_infer_batch(
        q_prior, A_world, o_world, actions,
        A_social, o_social, social_mask, state.lam)

    # 6. UPDATE LAMBDA
    lam_new = update_lambda(
        state.lam, q_new, q_prior,
        cfg.eta_lambda, cfg.kl_target, cfg.lambda_min, cfg.lambda_max)

    # 7. TRUST UPDATE
    alpha_new, beta_new, T_new = state.alpha_t, state.beta_t, T
    if cfg.trust_learning and state.alpha_t is not None:
        epsilon = categorical_surprisal(q_new, A_world, o_world, cfg.obs_x_index, adj)
        alpha_new, beta_new, T_new = trust_update(
            state.alpha_t, state.beta_t, epsilon, adj, cfg.trust_rho)

    # 8. RESOURCE FLOW
    r_new = state.r
    if cfg.resource_coupling and state.r is not None:
        W = jnp.asarray(flow_from_trust(np.asarray(T)))
        eta = jnp.asarray(inflow_share(np.asarray(T)))
        from src.world import h1
        x_chosen = jnp.array(cfg.pomdp.x_grid)[actions]
        h1_vals = h1(x_chosen, cfg.pomdp.world)
        fisher = h1_vals ** 2 / (cfg.pomdp.world.sigma ** 2)
        c_x = cfg.c0 * fisher / (state.r - cfg.r_min + EPS)
        r_new = jnp.asarray(flow_step(
            np.asarray(state.r), np.asarray(W), np.asarray(eta),
            np.asarray(c_x), cfg.R_in, cfg.alpha_flow, cfg.delta_decay))
        r_new = jnp.maximum(r_new, cfg.r_min)

    new_state = ContState(q=q_new, lam=lam_new, key=key_next,
                          alpha_t=alpha_new, beta_t=beta_new, r=r_new)

    info = {
        "mean_q": np.asarray(q_new.mean(axis=0)),
        "mean_lam": float(lam_new.mean()),
        "theta_star_t": theta_star_t,
    }
    if r_new is not None:
        info["mean_r"] = float(np.asarray(r_new).mean())
    return new_state, T_new, info


# ------------------------------------------------------------------
# Full run
# ------------------------------------------------------------------

def run_cont(cfg: ContConfig,
             D_per_agent: np.ndarray | None = None,
             social_mask_per_agent: np.ndarray | None = None,
             lam_per_agent: np.ndarray | None = None) -> dict:

    gm = build_generative_model(cfg.pomdp)
    T = build_trust(cfg.pomdp, cfg.n_agents, mean_degree=cfg.mean_degree,
                    kind=cfg.graph_kind, rewiring_p=cfg.rewiring_p,
                    seed=cfg.seed)
    adj = (T > 0).astype(float)
    state = init_cont(cfg, D_per_agent, lam_per_agent)

    if social_mask_per_agent is not None:
        s_mask = jnp.asarray(social_mask_per_agent, dtype=float)
    else:
        s_mask = jnp.full((cfg.n_agents,), cfg.social_mask, dtype=float)

    true_idx = cfg.pomdp.true_paradigm
    mean_qB = np.empty(cfg.n_steps)
    occ_B = np.empty(cfg.n_steps)
    mean_lam = np.empty(cfg.n_steps)
    theta_star_trace = np.empty(cfg.n_steps)
    mean_r = np.empty(cfg.n_steps) if cfg.resource_coupling else None
    infos: list[dict] = []

    for t in range(cfg.n_steps):
        state, T, info = cont_step(state, gm, T, adj, cfg, s_mask, t)
        q_np = np.asarray(state.q)
        mean_qB[t] = info["mean_q"][true_idx]
        occ_B[t] = float(np.mean(q_np[:, true_idx] > 0.5))
        mean_lam[t] = info["mean_lam"]
        theta_star_trace[t] = info["theta_star_t"]
        if mean_r is not None:
            mean_r[t] = info.get("mean_r", 0.0)
        infos.append(info)

    result = {
        "mean_qB": mean_qB,
        "occ_B": occ_B,
        "mean_lambda": mean_lam,
        "final_q": np.asarray(state.q),
        "final_lam": np.asarray(state.lam),
        "infos": infos,
        "T": np.asarray(T),
        "theta_star_trace": theta_star_trace,
    }
    if mean_r is not None:
        result["mean_r"] = mean_r
        result["final_r"] = np.asarray(state.r)
    return result
