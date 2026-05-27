"""Simplified active-inference model: theory-laden observation via hierarchical context.

Drops the POMDP action loop. Each step:
  1. TRANSITION — apply B_joint to previous joint posterior (proper AIF prior)
  2. OBSERVE    — environment samples outcomes from theta*(t) at a fixed experiment
  3. EMIT       — agents emit marginal q(theta) as soft paradigm beliefs
  4. ROUTE      — trust-weighted social observation (reuse step.readout_trust_mixture)
  5. INFER      — categorical Bayes on joint state (theta, c) using shared A_joint

The theory-ladenness mechanism is a latent context variable c in {0, 1}:
  c=0 ("normal science"): paradigm columns averaged — observations uninformative
  c=1 ("crisis/anomaly"):  original A_world columns — observations discriminate

The context prior is NOT reset each step. Instead, the joint posterior q(theta, c)
is carried forward through a transition model B = B_theta ⊗ B_c, where B_theta is
identity (truth is fixed) and B_c encodes a bias toward normal science:

  B_c = [[1 - eps_crisis,   eps_resolve  ],
         [    eps_crisis,  1 - eps_resolve]]

  eps_crisis  : P(c'=crisis | c=normal)  — rate of spontaneous anomaly detection
  eps_resolve : P(c'=normal | c=crisis)  — rate of crisis resolution (paradigm inertia)

Lock-in emerges from the dynamics: an agent committed to the wrong paradigm sees
data through c=0 (uninformative), which reinforces c=0 via B_c, further insulating
beliefs from paradigm-discriminating evidence. Breaking out requires sustained
anomalous evidence strong enough to overcome both the B_c bias and social coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from src.pomdp import agent_pop
from src.pomdp.gen_model import PomdpConfig, build_generative_model
from src.pomdp.step import build_trust, readout_trust_mixture
from src.world import theta_schedule

EPS = 1e-12


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SimpleConfig:
    pomdp: PomdpConfig = field(default_factory=PomdpConfig)

    # context transition (B_c)
    eps_crisis: float = 0.05     # P(c'=1 | c=0): spontaneous anomaly rate
    eps_resolve: float = 0.30    # P(c'=0 | c=1): crisis resolution rate

    # initial context prior
    D_c_normal: float = 0.9      # initial P(c=0) — starts in normal science

    # partial theory-ladenness: how much c=0 averages away discriminability
    alpha_tl: float = 1.0        # 1.0=fully averaged (current), 0.0=no averaging

    # paradigm leak: small drift of theta belief toward uniform each step
    eps_theta: float = 0.0       # 0.0=absorbing (B_theta=identity), >0=leaky

    n_context: int = 2
    obs_x_index: int = 2

    n_agents: int = 80
    n_steps: int = 200

    use_theta_schedule: bool = False

    graph_kind: str = "watts_strogatz"
    mean_degree: int = 4
    rewiring_p: float = 0.1

    social_mask: float = 1.0

    # trust learning
    trust_learning: bool = False
    trust_rho: float = 0.95      # forgetting rate for Gamma hyperparameters
    trust_alpha0: float = 1.0    # initial Gamma shape
    trust_beta0: float = 1.0     # initial Gamma rate

    seed: int = 0


# ------------------------------------------------------------------
# Joint generative-model builders (called once per run)
# ------------------------------------------------------------------

def build_joint_A_world(A_world: jnp.ndarray, K: int, C: int = 2,
                        alpha: float = 1.0) -> jnp.ndarray:
    """Joint likelihood A_joint[o, theta*C + c, a].

    c=0 columns: blend of paradigm-specific and average, controlled by alpha.
      alpha=1.0: fully averaged (identical across theta — uninformative).
      alpha=0.0: no averaging (c=0 same as c=1 — no theory-ladenness).
    c=1 columns: original A_world[:, theta, a] (fully discriminating).
    """
    n_o, _K, A = A_world.shape
    avg = jnp.mean(A_world, axis=1, keepdims=True)          # (n_o, 1, A)
    avg = jnp.broadcast_to(avg, (n_o, K, A))                # (n_o, K, A)

    A_joint = jnp.empty((n_o, K * C, A))
    for k in range(K):
        A_c0 = (1.0 - alpha) * A_world[:, k, :] + alpha * avg[:, k, :]
        A_joint = A_joint.at[:, k * C + 0, :].set(A_c0)
        A_joint = A_joint.at[:, k * C + 1, :].set(A_world[:, k, :])

    A_joint = A_joint / (A_joint.sum(axis=0, keepdims=True) + EPS)
    return A_joint


def build_joint_A_social(A_social: jnp.ndarray, K: int, C: int = 2) -> jnp.ndarray:
    """Joint social likelihood A_social_joint[k_obs, theta*C + c].

    Social observations carry info about theta only, not c.
    """
    A_joint = jnp.empty((K, K * C))
    for k in range(K):
        for c in range(C):
            A_joint = A_joint.at[:, k * C + c].set(A_social[:, k])
    A_joint = A_joint / (A_joint.sum(axis=0, keepdims=True) + EPS)
    return A_joint


def build_B_c(eps_crisis: float, eps_resolve: float) -> jnp.ndarray:
    """Context transition matrix B_c[c', c] (columns are source state).

    B_c[:, 0] = [1-eps_crisis, eps_crisis]     — from normal science
    B_c[:, 1] = [eps_resolve,  1-eps_resolve]  — from crisis
    """
    return jnp.array([[1.0 - eps_crisis, eps_resolve],
                       [eps_crisis, 1.0 - eps_resolve]])


# ------------------------------------------------------------------
# Per-step helpers
# ------------------------------------------------------------------

def transition_joint(q_joint: jnp.ndarray, B_c: jnp.ndarray,
                     K: int, C: int = 2,
                     eps_theta: float = 0.0) -> jnp.ndarray:
    """Apply B_theta ⊗ B_c transition to the joint posterior.

    B_c is applied within each theta block. If eps_theta > 0, a small leak
    toward uniform over theta is applied first (prevents absorbing commitment).
    """
    N = q_joint.shape[0]
    q = q_joint.reshape(N, K, C)                           # (N, K, C)
    if eps_theta > 0:
        uniform_theta = jnp.ones((K,)) / K
        q = (1.0 - eps_theta) * q + eps_theta * uniform_theta[None, :, None]
    q_new = jnp.einsum('ij,nkj->nki', B_c, q)             # (N, K, C)
    q_flat = q_new.reshape(N, K * C)
    return q_flat / (q_flat.sum(axis=1, keepdims=True) + EPS)


def marginalize_theta(q_joint: jnp.ndarray, K: int, C: int = 2) -> jnp.ndarray:
    """(N, K*C) joint posterior -> (N, K) marginal over theta."""
    N = q_joint.shape[0]
    return q_joint.reshape(N, K, C).sum(axis=2)


def marginalize_context(q_joint: jnp.ndarray, K: int, C: int = 2) -> jnp.ndarray:
    """(N, K*C) joint posterior -> (N, C) marginal over context."""
    N = q_joint.shape[0]
    return q_joint.reshape(N, K, C).sum(axis=1)


def sample_observations(A_world_orig: jnp.ndarray, cfg: SimpleConfig,
                        t: int, key: jax.Array, N: int
                        ) -> tuple[jnp.ndarray, float]:
    """Sample one-hot observations from the true paradigm at the fixed experiment."""
    action = cfg.obs_x_index

    if cfg.use_theta_schedule:
        theta_star = float(theta_schedule(jnp.asarray(t), cfg.pomdp.world))
        theta_vals = np.asarray(cfg.pomdp.theta_vals)
        lo, hi = theta_vals[0], theta_vals[-1]
        frac = (theta_star - lo) / (hi - lo + EPS)
        frac = float(jnp.clip(frac, 0.0, 1.0))
        p_o = ((1.0 - frac) * A_world_orig[:, 0, action]
               + frac * A_world_orig[:, 1, action])
    else:
        theta_star = float(cfg.pomdp.theta_vals[cfg.pomdp.true_paradigm])
        p_o = A_world_orig[:, cfg.pomdp.true_paradigm, action]

    keys = jax.random.split(key, N)
    log_p = jnp.log(p_o + EPS)
    o_idx = jax.vmap(lambda k: jax.random.categorical(k, log_p))(keys)
    o_world = jax.nn.one_hot(o_idx, cfg.pomdp.n_o)
    return o_world, theta_star


# ------------------------------------------------------------------
# Trust learning (categorical surprisal + Gamma-conjugate update)
# ------------------------------------------------------------------

def categorical_surprisal(q_theta: jnp.ndarray, A_world: jnp.ndarray,
                          o_world: jnp.ndarray, action: int,
                          adj: jnp.ndarray) -> jnp.ndarray:
    """Surprisal matrix eps_ij: how well did j's belief predict i's observation?

    eps_ij = -log sum_theta q_j(theta) * A_world[o_i, theta, action]

    q_theta : (N, K), A_world : (n_o, K, A), o_world : (N, n_o) one-hot,
    adj : (N, N) adjacency + self-loops.
    Returns (N, N) masked surprisal.
    """
    A_a = A_world[:, :, action]                               # (n_o, K)
    pred_j = q_theta @ A_a.T                                  # (N, n_o): j's predictive
    log_pred_j = jnp.log(pred_j + EPS)                        # (N, n_o)
    ll_at_i = jnp.sum(o_world[:, None, :] * log_pred_j[None, :, :], axis=2)  # (N, N)
    return -ll_at_i * adj


def trust_update(alpha_t: jnp.ndarray, beta_t: jnp.ndarray,
                 epsilon: jnp.ndarray, adj: jnp.ndarray,
                 rho: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Gamma-conjugate trust update. Returns (alpha_new, beta_new, T_new)."""
    alpha_new = rho * alpha_t + adj
    beta_new = rho * beta_t + epsilon
    gamma = (alpha_new / (beta_new + EPS)) * adj
    T_new = gamma / (gamma.sum(axis=1, keepdims=True) + EPS)
    return alpha_new, beta_new, T_new


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SimpleState:
    q_joint: jax.Array    # (N, K*C) joint belief over (paradigm, context)
    key: jax.Array
    alpha_t: jax.Array | None = None   # (N, N) Gamma shape for trust
    beta_t: jax.Array | None = None    # (N, N) Gamma rate for trust


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

def init_simple(cfg: SimpleConfig,
                D_per_agent: np.ndarray | None = None) -> SimpleState:
    gm = build_generative_model(cfg.pomdp)
    K, C = cfg.pomdp.n_paradigms, cfg.n_context
    D_c = jnp.array([cfg.D_c_normal, 1.0 - cfg.D_c_normal])

    if D_per_agent is None:
        D_theta = gm["D"]                                    # (K,)
        q0_joint = jnp.outer(D_theta, D_c)                   # (K, C)
        q0_joint = jnp.broadcast_to(q0_joint.ravel(), (cfg.n_agents, K * C)).copy()
    else:
        d = jnp.asarray(D_per_agent, dtype=float)
        d = d / (d.sum(axis=1, keepdims=True) + EPS)         # (N, K)
        q0_joint = (d[:, :, None] * D_c[None, None, :]).reshape(cfg.n_agents, K * C)

    N = cfg.n_agents
    alpha0 = None
    beta0 = None
    if cfg.trust_learning:
        alpha0 = jnp.full((N, N), cfg.trust_alpha0)
        beta0 = jnp.full((N, N), cfg.trust_beta0)

    return SimpleState(q_joint=q0_joint, key=jax.random.PRNGKey(cfg.seed),
                       alpha_t=alpha0, beta_t=beta0)


# ------------------------------------------------------------------
# Single step
# ------------------------------------------------------------------

def simple_step(state: SimpleState,
                gm_joint: dict,
                A_world_orig: jnp.ndarray,
                T: jnp.ndarray,
                adj: jnp.ndarray,
                cfg: SimpleConfig,
                social_mask: jnp.ndarray,
                t: int) -> tuple[SimpleState, jnp.ndarray, dict]:
    """One step. Returns (new_state, T_new, info)."""
    K = cfg.pomdp.n_paradigms
    C = cfg.n_context
    N = state.q_joint.shape[0]
    key_obs, key_next = jax.random.split(state.key)

    # 1. TRANSITION — proper AIF prior via B_joint
    B_c = gm_joint["B_c"]
    q_prior = transition_joint(state.q_joint, B_c, K, C,
                               eps_theta=cfg.eps_theta)       # (N, K*C)

    # 2. OBSERVE
    o_world, theta_star_t = sample_observations(
        A_world_orig, cfg, t, key_obs, N)

    # 3. EMIT + ROUTE (emit marginal q(theta) for social channel)
    q_theta = marginalize_theta(q_prior, K, C)                # (N, K)
    emissions = q_theta
    o_social = readout_trust_mixture(emissions, T)            # (N, K)

    # 4. INFER on joint state (theta, c)
    actions = jnp.full((N,), cfg.obs_x_index, dtype=jnp.int32)
    mask = social_mask

    q_joint_post = agent_pop.infer_state_batch(
        q_prior,
        gm_joint["A_world_joint"],
        o_world,
        actions,
        gm_joint["A_social_joint"],
        o_social,
        mask,
    )                                                         # (N, K*C)

    # 5. TRUST LEARNING (optional)
    alpha_new, beta_new, T_new = state.alpha_t, state.beta_t, T
    if cfg.trust_learning and state.alpha_t is not None:
        q_theta_post = marginalize_theta(q_joint_post, K, C)
        epsilon = categorical_surprisal(
            q_theta_post, A_world_orig, o_world, cfg.obs_x_index, adj)
        alpha_new, beta_new, T_new = trust_update(
            state.alpha_t, state.beta_t, epsilon, adj, cfg.trust_rho)

    new_state = SimpleState(q_joint=q_joint_post, key=key_next,
                            alpha_t=alpha_new, beta_t=beta_new)

    q_theta_post = marginalize_theta(q_joint_post, K, C)
    q_c_post = marginalize_context(q_joint_post, K, C)
    info = {
        "mean_q": np.asarray(q_theta_post.mean(axis=0)),
        "mean_q_c": np.asarray(q_c_post.mean(axis=0)),
        "theta_star_t": theta_star_t,
    }
    return new_state, T_new, info


# ------------------------------------------------------------------
# Full run
# ------------------------------------------------------------------

def run_simple(cfg: SimpleConfig,
               D_per_agent: np.ndarray | None = None,
               social_mask_per_agent: np.ndarray | None = None) -> dict:
    K = cfg.pomdp.n_paradigms
    C = cfg.n_context

    gm_orig = build_generative_model(cfg.pomdp)
    A_world_orig = gm_orig["A_world"]

    gm_joint = {
        "A_world_joint": build_joint_A_world(A_world_orig, K, C, cfg.alpha_tl),
        "A_social_joint": build_joint_A_social(gm_orig["A_social"], K, C),
        "B_c": build_B_c(cfg.eps_crisis, cfg.eps_resolve),
    }

    T = build_trust(cfg.pomdp, cfg.n_agents, mean_degree=cfg.mean_degree,
                    kind=cfg.graph_kind, rewiring_p=cfg.rewiring_p,
                    seed=cfg.seed)
    adj = (T > 0).astype(float)                               # adjacency + self-loops
    state = init_simple(cfg, D_per_agent)

    if social_mask_per_agent is not None:
        s_mask = jnp.asarray(social_mask_per_agent, dtype=float)
    else:
        s_mask = jnp.full((cfg.n_agents,), cfg.social_mask, dtype=float)

    true_paradigm_idx = cfg.pomdp.true_paradigm
    mean_qB = np.empty(cfg.n_steps)
    occ_B = np.empty(cfg.n_steps)
    mean_c_normal = np.empty(cfg.n_steps)
    theta_star_trace = np.empty(cfg.n_steps)
    infos: list[dict] = []

    for t in range(cfg.n_steps):
        state, T, info = simple_step(
            state, gm_joint, A_world_orig, T, adj, cfg, s_mask, t)
        q_theta = np.asarray(marginalize_theta(state.q_joint, K, C))
        mean_qB[t] = info["mean_q"][true_paradigm_idx]
        occ_B[t] = float(np.mean(q_theta[:, true_paradigm_idx] > 0.5))
        mean_c_normal[t] = info["mean_q_c"][0]
        theta_star_trace[t] = info["theta_star_t"]
        infos.append(info)

    return {
        "mean_qB": mean_qB,
        "occ_B": occ_B,
        "mean_c_normal": mean_c_normal,
        "final_q": np.asarray(marginalize_theta(state.q_joint, K, C)),
        "final_q_joint": np.asarray(state.q_joint),
        "infos": infos,
        "T": np.asarray(T),
        "theta_star_trace": theta_star_trace,
    }
