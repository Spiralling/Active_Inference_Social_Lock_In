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

    n_context: int = 2
    obs_x_index: int = 2

    n_agents: int = 80
    n_steps: int = 200

    use_theta_schedule: bool = False

    graph_kind: str = "watts_strogatz"
    mean_degree: int = 4
    rewiring_p: float = 0.1

    social_mask: float = 1.0
    seed: int = 0


# ------------------------------------------------------------------
# Joint generative-model builders (called once per run)
# ------------------------------------------------------------------

def build_joint_A_world(A_world: jnp.ndarray, K: int, C: int = 2) -> jnp.ndarray:
    """Joint likelihood A_joint[o, theta*C + c, a].

    c=0 columns: paradigm-averaged (identical across theta — uninformative).
    c=1 columns: original A_world[:, theta, a] (fully discriminating).
    """
    n_o, _K, A = A_world.shape
    avg = jnp.mean(A_world, axis=1, keepdims=True)          # (n_o, 1, A)
    avg = jnp.broadcast_to(avg, (n_o, K, A))                # (n_o, K, A)

    A_joint = jnp.empty((n_o, K * C, A))
    for k in range(K):
        A_joint = A_joint.at[:, k * C + 0, :].set(avg[:, k, :])
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
                     K: int, C: int = 2) -> jnp.ndarray:
    """Apply B_theta ⊗ B_c transition to the joint posterior.

    Since B_theta = identity, this applies B_c independently within each
    theta block: q_prior[theta, c'] = sum_c B_c[c', c] * q_post[theta, c].
    """
    N = q_joint.shape[0]
    q = q_joint.reshape(N, K, C)                           # (N, K, C)
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
# State
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SimpleState:
    q_joint: jax.Array    # (N, K*C) joint belief over (paradigm, context)
    key: jax.Array


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

    return SimpleState(q_joint=q0_joint, key=jax.random.PRNGKey(cfg.seed))


# ------------------------------------------------------------------
# Single step
# ------------------------------------------------------------------

def simple_step(state: SimpleState,
                gm_joint: dict,
                A_world_orig: jnp.ndarray,
                T: jnp.ndarray,
                cfg: SimpleConfig,
                t: int) -> tuple[SimpleState, dict]:
    K = cfg.pomdp.n_paradigms
    C = cfg.n_context
    N = state.q_joint.shape[0]
    key_obs, key_next = jax.random.split(state.key)

    # 1. TRANSITION — proper AIF prior via B_joint
    B_c = gm_joint["B_c"]
    q_prior = transition_joint(state.q_joint, B_c, K, C)     # (N, K*C)

    # 2. OBSERVE
    o_world, theta_star_t = sample_observations(
        A_world_orig, cfg, t, key_obs, N)

    # 3. EMIT + ROUTE (emit marginal q(theta) for social channel)
    q_theta = marginalize_theta(q_prior, K, C)                # (N, K)
    emissions = q_theta
    o_social = readout_trust_mixture(emissions, T)            # (N, K)

    # 4. INFER on joint state (theta, c)
    actions = jnp.full((N,), cfg.obs_x_index, dtype=jnp.int32)
    mask = jnp.broadcast_to(jnp.asarray(cfg.social_mask, dtype=float), (N,))

    q_joint_post = agent_pop.infer_state_batch(
        q_prior,
        gm_joint["A_world_joint"],
        o_world,
        actions,
        gm_joint["A_social_joint"],
        o_social,
        mask,
    )                                                         # (N, K*C)

    new_state = SimpleState(q_joint=q_joint_post, key=key_next)

    q_theta_post = marginalize_theta(q_joint_post, K, C)
    q_c_post = marginalize_context(q_joint_post, K, C)
    info = {
        "mean_q": np.asarray(q_theta_post.mean(axis=0)),
        "mean_q_c": np.asarray(q_c_post.mean(axis=0)),
        "theta_star_t": theta_star_t,
    }
    return new_state, info


# ------------------------------------------------------------------
# Full run
# ------------------------------------------------------------------

def run_simple(cfg: SimpleConfig,
               D_per_agent: np.ndarray | None = None) -> dict:
    K = cfg.pomdp.n_paradigms
    C = cfg.n_context

    gm_orig = build_generative_model(cfg.pomdp)
    A_world_orig = gm_orig["A_world"]

    gm_joint = {
        "A_world_joint": build_joint_A_world(A_world_orig, K, C),
        "A_social_joint": build_joint_A_social(gm_orig["A_social"], K, C),
        "B_c": build_B_c(cfg.eps_crisis, cfg.eps_resolve),
    }

    T = build_trust(cfg.pomdp, cfg.n_agents, mean_degree=cfg.mean_degree,
                    kind=cfg.graph_kind, rewiring_p=cfg.rewiring_p,
                    seed=cfg.seed)
    state = init_simple(cfg, D_per_agent)

    true_paradigm_idx = cfg.pomdp.true_paradigm
    mean_qB = np.empty(cfg.n_steps)
    occ_B = np.empty(cfg.n_steps)
    mean_c_normal = np.empty(cfg.n_steps)
    theta_star_trace = np.empty(cfg.n_steps)
    infos: list[dict] = []

    for t in range(cfg.n_steps):
        state, info = simple_step(state, gm_joint, A_world_orig, T, cfg, t)
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
