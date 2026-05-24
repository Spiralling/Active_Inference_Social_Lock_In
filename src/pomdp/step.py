"""The inter-agent cycle (plan §3): emit -> route -> infer -> act.

This is the part pymdp gives ZERO help with — routing one agent's belief into
another's observation. Per round:

  1. EMIT     each agent emits a soft paradigm signal = its belief q_j(theta).
  2. ROUTE    agent i's social observation is the trust-weighted mixture of its
              in-neighbours' emissions, o_social_i = sum_j T_ij q_j (T row-
              stochastic). The READ-OUT is a swappable function; default is this
              trust-weighted mixture (the softmax-precision reliability lives in
              A_social during inference, matching notebook 15). The seam is kept
              open so a cross-inhibition read-out, or a slow shared stigmergic
              field, can drop in later (Phase 3, mediating-field note).
  3. ACT      each agent picks an experiment from its EFE policy posterior
              (incl. the belief-utility term) and the environment returns an
              outcome sampled from the TRUE paradigm theta*.
  4. INFER    each agent does the exact categorical Bayes update on
              (own outcome, social signal), with a per-agent social_mask =
              trust-precision down-weight.

Hidden truth theta* is FIXED (B = identity): no transition, the posterior is
carried forward as the next prior. Paradigm "capture" = the population belief
locking onto a paradigm.

State is carried in a plain frozen ``PopulationState`` (numpy/JAX arrays); the
step is a pure function so notebooks can scan it without a stateful object.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from src.network import build_adjacency
from src.pomdp import agent_pop
from src.pomdp.gen_model import PomdpConfig, build_generative_model

EPS = 1e-12


# ----------------------------------------------------------------------
# Trust matrix (Phase 1: fixed from the graph; learning is Phase 2)
# ----------------------------------------------------------------------

def build_trust(cfg: PomdpConfig, n_agents: int, mean_degree: int = 4,
                kind: str = "watts_strogatz", rewiring_p: float = 0.1,
                seed: int = 0) -> jnp.ndarray:
    """Row-stochastic trust matrix T from the adjacency graph.

    Phase 1: uniform trust over the closed neighbourhood (self included), so the
    social-channel STRENGTH is carried entirely by the reliability kappa in
    ``A_social`` (the swept bifurcation knob), not by T. Trust LEARNING (a
    reliability estimate per edge) is Phase 2.
    """
    A = build_adjacency(n_agents=n_agents, mean_degree=mean_degree,
                        rewiring_p=rewiring_p, seed=seed, kind=kind)
    A_self = A + np.eye(n_agents, dtype=A.dtype)         # include self
    T = A_self / (A_self.sum(axis=1, keepdims=True) + EPS)
    return jnp.asarray(T)


# ----------------------------------------------------------------------
# Read-out: emissions -> per-agent social observation (SWAPPABLE)
# ----------------------------------------------------------------------

def readout_trust_mixture(emissions: jax.Array, T: jax.Array) -> jax.Array:
    """Default read-out: o_social_i = sum_j T_ij * q_j  (trust-weighted mean).

    emissions : (N, K) soft paradigm signals; T : (N, N) row-stochastic.
    Returns (N, K) soft categorical social observations. The softmax-precision
    nonlinearity that produced the NB15 fold lives downstream in A_social, not
    here — keeping this read-out linear makes the seam clean for a later
    cross-inhibition / stigmergic-field replacement.
    """
    return T @ emissions  # (N, K)


# ----------------------------------------------------------------------
# Population state + step
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PopulationState:
    q: jax.Array            # (N, K) beliefs over paradigms
    U: jax.Array            # (N, K) per-agent belief-utility (heterogeneity)
    key: jax.Array          # PRNG


def init_state(cfg: PomdpConfig, n_agents: int,
               U_per_agent: np.ndarray | None = None,
               D_per_agent: np.ndarray | None = None,
               seed: int = 0) -> PopulationState:
    """Initial beliefs (from D, optionally per-agent) and belief-utilities."""
    gm = build_generative_model(cfg)
    if D_per_agent is None:
        q0 = jnp.broadcast_to(gm["D"], (n_agents, cfg.n_paradigms))
    else:
        d = jnp.asarray(D_per_agent)
        q0 = d / (d.sum(axis=1, keepdims=True) + EPS)
    if U_per_agent is None:
        U = jnp.broadcast_to(gm["U"], (n_agents, cfg.n_paradigms))
    else:
        U = jnp.asarray(U_per_agent)
    return PopulationState(q=q0, U=U, key=jax.random.PRNGKey(seed))


def step(state: PopulationState, gm: dict, T: jax.Array, cfg: PomdpConfig,
         social_mask: jax.Array | float = 1.0,
         bu_mode: str = "plan_tilt",
         readout=readout_trust_mixture) -> tuple[PopulationState, dict]:
    """One inter-agent round. Returns (new_state, step_info).

    social_mask : per-agent (N,) or scalar trust-precision down-weight in [0,1].
    bu_mode     : belief-utility mechanism (see agent_pop.efe_terms).
    """
    N = state.q.shape[0]
    key_act, key_obs, key_next = jax.random.split(state.key, 3)
    mask = jnp.broadcast_to(jnp.asarray(social_mask, dtype=float), (N,))

    # 1. EMIT: soft belief signals.
    emissions = state.q                                     # (N, K)

    # 2. ROUTE: trust-weighted social observation per agent.
    o_social = readout(emissions, T)                        # (N, K)

    # 3. ACT: EFE policy posterior (with belief-utility) -> sample experiment.
    q_pi, terms = agent_pop.policy_posterior_batch(
        state.q, gm["A_world"], gm["C"], state.U,
        cfg.beta_U, cfg.gamma_policy, bu_mode=bu_mode,
    )                                                       # q_pi (N, A)
    act_keys = jax.random.split(key_act, N)
    actions = jax.vmap(agent_pop.sample_action)(q_pi, act_keys)   # (N,)

    # 4a. ENVIRONMENT: sample outcome from the TRUE paradigm theta*.
    A_world = gm["A_world"]                                  # (n_o, K, A)
    p_o = A_world[:, cfg.true_paradigm, actions].T          # (N, n_o)
    obs_keys = jax.random.split(key_obs, N)
    o_idx = jax.vmap(lambda k, p: jax.random.categorical(k, jnp.log(p + EPS)))(obs_keys, p_o)
    o_world = jax.nn.one_hot(o_idx, cfg.n_o)                 # (N, n_o)

    # 4b. INFER: exact categorical Bayes on (own outcome, social signal).
    q_new = agent_pop.infer_state_batch(
        state.q, A_world, o_world, actions, gm["A_social"], o_social, mask,
    )                                                       # (N, K)

    new_state = PopulationState(q=q_new, U=state.U, key=key_next)
    info = {
        "q_pi": np.asarray(q_pi),
        "actions": np.asarray(actions),
        "o_idx": np.asarray(o_idx),
        "neg_efe": np.asarray(terms["neg_efe"]),
        "epistemic": np.asarray(terms["epistemic"]),
        "belief_utility": np.asarray(terms["belief_utility"]),
        "mean_q": np.asarray(q_new.mean(axis=0)),
    }
    return new_state, info


def run(cfg: PomdpConfig, n_agents: int, n_steps: int,
        U_per_agent: np.ndarray | None = None,
        D_per_agent: np.ndarray | None = None,
        social_mask: jax.Array | float = 1.0,
        bu_mode: str = "plan_tilt",
        graph_kwargs: dict | None = None,
        seed: int = 0) -> dict:
    """Run the population for n_steps. Returns trajectories of mean belief and
    the per-step info (occupancy, EFE terms). Thin Python loop — fine for the
    Phase-1 scaffold; ``jax.lax.scan`` is an easy later optimisation."""
    gm = build_generative_model(cfg)
    T = build_trust(cfg, n_agents, seed=seed, **(graph_kwargs or {}))
    state = init_state(cfg, n_agents, U_per_agent, D_per_agent, seed=seed)

    mean_qB = np.empty(n_steps)
    occ_B = np.empty(n_steps)
    infos = []
    for t in range(n_steps):
        state, info = step(state, gm, T, cfg, social_mask, bu_mode)
        mean_qB[t] = info["mean_q"][cfg.true_paradigm]
        occ_B[t] = float(np.mean(np.asarray(state.q)[:, cfg.true_paradigm] > 0.5))
        infos.append(info)
    return {
        "mean_qB": mean_qB,        # population mean belief in the TRUE paradigm
        "occ_B": occ_B,            # fraction whose MAP = true paradigm
        "final_q": np.asarray(state.q),
        "infos": infos,
        "T": np.asarray(T),
    }
