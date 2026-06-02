"""Tests for the class-based kernel (src/structural/kernel.py).

The kernel (``Paradigm`` / ``Agent`` / ``Network``) is a *facade* over the trusted
``step`` rollouts, so the contract is equivalence: a ``Network`` rollout must equal
the legacy ``step.run_*`` it delegates to, bit-for-bit (up to float tolerance),
across every precision mode and a heterogeneous-``groups`` population. If these pass,
the readable object layer provably carries no behavioural drift -- the whole point of
the refactor.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.structural.phlogiston import StructuralConfig
from src.structural import step as S
from src.structural import observables as obs
from src.structural import phlogiston as ph
from src.structural.belief import add_fisher
from src.structural.world import sample_o, fisher_deposit_weighted
from src.structural.kernel import Network, Agent, Paradigm


KEY = jax.random.PRNGKey(0)


# ----------------------------------------------------------------------
# Network <-> step equivalence (the core contract).
# ----------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["heuristic", "derived", "derived_live"])
def test_run_final_matches_step(mode):
    cfg = StructuralConfig(n_agents=12, n_steps=18, precision_mode=mode,
                           core_governance=50.0, experiment_bias=0.5)
    state = S.init_state(cfg, KEY)
    ref = S.run_final(cfg, state)

    net = Network.init(cfg, KEY)
    out = net.run_final()

    assert jnp.allclose(out.agents.Pi, ref.Pi, atol=1e-5)
    assert jnp.allclose(out.agents.h, ref.h, atol=1e-5)


@pytest.mark.parametrize("mode", ["heuristic", "derived", "derived_live"])
def test_run_trace_matches_step(mode):
    cfg = StructuralConfig(n_agents=10, n_steps=20, precision_mode=mode,
                           core_governance=30.0, experiment_bias=0.4)
    state = S.init_state(cfg, KEY)
    ms_ref, gw_ref = S.run_trace(cfg, state)

    ms, gw = Network.init(cfg, KEY).run_trace()

    assert np.allclose(ms, np.asarray(ms_ref), atol=1e-5)
    assert np.allclose(gw, np.asarray(gw_ref), atol=1e-5)


def test_run_trace_precision_matches_step():
    cfg = StructuralConfig(n_agents=8, n_steps=16, precision_mode="derived",
                           core_governance=100.0)
    state = S.init_state(cfg, KEY)
    ms_ref, gw_ref, rho_ref = S.run_trace_precision(cfg, state)

    ms, gw, rho = Network.init(cfg, KEY).run_trace_precision()

    assert np.allclose(ms, np.asarray(ms_ref), atol=1e-5)
    assert np.allclose(rho, np.asarray(rho_ref), atol=1e-5)


def test_groups_population_matches_step():
    """Heterogeneous init (an entrenched high-precision bloc + reformers) must
    flow through the kernel identically to ``step.init_state(..., groups)``."""
    cfg = StructuralConfig(n_agents=10, n_steps=18)
    groups = [{"count": 5, "stance": -1.0, "prec_scale": 3.0, "label": "entrenched"},
              {"count": 5, "stance": 1.0, "prec_scale": 1.0, "label": "reformers"}]
    state = S.init_state(cfg, KEY, groups)
    ref = S.run_final(cfg, state)

    out = Network.init(cfg, KEY, groups).run_final()

    assert jnp.allclose(out.agents.Pi, ref.Pi, atol=1e-5)
    assert jnp.allclose(out.agents.h, ref.h, atol=1e-5)


def test_sbm_communities_disconnect_and_couple():
    """planted_sbm with one society per group: inter_prob=0 gives genuinely
    disconnected communities (zero cross-community trust -- echo chambers); inter_prob>0
    couples them."""
    from src.config import NetworkConfig
    groups = [{"count": 12, "paradigm": "phlogiston", "stance": -1.0},
              {"count": 12, "paradigm": "oxygen"}]
    cfg0 = StructuralConfig(n_agents=24,
                            network=NetworkConfig(kind="planted_sbm",
                                                  intra_prob=0.3, inter_prob=0.0))
    W0 = S.init_state(cfg0, KEY, groups).W
    cross0 = float(W0[:12, 12:].sum() + W0[12:, :12].sum())
    assert cross0 == 0.0                       # disconnected: no cross-community trust

    cfg1 = dataclasses_replace_inter(cfg0, 0.3)
    W1 = S.init_state(cfg1, KEY, groups).W
    cross1 = float(W1[:12, 12:].sum() + W1[12:, :12].sum())
    assert cross1 > 0.0                        # coupled


def dataclasses_replace_inter(cfg, inter):
    import dataclasses
    from src.config import NetworkConfig
    nc = dataclasses.replace(cfg.network, inter_prob=inter)
    return dataclasses.replace(cfg, network=nc)


def test_init_reuses_step_state():
    """``Network.init`` is exactly ``step.init_state`` re-wrapped -- same priors,
    same adjacency-derived weights ``W``."""
    cfg = StructuralConfig(n_agents=14)
    state = S.init_state(cfg, KEY)
    net = Network.init(cfg, KEY)
    assert jnp.allclose(net.agents.Pi, state.Pi)
    assert jnp.allclose(net.agents.h, state.h)
    assert jnp.allclose(net.W, state.W)
    assert net.names == state.names


# ----------------------------------------------------------------------
# Single-agent readable path (the comprehension surface).
# ----------------------------------------------------------------------

def test_agent_from_belief_roundtrip():
    cfg = StructuralConfig()
    para = Paradigm(cfg=cfg, name="phlogiston")
    prior = para.prior()
    a = Agent.from_belief(prior)
    assert jnp.allclose(a.belief.Pi, prior.Pi)
    assert jnp.allclose(a.belief.h, prior.h)
    assert a.names == prior.names


def test_agent_update_equals_add_fisher():
    """``Agent.update`` is precisely ``belief.add_fisher`` (no second copy)."""
    cfg = StructuralConfig()
    prior = Paradigm(cfg=cfg, name="phlogiston").prior()
    J = jnp.eye(prior.Pi.shape[0]) * 0.3
    j = jnp.ones(prior.h.shape[0]) * 0.2
    a = Agent.from_belief(prior).update(J, j)
    ref = add_fisher(prior, J, j)
    assert jnp.allclose(a.Pi, ref.Pi)
    assert jnp.allclose(a.h, ref.h)


def test_agent_observe_equals_manual_deposit():
    """One agent's ``observe`` == sample_o -> fisher_deposit_weighted -> add_fisher
    done by hand with the same key."""
    cfg = StructuralConfig()
    para = Paradigm(cfg=cfg, name="phlogiston")
    prior = para.prior()
    H = para.H()
    phi = para.phi_true_at(0)
    m = H.shape[0]
    w = jnp.ones(m)
    k = jax.random.PRNGKey(7)

    a = Agent.from_belief(prior).observe(H, phi, cfg.sigma_o, w, k)

    o = sample_o(H, phi, cfg.sigma_o, k)
    J, j = fisher_deposit_weighted(H, o, cfg.sigma_o, w)
    ref = add_fisher(prior, J, j)
    assert jnp.allclose(a.Pi, ref.Pi)
    assert jnp.allclose(a.h, ref.h)


def test_network_agent_slice():
    cfg = StructuralConfig(n_agents=9)
    net = Network.init(cfg, KEY)
    a = net.agent(3)
    assert a.Pi.shape == (len(net.names), len(net.names))
    assert jnp.allclose(a.Pi, net.agents.Pi[3])
    # readable scorers run on a single agent
    assert a.posterior_mean().shape == (len(net.names),)
    assert jnp.ndim(a.log_evidence()) == 0


# ----------------------------------------------------------------------
# Determinism / pytree sanity.
# ----------------------------------------------------------------------

def test_run_final_is_deterministic():
    cfg = StructuralConfig(n_agents=10, n_steps=12)
    a = Network.init(cfg, KEY).run_final()
    b = Network.init(cfg, KEY).run_final()
    assert jnp.allclose(a.agents.Pi, b.agents.Pi)


def test_network_is_pytree():
    """A Network is a JAX pytree: its array leaves include the agent stack and W,
    and the static cfg is not a leaf."""
    cfg = StructuralConfig(n_agents=8)
    net = Network.init(cfg, KEY)
    leaves = jax.tree_util.tree_leaves(net)
    # all leaves are arrays (cfg/names are static, not leaves)
    assert all(isinstance(x, jax.Array) for x in leaves)
    assert any(x.shape == (8,) + net.agents.Pi.shape[1:] for x in leaves)
