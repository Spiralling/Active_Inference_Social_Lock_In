"""Tests for the class-based kernel (src/structural/kernel.py).

Layer: structural (THE PAPER OBJECT) · Guards: src/structural/kernel.py ·
Map: tests/README.md

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


@pytest.mark.parametrize("mode", ["heuristic", "derived", "derived_live"])
def test_run_trace_net_final_matches_run_final(mode):
    """The full-net trace shares the ``_transition`` body with every other
    rollout, so its LAST frame must equal ``run_final`` bit-for-bit."""
    cfg = StructuralConfig(n_agents=9, n_steps=15, precision_mode=mode,
                           core_governance=40.0, experiment_bias=0.5)
    ref = S.run_final(cfg, S.init_state(cfg, KEY))

    Pi_t, h_t = Network.init(cfg, KEY).run_trace_net()

    assert Pi_t.shape == (cfg.n_steps, cfg.n_agents,
                          len(cfg.node_names), len(cfg.node_names))
    assert h_t.shape == (cfg.n_steps, cfg.n_agents, len(cfg.node_names))
    assert np.allclose(Pi_t[-1], np.asarray(ref.Pi), atol=1e-5)
    assert np.allclose(h_t[-1], np.asarray(ref.h), atol=1e-5)


def test_run_trace_net_order_parameter_matches_run_trace():
    """The order parameter recomputed off the last full-net frame must equal the
    last value of ``run_trace``'s order-parameter curve -- so the notebook's
    in-cell sanity check is sound."""
    cfg = StructuralConfig(n_agents=10, n_steps=20)
    net = Network.init(cfg, KEY)
    ms_ref, _ = net.run_trace()
    Pi_t, h_t = Network.init(cfg, KEY).run_trace_net()

    m_last = obs.order_parameter(
        jnp.asarray(Pi_t[-1]), jnp.asarray(h_t[-1]), cfg.node_names,
        ph.DISAGREEMENT_NODES, cfg.mu_phlog_mass, cfg.mu_oxy_mass)
    assert np.allclose(float(m_last), float(ms_ref[-1]), atol=1e-5)


def test_run_bridge_net_order_parameter_matches_run_bridge():
    """The full-net bridge trace must agree with ``run_bridge``'s order-parameter
    curve frame-for-frame (same scan body, same W schedule) -- so the notebook can
    derive the conditional structure off ``run_bridge_net`` and trust it lines up
    with the headline m(t)."""
    from src.config import NetworkConfig
    groups = [{"count": 9, "paradigm": "phlogiston", "stance": -1.0, "prec_scale": 8.0},
              {"count": 3, "paradigm": "oxygen"}]
    cfg = StructuralConfig(n_agents=12, n_steps=30, precision_mode="heuristic",
                           experiment_bias=1.0, t_shift=0,
                           network=NetworkConfig(kind="planted_sbm",
                                                 intra_prob=0.5, inter_prob=0.0))
    net = Network.init(cfg, KEY, groups)
    ms_ref, _ = net.run_bridge(10, 0.3, groups)
    Pi_t, h_t = Network.init(cfg, KEY, groups).run_bridge_net(10, 0.3, groups)

    m_t = np.array([
        float(obs.order_parameter(jnp.asarray(Pi_t[t]), jnp.asarray(h_t[t]),
                                  cfg.node_names, ph.DISAGREEMENT_NODES,
                                  cfg.mu_phlog_mass, cfg.mu_oxy_mass))
        for t in range(cfg.n_steps)])
    assert np.allclose(m_t, np.asarray(ms_ref), atol=1e-5)


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
    and the static cfg/graph are not leaves."""
    cfg = StructuralConfig(n_agents=8)
    net = Network.init(cfg, KEY)
    leaves = jax.tree_util.tree_leaves(net)
    # all leaves are arrays (cfg/graph/names are static, not leaves)
    assert all(isinstance(x, jax.Array) for x in leaves)
    assert any(x.shape == (8,) + net.agents.Pi.shape[1:] for x in leaves)


# ----------------------------------------------------------------------
# First-class graph layer: topology decoupled from beliefs (the rewrite).
# ----------------------------------------------------------------------

from src.structural import graphs


def test_network_stores_graph():
    """The Network carries a first-class graph (the topology), and its W is that
    graph's closed-neighbourhood normalisation -- not a throwaway adjacency."""
    cfg = StructuralConfig(n_agents=12)
    g = graphs.erdos_renyi(12, mean_degree=4, seed=0)
    net = Network.init(cfg, KEY, graph=g)
    assert net.graph is g
    assert jnp.allclose(net.W, g.trust_W())


def test_explicit_graph_decouples_topology_from_groups():
    """Topology and beliefs are independent: a heterogeneous belief population can
    sit on ANY graph family, with no groups-derived membership entanglement."""
    cfg = StructuralConfig(n_agents=10, n_steps=12)
    groups = [{"count": 5, "stance": -1.0, "prec_scale": 3.0},
              {"count": 5, "stance": 1.0, "prec_scale": 1.0}]
    for g in [graphs.complete(10), graphs.scale_free(10, mean_degree=4, seed=0),
              graphs.ring(10, mean_degree=2)]:
        net = Network.init(cfg, KEY, groups=groups, graph=g)
        assert net.graph.kind == g.kind
        out = net.run_final()                      # runs end to end on each topology
        assert out.agents.Pi.shape[0] == 10


def test_default_path_matches_explicit_config_graph():
    """Omitting ``graph`` builds it from cfg.network; passing the same graph
    explicitly must give an identical W (the default path is just sugar)."""
    from src.config import NetworkConfig
    cfg = StructuralConfig(n_agents=20,
                           network=NetworkConfig(kind="scale_free", mean_degree=4))
    implicit = Network.init(cfg, KEY)
    explicit = Network.init(cfg, KEY,
                            graph=graphs.scale_free(20, mean_degree=4, seed=cfg.seed))
    assert jnp.allclose(implicit.W, explicit.W)


def test_isolated_is_graph_operation():
    """``isolated`` removes the graph's edges (W -> identity) AND keeps topology and
    fusion matrix consistent -- it is now a graph op, not W-surgery."""
    cfg = StructuralConfig(n_agents=9)
    net = Network.init(cfg, KEY, graph=graphs.complete(9))
    iso = net.isolated()
    assert iso.graph.kind == "isolated"
    assert jnp.allclose(iso.W, jnp.eye(9))


def test_bridge_reads_membership_from_graph_not_groups():
    """``run_bridge`` needs no ``groups`` argument: the community membership lives
    on the stored graph. A community graph bridges; a non-community graph raises."""
    cfg = StructuralConfig(n_agents=16, n_steps=20, precision_mode="heuristic",
                           experiment_bias=1.0, t_shift=0)
    netc = Network.init(cfg, KEY,
                        graph=graphs.community([8, 8], intra=0.5, inter=0.0, seed=0))
    ms, _ = netc.run_bridge(10, 0.3)               # no groups threaded through
    assert ms.shape == (cfg.n_steps,)

    net_plain = Network.init(cfg, KEY, graph=graphs.complete(16))
    with pytest.raises(ValueError):
        net_plain.run_bridge(10, 0.3)              # no community membership
