"""The class-based kernel: ``Paradigm`` / ``Agent`` / ``Network``.

A readable, object-shaped *facade* over the structural primitives. Nothing here
is new mathematics — every method delegates to a function that already exists and
is already tested:

  * ``Agent.update``   -> ``belief.add_fisher``
  * ``Agent.observe``  -> ``world.sample_o`` + ``world.fisher_deposit_weighted``
  * ``Agent.receive``  -> ``step.fuse``
  * ``Network.init``   -> ``step.init_state`` (host-side graph build)
  * ``Network.run_*``  -> ``step.run_final`` / ``step.run_trace`` (the scan body
                          is ``step._transition``), wrapped in ``eqx.filter_jit``.

so there is structurally only *one* implementation of the dynamics; the classes
just give you a surface you can hold one agent at a time.

Why ``equinox.Module`` (and not a plain dataclass or a bag of arrays): an
``eqx.Module`` is a JAX pytree, so a *stack* of agents is literally an ``Agent``
whose ``Pi``/``h`` leaves carry a leading ``N`` axis, and ``jax.lax.scan`` /
``jax.vmap`` operate on the object directly. The single readable agent and the
batched compiled sweep are the *same code*. This mirrors ``src/population.py``
(the v2 scalar model), which is already an ``eqx.Module`` with a static ``cfg``
and an ``@eqx.filter_jit`` core; we copy that proven template.

The JIT boundary (the locked design decision -- see the plan):
  * ``Network.init`` is HOST-side: networkx adjacency, heterogeneous ``groups``
    priors, ``trust_weights``. Never traced.
  * ``Network.run_final`` / ``run_trace`` are ``@eqx.filter_jit`` over a single
    ``lax.scan``; the regime schedule ``phi_true_at`` unrolls at trace time
    (``cfg.n_steps`` is static) and the trajectory is read to the host exactly
    ONCE, at the end -- never the per-step ``float(m)`` device sync of the legacy
    ``step.run``.
``cfg`` is an ``eqx.field(static=True)`` frozen dataclass, so every Python branch
inside the scan body (``cfg.precision_mode``, ``names.index(...)``) resolves at
trace time -- exactly as ``population.py`` branches on ``cfg.policy.objective``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.network import build_adjacency
from src.structural import belief, bmr, world, step
from src.structural import phlogiston as ph
from src.structural.belief import GaussianBeliefNet
from src.structural.phlogiston import StructuralConfig


# ----------------------------------------------------------------------
# Paradigm -- the generative model handed to an agent at init.
# ----------------------------------------------------------------------

class Paradigm(eqx.Module):
    """The model an agent is given: a *prior topology* + the observation operator
    + the node vocabulary. A pure-static object (no array leaves) -- it is the OO
    face over the free functions in ``phlogiston.py``.

    ``name`` selects which of the two candidate prior topologies this paradigm is
    (``"phlogiston"`` -- the dense common-cause hub, or ``"oxygen"`` -- the direct
    mass law). Both live on the same ``cfg.node_names`` basis and share the same
    likelihood ``H`` (the locked design decision: paradigms differ by prior, never
    by likelihood), which is why their evidence can be compared at all.
    """

    cfg: StructuralConfig = eqx.field(static=True)
    name: str = eqx.field(static=True)

    @property
    def names(self) -> tuple[str, ...]:
        """The shared fusion basis (row/column order of every belief net)."""
        return self.cfg.node_names

    def prior(self) -> GaussianBeliefNet:
        """This paradigm's prior belief net -- the model the agent starts from
        (delegates to ``phlogiston.candidate_priors``)."""
        return ph.candidate_priors(self.cfg)[self.name]

    def H(self) -> jax.Array:
        """The (m, d) observation operator shared by both paradigms
        (``phlogiston.H_observable``)."""
        return ph.H_observable(self.cfg)

    def phi_true_at(self, t: int) -> jax.Array:
        """The world's true commitment vector at step ``t`` (the regime schedule;
        ``phlogiston.phi_true_at``). The *world* is not the paradigm -- this is
        carried here only so a single ``Paradigm`` is enough to drive the
        single-agent evidence race."""
        return ph.phi_true_at(self.cfg, t)

    def measured_nodes(self) -> tuple[str, ...]:
        """Row order of ``H`` -- every node except the hidden hub."""
        return ph.measured_nodes(self.cfg)

    def disagreement_mask(self) -> jax.Array:
        """(m,) 0/1 mask: 1 on the mass-law rows the two paradigms disagree on."""
        return ph.disagreement_row_mask(self.cfg)


# ----------------------------------------------------------------------
# Agent -- one agent's belief is a Gaussian Bayes net in information form.
# ----------------------------------------------------------------------

class Agent(eqx.Module):
    """A single agent whose entire model is a ``GaussianBeliefNet`` ``(Pi, h)``.

    ``Pi`` (d, d), ``h`` (d,) are dynamic array leaves; ``names`` is static. The
    methods are the per-agent lifecycle -- OBSERVE -> UPDATE, plus RECEIVE (fuse a
    neighbour's net) and the read-only scorers -- each a one-liner over an existing
    primitive.

    Batched vs single. Because ``Agent`` is a pytree, a *population* of N agents is
    an ``Agent`` whose leaves are ``Pi`` (N, d, d) / ``h`` (N, d); ``Network``
    drives that stacked form through ``jax.lax.scan``. The instance *methods* below
    (``observe``/``posterior_mean``/``log_evidence``) assume a SINGLE agent (2-D
    ``Pi``) -- they are the readable/debuggable path, e.g. ``net.agent(0).observe(...)``
    in a notebook. The batched dynamics never call them; they call the same
    underlying functions through ``step._transition``'s ``vmap`` -- so there is no
    second copy of the math, only two call shapes.
    """

    Pi: jax.Array
    h: jax.Array
    names: tuple[str, ...] = eqx.field(static=True)

    # -- construction --------------------------------------------------

    @classmethod
    def from_belief(cls, net: GaussianBeliefNet) -> "Agent":
        """Wrap a ``GaussianBeliefNet`` as an ``Agent`` (e.g. ``Agent.from_belief(
        paradigm.prior())`` -- an agent handed a paradigm's prior at init)."""
        return cls(Pi=net.Pi, h=net.h, names=net.names)

    @property
    def belief(self) -> GaussianBeliefNet:
        """This agent's model as a ``GaussianBeliefNet`` (single-agent view)."""
        return GaussianBeliefNet(Pi=self.Pi, h=self.h, names=self.names)

    # -- lifecycle: observe / update -----------------------------------

    def update(self, J: jax.Array, j: jax.Array) -> "Agent":
        """LEARN: deposit one observation's Fisher information ``(J, j)`` into the
        belief (delegates to ``belief.add_fisher``: ``Pi += J``, ``h += j``)."""
        net = belief.add_fisher(self.belief, J, j)
        return Agent(Pi=net.Pi, h=net.h, names=self.names)

    def observe(self, H: jax.Array, phi_true: jax.Array, sigma_o: float,
                weights: jax.Array, key: jax.Array) -> "Agent":
        """ACT + OBSERVE in one: draw a (weighted) experiment from the world and
        fold its information in. ``weights`` (m,) are the per-channel evidential
        precisions ``rho_k`` (1 = full attention, 0 = experiment skipped). Delegates
        to ``world.sample_o`` then ``world.fisher_deposit_weighted`` then
        ``self.update`` -- the single-agent mirror of one ``vmap`` lane of
        ``step._transition``."""
        o = world.sample_o(H, phi_true, sigma_o, key)
        J, j = world.fisher_deposit_weighted(H, o, sigma_o, weights)
        return self.update(J, j)

    # -- lifecycle: receive (fuse a peer's net) ------------------------

    def receive(self, other: "Agent") -> "Agent":
        """RECEIVE: fuse a peer's whole belief net by precision addition
        (``belief.combine``: ``Pi1 + Pi2``, ``h1 + h2``). Full-communication
        pooling of two equally-trusted agents; the precision-weighted mean rides
        along for free in ``h``. (Trust-weighted pooling over a whole neighbourhood
        is ``step.fuse``, which ``Network`` uses.)"""
        net = belief.combine(self.belief, other.belief)
        return Agent(Pi=net.Pi, h=net.h, names=self.names)

    # -- read-only scorers ---------------------------------------------

    def posterior_mean(self) -> jax.Array:
        """Posterior mean ``mu = Pi^{-1} h`` (solved, never inverted)."""
        return self.belief.mean()

    def log_evidence(self) -> jax.Array:
        """Closed-form Gaussian log-evidence (``bmr.log_evidence``)."""
        return bmr.log_evidence(self.belief)


# ----------------------------------------------------------------------
# Network -- adjacency + the stacked agents + the compiled round loop.
# ----------------------------------------------------------------------

class Network(eqx.Module):
    """The social graph: a *stack* of ``Agent`` on a shared basis, plus the
    row-stochastic trust/fusion weights ``W`` and the PRNG key.

    ``init`` builds the graph host-side (the user's "initialise the graph"); the
    ``run_*`` methods are the compiled rollout (the user's "then compile"). The
    per-round dynamics -- FUSE the trusted neighbourhood, then weighted-OBSERVE --
    are reused verbatim from ``step`` (``step._transition``); ``Network`` only adds
    the ``eqx.filter_jit`` wrapper and the object surface.
    """

    cfg: StructuralConfig = eqx.field(static=True)
    agents: Agent                 # stacked: Pi (N, d, d), h (N, d)
    W: jax.Array                  # (N, N) row-stochastic trust/fusion weights
    key: jax.Array

    # -- construction (host-side; never traced) ------------------------

    @classmethod
    def init(cls, cfg: StructuralConfig, key: jax.Array,
             groups: list[dict] | None = None) -> "Network":
        """Build the graph and hand each agent its model. Reuses
        ``step.init_state`` (networkx adjacency via ``src.network.build_adjacency``,
        homogeneous or heterogeneous ``groups`` priors, ``trust_weights``), then
        re-wraps the stacked ``(Pi, h)`` as an ``Agent`` pytree. All host-side --
        this is setup, deliberately outside the JIT boundary."""
        state = step.init_state(cfg, key, groups)
        agents = Agent(Pi=state.Pi, h=state.h, names=state.names)
        return cls(cfg=cfg, agents=agents, W=state.W, key=state.key)

    @property
    def n_agents(self) -> int:
        return self.agents.Pi.shape[0]

    @property
    def names(self) -> tuple[str, ...]:
        return self.agents.names

    def agent(self, i: int) -> Agent:
        """Slice out a single agent (the readable/debuggable view, e.g.
        ``net.agent(0).posterior_mean()``)."""
        return Agent(Pi=self.agents.Pi[i], h=self.agents.h[i],
                     names=self.agents.names)

    def _state(self) -> step.PopulationState:
        """Repackage as the legacy ``PopulationState`` so the trusted ``step.run_*``
        functions can be reused unchanged. Cheap (no copy of array data)."""
        return step.PopulationState(Pi=self.agents.Pi, h=self.agents.h,
                                    names=self.agents.names, W=self.W, key=self.key)

    def _with_arrays(self, Pi: jax.Array, h: jax.Array,
                     key: jax.Array) -> "Network":
        """Return a new ``Network`` with the agent stack / key replaced (immutable
        update via ``eqx.tree_at``, mirroring ``population.py``)."""
        agents = Agent(Pi=Pi, h=h, names=self.agents.names)
        return eqx.tree_at(lambda n: (n.agents, n.key), self, (agents, key))

    # -- compiled rollout ----------------------------------------------

    def run_final(self) -> "Network":
        """Fast-forward to the final state -- a single ``@eqx.filter_jit`` ``scan``.
        Returns a new ``Network`` carrying the final beliefs. Read any observable
        off it afterwards (e.g. ``observables.order_parameter(net.agents.Pi, ...)``)."""
        Pi, h, key = _run_final_jit(self)
        return self._with_arrays(Pi, h, key)

    def run_trace(self) -> tuple[np.ndarray, np.ndarray]:
        """Full per-step trajectory in one compiled ``scan``. Returns
        ``(order_parameter_t, gravimetric_attention_t)`` as host arrays, each
        ``(n_steps,)`` -- the population paradigm-shift curve and the
        self-censorship trace. The device->host transfer happens ONCE here, not
        per step."""
        ms, gws = _run_trace_jit(self)
        return np.asarray(ms), np.asarray(gws)

    def run_trace_index(self) -> np.ndarray:
        """Per-agent oxygen-index trajectory ``(n_steps, N)`` in one compiled
        ``scan``. The population curve is ``.mean(axis=1)``; a per-school curve is
        the mean over that school's agent columns -- so the heterogeneous-bloc
        decomposition needs no python loop."""
        return np.asarray(_run_trace_index_jit(self))

    def run_trace_bmr(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-step ``(m_t, deltaF_t)`` in one compiled ``scan``: the population
        order parameter and the population-mean BMR Bayes factor for phlogiston's
        falsifiable commitment. ``deltaF`` crossing zero is the refutation signal;
        its lead over the ``m`` half-crossing is the early-warning lead-time (E3)."""
        ms, dFs = _run_trace_bmr_jit(self)
        return np.asarray(ms), np.asarray(dFs)

    # -- timed-bridge rollout (Model B: enablement by a split) ---------

    def _coupled_W(self, inter_prob: float, groups: list[dict]) -> jax.Array:
        """Rebuild the trust weights ``W`` for the *same* agents with the
        communities bridged at density ``inter_prob`` (host-side; same seed and
        ``intra_prob`` as ``init``, so the within-community structure is unchanged
        and only the cross-community edges are added). ``groups`` carries the
        community membership."""
        membership = [gi for gi, g in enumerate(groups) for _ in range(g["count"])]
        nc = self.cfg.network
        A = build_adjacency(
            n_agents=self.n_agents, mean_degree=nc.mean_degree,
            rewiring_p=nc.rewiring_p, seed=self.cfg.seed, kind="planted_sbm",
            society_membership=membership, intra_prob=nc.intra_prob,
            inter_prob=inter_prob)
        A_self = jnp.asarray(A) + jnp.eye(self.n_agents)
        return step.trust_weights(A_self)

    def _bridge_W_seq(self, t_incubate: int, inter_prob: float,
                      groups: list[dict]) -> jax.Array:
        """The per-step fusion schedule for a timed bridge: the network's current
        (split) ``W`` for the first ``t_incubate`` steps, then the coupled ``W`` for
        the rest of the horizon. ``(n_steps, N, N)``."""
        T = self.cfg.n_steps
        t_inc = max(0, min(int(t_incubate), T))
        W_coupled = self._coupled_W(inter_prob, groups)
        return jnp.stack([self.W] * t_inc + [W_coupled] * (T - t_inc))

    def run_bridge(self, t_incubate: int, inter_prob: float,
                   groups: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Model B rollout: the population fuses inside its disconnected communities
        (the network's current split ``W``, from an ``inter_prob=0`` init) for
        ``t_incubate`` steps of *protected incubation*, then a bridge of density
        ``inter_prob`` opens and stays open for the rest of the horizon. ``groups``
        is the same society spec used at ``init`` (it carries the membership needed
        to rebuild the coupled graph). Returns ``(m_t, grav_t)`` like ``run_trace``;
        ``t_incubate=0`` is identical to an always-coupled run."""
        W_seq = self._bridge_W_seq(t_incubate, inter_prob, groups)
        ms, gws = _run_trace_schedule_jit(self, W_seq)
        return np.asarray(ms), np.asarray(gws)

    def run_bridge_index(self, t_incubate: int, inter_prob: float,
                         groups: list[dict]) -> np.ndarray:
        """Per-agent oxygen-index trajectory ``(n_steps, N)`` for the timed bridge
        (the index sibling of ``run_bridge``): reads each community's conviction
        *through* the moment the bridge opens. The population curve is
        ``.mean(axis=1)``; a per-community curve is the mean over that community's
        agent columns."""
        W_seq = self._bridge_W_seq(t_incubate, inter_prob, groups)
        return np.asarray(_run_trace_index_schedule_jit(self, W_seq))

    def isolated(self) -> "Network":
        """A copy with communication switched off: ``W`` replaced by the identity,
        so each agent only ever sees its own data (the no-fusion baseline). Lets you
        contrast ``net.run_trace()`` against ``net.isolated().run_trace()``."""
        N = self.n_agents
        return eqx.tree_at(lambda n: n.W, self, jnp.eye(N, dtype=self.W.dtype))

    def run_trace_precision(self
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Like ``run_trace`` but also returns the per-disagreement-channel
        evidential precision ``rho`` over time (``observables`` for the derived-
        precision experiments). Returns ``(m_t, grav_t, rho_disagree_t)``."""
        ms, gws, rho = _run_trace_precision_jit(self)
        return np.asarray(ms), np.asarray(gws), np.asarray(rho)

    def __repr__(self) -> str:
        N = self.n_agents
        return (f"Network(N={N}, d={len(self.names)}, "
                f"n_steps={self.cfg.n_steps}, "
                f"precision_mode={self.cfg.precision_mode!r})")


# ----------------------------------------------------------------------
# JIT-compiled cores. Each takes the ``Network`` (an eqx.Module pytree, so its
# array leaves are traced and ``cfg`` is static) and delegates the dynamics to the
# trusted ``step`` rollouts -- the PopulationState is rebuilt INSIDE the trace from
# already-traced arrays, so it never reaches jit as a (non-pytree) argument.
# ----------------------------------------------------------------------

@eqx.filter_jit
def _run_final_jit(net: Network) -> tuple[jax.Array, jax.Array, jax.Array]:
    final = step.run_final(net.cfg, net._state())
    return final.Pi, final.h, final.key


@eqx.filter_jit
def _run_trace_jit(net: Network) -> tuple[jax.Array, jax.Array]:
    return step.run_trace(net.cfg, net._state())


@eqx.filter_jit
def _run_trace_index_jit(net: Network) -> jax.Array:
    return step.run_trace_index(net.cfg, net._state())


@eqx.filter_jit
def _run_trace_bmr_jit(net: Network) -> tuple[jax.Array, jax.Array]:
    return step.run_trace_bmr(net.cfg, net._state())


@eqx.filter_jit
def _run_trace_precision_jit(net: Network
                             ) -> tuple[jax.Array, jax.Array, jax.Array]:
    return step.run_trace_precision(net.cfg, net._state())


@eqx.filter_jit
def _run_trace_schedule_jit(net: Network, W_seq: jax.Array
                            ) -> tuple[jax.Array, jax.Array]:
    return step.run_trace_schedule(net.cfg, net._state(), W_seq)


@eqx.filter_jit
def _run_trace_index_schedule_jit(net: Network, W_seq: jax.Array) -> jax.Array:
    return step.run_trace_index_schedule(net.cfg, net._state(), W_seq)
