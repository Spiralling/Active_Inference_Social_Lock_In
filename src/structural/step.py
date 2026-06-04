"""The multi-agent coupling loop: full-communication fusion of belief nets.

The Gaussian-structural sibling of ``src/pomdp/step.py``. Each agent carries an
entire paradigm net ``(Pi_i, h_i)`` on the shared global basis. "Full
communication" (the user's framing) means a peer transmits its *whole* belief
object, and the receiver fuses by precision addition over its trusted
neighbourhood:

    Pi_i <- sum_j W_ij Pi_j ,    h_i <- sum_j W_ij h_j ,

with ``W`` row-stochastic over the closed neighbourhood (self + neighbours) ---
exactly the weighting of ``src/inference.py:precision_pool``, lifted from
scalars to matrices. The elegance: because ``h = Pi mu`` adds linearly, this one
addition *automatically* carries the precision-weighted mean; there is no
separate mean-mixing step (contrast ``precision_pool``, which needs two lines
because it carries ``mu`` and ``tau`` separately).

Per round: FUSE over the trust graph, then OBSERVE (each agent draws data from
the regime-dependent world and deposits its Fisher information). State is a
stack of nets; ``step`` is a pure function of ``(state, key)`` (mirrors the
purity contract of the POMDP step).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.config import NetworkConfig
from src.network import build_adjacency
from src.structural import phlogiston as ph
from src.structural.phlogiston import StructuralConfig
from src.structural.world import fisher_deposit, fisher_deposit_weighted, sample_o
from src.structural import observables as obs
from src.structural import precision as P
from src.structural import efe
from src.structural import bmr, linalg


EPS = 1e-12


def trust_weights(A_self: jax.Array, gamma: jax.Array | None = None
                  ) -> jax.Array:
    """Row-stochastic fusion weights over the closed neighbourhood.

    ``A_self`` : (N, N) adjacency + identity (1 where j is i or a neighbour of i).
    ``gamma``  : optional (N, N) trust precisions; defaults to uniform (all 1).

    Returns ``W`` with ``W[i, j] = gamma_ij A_self_ij / sum_k gamma_ik A_self_ik``
    --- the same construction as ``src/inference.py:precision_pool``
    (``weights_raw = gamma * mask``, row-normalise).
    """
    g = jnp.ones_like(A_self) if gamma is None else gamma
    raw = g * A_self
    return raw / (raw.sum(axis=1, keepdims=True) + EPS)


def fuse(Pi: jax.Array, h: jax.Array, W: jax.Array
         ) -> tuple[jax.Array, jax.Array]:
    """Trust-weighted precision-addition fusion of a stack of belief nets.

    ``Pi`` (N, d, d), ``h`` (N, d), ``W`` (N, N) row-stochastic. Returns the
    fused ``(Pi', h')`` with ``Pi'_i = sum_j W_ij Pi_j`` and ``h'_i = sum_j W_ij
    h_j``. The precision-weighted mean is recovered for free from the fused
    potential.
    """
    Pi_new = jnp.einsum("ij,jab->iab", W, Pi)
    h_new = W @ h
    return Pi_new, h_new


@dataclass(frozen=True)
class PopulationState:
    """Stacked belief nets for N agents on a shared basis (mirrors
    ``pomdp.step.PopulationState``)."""

    Pi: jax.Array            # (N, d, d) per-agent precision matrices
    h: jax.Array             # (N, d) per-agent potentials
    names: tuple[str, ...]   # shared d-node basis
    W: jax.Array             # (N, N) row-stochastic trust/fusion weights
    key: jax.Array           # PRNG


def _group_prior(cfg: StructuralConfig, paradigm: str, stance: float,
                 prec_scale: float):
    """Build one ``(stance, prec_scale)``-patched prior for a group, from the
    chosen paradigm's builder.

    ``paradigm='phlogiston'`` (default, the incumbent): ``stance`` overrides the
    disagreement-node prior mean ``mu_phlog_mass`` and ``prec_scale`` scales the
    confidence knobs ``base_prec``/``hub_self_prec`` -- exactly the original
    homogeneous-groups behaviour.

    ``paradigm='oxygen'`` (the experimentalist *vanguard* -- agents who have
    already measured, so their net carries the anomaly<->mass-law coupling and the
    correct mean): built from ``ph.oxygen_prior``; ``stance`` overrides
    ``mu_oxy_mass`` and ``prec_scale`` also scales the gravimetric-block precision
    ``mass_prec`` (so ``prec_scale`` means "confidence in the mass law"). No new
    mechanism -- the vanguard is just this prior; its *disagreement* with the
    incumbents is what the EFE social term transmits."""
    if paradigm == "oxygen":
        cfg_g = dataclasses.replace(
            cfg,
            mu_oxy_mass=stance,
            base_prec=cfg.base_prec * prec_scale,
            hub_self_prec=cfg.hub_self_prec * prec_scale,
            mass_prec=cfg.mass_prec * prec_scale,
        )
        return ph.oxygen_prior(cfg_g)
    cfg_g = dataclasses.replace(
        cfg,
        mu_phlog_mass=stance,
        base_prec=cfg.base_prec * prec_scale,
        hub_self_prec=cfg.hub_self_prec * prec_scale,
    )
    return ph.phlogiston_prior(cfg_g)


def init_state(cfg: StructuralConfig, key: jax.Array,
               groups: list[dict] | None = None, *,
               W_override: jax.Array | None = None) -> PopulationState:
    """Initialise N agents on a social graph built by ``src/network.build_adjacency``
    (reused, not duplicated).

    ``W_override`` (keyword-only, default ``None``): supply the row-stochastic
    fusion matrix directly instead of deriving it from ``cfg.network`` here. The
    class layer (``kernel.Network.init``) uses this to drive the population from a
    first-class ``graphs.Graph`` (so the new graph families -- Erdos-Renyi, ring,
    lattice, complete -- and topology decoupled from the belief ``groups`` work
    without teaching this legacy builder every kind). ``None`` keeps the original
    behaviour byte-for-byte: the prior math below is identical either way; only the
    ``W`` source differs.

    ``groups=None`` (default): every agent holds the *same* phlogiston prior --
    the homogeneous baseline.

    ``groups``: a list of sub-population specs, each a dict with keys
    ``{"count": int, "stance": float, "prec_scale": float, "label": str,
       "paradigm": "phlogiston"|"oxygen", "stance_sd": float, "prec_sd": float}``.
    ``count`` and (for the homogeneous default) ``stance``/``prec_scale`` are as
    before. New optional keys:

      * ``paradigm`` (default ``"phlogiston"``): which prior builds the group --
        ``"oxygen"`` is the experimentalist *vanguard* (``ph.oxygen_prior``;
        already-measured agents that disagree with the incumbents on the anomaly
        node). See ``_group_prior``.
      * ``stance_sd`` / ``prec_sd`` (default ``0``): per-agent *within-group*
        heterogeneity. Each agent's stance is drawn ``~ N(stance, stance_sd)`` and
        its precision scale ``~ prec_scale * lognormal(0, prec_sd)``, and a prior is
        built per agent. ``sd = 0`` recovers the exact broadcast (byte-identical to
        the original homogeneous-groups path), so existing callers are unaffected.
        Non-zero sd gives a *contested middle* -- overlapping tails between camps --
        rather than two delta-spikes. Sampling uses a key derived from ``cfg.seed``
        (independent of the rollout key).

    Only the prior builders see the patched cfg; the world schedule and order
    parameter keep the canonical ``cfg``. Group counts must sum to
    ``cfg.n_agents``."""
    if groups is None:
        prior = ph.phlogiston_prior(cfg)
        N = cfg.n_agents
        Pi = jnp.broadcast_to(prior.Pi, (N,) + prior.Pi.shape).copy()
        h = jnp.broadcast_to(prior.h, (N,) + prior.h.shape).copy()
        names = prior.names
    else:
        N = sum(g["count"] for g in groups)
        if N != cfg.n_agents:
            raise ValueError(
                f"group counts sum to {N}, but cfg.n_agents={cfg.n_agents}")
        Pi_blocks, h_blocks = [], []
        names = cfg.node_names
        samp_key = jax.random.PRNGKey(cfg.seed)
        for g in groups:
            paradigm = g.get("paradigm", "phlogiston")
            count = g["count"]
            prec_scale = g.get("prec_scale", 1.0)
            default_stance = (cfg.mu_oxy_mass if paradigm == "oxygen"
                              else cfg.mu_phlog_mass)
            stance = g.get("stance", default_stance)
            stance_sd = g.get("stance_sd", 0.0)
            prec_sd = g.get("prec_sd", 0.0)

            if stance_sd == 0.0 and prec_sd == 0.0:
                # homogeneous group: build once, broadcast (the original path).
                p = _group_prior(cfg, paradigm, stance, prec_scale)
                Pi_blocks.append(jnp.broadcast_to(p.Pi, (count,) + p.Pi.shape))
                h_blocks.append(jnp.broadcast_to(p.h, (count,) + p.h.shape))
                names = p.names
            else:
                # Gaussian within-group heterogeneity: a prior per agent.
                samp_key, ks, kp = jax.random.split(samp_key, 3)
                stances = stance + stance_sd * jax.random.normal(ks, (count,))
                precs = prec_scale * jnp.exp(prec_sd * jax.random.normal(kp, (count,)))
                for a in range(count):
                    p = _group_prior(cfg, paradigm,
                                     float(stances[a]), float(precs[a]))
                    Pi_blocks.append(p.Pi[None])
                    h_blocks.append(p.h[None])
                    names = p.names
        Pi = jnp.concatenate(Pi_blocks, axis=0).copy()
        h = jnp.concatenate(h_blocks, axis=0).copy()

    if W_override is not None:
        return PopulationState(Pi=Pi, h=h, names=names, W=W_override, key=key)

    nc: NetworkConfig = cfg.network
    if nc.kind == "planted_sbm" and groups is not None:
        # Each group is a society: agents are wired densely WITHIN their group
        # (intra_prob) and sparsely BETWEEN groups (inter_prob). inter_prob=0
        # gives fully disconnected communities -- the echo-chamber setup, one
        # theory per community. Dial inter_prob up to couple the schools.
        membership = [gi for gi, g in enumerate(groups) for _ in range(g["count"])]
        A = build_adjacency(
            n_agents=N, mean_degree=nc.mean_degree, rewiring_p=nc.rewiring_p,
            seed=cfg.seed, kind="planted_sbm", society_membership=membership,
            intra_prob=nc.intra_prob, inter_prob=nc.inter_prob,
        )
    else:
        A = build_adjacency(
            n_agents=N, mean_degree=nc.mean_degree, rewiring_p=nc.rewiring_p,
            seed=cfg.seed, kind=nc.kind,
        )
    A_self = jnp.asarray(A) + jnp.eye(N)
    W = trust_weights(A_self)
    return PopulationState(Pi=Pi, h=h, names=names, W=W, key=key)


def _channel_weights(Pi: jax.Array, h: jax.Array, W: jax.Array,
                     cfg: StructuralConfig) -> jax.Array:
    """Per-agent observation weights over the ``H_observable`` rows, selected by
    ``cfg.precision_mode``. The weights are the per-channel evidential precision
    ``rho_k`` fed to ``fisher_deposit_weighted`` (``weights[k] == R^{-1}[k] sigma^2``);
    the deposit machinery is identical across modes, only the *source* of the gain
    differs. Branch is host-side on the static ``cfg.precision_mode`` string.

    ``heuristic``   : the imposed sigmoid gate -- attention falls on the disagreement
        rows once an agent *holds* phlogiston (the original behaviour; ``experiment_bias
        == 0`` => all-ones => plain deposit).
    ``derived``     : ``rho_k`` derived ONCE from the incumbent prior net's coupling-to-
        core (``phlogiston.derived_channel_precision``), broadcast across agents -- a
        paradigm-level constant, independent of any agent's live stance.
    ``derived_live``: ``rho_k`` from each agent's live (fused) covariance -- adaptive
        governance that tightens as a bloc entrenches.
    ``efe``         : expected-free-energy sampling (``src/structural/efe.py``) -- the
        ``derived`` pragmatic baseline PLUS the two prediction-error drives (self/world
        ``epistemic_weight`` and social/neighbour ``social_weight``). The social term
        reads the trust graph ``W``, so this is the only mode that needs it. With both
        weights 0 it is byte-identical to ``derived`` (the back-compat anchor).
    """
    n = Pi.shape[0]
    if cfg.observation_operator == "relational" and cfg.precision_mode == "efe":
        # The 'efe' drives index measured_nodes and build a base_rho over them, so they
        # do not yet line up with the relational operator's extra mass-balance row. Loud
        # guard rather than a cryptic shape error. Supported on the relational substrate:
        # 'heuristic' (default), 'derived' (paradigm-level gamma), 'derived_live' (the
        # per-agent adaptive gamma -- entrenchment self-silences via channel_precision_H_stack).
        raise NotImplementedError(
            "observation_operator='relational' is not yet wired for "
            "precision_mode='efe'; use 'heuristic', 'derived', or 'derived_live'.")
    if cfg.precision_mode == "heuristic":
        mass_idx = jnp.asarray([cfg.node_names.index(nm)
                                for nm in ph.DISAGREEMENT_NODES])
        mu = jax.vmap(jnp.linalg.solve)(Pi, h)                   # (N, d)
        frac = (mu[:, mass_idx].mean(axis=1) - cfg.mu_phlog_mass) \
            / (cfg.mu_oxy_mass - cfg.mu_phlog_mass)
        oxy = jnp.clip(frac, 0.0, 1.0)                           # (N,)
        return ph.attention_weights(oxy, cfg)                    # (N, m)
    if cfg.precision_mode == "derived":
        rho = ph.derived_channel_precision(cfg)                  # (m,) constant
        return jnp.broadcast_to(rho, (n, rho.shape[0]))          # (N, m)
    if cfg.precision_mode == "derived_live":
        if cfg.observation_operator == "relational":
            c = cfg.node_names.index(cfg.core_node)
            return P.channel_precision_H_stack(
                Pi, c, ph.gravimetric_H(cfg), cfg.core_governance, cfg.rho_max)  # (N, m)
        return P.channel_precision_stack(
            Pi, h, cfg.node_names, cfg.core_node, ph.measured_nodes(cfg),
            cfg.core_governance, cfg.rho_max, cfg.evidential_cost_kind)  # (N, m)
    if cfg.precision_mode == "efe":
        base_rho = ph.derived_channel_precision(cfg)             # (m,) pragmatic baseline
        measured_idx = jnp.asarray(
            [cfg.node_names.index(nm) for nm in ph.measured_nodes(cfg)])
        return efe.efe_channel_precision(
            Pi, h, W, base_rho, measured_idx,
            cfg.epistemic_weight, cfg.social_weight, cfg.rho_max)  # (N, m)
    raise ValueError(
        f"precision_mode must be 'heuristic'|'derived'|'derived_live'|'efe', "
        f"got {cfg.precision_mode!r}")


def _fuse_then_observe(Pi: jax.Array, h: jax.Array, W: jax.Array, key: jax.Array,
                       cfg: StructuralConfig, phi: jax.Array
                       ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``sharing_mode='posterior'``: peers transmit their *whole* belief net. FUSE
    (precision-addition over the trust graph) then weighted-OBSERVE. This is the original
    dynamics -- a conclusion (the full posterior, prior bias included) is what propagates."""
    Pi_f, h_f = fuse(Pi, h, W)
    H = ph.observation_operator(cfg)
    n = Pi.shape[0]
    key, *subs = jax.random.split(key, n + 1)
    subs = jnp.stack(subs)
    Wt = _channel_weights(Pi_f, h_f, W, cfg)                     # (N, m)

    def observe_one(Pi_i, h_i, k, w):
        o = sample_o(H, phi, cfg.sigma_o, k)
        J, j = fisher_deposit_weighted(H, o, cfg.sigma_o, w)
        return Pi_i + J, h_i + j

    Pi_new, h_new = jax.vmap(observe_one)(Pi_f, h_f, subs, Wt)
    return Pi_new, h_new, key


def _observe_pooled(Pi: jax.Array, h: jax.Array, W: jax.Array, key: jax.Array,
                    cfg: StructuralConfig, phi: jax.Array
                    ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``sharing_mode='observation'``: peers transmit *raw observations* (their per-step
    Fisher deposits), NOT their belief. Each agent keeps its OWN net (its prior never
    fuses) and adds the trust-weighted pool of its neighbours' deposits:

        Pi_i <- Pi_i + sum_j W_ij J_j ,    h_i <- h_i + sum_j W_ij j_j .

    Same trust graph ``W`` as the posterior mode (so the two differ ONLY in *what* is
    shared, not how strongly), but because the prior is never averaged in, a neighbour's
    prior *bias* does not propagate -- only the signal in its data does. Channel weights are
    read from each agent's OWN current belief (no fused belief exists here)."""
    H = ph.observation_operator(cfg)
    n = Pi.shape[0]
    key, *subs = jax.random.split(key, n + 1)
    subs = jnp.stack(subs)
    Wt = _channel_weights(Pi, h, W, cfg)                         # (N, m) from own belief

    def deposit_one(k, w):
        o = sample_o(H, phi, cfg.sigma_o, k)
        return fisher_deposit_weighted(H, o, cfg.sigma_o, w)     # (d,d), (d,)

    Js, js = jax.vmap(deposit_one)(subs, Wt)                     # (N,d,d), (N,d)
    Js_pool = jnp.einsum("ij,jab->iab", W, Js)                   # trust-weighted obs pool
    js_pool = W @ js
    return Pi + Js_pool, h + js_pool, key


def _transition(Pi: jax.Array, h: jax.Array, W: jax.Array, key: jax.Array,
                cfg: StructuralConfig, phi: jax.Array
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The pure one-round dynamics, returning the new ``(Pi, h, key)``. Shared by ``step``
    (which adds the observable read-out) and every ``run_*`` scan. Dispatches host-side on
    ``cfg.sharing_mode`` (``'posterior'`` = fuse whole beliefs, the DEFAULT and byte-identical
    to before; ``'observation'`` = pool raw observations, prior bias not transmitted), then
    applies the conviction tilt. ``cfg.precision_mode`` selects the per-channel gain; the
    default ``'heuristic', experiment_bias=0`` gives all-ones weights (plain ``fisher_deposit``).
    """
    if cfg.sharing_mode == "observation":
        Pi_new, h_new, key = _observe_pooled(Pi, h, W, key, cfg, phi)
    else:
        Pi_new, h_new, key = _fuse_then_observe(Pi, h, W, key, cfg, phi)

    # CONVICTION TILT (the second field): the motivated posterior q_lambda prop
    # p(s|o) e^{lambda U(s)} with U = T u, i.e. h <- h + lambda U. Guarded host-side on the
    # frozen-cfg float, so conviction_tilt=0 (DEFAULT) is byte-identical -- no tilt.
    if cfg.conviction_tilt != 0.0:
        u = ph.conviction_u(cfg)                                 # (d,)
        U = ph.conviction_field(Pi_new, h_new, cfg.node_names, u, cfg.conviction_alpha)
        h_new = h_new + cfg.conviction_tilt * U
    return Pi_new, h_new, key


def step(state: PopulationState, cfg: StructuralConfig, t: int
         ) -> tuple[PopulationState, dict]:
    """One round: FUSE belief nets over the trust graph, then OBSERVE and
    deposit Fisher information. Pure in ``(state, t)`` given the carried key.
    """
    phi = ph.phi_true_at(cfg, t)
    Pi_new, h_new, key = _transition(state.Pi, state.h, state.W, state.key, cfg, phi)
    new_state = PopulationState(Pi=Pi_new, h=h_new, names=state.names,
                                W=state.W, key=key)

    m = obs.order_parameter(Pi_new, h_new, state.names, ph.DISAGREEMENT_NODES,
                            cfg.mu_phlog_mass, cfg.mu_oxy_mass)
    out = {"t": t, "order_parameter": float(m)}
    return new_state, out


def run_final(cfg: StructuralConfig, state: PopulationState) -> PopulationState:
    """Fast-forward to the final state via ``jax.lax.scan`` (no per-step host
    sync) -- the rollout used by the long-horizon lock-in sweeps. The regime
    schedule is precomputed once (cheap, host-side) and scanned over, so the
    whole trajectory runs as a single compiled loop. Returns only the final
    ``PopulationState``; read the order parameter off it with
    ``observables.order_parameter``.
    """
    phis = jnp.stack([ph.phi_true_at(cfg, t) for t in range(cfg.n_steps)])  # (T, d)
    W = state.W

    def body(carry, phi):
        Pi, h, key = carry
        Pi, h, key = _transition(Pi, h, W, key, cfg, phi)
        return (Pi, h, key), None

    (Pi, h, key), _ = jax.lax.scan(body, (state.Pi, state.h, state.key), phis)
    return PopulationState(Pi=Pi, h=h, names=state.names, W=W, key=key)


def run_trace(cfg: StructuralConfig, state: PopulationState
              ) -> tuple[jax.Array, jax.Array]:
    """Full per-step trajectory via ``jax.lax.scan`` -- the fast equivalent of a
    ``step`` python loop, for long horizons. Returns ``(m_t, grav_attn_t)``, each
    ``(n_steps,)``: the population order parameter and the population-mean
    attention on the *gravimetric* (disagreement) experiments at each step. The
    latter visualises self-censorship -- a committed bloc's gravimetric attention
    falls toward ``1 - bias`` and stays there, so it never pulls the refuting
    distribution.
    """
    phis = jnp.stack([ph.phi_true_at(cfg, t) for t in range(cfg.n_steps)])  # (T, d)
    W = state.W
    mass_idx = jnp.asarray([cfg.node_names.index(n) for n in ph.DISAGREEMENT_NODES])
    dm = ph.disagreement_row_mask(cfg)                          # (m,) 0/1

    def body(carry, phi):
        Pi, h, key = carry
        Pi, h, key = _transition(Pi, h, W, key, cfg, phi)
        m = obs.order_parameter(Pi, h, state.names, ph.DISAGREEMENT_NODES,
                                cfg.mu_phlog_mass, cfg.mu_oxy_mass)
        mu = jax.vmap(jnp.linalg.solve)(Pi, h)
        oxy = jnp.clip((mu[:, mass_idx].mean(axis=1) - cfg.mu_phlog_mass)
                       / (cfg.mu_oxy_mass - cfg.mu_phlog_mass), 0.0, 1.0)
        Wt = ph.attention_weights(oxy, cfg)                     # (N, m)
        grav = (Wt @ dm) / dm.sum()                             # (N,) per-agent
        return (Pi, h, key), (m, grav.mean())

    _, (ms, gws) = jax.lax.scan(body, (state.Pi, state.h, state.key), phis)
    return ms, gws


def run_trace_schedule(cfg: StructuralConfig, state: PopulationState,
                       W_seq: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Like ``run_trace`` but with a *per-step* fusion weight schedule ``W_seq``
    ``(n_steps, N, N)`` instead of the single static ``state.W``: at step ``t`` the
    FUSE uses ``W_seq[t]``. This is the time-varying-network rollout -- the trust
    graph can change over the run, which is how the *timed bridge* (Model B) is
    driven: the population fuses inside disconnected communities for an incubation
    window, then a bridge opens. With a constant schedule (``W_seq[t] == state.W``
    for all ``t``) it is identical to ``run_trace`` -- one code path, no behavioural
    change. Returns ``(m_t, grav_t)`` exactly like ``run_trace``.
    """
    phis = jnp.stack([ph.phi_true_at(cfg, t) for t in range(cfg.n_steps)])  # (T, d)
    mass_idx = jnp.asarray([cfg.node_names.index(n) for n in ph.DISAGREEMENT_NODES])
    dm = ph.disagreement_row_mask(cfg)                          # (m,) 0/1

    def body(carry, inp):
        phi, W = inp
        Pi, h, key = carry
        Pi, h, key = _transition(Pi, h, W, key, cfg, phi)
        m = obs.order_parameter(Pi, h, state.names, ph.DISAGREEMENT_NODES,
                                cfg.mu_phlog_mass, cfg.mu_oxy_mass)
        mu = jax.vmap(jnp.linalg.solve)(Pi, h)
        oxy = jnp.clip((mu[:, mass_idx].mean(axis=1) - cfg.mu_phlog_mass)
                       / (cfg.mu_oxy_mass - cfg.mu_phlog_mass), 0.0, 1.0)
        Wt = ph.attention_weights(oxy, cfg)                     # (N, m)
        grav = (Wt @ dm) / dm.sum()                             # (N,) per-agent
        return (Pi, h, key), (m, grav.mean())

    _, (ms, gws) = jax.lax.scan(
        body, (state.Pi, state.h, state.key), (phis, W_seq))
    return ms, gws


def run_trace_index(cfg: StructuralConfig, state: PopulationState) -> jax.Array:
    """Per-agent oxygen-index trajectory ``(n_steps, N)`` via ``jax.lax.scan``:
    each agent's position on the phlogiston(0)->oxygen(1) axis at every step. The
    population curve is the row mean; a per-school curve is the mean over that
    school's agent columns -- so this one trace yields every aggregated order
    parameter without a python loop. Sibling of ``run_trace`` (which returns only
    the population mean); shares the same ``_transition`` body."""
    phis = jnp.stack([ph.phi_true_at(cfg, t) for t in range(cfg.n_steps)])  # (T, d)
    W = state.W
    mass_idx = jnp.asarray([cfg.node_names.index(n) for n in ph.DISAGREEMENT_NODES])

    def body(carry, phi):
        Pi, h, key = carry
        Pi, h, key = _transition(Pi, h, W, key, cfg, phi)
        mu = jax.vmap(jnp.linalg.solve)(Pi, h)
        oxy = jnp.clip((mu[:, mass_idx].mean(axis=1) - cfg.mu_phlog_mass)
                       / (cfg.mu_oxy_mass - cfg.mu_phlog_mass), 0.0, 1.0)
        return (Pi, h, key), oxy                                # (N,)

    _, oxys = jax.lax.scan(body, (state.Pi, state.h, state.key), phis)
    return oxys                                                 # (T, N)


def run_trace_net(cfg: StructuralConfig, state: PopulationState
                  ) -> tuple[jax.Array, jax.Array]:
    """Full per-agent ``(Pi, h)`` trajectory via ``jax.lax.scan``: returns
    ``(Pi_t, h_t)`` of shape ``(n_steps, N, d, d)`` and ``(n_steps, N, d)``.

    The heavyweight sibling of ``run_trace_index`` -- where that one reduces each
    step's state to a scalar oxygen-index *inside* the scan, this one emits the
    *whole* belief net per step, so the conditional / joint coupling structure can
    be read off the host afterwards (``bmr.schur_marginalize`` to condition out the
    hidden hub, ``observables.carryover_mass`` for the hub's structural importance,
    etc.). The scan body is the same ``_transition`` as every other rollout, so the
    trajectory is bit-identical to ``run_trace`` / ``run_final`` -- this just keeps
    the full state instead of a summary. Static ``W`` (no bridge schedule). Memory
    is ``O(n_steps * N * d^2)``; subsample the returned frames host-side for long
    horizons / large populations."""
    phis = jnp.stack([ph.phi_true_at(cfg, t) for t in range(cfg.n_steps)])  # (T, d)
    W = state.W

    def body(carry, phi):
        Pi, h, key = carry
        Pi, h, key = _transition(Pi, h, W, key, cfg, phi)
        return (Pi, h, key), (Pi, h)

    _, (Pi_t, h_t) = jax.lax.scan(body, (state.Pi, state.h, state.key), phis)
    return Pi_t, h_t                                            # (T,N,d,d),(T,N,d)


def run_trace_index_schedule(cfg: StructuralConfig, state: PopulationState,
                             W_seq: jax.Array) -> jax.Array:
    """Per-agent oxygen-index trajectory ``(n_steps, N)`` under a *per-step* fusion
    schedule ``W_seq`` ``(n_steps, N, N)`` -- the schedule sibling of
    ``run_trace_index`` (which uses the static ``state.W``). Lets a per-community
    curve be read *through* a network change, e.g. the clustered challenger's
    conviction during incubation and the kink at the moment the bridge opens. A
    constant schedule reproduces ``run_trace_index`` exactly."""
    phis = jnp.stack([ph.phi_true_at(cfg, t) for t in range(cfg.n_steps)])  # (T, d)
    mass_idx = jnp.asarray([cfg.node_names.index(n) for n in ph.DISAGREEMENT_NODES])

    def body(carry, inp):
        phi, W = inp
        Pi, h, key = carry
        Pi, h, key = _transition(Pi, h, W, key, cfg, phi)
        mu = jax.vmap(jnp.linalg.solve)(Pi, h)
        oxy = jnp.clip((mu[:, mass_idx].mean(axis=1) - cfg.mu_phlog_mass)
                       / (cfg.mu_oxy_mass - cfg.mu_phlog_mass), 0.0, 1.0)
        return (Pi, h, key), oxy                                # (N,)

    _, oxys = jax.lax.scan(body, (state.Pi, state.h, state.key), (phis, W_seq))
    return oxys                                                 # (T, N)


def run_trace_net_schedule(cfg: StructuralConfig, state: PopulationState,
                           W_seq: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Full per-agent ``(Pi, h)`` trajectory under a *per-step* fusion schedule
    ``W_seq`` ``(n_steps, N, N)`` -- the schedule sibling of ``run_trace_net`` (the
    "open the box" trace), so the conditional / joint structure can be read *through*
    a network change such as the timed bridge (a clustered challenger incubating in
    isolation, then a bridge opening into the mainstream). Returns ``(Pi_t, h_t)`` of
    shape ``(n_steps, N, d, d)`` / ``(n_steps, N, d)``. A constant schedule reproduces
    ``run_trace_net`` exactly. Memory is ``O(n_steps * N * d^2)``."""
    phis = jnp.stack([ph.phi_true_at(cfg, t) for t in range(cfg.n_steps)])  # (T, d)

    def body(carry, inp):
        phi, W = inp
        Pi, h, key = carry
        Pi, h, key = _transition(Pi, h, W, key, cfg, phi)
        return (Pi, h, key), (Pi, h)

    _, (Pi_t, h_t) = jax.lax.scan(body, (state.Pi, state.h, state.key), (phis, W_seq))
    return Pi_t, h_t                                            # (T,N,d,d),(T,N,d)


def run_trace_bmr(cfg: StructuralConfig, state: PopulationState
                  ) -> tuple[jax.Array, jax.Array]:
    """Per-step ``(m_t, deltaF_t)``: the population order parameter and the
    population-mean BMR Bayes factor for phlogiston's falsifiable commitment ("the
    calx is lighter"), via ``jax.lax.scan``. ``deltaF > 0`` => the data still
    vindicate the commitment; ``deltaF < 0`` => it is refuted. The crossing of
    ``deltaF_t`` through zero is the population's *refutation* signal; its lead over
    the ``m_t`` half-crossing is the early-warning lead-time (E3 -- self-sealing
    flatlines deltaF, a trusted vanguard restores it).

    Reuses ``bmr.prune_node_prior`` to sharpen the disagreement nodes to phlogiston's
    claimed value and ``linalg.savage_dickey`` (vmapped over agents) for the
    closed-form factor of each agent's live posterior against that reduced prior."""
    phis = jnp.stack([ph.phi_true_at(cfg, t) for t in range(cfg.n_steps)])  # (T, d)
    W = state.W
    phlog = ph.phlogiston_prior(cfg)
    reduced = bmr.prune_node_prior(phlog, ph.DISAGREEMENT_NODES, value=cfg.mu_phlog_mass)
    Pi0, h0, Pi0r, h0r = phlog.Pi, phlog.h, reduced.Pi, reduced.h

    def body(carry, phi):
        Pi, h, key = carry
        Pi, h, key = _transition(Pi, h, W, key, cfg, phi)
        dF = jax.vmap(lambda Pp, hp: linalg.savage_dickey(Pp, hp, Pi0, h0, Pi0r, h0r)[0]
                      )(Pi, h)                                   # (N,)
        m = obs.order_parameter(Pi, h, state.names, ph.DISAGREEMENT_NODES,
                                cfg.mu_phlog_mass, cfg.mu_oxy_mass)
        return (Pi, h, key), (m, dF.mean())

    _, (ms, dFs) = jax.lax.scan(body, (state.Pi, state.h, state.key), phis)
    return ms, dFs                                              # (T,), (T,)


def run_trace_precision(cfg: StructuralConfig, state: PopulationState
                        ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Like ``run_trace`` but ALSO returns the per-channel evidential precision on the
    disagreement (mass-law) channels at each step. Returns ``(m_t, grav_t, rho_disagree_t)``:

      ``m_t``            : (n_steps,) population order parameter.
      ``grav_t``         : (n_steps,) population-mean weight on the disagreement channels
                           (the self-censorship trace; matches ``run_trace``'s 2nd output).
      ``rho_disagree_t`` : (n_steps, n_disagree) the per-disagreement-channel evidential
                           precision (population mean), in the order of the disagreement
                           rows of ``H_observable``.

    The reported weights use ``_channel_weights`` in *all* precision modes (so one plotting
    path serves heuristic and derived): for ``precision_mode='derived'`` ``rho_disagree_t``
    is constant in ``t`` (the paradigm-level gain) and equals ``derived_channel_precision``
    sliced to the disagreement rows; at ``core_governance=0`` it is all ``rho_max``; for
    ``derived_live`` it falls as the bloc entrenches. ``run_trace``'s 2-tuple contract is
    left untouched -- this is an additive function.
    """
    phis = jnp.stack([ph.phi_true_at(cfg, t) for t in range(cfg.n_steps)])  # (T, d)
    W = state.W
    dm = ph.disagreement_row_mask(cfg)                          # (m,) 0/1
    meas = ph.measured_nodes(cfg)
    disagree_cols = jnp.asarray(
        [i for i, nm in enumerate(meas) if nm in ph.DISAGREEMENT_NODES], dtype=jnp.int32)

    def body(carry, phi):
        Pi, h, key = carry
        Pi, h, key = _transition(Pi, h, W, key, cfg, phi)
        m = obs.order_parameter(Pi, h, state.names, ph.DISAGREEMENT_NODES,
                                cfg.mu_phlog_mass, cfg.mu_oxy_mass)
        Wt = _channel_weights(Pi, h, state.W, cfg)              # (N, m)
        grav = (Wt @ dm) / dm.sum()                            # (N,) per-agent
        rho_dis = Wt[:, disagree_cols].mean(axis=0)            # (n_disagree,)
        return (Pi, h, key), (m, grav.mean(), rho_dis)

    _, (ms, gws, rho_dis) = jax.lax.scan(
        body, (state.Pi, state.h, state.key), phis)
    return ms, gws, rho_dis


def run(cfg: StructuralConfig, key: jax.Array
        ) -> tuple[PopulationState, list[dict]]:
    """Run the full multi-agent rollout; return final state and per-step outs."""
    state = init_state(cfg, key)
    history: list[dict] = []
    for t in range(cfg.n_steps):
        state, out = step(state, cfg, t)
        history.append(out)
    return state, history
