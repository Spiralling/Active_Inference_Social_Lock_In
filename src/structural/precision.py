"""Derived evidential precision: the dual of carry-over inertia, pointed outward.

The paradigm is a *precision field* over the augmented graph. The same hidden core
that sets precision on the **internal** edges -- giving the carry-over inertia
``lambda_i`` (the Schur fill-in ``Pi_ab Pi_bb^{-1} Pi_ba`` of ``bmr.carryover``: how
tightly a commitment is braced by the rest) -- also sets precision on the **sensory**
edges, giving the *evidential gain* ``rho_k`` on each observation channel ``k``. This
module derives ``rho_k`` from structure rather than asserting it (contrast the heuristic
``phlogiston.attention_weights`` sigmoid, which gates on the agent's held paradigm).

The cost (the dual of ``lambda_i``, read on the core). Dropping a measured node ``v``
leaves the carry-over fill-in ``carryover(net, (v,)) = Pi[:,v] Pi[v,v]^{-1} Pi[v,:]`` on
the survivors -- the same object whose Frobenius mass *is* ``lambda_v`` (how much ``v``
braces the whole net, ``observables.carryover_mass``). Its entry on the hidden **core**
``c`` is

    C_v = Pi[c, v]^2 / Pi[v, v]                    (the brace v puts on the core),

i.e. ``carryover(net, (v,))[c, c]``. So ``lambda_v`` reads the carry-over's total mass and
``C_v`` reads its core entry: one Schur fill-in, two faces of one precision field. This
form is non-negative and well-defined whenever ``Pi[v,v] > 0`` -- crucially it stays clean
even when the paradigm prior is an *improper* common-cause net (the phlogiston hub couples
to eight neighbours at 0.8 with self-precision 2.0, so its bare precision is indefinite and
the covariance ``Pi^{-1}`` has no proper core variance). It is the robust default.

Motivation (the Sherman-Morrison reading, exact for a *proper* belief). A channel reading
``v`` with precision ``rho`` adds ``rho e_v e_v^T`` to ``Pi``; the induced shift of the
core's posterior mean, per unit prediction error on ``v``, is ``~ rho * Sigma[c, v]``
(``Sigma = Pi^{-1}``), and forcing the stiff core to move by ``Delta`` costs
``~ Delta^2 / Sigma[c,c]``. So the expected expense of opening ``v`` to evidence is the
covariance coupling-to-core ``Sigma[c,v]^2 / Sigma[c,c]`` (or its scale-free normalization
``Sigma[c,v]^2/(Sigma[c,c]Sigma[v,v]) in [0,1]``). For a proper (PD) belief these rank the
channels the same way as ``C_v``; they are offered as ``cost_kind='fe'|'normalized'`` for
PD nets (e.g. a fused live belief, ``derived_live``), with the Schur ``cost_kind='carryover'``
the default that also handles the improper prior.

The gain function (the agent under-allocates precision to expensive channels -- expected
free energy, pointed outward):

    rho_k = rho_max / (1 + g * C_{v(k)}),    g = core_governance >= 0.

``g = 0`` gives ``rho_k = rho_max`` everywhere (paradigm-neutral; with ``rho_max = 1`` this
is *exactly* the unbiased deposit -- the back-compat anchor). Large ``g`` drives
``rho_k -> 0`` on channels coupled to the entrenched core, silencing them: evidential
lock-in. ``g`` is the phase-transition knob, the sensory-edge analogue of ``prec_scale``.

Because ``world.fisher_deposit_weighted`` already deposits ``H^T diag(w) H / sigma^2`` --
i.e. its per-row ``weights[k]`` *is* the per-channel observation precision -- ``rho_k``
plugs straight into the ``weights`` argument with no new deposit machinery.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.structural.belief import GaussianBeliefNet


def _coupling_from_precision(Pi: jax.Array, c: int, node_idx: jax.Array) -> jax.Array:
    """The carry-over brace of each measured node on the core, read off ``Pi`` directly:

        C_v = Pi[c, v]^2 / Pi[v, v]   == carryover(net, (v,))[c, c].

    ``Pi`` (d, d) precision matrix; ``c`` core index; ``node_idx`` (m,) measured indices.
    Non-negative whenever the measured-node self-precisions are positive; needs no inverse,
    so it is well-defined even when ``Pi`` is indefinite (an improper common-cause prior).
    Returns (m,).
    """
    pi_cv = Pi[c, node_idx]                          # (m,) direct core->v coupling
    pi_vv = Pi[node_idx, node_idx]                   # (m,) v self-precision (diagonal)
    return pi_cv ** 2 / pi_vv


def _coupling_from_cov(Sigma: jax.Array, c: int, node_idx: jax.Array,
                       cost_kind: str) -> jax.Array:
    """The Sherman-Morrison coupling-to-core cost from a *proper* covariance ``Sigma``:

        cost_kind == "fe"         :  Sigma[c,v]^2 / Sigma[c,c]
        cost_kind == "normalized" :  Sigma[c,v]^2 / (Sigma[c,c] * Sigma[v,v])   in [0, 1]

    Requires ``Sigma`` to be a genuine covariance (PD precision); see module docstring.
    Returns (m,).
    """
    s_cv = Sigma[c, node_idx]                        # (m,)  Sigma[core, v]
    s_cc = Sigma[c, c]                               # scalar core marginal variance
    num = s_cv ** 2
    if cost_kind == "normalized":
        s_vv = Sigma[node_idx, node_idx]             # (m,)  per-node variance (diagonal)
        return num / (s_cc * s_vv)
    if cost_kind == "fe":
        return num / s_cc
    raise ValueError(
        f"cov cost_kind must be 'fe' or 'normalized', got {cost_kind!r}")


def core_coupling(net: GaussianBeliefNet, core: str,
                  nodes: tuple[str, ...],
                  cost_kind: str = "carryover") -> jax.Array:
    """Per-node structural coupling-to-core cost ``C_v``.

        cost_kind == "carryover" (default) :  Pi[c,v]^2 / Pi[v,v]   (the Schur brace on
            the core; the literal dual of ``lambda_v``; robust to an improper prior)
        cost_kind == "fe"                  :  Sigma[c,v]^2 / Sigma[c,c]   (proper-belief
            Sherman-Morrison free-energy cost; needs a PD net)
        cost_kind == "normalized"          :  Sigma[c,v]^2 / (Sigma[c,c] Sigma[v,v]) in [0,1]

    with ``c = net.index(core)``. The hidden core need not be observable -- it is a
    coordinate of the joint belief. Returns ``(len(nodes),)`` in the order of ``nodes``.
    The ``carryover`` form reads ``net.Pi`` directly (no inverse); the covariance forms
    form ``net.cov()`` and require a proper (PD) belief (see module docstring).
    """
    c = net.index(core)
    node_idx = net.indices(nodes)
    if cost_kind == "carryover":
        return _coupling_from_precision(net.Pi, c, node_idx)
    return _coupling_from_cov(net.cov(), c, node_idx, cost_kind)


def rho_from_cost(cost: jax.Array, core_governance: float,
                  rho_max: float = 1.0) -> jax.Array:
    """Map coupling-to-core cost to evidential precision -- the gain function

        rho = rho_max / (1 + core_governance * cost).

    ``core_governance = 0`` gives ``rho = rho_max`` everywhere (recovers the unbiased
    deposit when ``rho_max = 1``); large ``core_governance`` drives ``rho -> 0`` on
    high-cost channels. Monotone decreasing in both ``core_governance`` and ``cost``
    (for non-negative ``cost``). Returns the same shape as ``cost``.
    """
    return rho_max / (1.0 + core_governance * cost)


def channel_precision(net: GaussianBeliefNet, core: str,
                      measured_nodes: tuple[str, ...],
                      core_governance: float, rho_max: float = 1.0,
                      cost_kind: str = "carryover",
                      gate_mask: jax.Array | None = None) -> jax.Array:
    """Derived per-channel evidential precision ``rho_k`` over the ``H_observable`` rows.

    ``measured_nodes`` is the row order of ``H_observable`` (i.e.
    ``phlogiston.measured_nodes(cfg)``); one ``rho`` per row. Composes ``core_coupling``
    then ``rho_from_cost``. Returns ``(m,)`` ready to pass as the ``weights`` argument of
    ``world.fisher_deposit_weighted``.

    ``gate_mask`` (optional ``(m,)`` 0/1): if given, governance is applied ONLY where the
    mask is 1 (``rho = rho_max`` elsewhere) -- the "restrict silencing to the disagreement
    rows" comparison. Default ``None`` = the pure structural derivation over all channels
    (the honest first-principles choice: a strong paradigm under-weights *every* channel
    whose accommodation is expensive, including hub-coupled agreement channels -- benign,
    because the agreement nodes' truth never flips).

    NOTE: ``rho`` here is a PARADIGM-LEVEL constant -- it depends on the (prior) net and
    ``core_governance``, not on any agent's live stance -- so it is computed once and
    broadcast across agents in ``step._transition``.
    """
    cost = core_coupling(net, core, measured_nodes, cost_kind)
    rho = rho_from_cost(cost, core_governance, rho_max)
    if gate_mask is not None:
        gate_mask = jnp.asarray(gate_mask)
        rho = jnp.where(gate_mask > 0, rho, rho_max)
    return rho


def _coupling_from_precision_H(Pi: jax.Array, c: int, H: jax.Array) -> jax.Array:
    """Relational generalization of ``C_v = Pi[c,v]^2 / Pi[v,v]`` to a channel that
    reads a *combination* of nodes ``H[k]`` (a row of an arbitrary observation operator):

        C_k = (H[k] . Pi[:, c])^2 / (H[k] . Pi . H[k]).

    For a direct read ``H[k] = e_v`` the numerator is ``Pi[v, c]^2`` and the denominator
    ``Pi[v, v]``, so it reduces *exactly* to ``_coupling_from_precision`` -- the node and
    relational operators rank the channels the same way on their shared direct rows.
    Reads ``Pi`` directly (no inverse), so it is well-defined even for the improper
    common-cause prior. ``Pi`` (d, d); ``c`` core index; ``H`` (m, d). Returns (m,).
    """
    pi_col_c = Pi[:, c]                               # (d,) core column
    num = (H @ pi_col_c) ** 2                          # (m,) (H[k] . Pi[:,c])^2
    quad = jnp.einsum("ki,ij,kj->k", H, Pi, H)         # (m,) H[k] Pi H[k]
    return num / quad


def channel_precision_H(net: GaussianBeliefNet, core: str, H: jax.Array,
                        core_governance: float, rho_max: float = 1.0,
                        gate_mask: jax.Array | None = None) -> jax.Array:
    """Derived per-channel evidential precision ``rho_k`` over the rows of an ARBITRARY
    observation operator ``H`` (m, d) -- the relational generalization of
    ``channel_precision`` (which assumes one row per node). Composes the relational cost
    ``_coupling_from_precision_H`` then ``rho_from_cost``; on a node operator (unit rows)
    it returns the same ``rho`` as ``channel_precision(cost_kind='carryover')``. Reads the
    precision directly (no inverse), robust to the improper hub prior. Returns ``(m,)``
    ready to pass as ``weights`` to ``world.fisher_deposit_weighted``.

    ``gate_mask`` (optional ``(m,)`` 0/1): restrict governance to the masked rows
    (``rho = rho_max`` elsewhere), as in ``channel_precision``.
    """
    c = net.index(core)
    cost = _coupling_from_precision_H(net.Pi, c, jnp.asarray(H))
    rho = rho_from_cost(cost, core_governance, rho_max)
    if gate_mask is not None:
        gate_mask = jnp.asarray(gate_mask)
        rho = jnp.where(gate_mask > 0, rho, rho_max)
    return rho


def channel_precision_H_stack(Pi: jax.Array, core_idx: int, H: jax.Array,
                              core_governance: float, rho_max: float = 1.0) -> jax.Array:
    """Per-agent derived ``rho_k`` over the rows of a relational operator ``H`` (m, d),
    from a live / fused ``(N, d, d)`` precision stack -- the relational sibling of
    ``channel_precision_stack`` and the ADAPTIVE (``derived_live``) governance path on the
    edge-moving substrate. vmaps the H-based cost ``_coupling_from_precision_H`` over
    agents (reads ``Pi`` directly, no inverse). Returns ``(N, m)``.

    The point of the live form: a more entrenched agent carries a larger precision mass, so
    its coupling-to-core cost on the disconfirming (mass-balance) row is larger, so its
    ``rho`` on that row is smaller -- self-silencing tightens with entrenchment *on its own*,
    which is how a heterogeneous population can lock its core while its belt stays open
    without any per-agent knob being dialled. Valid once the live belief is proper enough
    that the row quadratic ``H[k] Pi H[k] > 0`` (true after a few deposits); see the module
    docstring's PD caveat.
    """
    H = jnp.asarray(H)
    costs = jax.vmap(lambda P: _coupling_from_precision_H(P, core_idx, H))(Pi)   # (N, m)
    return rho_from_cost(costs, core_governance, rho_max)


def channel_precision_stack(Pi: jax.Array, h: jax.Array,
                            names: tuple[str, ...], core: str,
                            measured_nodes: tuple[str, ...],
                            core_governance: float, rho_max: float = 1.0,
                            cost_kind: str = "carryover") -> jax.Array:
    """Per-agent derived ``rho_k`` from a live / fused ``(N, d, d)`` belief stack -- the
    optional ADAPTIVE-governance variant (``rho`` read from each agent's *current* belief).

    With ``cost_kind='carryover'`` (default) reads each agent's ``Pi`` directly (no
    inverse). With a covariance ``cost_kind`` ('fe'/'normalized') it vmaps the inverse over
    the stack -- only valid once the live belief is proper (PD), e.g. after fusion + data;
    use sparingly (``N`` inverses per call) and prefer the prior-derived default in
    ``channel_precision``, which avoids both the inverse and the ``rho<->Sigma`` feedback
    loop. Returns ``(N, m)``. ``h`` is unused (cost depends on the precision / covariance
    alone) but kept so the call site mirrors the fused ``(Pi, h)`` pair.
    """
    c = names.index(core)
    node_idx = jnp.asarray([names.index(v) for v in measured_nodes], dtype=jnp.int32)
    if cost_kind == "carryover":
        costs = jax.vmap(lambda P: _coupling_from_precision(P, c, node_idx))(Pi)
    else:
        Sigmas = jax.vmap(jnp.linalg.inv)(Pi)                              # (N, d, d)
        costs = jax.vmap(lambda S: _coupling_from_cov(S, c, node_idx, cost_kind))(Sigmas)
    return rho_from_cost(costs, core_governance, rho_max)                   # (N, m)
