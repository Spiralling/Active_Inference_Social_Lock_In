"""Dual-field scoring: epistemic evidence plus utility on belief means."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from . import bmr
from .belief import GaussianBeliefNet


@dataclass(frozen=True)
class PrecisionUtilityNet:
    """A Gaussian belief net with an intrinsic utility field over nodes."""

    names: tuple[str, ...]
    Pi: jax.Array
    h: jax.Array
    u: jax.Array
    alpha: float = 0.5

    def dim(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def to_belief(self) -> GaussianBeliefNet:
        return GaussianBeliefNet(Pi=self.Pi, h=self.h, names=self.names)

    def mean(self) -> jax.Array:
        return self.to_belief().mean()

    def cov(self) -> jax.Array:
        return self.to_belief().cov()

    def utility_operator(self) -> jax.Array:
        """Row-stochastic propagation operator from absolute off-diagonal precision."""
        d = self.dim()
        off_diag = jnp.abs(self.Pi) * (1.0 - jnp.eye(d, dtype=self.Pi.dtype))
        row_sum = jnp.sum(off_diag, axis=1, keepdims=True)
        W = jnp.where(row_sum > 0.0, off_diag / row_sum, 0.0)
        isolated = jnp.squeeze(row_sum <= 0.0, axis=1)
        return W + jnp.diag(isolated.astype(self.Pi.dtype))

    def effective_utility(self) -> jax.Array:
        """Solve the utility field fixed-point: (I - alpha W) u_eff = u."""
        d = self.dim()
        W = self.utility_operator()
        A = jnp.eye(d, dtype=self.Pi.dtype) - self.alpha * W
        return jnp.linalg.solve(A, self.u)

    def cost_field(self) -> jax.Array:
        """Per-node Schur carry-over cost mass from precision couplings."""
        d = self.dim()
        if d == 1:
            return jnp.zeros((1,), dtype=self.Pi.dtype)

        costs = []
        for i in range(d):
            keep = [k for k in range(d) if k != i]
            Pi_mi_i = self.Pi[jnp.asarray(keep), i : i + 1]
            Pi_i_mi = self.Pi[i : i + 1, jnp.asarray(keep)]
            inv_Pi_ii = 1.0 / self.Pi[i, i]
            carry = (Pi_mi_i * inv_Pi_ii) @ Pi_i_mi
            costs.append(jnp.linalg.norm(carry, ord="fro"))
        return jnp.asarray(costs)

    def score_state(self) -> jax.Array:
        """Utility score of the current posterior state (u_eff dot mean)."""
        return jnp.dot(self.effective_utility(), self.mean())


def conviction_field(Pi: jax.Array, h: jax.Array, names: tuple[str, ...],
                     u: jax.Array, alpha: float = 0.5) -> jax.Array:
    """The conviction field ``U = T u`` for a *stack* of belief nets -- the general
    primitive (this is its natural home: it just wraps
    ``PrecisionUtilityNet.effective_utility``, which solves ``(I - alpha W) u_eff = u``
    with ``W`` the row-stochastic propagation operator read off each net's off-diagonal
    precision).

    ``Pi`` (N, d, d), ``h`` (N, d). ``u`` is either a single ``(d,)`` utility broadcast to
    every agent (the homogeneous case -- byte-identical to the original
    ``phlogiston.conviction_field``) or a per-agent ``(N, d)`` stack (the heterogeneous
    case: each community tilts toward its *own* favoured theory). Returns ``(N, d)`` -- the
    value field propagated along each agent's *current* couplings, so it rides the structure
    as the relational substrate reshapes it. Well-defined for any ``alpha < 1`` regardless of
    ``Pi`` definiteness (``W`` is row-stochastic), so it is safe on an improper hub prior.

    The motivated (value-tilted) posterior is ``q_lambda(s) prop p(s|o) e^{lambda U(s)}``;
    for a linear utility on a Gaussian that is one shift of the potential, ``h <- h +
    lambda U`` applied each step (see ``step._transition`` / ``simulation.run_simulation``)."""
    u_arr = jnp.asarray(u)
    if u_arr.ndim == 1:
        def one(Pi_i, h_i):
            return PrecisionUtilityNet(names=names, Pi=Pi_i, h=h_i,
                                       u=u_arr, alpha=alpha).effective_utility()
        return jax.vmap(one)(Pi, h)

    def one_u(Pi_i, h_i, u_i):
        return PrecisionUtilityNet(names=names, Pi=Pi_i, h=h_i,
                                   u=u_i, alpha=alpha).effective_utility()
    return jax.vmap(one_u)(Pi, h, u_arr)


def cost_field_live(Pi: jax.Array, alpha: float = 0.5) -> jax.Array:
    """The revision-cost field ``kappa = T 1 - 1`` for a *stack* of belief nets, read from
    each net's *live* couplings.

    The paper's revision cost is the propagation operator applied to the unit source,
    ``kappa = T 1``: how much downstream mass must re-equilibrate when a commitment moves.
    Crucially this must be computed on the UNNORMALIZED coupling mass ``A = |Pi| (1 - I)``:
    the row-stochastic operator used for the conviction field would give ``T 1 =
    1/(1-alpha) 1`` -- constant across nodes, carrying no information. Here each agent's
    ``A`` is rescaled by ``alpha / rho(A)`` (``rho`` the Perron root via ``eigvalsh``; ``A``
    symmetric nonnegative), which (i) makes the resolvent well-defined for any live ``Pi``
    (spectral radius exactly ``alpha < 1``), and (ii) makes ``kappa`` invariant to the
    global ``|Pi|`` growth the forgetting equilibrium produces -- the same invariance
    argument as ``reliability.pairwise_structure_z2``. The identity term is subtracted so
    a node with no couplings (e.g. a pinned, not-yet-awakened slot) reads exactly ``0``.

    ``Pi`` (N, d, d) -> ``kappa`` (N, d), nonnegative, monotone in coupling mass: a hub
    propagating many strong chains costs more to revise than a belt node."""
    Pi_arr = jnp.asarray(Pi)
    d = Pi_arr.shape[-1]
    eye = jnp.eye(d, dtype=Pi_arr.dtype)
    ones = jnp.ones((d,), dtype=Pi_arr.dtype)

    def one(Pi_i):
        A = jnp.abs(Pi_i) * (1.0 - eye)
        A = 0.5 * (A + A.T)
        rho = jnp.max(jnp.linalg.eigvalsh(A))
        B = (alpha / jnp.maximum(rho, 1e-12)) * A
        return jnp.linalg.solve(eye - B, ones) - 1.0

    return jax.vmap(one)(Pi_arr)


@dataclass(frozen=True)
class ScoreBreakdown:
    delta_F: jax.Array
    delta_U: jax.Array
    delta_J: jax.Array
    favour_reduced: bool
    reduced_post: GaussianBeliefNet


def score_edit_with_utility(
    net_post: GaussianBeliefNet,
    net_prior: GaussianBeliefNet,
    reduced_prior: GaussianBeliefNet,
    intrinsic_utility: jax.Array,
    alpha: float = 0.5,
    beta_u: float = 1.0,
) -> ScoreBreakdown:
    """Score a structural edit with evidence (Delta F) plus utility shift (Delta U)."""
    bmr_out = bmr.bmr_prune(net_post, net_prior, reduced_prior)
    reduced_post = bmr_out["tilde_post"]
    delta_F = bmr_out["delta_F"]

    u = jnp.asarray(intrinsic_utility)
    if u.shape != (len(net_post.names),):
        raise ValueError(
            "intrinsic_utility must have shape "
            f"({len(net_post.names)},), got {u.shape}"
        )

    full_net = PrecisionUtilityNet(
        names=net_post.names, Pi=net_post.Pi, h=net_post.h, u=u, alpha=alpha
    )
    reduced_net = PrecisionUtilityNet(
        names=reduced_post.names,
        Pi=reduced_post.Pi,
        h=reduced_post.h,
        u=u,
        alpha=alpha,
    )

    delta_U = reduced_net.score_state() - full_net.score_state()
    delta_J = delta_F + beta_u * delta_U
    return ScoreBreakdown(
        delta_F=delta_F,
        delta_U=delta_U,
        delta_J=delta_J,
        favour_reduced=bool(delta_J > 0.0),
        reduced_post=reduced_post,
    )
