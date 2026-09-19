"""The decision layer: pricing structural moves with the model evidence.

Where ``src/structural/belief.py`` holds transforms that always succeed, this
module holds the read-only *scorers* — the only place a matrix inverse appears.
Three jobs, all from ``notes/bmr_feynman_linear_algebra.tex``:

  - ``log_evidence``       : the closed-form Gaussian log-normalizer (log Z).
  - ``schur_marginalize``  : REDUCE by marginalizing — drop a block, survivors
                             keep the Schur complement. This is the carry-over:
                             removing a hub writes fill-in edges among its former
                             neighbours.
  - ``bmr_prune``          : REDUCE by pruning (Bayesian Model Reduction) — same
                             likelihood, sharper prior; the log Bayes factor for
                             the edit has a closed form for Gaussians (the matrix
                             Savage-Dickey / Friston-Penny post-hoc identity).

Plus prior-edit helpers (``prune_node_prior``, ``prune_edge_prior``) that build
the reduced prior ``bmr_prune`` scores.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.structural.belief import GaussianBeliefNet
from src.structural import linalg


LOG_2PI = linalg.LOG_2PI
SPIKE_PRECISION = linalg.SPIKE_PRECISION   # precision used to pin a node to a sharp value


def log_evidence(net: GaussianBeliefNet) -> jax.Array:
    """Closed-form Gaussian log-normalizer (note Eq. mgf / §6):

        Z   = (2pi)^{d/2} det(Pi)^{-1/2} exp( 1/2 h^T Pi^{-1} h )
        logZ = 1/2 [ h^T Pi^{-1} h  -  logdet(Pi)  +  d log(2pi) ]

    Implemented with ``slogdet`` (numerically stable log-determinant) and
    ``solve`` for ``h^T Pi^{-1} h`` — the inverse is never formed. Returns a
    scalar. Per-candidate-paradigm running log-evidence is the difference
    ``log_evidence(posterior) - log_evidence(prior)`` accumulated as data arrive
    (see ``src/structural/agent.py``); its drift rate is ``-KL(P || p_M)``.

    Thin wrapper over ``linalg.log_evidence`` (the reusable array kernel).
    """
    return linalg.log_evidence(net.Pi, net.h)


def _partition(net: GaussianBeliefNet, drop: tuple[str, ...]
               ) -> tuple[tuple[str, ...], jax.Array, jax.Array]:
    """Split a basis into the kept block ``a`` and the dropped block ``b``.

    Returns ``(keep_names, a, b)`` with ``a`` / ``b`` the integer index arrays
    (via ``GaussianBeliefNet.indices``) of survivors and dropped nodes. Raises if
    ``drop`` names are absent from the basis or would empty it.
    """
    drop_set = set(drop)
    missing = drop_set - set(net.names)
    if missing:
        raise ValueError(f"drop names not in basis: {missing}")
    keep_names = tuple(n for n in net.names if n not in drop_set)
    if not keep_names:
        raise ValueError("cannot marginalize away the entire basis")
    return keep_names, net.indices(keep_names), net.indices(drop)


def schur_marginalize(net: GaussianBeliefNet, drop: tuple[str, ...]
                      ) -> GaussianBeliefNet:
    """REDUCE by marginalizing: integrate out the ``drop`` block, exactly
    (note Eq. schur, the carry-over).

    Partition into the kept block ``a`` (survivors, order preserved) and the
    dropped block ``b``. The survivors keep the Schur complement:

        Pi_a^marg = Pi_aa - Pi_ab Pi_bb^{-1} Pi_ba
        h_a^marg  = h_a   - Pi_ab Pi_bb^{-1} h_b

    The subtracted term writes new off-diagonal entries (fill-in) among the
    removed block's former neighbours — the deleted node leaves its ghost behind
    as fresh couplings. Both ``Pi_bb^{-1} (.)`` are computed with ``solve``.
    """
    keep_names, a, b = _partition(net, drop)
    Pi_marg, h_marg = linalg.schur_marginalize(net.Pi, net.h, a, b)
    return GaussianBeliefNet(Pi=Pi_marg, h=h_marg, names=keep_names)


def condition(net: GaussianBeliefNet, drop: tuple[str, ...]
              ) -> GaussianBeliefNet:
    """REDUCE by conditioning: in information form this is just restricting to
    the kept block (note §3 duality table — conditioning is the *cheap* removal
    in precision coordinates, marginalizing is the expensive one).

        Pi_a^cond = Pi_aa ,   h_a^cond = h_a

    This is the exact adjoint of ``belief.border``: bordering appends a node and
    leaves the incumbent block untouched, so conditioning it away returns the
    original *exactly*. (Marginalizing, by contrast, leaves the carry-over ghost
    ``Pi_ab Pi_bb^{-1} Pi_ba`` behind — the note's "returns exactly" claim holds
    for conditioning, not for the Schur complement.)
    """
    keep_names, a, _ = _partition(net, drop)
    Pi_c, h_c = linalg.condition(net.Pi, net.h, a)
    return GaussianBeliefNet(Pi=Pi_c, h=h_c, names=keep_names)


def carryover(net: GaussianBeliefNet, drop: tuple[str, ...]) -> jax.Array:
    """The carry-over (fill-in) matrix ``Pi_ab Pi_bb^{-1} Pi_ba`` alone — the
    ghost a marginalized block leaves among its former neighbours. Its mass is
    the literal "importance" of the dropped block (note §9)."""
    _, a, b = _partition(net, drop)
    return linalg.carryover_fillin(net.Pi, a, b)


def bmr_prune(net_post: GaussianBeliefNet,
              net_prior: GaussianBeliefNet,
              reduced_prior: GaussianBeliefNet) -> dict:
    """REDUCE by pruning: the closed-form log Bayes factor for a structural edit
    (note §4-§6; matrix Savage-Dickey / Friston-Penny post-hoc identity).

    A reduced model shares the *same likelihood* but has a sharper prior
    ``reduced_prior`` = (tildePi0, tildeh0) — e.g. a coupling pinned to zero. The
    shared likelihood deposit is ``J = Pi_post - Pi0``, ``j = h_post - h0``, so
    the reduced posterior is obtained by swapping the prior:

        tildePi_post = tildePi0 + (Pi_post - Pi0)
        tildeh_post  = tildeh0  + (h_post  - h0)

    Because the likelihood normalizer cancels between the two models, the log
    Bayes factor is a difference of four closed-form log-evidences:

        Delta F = log(tildeZ / Z)
                = [logZ(tildePi_post, tildeh_post) - logZ(tildePi0, tildeh0)]
                - [logZ(Pi_post,      h_post)      - logZ(Pi0,      h0)]

    Exact for Gaussians (no Laplace approximation). ``Delta F > 0`` means the
    data are content with the edit and the reduced (pruned) model is favoured;
    ``Delta F < 0`` means the data hold the coupling away from zero and the full
    model is kept. Returns ``{"delta_F", "favour_reduced", "tilde_post"}``.
    """
    for other in (net_prior, reduced_prior):
        if other.names != net_post.names:
            raise ValueError("bmr_prune requires all three nets on the same basis")

    delta_F, Pi_rp, h_rp = linalg.savage_dickey(
        net_post.Pi, net_post.h,
        net_prior.Pi, net_prior.h,
        reduced_prior.Pi, reduced_prior.h,
    )
    tilde_post = GaussianBeliefNet(Pi=Pi_rp, h=h_rp, names=net_post.names)
    return {
        "delta_F": delta_F,
        "favour_reduced": bool(delta_F > 0),
        "tilde_post": tilde_post,
    }


# ----------------------------------------------------------------------
# Prior-edit helpers: construct the reduced prior bmr_prune scores.
# ----------------------------------------------------------------------

def prune_node_prior(net_prior: GaussianBeliefNet, nodes: tuple[str, ...],
                     spike: float = SPIKE_PRECISION, value: float = 0.0
                     ) -> GaussianBeliefNet:
    """Build a reduced prior that pins ``nodes`` to a sharp value: zero all their
    couplings (off-diagonal) and sharpen their diagonal to a ``spike`` centred at
    ``value`` (potential ``h_i = spike * value`` so the mean is ``value``).

    Two uses:
      * ``value = 0`` — "pin the node out" (prune a whole hub toward absence).
      * ``value = <claimed value>`` — sharpen a paradigm's specific commitment to
        the value it asserts, so ``bmr_prune`` tests whether the data still
        support that claim (``Delta F > 0`` while the claim holds, flipping
        negative once the data refute it).

    Survivors' prior block is untouched; the carry-over only shows up in the
    *posterior* via ``bmr_prune``.
    """
    idx = jnp.asarray([net_prior.names.index(n) for n in nodes])
    Pi, h = linalg.spike_prior(net_prior.Pi, net_prior.h, idx,
                               value=value, spike=spike)
    return GaussianBeliefNet(Pi=Pi, h=h, names=net_prior.names)


def prune_edge_prior(net_prior: GaussianBeliefNet, edge: tuple[str, str]
                     ) -> GaussianBeliefNet:
    """Build a reduced prior that pins a single coupling ``edge = (u, v)`` to
    zero (the edge vanishes), leaving everything else intact. A rank-2 symmetric
    edit."""
    u, v = net_prior.names.index(edge[0]), net_prior.names.index(edge[1])
    Pi = linalg.zero_edge_prior(net_prior.Pi, u, v)
    return GaussianBeliefNet(Pi=Pi, h=net_prior.h, names=net_prior.names)
