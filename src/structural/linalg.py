"""Information-form Gaussian linear algebra: the reusable array kernel.

Pure functions on raw arrays (``Pi``, ``h``, ``H``, integer index arrays) -- NO
``GaussianBeliefNet``, no ``names``, no basis bookkeeping, no scenario knowledge.
This is the single place the matrix math actually lives; ``belief.py``, ``bmr.py``
and ``world.py`` are thin wrappers that add named-basis bookkeeping and delegate
the numbers here. Reuse these anywhere you have an information-form Gaussian,
independent of the phlogiston scenario.

Information form. A Gaussian over ``x in R^d`` is carried as ``(Pi, h)`` with

    p(x) propto exp( -1/2 x^T Pi x + h^T x ),     Pi = Sigma^{-1},  h = Pi mu.

The whole point of these coordinates: *learning is addition* (``Pi += J``), *fusion
is addition* (``Pi1 + Pi2``), *marginalizing is a Schur complement*, and the
log-normalizer is closed form. Every function below is one of those identities,
written once, on bare arrays.

Block operations take integer index arrays ``keep`` / ``drop`` (e.g. from
``GaussianBeliefNet.indices``) rather than names, so they compose with any basis.
``solve`` is used wherever an inverse would otherwise appear; ``info_cov`` is the
only function that forms ``Pi^{-1}`` explicitly, and it says so.

See ``notes/bmr_feynman_linear_algebra.tex`` for the derivations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


LOG_2PI = float(jnp.log(2.0 * jnp.pi))
SPIKE_PRECISION = 1e6   # precision used to pin a coordinate to a sharp value (BMR)


# ----------------------------------------------------------------------
# Solves (mean / covariance from information form).
# ----------------------------------------------------------------------

def info_mean(Pi: jax.Array, h: jax.Array) -> jax.Array:
    """Posterior mean ``mu = Pi^{-1} h``, solved (never inverted)."""
    return jnp.linalg.solve(Pi, h)


def info_cov(Pi: jax.Array) -> jax.Array:
    """Posterior covariance ``Sigma = Pi^{-1}``. This is the ONE function that
    forms the inverse explicitly -- prefer ``info_mean`` / ``log_evidence`` (which
    solve) when you only need a mean or a score."""
    return jnp.linalg.inv(Pi)


# ----------------------------------------------------------------------
# The closed-form log-normalizer.
# ----------------------------------------------------------------------

def log_evidence(Pi: jax.Array, h: jax.Array) -> jax.Array:
    """Closed-form Gaussian log-normalizer

        logZ = 1/2 [ h^T Pi^{-1} h  -  logdet(Pi)  +  d log(2pi) ].

    Uses ``slogdet`` (stable log-determinant) and ``solve`` for the quadratic; the
    inverse is never formed. ``d = Pi.shape[-1]``. Accumulating the *difference*
    ``log_evidence(post) - log_evidence(prior)`` as data arrive gives a paradigm's
    running model evidence (drift rate ``-KL(P || p_M)``)."""
    _sign, logabsdet = jnp.linalg.slogdet(Pi)
    quad = h @ jnp.linalg.solve(Pi, h)
    d = Pi.shape[-1]
    return 0.5 * (quad - logabsdet + d * LOG_2PI)


# ----------------------------------------------------------------------
# Learn / fuse -- both are addition in information form.
# ----------------------------------------------------------------------

def add_information(Pi: jax.Array, h: jax.Array, J: jax.Array, j: jax.Array
                    ) -> tuple[jax.Array, jax.Array]:
    """Deposit information (LEARN) or fuse an independent belief (FUSE): the same
    running sum ``Pi + J``, ``h + j``. The matrix grows in entries, not in size."""
    return Pi + J, h + j


def fuse_stack(Pi: jax.Array, h: jax.Array, W: jax.Array
               ) -> tuple[jax.Array, jax.Array]:
    """Trust-weighted precision-addition fusion of a *stack* of nets.

    ``Pi`` (N, d, d), ``h`` (N, d), ``W`` (N, N) row-stochastic. Returns
    ``Pi'_i = sum_j W_ij Pi_j``, ``h'_i = sum_j W_ij h_j`` -- the precision-weighted
    mean recovered for free in the fused ``h``. (Opinion pooling = matrix addition.)
    """
    return jnp.einsum("ij,jab->iab", W, Pi), W @ h


def row_stochastic(weights: jax.Array, eps: float = 1e-12) -> jax.Array:
    """Normalise each row to sum to 1 -- the closed-neighbourhood fusion weights.
    ``weights`` (N, N) non-negative (e.g. ``trust * adjacency``)."""
    return weights / (weights.sum(axis=1, keepdims=True) + eps)


# ----------------------------------------------------------------------
# Likelihood: the Fisher-information deposit of one linear-Gaussian observation.
# ----------------------------------------------------------------------

def fisher_deposit(H: jax.Array, o: jax.Array, sigma_o: float
                   ) -> tuple[jax.Array, jax.Array]:
    """Information from one observation ``o = H x + eps``, ``eps ~ N(0, sigma_o^2 I)``:

        J = H^T H / sigma_o^2   (d, d),    j = H^T o / sigma_o^2   (d,).

    These are the Gaussian-likelihood precision and potential contributions; feed
    them to ``add_information``. ``H`` is (m, d), ``o`` is (m,)."""
    inv_var = 1.0 / (sigma_o ** 2)
    return (H.T @ H) * inv_var, (H.T @ o) * inv_var


def fisher_deposit_weighted(H: jax.Array, o: jax.Array, sigma_o: float,
                            weights: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-channel attention-weighted deposit: ``J = H^T diag(w) H / sigma_o^2``,
    ``j = H^T diag(w) o / sigma_o^2``. ``weights`` (m,) in [0, inf): ``w_k`` is the
    evidential precision (gain) on channel ``k`` -- ``1`` = full, ``0`` = the
    experiment is not run. All-ones recovers ``fisher_deposit`` exactly. Each row is
    scaled by ``sqrt(w_k)`` and pushed through the ordinary deposit."""
    sw = jnp.sqrt(jnp.clip(weights, 0.0, None))
    return fisher_deposit(sw[:, None] * H, sw * o, sigma_o)


# ----------------------------------------------------------------------
# Reduction: Schur complement (marginalize) and conditioning.
# ----------------------------------------------------------------------

def carryover_fillin(Pi: jax.Array, keep: jax.Array, drop: jax.Array
                     ) -> jax.Array:
    """The carry-over (fill-in) block ``Pi_ab Pi_bb^{-1} Pi_ba`` a marginalized
    block leaves among its former neighbours -- the ghost of the removed nodes.
    ``keep`` / ``drop`` are integer index arrays. Its Frobenius mass IS the
    "importance" of the dropped block."""
    Pi_ab = Pi[jnp.ix_(keep, drop)]
    Pi_bb = Pi[jnp.ix_(drop, drop)]
    Pi_ba = Pi[jnp.ix_(drop, keep)]
    return Pi_ab @ jnp.linalg.solve(Pi_bb, Pi_ba)


def schur_marginalize(Pi: jax.Array, h: jax.Array,
                      keep: jax.Array, drop: jax.Array
                      ) -> tuple[jax.Array, jax.Array]:
    """REDUCE by marginalizing out the ``drop`` block, exactly:

        Pi_a^marg = Pi_aa - Pi_ab Pi_bb^{-1} Pi_ba   (= Pi_aa - carryover_fillin)
        h_a^marg  = h_a   - Pi_ab Pi_bb^{-1} h_b.

    Survivors keep the Schur complement; the subtracted term is the fill-in among
    the removed block's neighbours. Both ``Pi_bb^{-1}(.)`` use ``solve``."""
    Pi_aa = Pi[jnp.ix_(keep, keep)]
    Pi_ab = Pi[jnp.ix_(keep, drop)]
    Pi_bb = Pi[jnp.ix_(drop, drop)]
    Pi_ba = Pi[jnp.ix_(drop, keep)]
    Pi_marg = Pi_aa - Pi_ab @ jnp.linalg.solve(Pi_bb, Pi_ba)
    h_marg = h[keep] - Pi_ab @ jnp.linalg.solve(Pi_bb, h[drop])
    return Pi_marg, h_marg


def condition(Pi: jax.Array, h: jax.Array, keep: jax.Array
              ) -> tuple[jax.Array, jax.Array]:
    """REDUCE by conditioning: in information form this is just *restricting* to the
    kept block, ``Pi_aa``, ``h_a``. (Conditioning is the cheap removal in precision
    coordinates; marginalizing is the expensive Schur one. Conditioning is the exact
    adjoint of ``border``.)"""
    return Pi[jnp.ix_(keep, keep)], h[keep]


# ----------------------------------------------------------------------
# Expansion: border a net with a new coordinate.
# ----------------------------------------------------------------------

def border(Pi: jax.Array, h: jax.Array, couplings: jax.Array,
           Pi_diag: float, h_new: float = 0.0
           ) -> tuple[jax.Array, jax.Array]:
    """EXPAND a (d, d) net to (d+1, d+1) by appending a new coordinate as a
    bordering row/column:

        Pi_new = [[ Pi,          couplings[:, None] ],
                  [ couplings,    Pi_diag           ]]

    ``couplings`` (d,) is the newcomer's off-diagonal block; no incumbent entry is
    touched. Bordering then ``schur_marginalize``-ing the same coordinate returns
    the original exactly. ``h`` grows by ``h_new``."""
    couplings = jnp.asarray(couplings)
    top = jnp.concatenate([Pi, couplings[:, None]], axis=1)              # (d, d+1)
    bottom = jnp.concatenate([couplings, jnp.asarray([Pi_diag])])[None, :]  # (1, d+1)
    Pi_new = jnp.concatenate([top, bottom], axis=0)                     # (d+1, d+1)
    h_new_vec = jnp.concatenate([h, jnp.asarray([h_new])])
    return Pi_new, h_new_vec


# ----------------------------------------------------------------------
# Bayesian Model Reduction: the closed-form log Bayes factor for a prior edit.
# ----------------------------------------------------------------------

def savage_dickey(Pi_post: jax.Array, h_post: jax.Array,
                  Pi0: jax.Array, h0: jax.Array,
                  Pi0_red: jax.Array, h0_red: jax.Array
                  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """REDUCE by pruning: the closed-form log Bayes factor for swapping the prior
    ``(Pi0, h0)`` for a sharper ``(Pi0_red, h0_red)`` under the *same* likelihood
    (matrix Savage-Dickey / Friston-Penny post-hoc identity).

    The shared likelihood deposit is ``J = Pi_post - Pi0``, ``j = h_post - h0``, so
    the reduced posterior is ``(Pi0_red + J, h0_red + j)`` and

        Delta F = [logZ(red_post) - logZ(red_prior)] - [logZ(post) - logZ(prior)].

    Exact for Gaussians. ``Delta F > 0`` => the data are content with the edit (the
    reduced model is favoured); ``< 0`` => the data hold the full model. Returns
    ``(delta_F, Pi_red_post, h_red_post)``."""
    J = Pi_post - Pi0
    j = h_post - h0
    Pi_rp = Pi0_red + J
    h_rp = h0_red + j
    delta_F = (
        (log_evidence(Pi_rp, h_rp) - log_evidence(Pi0_red, h0_red))
        - (log_evidence(Pi_post, h_post) - log_evidence(Pi0, h0))
    )
    return delta_F, Pi_rp, h_rp


def spike_prior(Pi: jax.Array, h: jax.Array, idx: jax.Array,
                value: float = 0.0, spike: float = SPIKE_PRECISION
                ) -> tuple[jax.Array, jax.Array]:
    """Build a reduced prior that pins coordinates ``idx`` to a sharp ``value``:
    zero their couplings (whole row/col) and set the diagonal to ``spike`` with
    potential ``spike * value`` (so the pinned mean is ``value``). ``value = 0``
    prunes a node toward absence; ``value = <claimed>`` sharpens a commitment to its
    asserted value (so a downstream ``savage_dickey`` tests whether the data still
    support the claim). ``idx`` is an integer index array."""
    Pi = Pi.at[idx, :].set(0.0)
    Pi = Pi.at[:, idx].set(0.0)
    Pi = Pi.at[idx, idx].set(spike)
    h = h.at[idx].set(spike * value)
    return Pi, h


def zero_edge_prior(Pi: jax.Array, u: int, v: int) -> jax.Array:
    """Pin a single coupling ``(u, v)`` to zero (the edge vanishes), leaving
    everything else intact -- a rank-2 symmetric edit. Returns the edited ``Pi``."""
    return Pi.at[u, v].set(0.0).at[v, u].set(0.0)
