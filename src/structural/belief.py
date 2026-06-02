"""The one object: a Gaussian belief in information (precision) form.

This is the matrix generalisation of ``src/inference.py``. Where that module
carries a scalar belief over a single theta as ``(mu, tau)`` (mean and
precision), here a belief over a *vector* of theoretical commitments
``phi in R^d`` is carried in information form as ``(Pi, h)``:

    p(phi) propto exp( -1/2 phi^T Pi phi + h^T phi ),

with precision matrix ``Pi = Sigma^{-1}`` (d x d) and potential
``h = Pi @ mu`` (d,). Read ``Pi`` as the paradigm net written as numbers: the
diagonal ``Pi[i,i]`` is self-precision (belief in commitment i on its own), the
off-diagonal ``Pi[i,j]`` is the inferential coupling (an edge); a zero
off-diagonal is a missing edge (conditional independence). See
``notes/bmr_feynman_linear_algebra.tex`` for the full picture.

This module holds the *closed algebra* — transforms that always succeed and
never need a matrix inverse:

  - ``vague_prior``  : an uninformative belief, ``Pi = kappa I``.
  - ``add_fisher``   : LEARN. Deposit one observation's Fisher information,
                       ``Pi += J``, ``h += j``. Size unchanged.
  - ``combine``      : fuse two *independent* beliefs by ADDITION,
                       ``Pi1 + Pi2``, ``h1 + h2`` (note Eq. add).
  - ``border``       : EXPAND. Append a new node as a new row/column whose
                       off-diagonal is the newcomer's couplings (note Fig. border).

The scoring/decision layer that needs the inverse (log-evidence, Schur
complement, BMR pruning) lives in ``src/structural/bmr.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.structural import linalg


@dataclass(frozen=True)
class GaussianBeliefNet:
    """A Gaussian belief over commitments ``phi`` in information form.

    ``Pi``    : (d, d) precision / information matrix, symmetric PSD.
    ``h``     : (d,) potential vector = Pi @ mu.
    ``names`` : length-d tuple naming the basis; the order of ``names`` IS the
                row/column order of ``Pi`` and ``h``. Two beliefs may only be
                combined when their ``names`` match exactly (same shared basis).

    The dataclass is frozen and treated as immutable; every operation returns a
    fresh ``GaussianBeliefNet``. ``names`` is static metadata (it does not enter
    array math), so this is intentionally a plain dataclass rather than an
    equinox.Module — the structural algebra runs on small (d ~ 10) matrices and
    is not vmapped over the basis.
    """

    Pi: jax.Array
    h: jax.Array
    names: tuple[str, ...]

    @property
    def dim(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        """Row/column index of a named commitment."""
        return self.names.index(name)

    def indices(self, names: tuple[str, ...]) -> jax.Array:
        """Row/column indices of a tuple of named commitments."""
        return jnp.asarray([self.names.index(n) for n in names], dtype=jnp.int32)

    def mean(self) -> jax.Array:
        """Posterior mean mu = Pi^{-1} h, solved (never inverted)."""
        return linalg.info_mean(self.Pi, self.h)

    def cov(self) -> jax.Array:
        """Posterior covariance Sigma = Pi^{-1}. Forms the inverse — use
        sparingly; prefer ``mean`` / ``bmr.log_evidence`` which solve instead."""
        return linalg.info_cov(self.Pi)


def vague_prior(names: tuple[str, ...], kappa: float = 1e-3) -> GaussianBeliefNet:
    """An (almost) uninformative belief on the given basis: ``Pi = kappa I``,
    ``h = 0``.

    This is the "an agent who does not track a node holds a vague prior on it"
    device that lets every agent live on a common global basis so beliefs add
    (see ``src/structural/step.py:fuse``). ``kappa`` is small but strictly
    positive so ``Pi`` stays invertible for the log-evidence / Schur steps; too
    large and a vague prior contaminates fusion, too small and ``Pi`` is
    ill-conditioned for ``slogdet``.
    """
    d = len(names)
    Pi = kappa * jnp.eye(d)
    h = jnp.zeros((d,))
    return GaussianBeliefNet(Pi=Pi, h=h, names=names)


def add_fisher(net: GaussianBeliefNet, J: jax.Array, j: jax.Array
               ) -> GaussianBeliefNet:
    """LEARN: deposit one observation's information (note Eq. accum).

    ``J`` : (d, d) Fisher-information deposit (precision contribution).
    ``j`` : (d,) potential contribution.

    Posterior in information form is a running sum:
        Pi' = Pi + J ,   h' = h + j.
    The matrix grows in entries, not in size. Compare the scalar
    ``src/inference.py:private_update`` step ``tau' = tau + h1(x)^2/sigma^2``:
    this is its matrix lift. (Array math delegated to ``linalg.add_information``.)
    """
    Pi, h = linalg.add_information(net.Pi, net.h, J, j)
    return GaussianBeliefNet(Pi=Pi, h=h, names=net.names)


def combine(net1: GaussianBeliefNet, net2: GaussianBeliefNet
            ) -> GaussianBeliefNet:
    """Fuse two INDEPENDENT beliefs on the same basis by addition (note Eq. add).

        Pi = Pi1 + Pi2 ,   h = h1 + h2.

    Multiplying densities became adding matrices: this is why combination is
    cheap and why ``step.fuse`` is a one-liner. Requires identical ``names``.
    """
    if net1.names != net2.names:
        raise ValueError(
            "combine requires identical names (shared basis); "
            f"got {net1.names} vs {net2.names}"
        )
    Pi, h = linalg.add_information(net1.Pi, net1.h, net2.Pi, net2.h)
    return GaussianBeliefNet(Pi=Pi, h=h, names=net1.names)


def border(net: GaussianBeliefNet, new_name: str,
           Pi_diag: float, couplings: jax.Array,
           h_new: float = 0.0) -> GaussianBeliefNet:
    """EXPAND: append a genuinely new commitment as a bordering row/column
    (note Fig. border).

    ``new_name`` : name of the newcomer (must not already be in the basis).
    ``Pi_diag``  : the newcomer's self-precision (new diagonal entry).
    ``couplings``: (d,) the newcomer's couplings to the incumbents — the
                   off-diagonal block ``Pi_ab`` (the conditionals p(k | new)).
    ``h_new``    : the newcomer's potential entry (default 0).

    The matrix grows from d x d to (d+1) x (d+1):
        Pi_new = [[ Pi,            couplings[:, None] ],
                  [ couplings[None, :], Pi_diag        ]]
    No incumbent entry is touched. Bordering then Schur-complementing the same
    node returns the original exactly (see ``bmr.schur_marginalize``).
    """
    if new_name in net.names:
        raise ValueError(f"{new_name!r} already in basis {net.names}")
    couplings = jnp.asarray(couplings)
    d = net.dim
    if couplings.shape != (d,):
        raise ValueError(f"couplings must have shape ({d},), got {couplings.shape}")

    Pi_new, h_new_vec = linalg.border(net.Pi, net.h, couplings, Pi_diag, h_new)
    return GaussianBeliefNet(Pi=Pi_new, h=h_new_vec,
                             names=net.names + (new_name,))
