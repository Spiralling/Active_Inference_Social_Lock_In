"""The linear-Gaussian data-generating process for the structural model.

The matrix generalisation of ``src/world.py``. There the scalar world is
``o ~ N(h0(x) + theta h1(x), sigma^2)``; here the world reads the whole vector
of commitments ``phi in R^d`` through an observation operator and a single
experiment yields

    o = H @ phi_true + eps ,    eps ~ N(0, Sigma_o),

where ``H`` is (m, d) — row ``k`` is the linear functional of the commitments
that experiment-modality ``k`` measures. Keeping the likelihood linear-Gaussian
is what makes every belief stay Gaussian and BMR stay closed-form (the locked
design decision: paradigms differ by *prior topology*, never by the likelihood).

Each observation deposits Fisher information

    J = H^T Sigma_o^{-1} H   (d, d) precision contribution
    j = H^T Sigma_o^{-1} o   (d,)   potential contribution

which ``src/structural/belief.add_fisher`` accumulates. The observation operator
``H`` is held constant (``phlogiston.H_observable``); the "regime" lives in the
*world truth* — ``phlogiston.phi_true_at`` flips the mass-law commitments at
``t_shift`` (the matrix analogue of ``src/world.theta_schedule``). That flip is
what re-orders the candidate paradigms' evidence drifts and produces the crossing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.structural import linalg


def fisher_deposit(H: jax.Array, o: jax.Array, sigma_o: float
                   ) -> tuple[jax.Array, jax.Array]:
    """Information deposited by one observation ``o`` under operator ``H``.

    ``H``       : (m, d) observation operator (m modalities, d commitments).
    ``o``       : (m,) observed values.
    ``sigma_o`` : observation noise std (isotropic, Sigma_o = sigma_o^2 I).

    Returns ``(J, j)`` with ``J = H^T H / sigma_o^2`` (d, d) and
    ``j = H^T o / sigma_o^2`` (d,). These are exactly the Gaussian-likelihood
    precision and potential contributions; ``add_fisher`` sums them into a
    belief. (Isotropic Sigma_o keeps the deposit a clean ``H^T H`` Gram matrix;
    a full Sigma_o would interpose ``solve(Sigma_o, .)``.)

    Thin wrapper over ``linalg.fisher_deposit`` (the reusable array kernel); kept
    here as the scenario-facing name the rest of ``src/structural`` imports.
    """
    return linalg.fisher_deposit(H, o, sigma_o)


def fisher_deposit_weighted(H: jax.Array, o: jax.Array, sigma_o: float,
                            weights: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-modality *attention-weighted* information deposit -- the mechanism of
    biased experiment selection.

    ``weights`` : (m,) in [0, 1], one per measurement row. ``w_k = 1`` runs
    experiment ``k`` at full precision; ``w_k = 0`` means the agent simply does
    not perform that experiment, so it gathers *no* information on those
    commitments. Returns ``J = H^T diag(w) H / sigma_o^2`` and
    ``j = H^T diag(w) o / sigma_o^2`` -- i.e. each row is scaled by ``sqrt(w_k)``
    and fed through the ordinary deposit, so all-ones weights recover
    ``fisher_deposit`` exactly. An agent that down-weights the rows its current
    model deems irrelevant (e.g. a phlogistonist skipping careful gravimetry)
    therefore receives *different data about the same world* than a peer with a
    different model -- the heart of theory-laden, motivated observation.

    Thin wrapper over ``linalg.fisher_deposit_weighted``.
    """
    return linalg.fisher_deposit_weighted(H, o, sigma_o, weights)


def sample_o(H: jax.Array, phi_true: jax.Array, sigma_o: float,
             key: jax.Array) -> jax.Array:
    """Sample one observation ``o = H @ phi_true + eps``, ``eps ~ N(0, sigma_o^2 I)``.

    ``H`` is (m, d); returns (m,). Used by the multi-agent step to feed each
    agent fresh data; agents share the same ``phi_true`` (one world) but draw
    independent noise.
    """
    mean = H @ phi_true
    noise = jax.random.normal(key, mean.shape) * sigma_o
    return mean + noise
