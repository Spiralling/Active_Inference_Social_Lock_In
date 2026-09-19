"""Expected-free-energy attention: where to look = where you expect prediction error.

The sampling policy that chooses each agent's per-channel evidential precision ``rho_k``
(the weights fed to ``world.fisher_deposit_weighted``) as the minimiser of expected free
energy. For a Gaussian belief this is a *uniform functional on the belief net* -- the same
rule applied to every measured commitment, no privileged target node -- and it has exactly
TWO drivers, which are the same object (a discrepancy between one model's prediction and
another's) differing only in who the "other" is:

  1. **agent <-> world**  (``predictive_variance``): the agent's own predictive uncertainty
     about node v. You expect to be surprised in proportion to how loosely you hold v, so
     you look where you are uncertain. This term **self-seals**: as evidence sharpens v the
     uncertainty falls, the epistemic pull vanishes, and a confident-but-wrong agent stops
     sampling the very node that would correct it. Self-sealing is therefore *emergent*, not
     a tuned gate.

  2. **agent <-> neighbour** (``social_surprisal``): the trust-weighted, precision-weighted
     discrepancy between a neighbour's prediction and yours, per channel -- the structural
     analogue of ``src/trust.py:surprisal_matrix``. A trusted, confident neighbour who
     predicts something different about node v injects a prediction error you did not have to
     sample to receive. This is the term that can **break self-sealing**: it enters through
     trust (the door left open), not through your own attention (the door nailed shut). The
     experimentalist "vanguard" is just a sub-population initialised from the *oxygen* prior
     (``phlogiston.oxygen_prior``) -- they have already measured, so their net disagrees with
     the incumbents on the anomaly node, and that disagreement is exactly this social term.

Robustness / cost decision (matters because the phlogiston prior is an *improper* common-cause
net -- indefinite, so ``Pi^{-1}`` has no proper variance): both drivers are read from the
precision matrix's **diagonal** ``Pi[v,v]`` (the *conditional* precision of v given the rest),
never from a matrix inverse. ``1/Pi[v,v]`` is the conditional predictive variance -- always
positive when ``Pi[v,v] > 0``, needs no inversion (cheap inside the jitted scan), and is
arguably the more apt epistemic quantity ("what would measuring v tell me, holding the rest of
my beliefs fixed"). Means come from ``solve(Pi, h)`` -- exactly as ``observables`` and
``step.run_trace`` already do on these same nets.

These pure-array functions are the ``linalg``-style spec; the dynamics wire them in via
``step._channel_weights`` under ``precision_mode='efe'``. See ``notes/efe_attention.tex``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def conditional_precision(Pi: jax.Array, measured_idx: jax.Array) -> jax.Array:
    """The conditional precision ``Pi[v,v]`` of each measured node -- how tightly v is held
    given the rest of the net. ``Pi`` is ``(..., d, d)`` (single net or an ``(N,d,d)`` stack);
    ``measured_idx`` (m,) selects the measured nodes. Returns ``(..., m)``. Reads the diagonal
    only -- no inverse, so it is well-defined even for an indefinite (improper) prior."""
    diag = jnp.diagonal(Pi, axis1=-2, axis2=-1)        # (..., d)
    return diag[..., measured_idx]                      # (..., m)


def predictive_variance(Pi: jax.Array, measured_idx: jax.Array) -> jax.Array:
    """The agent<->world expected prediction error: the conditional predictive variance
    ``1 / Pi[v,v]`` of each measured node. High where the agent is uncertain (look there);
    falls to zero as evidence sharpens v (stop looking) -- the self-sealing term. ``(..., m)``."""
    return 1.0 / conditional_precision(Pi, measured_idx)


def social_surprisal(mu: jax.Array, Pi: jax.Array, W: jax.Array,
                     measured_idx: jax.Array) -> jax.Array:
    """The agent<->neighbour expected prediction error -- the structural ``surprisal_matrix``.

    ``mu`` (N, d) per-agent posterior means (``solve(Pi, h)``), ``Pi`` (N, d, d),
    ``W`` (N, N) row-stochastic trust/fusion weights, ``measured_idx`` (m,). For each agent i
    and channel k (measuring node ``v = measured_idx[k]``):

        PE[i, k] = sum_j  W_ij * Pi_j[v, v] * ( mu_j[v] - mu_i[v] )^2

    -- the trust-weighted (``W_ij``), confidence-weighted (neighbour j's conditional precision
    ``Pi_j[v,v]``) squared discrepancy between j's prediction of node v and i's. Large exactly
    where a *trusted, confident* neighbour disagrees with you about v. Returns ``(N, m)``.

    This is the quantity a vanguard transmits: it does not depend on i's own certainty (so it
    can fire on a node i has stopped sampling), only on the disagreement carried across the
    trust graph -- the formal "a trusted peer points at the node you'd nailed shut."
    """
    muk = mu[:, measured_idx]                          # (N, m) each agent's predicted observable
    pik = conditional_precision(Pi, measured_idx)      # (N, m) each agent's confidence on v
    disc = muk[None, :, :] - muk[:, None, :]           # (N, N, m): [i,j,k] = mu_j - mu_i
    weight = W[:, :, None] * pik[None, :, :]           # (N, N, m): W_ij * Pi_j[v,v]
    return (weight * disc ** 2).sum(axis=1)            # (N, m), sum over neighbours j


def _squash(x: jax.Array) -> jax.Array:
    """Bounded, monotone ``x / (1 + x)`` mapping a non-negative drive to [0, 1) -- keeps an
    unbounded prediction-error term (e.g. a vague node's large variance, or a big disagreement)
    from dominating the allocation. Order-preserving, so it never changes *which* channel is
    most salient, only the saturation."""
    return x / (1.0 + x)


def efe_channel_precision(Pi: jax.Array, h: jax.Array, W: jax.Array,
                          base_rho: jax.Array, measured_idx: jax.Array,
                          epistemic_weight: float, social_weight: float,
                          rho_max: float = 1.0) -> jax.Array:
    """Per-agent per-channel evidential precision ``rho_k`` from expected free energy.

    ``Pi`` (N, d, d), ``h`` (N, d), ``W`` (N, N), ``base_rho`` (m,) the *pragmatic* baseline
    allocation (pass ``phlogiston.derived_channel_precision(cfg)`` -- the paradigm's suppressive
    core-coupling gain, so this reduces to ``precision_mode='derived'`` when the epistemic terms
    are off), ``measured_idx`` (m,). Returns ``(N, m)``.

        drive[i,k] = epistemic_weight * squash(predictive_variance) + social_weight * squash(social_surprisal)
        rho[i,k]   = min( rho_max,  base_rho[k] + rho_max * drive[i,k] )

    The drive is **additive** over the pragmatic baseline and clipped at ``rho_max``: it can
    only *re-open* a channel the paradigm suppressed (push ``rho`` back up toward the ceiling),
    never exceed it. With ``epistemic_weight = social_weight = 0`` the drive is zero and
    ``rho = min(rho_max, base_rho) = base_rho`` (since ``base_rho <= rho_max`` by construction)
    -- byte-identical to the ``derived`` mode. So the social term is precisely the thing that
    lifts a self-sealed channel back open when a trusted neighbour disagrees.
    """
    pe_self = predictive_variance(Pi, measured_idx)            # (N, m)
    mu = jax.vmap(jnp.linalg.solve)(Pi, h)                     # (N, d)
    pe_soc = social_surprisal(mu, Pi, W, measured_idx)         # (N, m)
    drive = epistemic_weight * _squash(pe_self) + social_weight * _squash(pe_soc)
    return jnp.minimum(rho_max, base_rho[None, :] + rho_max * drive)
