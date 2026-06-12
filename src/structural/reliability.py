"""Inferred reliability: per-channel evidence weights and content-gated trust.

Gaussian likelihoods are outlier-resistant the wrong way round (O'Hagan 1979;
O'Hagan & Pericchi 2012): when two information sources conflict, precision
addition averages them into a *confident compromise* -- the posterior becomes
more certain about a configuration neither source supports, and the conflict
leaves no trace in the belief state. The standard repair is a heavy-tailed
(Student-t) likelihood, carried here as a **scale mixture of Gaussians**: each
channel's noise precision is itself a latent with a Gamma prior, and its
posterior expectation given the standardized one-step residual ``z_k`` is

    lambda_k = (nu + 1) / (nu + z_k^2),

the classic Student-t reweighting (one E-step of the EM view of t-regression).
On-prediction data (``z ~ 1``) keep full weight; a gross outlier is discounted
as ``~ nu / z^2`` -- the channel is *inferred unreliable* rather than averaged
in. Small ``nu`` => quick to call an outlier; ``nu -> inf`` recovers the
Gaussian. The same functional gates **trust between agents**:

    gamma_ij = (nu_s + 1) / (nu_s + z_ij^2),

with ``z_ij^2`` the (diagonal-approximation) Mahalanobis disagreement between
i's and j's posteriors, SUMMED over the measured nodes -- summed, not meaned,
because a *localized* disagreement must be able to sever trust (a mean over
many agreeing nodes would dilute it below any gate threshold). Uniform fusion
is a contraction, so inter-agent divergence is impossible by construction;
gating the fusion weights by ``gamma`` is what makes stable camps attainable.

House style: pure functions on raw arrays (the ``linalg.py`` / ``efe.py``
kernel discipline) -- no ``GaussianBeliefNet``, no names, no cfg. Variance
reads are diagonal-only where they enter pairwise gates (``1/Pi[v,v]``, the
conditional predictive variance -- well-defined even on the improper hub
prior, exactly as ``efe.conditional_precision`` reads it); the per-channel
predictive variance pays one stacked solve and clips against the improper
prior. The dynamics wire these in via ``step._fuse_then_observe`` /
``step._observe_pooled`` (``cfg.reliability_nu``) and ``step._gated_W``
(``cfg.social_nu``); both default to ``None`` => off, byte-identical.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def predictive_channel_variance(Pi: jax.Array, H: jax.Array, sigma_o: float
                                ) -> jax.Array:
    """Per-channel predictive variance of the observations under belief ``(Pi, .)``:

        pred_var_k = sigma_o^2 + (H Pi^{-1} H^T)[k, k],

    via one stacked solve ``S = solve(Pi, H^T)`` (never an inverse). The model
    term is clipped at 0: the phlogiston hub prior is improper (indefinite), so
    a raw ``H Pi^{-1} H^T`` diagonal can go negative -- the clip floors the
    predictive variance at the observation noise, which only makes the residual
    standardization *more* conservative. ``Pi`` (d, d), ``H`` (m, d) -> (m,)."""
    S = jnp.linalg.solve(Pi, H.T)                       # (d, m)
    model_var = jnp.einsum("kd,dk->k", H, S)            # diag(H @ Pi^{-1} @ H^T)
    return sigma_o ** 2 + jnp.clip(model_var, 0.0, jnp.inf)


def channel_z2(Pi: jax.Array, h: jax.Array, H: jax.Array, o: jax.Array,
               sigma_o: float) -> jax.Array:
    """Squared standardized one-step residual per channel:

        z_k^2 = (o_k - (H mu)_k)^2 / pred_var_k,

    the agent's own surprise at observation ``o`` BEFORE depositing it (mean
    from ``solve(Pi, h)``, variance from ``predictive_channel_variance``).
    ``z^2 ~ 1`` is on-prediction; ``z^2 >> nu`` is what the Student-t gate
    reads as an unreliable channel. Returns (m,)."""
    mu = jnp.linalg.solve(Pi, h)
    resid = o - H @ mu
    return resid ** 2 / predictive_channel_variance(Pi, H, sigma_o)


def student_t_weight(z2: jax.Array, nu: float) -> jax.Array:
    """The Student-t scale-mixture reliability weight ``(nu + 1) / (nu + z2)`` --
    the posterior expectation of a channel's latent noise scale given its
    standardized residual. ``z2 = 0`` gives the maximum ``(nu+1)/nu``; large
    ``z2`` decays as ``~ nu/z2`` (the outlier is discounted, not averaged in).
    Monotone decreasing in ``z2``; ``nu -> inf`` recovers weight 1 everywhere."""
    return (nu + 1.0) / (nu + z2)


def channel_reliability(Pi: jax.Array, h: jax.Array, H: jax.Array, o: jax.Array,
                        sigma_o: float, nu: float) -> jax.Array:
    """Per-channel inferred reliability ``lambda_k`` of one observation ``o``
    against the agent's pre-deposit belief ``(Pi, h)`` -- the composition
    ``student_t_weight(channel_z2(...), nu)``. Multiplies the existing channel
    weights ``Wt`` in ``step``, so it composes with every ``precision_mode``
    (a deposit gate like attention, but inferred from surprise, not chosen).
    Returns (m,)."""
    return student_t_weight(channel_z2(Pi, h, H, o, sigma_o), nu)


def pairwise_disagreement_z2(mu: jax.Array, Pi: jax.Array,
                             measured_idx: jax.Array) -> jax.Array:
    """Diagonal-approximation Mahalanobis disagreement between every pair of
    agents, SUMMED over the measured nodes:

        z_ij^2 = sum_v (mu_j[v] - mu_i[v])^2 / (1/Pi_i[v,v] + 1/Pi_j[v,v]).

    Summed, not meaned: a localized disagreement (one contested node among many
    agreed ones) must be able to sever trust; the mean dilutes it below any
    gate. Variances are diagonal reads ``1/Pi[v,v]`` (conditional predictive
    variance -- the ``efe.py`` convention, improper-prior-safe, no O(N^2)
    solves). ``mu`` (N, d), ``Pi`` (N, d, d), ``measured_idx`` (m,) -> (N, N)
    symmetric with zero diagonal."""
    muk = mu[:, measured_idx]                                       # (N, m)
    var = 1.0 / jnp.diagonal(Pi, axis1=-2, axis2=-1)[:, measured_idx]  # (N, m)
    diff = muk[None, :, :] - muk[:, None, :]                        # (N, N, m)
    denom = var[:, None, :] + var[None, :, :]                       # (N, N, m)
    return (diff ** 2 / denom).sum(axis=-1)                         # (N, N)


def social_gamma(Pi: jax.Array, h: jax.Array, measured_idx: jax.Array,
                 nu_s: float) -> jax.Array:
    """Content-gated trust precisions ``gamma_ij = (nu_s+1)/(nu_s+z_ij^2)`` from
    the stack of posteriors -- the Student-t gate applied to belief disagreement
    instead of data surprise. The diagonal is the maximum ``(nu_s+1)/nu_s``
    (``z_ii = 0``: self gets full trust); row-normalization in
    ``step.trust_weights`` absorbs the overall scale, so only the *ratios*
    matter. Agents who have diverged stop averaging with each other -- the
    mechanism of camps. ``Pi`` (N, d, d), ``h`` (N, d) -> (N, N)."""
    mu = jax.vmap(jnp.linalg.solve)(Pi, h)                          # (N, d)
    z2 = pairwise_disagreement_z2(mu, Pi, measured_idx)
    return student_t_weight(z2, nu_s)


def pairwise_structure_z2(Pi: jax.Array, pairs: jax.Array,
                          scale_floor: float = 0.1) -> jax.Array:
    """Pairwise STRUCTURE disagreement: how differently two agents WIRE the world.

    Where ``pairwise_disagreement_z2`` reads the divergence of the posteriors'
    *centers* (opinions), this reads the divergence of their *coupling pattern*
    -- the contested off-diagonal precision entries ``x_ik = Pi_i[a_k, c_k]``:

        z_ij^2 = sum_k (x_ik - x_jk)^2 / max((x_ik^2 + x_jk^2) / 2, scale_floor^2).

    The normalization is the relative pairwise-symmetric scale, so each
    coupling's contribution is bounded in ``[0, 4]`` (opposite-sign couplings
    -> 4; one holds, one lacks -> 2; agreement -> ~0) and the read is INVARIANT
    to the global precision growth the omega-forgetting equilibrium produces
    (both agents' couplings scaling together leaves z^2 unchanged). SUMMED over
    the contested pairs, not meaned -- a localized wiring disagreement must be
    able to sever trust. ``scale_floor`` keeps two agents who both *lack* a
    coupling (x ~ 0 vs x ~ 0) reading as agreement instead of 0/0.

    Fed to ``student_t_weight`` this is the structure-reading trust gate: agents
    extend trust by HOW YOU WIRE THE WORLD, not what opinions you hold -- the
    gate that can protect a pluralism living in the wiring while the means-gate
    (whose camps agree in means) cannot. ``Pi`` (N, d, d), ``pairs`` (k, 2) int
    -> (N, N) symmetric with zero diagonal."""
    x = Pi[:, pairs[:, 0], pairs[:, 1]]                             # (N, k)
    diff = x[:, None, :] - x[None, :, :]                            # (N, N, k)
    scale = jnp.maximum(0.5 * (x[:, None, :] ** 2 + x[None, :, :] ** 2),
                        scale_floor ** 2)                           # (N, N, k)
    return (diff ** 2 / scale).sum(axis=-1)                         # (N, N)
