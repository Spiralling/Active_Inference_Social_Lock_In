"""The categorical active-inference agent: state inference + EFE policy.

Reimplemented in pure JAX (vmappable over N agents), mirroring pymdp's
A/B/C/D/E + expected-free-energy semantics. With a single inferred factor theta
and B = identity, state inference is exact one-line categorical Bayes, so
pymdp's iterative FPI is unnecessary; the equations here are the same ones
``pymdp.control.update_posterior_policies`` computes, plus the load-bearing
**belief-utility term** that pymdp does not compute natively.

Standard EFE for an experiment a (all "good", we MAXIMISE negative EFE):
    epistemic(a) = E_{o~Qo(.|a)} KL[ q(theta|o,a) || q(theta) ]      (salience)
    pragmatic(a) = E_{o~Qo(.|a)} [ C(o) ]                            (preference)
Belief-utility (Hyland, ours):
    belief_utility(a) = sum_theta U(theta) * q(theta | a)            (value of the
                        EXPECTED posterior under a)
    neg_efe(a) = epistemic(a) + pragmatic(a) + beta_U * belief_utility(a)
    q_pi = softmax( gamma_policy * neg_efe )

belief-utility enters policy evaluation ONLY — never the state update — so
inference stays Bayesian and we avoid the precision-additive blow-up of nb14.
Regression contract: beta_U = 0 (or flat U) recovers the pure pymdp G.

All functions are pure and operate on a single agent's (q, gm); ``vmap`` over
the batch in ``step``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

EPS = 1e-12


def _safe_log(x):
    return jnp.log(x + EPS)


def _softmax(x, axis=-1):
    x = x - jnp.max(x, axis=axis, keepdims=True)
    e = jnp.exp(x)
    return e / (jnp.sum(e, axis=axis, keepdims=True) + EPS)


# ----------------------------------------------------------------------
# State inference (exact categorical Bayes; single factor theta, B=identity)
# ----------------------------------------------------------------------

def infer_state(prior_q: jax.Array,
                A_world: jax.Array, o_world: jax.Array, action: jax.Array,
                A_social: jax.Array, o_social: jax.Array,
                social_mask: jax.Array) -> jax.Array:
    """Posterior q(theta) after one experiment outcome + one social signal.

    prior_q   : (K,) current belief over paradigms.
    A_world   : (n_o, K, A) likelihood; o_world : (n_o,) one-hot/soft outcome;
                action : scalar int index into the experiment axis.
    A_social  : (K, K) reliability; o_social : (K,) soft reported-paradigm vec.
    social_mask : scalar in [0, 1] — trust-precision down-weight on the social
                channel (1 = full weight, 0 = ignore social signal).

    log q' = log prior
           + sum_o o_world[o] log A_world[o, :, action]            (world modality)
           + social_mask * sum_o o_social[o] log A_social[o, :]    (social modality)
    """
    A_a = A_world[:, :, action]                       # (n_o, K)
    ll_world = jnp.sum(o_world[:, None] * _safe_log(A_a), axis=0)   # (K,)
    ll_social = jnp.sum(o_social[:, None] * _safe_log(A_social), axis=0)  # (K,)
    log_post = _safe_log(prior_q) + ll_world + social_mask * ll_social
    return _softmax(log_post)


# ----------------------------------------------------------------------
# Expected free energy over experiments (+ belief-utility term)
# ----------------------------------------------------------------------

def efe_terms(q: jax.Array, A_world: jax.Array, C: jax.Array, U: jax.Array,
              beta_U: float, bu_mode: str = "confirm") -> dict:
    """Per-action EFE decomposition. Returns dict of (A,) arrays.

    Qo(o|a)        = sum_theta q(theta) A_world[o, theta, a]
    post(theta|o,a) propto q(theta) A_world[o, theta, a]
    epistemic(a)   = sum_o Qo(o|a) sum_theta post log(post / q)        (salience)
    pragmatic(a)   = sum_o Qo(o|a) C(o)                                (preference)
    neg_efe(a)     = epistemic + pragmatic + beta_U * belief_utility(a)

    Belief-utility (``bu_mode``):
      * ``"confirm"`` (default) — motivated reasoning, the **non-vacuous** form:
            bu(a) = sum_theta U(theta) * E_o[ post(theta|o,a)^2 ]
        valuing CERTAINTY in the preferred paradigm. Convex in the posterior, so
        it rewards experiments whose likely outcomes confirm a high-U paradigm.
      * ``"expected_belief"`` — the linear form sum_theta U(theta) q(theta|a).
        Kept ONLY to *demonstrate* its vacuity: by the martingale property
        E_o[post] = prior, so this is constant across actions and never affects
        the policy. Do not use for real runs.

      * ``"plan_tilt"`` — risk-sensitive / "hopeful" planning. The PREDICTIVE
        and expected posterior are computed under a utility-tilted planning
        belief  q_plan ∝ q * exp(beta_U * U)  (the agent plans *as if* it
        already leaned toward the preferred paradigm), while STATE INFERENCE
        still uses the true Bayesian q (untouched). This is the faithful lift of
        notebook 16's standing belief-utility field h_U into the policy: it is
        asymmetric between paradigms, so it can drive paradigm *capture* toward
        a specific theory, yet leaves the belief update Bayesian. ``belief_utility``
        is reported as the tilt magnitude KL(q_plan||q) for diagnostics.

    All modes reduce to pure (epistemic + pragmatic) EFE when beta_U = 0 or U flat.
    """
    if bu_mode == "plan_tilt":
        q_plan = _softmax(_safe_log(q) + beta_U * U)
        joint = q_plan[None, :, None] * A_world
        Qo = jnp.sum(joint, axis=1)
        post = joint / (Qo[:, None, :] + EPS)
        kl_post = jnp.sum(post * (_safe_log(post) - _safe_log(q_plan)[None, :, None]), axis=1)
        epistemic = jnp.sum(Qo * kl_post, axis=0)
        pragmatic = jnp.sum(Qo * C[:, None], axis=0)
        q_theta_given_a = jnp.sum(Qo[:, None, :] * post, axis=0)
        tilt = jnp.sum(q_plan * (_safe_log(q_plan) - _safe_log(q)))  # scalar KL
        return {
            "epistemic": epistemic,
            "pragmatic": pragmatic,
            "belief_utility": jnp.broadcast_to(tilt, epistemic.shape),
            "neg_efe": epistemic + pragmatic,
            "q_theta_given_a": q_theta_given_a,
        }

    # joint[o, theta, a] = q(theta) * A_world[o, theta, a]
    joint = q[None, :, None] * A_world                      # (n_o, K, A)
    Qo = jnp.sum(joint, axis=1)                             # (n_o, A) predictive
    post = joint / (Qo[:, None, :] + EPS)                   # (n_o, K, A) posterior(theta|o,a)

    # epistemic = E_o KL(post||prior)
    kl = jnp.sum(post * (_safe_log(post) - _safe_log(q)[None, :, None]), axis=1)  # (n_o, A)
    epistemic = jnp.sum(Qo * kl, axis=0)                    # (A,)

    # pragmatic = E_o[C(o)]
    pragmatic = jnp.sum(Qo * C[:, None], axis=0)            # (A,)

    # expected posterior (always == prior by the tower rule; kept for diagnostics)
    q_theta_given_a = jnp.sum(Qo[:, None, :] * post, axis=0)  # (K, A)

    if bu_mode == "expected_belief":
        bu = jnp.sum(U[:, None] * q_theta_given_a, axis=0)   # (A,) — VACUOUS
    elif bu_mode == "confirm":
        # E_o[ post(theta|o,a)^2 ] = expected squared posterior (convex => non-vacuous)
        E_post_sq = jnp.sum(Qo[:, None, :] * post ** 2, axis=0)   # (K, A)
        bu = jnp.sum(U[:, None] * E_post_sq, axis=0)              # (A,)
    else:
        raise ValueError(f"unknown bu_mode {bu_mode!r}")

    neg_efe = epistemic + pragmatic + beta_U * bu
    return {
        "epistemic": epistemic,
        "pragmatic": pragmatic,
        "belief_utility": bu,
        "neg_efe": neg_efe,
        "q_theta_given_a": q_theta_given_a,
    }


def policy_posterior(q: jax.Array, A_world: jax.Array, C: jax.Array,
                     U: jax.Array, beta_U: float, gamma_policy: float,
                     bu_mode: str = "confirm") -> tuple[jax.Array, dict]:
    """q_pi over experiments = softmax(gamma_policy * neg_efe). Returns (q_pi, terms)."""
    terms = efe_terms(q, A_world, C, U, beta_U, bu_mode=bu_mode)
    q_pi = _softmax(gamma_policy * terms["neg_efe"])
    return q_pi, terms


def sample_action(q_pi: jax.Array, key: jax.Array) -> jax.Array:
    """Sample an experiment index from the policy posterior."""
    return jax.random.categorical(key, _safe_log(q_pi))


# ----------------------------------------------------------------------
# Batched (population) wrappers — vmap over N agents
# ----------------------------------------------------------------------

# q_batch: (N, K); A_world shared; per-agent U allowed (N, K); etc.

def infer_state_batch(prior_q, A_world, o_world, action, A_social, o_social,
                      social_mask):
    """vmap of infer_state over agents. Shapes: prior_q (N,K), o_world (N,n_o),
    action (N,), o_social (N,K), social_mask (N,). A_world/A_social shared."""
    f = lambda pq, ow, a, osoc, m: infer_state(pq, A_world, ow, a, A_social, osoc, m)
    return jax.vmap(f)(prior_q, o_world, action, o_social, social_mask)


def policy_posterior_batch(q, A_world, C, U, beta_U, gamma_policy,
                           bu_mode: str = "confirm"):
    """vmap of policy_posterior over agents. q (N,K), U (N,K) per-agent
    belief-utility (heterogeneity = the symmetry-breaking field)."""
    f = lambda qi, Ui: policy_posterior(qi, A_world, C, Ui, beta_U, gamma_policy,
                                        bu_mode=bu_mode)
    return jax.vmap(f)(q, U)
