"""Represented rivals: q(phi, m) -- agents that hold candidate WIRINGS, not one net.

The structural-divergence diagnosis (paper/slides/structural_divergence_talk.pdf): a
single information-form Gaussian ``(Pi, h)`` has no coordinate that can mean "torn
between two hypotheses" -- pooling is a contraction, conflict resolves by confident
compromise, and the evidence that DISCRIMINATES two wirings is erased at deposit time
(the Fisher deposit ``J = H^T diag(w) H / sigma^2`` never sees the observation). This
module supplies the missing slot, the talk's slide-12 requirements made executable:

  1. **Multiple structural hypotheses per agent** -- each agent carries K candidate
     nets ``(Pi_ik, h_ik)`` plus log-weights ``L_ik`` over a model index m. Each step
     every candidate is scored by its PREQUENTIAL predictive density BEFORE the shared
     deposit lands (``log p(o_{1:T} | M_k) = sum_t log p(o_t | o_{<t}, M_k)``), so
     structure-discriminating evidence ACCUMULATES in ``L`` instead of being erased.
  2. **Divergence as state** -- :func:`gaussian_kl` / :func:`rival_divergence` carry
     the gap between an agent's live candidates (read-outs here; acting on *where*
     two wirings disagree -- experiment selection -- is the named extension).
  3. **Communication of hypotheses** -- :func:`pool_log_weights`: damped log-linear
     pooling of the log-weights over the trust graph (``L <- (1-a)L + a W L``), i.e.
     agents transmit *which structures they entertain and how strongly*. Net fusion is
     HYPOTHESIS-ALIGNED (:func:`fuse_within_frame`): candidate k fuses only with
     neighbours' candidate k -- parameter beliefs are shared within a frame, never
     averaged across frames. Incommensurability is structural, not stipulated.

Why this buys WALLS where trust gates only buy time: the pooled log-weight dynamics
have a per-agent evidence SOURCE ``s_i(t)`` (the per-step log-likelihood differences
its own theory-laden observations produce). With damped pooling at rate ``alpha_m``,
deviations from the population mean converge to ``(I - W)^+ (s - s_bar)/alpha_m`` --
on a complete graph with two symmetric camps, a CONSTANT log-odds gap of about
``s_gap / (2 alpha_m)`` nats. A few nats saturate the softmax: two camps, both
internally certain, holding DIFFERENT wirings, at a genuine fixed point of the
coupled dynamics -- not the metastable slow-mixing of the Student-t gates
(``gated_divergence`` et al.: "consensus remains the only fixed point").

Mechanism note, stated honestly: anchored forgetting on the candidate nets
(``Pi_ik <- Pi0_k + omega (Pi_ik - Pi0_k)``) is PART of the mechanism -- without it
every candidate's precision grows without bound, their predictive densities converge,
and the per-step source decays ~1/t. The Gaussian baselines have no candidate object
to anchor to (the rival was averaged away at deposit time); that asymmetry is
constitutive of "keeping the rival represented", not a tuning trick.

House style: pure functions on raw arrays; one frozen :class:`RivalConfig`; a compact
``lax.scan`` engine (:func:`run_rival`) -- the mixture state ``(N, K, d, d)`` is a
different object from the single-net engines and deliberately lives here, not in
``step.py`` / ``simulation.py``. K = 1 reduces to plain fuse-then-observe, which is
how the same engine runs the Gaussian baselines on byte-identical observation streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from src.structural import reliability as rel
from src.structural.linalg import LOG_2PI
from src.structural.step import trust_weights
from src.structural.world import sample_o, fisher_deposit_weighted


# ----------------------------------------------------------------------
# 1. The prequential score: full predictive log-density of one observation.
# ----------------------------------------------------------------------

def log_predictive(Pi: jax.Array, h: jax.Array, H: jax.Array, o: jax.Array,
                   sigma_o: float, jitter: float = 0.0) -> jax.Array:
    """``log N(o; H mu, sigma_o^2 I + H Pi^{-1} H^T)`` -- the marginal likelihood of
    one observation under belief ``(Pi, h)``, the quantity whose running sum is the
    prequential model evidence ``log p(o_{1:t} | M)``.

    Solves, never inverts: ``S = solve(Pi, H^T)`` then one (m, m) Cholesky.
    PD CONTRACT: ``Pi`` must be positive-definite (true for the cosmology presets --
    diagonally dominant -- and for any ``LinearGaussianBN.to_info()`` compile). The
    indefinite phlogiston hub prior is OUT of contract; ``jitter`` is the escape
    hatch for near-singular cases, never a clip (a clipped covariance is not a
    density). ``Pi`` (d, d), ``h`` (d,), ``H`` (m, d), ``o`` (m,) -> scalar."""
    m = H.shape[0]
    mu = jnp.linalg.solve(Pi, h)
    S = jnp.linalg.solve(Pi, H.T)                          # (d, m)
    C = sigma_o ** 2 * jnp.eye(m) + H @ S
    C = 0.5 * (C + C.T) + jitter * jnp.eye(m)
    cf = cho_factor(C, lower=True)
    r = o - H @ mu
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(cf[0])))
    return -0.5 * (m * LOG_2PI + logdet + r @ cho_solve(cf, r))


def log_predictive_weighted(Pi: jax.Array, h: jax.Array, H: jax.Array, o: jax.Array,
                            sigma_o: float, w: jax.Array,
                            jitter: float = 0.0) -> jax.Array:
    """Attention-weighted predictive density: row ``k`` of the observation model is
    scaled by ``sqrt(w_k)`` -- EXACTLY the likelihood whose Fisher information is
    ``fisher_deposit_weighted``'s ``J = H^T diag(w) H / sigma^2``, so the score and
    the deposit describe the same theory-laden experiment. A ``w_k = 0`` row
    contributes a candidate-INDEPENDENT constant (it cancels in the softmax over
    models): an experiment you do not run discriminates nothing."""
    sw = jnp.sqrt(w)
    return log_predictive(Pi, h, sw[:, None] * H, sw * o, sigma_o, jitter)


def score_candidates(Pi_k: jax.Array, h_k: jax.Array, H: jax.Array, o: jax.Array,
                     sigma_o: float, w: jax.Array, jitter: float = 0.0) -> jax.Array:
    """One agent's per-candidate prequential scores for one observation.
    ``Pi_k`` (K, d, d), ``h_k`` (K, d) -> (K,)."""
    return jax.vmap(
        lambda P, q: log_predictive_weighted(P, q, H, o, sigma_o, w, jitter)
    )(Pi_k, h_k)


# ----------------------------------------------------------------------
# 2. The model index: posterior, hypothesis communication, hypothesis gate.
# ----------------------------------------------------------------------

def model_posterior(L: jax.Array) -> jax.Array:
    """``q(m) = softmax(L)`` per row (logsumexp-stable; shift-invariant)."""
    return jax.nn.softmax(L, axis=-1)


def pool_log_weights(L: jax.Array, W: jax.Array, alpha_m: float) -> jax.Array:
    """COMMUNICATION OF HYPOTHESES: damped log-linear pooling of the model
    log-weights over the trust graph,

        L <- (1 - alpha_m) L + alpha_m (W @ L).

    Log-linear (not linear) because log-weights are the additive evidence scale:
    pooling is DeGroot in log space, and each agent's own per-step scores act as a
    persistent source term. Deviations from consensus converge to
    ``(I - W)^+ (s - s_bar) / alpha_m`` -- a CLOSED-FORM wall height: camps with
    different theory-laden evidence streams hold a constant log-odds gap
    ``~ s_gap / (2 alpha_m)`` on a complete graph. ``L`` (N, K), ``W`` (N, N)."""
    return (1.0 - alpha_m) * L + alpha_m * (W @ L)


def categorical_jsd(q: jax.Array, eps: float = 1e-12) -> jax.Array:
    """Pairwise Jensen-Shannon divergence between the rows of ``q`` (N, K) ->
    (N, N), symmetric, zero diagonal, bounded by log 2. The hypothesis-space
    disagreement read-out (and the statistic of the m-gate below)."""
    qi, qj = q[:, None, :], q[None, :, :]
    mix = 0.5 * (qi + qj)

    def kl(p, r):
        return jnp.sum(p * (jnp.log(p + eps) - jnp.log(r + eps)), axis=-1)

    return 0.5 * kl(qi, mix) + 0.5 * kl(qj, mix)


def m_gated_weights(L: jax.Array, A_self: jax.Array, nu_m: float) -> jax.Array:
    """Trust gated on HYPOTHESIS disagreement: the house Student-t functional
    (``reliability.student_t_weight``) pointed at the categorical JSD between
    agents' model posteriors, row-normalized over the graph support. The named
    rescue for the lone-discoverer problem (plain log-linear pooling lets a
    majority swamp a minority's evidence); not used by the symmetric-camps demo."""
    gamma = rel.student_t_weight(categorical_jsd(model_posterior(L)), nu_m)
    return trust_weights(A_self, gamma)


# ----------------------------------------------------------------------
# 3. Hypothesis-aligned fusion + anchored forgetting.
# ----------------------------------------------------------------------

def fuse_within_frame(Pi: jax.Array, h: jax.Array, W: jax.Array
                      ) -> tuple[jax.Array, jax.Array]:
    """Trust-weighted precision fusion WITHIN each hypothesis frame: candidate k
    fuses only with neighbours' candidate k. Parameter beliefs are shared inside a
    frame; frames are never averaged into each other -- the cross-frame averaging
    that produced slide 10's blurred third wiring cannot occur by construction.
    ``Pi`` (N, K, d, d), ``h`` (N, K, d); K = 1 is exactly ``step.fuse``."""
    return (jnp.einsum("ij,jkab->ikab", W, Pi),
            jnp.einsum("ij,jkd->ikd", W, h))


def anchored_forget(Pi: jax.Array, h: jax.Array, Pi0: jax.Array, h0: jax.Array,
                    omega: float) -> tuple[jax.Array, jax.Array]:
    """Relax every candidate toward ITS OWN reference prior:
    ``Pi_ik <- Pi0_k + omega (Pi_ik - Pi0_k)`` (and likewise h). ``omega = 1`` is a
    no-op. Part of the rival mechanism: it bounds each candidate's precision so the
    candidates' predictive densities stay distinct and the per-step log-odds source
    stays stationary (without it the source decays ~1/t and the wall erodes). It
    also bounds the TRANSMITTED confidence in fusion: under ``omega = 1`` a sender's
    potential scales with its unboundedly growing precision, so even a severed trust
    gate leaks means at rate ~ (tiny weight) x (huge precision) -- the deep reason
    means-gates are metastable at long horizons.
    ``Pi`` (N, K, d, d); ``Pi0`` (K, d, d) shared menu or (N, K, d, d) per agent."""
    P0 = Pi0 if Pi0.ndim == Pi.ndim else Pi0[None]
    q0 = h0 if h0.ndim == h.ndim else h0[None]
    return (P0 + omega * (Pi - P0), q0 + omega * (h - q0))


# ----------------------------------------------------------------------
# 4. Divergence as state (read-outs).
# ----------------------------------------------------------------------

def gaussian_kl(Pi1: jax.Array, h1: jax.Array, Pi2: jax.Array, h2: jax.Array
                ) -> jax.Array:
    """``KL(N1 || N2)`` in information form (both PD):

        0.5 [ tr(Pi1^{-1} Pi2) - d + dmu^T Pi2 dmu + logdet Pi1 - logdet Pi2 ].
    """
    d = Pi1.shape[-1]
    mu1 = jnp.linalg.solve(Pi1, h1)
    mu2 = jnp.linalg.solve(Pi2, h2)
    dmu = mu1 - mu2
    tr = jnp.trace(jnp.linalg.solve(Pi1, Pi2))
    s1 = jnp.linalg.slogdet(Pi1)[1]
    s2 = jnp.linalg.slogdet(Pi2)[1]
    return 0.5 * (tr - d + dmu @ (Pi2 @ dmu) + s1 - s2)


def gaussian_sym_kl(Pi1: jax.Array, h1: jax.Array, Pi2: jax.Array, h2: jax.Array
                    ) -> jax.Array:
    """Symmetrized KL (honest naming: this is sym-KL, not the JSD)."""
    return 0.5 * (gaussian_kl(Pi1, h1, Pi2, h2) + gaussian_kl(Pi2, h2, Pi1, h1))


def rival_divergence(Pi_k: jax.Array, h_k: jax.Array, L: jax.Array) -> jax.Array:
    """DIVERGENCE AS STATE: per agent, the sym-KL between its two strongest live
    candidates -- how torn the agent is, and where acting on disagreement (crucial
    experiments) would read from. ``Pi_k`` (N, K, d, d), ``L`` (N, K) -> (N,)."""
    def one(P, q, l):
        _, ix = jax.lax.top_k(l, 2)
        return gaussian_sym_kl(P[ix[0]], q[ix[0]], P[ix[1]], q[ix[1]])
    return jax.vmap(one)(Pi_k, h_k, L)


# ----------------------------------------------------------------------
# 5. The engine.
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RivalConfig:
    """Everything the rival engine needs beyond the per-run state.

    ``Pi0`` / ``h0`` : (K, d, d) / (K, d) the candidate reference priors (the menu).
    ``H``            : (m, d) observation operator. ``sigma_o`` : noise std.
    ``omega``        : anchored forgetting on the candidate nets (1 = off).
    ``alpha_m``      : hypothesis-pooling rate (wall height ~ s_gap / (2 alpha_m)).
    ``rho_L``        : decay on L (re-openability knob; 0 = evidence never expires).
    ``nu_m``         : optional Student-t gate on hypothesis disagreement (None = off).
    ``social_nu`` / ``gate_idx`` : optional means-gate for K = 1 BASELINES (the
                       ``reliability.social_gamma`` gate; None = fixed trust).
    ``fuse_nets``    : hypothesis-aligned net fusion on/off.
    ``jitter``       : numerical escape hatch for ``log_predictive`` (default 0).
    """

    Pi0: jax.Array
    h0: jax.Array
    H: jax.Array
    sigma_o: float
    omega: float = 1.0
    alpha_m: float = 0.05
    rho_L: float = 0.0
    nu_m: float | None = None
    social_nu: float | None = None
    gate_idx: tuple = ()
    fuse_nets: bool = True
    jitter: float = 0.0
    # optional PER-AGENT anchors (N, K, d, d)/(N, K, d) -- the K = 1 baselines with
    # omega < 1 need each camp to relax toward its OWN prior, not a shared menu.
    Pi0_agent: jax.Array | None = None
    h0_agent: jax.Array | None = None

    @property
    def n_candidates(self) -> int:
        return int(self.Pi0.shape[0])

    @property
    def dim(self) -> int:
        return int(self.Pi0.shape[-1])


def init_uniform(cfg: RivalConfig, n: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Every agent starts at the candidate menu with uniform log-weights:
    ``Pi (N, K, d, d)``, ``h (N, K, d)``, ``L (N, K) = 0``."""
    K = cfg.n_candidates
    return (jnp.broadcast_to(cfg.Pi0[None], (n,) + cfg.Pi0.shape),
            jnp.broadcast_to(cfg.h0[None], (n,) + cfg.h0.shape),
            jnp.zeros((n, K)))


def _step(carry, phi, cfg: RivalConfig, W, A_self, w_att):
    """One round: gate -> communicate hypotheses -> fuse within frame -> score
    (prequential, BEFORE deposit) -> deposit into every candidate -> forget.
    The RNG layout (one split + N keys per step) is independent of K, the gates
    and ``fuse_nets``, so every condition sees byte-identical observations."""
    Pi, h, L, key = carry
    N = Pi.shape[0]
    K = cfg.n_candidates

    # 1. gates (static branches on the frozen cfg)
    Wm = m_gated_weights(L, A_self, cfg.nu_m) if cfg.nu_m is not None else W
    if cfg.social_nu is not None:                       # K = 1 baseline means-gate
        gamma = rel.social_gamma(Pi[:, 0], h[:, 0],
                                 jnp.asarray(cfg.gate_idx, dtype=int), cfg.social_nu)
        Wn = trust_weights(A_self, gamma)
    else:
        Wn = W

    # 2. communicate hypotheses
    if K > 1:
        L = pool_log_weights(L, Wm, cfg.alpha_m)

    # 3. fuse parameter beliefs within each hypothesis frame
    if cfg.fuse_nets:
        Pi, h = fuse_within_frame(Pi, h, Wn)

    # 4. observe: score every candidate BEFORE the shared deposit lands
    key, sk = jax.random.split(key)
    keys = jax.random.split(sk, N)

    def obs_one(P_k, q_k, k_i, w_i):
        o = sample_o(cfg.H, phi, cfg.sigma_o, k_i)
        s = score_candidates(P_k, q_k, cfg.H, o, cfg.sigma_o, w_i, cfg.jitter)
        J, j = fisher_deposit_weighted(cfg.H, o, cfg.sigma_o, w_i)
        return s, J, j

    s_all, J_all, j_all = jax.vmap(obs_one)(Pi, h, keys, w_att)
    L = L + s_all
    Pi = Pi + J_all[:, None, :, :]
    h = h + j_all[:, None, :]

    # 5. anchored forgetting + optional evidence decay
    aP = cfg.Pi0 if cfg.Pi0_agent is None else cfg.Pi0_agent
    ah = cfg.h0 if cfg.h0_agent is None else cfg.h0_agent
    Pi, h = anchored_forget(Pi, h, aP, ah, cfg.omega)
    L = (1.0 - cfg.rho_L) * L

    # per-step read-outs: MAP-candidate effective net mean + how torn each agent is
    onehot = jax.nn.one_hot(jnp.argmax(L, axis=-1), K)
    Pi_eff = jnp.einsum("nk,nkab->nab", onehot, Pi)
    h_eff = jnp.einsum("nk,nkd->nd", onehot, h)
    mu_eff = jax.vmap(jnp.linalg.solve)(Pi_eff, h_eff)
    D_top2 = rival_divergence(Pi, h, L) if K > 1 else jnp.zeros((N,))
    return (Pi, h, L, key), (L, mu_eff, D_top2)


def run_rival(cfg: RivalConfig, W: jax.Array, A_self: jax.Array, w_att: jax.Array,
              phis: jax.Array, seed: int = 0,
              Pi_init: jax.Array | None = None,
              h_init: jax.Array | None = None,
              L_init: jax.Array | None = None) -> dict:
    """Scan the rival dynamics over a precomputed world ``phis`` (T, d).

    ``W`` (N, N) fixed row-stochastic trust; ``A_self`` (N, N) adjacency + I (the
    gates' support); ``w_att`` (N, m) per-agent attention over observation rows.
    ``Pi_init`` / ``h_init`` / ``L_init`` override the uniform menu start -- the
    K = 1 baselines hand each camp a different single-candidate prior, and
    multi-PHASE runs (e.g. a shared normal-science era followed by per-community
    attention regimes) chain the final state of one call into the next. Returns
    host arrays: ``L_t (T, N, K)``, ``q_t``, ``mu_eff_t (T, N, d)``,
    ``D_top2_t (T, N)`` and the final state."""
    import numpy as np

    n = int(W.shape[0])
    Pi, h, L = init_uniform(cfg, n)
    if Pi_init is not None:
        Pi, h = jnp.asarray(Pi_init), jnp.asarray(h_init)
    if L_init is not None:
        L = jnp.asarray(L_init)
    key = jax.random.PRNGKey(seed)

    def step_fn(carry, phi):
        return _step(carry, phi, cfg, W, A_self, w_att)

    (Pi, h, L, _), (L_t, mu_eff_t, D_t) = jax.lax.scan(
        step_fn, (Pi, h, L, key), jnp.asarray(phis))
    L_t = np.asarray(L_t)
    return {
        "L_t": L_t,                                     # (T, N, K)
        "q_t": np.asarray(jax.nn.softmax(jnp.asarray(L_t), axis=-1)),
        "mu_eff_t": np.asarray(mu_eff_t),               # (T, N, d)
        "D_top2_t": np.asarray(D_t),                    # (T, N)
        "Pi": np.asarray(Pi), "h": np.asarray(h), "L": np.asarray(L),
    }
