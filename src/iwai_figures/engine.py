"""The numerical kernel used by both supplied manuscript versions.

Small Gaussian information-form updates, with supplied candidate support.
No utility, graph discovery, model pruning or policy selection is implemented.
Arithmetic and random-key order are retained from the figure-producing code.
"""

from __future__ import annotations
from dataclasses import dataclass
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

LOG_2PI = float(jnp.log(2.0 * jnp.pi))


def fisher_deposit(
    H: jax.Array, o: jax.Array, sigma_o: float
) -> tuple[jax.Array, jax.Array]:
    """Information from one observation ``o = H x + eps``, ``eps ~ N(0, sigma_o^2 I)``:

        J = H^T H / sigma_o^2   (d, d),    j = H^T o / sigma_o^2   (d,).

    These are the Gaussian-likelihood precision and potential contributions; feed
    them to ``add_information``. ``H`` is (m, d), ``o`` is (m,)."""
    inv_var = 1.0 / (sigma_o**2)
    return (H.T @ H) * inv_var, (H.T @ o) * inv_var


def fisher_deposit_weighted(
    H: jax.Array, o: jax.Array, sigma_o: float, weights: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Per-channel attention-weighted deposit: ``J = H^T diag(w) H / sigma_o^2``,
    ``j = H^T diag(w) o / sigma_o^2``. ``weights`` (m,) in [0, inf): ``w_k`` is the
    evidential precision (gain) on channel ``k`` -- ``1`` = full, ``0`` = the
    experiment is not run. All-ones recovers ``fisher_deposit`` exactly. Each row is
    scaled by ``sqrt(w_k)`` and pushed through the ordinary deposit."""
    sw = jnp.sqrt(jnp.clip(weights, 0.0, None))
    return fisher_deposit(sw[:, None] * H, sw * o, sigma_o)


def sample_o(
    H: jax.Array, phi_true: jax.Array, sigma_o: float, key: jax.Array
) -> jax.Array:
    """Sample one observation ``o = H @ phi_true + eps``, ``eps ~ N(0, sigma_o^2 I)``.

    ``H`` is (m, d); returns (m,). Used by the multi-agent step to feed each
    agent fresh data; agents share the same ``phi_true`` (one world) but draw
    independent noise.
    """
    mean = H @ phi_true
    noise = jax.random.normal(key, mean.shape) * sigma_o
    return mean + noise


def student_t_weight(z2: jax.Array, nu: float) -> jax.Array:
    """Reliability weight (nu + 1) / (nu + z2), used here with nu=.01.

    The social gate applies this function to a summed disagreement metric;
    it is not claimed to be an exact multivariate Student-t likelihood.
    Large disagreement gives weight asymptotic to (nu + 1) / z2.
    """
    return (nu + 1.0) / (nu + z2)


def pairwise_disagreement_z2(
    mu: jax.Array, Pi: jax.Array, measured_idx: jax.Array
) -> jax.Array:
    """Sum diagonal-normalized observable mean disagreements across all selected coordinates."""
    muk = mu[:, measured_idx]  # (N, m)
    var = 1.0 / jnp.diagonal(Pi, axis1=-2, axis2=-1)[:, measured_idx]  # (N, m)
    diff = muk[None, :, :] - muk[:, None, :]  # (N, N, m)
    denom = var[:, None, :] + var[None, :, :]  # (N, N, m)
    return (diff**2 / denom).sum(axis=-1)  # (N, N)


def social_gamma(
    Pi: jax.Array, h: jax.Array, measured_idx: jax.Array, nu_s: float
) -> jax.Array:
    """Student-t social reliability from current mean disagreement."""
    mu = jax.vmap(jnp.linalg.solve)(Pi, h)  # (N, d)
    z2 = pairwise_disagreement_z2(mu, Pi, measured_idx)
    return student_t_weight(z2, nu_s)


def log_predictive(
    Pi: jax.Array,
    h: jax.Array,
    H: jax.Array,
    o: jax.Array,
    sigma_o: float,
    jitter: float = 0.0,
) -> jax.Array:
    """Gaussian predictive log density before assimilating the new observation."""
    m = H.shape[0]
    mu = jnp.linalg.solve(Pi, h)
    S = jnp.linalg.solve(Pi, H.T)  # (d, m)
    C = sigma_o**2 * jnp.eye(m) + H @ S
    C = 0.5 * (C + C.T) + jitter * jnp.eye(m)
    cf = cho_factor(C, lower=True)
    r = o - H @ mu
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(cf[0])))
    return -0.5 * (m * LOG_2PI + logdet + r @ cho_solve(cf, r))


def log_predictive_weighted(
    Pi: jax.Array,
    h: jax.Array,
    H: jax.Array,
    o: jax.Array,
    sigma_o: float,
    w: jax.Array,
    jitter: float = 0.0,
) -> jax.Array:
    """Attention-weighted predictive density, consistent with the weighted deposit."""
    sw = jnp.sqrt(w)
    return log_predictive(Pi, h, sw[:, None] * H, sw * o, sigma_o, jitter)


def score_candidates(
    Pi_k: jax.Array,
    h_k: jax.Array,
    H: jax.Array,
    o: jax.Array,
    sigma_o: float,
    w: jax.Array,
    jitter: float = 0.0,
) -> jax.Array:
    """One agent's per-candidate prequential scores for one observation.
    ``Pi_k`` (K, d, d), ``h_k`` (K, d) -> (K,)."""
    return jax.vmap(
        lambda P, q: log_predictive_weighted(P, q, H, o, sigma_o, w, jitter)
    )(Pi_k, h_k)


def pool_log_weights(L: jax.Array, W: jax.Array, alpha_m: float) -> jax.Array:
    """Damped log-linear pooling of per-agent candidate log support."""
    return (1.0 - alpha_m) * L + alpha_m * (W @ L)


def fuse_within_frame(
    Pi: jax.Array, h: jax.Array, W: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Pool natural parameters only between matching candidates."""
    return (jnp.einsum("ij,jkab->ikab", W, Pi), jnp.einsum("ij,jkd->ikd", W, h))


def anchored_forget(
    Pi: jax.Array, h: jax.Array, Pi0: jax.Array, h0: jax.Array, omega: float
) -> tuple[jax.Array, jax.Array]:
    """Relax updated natural parameters towards fixed reference beliefs."""
    P0 = Pi0 if Pi0.ndim == Pi.ndim else Pi0[None]
    q0 = h0 if h0.ndim == h.ndim else h0[None]
    return (P0 + omega * (Pi - P0), q0 + omega * (h - q0))


@dataclass(frozen=True)
class RivalConfig:
    """Fixed scientific settings; unused legacy gates and score decay are rejected."""

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
    return (
        jnp.broadcast_to(cfg.Pi0[None], (n,) + cfg.Pi0.shape),
        jnp.broadcast_to(cfg.h0[None], (n,) + cfg.h0.shape),
        jnp.zeros((n, K)),
    )


def trust_weights(A_self, gamma=None):
    """Legacy binary-contact gate; preserve its archived epsilon denominator."""
    g = jnp.ones_like(A_self) if gamma is None else gamma
    raw = g * A_self
    return raw / (raw.sum(axis=1, keepdims=True) + 1e-12)


def _step(carry, phi, cfg, W, A_self, w_att, track_means=False):
    """Communicate, score fresh data, deposit evidence, then retain/anchor.

    One key split and N agent keys per round, independent of candidate count.
    This extracted kernel only supports mechanisms exercised by these figures.
    """
    if cfg.nu_m is not None or cfg.rho_L != 0.0:
        raise ValueError(
            "These figure experiments disable support gates and score decay"
        )
    if track_means:
        raise ValueError("Readouts are computed explicitly by the rollout wrapper")
    Pi, h, L, key = carry
    N = Pi.shape[0]
    K = cfg.n_candidates
    Wm = W
    if cfg.social_nu is not None:
        gamma = social_gamma(
            Pi[:, 0], h[:, 0], jnp.asarray(cfg.gate_idx, dtype=int), cfg.social_nu
        )
        Wn = trust_weights(A_self, gamma)
    else:
        Wn = W
    if K > 1:
        L = pool_log_weights(L, Wm, cfg.alpha_m)
    if cfg.fuse_nets:
        Pi, h = fuse_within_frame(Pi, h, Wn)
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
    aP = cfg.Pi0 if cfg.Pi0_agent is None else cfg.Pi0_agent
    ah = cfg.h0 if cfg.h0_agent is None else cfg.h0_agent
    Pi, h = anchored_forget(Pi, h, aP, ah, cfg.omega)
    L = (1.0 - cfg.rho_L) * L
    return (Pi, h, L, key), (L, jnp.zeros((N, cfg.dim)), jnp.zeros((N,)))


def to_info(B: jax.Array, b: jax.Array, s: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Compile CPD parameters into the joint information form.

        A = I - B ,    Pi = A^T diag(1/s) A ,    h = A^T diag(1/s) b .

    Valid for any ``B`` with ``(I - B)`` invertible (always true for a DAG, where
    ``B`` is nilpotent). ``s`` must be strictly positive.
    """
    d = B.shape[-1]
    A = jnp.eye(d) - B
    Dinv = 1.0 / s
    AtD = A.T * Dinv[None, :]  # A^T @ diag(Dinv)
    Pi = AtD @ A
    h = AtD @ b
    return Pi, h
