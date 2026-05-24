"""Generative model for the fixed-state epistemic POMDP (Phase 1).

The David-Hyland reframing (2026-05-20), confirmed option B:

  * Hidden state theta in {0, ..., K-1} = which paradigm/theory is correct.
    FIXED truth (B = strict identity); the agent infers q(theta). Paradigm
    "capture" is movement of the belief, an absorbing consensus.
  * A "theory" is a likelihood A_theta(o | a): for paradigm theta and
    experiment a, the distribution over discretized outcomes o. Theory-
    ladenness = the columns A[:, theta, a] differ across theta, so the same
    (experiment, outcome) is read differently depending on which paradigms are
    entertained. Underdetermination = the columns are close (low discriminability).
  * Belief-utility U(theta): a preference over WHICH paradigm is true, separate
    from the probability q(theta). It enters EFE / policy evaluation only (see
    ``agent_pop``), never the belief update — this is the Hyland lambda.U done
    right and avoids the precision-additive blow-up of notebook 14.

Discretization of the legacy world (``src/world.py``): for experiment x,
paradigm theta predicts o ~ N(h0(x) + theta_val(theta) * h1(x), sigma^2). We
bin o into ``n_o`` categories and integrate the Gaussian over each bin to get
A_world[o | theta, x]. The Fisher-information discriminability h1(x)^2/sigma^2
becomes the per-experiment KL between paradigm columns (``discriminability``),
the ``d`` knob whose role notebook 16 mapped.

Single hidden factor by design: the "state of nature s" of option B is, in the
minimal faithful cut, identified with the true paradigm theta* (the environment
generates from theta*). A separate fixed s-factor can be added later if the
truth must be richer than "one of the candidate paradigms"; the docstrings flag
where that would slot in. All arrays are plain JAX/numpy following pymdp's
A/B/C/D/E conventions (axis 0 = outcome/state being distributed over).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from src.config import WorldConfig
from src.world import h0, h1


EPS = 1e-12


@dataclass(frozen=True)
class PomdpConfig:
    """Configuration for the categorical epistemic-POMDP generative model.

    Mirrors the ``src/config.py`` frozen-dataclass style.
    """

    # --- paradigms / hidden state ---
    n_paradigms: int = 2                 # K; theta in {0..K-1}
    theta_vals: tuple[float, ...] = (0.0, 1.0)   # theta-value per paradigm (Newtonian / relativistic)
    true_paradigm: int = 1               # theta* the environment generates from (B true, matches nb16)

    # --- experiments (actions) and outcome discretization ---
    x_grid: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 3.0)  # available experiments
    n_o: int = 5                         # outcome bins (the n_o / discriminability knob)
    o_lo: float = -6.0                   # outcome binning range
    o_hi: float = 12.0

    # --- world (reused, discretized) ---
    world: WorldConfig = field(default_factory=WorldConfig)

    # --- preferences / belief-utility ---
    # C over the outcome modality: flat by default (epistemic foraging baseline).
    # belief-utility U(theta): preference over which paradigm is true.
    belief_utility: tuple[float, ...] = (0.0, 0.0)   # U(theta); 0 = no motivation
    beta_U: float = 1.0                  # weight of the belief-utility term in EFE
    gamma_policy: float = 1.0            # policy precision (softmax temperature on EFE)

    # --- social channel ---
    q_reliability: float = 0.80          # trust-as-precision reliability for A_social

    # --- prior ---
    D0: tuple[float, ...] | None = None  # initial q(theta); None -> uniform

    def __post_init__(self):
        if len(self.theta_vals) != self.n_paradigms:
            raise ValueError("theta_vals must have length n_paradigms")
        if len(self.belief_utility) != self.n_paradigms:
            raise ValueError("belief_utility must have length n_paradigms")


# ----------------------------------------------------------------------
# A — observation likelihood (per paradigm, per experiment)
# ----------------------------------------------------------------------

def outcome_bin_edges(cfg: PomdpConfig) -> np.ndarray:
    """``n_o + 1`` bin edges spanning [o_lo, o_hi], outer bins extend to +-inf."""
    inner = np.linspace(cfg.o_lo, cfg.o_hi, cfg.n_o - 1)
    return np.concatenate([[-np.inf], inner, [np.inf]])


def build_A_world(cfg: PomdpConfig) -> jnp.ndarray:
    """A_world[o, theta, a] = P(outcome o | paradigm theta, experiment x_a).

    Shape (n_o, K, n_actions). Each column over o sums to 1 (a normalized
    categorical). Built by integrating N(h0(x)+theta_val*h1(x), sigma^2) over
    the outcome bins. This is the theory-ladenness object: differing columns
    across theta mean the same outcome is interpreted differently.
    """
    wc = cfg.world
    sigma = wc.sigma
    x = np.asarray(cfg.x_grid)
    edges = outcome_bin_edges(cfg)                       # (n_o + 1,)
    h0x = np.asarray(h0(jnp.asarray(x), wc))             # (A,)
    h1x = np.asarray(h1(jnp.asarray(x), wc))             # (A,)

    A = np.zeros((cfg.n_o, cfg.n_paradigms, len(x)))
    for k in range(cfg.n_paradigms):
        mean = h0x + cfg.theta_vals[k] * h1x             # (A,)
        # cdf at edges per action: shape (n_o+1, A)
        cdf = norm.cdf(edges[:, None], loc=mean[None, :], scale=sigma)
        probs = np.diff(cdf, axis=0)                     # (n_o, A)
        A[:, k, :] = probs
    A = A / (A.sum(axis=0, keepdims=True) + EPS)         # normalize columns
    return jnp.asarray(A)


def build_A_social(cfg: PomdpConfig) -> jnp.ndarray:
    """A_social[o_soc, theta] reliability matrix (K x K).

    Diagonal = q (a report for paradigm theta is correct with prob q); off-
    diagonal mass split uniformly over the other paradigms. For K=2 this is the
    notebook-15 reliability matrix [[q, 1-q], [1-q, q]]. kappa = log(q/(1-q)) is
    the trust-as-message-precision that drove the fold.
    """
    K = cfg.n_paradigms
    q = cfg.q_reliability
    off = (1.0 - q) / max(K - 1, 1)
    A = np.full((K, K), off)
    np.fill_diagonal(A, q)
    A = A / (A.sum(axis=0, keepdims=True) + EPS)
    return jnp.asarray(A)


# ----------------------------------------------------------------------
# B — transition (strict identity: fixed truth, absorbing capture)
# ----------------------------------------------------------------------

def build_B(cfg: PomdpConfig) -> jnp.ndarray:
    """B[theta', theta] = identity. The paradigm truth does not change and the
    agent's actions are measurements, not control (David's core point)."""
    return jnp.eye(cfg.n_paradigms)


# ----------------------------------------------------------------------
# C / D / U
# ----------------------------------------------------------------------

def build_C(cfg: PomdpConfig) -> jnp.ndarray:
    """Log-preference over outcomes; flat => pure epistemic foraging baseline.

    Returned as a (n_o,) log-probability vector (pymdp C convention is log-pref).
    """
    return jnp.zeros((cfg.n_o,))


def build_D(cfg: PomdpConfig) -> jnp.ndarray:
    """Prior q(theta) at t=0. Uniform unless ``D0`` is given."""
    if cfg.D0 is None:
        return jnp.full((cfg.n_paradigms,), 1.0 / cfg.n_paradigms)
    d = jnp.asarray(cfg.D0)
    return d / (d.sum() + EPS)


def build_U(cfg: PomdpConfig) -> jnp.ndarray:
    """Belief-utility vector U(theta). Enters EFE only (see agent_pop)."""
    return jnp.asarray(cfg.belief_utility)


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = p + EPS
    q = q + EPS
    return float(np.sum(p * np.log(p / q)))


def discriminability(cfg: PomdpConfig, A_world: jnp.ndarray | None = None
                     ) -> np.ndarray:
    """Per-experiment theory discriminability d_a (the nb16 ``d`` knob).

    For K=2: symmetric KL between the two paradigm columns at each experiment.
    For K>2: mean pairwise symmetric KL. High d => experiment sharply
    separates the paradigms (strong evidence, flattens the fold); low d =>
    underdetermined / theory-laden.
    """
    A = np.asarray(build_A_world(cfg) if A_world is None else A_world)
    n_actions = A.shape[2]
    out = np.zeros(n_actions)
    K = cfg.n_paradigms
    for a in range(n_actions):
        pairs = []
        for i in range(K):
            for j in range(i + 1, K):
                ci, cj = A[:, i, a], A[:, j, a]
                pairs.append(0.5 * (_kl(ci, cj) + _kl(cj, ci)))
        out[a] = float(np.mean(pairs)) if pairs else 0.0
    return out


def build_generative_model(cfg: PomdpConfig) -> dict:
    """Assemble all generative-model arrays into one dict (the agent's model)."""
    return {
        "A_world": build_A_world(cfg),     # (n_o, K, A)
        "A_social": build_A_social(cfg),   # (K, K)
        "B": build_B(cfg),                 # (K, K) identity
        "C": build_C(cfg),                 # (n_o,)
        "D": build_D(cfg),                 # (K,)
        "U": build_U(cfg),                 # (K,)
        "d": jnp.asarray(discriminability(cfg)),  # (A,)
    }
