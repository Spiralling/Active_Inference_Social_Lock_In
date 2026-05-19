"""Configuration for the v2 paradigm-persistence model.

Hierarchy of frozen dataclasses; `ModelConfig` aggregates one of each
sub-config plus the population-level scalars. ``build()`` is the single
factory that constructs (Population, World, Policy, Trust) state with
consistent seeds.

Maps to IWAI26_Variational_norm_learning v2:
  - WorldConfig   ↔ PDF §1.2 (regression family h0, h1; sigma; theta_star schedule)
  - PolicyConfig  ↔ PDF §4   (softmax over EFE = EIG − cost)
  - TrustConfig   ↔ PDF §2/3 (Gamma-conjugate trust update with forgetting)
  - NetworkConfig ↔ PDF §2   (graph topology; SBM machinery retained from v1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ----------------------------------------------------------------------
# World — the parameterised regression family.
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class WorldConfig:
    """The world that generates observations o ~ N(h0(x) + theta* h1(x), sigma^2).

    The default (h0_kind='linear_half', h1_kind='power', h1_k=3) is the
    relativistic-correction case: h0(x) = x/2, h1(x) = x^3.
    """

    h0_kind: str = "linear_half"        # "linear_half" | "linear" | "zero"
    h1_kind: str = "power"               # "power" | "window" | "threshold" | "oscillatory"
    h1_k: int = 3                        # exponent for "power"
    h1_window_x0: float = 5.0            # centre for "window"
    h1_window_w: float = 1.0             # width for "window"
    h1_threshold_xc: float = 5.0         # critical x for "threshold"
    h1_omega: float = 1.0                # angular freq for "oscillatory"

    sigma: float = 1.0                   # observation noise std

    # Truth theta*(t) schedule.
    theta_star_pre: float = 0.0          # Newtonian baseline
    theta_star_post: float = 1.0         # relativistic correction magnitude
    schedule: str = "step"               # "step" | "reversal" | "ramp"
    schedule_t_shift: int = 50           # first env change at this step
    schedule_t_reverse: int | None = None  # only used for "reversal"
    schedule_ramp_T: int = 30            # ramp duration for "ramp"


# ----------------------------------------------------------------------
# Policy — active inference experimental design (PDF §4).
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyConfig:
    """Agents choose x via softmax over a per-objective gain functional.

    Two objectives are supported:

    ``"eig_minus_cost"`` (v2 baseline):
        pi_i(x) ∝ exp(beta_exp * (EIG_i(x) − c(x; cost_slope, cost_kind)))
        — the original informational design used by notebooks 08–11.
        Preserved for Tier-3 regression.

    ``"resource_gain"`` (trust-centric refactor):
        pi_i(x) ∝ exp(beta_exp * Ê[Δr_i | x])
        — 4-term decomposition: trust-weighted internal flow + aggregate-
        trust exogenous share − cost(x; r_i) − decay. Cost reads c0 and the
        Fisher-coupled barrier from ``ResourceConfig`` rather than the
        legacy cost_slope/cost_kind fields.
    """

    # Default grid spans the discriminating regime for h1(x)=x^3 without
    # immediately saturating the posterior. With sigma=1 and x=3,
    # Fisher info h1(x)^2/sigma^2 = 729 — a single high-x observation
    # contributes a lot but doesn't pin theta in one step.
    x_grid: tuple[float, ...] = (0.1, 0.3, 1.0, 2.0, 3.0)
    beta_exp: float = 1.0                # softmax temperature (inverse)
    objective: str = "eig_minus_cost"    # "eig_minus_cost" | "resource_gain"
    cost_slope: float = 1.0              # legacy: cost grows over the grid
    cost_kind: str = "quadratic"         # legacy: "abs" | "quadratic" | "exponential"


# ----------------------------------------------------------------------
# Inference — the agent's belief update over theta.
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class InferenceConfig:
    """Belief update over theta. ``posterior_rho`` is exponential forgetting
    that reverts the posterior toward the (mu_0, tau_0) paradigm prior.

    At ``posterior_rho = 1.0`` (default) the update is the standard
    ratcheting conjugate-Gaussian step — precision only ever grows and the
    mean never reverts, so the population is a gradient flow to consensus.
    At ``posterior_rho < 1`` an unobserved belief component relaxes
    geometrically back toward the prior mean (rate 1 - rho), while
    observations pull it toward the truth — a standing tension between
    accumulated evidence and the anchoring paradigm. Forgetting is what
    lets the dynamics sustain motion instead of settling onto the truth.
    """

    posterior_rho: float = 1.0           # 1.0 = no forgetting (ratchet)


# ----------------------------------------------------------------------
# Trust — Gamma-conjugate precision learning (PDF §2.3, eqs 6/8).
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class TrustConfig:
    """Per-edge sufficient statistics (alpha_ij, beta_ij); gamma = alpha/beta.

    Exponential forgetting with rate rho. learning=False freezes alpha/beta
    for ablation studies (PDF E1 condition with trust-learning off).
    """

    rho: float = 0.99                    # exponential forgetting; window ≈ 1/(1−ρ)
    prior_n0: float = 1.0                # alpha(0) per edge (incl. self)
    prior_eps0: float = 1.0              # so beta(0) = n0 * eps0 → initial gamma = 1/eps0
    learning: bool = True                # gate the alpha/beta update


# ----------------------------------------------------------------------
# Resource — exogenous inflow + internal trust-mediated flow.
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ResourceConfig:
    """Per-agent endogenous resource r_i and its update.

    The recursion (PDF §3 / Eq. flow-step) is

        r_i(t+1) = (1−δ)·r_i(t)
                 + α·Σ_j W_{ji}·r_j(t)             # internal flow from neighbours
                 − c(x_i; r_i)                      # apparatus cost (barrier)
                 + R_in · η_i(t+1)                  # exogenous share by aggregate trust

    where W = row-normalise(Γ) and η = col-sum-normalise(Γ) are deterministic
    readouts of the trust matrix (not separate state).

    Cost: c(x; r) = c0 · h1(x)^2 / σ^2 · φ(r) with barrier
        φ(r) = 1 / max(r − r_min, ε)
    so the Fisher information (Axis-C lever) is multiplicatively scaled by a
    resource barrier that diverges as r → r_min.
    """

    R_in: float = 0.0                    # exogenous inflow scale (Axis B)
    alpha_flow: float = 0.0              # fraction of r flowing along W per step
    delta_decay: float = 0.0             # geometric resource decay per step
    r0: float = 1.0                      # initial r_i, uniform
    c0: float = 0.0                      # cost scale (Axis C)
    r_min: float = 0.0                   # barrier floor
    fisher_cost_steepness: float = 1.0   # multiplier on h1(x)^2/sigma^2 in cost
    barrier_eps: float = 1e-3            # numerical floor for φ(r)


# ----------------------------------------------------------------------
# Utility — Mal-faithful socially-induced posterior target (λ-modified update).
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class UtilityConfig:
    """The Hyland-style cost-of-mind-change term U(q_i; {q_j, r_j}_j).

    Given the standard Bayes posterior (mu_data, tau_data) and a resource-
    weighted Gaussian target (mu_tgt, tau_tgt) computed from neighbours' Mal-
    faithful posteriors, the λ-modified update is the precision-weighted
    blend

        tau' = tau_data + lambda_mc * tau_tgt
        mu'  = (tau_data * mu_data + lambda_mc * tau_tgt * mu_tgt) / tau'

    At lambda_mc = 0 this reduces to standard Bayes (Tier-1 test #8).
    """

    lambda_mc: float = 0.0               # 0.0 ⇒ standard Bayes


# ----------------------------------------------------------------------
# Network — graph topology.
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class NetworkConfig:
    """Graph kind and density. Wraps src.network.build_adjacency."""

    kind: str = "watts_strogatz"         # "scale_free" | "watts_strogatz" | "planted_sbm"
    mean_degree: int = 4
    rewiring_p: float = 0.1              # WS only
    intra_prob: float = 0.04             # SBM only
    inter_prob: float = 0.003            # SBM only


# ----------------------------------------------------------------------
# Top-level model config — aggregates the four sub-configs.
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """One handle for an entire v2 simulation."""

    n_agents: int = 80
    mu_0: float = 0.0                    # initial posterior mean over theta
    tau_0: float = 1.0                   # initial posterior precision
    seed: int = 0

    world: WorldConfig = field(default_factory=WorldConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)


# ----------------------------------------------------------------------
# Factory.
# ----------------------------------------------------------------------

def build(cfg: ModelConfig | None = None) -> "Population":
    """Construct a v2 Population from a ModelConfig."""
    from src.population import Population
    import jax

    cfg = cfg or ModelConfig()
    key = jax.random.PRNGKey(cfg.seed)
    return Population.init(cfg, key)
